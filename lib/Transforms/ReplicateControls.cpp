//===- ReplicateControls.cpp - bound the loads of the slot controls -------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The static fold gives each region of state and each group of links its own
// copy of the slot's controls, but a copy still drives every store of its
// region: the address and write enable pins of hundreds to thousands of LUTs
// on one net, with no logic on the way. Routed on the V80, such a net takes
// 2.6-3.5 ns of a 4 ns period. Here each such register is replicated, every
// replica loaded from the same next value in lockstep, so that no net drives
// more than about `loads` pins.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

namespace foldsim {
#define GEN_PASS_DEF_REPLICATECONTROLS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// A memory this shallow is LUTs or flops, whose address and enable pins
/// grow with its width; a deeper one is block RAM.
constexpr uint64_t kLutDepth = 64;

/// Ops that only route a register's bits: copied along with a replica.
bool isWiring(Operation *op) {
  if (isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp, hw::BitcastOp>(
          op))
    return true;
  // An inversion, which synthesis absorbs into the pins it drives.
  if (auto x = dyn_cast<comb::XorOp>(op))
    return x.getNumOperands() == 2 &&
           llvm::any_of(x.getOperands(), [](Value v) {
             auto c = v.getDefiningOp<hw::ConstantOp>();
             return c && c.getValue().isAllOnes();
           });
  return false;
}

/// Logic of one LUT: a one-bit result of at most six input bits. With
/// `fanout-logic`, copied along with a replica like wiring, so that the
/// loads past it are bounded too.
bool isLutLogic(Operation *op) {
  if (!isa<comb::AndOp, comb::OrOp, comb::XorOp, comb::MuxOp, comb::ICmpOp>(op) ||
      op->getNumResults() != 1 || hw::getBitWidth(op->getResult(0).getType()) != 1)
    return false;
  int64_t bits = 0;
  for (auto operand : op->getOperands())
    bits += hw::getBitWidth(operand.getType());
  return bits <= 6;
}

bool isDontTouch(Operation *op) {
  auto attrs = sv::getSVAttributes(op);
  return attrs && llvm::any_of(attrs, [](Attribute a) {
           auto sva = dyn_cast<sv::SVAttributeAttr>(a);
           return sva && sva.getName().getValue() == "DONT_TOUCH";
         });
}

uint64_t widthOf(Value value) {
  return std::max<int64_t>(1, hw::getBitWidth(value.getType()));
}

/// The pins one bit of a port's address, enable or mask drives.
uint64_t memoryLoads(seq::FirMemType type, uint64_t width) {
  uint64_t depth = type.getDepth();
  if (depth <= kLutDepth)
    return width * llvm::divideCeil(depth, 32);
  return llvm::divideCeil(width, 36) * llvm::divideCeil(depth, 1024);
}

/// The bits of a memory one mask bit writes.
uint64_t maskGrain(seq::FirMemType type) {
  auto maskWidth = type.getMaskWidth();
  return maskWidth && *maskWidth ? type.getWidth() / *maskWidth : 1;
}

/// The pins a use of a register's bit drives (see the pass description).
uint64_t loadsOf(OpOperand &use) {
  auto *op = use.getOwner();
  unsigned n = use.getOperandNumber();
  auto port = [&](Value memory, Value data) -> uint64_t {
    auto type = cast<seq::FirMemType>(memory.getType());
    if (use.get() == data)
      return 1;
    if (auto write = dyn_cast<seq::FirMemWriteOp>(op);
        write && write.getMask() == use.get())
      return memoryLoads(type, maskGrain(type));
    return memoryLoads(type, type.getWidth());
  };
  if (auto read = dyn_cast<seq::FirMemReadOp>(op))
    return port(read.getMemory(), Value{});
  if (auto write = dyn_cast<seq::FirMemWriteOp>(op))
    return port(write.getMemory(), write.getData());
  if (auto mux = dyn_cast<comb::MuxOp>(op))
    return n == 0 ? widthOf(mux.getResult()) : 1;
  if (auto get = dyn_cast<hw::ArrayGetOp>(op))
    return use.get() == get.getIndex() ? widthOf(get.getResult()) : 1;
  if (auto reg = dyn_cast<seq::FirRegOp>(op))
    return use.get() == reg.getReset() ? widthOf(reg.getResult()) : 1;
  return 1;
}

/// The uses a register's bits end at, past wiring; and the wiring on the way.
struct Fanout {
  SmallVector<OpOperand *> sinks;
  llvm::SmallPtrSet<Operation *, 8> wiring;
};

Fanout fanoutOf(Value reg, bool logic = false) {
  Fanout fanout;
  SmallVector<Value> worklist{reg};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (auto &use : value.getUses()) {
      auto *owner = use.getOwner();
      if (isWiring(owner) || (logic && isLutLogic(owner))) {
        if (fanout.wiring.insert(owner).second)
          for (auto result : owner->getResults())
            worklist.push_back(result);
        continue;
      }
      fanout.sinks.push_back(&use);
    }
  }
  return fanout;
}

/// Splits `memory` by bits into parts of at most `loads` loads per port pin,
/// in whole mask bits where a mask bit is narrower than that.
void splitMemory(seq::FirMemOp memory, uint64_t loads) {
  auto type = memory.getType();
  uint64_t width = type.getWidth();
  uint64_t grain = maskGrain(type);
  uint64_t part =
      std::max<uint64_t>(1, loads / llvm::divideCeil(type.getDepth(), 32));
  if (grain <= part)
    part = part / grain * grain;
  if (width <= part)
    return;
  // Parts of equal width, but the last.
  uint64_t count = llvm::divideCeil(width, part);
  if (grain <= part)
    part = llvm::divideCeil(llvm::divideCeil(width, grain), count) * grain;
  else
    part = llvm::divideCeil(width, count);
  OpBuilder b(memory);
  struct Part {
    uint64_t low, width;
    seq::FirMemOp memory;
  };
  SmallVector<Part> parts;
  for (uint64_t low = 0, bits; low < width; low += bits) {
    bits = std::min(part, width - low);
    if (grain > part)
      bits = std::min(bits, (low / grain + 1) * grain - low);
    auto partType = seq::FirMemType::get(
        b.getContext(), type.getDepth(), bits,
        type.getMaskWidth()
            ? std::optional<uint32_t>(std::max<uint64_t>(1, bits / grain))
            : std::nullopt);
    auto partMemory = seq::FirMemOp::create(
        b, memory.getLoc(), partType, memory.getReadLatency(),
        memory.getWriteLatency(), memory.getRuw(), memory.getWuw(),
        b.getStringAttr(memory.getName().value_or("mem") + "_p" +
                        Twine(parts.size())),
        hw::InnerSymAttr{}, seq::FirMemInitAttr{}, memory.getPrefixAttr(),
        memory.getOutputFileAttr());
    partMemory->setDialectAttrs(memory->getDialectAttrs());
    parts.push_back({low, bits, partMemory});
  }
  for (auto *user : llvm::make_early_inc_range(memory->getUsers())) {
    OpBuilder ub(user);
    auto loc = user->getLoc();
    if (auto read = dyn_cast<seq::FirMemReadOp>(user)) {
      SmallVector<Value> data;
      for (auto &p : llvm::reverse(parts))
        data.push_back(seq::FirMemReadOp::create(ub, loc, p.memory,
                                                 read.getAddress(),
                                                 read.getClk(),
                                                 read.getEnable()));
      read.getData().replaceAllUsesWith(
          comb::ConcatOp::create(ub, loc, data).getResult());
    } else {
      auto write = cast<seq::FirMemWriteOp>(user);
      for (auto &p : parts) {
        Value data = comb::ExtractOp::create(ub, loc, write.getData(), p.low,
                                             p.width);
        Value mask, enable = write.getEnable();
        if (write.getMask()) {
          mask = comb::ExtractOp::create(
              ub, loc, write.getMask(), p.low / grain,
              std::max<uint64_t>(1, p.width / grain));
          // A part of one mask bit has a memory of mask width 1, which is
          // no mask (lowering ignores it): its bit enables the write.
          if (p.width <= grain) {
            enable = enable ? Value(comb::AndOp::create(ub, loc, enable, mask))
                            : mask;
            mask = Value();
          }
        }
        seq::FirMemWriteOp::create(ub, loc, p.memory, write.getAddress(),
                                   write.getClk(), enable, data, mask);
      }
    }
    user->erase();
  }
  memory.erase();
}

/// Splits a select by bits into selects of at most `loads` bits.
void splitMux(comb::MuxOp mux, uint64_t loads) {
  uint64_t width = widthOf(mux.getResult());
  uint64_t count = llvm::divideCeil(width, loads);
  if (count < 2)
    return;
  uint64_t part = llvm::divideCeil(width, count);
  OpBuilder b(mux);
  auto loc = mux.getLoc();
  auto bitsType = b.getIntegerType(width);
  auto asBits = [&](Value v) -> Value {
    return v.getType() == bitsType ? v
                                   : hw::BitcastOp::create(b, loc, bitsType, v);
  };
  Value t = asBits(mux.getTrueValue()), f = asBits(mux.getFalseValue());
  SmallVector<Value> parts;
  for (uint64_t low = 0; low < width; low += part) {
    uint64_t bits = std::min(part, width - low);
    parts.push_back(comb::MuxOp::create(
        b, loc, mux.getCond(), comb::ExtractOp::create(b, loc, t, low, bits),
        comb::ExtractOp::create(b, loc, f, low, bits), mux.getTwoStateAttr()));
  }
  std::reverse(parts.begin(), parts.end());
  Value result = comb::ConcatOp::create(b, loc, parts);
  if (result.getType() != mux.getType())
    result = hw::BitcastOp::create(b, loc, mux.getType(), result);
  mux.getResult().replaceAllUsesWith(result);
  mux.erase();
}

struct ReplicateControlsPass
    : public foldsim::impl::ReplicateControlsBase<ReplicateControlsPass> {
  using ReplicateControlsBase::ReplicateControlsBase;

  void runOnOperation() override {
    auto module = getOperation();
    auto *body = module.getBodyBlock();
    SmallVector<seq::FirRegOp> controls;
    for (auto reg : body->getOps<seq::FirRegOp>())
      if (reg->hasAttr(foldsim::kControlAttrName))
        controls.push_back(reg);
    if (loads) {
      splitWidePorts(controls, loads);
      replicate(body, controls, loads);
    }
    for (auto reg : body->getOps<seq::FirRegOp>())
      reg->removeAttr(foldsim::kControlAttrName);
    // `fanout`: any register of more loads, the target's own included (a
    // reset or select a copy's state takes from one flop).
    if (fanout) {
      SmallVector<seq::FirRegOp> heavy;
      for (auto reg : body->getOps<seq::FirRegOp>()) {
        // Not a control replicated above (DONT_TOUCH: it and its replicas).
        if (reg.getInnerSymAttr() || isDontTouch(reg))
          continue;
        uint64_t total = 0;
        for (auto *use : fanoutOf(reg.getResult(), fanoutLogic).sinks)
          total += loadsOf(*use);
        if (total > fanout)
          heavy.push_back(reg);
      }
      splitWidePorts(heavy, fanout);
      replicate(body, heavy, fanout, fanoutLogic);
    }
  }

  /// A memory of LUTs, or a select, whose one port a control drives with
  /// more than `loads` pins, split by bits so that the parts' ports can take
  /// replicas of their own.
  void splitWidePorts(ArrayRef<seq::FirRegOp> controls, uint64_t loads) {
    llvm::SetVector<Operation *> memories, muxes;
    for (auto reg : controls) {
      for (auto *use : fanoutOf(reg.getResult()).sinks) {
        if (loadsOf(*use) <= loads)
          continue;
        auto *owner = use->getOwner();
        if (isa<comb::MuxOp>(owner)) {
          muxes.insert(owner);
          continue;
        }
        Value memory;
        if (auto read = dyn_cast<seq::FirMemReadOp>(owner))
          memory = read.getMemory();
        else if (auto write = dyn_cast<seq::FirMemWriteOp>(owner))
          memory = write.getMemory();
        auto op = memory ? memory.getDefiningOp<seq::FirMemOp>() : nullptr;
        if (op && op.getType().getDepth() <= kLutDepth && !op.getInit() &&
            !op.getInnerSymAttr() &&
            llvm::all_of(op->getUsers(), [](Operation *user) {
              return isa<seq::FirMemReadOp, seq::FirMemWriteOp>(user);
            }))
          memories.insert(op);
      }
    }
    for (auto *op : memories)
      splitMemory(cast<seq::FirMemOp>(op), loads);
    for (auto *op : muxes)
      splitMux(cast<comb::MuxOp>(op), loads);
  }

  /// Each control of more than `loads` loads becomes replicas, each with a
  /// group of its loads in module order. A replica's driver, itself a
  /// control or its replica, comes first in the module, so the controls are
  /// taken last first: a driver counts the replicas of what it drives.
  void replicate(Block *body, ArrayRef<seq::FirRegOp> controls, uint64_t loads,
                 bool logic = false) {
    auto *context = body->getParentOp()->getContext();
    // Kept apart: replicas are equal registers, which synthesis merges back
    // into one (Vivado's equivalent register removal) unless each is kept,
    // and which implementation may merge again (opt_design's equivalent
    // driver merge, which ignores KEEP) unless each is DONT_TOUCH. DONT_TOUCH
    // also keeps phys_opt from rebuilding a replica's fanout, which is the
    // compiler's choice here, and retiming from moving it.
    auto dontTouch = sv::SVAttributeAttr::get(context, "DONT_TOUCH", R"("true")",
                                              /*emitAsComment=*/false);
    DenseMap<Operation *, unsigned> position;
    for (auto [i, op] : llvm::enumerate(*body))
      position[&op] = i;
    auto positionOf = [&](Operation *op) {
      while (op->getBlock() != body)
        op = op->getParentOp();
      return position.lookup(op);
    };
    for (auto reg : llvm::reverse(controls)) {
      if (reg.getInnerSymAttr())
        continue;
      auto fanout = fanoutOf(reg.getResult(), logic);
      uint64_t total = 0;
      for (auto *use : fanout.sinks)
        total += loadsOf(*use);
      if (total <= loads)
        continue;
      auto sinks = fanout.sinks;
      llvm::stable_sort(sinks, [&](OpOperand *x, OpOperand *y) {
        return positionOf(x->getOwner()) < positionOf(y->getOwner());
      });
      // Groups in module order, each of at most `loads` loads but a single
      // use of more.
      SmallVector<SmallVector<OpOperand *>> groups;
      uint64_t used = 0;
      for (auto *use : sinks) {
        uint64_t n = loadsOf(*use);
        if (groups.empty() || (used && used + n > loads)) {
          groups.emplace_back();
          used = 0;
        }
        groups.back().push_back(use);
        used += n;
      }
      // One use of more loads than a group: left to synthesis, which may
      // replicate it itself.
      if (groups.size() < 2)
        continue;
      sv::addSVAttributes(reg, {dontTouch});
      OpBuilder b(context);
      b.setInsertionPointAfter(reg);
      for (unsigned g = 1; g < groups.size(); ++g) {
        auto replica = cast<seq::FirRegOp>(b.clone(*reg));
        replica.setNameAttr(
            b.getStringAttr(reg.getName() + "_rep" + Twine(g)));
        position[replica] = position.lookup(reg);
        // The wiring from the register to each load of the group, copied
        // from the replica.
        IRMapping mapping;
        mapping.map(reg.getResult(), replica.getResult());
        std::function<Value(Value)> mapped = [&](Value value) -> Value {
          if (auto known = mapping.lookupOrNull(value))
            return known;
          auto *op = value.getDefiningOp();
          if (!op || !fanout.wiring.contains(op))
            return value;
          OpBuilder wb(op);
          wb.setInsertionPointAfter(op);
          auto *clone = wb.clone(*op);
          // A copy of the wiring sorts first, as an op the module order
          // did not number; entered so that a lookup of it never misses.
          position[clone] = 0;
          for (auto &operand : clone->getOpOperands())
            operand.set(mapped(operand.get()));
          for (auto [old, copied] :
               llvm::zip(op->getResults(), clone->getResults()))
            mapping.map(old, copied);
          return mapping.lookup(value);
        };
        for (auto *use : groups[g])
          use->set(mapped(use->get()));
      }
      // The wiring no load uses any more.
      for (bool erased = true; erased;) {
        SmallVector<Operation *> unused;
        for (auto *op : fanout.wiring)
          if (op->use_empty())
            unused.push_back(op);
        erased = !unused.empty();
        for (auto *op : unused) {
          // Its position stays: erasing it would leave a tombstone, and a
          // module of many replicated controls fills the map with them, so
          // that every lookup that misses walks most of the table. An op
          // created later at its address is entered anew (a replica or a
          // copy of the wiring, the only ops this creates).
          fanout.wiring.erase(op);
          op->erase();
        }
      }
    }
  }
};

} // namespace
