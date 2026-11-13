//===- LocalizeReadAddresses.cpp - bring a read's address registers home -===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A memory that asks for block RAM reads synchronously once the registers of
// its asynchronous read's address are beside it (seq-retime-firmem-reads,
// which inlines the private modules between them). When some of those
// registers sit in a class kept as an instance (a deeper fold: the class
// boundary must stay), the address arrives on the instance's output and the
// read cannot be retimed. This pass keeps the boundary and copies the
// registers instead: the class gets an output with the next value of that
// output (its registers' next values, synchronous resets applied, through
// the same logic), and the memory's module a register of it, equal to the
// instance's output on every cycle after the first edge; the read's address
// is computed from that register. seq-retime-firmem-reads then finds its
// address registers at home.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace foldsim {
#define GEN_PASS_DEF_LOCALIZEREADADDRESSES
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {
enum class Hint { None, Block, Distributed };

/// The memory's synthesis hint, as seq-retime-firmem-reads reads it.
Hint hintOf(Operation *op) {
  auto attrs = sv::getSVAttributes(op);
  if (!attrs)
    return Hint::None;
  for (auto attr : attrs.getAsRange<sv::SVAttributeAttr>()) {
    auto name = attr.getName().getValue();
    if ((name != "ram_style" && name != "ramstyle" && name != "syn_ramstyle") ||
        !attr.getExpression())
      continue;
    std::string text = attr.getExpression().getValue().trim().trim('"').lower();
    StringRef value(text);
    for (auto style : {"block", "block_ram", "bram", "ultra", "m9k", "m10k",
                       "m20k", "m144k", "m-ram", "mram"})
      if (value.contains(style))
        return Hint::Block;
    for (auto style :
         {"distributed", "registers", "logic", "mlab", "select_ram", "lutram"})
      if (value.contains(style))
        return Hint::Distributed;
  }
  return Hint::None;
}

/// Logic a cone may pass through: no regions, no side effects, not a read.
bool isPureCone(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         !isa<seq::FirMemReadOp, seq::FirMemReadWriteOp, hw::InstanceOp,
              seq::FirRegOp>(op);
}

/// The cone of `value` within its block: the pure ops in program order
/// (`ops`) and its leaves. False if a leaf is neither a constant, a register
/// with no asynchronous reset (`regs`), nor (with `insts`) an instance's
/// result.
bool cone(Value value, SetVector<Operation *> &ops, SetVector<Value> &regs,
          SetVector<Value> *insts) {
  SmallVector<Value> stack{value};
  DenseSet<Value> seen;
  while (!stack.empty()) {
    Value v = stack.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    Operation *op = v.getDefiningOp();
    if (!op)
      return false; // a block argument: an input of the module
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      if (reg.getIsAsync())
        return false;
      regs.insert(v);
      continue;
    }
    if (isa<hw::InstanceOp>(op)) {
      if (!insts)
        return false;
      insts->insert(v);
      continue;
    }
    if (op->hasTrait<OpTrait::ConstantLike>())
      continue;
    if (!isPureCone(op))
      return false;
    ops.insert(op);
    for (Value operand : op->getOperands())
      stack.push_back(operand);
  }
  return true;
}

/// Clones the cone's ops before `at`, in program order, with `map` applied.
Value cloneCone(OpBuilder &b, Value value, SetVector<Operation *> &ops,
                IRMapping &map) {
  // Operands before users (a graph region may use a value before its op).
  DenseSet<Operation *> done;
  SmallVector<std::pair<Operation *, bool>> stack;
  for (Operation *op : ops)
    stack.push_back({op, false});
  while (!stack.empty()) {
    auto [op, expanded] = stack.pop_back_val();
    if (done.contains(op))
      continue;
    if (expanded) {
      done.insert(op);
      b.clone(*op, map);
      continue;
    }
    stack.push_back({op, true});
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp(); def && ops.contains(def) && !done.contains(def))
        stack.push_back({def, false});
  }
  return map.lookupOrDefault(value);
}

struct LocalizeReadAddressesPass
    : public foldsim::impl::LocalizeReadAddressesBase<LocalizeReadAddressesPass> {
  using LocalizeReadAddressesBase::LocalizeReadAddressesBase;
  void runOnOperation() override;
};
} // namespace

void LocalizeReadAddressesPass::runOnOperation() {
  ModuleOp top = getOperation();
  DenseMap<StringAttr, hw::HWModuleOp> modules;
  for (auto m : top.getOps<hw::HWModuleOp>())
    modules[m.getModuleNameAttr()] = m;

  // The next-value port of each (module, output) made so far.
  DenseMap<std::pair<Operation *, unsigned>, unsigned> nextPort;
  unsigned localized = 0;

  // The memories this pass helps: asynchronous reads (latency 0, writes 1)
  // that ask for block RAM, or (mode auto) unhinted ones of min-bits bits.
  SmallVector<seq::FirMemOp> memories;
  top.walk([&](seq::FirMemOp mem) {
    if (mem.getReadLatency() != 0 || mem.getWriteLatency() != 1)
      return;
    auto type = mem.getType();
    Hint hint = hintOf(mem);
    if (hint == Hint::Block ||
        (hint == Hint::None && mode == "auto" &&
         type.getDepth() * type.getWidth() >= minBits))
      memories.push_back(mem);
  });

  for (auto mem : memories) {
    // The clock of its writes: the copy registers take it.
    Value clock;
    for (auto *user : mem->getUsers())
      if (auto write = dyn_cast<seq::FirMemWriteOp>(user)) {
        clock = write.getClk();
        break;
      }
    if (!clock)
      continue;
    SmallVector<seq::FirMemReadOp> reads;
    for (auto *user : mem->getUsers())
      if (auto read = dyn_cast<seq::FirMemReadOp>(user))
        reads.push_back(read);
    for (auto read : reads) {
      SetVector<Operation *> ops;
      SetVector<Value> regs, insts;
      if (!cone(read.getAddress(), ops, regs, &insts) || insts.empty())
        continue;
      // Each instance result: its module's output must come from registers
      // on the clock the instance receives as the memory's write clock.
      bool ok = true;
      for (Value v : insts) {
        auto inst = cast<hw::InstanceOp>(v.getDefiningOp());
        auto it = modules.find(inst.getReferencedModuleNameAttr());
        if (it == modules.end()) {
          ok = false;
          break;
        }
        auto child = it->second;
        unsigned k = cast<OpResult>(v).getResultNumber();
        Value out = child.getBodyBlock()->getTerminator()->getOperand(k);
        SetVector<Operation *> cops;
        SetVector<Value> cregs;
        if (!cone(out, cops, cregs, nullptr) || cregs.empty()) {
          ok = false;
          break;
        }
        for (Value r : cregs) {
          auto arg = dyn_cast<BlockArgument>(r.getDefiningOp<seq::FirRegOp>().getClk());
          if (!arg || inst.getOperand(arg.getArgNumber()) != clock) {
            ok = false;
            break;
          }
        }
        if (!ok)
          break;
      }
      if (!ok)
        continue;

      // Next-value outputs where missing.
      for (Value v : insts) {
        auto inst = cast<hw::InstanceOp>(v.getDefiningOp());
        auto child = modules[inst.getReferencedModuleNameAttr()];
        unsigned k = cast<OpResult>(v).getResultNumber();
        auto key = std::make_pair(child.getOperation(), k);
        if (nextPort.count(key))
          continue;
        auto *term = child.getBodyBlock()->getTerminator();
        Value out = term->getOperand(k);
        SetVector<Operation *> cops;
        SetVector<Value> cregs;
        cone(out, cops, cregs, nullptr);
        OpBuilder b(term);
        IRMapping map;
        for (Value r : cregs) {
          auto reg = r.getDefiningOp<seq::FirRegOp>();
          Value next = reg.getNext();
          if (reg.getReset())
            next = comb::MuxOp::create(b, reg.getLoc(), reg.getReset(),
                                       reg.getResetValue(), next);
          map.map(r, next);
        }
        Value nv = cloneCone(b, out, cops, map);
        auto name = b.getStringAttr(child.getOutputName(k) + "_foldsim_next");
        unsigned port = child.getNumOutputPorts();
        child.appendOutputs({{name, nv}});
        nextPort[key] = port;
        // Every instance of the module: its new result.
        SmallVector<hw::InstanceOp> uses;
        top.walk([&](hw::InstanceOp i) {
          if (i.getReferencedModuleNameAttr() == child.getModuleNameAttr())
            uses.push_back(i);
        });
        for (auto old : uses) {
          OpBuilder ib(old);
          auto fresh = hw::InstanceOp::create(
              ib, old.getLoc(), child, old.getInstanceNameAttr(),
              SmallVector<Value>(old.getOperands()), old.getParametersAttr(),
              old.getInnerSymAttr());
          fresh->setDiscardableAttrs(old->getDiscardableAttrDictionary());
          for (unsigned j = 0; j < old.getNumResults(); ++j)
            old.getResult(j).replaceAllUsesWith(fresh.getResult(j));
          old.erase();
        }
      }
      // The instances were rebuilt: find the address's cone again, its
      // instance leaves now the new instances' results.
      ops.clear();
      regs.clear();
      insts.clear();
      cone(read.getAddress(), ops, regs, &insts);
      // In the memory's module: a register of each next value, and the
      // address from those registers.
      OpBuilder b(read);
      IRMapping map;
      for (Value v : insts) {
        auto inst = cast<hw::InstanceOp>(v.getDefiningOp());
        auto child = modules[inst.getReferencedModuleNameAttr()];
        unsigned k = cast<OpResult>(v).getResultNumber();
        Value next = inst.getResult(nextPort[{child.getOperation(), k}]);
        auto name = b.getStringAttr(inst.getInstanceName() + "." +
                                    child.getOutputName(k) + "_foldsim_local");
        map.map(v, seq::FirRegOp::create(b, read.getLoc(), next, clock, name));
      }
      read.getAddressMutable().assign(cloneCone(b, read.getAddress(), ops, map));
      ++localized;
    }
  }
  if (verbose && localized)
    llvm::errs() << "foldsim-localize-read-addresses: " << localized
                 << " reads' address registers copied beside their memory\n";
}
