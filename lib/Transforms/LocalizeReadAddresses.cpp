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
// registers sit across a class boundary that must stay (a deeper fold keeps
// the class as an instance), the address arrives on an instance's output or
// on the module's own input, and the read cannot be retimed. This pass keeps
// every boundary and copies the registers instead.
//
// The next value of a signal, NV, is the value it takes after the next edge
// of the memory's clock: for a register, its next input (a synchronous reset
// applied); for logic, the same logic over its operands' next values; for a
// kept instance's output, a new output `<output>_foldsim_next` of its module
// carrying that output's next value; for a module's input, a new input
// `<input>_foldsim_next` that every instance of the module feeds with the
// next value of what it connects there. Both recurse, so registers any number
// of class levels down (an output of a class nested in a class) or up (an
// input from the parent, or the parent's parent) are reached. The memory's
// module gets a register of each next value at the boundary (`..._foldsim_
// local`), equal to the boundary signal on every cycle after the first edge,
// and the read's address is computed from those registers; seq-retime-firmem-
// reads then finds its address registers at home. Every register reached must
// be clocked, through the instances, by the memory's write clock.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

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

/// The cone of `value` within its block: the pure ops (`ops`) and its
/// boundary leaves (`leaves`: a kept instance's results and the module's
/// inputs); registers and constants end it. False if it reaches anything
/// else (a memory read, a register with an asynchronous reset, ...).
bool cone(Value value, SetVector<Operation *> &ops, SetVector<Value> &leaves) {
  SmallVector<Value> stack{value};
  DenseSet<Value> seen;
  while (!stack.empty()) {
    Value v = stack.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    Operation *op = v.getDefiningOp();
    if (!op || isa<hw::InstanceOp>(op)) {
      leaves.insert(v);
      continue;
    }
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      if (reg.getIsAsync())
        return false;
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

/// Clones the cone's ops before `b`'s point, in program order, with `map`
/// applied.
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
      if (Operation *def = operand.getDefiningOp();
          def && ops.contains(def) && !done.contains(def))
        stack.push_back({def, false});
  }
  return map.lookupOrDefault(value);
}

using Port = std::pair<Operation *, unsigned>; // (hw.module, port index)

/// The next-value closure over the class hierarchy.
struct Closure {
  DenseMap<StringAttr, hw::HWModuleOp> modules;
  DenseMap<Operation *, SmallVector<hw::InstanceOp>> uses; // module -> instances
  DenseMap<Value, Value> roots;
  DenseMap<std::pair<Value, Value>, bool> able;

  void index(ModuleOp top) {
    modules.clear();
    uses.clear();
    for (auto m : top.getOps<hw::HWModuleOp>())
      modules[m.getModuleNameAttr()] = m;
    top.walk([&](hw::InstanceOp i) {
      auto it = modules.find(i.getReferencedModuleNameAttr());
      if (it != modules.end())
        uses[it->second.getOperation()].push_back(i);
    });
  }

  hw::HWModuleOp moduleOf(Value v) {
    Operation *parent = isa<BlockArgument>(v)
                            ? cast<BlockArgument>(v).getOwner()->getParentOp()
                            : v.getDefiningOp()->getParentOp();
    return dyn_cast_or_null<hw::HWModuleOp>(parent);
  }

  hw::HWModuleOp child(hw::InstanceOp inst) {
    auto it = modules.find(inst.getReferencedModuleNameAttr());
    return it == modules.end() ? hw::HWModuleOp() : it->second;
  }

  /// The clock `clk` is driven by: up through module inputs to the value every
  /// instance connects there (all instances alike), or null if they differ.
  Value root(Value clk) {
    if (auto it = roots.find(clk); it != roots.end())
      return it->second;
    roots[clk] = Value();
    Value r = clk;
    if (auto arg = dyn_cast<BlockArgument>(clk)) {
      auto m = moduleOf(clk);
      auto it = m ? uses.find(m.getOperation()) : uses.end();
      if (it != uses.end() && !it->second.empty()) {
        r = Value();
        for (auto inst : it->second) {
          Value up = root(inst.getOperand(arg.getArgNumber()));
          if (!up || (r && up != r)) {
            r = Value();
            break;
          }
          r = up;
        }
      }
    }
    return roots[clk] = r;
  }

  /// Whether the next value of `v` (on the clock rooted at `clk`) can be
  /// computed: its cone, followed across kept instances and module inputs,
  /// ends at constants and synchronous registers on that clock.
  bool can(Value v, Value clk) {
    auto key = std::make_pair(v, clk);
    if (auto it = able.find(key); it != able.end())
      return it->second;
    able[key] = false; // a cycle through the hierarchy is refused
    bool ok = false;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto m = moduleOf(v);
      auto it = m ? uses.find(m.getOperation()) : uses.end();
      ok = it != uses.end() && !it->second.empty() &&
           !isa<seq::ClockType>(v.getType());
      if (ok)
        for (auto inst : it->second)
          if (!can(inst.getOperand(arg.getArgNumber()), clk)) {
            ok = false;
            break;
          }
    } else if (auto inst = v.getDefiningOp<hw::InstanceOp>()) {
      auto c = child(inst);
      ok = c && can(c.getBodyBlock()->getTerminator()->getOperand(
                        cast<OpResult>(v).getResultNumber()),
                    clk);
    } else if (auto reg = v.getDefiningOp<seq::FirRegOp>()) {
      ok = !reg.getIsAsync() && root(reg.getClk()) == clk;
    } else if (v.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {
      ok = true;
    } else if (isPureCone(v.getDefiningOp())) {
      ok = llvm::all_of(v.getDefiningOp()->getOperands(),
                        [&](Value o) { return can(o, clk); });
    }
    return able[key] = ok;
  }

  /// The boundary ports the next value of `v` needs (after `can` said yes).
  void need(Value v, DenseSet<Value> &seen, SetVector<Port> &outs,
            SetVector<Port> &ins) {
    if (!seen.insert(v).second)
      return;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto m = moduleOf(v);
      ins.insert({m.getOperation(), arg.getArgNumber()});
      for (auto inst : uses[m.getOperation()])
        need(inst.getOperand(arg.getArgNumber()), seen, outs, ins);
    } else if (auto inst = v.getDefiningOp<hw::InstanceOp>()) {
      auto c = child(inst);
      unsigned k = cast<OpResult>(v).getResultNumber();
      outs.insert({c.getOperation(), k});
      need(c.getBodyBlock()->getTerminator()->getOperand(k), seen, outs, ins);
    } else if (auto *op = v.getDefiningOp();
               op && !isa<seq::FirRegOp>(op) &&
               !op->hasTrait<OpTrait::ConstantLike>()) {
      for (Value o : op->getOperands())
        need(o, seen, outs, ins);
    }
  }
};

struct LocalizeReadAddressesPass
    : public foldsim::impl::LocalizeReadAddressesBase<LocalizeReadAddressesPass> {
  using LocalizeReadAddressesBase::LocalizeReadAddressesBase;
  void runOnOperation() override;
};
} // namespace

void LocalizeReadAddressesPass::runOnOperation() {
  ModuleOp top = getOperation();
  Closure cl;
  cl.index(top);

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

  // 1. The reads whose address crosses a boundary and whose next value can be
  //    computed, and the boundary ports that takes.
  struct Job {
    seq::FirMemReadOp read;
    Value clock; // the memory's write clock, in its module
  };
  SmallVector<Job> jobs;
  SetVector<Port> outs, ins;
  DenseSet<Value> seen;
  for (auto mem : memories) {
    Value clock;
    for (auto *user : mem->getUsers())
      if (auto write = dyn_cast<seq::FirMemWriteOp>(user)) {
        clock = write.getClk();
        break;
      }
    Value root = clock ? cl.root(clock) : Value();
    if (!root)
      continue;
    for (auto *user : mem->getUsers()) {
      auto read = dyn_cast<seq::FirMemReadOp>(user);
      if (!read)
        continue;
      SetVector<Operation *> ops;
      SetVector<Value> leaves;
      if (!cone(read.getAddress(), ops, leaves) || leaves.empty())
        continue;
      if (!llvm::all_of(leaves, [&](Value l) { return cl.can(l, root); })) {
        if (verbose)
          mem.emitRemark("foldsim-localize-read-addresses: a read's address "
                         "crosses a class boundary to other than synchronous "
                         "registers on the memory's clock; left alone");
        continue;
      }
      for (Value l : leaves)
        cl.need(l, seen, outs, ins);
      jobs.push_back({read, clock});
    }
  }
  if (jobs.empty())
    return;

  // 2. The ports, on placeholders, and every instance of a changed module
  //    rebuilt once with them.
  auto placeholder = [](OpBuilder &b, Location loc, Type type) -> Value {
    return UnrealizedConversionCastOp::create(b, loc, TypeRange{type},
                                              ValueRange{})
        .getResult(0);
  };
  SmallVector<Operation *> holders;
  DenseMap<Port, unsigned> outPort, inPort; // -> the new port's index
  MapVector<Operation *, unsigned> changed; // module -> its inputs before
  for (auto [m, k] : outs) {
    auto mod = cast<hw::HWModuleOp>(m);
    changed.insert({m, mod.getNumInputPorts()});
    auto *term = mod.getBodyBlock()->getTerminator();
    OpBuilder b(term);
    Value v = placeholder(b, term->getLoc(), term->getOperand(k).getType());
    holders.push_back(v.getDefiningOp());
    outPort[{m, k}] = mod.getNumOutputPorts();
    mod.appendOutputs(
        {{b.getStringAttr(mod.getOutputName(k) + "_foldsim_next"), v}});
  }
  for (auto [m, k] : ins) {
    auto mod = cast<hw::HWModuleOp>(m);
    changed.insert({m, mod.getNumInputPorts()});
    Type type = mod.getBodyBlock()->getArgument(k).getType();
    std::string name = (mod.getInputName(k) + "_foldsim_next").str();
    inPort[{m, k}] = mod.getNumInputPorts();
    mod.appendInput(name, type);
  }
  for (auto [m, before] : changed) {
    auto mod = cast<hw::HWModuleOp>(m);
    for (auto old : cl.uses[m]) {
      OpBuilder ib(old);
      SmallVector<Value> operands(old.getOperands());
      for (unsigned j = before; j < mod.getNumInputPorts(); ++j) {
        operands.push_back(placeholder(
            ib, old.getLoc(), mod.getBodyBlock()->getArgument(j).getType()));
        holders.push_back(operands.back().getDefiningOp());
      }
      auto fresh = hw::InstanceOp::create(
          ib, old.getLoc(), mod, old.getInstanceNameAttr(), operands,
          old.getParametersAttr(), old.getInnerSymAttr());
      fresh->setDiscardableAttrs(old->getDiscardableAttrDictionary());
      for (unsigned j = 0; j < old.getNumResults(); ++j)
        old.getResult(j).replaceAllUsesWith(fresh.getResult(j));
      old.erase();
    }
  }
  cl.index(top);

  // 3. The next values: NV(v) in v's module, built before its terminator.
  DenseMap<Value, Value> nexts;
  std::function<Value(Value)> next = [&](Value v) -> Value {
    if (auto it = nexts.find(v); it != nexts.end())
      return it->second;
    auto mod = cl.moduleOf(v);
    OpBuilder b(mod.getBodyBlock()->getTerminator());
    Value nv;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      nv = mod.getBodyBlock()->getArgument(
          inPort.lookup({mod.getOperation(), arg.getArgNumber()}));
    } else if (auto inst = v.getDefiningOp<hw::InstanceOp>()) {
      auto c = cl.child(inst);
      nv = inst.getResult(outPort.lookup(
          {c.getOperation(), cast<OpResult>(v).getResultNumber()}));
    } else if (auto reg = v.getDefiningOp<seq::FirRegOp>()) {
      nv = reg.getNext();
      if (reg.getReset())
        nv = comb::MuxOp::create(b, reg.getLoc(), reg.getReset(),
                                 reg.getResetValue(), nv);
    } else if (v.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {
      nv = v;
    } else {
      Operation *op = v.getDefiningOp();
      IRMapping map;
      for (Value o : op->getOperands())
        map.map(o, next(o));
      Operation *copy = b.clone(*op, map);
      nv = copy->getResult(cast<OpResult>(v).getResultNumber());
    }
    return nexts[v] = nv;
  };
  for (auto [port, idx] : outPort) {
    auto mod = cast<hw::HWModuleOp>(port.first);
    auto *term = mod.getBodyBlock()->getTerminator();
    term->setOperand(idx, next(term->getOperand(port.second)));
  }
  for (auto [port, idx] : inPort)
    for (auto inst : cl.uses[port.first])
      inst->setOperand(idx, next(inst.getOperand(port.second)));

  // 4. Each read: a register of the next value at every boundary leaf, and
  //    the address from them.
  unsigned localized = 0;
  for (auto &job : jobs) {
    auto read = job.read;
    SetVector<Operation *> ops;
    SetVector<Value> leaves;
    cone(read.getAddress(), ops, leaves);
    auto mod = cl.moduleOf(read.getResult());
    OpBuilder b(read);
    IRMapping map;
    for (Value l : leaves) {
      std::string name;
      if (auto arg = dyn_cast<BlockArgument>(l)) {
        name = (mod.getInputName(arg.getArgNumber()) + "_foldsim_local").str();
      } else {
        auto inst = cast<hw::InstanceOp>(l.getDefiningOp());
        auto c = cl.child(inst);
        name = (inst.getInstanceName() + "." +
                c.getOutputName(cast<OpResult>(l).getResultNumber()) +
                "_foldsim_local")
                   .str();
      }
      map.map(l, seq::FirRegOp::create(b, read.getLoc(), next(l), job.clock,
                                       b.getStringAttr(name)));
    }
    read.getAddressMutable().assign(
        cloneCone(b, read.getAddress(), ops, map));
    ++localized;
  }
  for (auto *h : holders) {
    if (!h->use_empty()) {
      h->emitError("foldsim-localize-read-addresses: a next-value port was "
                   "left unconnected");
      return signalPassFailure();
    }
    h->erase();
  }
  if (verbose && localized)
    llvm::errs() << "foldsim-localize-read-addresses: " << localized
                 << " reads' address registers copied beside their memory ("
                 << outs.size() << " next-value outputs, " << ins.size()
                 << " next-value inputs)\n";
}
