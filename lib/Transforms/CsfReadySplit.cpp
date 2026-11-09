//===- CsfReadySplit.cpp - Ready-cone lanes of the constant-shell fold ----===//
//
// The constant-shell fold's ready-cone copies (`csf-ready-copies` with
// `csf-ready-cones`), built as a netlist rewrite before the fold.
//
// A class's ready pass evaluates its late outputs (the outputs its inputs
// reach through logic) at a slot of the class's copy, from the registers
// those outputs read (kept apart from pages and rows: read at any slot). The
// schedule-only model of Prototype 1 put those passes on k small copies of
// the ready cone (lanes), off the copy's slots. Here the lanes are made real:
// the cone becomes a stateless model of its own, `<model>_rc`, whose inputs
// are the model's inputs it reads and the registers it reads (one output of
// the model, `csf_ready_regs`, the registers concatenated), and whose outputs
// are the model's late outputs. Everything after is the fold as it is:
//
//  - `csf_ready_regs` is an early output of the model, a bare concatenation of
//    registers: written at the model's state pass from their next values, read
//    by the cone's instance the frame after (a link; the cone's rows of ready
//    registers, in the design's terms);
//  - the cone model's instances are a class; `copies=<model>_rc:k` makes its k
//    copies the lanes; its evaluation (and any ready pass of its own, by
//    level) is placed by the schedule like any class's;
//  - the model keeps no ready pass for the outputs moved (none at all if every
//    late output moved), so its ready registers page or go to rows like any
//    other register.
//
// Exact by construction: the cone's logic is the model's, cloned; its register
// operands are the registers' values of the same target cycle (the early
// link), its inputs the same nets.
//
// An output whose cone reaches a memory read or a black box stays with the
// model (its ready pass stays on the model's copy).
//
//===----------------------------------------------------------------------===//

#include "CsfReadySplit.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace circt;

namespace foldsim {

/// Logic the cone model may hold: combinational and aggregate ops, no
/// state, no memory, no black box.
static bool isConeLogic(Operation *op) {
  if (op->getNumRegions() || op->getNumResults() == 0)
    return false;
  if (isa<comb::CombDialect>(op->getDialect()))
    return true;
  if (isa<hw::InstanceOp, hw::OutputOp>(op))
    return false;
  if (isa<hw::WireOp>(op))
    return true;
  return op->getDialect()->getNamespace() == "hw" && isMemoryEffectFree(op);
}

namespace {
/// A value of the model as the cone sees it.
struct Cone {
  /// Logic alone down to registers, constants and inputs.
  bool pure = true;
  /// An input (not the clock) reaches it.
  bool late = false;
};
} // namespace

LogicalResult splitReadyCones(ModuleOp top, ArrayRef<std::string> models,
                              SmallVectorImpl<ReadySplit> &done) {
  for (auto &name : models) {
    auto model = top.lookupSymbol<hw::HWModuleOp>(name);
    if (!model)
      return top.emitError("csf-ready-cones: no model '") << name << "'";
    ReadySplit result;
    result.model = name;
    result.cone = name + "_rc";
    if (top.lookupSymbol(result.cone))
      return model.emitError("csf-ready-cones: a module '")
             << result.cone << "' exists";
    auto *body = model.getBodyBlock();
    auto output = cast<hw::OutputOp>(body->getTerminator());

    // Each value's kind, by a post-order walk from the outputs (the body is a
    // graph region: no op order to rely on). Registers, constants and inputs
    // end a cone; anything else not logic makes it impure.
    DenseMap<Value, Cone> kind;
    SmallVector<Operation *> order; // the cones' logic, operands first
    auto classify = [&](Value root) {
      SmallVector<std::pair<Value, bool>> stack{{root, false}};
      while (!stack.empty()) {
        auto [value, expanded] = stack.pop_back_val();
        if (!expanded && kind.count(value))
          continue;
        if (auto arg = dyn_cast<BlockArgument>(value)) {
          kind[value] = {/*pure=*/!isa<seq::ClockType>(arg.getType()),
                         /*late=*/!isa<seq::ClockType>(arg.getType())};
          continue;
        }
        auto *op = value.getDefiningOp();
        if (isa<seq::FirRegOp, hw::ConstantOp, hw::AggregateConstantOp>(op)) {
          kind[value] = {true, false};
          continue;
        }
        if (!isConeLogic(op)) {
          for (auto result : op->getResults())
            kind[result] = {false, false};
          continue;
        }
        if (!expanded) {
          // A cycle through logic alone (none in a well-formed netlist)
          // reads as impure.
          for (auto result : op->getResults())
            kind[result] = {false, false};
          stack.push_back({value, true});
          for (auto operand : op->getOperands())
            if (!kind.count(operand))
              stack.push_back({operand, false});
          continue;
        }
        Cone cone;
        for (auto operand : op->getOperands()) {
          auto it = kind.find(operand);
          assert(it != kind.end() && "an operand classified first");
          cone.pure &= it->second.pure;
          cone.late |= it->second.late;
        }
        for (auto result : op->getResults())
          kind[result] = cone;
        order.push_back(op);
      }
    };
    SmallVector<unsigned> moved;
    for (auto [o, operand] : llvm::enumerate(output.getOperands())) {
      classify(operand);
      auto cone = kind.lookup(operand);
      if (!cone.late)
        continue;
      if (cone.pure)
        moved.push_back(o);
      else
        ++result.kept;
    }
    result.moved = moved.size();
    if (moved.empty()) {
      done.push_back(result);
      continue;
    }

    // The moved outputs' cones: their logic in operand order, the registers
    // and inputs they read.
    DenseSet<Operation *> inCone;
    llvm::SetVector<Operation *> regs;
    llvm::SetVector<unsigned> args;
    {
      SmallVector<Value> stack;
      DenseSet<Value> seen;
      for (unsigned o : moved)
        stack.push_back(output.getOperand(o));
      while (!stack.empty()) {
        Value value = stack.pop_back_val();
        if (!seen.insert(value).second)
          continue;
        if (auto arg = dyn_cast<BlockArgument>(value)) {
          args.insert(arg.getArgNumber());
          continue;
        }
        auto *op = value.getDefiningOp();
        if (isa<seq::FirRegOp>(op)) {
          regs.insert(op);
          continue;
        }
        if (!inCone.insert(op).second)
          continue;
        for (auto operand : op->getOperands())
          stack.push_back(operand);
      }
    }
    SmallVector<Operation *> coneOps;
    for (auto *op : order)
      if (inCone.contains(op))
        coneOps.push_back(op);
    // Constants (no operands) are not in `order`: first.
    for (auto &op : *body)
      if (inCone.contains(&op) && isa<hw::ConstantOp, hw::AggregateConstantOp>(op))
        coneOps.insert(coneOps.begin(), &op);
    result.ops = coneOps.size();
    result.regs = regs.size();
    for (auto *reg : regs)
      result.regBits += hw::getBitWidth(reg->getResult(0).getType());

    auto *context = top.getContext();
    auto intType = [&](uint64_t width) {
      return IntegerType::get(context, width);
    };
    auto modelPorts = model.getPortList();
    SmallVector<hw::PortInfo> ins, outs;
    for (auto &port : modelPorts)
      (port.dir == hw::ModulePort::Direction::Output ? outs : ins)
          .push_back(port);

    // The cone model.
    SmallVector<hw::PortInfo> ports;
    unsigned numIn = 0, numOut = 0;
    auto addPort = [&](StringAttr portName, Type type, bool input) {
      hw::PortInfo port;
      port.name = portName;
      port.type = type;
      port.dir = input ? hw::ModulePort::Direction::Input
                       : hw::ModulePort::Direction::Output;
      port.argNum = input ? numIn++ : numOut++;
      ports.push_back(port);
    };
    SmallVector<unsigned> argList(args.begin(), args.end());
    llvm::sort(argList);
    for (unsigned a : argList)
      addPort(ins[a].name, ins[a].type, true);
    bool hasRegs = result.regBits > 0;
    if (hasRegs)
      addPort(StringAttr::get(context, kReadyRegsPort), intType(result.regBits),
              true);
    for (unsigned o : moved)
      addPort(outs[o].name, outs[o].type, false);
    OpBuilder mb(model);
    mb.setInsertionPointAfter(model);
    auto cone = hw::HWModuleOp::create(mb, model.getLoc(),
                                       StringAttr::get(context, result.cone),
                                       hw::ModulePortInfo(ports));
    auto *coneBody = cone.getBodyBlock();
    coneBody->getTerminator()->erase();
    OpBuilder cb = OpBuilder::atBlockEnd(coneBody);
    IRMapping mapping;
    for (auto [k, a] : llvm::enumerate(argList))
      mapping.map(body->getArgument(a), coneBody->getArgument(k));
    uint64_t offset = 0;
    if (hasRegs) {
      Value all = coneBody->getArgument(argList.size());
      for (auto *reg : regs) {
        Value value = reg->getResult(0);
        uint64_t width = hw::getBitWidth(value.getType());
        Value bits = cb.createOrFold<comb::ExtractOp>(reg->getLoc(),
                                                      intType(width), all,
                                                      offset);
        if (bits.getType() != value.getType())
          bits = cb.createOrFold<hw::BitcastOp>(reg->getLoc(), value.getType(),
                                                bits);
        mapping.map(value, bits);
        offset += width;
      }
    }
    for (auto *op : coneOps)
      cb.clone(*op, mapping);
    SmallVector<Value> coneOuts;
    for (unsigned o : moved)
      coneOuts.push_back(mapping.lookup(output.getOperand(o)));
    hw::OutputOp::create(cb, output.getLoc(), coneOuts);

    // The model: its ready registers out, the moved outputs gone.
    SmallVector<Value> newOuts;
    {
      DenseSet<unsigned> movedSet(moved.begin(), moved.end());
      for (auto [o, operand] : llvm::enumerate(output.getOperands()))
        if (!movedSet.contains(o))
          newOuts.push_back(operand);
    }
    SmallVector<std::pair<unsigned, hw::PortInfo>> insertOuts;
    if (hasRegs) {
      OpBuilder ob(output);
      SmallVector<Value> bits;
      for (auto *reg : llvm::reverse(regs)) {
        Value value = reg->getResult(0);
        if (!isa<IntegerType>(value.getType()))
          value = hw::BitcastOp::create(
              ob, reg->getLoc(), intType(hw::getBitWidth(value.getType())),
              value);
        bits.push_back(value);
      }
      Value all = bits.size() == 1
                      ? bits.front()
                      : Value(comb::ConcatOp::create(ob, output.getLoc(), bits));
      newOuts.push_back(all);
      hw::PortInfo port;
      port.name = StringAttr::get(context, kReadyRegsPort);
      port.type = all.getType();
      port.dir = hw::ModulePort::Direction::Output;
      insertOuts.push_back({unsigned(outs.size()), port});
    }
    model.modifyPorts({}, insertOuts, {}, moved);
    output->setOperands(newOuts);
    // The moved outputs' logic nothing else uses.
    for (auto *op : llvm::reverse(coneOps))
      if (op->use_empty())
        op->erase();

    // Each instance: the model's, and the cone's beside it.
    SmallVector<hw::InstanceOp> instances;
    top.walk([&](hw::InstanceOp inst) {
      if (inst.getModuleName() == name)
        instances.push_back(inst);
    });
    for (auto inst : instances) {
      OpBuilder ib(inst);
      SmallVector<Value> inputs(inst.getInputs());
      auto now = hw::InstanceOp::create(ib, inst.getLoc(), model,
                                        inst.getInstanceNameAttr(), inputs,
                                        inst.getParametersAttr(),
                                        inst.getInnerSymAttr());
      SmallVector<Value> coneIns;
      for (unsigned a : argList)
        coneIns.push_back(inputs[a]);
      if (hasRegs)
        coneIns.push_back(now.getResult(now.getNumResults() - 1));
      auto coneInst = hw::InstanceOp::create(
          ib, inst.getLoc(), cone,
          StringAttr::get(context, inst.getInstanceName() + "_rc"), coneIns);
      for (auto attr : inst->getDiscardableAttrs()) {
        now->setAttr(attr.getName(), attr.getValue());
        coneInst->setAttr(attr.getName(), attr.getValue());
      }
      unsigned kept = 0, k = 0;
      for (unsigned o = 0; o < inst.getNumResults(); ++o) {
        if (k < moved.size() && moved[k] == o)
          inst.getResult(o).replaceAllUsesWith(coneInst.getResult(k++));
        else
          inst.getResult(o).replaceAllUsesWith(now.getResult(kept++));
      }
      inst.erase();
    }
    done.push_back(result);
  }
  return success();
}

} // namespace foldsim
