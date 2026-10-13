//===- FuseModels.cpp - Simulate pairs of models as one -------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two models of a netlist that exchange bits combinationally both ways take a
// thread of each a round trip through the channels between them, every target
// cycle. Pairing their instances and simulating each pair as one flat model
// makes those paths internal to it.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "foldsim/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

namespace foldsim {
#define GEN_PASS_DEF_FUSEMODELS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

struct FuseModelsPass
    : public foldsim::impl::FuseModelsBase<FuseModelsPass> {
  using FuseModelsBase::FuseModelsBase;
  void runOnOperation() override;
  LogicalResult fuse(hw::HWModuleOp netlist, hw::HWModuleOp first,
                     hw::HWModuleOp second);
};

} // namespace

/// The instances whose outputs `value` is computed from, through the
/// netlist's wiring.
static void collectSources(Value value, SmallPtrSetImpl<Operation *> &sources,
                           DenseSet<Value> &seen) {
  if (!seen.insert(value).second)
    return;
  auto *op = value.getDefiningOp();
  if (!op)
    return;
  if (isa<hw::InstanceOp>(op)) {
    sources.insert(op);
    return;
  }
  for (auto operand : op->getOperands())
    collectSources(operand, sources, seen);
}

/// Whether `value` is computed only from outputs of `partner` and constants,
/// by ops without regions or side effects.
static bool isFromPartner(Value value, Operation *partner,
                          DenseMap<Value, bool> &cache) {
  if (auto it = cache.find(value); it != cache.end())
    return it->second;
  auto *op = value.getDefiningOp();
  bool result;
  if (!op)
    result = false;
  else if (op == partner)
    result = true;
  else if (isa<hw::InstanceOp>(op) || op->getNumRegions() != 0 ||
           !isMemoryEffectFree(op))
    result = false;
  else {
    cache[value] = false;
    result = llvm::all_of(op->getOperands(), [&](Value operand) {
      return isFromPartner(operand, partner, cache);
    });
  }
  return cache[value] = result;
}

/// Whether `a`, from outputs of `partnerA`, and `b`, from outputs of
/// `partnerB`, are the same logic of the same outputs.
static bool isSameLogic(Value a, Value b, Operation *partnerA,
                        Operation *partnerB) {
  auto *opA = a.getDefiningOp(), *opB = b.getDefiningOp();
  if (!opA || !opB)
    return false;
  if (opA == partnerA || opB == partnerB)
    return opA == partnerA && opB == partnerB &&
           cast<OpResult>(a).getResultNumber() ==
               cast<OpResult>(b).getResultNumber();
  if (opA->getName() != opB->getName() ||
      opA->getAttrDictionary() != opB->getAttrDictionary() ||
      cast<OpResult>(a).getResultNumber() !=
          cast<OpResult>(b).getResultNumber() ||
      opA->getNumOperands() != opB->getNumOperands() ||
      opA->getResultTypes() != opB->getResultTypes())
    return false;
  for (auto [x, y] : llvm::zip(opA->getOperands(), opB->getOperands()))
    if (!isSameLogic(x, y, partnerA, partnerB))
      return false;
  return true;
}

LogicalResult FuseModelsPass::fuse(hw::HWModuleOp netlist,
                                         hw::HWModuleOp first,
                                         hw::HWModuleOp second) {
  std::array<hw::HWModuleOp, 2> models = {first, second};
  std::array<SmallVector<hw::InstanceOp>, 2> instances;
  for (auto instance : netlist.getBodyBlock()->getOps<hw::InstanceOp>())
    for (unsigned m = 0; m < 2; ++m)
      if (instance.getModuleNameAttr().getAttr() == models[m].getSymNameAttr())
        instances[m].push_back(instance);
  if (instances[0].empty() || instances[0].size() != instances[1].size())
    return netlist.emitError("cannot pair the instances of '")
           << first.getSymName() << "' (" << instances[0].size() << ") and '"
           << second.getSymName() << "' (" << instances[1].size() << ")";
  for (auto model : models)
    if (!model.getBodyBlock()->getOps<hw::InstanceOp>().empty())
      return model.emitError("a fused model must be flat");

  // Each instance of the first model with the one of the second it exchanges
  // bits with.
  auto sourcesOf = [](hw::InstanceOp instance) {
    SmallPtrSet<Operation *, 8> sources;
    DenseSet<Value> seen;
    for (auto operand : instance.getInputs())
      collectSources(operand, sources, seen);
    return sources;
  };
  DenseMap<Operation *, SmallPtrSet<Operation *, 8>> sources;
  for (auto &list : instances)
    for (auto instance : list)
      sources[instance] = sourcesOf(instance);
  SmallVector<hw::InstanceOp> partnerOf;
  SmallPtrSet<Operation *, 8> taken;
  for (auto a : instances[0]) {
    hw::InstanceOp partner;
    for (auto b : instances[1]) {
      if (!sources[a].contains(b) && !sources[b].contains(a))
        continue;
      if (partner)
        return a.emitError("connected to more than one '")
               << second.getSymName() << "'";
      partner = b;
    }
    if (!partner || !taken.insert(partner).second)
      return a.emitError("no '")
             << second.getSymName() << "' of its own to fuse with";
    partnerOf.push_back(partner);
  }
  auto pair = [&](unsigned i, unsigned m) -> hw::InstanceOp {
    return m == 0 ? instances[0][i] : partnerOf[i];
  };
  unsigned numPairs = instances[0].size();

  // The clock both models take, and each input: inside, from the other model,
  // or a port of the fused model.
  auto *context = netlist.getContext();
  SmallVector<hw::PortInfo> ports;
  struct Input {
    unsigned model, index;
  };
  SmallVector<Input> external;
  // The model input each input of the fused model is, but its clock.
  SmallVector<std::optional<Input>> inputOf;
  // For an input inside, its logic in the first pair.
  std::array<DenseMap<unsigned, Value>, 2> internal;
  std::optional<unsigned> clockPort;
  std::array<std::optional<unsigned>, 2> clockIndex;
  for (unsigned m = 0; m < 2; ++m) {
    auto model = models[m];
    for (auto [index, type] : llvm::enumerate(model.getInputTypes())) {
      auto name = model.getInputName(index);
      if (isa<seq::ClockType>(type)) {
        if (clockIndex[m])
          return model.emitError("a fused model must have one clock");
        clockIndex[m] = index;
        for (unsigned i = 0; i < numPairs; ++i)
          if (pair(i, m).getOperand(index) !=
              pair(i, 0).getOperand(*clockIndex[0]))
            return pair(i, m).emitError("not on the clock of '")
                   << pair(i, 0).getInstanceName() << "'";
        if (!clockPort) {
          clockPort = ports.size();
          hw::PortInfo port;
          port.name = StringAttr::get(context, name);
          port.type = type;
          port.dir = hw::ModulePort::Direction::Input;
          ports.push_back(port);
          inputOf.push_back(std::nullopt);
        }
        continue;
      }
      bool inside = true;
      for (unsigned i = 0; i < numPairs && inside; ++i) {
        DenseMap<Value, bool> cache;
        auto operand = pair(i, m).getOperand(index);
        inside = isFromPartner(operand, pair(i, 1 - m), cache) &&
                 isSameLogic(pair(0, m).getOperand(index), operand,
                             pair(0, 1 - m), pair(i, 1 - m));
      }
      if (inside) {
        internal[m][index] = pair(0, m).getOperand(index);
        continue;
      }
      external.push_back({m, unsigned(index)});
      inputOf.push_back(external.back());
      hw::PortInfo port;
      port.name = StringAttr::get(context, model.getSymName() + "_" + name);
      port.type = type;
      port.dir = hw::ModulePort::Direction::Input;
      ports.push_back(port);
    }
  }
  if (!clockPort)
    return first.emitError("a fused model must have a clock");
  // An output only the other model's inputs inside read is not a port.
  std::function<bool(Value, unsigned, unsigned)> readOutside =
      [&](Value value, unsigned i, unsigned m) {
        for (auto &use : value.getUses()) {
          auto *user = use.getOwner();
          if (user == pair(i, 1 - m).getOperation()) {
            if (!internal[1 - m].count(use.getOperandNumber()))
              return true;
            continue;
          }
          if (isa<hw::InstanceOp>(user) || user->getNumRegions() != 0 ||
              !isMemoryEffectFree(user) || user->getNumResults() == 0)
            return true;
          for (auto result : user->getResults())
            if (readOutside(result, i, m))
              return true;
        }
        return false;
      };
  std::array<SmallVector<bool>, 2> exported;
  for (unsigned m = 0; m < 2; ++m)
    for (unsigned index = 0; index < models[m].getNumOutputPorts(); ++index) {
      bool outside = false;
      for (unsigned i = 0; i < numPairs && !outside; ++i)
        outside = readOutside(pair(i, m).getResult(index), i, m);
      exported[m].push_back(outside);
    }
  for (unsigned m = 0; m < 2; ++m)
    for (auto [index, type] : llvm::enumerate(models[m].getOutputTypes())) {
      if (!exported[m][index])
        continue;
      hw::PortInfo port;
      port.name = StringAttr::get(context, models[m].getSymName() + "_" +
                                               models[m].getOutputName(index));
      port.type = type;
      port.dir = hw::ModulePort::Direction::Output;
      ports.push_back(port);
    }

  // The fused model: both models' logic, each reading the other's outputs
  // for the inputs inside.
  OpBuilder top(second);
  top.setInsertionPointAfter(second);
  auto name = (first.getSymName() + "_" + second.getSymName()).str();
  auto fused = hw::HWModuleOp::create(
      top, first.getLoc(), top.getStringAttr(name), hw::ModulePortInfo(ports));
  fused.setPrivate();
  auto *body = fused.getBodyBlock();
  body->getTerminator()->erase();
  auto b = ImplicitLocOpBuilder::atBlockEnd(first.getLoc(), body);
  BackedgeBuilder backedges(b, first.getLoc());
  std::array<IRMapping, 2> mappings;
  std::array<DenseMap<unsigned, Backedge>, 2> pending;
  for (unsigned m = 0; m < 2; ++m) {
    auto *modelBody = models[m].getBodyBlock();
    mappings[m].map(modelBody->getArgument(*clockIndex[m]),
                    body->getArgument(*clockPort));
    for (auto [index, arg] : llvm::enumerate(modelBody->getArguments())) {
      if (!internal[m].count(index))
        continue;
      pending[m].try_emplace(index, backedges.get(arg.getType()));
      mappings[m].map(arg, pending[m].find(index)->second);
    }
  }
  for (auto [arg, input] : llvm::enumerate(inputOf))
    if (input)
      mappings[input->model].map(
          models[input->model].getBodyBlock()->getArgument(input->index),
          body->getArgument(arg));
  // A model's logic is a graph region: every op is cloned, then its operands,
  // some defined further on, are set.
  std::array<SmallVector<Value>, 2> outputs;
  for (unsigned m = 0; m < 2; ++m) {
    SmallVector<std::pair<Operation *, Operation *>> cloned;
    for (auto &op : models[m].getBodyBlock()->without_terminator()) {
      if (op.getNumRegions() != 0)
        return op.emitError("a fused model must have no ops with regions");
      auto *clone = b.clone(op);
      mappings[m].map(op.getResults(), clone->getResults());
      cloned.push_back({&op, clone});
    }
    for (auto [op, clone] : cloned)
      for (auto &operand : clone->getOpOperands())
        operand.set(mappings[m].lookupOrDefault(
            op->getOperand(operand.getOperandNumber())));
    for (auto value : models[m].getBodyBlock()->getTerminator()->getOperands())
      outputs[m].push_back(mappings[m].lookup(value));
  }
  // An input inside: its wiring in the netlist, on the other model's outputs.
  // That may be an input inside of the other model in turn, so each takes its
  // value once all are known.
  DenseMap<Value, Value> targetOf;
  SmallVector<std::pair<Backedge, Value>> targets;
  for (unsigned m = 0; m < 2; ++m) {
    auto partner = pair(0, 1 - m);
    for (auto &[index, root] : internal[m]) {
      IRMapping mapping;
      for (auto [result, value] :
           llvm::zip(partner.getResults(), outputs[1 - m]))
        mapping.map(result, value);
      std::function<Value(Value)> cloneLogic = [&](Value value) -> Value {
        if (auto mapped = mapping.lookupOrNull(value))
          return mapped;
        auto *op = value.getDefiningOp();
        for (auto operand : op->getOperands())
          cloneLogic(operand);
        auto *clone = b.clone(*op, mapping);
        return clone->getResult(cast<OpResult>(value).getResultNumber());
      };
      auto backedge = pending[m].find(index)->second;
      auto value = cloneLogic(root);
      targetOf[backedge] = value;
      targets.push_back({backedge, value});
    }
  }
  SmallVector<Value> results;
  for (unsigned m = 0; m < 2; ++m)
    for (auto [index, value] : llvm::enumerate(outputs[m]))
      if (exported[m][index])
        results.push_back(value);
  hw::OutputOp::create(b, results);
  for (auto [backedge, value] : targets) {
    unsigned steps = 0;
    while (targetOf.count(value)) {
      value = targetOf[value];
      if (++steps > targets.size())
        return first.emitError("inputs of the fused models wired in a loop");
    }
    backedge.setValue(value);
  }

  // Each pair becomes an instance of it.
  for (unsigned i = 0; i < numPairs; ++i) {
    auto a = pair(i, 0);
    OpBuilder nb(a);
    SmallVector<Value> operands;
    for (auto &input : inputOf)
      operands.push_back(input ? pair(i, input->model).getOperand(input->index)
                               : a.getOperand(*clockIndex[0]));
    auto instance = hw::InstanceOp::create(nb, a.getLoc(), fused,
                                           a.getInstanceNameAttr(), operands);
    unsigned result = 0;
    for (unsigned m = 0; m < 2; ++m)
      for (auto [index, old] : llvm::enumerate(pair(i, m).getResults()))
        if (exported[m][index])
          old.replaceAllUsesWith(instance.getResult(result++));
  }
  // The pairs go, and with them the wiring of the inputs inside.
  SetVector<Operation *> wiring;
  std::function<void(Value)> collect = [&](Value value) {
    for (auto *user : value.getUsers())
      if (!isa<hw::InstanceOp>(user) && wiring.insert(user))
        for (auto result : user->getResults())
          collect(result);
  };
  for (unsigned i = 0; i < numPairs; ++i)
    for (unsigned m = 0; m < 2; ++m)
      for (auto old : pair(i, m).getResults())
        collect(old);
  for (unsigned i = 0; i < numPairs; ++i)
    for (unsigned m = 0; m < 2; ++m)
      pair(i, m)->dropAllReferences();
  for (unsigned i = 0; i < numPairs; ++i)
    for (unsigned m = 0; m < 2; ++m)
      pair(i, m)->dropAllUses();
  for (unsigned i = 0; i < numPairs; ++i) {
    pair(i, 1).erase();
    pair(i, 0).erase();
  }
  for (bool erased = true; erased;) {
    erased = false;
    for (auto *op : wiring)
      if (op && op->use_empty()) {
        wiring.remove(op);
        op->erase();
        erased = true;
        break;
      }
  }
  if (!wiring.empty())
    return wiring.front()->emitError("read outside the fused model");
  return success();
}

void FuseModelsPass::runOnOperation() {
  auto top = getOperation();
  auto netlist = dyn_cast_or_null<hw::HWModuleOp>(
      SymbolTable::lookupSymbolIn(top, moduleName));
  if (!netlist || !netlist->hasAttr("foldsim.netlist")) {
    top.emitError("no netlist '") << moduleName << "'";
    return signalPassFailure();
  }
  for (auto &group : groups) {
    auto [firstName, secondName] = StringRef(group).split('+');
    auto first = dyn_cast_or_null<hw::HWModuleOp>(
        SymbolTable::lookupSymbolIn(top, firstName));
    auto second = dyn_cast_or_null<hw::HWModuleOp>(
        SymbolTable::lookupSymbolIn(top, secondName));
    if (!first || !second || first == second) {
      top.emitError("group '") << group << "' is not two models";
      return signalPassFailure();
    }
    if (failed(fuse(netlist, first, second)))
      return signalPassFailure();
    for (auto model : {first, second})
      if (SymbolTable::symbolKnownUseEmpty(model, top))
        model.erase();
  }
}
