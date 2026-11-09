//===- SlrSplit.cpp - split a module's logic into parts, one per SLR -------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A module too large for one SLR (a class copy of a folded simulator) is cut
// into parts, each a module of its own instantiated in its place, so that a
// floorplan can put each part in its own SLR. The parts are given, op by op,
// by a map (foldsim_slr_split.py --apply-map: a min-cut over the module's
// netlist). Every value that crosses from one part to another leaves its
// part from a register (a flop, a registered memory read, a black box's
// registered output): the cut changes no cycle. Logic a part reads from
// another part's logic is copied into it, and the ops the map marks as
// replicated (constants, the fold's counters and their logic) are copied
// into every part that reads them. See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

namespace foldsim {
#define GEN_PASS_DEF_SLRSPLIT
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

bool isConstant(Operation *op);

/// Wiring: an op whose result bits are some of its operands' bits, no logic.
bool isWiring(Operation *op) {
  if (isa<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp, hw::BitcastOp,
          hw::StructExtractOp, hw::StructCreateOp, hw::ArrayCreateOp,
          hw::ArrayConcatOp, hw::WireOp>(op))
    return true;
  // An element or slice at a constant index.
  if (auto get = dyn_cast<hw::ArrayGetOp>(op))
    return get.getIndex().getDefiningOp<hw::ConstantOp>() != nullptr;
  if (auto slice = dyn_cast<hw::ArraySliceOp>(op))
    return slice.getLowIndex().getDefiningOp<hw::ConstantOp>() != nullptr;
  return false;
}

/// A value that leaves its op from a register: a flop, a memory's registered
/// read, or an instance's output (a black box registers its outputs; another
/// module is a part of its own), or wiring of such values (a register's
/// slice is still the register's output).
bool registeredSource(Value value, DenseMap<Value, bool> &memo) {
  auto *op = value.getDefiningOp();
  if (!op)
    return true;
  if (auto it = memo.find(value); it != memo.end())
    return it->second;
  memo[value] = false;
  bool r = false;
  if (isa<seq::FirRegOp, seq::CompRegOp, hw::InstanceOp>(op))
    r = true;
  else if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
    if (auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>())
      r = memory.getReadLatency() > 0;
  } else if (isWiring(op)) {
    r = true;
    for (auto operand : op->getOperands()) {
      auto *def = operand.getDefiningOp();
      if (def && isConstant(def))
        continue;
      if (!def || !registeredSource(operand, memo)) {
        r = def != nullptr ? false : r;
        if (!r)
          break;
      }
    }
  }
  memo[value] = r;
  return r;
}

bool isConstant(Operation *op) {
  return isa<hw::ConstantOp, hw::AggregateConstantOp>(op);
}

/// Logic that may be copied into another part: no state, no side effects.
bool copyableLogic(Operation *op) {
  if (op->getNumRegions() || isa<seq::FirRegOp, seq::CompRegOp, seq::FirMemOp,
                                 seq::FirMemReadOp, seq::FirMemWriteOp,
                                 seq::FirMemReadWriteOp, hw::InstanceOp>(op))
    return false;
  return isMemoryEffectFree(op);
}

Operation *memoryOf(Operation *op) {
  if (auto read = dyn_cast<seq::FirMemReadOp>(op))
    return read.getMemory().getDefiningOp();
  if (auto write = dyn_cast<seq::FirMemWriteOp>(op))
    return write.getMemory().getDefiningOp();
  if (auto rw = dyn_cast<seq::FirMemReadWriteOp>(op))
    return rw.getMemory().getDefiningOp();
  return nullptr;
}

struct SlrSplitPass : public foldsim::impl::SlrSplitBase<SlrSplitPass> {
  using SlrSplitBase::SlrSplitBase;
  void runOnOperation() override;
};

void SlrSplitPass::runOnOperation() {
  auto top = getOperation();
  auto *context = top.getContext();
  auto module = top.lookupSymbol<hw::HWModuleOp>(moduleName);
  if (!module) {
    top.emitError("slr-split: no module '") << moduleName << "'";
    return signalPassFailure();
  }
  if (parts < 2) {
    module.emitError("slr-split: parts must be at least 2");
    return signalPassFailure();
  }
  auto *body = module.getBodyBlock();
  auto *terminator = body->getTerminator();
  SmallVector<Operation *> ops;
  for (auto &op : body->without_terminator())
    ops.push_back(&op);

  // The map: a part (or -1: copied into every part that reads it) per op, in
  // the body's order.
  SmallVector<int> given;
  {
    std::ifstream in(mapPath);
    if (!in) {
      module.emitError("slr-split: cannot read map '") << mapPath << "'";
      return signalPassFailure();
    }
    long v;
    while (in >> v)
      given.push_back(int(v));
  }
  if (given.size() != ops.size()) {
    module.emitError("slr-split: the map has ")
        << given.size() << " entries for " << ops.size() << " ops";
    return signalPassFailure();
  }
  DenseMap<Operation *, int> partOf;
  for (auto [k, op] : llvm::enumerate(ops)) {
    int p = isConstant(op) ? -1 : given[k];
    if (p < -1 || p >= int(parts)) {
      op->emitError("slr-split: part ") << p << " of " << parts;
      return signalPassFailure();
    }
    partOf[op] = p;
  }
  // A memory's ports are where the memory is.
  for (auto *op : ops)
    if (auto *memory = memoryOf(op))
      if (partOf[memory] != partOf[op]) {
        op->emitError("slr-split: a port of a memory in another part (")
            << partOf[op] << ", its memory " << partOf[memory] << ")";
        return signalPassFailure();
      }

  // Each part's ops: its own, the replicated ones it reads, and copies of the
  // logic of other parts it reads.
  SmallVector<DenseSet<Operation *>> in(parts);
  DenseMap<Value, bool> regMemo;
  unsigned copiedLogic = 0;
  bool failedClosure = false;
  for (unsigned p = 0; p < parts; ++p) {
    SmallVector<Operation *> work;
    auto add = [&](Operation *op) {
      if (in[p].insert(op).second)
        work.push_back(op);
    };
    for (auto *op : ops)
      if (partOf[op] == int(p))
        add(op);
    while (!work.empty()) {
      auto *op = work.pop_back_val();
      // A copied memory takes its writes along.
      if (isa<seq::FirMemOp>(op) && partOf[op] < 0)
        for (auto *user : op->getUsers())
          if (isa<seq::FirMemWriteOp, seq::FirMemReadWriteOp>(user))
            add(user);
      if (auto *memory = memoryOf(op))
        add(memory);
      for (auto operand : op->getOperands()) {
        auto *def = operand.getDefiningOp();
        if (!def || in[p].contains(def))
          continue;
        int q = partOf[def];
        if (q < 0) {
          add(def);
          continue;
        }
        if (registeredSource(operand, regMemo))
          continue;
        if (!copyLogic || !copyableLogic(def)) {
          def->emitError("slr-split: a value of part ")
              << q << " read in part " << p
              << " is not from a register (a combinational "
              << (isa<seq::FirMemReadOp>(def) ? "memory read" : "value")
              << ")";
          failedClosure = true;
          continue;
        }
        ++copiedLogic;
        add(def);
      }
    }
  }
  if (failedClosure)
    return signalPassFailure();
  // The replicated ops must be closed: a register copied into a part reads
  // only what the part has or registers of other parts.
  // (Ensured above: every operand of an op in a part is in it, a register
  // of another part, or a module input.)

  // Each part's inputs (values from outside it) and outputs (its own values
  // read outside it: by another part or the module's output).
  SmallVector<SmallVector<Value>> inputs(parts), outputs(parts);
  DenseSet<Value> needed;
  for (unsigned p = 0; p < parts; ++p) {
    DenseSet<Value> seen;
    for (auto *op : ops) {
      if (!in[p].contains(op))
        continue;
      for (auto operand : op->getOperands()) {
        auto *def = operand.getDefiningOp();
        if (def && in[p].contains(def))
          continue;
        if (seen.insert(operand).second) {
          inputs[p].push_back(operand);
          needed.insert(operand);
        }
      }
    }
  }
  for (auto operand : terminator->getOperands())
    needed.insert(operand);
  SmallVector<DenseSet<Value>> inputSet(parts);
  for (unsigned p = 0; p < parts; ++p)
    inputSet[p].insert(inputs[p].begin(), inputs[p].end());
  unsigned cutValues = 0;
  uint64_t cutBits = 0;
  for (unsigned p = 0; p < parts; ++p)
    for (auto *op : ops)
      if (partOf[op] == int(p))
        for (auto result : op->getResults())
          if (needed.contains(result)) {
            outputs[p].push_back(result);
            bool cut = false;
            for (unsigned q = 0; q < parts; ++q)
              cut |= q != p && inputSet[q].contains(result);
            if (cut) {
              ++cutValues;
              cutBits += std::max<int64_t>(0, hw::getBitWidth(result.getType()));
            }
          }
  // A value the module's output reads from a replicated op: from part 0.
  for (auto operand : terminator->getOperands())
    if (auto *def = operand.getDefiningOp())
      if (partOf[def] < 0 && !llvm::is_contained(outputs[0], operand)) {
        in[0].insert(def);
        outputs[0].push_back(operand);
      }

  auto nameOf = [&](Value value) -> std::string {
    if (auto arg = dyn_cast<BlockArgument>(value))
      return module.getInputName(arg.getArgNumber()).str();
    auto *op = value.getDefiningOp();
    if (auto instance = dyn_cast<hw::InstanceOp>(op))
      return (instance.getInstanceName() + "_" +
              instance.getOutputName(cast<OpResult>(value).getResultNumber())
                  .getValue())
          .str();
    if (auto name = op->getAttrOfType<StringAttr>("name"))
      return name.getValue().str();
    return "";
  };

  // The parts' modules and their instances (at first on the original
  // values, rewired below).
  SmallVector<hw::InstanceOp> instances;
  OpBuilder ib(terminator);
  for (unsigned p = 0; p < parts; ++p) {
    SmallVector<hw::PortInfo> ports;
    llvm::StringSet<> names;
    auto addPort = [&](Value value, bool isInput, size_t argNum) {
      std::string base = nameOf(value);
      if (base.empty())
        base = isInput ? "in" : "out";
      std::string name = base;
      for (unsigned n = 1; !names.insert(name).second; ++n)
        name = base + "_" + std::to_string(n);
      hw::PortInfo port;
      port.name = StringAttr::get(context, name);
      port.type = value.getType();
      port.dir = isInput ? hw::ModulePort::Direction::Input
                         : hw::ModulePort::Direction::Output;
      port.argNum = argNum;
      ports.push_back(port);
    };
    for (auto [i, value] : llvm::enumerate(inputs[p]))
      addPort(value, true, i);
    for (auto [i, value] : llvm::enumerate(outputs[p]))
      addPort(value, false, i);
    OpBuilder mb(module);
    auto partModule = hw::HWModuleOp::create(
        mb, module.getLoc(),
        StringAttr::get(context, moduleName + "_p" + Twine(p)),
        hw::ModulePortInfo(ports));
    partModule.setPrivate();
    auto *partBody = partModule.getBodyBlock();
    partBody->getTerminator()->erase();
    IRMapping map;
    for (auto [value, arg] : llvm::zip(inputs[p], partBody->getArguments()))
      map.map(value, arg);
    OpBuilder pb = OpBuilder::atBlockEnd(partBody);
    SmallVector<Operation *> clones;
    for (auto *op : ops)
      if (in[p].contains(op))
        clones.push_back(pb.clone(*op, map));
    // A graph region: an operand may be defined further down.
    for (auto *clone : clones)
      for (auto &operand : clone->getOpOperands())
        if (auto mapped = map.lookupOrNull(operand.get()))
          operand.set(mapped);
    SmallVector<Value> results;
    for (auto value : outputs[p])
      results.push_back(map.lookup(value));
    hw::OutputOp::create(pb, module.getLoc(), results);
    instances.push_back(hw::InstanceOp::create(
        ib, module.getLoc(), partModule,
        StringAttr::get(context, prefix + moduleName + "_p" + Twine(p)),
        inputs[p]));
  }
  // Each part's outputs replace the original values: in the instances'
  // operands and the module's output.
  DenseMap<Value, Value> replacement;
  for (unsigned p = 0; p < parts; ++p)
    for (auto [value, result] : llvm::zip(outputs[p], instances[p].getResults()))
      replacement[value] = result;
  auto rewire = [&](Operation *user) {
    for (auto &operand : user->getOpOperands())
      if (auto it = replacement.find(operand.get()); it != replacement.end())
        operand.set(it->second);
  };
  for (auto instance : instances)
    rewire(instance);
  rewire(terminator);
  for (auto *op : ops)
    op->dropAllReferences();
  for (auto *op : ops)
    op->erase();
  numCutValues += cutValues;
  numCutBits += cutBits;
  numCopiedLogic += copiedLogic;
  llvm::errs() << "foldsim-slr-split: " << moduleName << ": " << parts
               << " parts, " << cutValues << " values (" << cutBits
               << " bits) cross from a register, " << copiedLogic
               << " ops of logic copied into a reading part\n";
  for (unsigned p = 0; p < parts; ++p)
    llvm::errs() << "  part " << p << ": " << in[p].size() << " ops, "
                 << inputs[p].size() << " inputs, " << outputs[p].size()
                 << " outputs\n";
}

} // namespace
