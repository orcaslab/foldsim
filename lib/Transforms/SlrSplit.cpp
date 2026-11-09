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
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
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
#include <map>
#include <tuple>

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
  /// crossing=txrx: the module retimed so that every crossing is a register
  /// pair (send, receive); see Passes.td.
  LogicalResult registerCrossings(hw::HWModuleOp module,
                                  SmallVectorImpl<Operation *> &ops,
                                  DenseMap<Operation *, int> &partOf);
};

/// crossing=txrx. Each op gets a lag (host cycles later than in the original
/// module), every memory a read lag and a write lag: an operand arriving
/// earlier than its user's lag goes through that many registers (the
/// difference), a value crossing parts through a send register in its part
/// (a copy of its flop, or a register on it) and a receive register in the
/// reader's (USER_SLL_REG, DONT_TOUCH). A memory's writes may land up to
/// `slack` host cycles later than its reads are taken (never earlier): a
/// row's write and the next read of it are further apart (a fold store's
/// state gap). A loop through registers alone keeps its lag (a positive cycle:
/// the map cut it). The module's outputs all come `latency` host cycles late:
/// the fold's split-latency for the class.
LogicalResult SlrSplitPass::registerCrossings(hw::HWModuleOp module,
                                              SmallVectorImpl<Operation *> &ops,
                                              DenseMap<Operation *, int> &partOf) {
  auto *context = module.getContext();
  auto *body = module.getBodyBlock();
  auto *terminator = body->getTerminator();
  Value clock;
  for (auto arg : body->getArguments())
    if (isa<seq::ClockType>(arg.getType())) {
      clock = arg;
      break;
    }
  if (!clock)
    return module.emitError("slr-split: crossing=txrx needs a clock input");
  // Variables: one per op; a memory's is its read lag, its write lag one more.
  DenseMap<Operation *, unsigned> var;
  DenseMap<Operation *, unsigned> writeVar;
  unsigned n = 0;
  for (auto *op : ops) {
    if (partOf[op] < 0 && !memoryOf(op) && !isa<seq::FirMemOp>(op))
      continue;
    var[op] = n++;
    if (isa<seq::FirMemOp>(op))
      writeVar[op] = n++;
  }
  // The variable an op's lag is: a port's is its memory's (read or write).
  auto lagVar = [&](Operation *op) -> int {
    if (auto *memory = memoryOf(op)) {
      if (isa<seq::FirMemReadOp>(op))
        return var.lookup(memory);
      return writeVar.lookup(memory);
    }
    auto it = var.find(op);
    return it == var.end() ? -1 : int(it->second);
  };
  auto partOfOp = [&](Operation *op) {
    if (auto *memory = memoryOf(op))
      return partOf[memory];
    return partOf[op];
  };
  struct Edge {
    unsigned from, to;
    int weight;
  };
  SmallVector<Edge> edges;
  auto isFlop = [](Value v) {
    return v.getDefiningOp<seq::FirRegOp>() != nullptr;
  };
  auto need = [&](Value v, Operation *user) -> int {
    auto *def = v.getDefiningOp();
    if (!def || partOf[def] < 0 || isConstant(def))
      return 0;
    if (partOfOp(def) == partOfOp(user))
      return 0;
    return isFlop(v) ? 1 : 2;
  };
  for (auto *op : ops) {
    int to = lagVar(op);
    if (to < 0 || isa<seq::FirMemOp>(op))
      continue;
    for (auto operand : op->getOperands()) {
      if (isa<seq::ClockType>(operand.getType()))
        continue;
      auto *def = operand.getDefiningOp();
      if (!def || isConstant(def) || partOf[def] < 0)
        continue;
      if (isa<seq::FirMemOp>(def))
        continue; // the port's memory operand
      int from = lagVar(def);
      if (from < 0)
        continue;
      edges.push_back({unsigned(from), unsigned(to), need(operand, op)});
    }
  }
  // A memory is elastic (its writes may land up to `slack` later than its
  // reads are taken) when it is read combinationally and every port's address
  // is the fold's control: rows by instance (copied control, wiring of it,
  // constants), each row read again only at its instance's next evaluation. Any other memory (a target
  // RAM addressed by data, a buffer read soon after its write) keeps its
  // writes and reads in step.
  auto controlOnly = [&](Value v) {
    SmallVector<Value> work{v};
    DenseSet<Value> seen;
    while (!work.empty()) {
      Value x = work.pop_back_val();
      if (!seen.insert(x).second)
        continue;
      auto *def = x.getDefiningOp();
      if (!def)
        return false;
      if (isConstant(def) || partOf.lookup(def) < 0)
        continue;
      if (!isWiring(def))
        return false;
      for (auto operand : def->getOperands())
        work.push_back(operand);
    }
    return true;
  };
  unsigned elastic = 0, rigid = 0;
  for (auto &[memory, w] : writeVar) {
    unsigned r = var[memory];
    // A store read through a register (block or ultra RAM) is read ahead of
    // its pass, as close after its row's write as the schedule allows
    // (fold-store-lead): never elastic.
    bool isElastic = cast<seq::FirMemOp>(memory).getReadLatency() == 0;
    for (auto *user : memory->getUsers()) {
      Value address;
      if (auto read = dyn_cast<seq::FirMemReadOp>(user))
        address = read.getAddress();
      else if (auto write = dyn_cast<seq::FirMemWriteOp>(user))
        address = write.getAddress();
      else {
        isElastic = false;
        break;
      }
      isElastic &= controlOnly(address);
    }
    (isElastic ? elastic : rigid)++;
    edges.push_back({r, w, 0});
    edges.push_back({w, r, isElastic ? -int(slack) : 0});
  }
  llvm::errs() << "foldsim-slr-split: crossing=txrx: " << elastic
               << " memories elastic (addressed by the fold's control), "
               << rigid << " in step\n";
  // Longest paths from 0 (every lag at least 0); a lag past `latency` plus
  // the slack has gone round a positive cycle.
  SmallVector<int> lag(n, 0);
  bool changed = true;
  int bound = int(latency) + int(slack) + 4;
  for (unsigned it = 0; changed; ++it) {
    changed = false;
    for (auto &e : edges)
      if (lag[e.from] + e.weight > lag[e.to]) {
        lag[e.to] = lag[e.from] + e.weight;
        changed = true;
        if (lag[e.to] > bound) {
          for (auto *op : ops)
            if (lagVar(op) == int(e.to))
              return op->emitError("slr-split: crossing=txrx: no lags within ")
                     << unsigned(latency) << " host cycles: a loop of registers alone "
                                   "crosses parts, or crossings in series pass "
                                   "the latency (this op's lag reached "
                     << lag[e.to] << ")";
          return module.emitError("slr-split: crossing=txrx: no lags");
        }
      }
  }
  int outLag = 0;
  for (auto operand : terminator->getOperands())
    if (auto *def = operand.getDefiningOp())
      if (int v = lagVar(def); v >= 0)
        outLag = std::max(outLag, lag[v]);
  if (outLag > int(latency))
    return module.emitError("slr-split: crossing=txrx: the outputs need ")
           << outLag << " host cycles, past latency=" << unsigned(latency);

  // The registers: chains per (value, delay, part, crossing).
  auto attrs = [&](Operation *reg) {
    sv::addSVAttributes(
        reg, {sv::SVAttributeAttr::get(context, "USER_SLL_REG", R"("TRUE")",
                                       false),
              sv::SVAttributeAttr::get(context, "DONT_TOUCH", R"("TRUE")",
                                       false)});
  };
  std::map<std::tuple<void *, int, int, int>, Value> chains;
  // A send register copied from a flop takes the flop's operands as they are
  // once its own are delayed.
  SmallVector<std::pair<Operation *, Operation *>> clones;
  OpBuilder b(terminator);
  unsigned sent = 0, received = 0, aligned = 0;
  uint64_t crossBits = 0;
  auto nameOf = [&](Value v) -> std::string {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return module.getInputName(arg.getArgNumber()).str();
    if (auto *def = v.getDefiningOp())
      if (auto name = def->getAttrOfType<StringAttr>("name"))
        return name.getValue().str();
    return "split";
  };
  auto newReg = [&](Value next, const Twine &name, int part) -> Operation * {
    auto reg = seq::FirRegOp::create(b, module.getLoc(), next, clock, b.getStringAttr(name));
    partOf[reg] = part;
    ops.push_back(reg);
    return reg;
  };
  // A value delayed by `d` for a reader in part `to`: through a send and a
  // receive register when it crosses, alignment registers otherwise.
  auto delayed = [&](Value v, int d, int to, bool cross) -> Value {
    auto key = std::make_tuple(v.getAsOpaquePointer(), d, to, int(cross));
    if (auto it = chains.find(key); it != chains.end())
      return it->second;
    auto *def = v.getDefiningOp();
    int from = def && partOf.lookup(def) >= 0 ? partOfOp(def) : to;
    std::string base = nameOf(v) + "_lag" + std::to_string(d);
    Value cur = v;
    int left = d;
    if (cross) {
      Operation *tx;
      if (auto reg = v.getDefiningOp<seq::FirRegOp>()) {
        // The send register: a copy of the flop, its own (it drives the
        // line alone), the same cycle.
        tx = b.clone(*reg);
        tx->setAttr("name", b.getStringAttr(base + "_tx"));
        clones.push_back({tx, reg});
        partOf[tx] = from;
        ops.push_back(tx);
      } else {
        tx = newReg(cur, base + "_tx", from);
        --left;
      }
      attrs(tx);
      ++sent;
      crossBits += std::max<int64_t>(0, hw::getBitWidth(v.getType()));
      cur = tx->getResult(0);
      auto *rx = newReg(cur, base + "_rx", to);
      attrs(rx);
      ++received;
      --left;
      cur = rx->getResult(0);
    }
    for (int k = 0; k < left; ++k) {
      cur = newReg(cur, base + "_a" + std::to_string(k), to)->getResult(0);
      ++aligned;
    }
    chains[key] = cur;
    return cur;
  };
  SmallVector<Operation *> users(ops.begin(), ops.end());
  for (auto *op : users) {
    int to = lagVar(op);
    if (to < 0 || isa<seq::FirMemOp>(op))
      continue;
    for (auto &operand : op->getOpOperands()) {
      Value v = operand.get();
      if (isa<seq::ClockType>(v.getType()) || v.getDefiningOp<seq::FirMemOp>())
        continue;
      auto *def = v.getDefiningOp();
      if (def && isConstant(def))
        continue;
      int from = (def && partOf[def] >= 0) ? lagVar(def) : -1;
      int d = lag[to] - (from >= 0 ? lag[from] : 0);
      bool cross = def && partOf[def] >= 0 && need(v, op) > 0;
      if (d <= 0 && !cross)
        continue;
      operand.set(delayed(v, d, partOfOp(op), cross));
    }
  }
  for (auto [tx, reg] : clones)
    for (unsigned i = 0; i < reg->getNumOperands(); ++i)
      tx->setOperand(i, reg->getOperand(i));
  // The outputs, all `latency` late.
  for (auto &operand : terminator->getOpOperands()) {
    Value v = operand.get();
    auto *def = v.getDefiningOp();
    int from = (def && partOf[def] >= 0) ? lagVar(def) : -1;
    int d = int(latency) - (from >= 0 ? lag[from] : 0);
    if (d > 0) {
      int part = def && partOf[def] >= 0 ? partOfOp(def) : 0;
      operand.set(delayed(v, d, part, false));
    }
  }
  llvm::errs() << "foldsim-slr-split: crossing=txrx: " << sent
               << " send and " << received << " receive registers ("
               << crossBits << " bits), " << aligned
               << " alignment registers; outputs " << latency
               << " host cycles late (the most lag " << outLag << ")\n";
  return success();
}

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

  if (crossing == "txrx") {
    if (failed(registerCrossings(module, ops, partOf)))
      return signalPassFailure();
  } else if (crossing != "source") {
    module.emitError("slr-split: crossing must be source or txrx");
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
