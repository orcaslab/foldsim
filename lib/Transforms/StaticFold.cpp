//===- StaticFold.cpp - A statically scheduled folded simulator ------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Folds a netlist of models into a simulator whose schedule is fixed at
// compile time: a target cycle is a frame of II host cycles, and in each slot
// of a frame each class's copy evaluates one of its instances in one pass.
// Nothing is decided at run time but whether the next frame starts.
//
//===----------------------------------------------------------------------===//

#include "FAME5.h"
#include "Pipelines.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "foldsim/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>

namespace foldsim {
#define GEN_PASS_DEF_FOLDSTATIC
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

struct FoldStaticPass : public foldsim::impl::FoldStaticBase<FoldStaticPass> {
  using FoldStaticBase::FoldStaticBase;
  void runOnOperation() override;
};

/// Bits of a consumer's port: bits of one instance's output, a netlist input,
/// or a constant.
struct Piece {
  /// The producer, or -1 for a constant.
  int producer = -1;
  unsigned inst = 0, low = 0, width = 0;
  APInt constant;
};
using Pieces = SmallVector<Piece>;

/// A model's fixed-latency pipelines, folded: each simulated by an unthreaded
/// copy on the host clock, its results kept per instance in a store that is
/// target state (as `foldsim-fame5` with `fold-pipelines` folds them).
struct Fold {
  seq::FirMemOp store;
  Value threadedStore;
  Value slot, thread, valid, data;
};
struct Folding {
  /// The unthreaded copies' ops: on the host clock, not target state.
  DenseSet<Operation *> physical;
  SmallVector<Fold> folds;
  DenseMap<Operation *, unsigned> foldOfStore;
  std::unique_ptr<BackedgeBuilder> edges;
  Backedge physClock, thread, edge, reset;
};

/// The passes of an instance's evaluation in a frame.
enum Pass : unsigned { kReady = 0, kState = 1 };

struct Class {
  hw::HWModuleOp model;
  std::string name;
  SmallVector<hw::InstanceOp> instances;
  SmallVector<std::string> inputNames, outputNames;
  SmallVector<Type> inputTypes, outputTypes;
  int clockInput = -1;
  /// The class it was split from by copies, or -1.
  int splitFrom = -1;
  /// Nested folding: the class of its instances' parents (-1 for a root),
  /// and its depth in the class tree.
  int parent = -1;
  unsigned depth = 0;
  /// Per input: driven by constants alone in every instance.
  SmallVector<bool> staticInput;
  /// Per output: late (depends on an input, a memory's read data or a black
  /// box), and the inputs it depends on combinationally.
  SmallVector<bool> late;
  SmallVector<llvm::BitVector> deps;
  /// Per output, its producer; per input, its consumer (-1 for the clock).
  SmallVector<int> producerOf, consumerOf;
  /// Logic levels, estimated: from each input to the state it reaches, and
  /// to each output from the state and inputs it comes from; and whether an
  /// input reaches an output combinationally.
  SmallVector<int> inputDepth, outputDepth;
  llvm::BitVector inputToOutput;
  /// Whether its memories include a registered read: its state is then read
  /// a host cycle later after a state pass.
  bool syncReads = false;
  /// The schedule: whether it has a ready pass, and each instance's slots.
  bool hasReady = false;
  SmallVector<int> slot[2];
  /// Its folded pipelines, if any.
  std::unique_ptr<Folding> folding;
  /// Built: the copy and its instance in the simulator.
  hw::HWModuleOp copy;
  unsigned threads() const { return instances.size(); }
};

/// A source of bits: an output of a model, or an input of the netlist.
struct Producer {
  std::string name;
  /// The class, or -1 for an input of the netlist; the output or input.
  int cls = -1;
  unsigned port = 0, width = 0;
  /// For a class output: late, and then whether written at the ready pass.
  bool late = false, atReady = false;
  /// Read by any consumer.
  bool read = false;
};

/// A sink of bits: an input of a model, or an output of the netlist.
struct Consumer {
  std::string name;
  /// The class, or -1 for an output of the netlist; the input or output.
  int cls = -1;
  unsigned port = 0, width = 0;
  /// Each instance's bits, low ones first.
  SmallVector<Pieces> pieces;
};

/// Host cycles from a link's write to the first slot that reads it: the write
/// lands at the end of its slot, and the read register is loaded a slot early.
/// Forwarded, a write reaches a read register loaded in its own slot.
static constexpr int kLinkLatency = 2, kForwardedLinkLatency = 1;

/// One evaluation of a frame: an instance of a class in a pass.
struct Node {
  unsigned cls, inst;
  Pass pass;
};

class StaticFolder {
public:
  StaticFolder(ModuleOp top, hw::HWModuleOp netlist, StringRef simName,
               unsigned regionBits, StringRef report,
               ArrayRef<std::string> copies, bool nest, bool scheduleOnly,
               bool oneCycleLinks, bool overlapFrames, bool modulo,
               bool interleaveCopies, bool zeroLatencyLinks)
      : top(top), netlist(netlist), simName(simName), regionBits(regionBits),
        reportPath(report), copies(copies), nest(nest),
        scheduleOnly(scheduleOnly), oneCycleLinks(oneCycleLinks),
        overlapFrames(overlapFrames), modulo(modulo),
        interleaveCopies(interleaveCopies), zeroLatencyLinks(zeroLatencyLinks),
        linkLatency(oneCycleLinks ? kForwardedLinkLatency : kLinkLatency) {}
  LogicalResult run();

private:
  void inlineNetlists();
  LogicalResult collect();
  LogicalResult splitCopies();
  LogicalResult nestInstances();
  LogicalResult decompose(Value value, unsigned low, unsigned width,
                          Pieces &pieces);
  LogicalResult analyze(Class &cls);
  LogicalResult schedule();
  /// Places every evaluation of the given choice of ready passes; II, or 0
  /// if the evaluations depend on each other in a cycle.
  unsigned place(ArrayRef<bool> ready, SmallVectorImpl<SmallVector<int>> &slots,
                 SmallVectorImpl<bool> &atReady);
  /// Places the evaluations with consecutive frames overlapped: the shortest
  /// frame at which every copy's evaluations fit, modulo the frame, and every
  /// dependence across frames holds; 0 if none does.
  unsigned placeModulo(
      ArrayRef<Node> nodes,
      ArrayRef<SmallVector<std::pair<unsigned, int>>> preds,
      ArrayRef<SmallVector<std::pair<unsigned, int>>> succs,
      ArrayRef<int> height, ArrayRef<bool> atReady,
      SmallVectorImpl<SmallVector<int>> &slots);
  LogicalResult buildCopy(Class &cls);
  LogicalResult buildSim();

  ModuleOp top;
  hw::HWModuleOp netlist;
  StringRef simName;
  unsigned regionBits;
  StringRef reportPath;
  ArrayRef<std::string> copies;
  bool nest;
  /// Nested folding: the netlist each instance was inlined from (a path of
  /// instance names), and the instances that are a netlist's own logic.
  DenseMap<Operation *, std::string> containerOf;
  DenseSet<Operation *> logicInstances;
  bool scheduleOnly;
  bool oneCycleLinks, overlapFrames, modulo, interleaveCopies;
  bool zeroLatencyLinks;
  /// Whether producer `p`'s link to class `c`'s input may be taken in the
  /// producer's own host cycle.
  bool zeroLink(unsigned p, unsigned c, unsigned input) const;
  /// Host cycles from a producer's slot to its reader's.
  int linkLatency;
  std::vector<Class> classes;
  std::vector<Producer> producers;
  std::vector<Consumer> consumers;
  DenseMap<Operation *, std::pair<unsigned, unsigned>> instanceOf;
  llvm::SmallPtrSet<Operation *, 4> inlined;
  /// The producer of each input of the netlist, or -1 for a clock.
  SmallVector<int> inputProducer;
  /// The frame length, and the slot of each netlist output's token.
  unsigned ii = 0;
  SmallVector<int> outputSlot;
  SmallVector<bool> outputEarly;
};

} // namespace

//===----------------------------------------------------------------------===//
// The netlist
//===----------------------------------------------------------------------===//

/// Separates the instance names of a container path: instance names may
/// hold '/'.
static constexpr char kPathSeparator = '\x1f';

void StaticFolder::inlineNetlists() {
  auto *body = netlist.getBodyBlock();
  for (auto instance : body->getOps<hw::InstanceOp>()) {
    containerOf[instance] = "";
    if (instance.getInstanceName() == "logic")
      logicInstances.insert(instance);
  }
  for (bool changed = true; changed;) {
    changed = false;
    for (auto instance :
         llvm::make_early_inc_range(body->getOps<hw::InstanceOp>())) {
      auto callee =
          dyn_cast_or_null<hw::HWModuleOp>(SymbolTable::lookupSymbolIn(
              top, instance.getModuleNameAttr().getAttr()));
      if (!callee || !callee->hasAttr("foldsim.netlist"))
        continue;
      inlined.insert(callee);
      auto *calleeBody = callee.getBodyBlock();
      IRMapping map;
      for (auto [arg, operand] :
           llvm::zip(calleeBody->getArguments(), instance.getInputs()))
        map.map(arg, operand);
      OpBuilder builder(instance);
      SmallVector<Operation *> clones;
      for (auto &op : calleeBody->without_terminator()) {
        auto *clone = builder.clone(op, map);
        if (auto inner = dyn_cast<hw::InstanceOp>(clone)) {
          containerOf[inner] = containerOf[instance] + kPathSeparator +
                               instance.getInstanceName().str();
          if (inner.getInstanceName() == "logic")
            logicInstances.insert(inner);
          inner.setInstanceNameAttr(builder.getStringAttr(
              instance.getInstanceName() + "_" + inner.getInstanceName()));
        }
        clones.push_back(clone);
      }
      // A graph region: an operand may be defined further down.
      for (auto *clone : clones)
        for (auto &operand : clone->getOpOperands())
          if (auto mapped = map.lookupOrNull(operand.get()))
            operand.set(mapped);
      for (auto [result, output] :
           llvm::zip(instance.getResults(),
                     calleeBody->getTerminator()->getOperands()))
        result.replaceAllUsesWith(map.lookupOrDefault(output));
      containerOf.erase(instance);
      logicInstances.erase(instance);
      instance.erase();
      changed = true;
    }
  }
}

/// The bits of an aggregate constant of `type`, the first field or element
/// the most significant, as `hw.bitcast` lays them out.
static APInt aggregateBits(Attribute attr, Type type) {
  if (auto integer = dyn_cast<IntegerAttr>(attr))
    return integer.getValue().zextOrTrunc(hw::getBitWidth(type));
  auto fields = cast<ArrayAttr>(attr);
  SmallVector<Type> types;
  if (auto structType = hw::type_dyn_cast<hw::StructType>(type))
    for (auto field : structType.getElements())
      types.push_back(field.type);
  else
    types.assign(fields.size(),
                 hw::type_cast<hw::ArrayType>(type).getElementType());
  std::optional<APInt> bits;
  for (auto [field, fieldType] : llvm::zip(fields, types)) {
    auto fieldBits = aggregateBits(field, fieldType);
    bits = bits ? bits->concat(fieldBits) : fieldBits;
  }
  return *bits;
}

LogicalResult StaticFolder::decompose(Value value, unsigned low,
                                      unsigned width, Pieces &pieces) {
  if (width == 0)
    return success();
  auto append = [&](Piece piece) {
    if (!pieces.empty()) {
      auto &last = pieces.back();
      if (piece.producer < 0 && last.producer < 0) {
        last.constant = piece.constant.concat(last.constant);
        last.width += piece.width;
        return;
      }
      if (piece.producer >= 0 && piece.producer == last.producer &&
          piece.inst == last.inst && last.low + last.width == piece.low) {
        last.width += piece.width;
        return;
      }
    }
    pieces.push_back(piece);
  };
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    int producer = inputProducer[arg.getArgNumber()];
    if (producer < 0)
      return netlist.emitError("a clock is not bits of a channel");
    append(Piece{producer, 0, low, width, APInt()});
    return success();
  }
  auto *op = value.getDefiningOp();
  if (auto instance = dyn_cast<hw::InstanceOp>(op)) {
    auto [c, inst] = instanceOf.lookup(instance);
    auto output = cast<OpResult>(value).getResultNumber();
    append(Piece{classes[c].producerOf[output], inst, low, width, APInt()});
    return success();
  }
  if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
    append(Piece{-1, 0, 0, width, constant.getValue().extractBits(width, low)});
    return success();
  }
  if (auto extract = dyn_cast<comb::ExtractOp>(op))
    return decompose(extract.getInput(), low + extract.getLowBit(), width,
                     pieces);
  if (auto bitcast = dyn_cast<hw::BitcastOp>(op))
    return decompose(bitcast.getInput(), low, width, pieces);
  if (auto replicate = dyn_cast<comb::ReplicateOp>(op)) {
    unsigned inputWidth = hw::getBitWidth(replicate.getInput().getType());
    for (unsigned bit = low; bit < low + width;) {
      unsigned offset = bit % inputWidth;
      unsigned run = std::min(inputWidth - offset, low + width - bit);
      if (failed(decompose(replicate.getInput(), offset, run, pieces)))
        return failure();
      bit += run;
    }
    return success();
  }
  if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
    unsigned base = 0;
    for (auto input : llvm::reverse(concat.getInputs())) {
      unsigned inputWidth = hw::getBitWidth(input.getType());
      unsigned lo = std::max(low, base);
      unsigned hi = std::min(low + width, base + inputWidth);
      if (lo < hi && failed(decompose(input, lo - base, hi - lo, pieces)))
        return failure();
      base += inputWidth;
    }
    return success();
  }
  return op->emitError("not wiring of a netlist");
}

LogicalResult StaticFolder::collect() {
  auto *body = netlist.getBodyBlock();
  for (auto aggregate :
       llvm::make_early_inc_range(body->getOps<hw::AggregateConstantOp>())) {
    OpBuilder builder(aggregate);
    auto bits = aggregateBits(aggregate.getFieldsAttr(), aggregate.getType());
    Value value = hw::ConstantOp::create(builder, aggregate.getLoc(), bits);
    value = hw::BitcastOp::create(builder, aggregate.getLoc(),
                                  aggregate.getType(), value);
    aggregate.replaceAllUsesWith(value);
    aggregate.erase();
  }
  DenseMap<Operation *, unsigned> classOf;
  for (auto &op : body->without_terminator()) {
    auto instance = dyn_cast<hw::InstanceOp>(op);
    if (!instance) {
      if (!isa<hw::ConstantOp, comb::ExtractOp, comb::ConcatOp,
               comb::ReplicateOp, hw::BitcastOp>(op))
        return op.emitError("not wiring or a model in a netlist");
      continue;
    }
    auto model = dyn_cast_or_null<hw::HWModuleOp>(SymbolTable::lookupSymbolIn(
        top, instance.getModuleNameAttr().getAttr()));
    if (!model)
      return instance.emitError("a model must be a module");
    auto [it, inserted] = classOf.try_emplace(model, classes.size());
    if (inserted) {
      auto &cls = classes.emplace_back();
      cls.model = model;
      cls.name = model.getModuleName().str();
      for (auto [i, name] : llvm::enumerate(instance.getArgNames())) {
        cls.inputNames.push_back(cast<StringAttr>(name).getValue().str());
        auto type = instance.getOperand(i).getType();
        cls.inputTypes.push_back(type);
        if (isa<seq::ClockType>(type)) {
          if (cls.clockInput >= 0)
            return instance.emitError("a model has a single clock input");
          cls.clockInput = i;
        }
      }
      for (auto [o, name] : llvm::enumerate(instance.getResultNames())) {
        cls.outputNames.push_back(cast<StringAttr>(name).getValue().str());
        cls.outputTypes.push_back(instance.getResult(o).getType());
      }
      if (cls.clockInput < 0)
        return instance.emitError("a model has a clock input");
    }
    auto &cls = classes[it->second];
    instanceOf[instance] = {it->second, cls.instances.size()};
    cls.instances.push_back(instance);
  }
  if (nest && failed(nestInstances()))
    return failure();
  if (failed(splitCopies()))
    return failure();

  // The producers: the netlist's inputs, and the models' outputs.
  for (auto [index, arg] : llvm::enumerate(body->getArguments())) {
    if (isa<seq::ClockType>(arg.getType())) {
      inputProducer.push_back(-1);
      continue;
    }
    inputProducer.push_back(producers.size());
    auto &producer = producers.emplace_back();
    producer.name = netlist.getInputName(index).str();
    producer.port = index;
    producer.width = hw::getBitWidth(arg.getType());
  }
  for (auto [c, cls] : llvm::enumerate(classes)) {
    for (auto [o, name] : llvm::enumerate(cls.outputNames)) {
      cls.producerOf.push_back(producers.size());
      auto &producer = producers.emplace_back();
      producer.name = cls.name + "_" + name;
      producer.cls = c;
      producer.port = o;
      producer.width = hw::getBitWidth(cls.outputTypes[o]);
    }
  }

  // The consumers: the models' inputs, and the netlist's outputs.
  for (auto [c, cls] : llvm::enumerate(classes)) {
    for (auto [i, name] : llvm::enumerate(cls.inputNames)) {
      if (int(i) == cls.clockInput) {
        cls.consumerOf.push_back(-1);
        continue;
      }
      cls.consumerOf.push_back(consumers.size());
      auto &consumer = consumers.emplace_back();
      consumer.name = cls.name + "_" + name;
      consumer.cls = c;
      consumer.port = i;
      consumer.width = hw::getBitWidth(cls.inputTypes[i]);
      for (auto instance : cls.instances)
        if (failed(decompose(instance.getOperand(i), 0, consumer.width,
                             consumer.pieces.emplace_back())))
          return failure();
    }
  }
  auto *output = body->getTerminator();
  for (auto [index, operand] : llvm::enumerate(output->getOperands())) {
    auto &consumer = consumers.emplace_back();
    consumer.name = netlist.getOutputName(index).str();
    consumer.port = index;
    consumer.width = hw::getBitWidth(operand.getType());
    if (failed(decompose(operand, 0, consumer.width,
                         consumer.pieces.emplace_back())))
      return failure();
  }
  for (auto &consumer : consumers)
    for (auto &pieces : consumer.pieces)
      for (auto &piece : pieces)
        if (piece.producer >= 0)
          producers[piece.producer].read = true;
  return success();
}

LogicalResult StaticFolder::splitCopies() {
  llvm::StringMap<unsigned> copiesOf;
  for (auto &entry : copies) {
    auto split = StringRef(entry).split(':');
    StringRef name = split.first, count = split.second;
    unsigned k = 0;
    if (count.getAsInteger(10, k) || k == 0)
      return netlist.emitError("copies: '") << entry
                                            << "' is not <model>:<count>";
    if (llvm::none_of(classes, [&](auto &cls) { return cls.name == name; }))
      return netlist.emitError("copies: no model '") << name << "'";
    copiesOf[name] = k;
  }
  unsigned numClasses = classes.size();
  for (unsigned c = 0; c < numClasses; ++c) {
    unsigned k = copiesOf.lookup(classes[c].name);
    if (k <= 1)
      continue;
    unsigned n = classes[c].instances.size();
    if (n % k)
      return classes[c].instances[0].emitError("copies: ")
             << n << " instances of '" << classes[c].name
             << "' do not split into " << k << " copies";
    unsigned per = n / k;
    auto instances = classes[c].instances;
    // Copy j's t-th instance: contiguous, or every k-th.
    auto pick = [&](unsigned j, unsigned t) {
      return instances[interleaveCopies ? t * k + j : j * per + t];
    };
    classes[c].instances.clear();
    for (unsigned t = 0; t < per; ++t) {
      classes[c].instances.push_back(pick(0, t));
      instanceOf[pick(0, t)] = {c, t};
    }
    for (unsigned j = 1; j < k; ++j) {
      // Each copy is built from a model of its own.
      auto model = classes[c].model;
      OpBuilder b(model);
      b.setInsertionPointAfter(model);
      auto clone = cast<hw::HWModuleOp>(b.clone(*model));
      std::string name = classes[c].name + "_copy" + std::to_string(j);
      clone.setSymName(name);
      Class copy;
      copy.model = clone;
      copy.name = name;
      copy.inputNames = classes[c].inputNames;
      copy.outputNames = classes[c].outputNames;
      copy.inputTypes = classes[c].inputTypes;
      copy.outputTypes = classes[c].outputTypes;
      copy.clockInput = classes[c].clockInput;
      copy.splitFrom = c;
      copy.parent = classes[c].parent;
      copy.depth = classes[c].depth;
      unsigned index = classes.size();
      for (unsigned t = 0; t < per; ++t) {
        auto instance = pick(j, t);
        copy.instances.push_back(instance);
        instanceOf[instance] = {index, t};
      }
      classes.push_back(std::move(copy));
    }
  }
  return success();
}

LogicalResult StaticFolder::nestInstances() {
  // Each instance's parent: the logic of the netlist it was inlined from, or,
  // for a netlist's logic, the logic of the netlist that one was inlined from.
  llvm::StringMap<Operation *> logicAt;
  for (auto &cls : classes)
    for (auto instance : cls.instances)
      if (logicInstances.contains(instance))
        logicAt[containerOf.lookup(instance)] = instance;
  auto up = [](StringRef path) {
    auto pos = path.rfind(kPathSeparator);
    return pos == StringRef::npos ? StringRef() : path.substr(0, pos);
  };
  DenseMap<Operation *, Operation *> parentOf;
  for (auto &cls : classes)
    for (auto instance : cls.instances) {
      std::string path = containerOf.lookup(instance);
      if (logicInstances.contains(instance)) {
        if (path.empty())
          continue;
        path = up(path).str();
      }
      if (auto *parent = logicAt.lookup(path); parent && parent != instance)
        parentOf[instance] = parent;
    }
  // A class's parents are all of one class, and the classes form a tree.
  for (auto [c, cls] : llvm::enumerate(classes)) {
    for (auto [t, instance] : llvm::enumerate(cls.instances)) {
      auto *parent = parentOf.lookup(instance);
      int pc = parent ? int(instanceOf.lookup(parent).first) : -1;
      if (t > 0 && pc != cls.parent)
        return instance.emitError("nest: the instances of '")
               << cls.name << "' have parents of different classes";
      cls.parent = pc;
    }
  }
  SmallVector<unsigned> order;
  for (auto [c, cls] : llvm::enumerate(classes)) {
    unsigned d = 0;
    for (int p = cls.parent; p >= 0; p = classes[p].parent)
      if (++d > classes.size())
        return netlist.emitError("nest: classes nested in a cycle");
    cls.depth = d;
    order.push_back(c);
  }
  llvm::stable_sort(order, [&](unsigned x, unsigned y) {
    return classes[x].depth < classes[y].depth;
  });
  // Each class's instances numbered by their parents' numbers, so that the
  // children of a parent are contiguous (a copy takes whole parents).
  for (auto c : order) {
    auto &cls = classes[c];
    if (cls.parent >= 0)
      llvm::stable_sort(cls.instances, [&](hw::InstanceOp x, hw::InstanceOp y) {
        return instanceOf.lookup(parentOf.lookup(x)).second <
               instanceOf.lookup(parentOf.lookup(y)).second;
      });
    for (auto [t, instance] : llvm::enumerate(cls.instances))
      instanceOf[instance] = {unsigned(c), unsigned(t)};
  }
  return success();
}

//===----------------------------------------------------------------------===//
// What each output depends on
//===----------------------------------------------------------------------===//

/// The logic levels a link taken in its producer's host cycle may span: the
/// logic screen's gate.
static constexpr int kZeroLinkLevels = 12;

/// An operation's logic levels, estimated: none for wiring, two for an adder
/// or a wide comparison, four for a multiplier or divider, one otherwise.
static int logicLevels(Operation *op) {
  if (isa<hw::ConstantOp, hw::AggregateConstantOp, comb::ExtractOp,
          comb::ConcatOp, comb::ReplicateOp, hw::BitcastOp, hw::StructCreateOp,
          hw::StructExtractOp, hw::StructExplodeOp, hw::ArrayCreateOp,
          hw::ArraySliceOp, hw::ArrayConcatOp, hw::WireOp>(op))
    return 0;
  if (isa<comb::MulOp, comb::DivUOp, comb::DivSOp, comb::ModUOp,
          comb::ModSOp>(op))
    return 4;
  if (isa<comb::AddOp, comb::SubOp>(op))
    return 2;
  if (auto cmp = dyn_cast<comb::ICmpOp>(op))
    return hw::getBitWidth(cmp.getLhs().getType()) > 6 ? 2 : 1;
  if (auto get = dyn_cast<hw::ArrayGetOp>(op)) {
    auto size = hw::type_cast<hw::ArrayType>(get.getInput().getType()).getNumElements();
    return std::max(1u, (llvm::Log2_64_Ceil(size) + 1) / 2);
  }
  return 1;
}

LogicalResult StaticFolder::analyze(Class &cls) {
  auto *body = cls.model.getBodyBlock();
  unsigned numInputs = cls.inputNames.size();
  // An input is static if every instance drives it with constants alone.
  for (unsigned i = 0; i < numInputs; ++i) {
    int c = cls.consumerOf[i];
    cls.staticInput.push_back(
        c >= 0 && llvm::all_of(consumers[c].pieces, [](const Pieces &pieces) {
          return llvm::all_of(pieces,
                              [](const Piece &p) { return p.producer < 0; });
        }));
  }
  struct Info {
    llvm::BitVector inputs;
    bool memory = false, box = false;
  };
  DenseMap<Value, Info> info;
  // Depth first from each output, through the logic, stopping at registers,
  // registered reads and constants.
  auto visit = [&](Value root) -> const Info & {
    SmallVector<std::pair<Value, bool>> stack{{root, false}};
    while (!stack.empty()) {
      auto [value, expanded] = stack.pop_back_val();
      if (!expanded && info.count(value))
        continue;
      Info result;
      result.inputs.resize(numInputs);
      SmallVector<Value> operands;
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        if (int(arg.getArgNumber()) != cls.clockInput &&
            !cls.staticInput[arg.getArgNumber()])
          result.inputs.set(arg.getArgNumber());
        info[value] = result;
        continue;
      }
      auto *op = value.getDefiningOp();
      if (isa<seq::FirRegOp, hw::ConstantOp, hw::AggregateConstantOp>(op)) {
        info[value] = result;
        continue;
      }
      if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
        result.memory = true;
        auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
        if (memory && memory.getReadLatency() > 0) {
          info[value] = result;
          continue;
        }
        operands.push_back(read.getAddress());
        if (read.getEnable())
          operands.push_back(read.getEnable());
      } else if (isa<hw::InstanceOp>(op)) {
        result.box = true;
        info[value] = result;
        continue;
      } else {
        operands.append(op->operand_begin(), op->operand_end());
      }
      if (!expanded) {
        // Placeholder against cycles, then the operands first.
        info[value] = result;
        stack.push_back({value, true});
        for (auto operand : operands)
          if (!info.count(operand))
            stack.push_back({operand, false});
        continue;
      }
      for (auto operand : operands) {
        auto &from = info[operand];
        result.inputs |= from.inputs;
        result.memory |= from.memory;
        result.box |= from.box;
      }
      info[value] = result;
    }
    return info[root];
  };
  auto *output = body->getTerminator();
  for (auto operand : output->getOperands()) {
    auto &deps = visit(operand);
    cls.deps.push_back(deps.inputs);
    cls.late.push_back(deps.inputs.any() || deps.memory || deps.box);
  }
  for (auto &op : *body)
    if (auto memory = dyn_cast<seq::FirMemOp>(op))
      if (memory.getReadLatency() > 0)
        cls.syncReads = true;
  for (auto [o, late] : llvm::enumerate(cls.late))
    producers[cls.producerOf[o]].late = late;

  // Logic levels, estimated, on each side of the ports.
  cls.inputToOutput.resize(numInputs);
  for (auto &deps : cls.deps)
    cls.inputToOutput |= deps;
  DenseMap<Value, int> toOutput;
  std::function<int(Value)> levelsTo = [&](Value value) -> int {
    if (auto it = toOutput.find(value); it != toOutput.end())
      return it->second;
    toOutput[value] = 0;
    int levels = 0;
    if (auto *op = value.getDefiningOp()) {
      auto read = dyn_cast<seq::FirMemReadOp>(op);
      auto memory = read ? read.getMemory().getDefiningOp<seq::FirMemOp>()
                         : seq::FirMemOp();
      if (read && (!memory || memory.getReadLatency() == 0))
        levels = 1 + levelsTo(read.getAddress());
      else if (!read && !isa<seq::FirRegOp, hw::InstanceOp>(op))
        for (auto operand : op->getOperands())
          levels = std::max(levels, logicLevels(op) + levelsTo(operand));
    }
    return toOutput[value] = levels;
  };
  for (auto operand : output->getOperands())
    cls.outputDepth.push_back(levelsTo(operand));
  DenseMap<Value, int> toState;
  std::function<int(Value)> levelsFrom = [&](Value value) -> int {
    if (auto it = toState.find(value); it != toState.end())
      return it->second;
    toState[value] = 0;
    int levels = 0;
    for (auto &use : value.getUses()) {
      auto *user = use.getOwner();
      if (isa<seq::FirRegOp, seq::FirMemWriteOp, hw::InstanceOp, hw::OutputOp>(
              user))
        continue;
      if (auto read = dyn_cast<seq::FirMemReadOp>(user)) {
        auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
        if (!memory || memory.getReadLatency() == 0)
          levels = std::max(levels, 1 + levelsFrom(read.getData()));
        continue;
      }
      for (auto result : user->getResults())
        levels = std::max(levels, logicLevels(user) + levelsFrom(result));
    }
    return toState[value] = levels;
  };
  for (auto arg : body->getArguments())
    cls.inputDepth.push_back(levelsFrom(arg));
  return success();
}

bool StaticFolder::zeroLink(unsigned p, unsigned c, unsigned input) const {
  auto &producer = producers[p];
  if (!zeroLatencyLinks || producer.cls < 0 || !producer.late)
    return false;
  auto &cls = classes[c];
  if (cls.inputToOutput.test(input))
    return false;
  // The link's choice of source and the read register's bypass, between.
  return classes[producer.cls].outputDepth[producer.port] +
             cls.inputDepth[input] + 3 <=
         kZeroLinkLevels;
}

//===----------------------------------------------------------------------===//
// The schedule
//===----------------------------------------------------------------------===//

/// Host cycles from a late netlist output's slot to the next frame's first:
/// the token is offered at the slot, the endpoint finishes its cycle the host
/// cycle after and offers its next tokens the one after that, and the next
/// frame starts four host cycles after it is decided on them.
static constexpr int kEnvRoundTrip = 6;

/// Host cycles from an instance's state pass to its next evaluation's first:
/// its registers, written at the end of the state pass, reach the next one's
/// read register through a bypass (`buildCopy`); a registered read's data,
/// out of the memory a host cycle after the state pass, a host cycle later.
static int writeBackLatency(const Class &cls) { return cls.syncReads ? 2 : 1; }

unsigned StaticFolder::place(ArrayRef<bool> ready,
                             SmallVectorImpl<SmallVector<int>> &slots,
                             SmallVectorImpl<bool> &atReady) {
  // Which late outputs each class writes at its ready pass: those whose
  // inputs come from constants, the netlist's inputs, early outputs and late
  // outputs written at ready passes.
  atReady.assign(producers.size(), false);
  for (auto [p, producer] : llvm::enumerate(producers))
    if (producer.cls >= 0 && producer.late && ready[producer.cls])
      atReady[p] = true;
  auto inputAtFrameStart = [&](unsigned c, unsigned input) {
    int k = classes[c].consumerOf[input];
    if (k < 0)
      return true;
    for (auto &pieces : consumers[k].pieces)
      for (auto &piece : pieces) {
        if (piece.producer < 0)
          continue;
        auto &producer = producers[piece.producer];
        if (producer.cls >= 0 && producer.late && !atReady[piece.producer])
          return false;
      }
    return true;
  };
  for (bool changed = true; changed;) {
    changed = false;
    for (auto [p, producer] : llvm::enumerate(producers)) {
      if (!atReady[p])
        continue;
      auto &cls = classes[producer.cls];
      for (auto input : cls.deps[producer.port].set_bits())
        if (!inputAtFrameStart(producer.cls, input)) {
          atReady[p] = false;
          changed = true;
          break;
        }
    }
  }

  // The evaluations, and what each waits for within the frame.
  std::vector<Node> nodes;
  std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned> nodeIndex;
  for (auto [c, cls] : llvm::enumerate(classes))
    for (unsigned i = 0; i < cls.threads(); ++i)
      for (auto pass : {kReady, kState}) {
        if (pass == kReady && !ready[c])
          continue;
        nodeIndex[{c, i, pass}] = nodes.size();
        nodes.push_back({unsigned(c), i, pass});
      }
  std::vector<SmallVector<std::pair<unsigned, int>>> preds(nodes.size());
  for (auto [n, node] : llvm::enumerate(nodes)) {
    auto &cls = classes[node.cls];
    if (node.pass == kState && ready[node.cls])
      preds[n].push_back({nodeIndex[{node.cls, node.inst, kReady}], 1});
    // The inputs this pass reads: all of them for the state pass, those the
    // outputs it writes depend on for the ready pass.
    llvm::BitVector reads(cls.inputNames.size(), node.pass == kState);
    if (node.pass == kReady)
      for (auto [o, late] : llvm::enumerate(cls.late))
        if (atReady[cls.producerOf[o]])
          reads |= cls.deps[o];
    for (auto input : reads.set_bits()) {
      int k = cls.consumerOf[input];
      if (k < 0)
        continue;
      for (auto &piece : consumers[k].pieces[node.inst]) {
        if (piece.producer < 0)
          continue;
        auto &producer = producers[piece.producer];
        if (producer.cls < 0 || !producer.late)
          continue;
        auto pass = atReady[piece.producer] ? kReady : kState;
        preds[n].push_back(
            {nodeIndex[{unsigned(producer.cls), piece.inst, pass}],
             zeroLink(piece.producer, node.cls, input) ? 0 : linkLatency});
      }
    }
  }

  // Longest path to a sink, as the priority; a cycle fails.
  SmallVector<int> height(nodes.size(), -1), state(nodes.size(), 0);
  std::vector<SmallVector<std::pair<unsigned, int>>> succs(nodes.size());
  for (unsigned n = 0; n < nodes.size(); ++n)
    for (auto [p, latency] : preds[n])
      succs[p].push_back({n, latency});
  SmallVector<unsigned> order;
  bool cyclic = false;
  std::function<void(unsigned)> dfs = [&](unsigned n) {
    state[n] = 1;
    int h = 0;
    for (auto [s, latency] : succs[n]) {
      if (state[s] == 1) {
        cyclic = true;
        continue;
      }
      if (state[s] == 0)
        dfs(s);
      h = std::max(h, height[s] + latency);
    }
    height[n] = h;
    state[n] = 2;
    order.push_back(n);
  };
  for (unsigned n = 0; n < nodes.size(); ++n)
    if (!state[n])
      dfs(n);
  if (cyclic)
    return 0;
  std::reverse(order.begin(), order.end());


  // List scheduling: in topological order, highest first among the ready.
  SmallVector<int> slotOf(nodes.size(), -1);
  std::vector<std::set<int>> busy(classes.size());
  SmallVector<unsigned> waiting(nodes.size(), 0);
  for (unsigned n = 0; n < nodes.size(); ++n)
    waiting[n] = preds[n].size();
  auto cmp = [&](unsigned a, unsigned b) {
    if (height[a] != height[b])
      return height[a] < height[b];
    return a > b;
  };
  std::vector<unsigned> heap;
  for (unsigned n = 0; n < nodes.size(); ++n)
    if (!waiting[n])
      heap.push_back(n);
  std::make_heap(heap.begin(), heap.end(), cmp);
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), cmp);
    unsigned n = heap.back();
    heap.pop_back();
    int earliest = 0;
    for (auto [p, latency] : preds[n])
      earliest = std::max(earliest, slotOf[p] + latency);
    auto &taken = busy[nodes[n].cls];
    while (taken.count(earliest))
      ++earliest;
    taken.insert(earliest);
    slotOf[n] = earliest;
    for (auto [s, latency] : succs[n])
      if (!--waiting[s]) {
        heap.push_back(s);
        std::push_heap(heap.begin(), heap.end(), cmp);
      }
  }

  slots.assign(classes.size() * 2, {});
  for (auto [c, cls] : llvm::enumerate(classes))
    for (auto pass : {kReady, kState})
      slots[c * 2 + pass].assign(cls.threads(), -1);
  int length = 0;
  for (auto [n, node] : llvm::enumerate(nodes)) {
    slots[node.cls * 2 + node.pass][node.inst] = slotOf[n];
    length = std::max(length, slotOf[n] + 1);
  }

  // Across frames: an instance's next read of its state after its write (a
  // threaded class's registers written a host cycle late, from a stage); an
  // early output's reads in the next frame after its write.
  auto first = [&](unsigned c, unsigned i) {
    int s = slots[c * 2 + kState][i];
    int r = slots[c * 2 + kReady][i];
    return r >= 0 ? std::min(r, s) : s;
  };
  for (auto [c, cls] : llvm::enumerate(classes))
    for (unsigned i = 0; i < cls.threads(); ++i) {
      int s = slots[c * 2 + kState][i];
      length = std::max(length, s - first(c, i) + writeBackLatency(cls));
    }
  for (auto &consumer : consumers) {
    if (consumer.cls < 0)
      continue;
    for (auto [i, pieces] : llvm::enumerate(consumer.pieces))
      for (auto &piece : pieces) {
        if (piece.producer < 0)
          continue;
        auto &producer = producers[piece.producer];
        if (producer.cls < 0 || producer.late)
          continue;
        int written = slots[producer.cls * 2 + kState][piece.inst];
        length = std::max(length, written - first(consumer.cls, i) +
                                      linkLatency);
      }
  }

  // Each netlist output's token: an early one's next value at (or, without
  // forwarding, a slot after) its producers' state passes, a late one's at
  // its writes. A late token closes the environment's cycle: offered at its
  // slot, taken by an endpoint's staged token end a host cycle later, which
  // finishes then and offers its next tokens the host cycle after; the next
  // frame is decided four host cycles before this one ends, on those tokens,
  // or, with overlapped frames, counting the pushes still due.
  outputSlot.assign(consumers.size(), -1);
  outputEarly.assign(consumers.size(), true);
  for (auto [k, consumer] : llvm::enumerate(consumers)) {
    if (consumer.cls >= 0)
      continue;
    int at = 0;
    for (auto &piece : consumer.pieces[0]) {
      if (piece.producer < 0)
        continue;
      auto &producer = producers[piece.producer];
      if (producer.cls < 0) {
        outputEarly[k] = false;
        continue;
      }
      if (producer.late)
        outputEarly[k] = false;
      auto pass = atReady[piece.producer] ? kReady : kState;
      at = std::max(at, slots[producer.cls * 2 + pass][piece.inst] +
                            linkLatency - 1);
    }
    outputSlot[k] = at;
    length = std::max(length, at + (outputEarly[k] ? 1 : kEnvRoundTrip));
  }
  // Overlapped, the decision for the next frame may fall before this one
  // begins: the inputs' and outputs' queues count the frames decided ahead.
  length = std::max(length, 2);
  // Consecutive frames overlapped, if that frame is shorter: a frame placed
  // alone is one of them, at its own length.
  if (modulo) {
    SmallVector<SmallVector<int>> overlapped;
    SmallVector<int> aloneSlot = outputSlot;
    SmallVector<bool> aloneEarly = outputEarly;
    unsigned period = placeModulo(nodes, preds, succs, height, atReady, overlapped);
    if (period && period < unsigned(length)) {
      slots = std::move(overlapped);
      return period;
    }
    outputSlot = aloneSlot;
    outputEarly = aloneEarly;
  }
  return length;
}

unsigned StaticFolder::placeModulo(
    ArrayRef<Node> nodes, ArrayRef<SmallVector<std::pair<unsigned, int>>> preds,
    ArrayRef<SmallVector<std::pair<unsigned, int>>> succs, ArrayRef<int> height,
    ArrayRef<bool> atReady, SmallVectorImpl<SmallVector<int>> &slots) {
  // No frame is shorter than a copy's evaluations of it.
  SmallVector<unsigned> evaluations(classes.size(), 0);
  for (auto &node : nodes)
    ++evaluations[node.cls];
  unsigned lower = 2;
  for (auto count : evaluations)
    lower = std::max(lower, count);
  auto cmp = [&](unsigned a, unsigned b) {
    if (height[a] != height[b])
      return height[a] < height[b];
    return a > b;
  };
  SmallVector<bool> feedsOutput(producers.size(), false);
  for (auto &consumer : consumers)
    if (consumer.cls < 0)
      for (auto &piece : consumer.pieces[0])
        if (piece.producer >= 0)
          feedsOutput[piece.producer] = true;
  for (unsigned period = lower; period <= 4 * lower + 64; ++period) {
    // List scheduling, a copy's slots taken modulo the frame; an evaluation
    // runs at most one frame late.
    SmallVector<int> slotOf(nodes.size(), -1);
    std::vector<std::set<int>> busy(classes.size());
    SmallVector<unsigned> waiting(nodes.size(), 0);
    for (unsigned n = 0; n < nodes.size(); ++n)
      waiting[n] = preds[n].size();
    std::vector<unsigned> heap;
    for (unsigned n = 0; n < nodes.size(); ++n)
      if (!waiting[n])
        heap.push_back(n);
    std::make_heap(heap.begin(), heap.end(), cmp);
    bool fits = true;
    while (fits && !heap.empty()) {
      std::pop_heap(heap.begin(), heap.end(), cmp);
      unsigned n = heap.back();
      heap.pop_back();
      int earliest = 0;
      for (auto [p, latency] : preds[n])
        earliest = std::max(earliest, slotOf[p] + latency);
      auto &taken = busy[nodes[n].cls];
      while (taken.count(earliest % period))
        ++earliest;
      if (earliest >= int(2 * period)) {
        fits = false;
        break;
      }
      taken.insert(earliest % period);
      slotOf[n] = earliest;
      for (auto [s, latency] : succs[n])
        if (!--waiting[s]) {
          heap.push_back(s);
          std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    if (!fits)
      continue;

    // Then each evaluation as late as its readers allow, latest readers
    // first: a ready pass close to its readers shortens the span from an
    // instance's first evaluation to its state write, which bounds the frame.
    // A state pass after a ready pass stays, since moving it lengthens that
    // span.
    SmallVector<unsigned> byStart(nodes.size());
    for (unsigned n = 0; n < nodes.size(); ++n)
      byStart[n] = n;
    llvm::stable_sort(byStart,
                      [&](unsigned a, unsigned b) { return slotOf[a] > slotOf[b]; });
    for (auto n : byStart) {
      auto &node = nodes[n];
      if (node.pass == kState &&
          llvm::any_of(preds[n], [&](auto pred) {
            return nodes[pred.first].cls == node.cls &&
                   nodes[pred.first].inst == node.inst;
          }))
        continue;
      int latest = INT_MAX;
      for (auto [s, latency] : succs[n])
        latest = std::min(latest, slotOf[s] - latency);
      // A netlist output's token leaves in time for the environment.
      for (auto port : classes[node.cls].producerOf) {
        auto &producer = producers[port];
        bool at = producer.late && atReady[port] ? node.pass == kReady
                                                  : node.pass == kState;
        if (!feedsOutput[port] || !at)
          continue;
        latest = std::min<int>(latest, int(period) - linkLatency + 1 -
                                           (producer.late ? kEnvRoundTrip : 1));
      }
      if (latest == INT_MAX)
        continue;
      auto &taken = busy[node.cls];
      taken.erase(slotOf[n] % period);
      int at = slotOf[n];
      for (int t = latest; t > slotOf[n]; --t)
        if (!taken.count(t % period)) {
          at = t;
          break;
        }
      taken.insert(at % period);
      slotOf[n] = at;
    }

    slots.assign(classes.size() * 2, {});
    for (auto [c, cls] : llvm::enumerate(classes))
      for (auto pass : {kReady, kState})
        slots[c * 2 + pass].assign(cls.threads(), -1);
    for (auto [n, node] : llvm::enumerate(nodes))
      slots[node.cls * 2 + node.pass][node.inst] = slotOf[n];
    auto first = [&](unsigned c, unsigned i) {
      int s = slots[c * 2 + kState][i];
      int r = slots[c * 2 + kReady][i];
      return r >= 0 ? std::min(r, s) : s;
    };
    auto last = [&](unsigned c, unsigned i) {
      return std::max(slots[c * 2 + kState][i], slots[c * 2 + kReady][i]);
    };
    bool ok = true;
    // An instance's next frame reads its state after this one's write.
    for (auto [c, cls] : llvm::enumerate(classes))
      for (unsigned i = 0; ok && i < cls.threads(); ++i) {
        ok = slots[c * 2 + kState][i] - first(c, i) + writeBackLatency(cls) <=
             int(period);
      }
    // A late link holds one frame's value: its readers read it before its
    // producer writes the next frame's.
    for (unsigned n = 0; ok && n < nodes.size(); ++n)
      for (auto [p, latency] : preds[n])
        if (slotOf[n] - slotOf[p] >= int(period))
          ok = false;
    // An early link holds two, by frame: the next frame's write lands after
    // its reads of this frame's, and before them in the next.
    for (auto &consumer : consumers) {
      if (!ok || consumer.cls < 0)
        continue;
      for (auto [i, pieces] : llvm::enumerate(consumer.pieces))
        for (auto &piece : pieces) {
          if (piece.producer < 0)
            continue;
          auto &producer = producers[piece.producer];
          if (producer.cls < 0 || producer.late)
            continue;
          int written = slots[producer.cls * 2 + kState][piece.inst];
          if (written - first(consumer.cls, i) + linkLatency > int(period) ||
              last(consumer.cls, i) - written >= int(period))
            ok = false;
        }
    }
    if (!ok)
      continue;

    // The netlist's outputs leave within their own frame, a late one in time
    // for the environment's round trip.
    outputSlot.assign(consumers.size(), -1);
    outputEarly.assign(consumers.size(), true);
    for (auto [k, consumer] : llvm::enumerate(consumers)) {
      if (consumer.cls >= 0)
        continue;
      int at = 0;
      for (auto &piece : consumer.pieces[0]) {
        if (piece.producer < 0)
          continue;
        auto &producer = producers[piece.producer];
        if (producer.cls < 0) {
          outputEarly[k] = false;
          continue;
        }
        if (producer.late)
          outputEarly[k] = false;
        auto pass = atReady[piece.producer] ? kReady : kState;
        at = std::max(at, slots[producer.cls * 2 + pass][piece.inst] +
                              linkLatency - 1);
      }
      outputSlot[k] = at;
      if (at + (outputEarly[k] ? 1 : kEnvRoundTrip) > int(period))
        ok = false;
    }
    if (ok)
      return period;
  }
  return 0;
}

LogicalResult StaticFolder::schedule() {
  // A ready pass for any choice of the classes with late outputs: few classes,
  // so every choice is placed and the shortest frame kept. Copies multiply
  // the classes: past a dozen, every choice of the models is placed with the
  // copies of a model alike, then single classes are flipped while the frame
  // gets shorter.
  auto late = [](const Class &cls) {
    return llvm::any_of(cls.late, [](bool late) { return late; });
  };
  SmallVector<unsigned> candidates;
  for (auto [c, cls] : llvm::enumerate(classes))
    if (late(cls))
      candidates.push_back(c);
  bool alike = candidates.size() > 12;
  if (alike)
    llvm::erase_if(candidates,
                   [&](unsigned c) { return classes[c].splitFrom >= 0; });
  if (candidates.size() > 12)
    return netlist.emitError("too many models with late outputs to choose "
                             "their ready passes: ")
           << candidates.size();
  unsigned best = 0;
  SmallVector<bool> bestReady;
  SmallVector<SmallVector<int>> bestSlots;
  SmallVector<bool> bestAtReady;
  SmallVector<int> bestOutputSlot;
  SmallVector<bool> bestOutputEarly;
  auto tryReady = [&](const SmallVector<bool> &ready) {
    SmallVector<SmallVector<int>> slots;
    SmallVector<bool> atReady;
    unsigned length = place(ready, slots, atReady);
    if (!length || (best && length >= best))
      return false;
    best = length;
    bestReady = ready;
    bestSlots = slots;
    bestAtReady = atReady;
    bestOutputSlot = outputSlot;
    bestOutputEarly = outputEarly;
    return true;
  };
  for (uint64_t mask = 0; mask < (uint64_t(1) << candidates.size()); ++mask) {
    SmallVector<bool> ready(classes.size(), false);
    for (auto [k, c] : llvm::enumerate(candidates))
      ready[c] = (mask >> k) & 1;
    if (alike)
      for (auto [c, cls] : llvm::enumerate(classes))
        if (cls.splitFrom >= 0)
          ready[c] = ready[cls.splitFrom];
    tryReady(ready);
  }
  for (bool shorter = alike && best; shorter;) {
    shorter = false;
    for (auto [c, cls] : llvm::enumerate(classes)) {
      if (!late(cls))
        continue;
      auto ready = bestReady;
      ready[c] = !ready[c];
      shorter |= tryReady(ready);
    }
  }
  if (!best)
    return netlist.emitError("the evaluations of a frame depend on each other "
                             "in a cycle for every choice of ready passes");
  ii = best;
  outputSlot = bestOutputSlot;
  outputEarly = bestOutputEarly;
  for (auto [p, producer] : llvm::enumerate(producers))
    producer.atReady = bestAtReady[p];
  for (auto [c, cls] : llvm::enumerate(classes)) {
    cls.hasReady = bestReady[c];
    cls.slot[kReady] = bestSlots[c * 2 + kReady];
    cls.slot[kState] = bestSlots[c * 2 + kState];
  }

  std::string text;
  llvm::raw_string_ostream os(text);
  os << "foldsim-fold-static: II = " << ii << "\n";
  for (auto &cls : classes) {
    os << "  " << cls.name << " x" << cls.threads()
       << (cls.hasReady ? " (ready pass)" : "");
    if (nest)
      os << " depth " << cls.depth
         << (cls.parent >= 0 ? " under " + classes[cls.parent].name : "");
    os << "\n";
    for (auto pass : {kReady, kState}) {
      if (pass == kReady && !cls.hasReady)
        continue;
      os << (pass == kReady ? "    ready:" : "    state:");
      for (auto s : cls.slot[pass])
        os << " " << s;
      os << "\n";
    }
    for (auto [o, late] : llvm::enumerate(cls.late)) {
      auto &producer = producers[cls.producerOf[o]];
      if (!late || !producer.read)
        continue;
      os << "    late " << cls.outputNames[o]
         << (producer.atReady ? " @ready" : " @state") << " <-";
      for (auto input : cls.deps[o].set_bits())
        os << " " << cls.inputNames[input];
      os << "\n";
    }
  }
  for (auto [k, consumer] : llvm::enumerate(consumers))
    if (consumer.cls < 0)
      os << "  output " << consumer.name << " @" << outputSlot[k]
         << (outputEarly[k] ? " (next)" : "") << "\n";
  llvm::errs() << os.str();
  if (!reportPath.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream file(reportPath, ec);
    if (ec)
      return netlist.emitError("report: ") << ec.message();
    file << os.str();
  }
  return success();
}


//===----------------------------------------------------------------------===//
// Fixed-latency pipelines
//===----------------------------------------------------------------------===//

/// Bits `[low, low + width)` of the integer `value`, looking through wiring.
static Value sliceBits(ImplicitLocOpBuilder &b, Value value, unsigned low,
                       unsigned width) {
  if (low == 0 && hw::getBitWidth(value.getType()) == width)
    return value;
  if (auto extract = value.getDefiningOp<comb::ExtractOp>())
    return sliceBits(b, extract.getInput(), extract.getLowBit() + low, width);
  if (auto concat = value.getDefiningOp<comb::ConcatOp>()) {
    SmallVector<Value> pieces;
    unsigned operandLow = hw::getBitWidth(concat.getType());
    for (auto operand : concat.getInputs()) {
      unsigned operandWidth = hw::getBitWidth(operand.getType());
      operandLow -= operandWidth;
      unsigned lo = std::max(low, operandLow);
      unsigned hi = std::min(low + width, operandLow + operandWidth);
      if (lo < hi)
        pieces.push_back(sliceBits(b, operand, lo - operandLow, hi - lo));
    }
    return pieces.size() == 1 ? pieces[0]
                              : b.createOrFold<comb::ConcatOp>(pieces);
  }
  return b.createOrFold<comb::ExtractOp>(value, low, width);
}

/// `value` as `type`, looking through bitcasts.
static Value castBits(ImplicitLocOpBuilder &b, Value value, Type type) {
  if (value.getType() == type)
    return value;
  if (auto cast = value.getDefiningOp<hw::BitcastOp>())
    return castBits(b, cast.getInput(), type);
  return b.createOrFold<hw::BitcastOp>(type, value);
}

/// The value made of runs of bits of other values: (value, low bit, count),
/// the least significant run first.
static Value fromRuns(ImplicitLocOpBuilder &b,
                      ArrayRef<std::tuple<Value, unsigned, unsigned>> runs) {
  SmallVector<Value> pieces;
  for (auto &[value, low, count] : llvm::reverse(runs)) {
    auto all = hw::getBitWidth(value.getType());
    pieces.push_back(
        sliceBits(b, castBits(b, value, b.getIntegerType(all)), low, count));
  }
  return pieces.size() == 1 ? pieces[0] : b.createOrFold<comb::ConcatOp>(pieces);
}

/// Appends bit `bit` of `value` to `runs`.
static void addBit(SmallVectorImpl<std::tuple<Value, unsigned, unsigned>> &runs,
                   Value value, unsigned bit) {
  if (!runs.empty()) {
    auto &[v, low, count] = runs.back();
    if (v == value && low + count == bit) {
      ++count;
      return;
    }
  }
  runs.push_back({value, bit, 1});
}

static bool inCone(Operation *op) {
  if (!op || op->getNumRegions() || isa<hw::InstanceOp, hw::OutputOp>(op))
    return false;
  auto ns = op->getDialect()->getNamespace();
  return ns == "comb" || ns == "hw";
}

/// Folds the fixed-latency pipelines of `model`, of `threads` instances, as
/// `foldsim-fame5` does: an unthreaded copy of each on the host clock, fed the
/// slot's instance's issue, and a per-instance store of its results that is
/// target state, read at the instance's count of enabled edges modulo its
/// depth. A static copy evaluates an instance's state pass every frame, II
/// host cycles apart, so its edges are at least two host cycles apart, as the
/// fold needs. The copy's host clock, slot instance, state pass and the store's
/// writes are filled in when the copy is built.
static LogicalResult foldPipelines(hw::HWModuleOp model, unsigned threads,
                                   std::unique_ptr<Folding> &out) {
  auto *body = model.getBodyBlock();
  ImplicitLocOpBuilder b(model.getLoc(), body->getTerminator());
  foldsim::PipelineAnalysis analysis(model);
  if (failed(analysis.run(b)))
    return failure();
  if (analysis.pipelines.empty())
    return success();
  out = std::make_unique<Folding>();
  auto &f = *out;
  auto *context = model.getContext();
  auto i1 = b.getI1Type();
  Value modelClock;
  for (auto [i, port] : llvm::enumerate(model.getPortList()))
    if (isa<seq::ClockType>(port.type) &&
        port.dir == hw::ModulePort::Direction::Input)
      modelClock = body->getArgument(port.argNum);
  if (!modelClock)
    return model.emitError("a model has a clock input");
  f.edges = std::make_unique<BackedgeBuilder>(b, model.getLoc());
  f.physClock = f.edges->get(seq::ClockType::get(context));
  auto idxType = b.getIntegerType(llvm::Log2_32_Ceil(threads));
  f.thread = f.edges->get(idxType);
  f.edge = f.edges->get(i1);
  f.reset = f.edges->get(i1);
  auto hostReg = [&](Value next, const Twine &regName,
                     bool cleared = false) -> Value {
    auto type = next.getType();
    seq::FirRegOp reg =
        cleared ? seq::FirRegOp::create(
                      b, next, f.physClock, b.getStringAttr(regName), f.reset,
                      hw::ConstantOp::create(b, type, 0), hw::InnerSymAttr{},
                      /*isAsync=*/false, b.getIntegerAttr(type, 0))
                : seq::FirRegOp::create(b, next, f.physClock,
                                        b.getStringAttr(regName));
    f.physical.insert(reg);
    return reg;
  };

  auto &pipelines = analysis.pipelines;
  auto &registers = analysis.registers;
  auto &boxes = analysis.boxes;
  struct Bit {
    int reg = -1, box = -1;
    Value result;
    unsigned bit = 0;
  };
  std::vector<SmallVector<Bit>> words(pipelines.size());
  DenseMap<std::pair<unsigned, unsigned>, unsigned> regBitAt;
  DenseMap<std::pair<Value, unsigned>, unsigned> boxBitAt;
  for (auto [ri, reg] : llvm::enumerate(registers))
    for (unsigned bit = 0; bit < reg.layer.size(); ++bit) {
      if (!reg.layer[bit])
        continue;
      auto pi = reg.pipeline[bit];
      if (reg.layer[bit] != pipelines[pi].layers)
        continue;
      regBitAt[{ri, bit}] = words[pi].size();
      words[pi].push_back({int(ri), -1, {}, bit});
    }
  auto boxPorts = [&](foldsim::PipelineBox &box) {
    auto *module = SymbolTable::lookupNearestSymbolFrom(
        box.inst, box.inst.getReferencedModuleNameAttr());
    return cast<hw::HWModuleLike>(module).getPortList();
  };
  DenseMap<unsigned, unsigned> boxPipeline;
  for (auto [pi, p] : llvm::enumerate(pipelines))
    for (auto bi : p.boxes)
      boxPipeline[bi] = pi;
  for (auto [bi, box] : llvm::enumerate(boxes)) {
    auto pi = boxPipeline[bi];
    if (box.outLayer != pipelines[pi].layers)
      continue;
    for (auto &port : boxPorts(box)) {
      if (port.dir == hw::ModulePort::Direction::Input ||
          !llvm::is_contained(box.contract.dataOut, port.getName()))
        continue;
      auto result = box.inst.getResult(port.argNum);
      for (int64_t k = 0; k < hw::getBitWidth(result.getType()); ++k) {
        boxBitAt[{result, unsigned(k)}] = words[pi].size();
        words[pi].push_back({-1, int(bi), result, unsigned(k)});
      }
    }
  }

  SmallVector<Value> storeRead(pipelines.size());
  for (unsigned pi = 0; pi < pipelines.size(); ++pi) {
    auto &p = pipelines[pi];
    auto prefix = ("fold" + Twine(pi)).str();
    unsigned depth = p.depth();
    unsigned layers = p.layers;
    bool stored = p.lastLayerStored;
    unsigned outAt = depth + 1;
    unsigned readable = outAt + 2;
    unsigned firstRead = stored ? 2 * layers : 2 * (layers - 1);
    if (readable > firstRead)
      return model.emitError("folded pipeline ")
             << pi << " of " << layers << " layers: its results are back "
             << readable << " host cycles after issue, after their first "
             << "read at " << firstRead;

    // The instance's slot: its enabled edges modulo the store's depth.
    auto slotType = b.getIntegerType(llvm::Log2_32_Ceil(depth));
    auto slotNext = f.edges->get(slotType);
    Value slot = seq::FirRegOp::create(b, slotNext, modelClock,
                                       b.getStringAttr(prefix + "_slot"));
    // The count is state of no reset: from any value it starts in range, so
    // that each result is read where it was written from the first edge on.
    if (!llvm::isPowerOf2_32(depth)) {
      Value size = hw::ConstantOp::create(b, slotType, depth);
      Value over = comb::ICmpOp::create(b, comb::ICmpPredicate::uge, slot, size);
      slot = comb::MuxOp::create(b, over, comb::SubOp::create(b, slot, size),
                                 slot);
    }
    Value last = comb::ICmpOp::create(
        b, comb::ICmpPredicate::eq, slot,
        hw::ConstantOp::create(b, slotType, depth - 1));
    Value step = comb::MuxOp::create(
        b, last, hw::ConstantOp::create(b, slotType, 0),
        comb::AddOp::create(b, slot, hw::ConstantOp::create(b, slotType, 1)));
    slotNext.setValue(p.enable
                          ? Value(comb::MuxOp::create(b, p.enable, step, slot))
                          : step);

    unsigned width = words[pi].size();
    auto memType = seq::FirMemType::get(context, depth, width, 1);
    auto store = seq::FirMemOp::create(
        b, memType, /*readLatency=*/0, /*writeLatency=*/1, seq::RUW::Undefined,
        seq::WUW::Undefined, b.getStringAttr(prefix + "_store"),
        hw::InnerSymAttr{}, seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
    storeRead[pi] = seq::FirMemReadOp::create(
        b, store, slot, modelClock, hw::ConstantOp::create(b, i1, 1));
    f.foldOfStore[store] = f.folds.size();

    // The unthreaded copy.
    DenseMap<Value, Value> copies;
    SmallVector<std::pair<unsigned, Backedge>> copyNexts;
    for (auto ri : p.registers) {
      auto &reg = registers[ri];
      bool copied = false;
      for (unsigned bit = 0; bit < reg.layer.size(); ++bit)
        if (reg.layer[bit] && reg.pipeline[bit] == pi &&
            (stored || reg.layer[bit] < layers))
          copied = true;
      if (!copied)
        continue;
      auto type = b.getIntegerType(reg.layer.size());
      auto next = f.edges->get(type);
      auto copy = hostReg(next, reg.reg.getName() + "_fold");
      copies[reg.reg.getResult()] =
          castBits(b, copy, reg.reg.getResult().getType());
      copyNexts.push_back({ri, next});
    }
    SmallVector<std::pair<unsigned, hw::InstanceOp>> boxCopies;
    for (auto bi : p.boxes)
      boxCopies.push_back({bi, {}});
    bool cloneFailed = false;
    std::function<Value(Value)> overCopy = [&](Value value) -> Value {
      if (auto it = copies.find(value); it != copies.end())
        return it->second;
      if (!analysis.dependsOnPipeline(value, pi))
        return value;
      auto *op = value.getDefiningOp();
      if (!op || !inCone(op)) {
        if (!cloneFailed)
          (op ? op->emitError() : model.emitError())
              << "folded pipeline " << pi << " depends on this other than "
              << "through logic";
        cloneFailed = true;
        return value;
      }
      IRMapping mapping;
      for (auto operand : op->getOperands())
        mapping.map(operand, overCopy(operand));
      auto *clone = b.clone(*op, mapping);
      for (auto [old, cloned] :
           llvm::zip(op->getResults(), clone->getResults()))
        copies[old] = cloned;
      return copies[value];
    };
    // The issue, registered: the first layer's logic cloned on its leaves,
    // each registered once.
    DenseMap<Value, Value> issued;
    std::function<Value(Value)> issueCopy = [&](Value value) -> Value {
      if (auto it = issued.find(value); it != issued.end())
        return it->second;
      auto *op = value.getDefiningOp();
      if (op && inCone(op) && op->getNumOperands() == 0)
        return issued[value] = value;
      if (!op || !inCone(op))
        return issued[value] = hostReg(value, prefix + "_issue");
      IRMapping mapping;
      for (auto operand : op->getOperands())
        mapping.map(operand, issueCopy(operand));
      auto *clone = b.clone(*op, mapping);
      for (auto [old, cloned] :
           llvm::zip(op->getResults(), clone->getResults()))
        issued[old] = cloned;
      return issued[value];
    };
    for (auto &[bi, copy] : boxCopies) {
      auto &box = boxes[bi];
      SmallVector<Value> operands;
      Value clockConv;
      for (auto &port : boxPorts(box)) {
        if (port.dir != hw::ModulePort::Direction::Input)
          continue;
        Value operand = box.inst.getOperand(port.argNum);
        auto name = port.getName();
        if (name == box.contract.clock)
          operand = isa<seq::ClockType>(port.type)
                        ? Value(f.physClock)
                        : (clockConv = seq::FromClockOp::create(b, f.physClock));
        else if (name == box.contract.enable)
          operand = hw::ConstantOp::create(b, port.type, 1);
        else if (llvm::is_contained(box.contract.dataIn, name))
          operand = box.inLayer > 0 ? Value() : issueCopy(operand);
        operands.push_back(operand);
      }
      for (auto &operand : operands)
        if (!operand)
          operand = hw::ConstantOp::create(b, b.getI1Type(), 0);
      auto *module = SymbolTable::lookupNearestSymbolFrom(
          box.inst, box.inst.getReferencedModuleNameAttr());
      copy = hw::InstanceOp::create(
          b, module, b.getStringAttr(box.inst.getInstanceName() + "_fold"),
          operands);
      f.physical.insert(copy);
      if (clockConv)
        f.physical.insert(clockConv.getDefiningOp());
      for (auto [old, cloned] :
           llvm::zip(box.inst.getResults(), copy.getResults()))
        copies[old] = cloned;
    }
    for (auto &[bi, copy] : boxCopies) {
      auto &box = boxes[bi];
      if (box.inLayer == 0)
        continue;
      unsigned i = 0;
      for (auto &port : boxPorts(box)) {
        if (port.dir != hw::ModulePort::Direction::Input)
          continue;
        if (llvm::is_contained(box.contract.dataIn, port.getName()))
          copy->setOperand(i, overCopy(box.inst.getOperand(port.argNum)));
        ++i;
      }
    }
    DenseMap<unsigned, Value> dataCopy;
    auto copiedData = [&](unsigned ri) {
      auto &v = dataCopy[ri];
      if (!v)
        v = overCopy(registers[ri].data);
      return v;
    };
    for (auto &[ri, next] : copyNexts) {
      auto &reg = registers[ri];
      SmallVector<std::tuple<Value, unsigned, unsigned>> runs;
      for (unsigned bit = 0; bit < reg.layer.size(); ++bit) {
        bool later = reg.layer[bit] > 1 && reg.pipeline[bit] == pi;
        bool first = reg.layer[bit] == 1 && reg.pipeline[bit] == pi;
        addBit(runs,
               later ? copiedData(ri) : first ? issueCopy(reg.data) : reg.data,
               bit);
      }
      next.setValue(fromRuns(b, runs));
    }
    if (cloneFailed)
      return failure();

    SmallVector<std::tuple<Value, unsigned, unsigned>> runs;
    for (auto &bit : words[pi]) {
      if (bit.box >= 0)
        addBit(runs, copies[bit.result], bit.bit);
      else if (stored)
        addBit(runs, copies[registers[bit.reg].reg.getResult()], bit.bit);
      else
        addBit(runs, copiedData(bit.reg), bit.bit);
    }
    Value word = fromRuns(b, runs);

    // The tags, along with the copy, and the registered write.
    Value valid = p.enable ? Value(comb::AndOp::create(b, f.edge, p.enable))
                           : Value(f.edge);
    Value slotTag = slot, threadTag = f.thread;
    for (unsigned k = 1; k <= outAt + 1; ++k) {
      valid = hostReg(valid, prefix + "_valid" + Twine(k), /*cleared=*/true);
      slotTag = hostReg(slotTag, prefix + "_slot" + Twine(k));
      threadTag = hostReg(threadTag, prefix + "_thread" + Twine(k));
    }
    Value data = hostReg(word, prefix + "_wdata");
    f.folds.push_back({store, {}, slotTag, threadTag, valid, data});
  }

  // The target keeps each pipeline register's bits that stay, the last layer
  // taking its next value from the store; a stored last layer is read from
  // the store; the other layers are gone.
  auto zeroBits = [&](unsigned count) {
    return hw::ConstantOp::create(b, b.getIntegerType(count), 0);
  };
  for (auto [ri, reg] : llvm::enumerate(registers)) {
    auto width = unsigned(reg.layer.size());
    SmallVector<unsigned> kept, other;
    for (unsigned bit = 0; bit < width; ++bit) {
      if (!reg.layer[bit])
        other.push_back(bit);
      else if (!pipelines[reg.pipeline[bit]].lastLayerStored &&
               reg.layer[bit] == pipelines[reg.pipeline[bit]].layers)
        kept.push_back(bit);
    }
    auto name = reg.reg.getName();
    Value keptReg, otherReg;
    if (!kept.empty()) {
      SmallVector<std::tuple<Value, unsigned, unsigned>> runs;
      for (auto bit : kept)
        addBit(runs, storeRead[reg.pipeline[bit]], regBitAt[{ri, bit}]);
      Value data = fromRuns(b, runs);
      auto next = f.edges->get(data.getType());
      keptReg = seq::FirRegOp::create(b, next, reg.reg.getClk(),
                                      b.getStringAttr(name));
      next.setValue(reg.enable ? Value(comb::MuxOp::create(b, reg.enable, data,
                                                           keptReg))
                               : data);
    }
    if (!other.empty()) {
      SmallVector<std::tuple<Value, unsigned, unsigned>> runs;
      for (auto bit : other)
        addBit(runs, reg.reg.getNext(), bit);
      auto next = f.edges->get(b.getIntegerType(other.size()));
      otherReg = seq::FirRegOp::create(b, next, reg.reg.getClk(),
                                       b.getStringAttr(name));
      next.setValue(fromRuns(b, runs));
    }
    SmallVector<std::tuple<Value, unsigned, unsigned>> runs;
    Value zeros = zeroBits(width);
    unsigned k = 0, o = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
      if (!reg.layer[bit]) {
        addBit(runs, otherReg, o++);
        continue;
      }
      auto &p = pipelines[reg.pipeline[bit]];
      if (reg.layer[bit] != p.layers)
        addBit(runs, zeros, bit);
      else if (p.lastLayerStored)
        addBit(runs, storeRead[reg.pipeline[bit]], regBitAt[{ri, bit}]);
      else
        addBit(runs, keptReg, k++);
    }
    Value value = castBits(b, fromRuns(b, runs), reg.reg.getType());
    reg.reg.getResult().replaceAllUsesWith(value);
    reg.reg.erase();
  }
  for (auto [bi, box] : llvm::enumerate(boxes)) {
    auto &p = pipelines[boxPipeline[bi]];
    for (auto result : box.inst.getResults()) {
      if (result.use_empty())
        continue;
      auto w = hw::getBitWidth(result.getType());
      SmallVector<std::tuple<Value, unsigned, unsigned>> runs;
      if (box.outLayer == p.layers) {
        for (int64_t k = 0; k < w; ++k)
          addBit(runs, storeRead[boxPipeline[bi]],
                 boxBitAt[{result, unsigned(k)}]);
        result.replaceAllUsesWith(
            castBits(b, fromRuns(b, runs), result.getType()));
      } else {
        result.replaceAllUsesWith(castBits(b, zeroBits(w), result.getType()));
      }
    }
    box.inst.erase();
  }

  // What only the pipelines' registers used, and the analysis' peeled writes,
  // are dead.
  for (bool changed = true; changed;) {
    changed = false;
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
      if (f.physical.contains(&op) || !op.use_empty() || op.getNumResults() == 0)
        continue;
      if (inCone(&op) || isa<seq::FromClockOp>(op)) {
        op.erase();
        changed = true;
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// A class's copy
//===----------------------------------------------------------------------===//

/// The regions sharing a copy of the slot controls two host cycles ahead.
static constexpr unsigned kRegionsPerGroup = 16;

static seq::FirRegOp keep(seq::FirRegOp reg) {
  sv::addSVAttributes(reg, {sv::SVAttributeAttr::get(reg.getContext(), "keep",
                                                     R"("true")",
                                                     /*emitAsComment=*/false)});
  return reg;
}

static Value asBits(ImplicitLocOpBuilder &b, Value value) {
  auto width = hw::getBitWidth(value.getType());
  return b.createOrFold<hw::BitcastOp>(b.getIntegerType(width), value);
}

/// The value `value` takes in the next target cycle, every register replaced
/// by its next value; built for a value that depends on no input but static
/// ones, nor on a memory's read data.
static Value nextValue(ImplicitLocOpBuilder &b, Value root,
                       DenseMap<Value, Value> &nexts) {
  SmallVector<std::pair<Value, bool>> stack{{root, false}};
  while (!stack.empty()) {
    auto [value, expanded] = stack.pop_back_val();
    if (!expanded && nexts.count(value))
      continue;
    auto *op = value.getDefiningOp();
    if (!op || isa<hw::ConstantOp, hw::AggregateConstantOp>(op)) {
      nexts[value] = value;
      continue;
    }
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      Value next = reg.getNext();
      if (reg.getReset())
        next = comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), next);
      nexts[value] = next;
      continue;
    }
    if (!expanded) {
      stack.push_back({value, true});
      for (auto operand : op->getOperands())
        if (!nexts.count(operand))
          stack.push_back({operand, false});
      continue;
    }
    bool same = llvm::all_of(op->getOperands(), [&](Value operand) {
      return nexts.lookup(operand) == operand;
    });
    if (same) {
      for (auto result : op->getResults())
        nexts[result] = result;
      continue;
    }
    IRMapping mapping;
    for (auto operand : op->getOperands())
      mapping.map(operand, nexts.lookup(operand));
    auto *clone = b.clone(*op, mapping);
    for (auto [old, cloned] : llvm::zip(op->getResults(), clone->getResults()))
      nexts[old] = cloned;
  }
  return nexts.lookup(root);
}

LogicalResult StaticFolder::buildCopy(Class &cls) {
  auto model = cls.model;
  auto *context = model.getContext();
  auto *modelBody = model.getBodyBlock();
  unsigned threads = cls.threads();
  unsigned idxWidth = threads > 1 ? llvm::Log2_32_Ceil(threads) : 0;
  if (threads > 1 && !llvm::isPowerOf2_32(threads))
    return model.emitError("a class of ") << threads
                                          << " instances: a copy needs a "
                                             "power of two";
  auto *folding = cls.folding.get();
  auto physical = [&](Operation *op) {
    return folding && folding->physical.contains(op);
  };
  // One instance: a black box with a pipeline contract stays, stepped through
  // its clock enable at the state pass; the clock reaches it, as a bit if its
  // port is one, and nothing else.
  DenseMap<Operation *, foldsim::PipelineContract> boxes;
  auto toBoxClock = [&](seq::FromClockOp op) {
    return llvm::all_of(op->getUses(), [&](OpOperand &use) {
      auto it = boxes.find(use.getOwner());
      if (it == boxes.end())
        return false;
      auto inst = cast<hw::InstanceOp>(use.getOwner());
      return inst.getInputName(use.getOperandNumber()) == it->second.clock;
    });
  };
  for (auto &op : *modelBody) {
    auto inst = dyn_cast<hw::InstanceOp>(op);
    if (!inst || physical(&op) || threads > 1)
      continue;
    auto *module = SymbolTable::lookupNearestSymbolFrom(
        inst, inst.getReferencedModuleNameAttr());
    auto contract =
        module ? foldsim::getPipelineContract(module)
               : FailureOr<std::optional<foldsim::PipelineContract>>(
                     std::optional<foldsim::PipelineContract>{});
    if (failed(contract))
      return failure();
    if (!*contract)
      return op.emitError("a black box without a pipeline contract in a "
                          "statically folded model of one instance");
    boxes[inst] = **contract;
  }
  for (auto &op : *modelBody) {
    if (physical(&op) || boxes.count(&op))
      continue;
    if (isa<hw::InstanceOp>(op))
      return op.emitError("a black box in a statically folded model of "
                          "several instances outside a folded pipeline");
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      if (reg.getIsAsync())
        return reg.emitError("asynchronous reset in a folded model");
      if (reg.getClk() != modelBody->getArgument(cls.clockInput))
        return reg.emitError("register not on the model's clock");
    }
    if (auto memory = dyn_cast<seq::FirMemOp>(op)) {
      if (memory.getInit())
        return memory.emitError("initialized memory in a folded model");
      if (memory.getReadLatency() > 1 || memory.getWriteLatency() != 1)
        return memory.emitError("a folded memory has a read latency of 0 or "
                                "1 and a write latency of 1");
    }
    if (auto clock = dyn_cast<seq::FromClockOp>(op); clock && !toBoxClock(clock))
      return op.emitError("a clock used as a bit in a folded model");
  }

  // The early outputs' next values, before any register is replaced.
  auto *modelOutput = modelBody->getTerminator();
  ImplicitLocOpBuilder mb(model.getLoc(), modelOutput);
  DenseMap<Value, Value> nexts;
  SmallVector<Value> results;
  for (auto [o, value] : llvm::enumerate(modelOutput->getOperands()))
    results.push_back(cls.late[o] ? value : nextValue(mb, value, nexts));

  // The copy's ports: the host clock, the slot's instance and whether it is a
  // state pass three host cycles ahead, then the model's inputs and outputs.
  SmallVector<hw::PortInfo> ports;
  size_t numIn = 0, numOut = 0;
  auto addPort = [&](const Twine &portName, Type type, bool isInput) {
    hw::PortInfo port;
    port.name = StringAttr::get(context, portName);
    port.type = type;
    port.dir = isInput ? hw::ModulePort::Direction::Input
                       : hw::ModulePort::Direction::Output;
    port.argNum = isInput ? numIn++ : numOut++;
    ports.push_back(port);
  };
  addPort("hostClock", seq::ClockType::get(context), true);
  addPort("state3", IntegerType::get(context, 1), true);
  if (threads > 1)
    addPort("inst3", IntegerType::get(context, idxWidth), true);
  SmallVector<unsigned> inputArg(cls.inputNames.size(), 0);
  for (auto [i, name] : llvm::enumerate(cls.inputNames)) {
    if (int(i) == cls.clockInput)
      continue;
    inputArg[i] = numIn;
    addPort(name, cls.inputTypes[i], true);
  }
  for (auto [o, name] : llvm::enumerate(cls.outputNames))
    addPort(name, cls.outputTypes[o], false);
  OpBuilder tb(model);
  auto copy = hw::HWModuleOp::create(tb, model.getLoc(),
                                     StringAttr::get(context, cls.name + "_static"),
                                     hw::ModulePortInfo(ports));
  auto *body = copy.getBodyBlock();
  body->getTerminator()->erase();
  body->getOperations().splice(body->end(), modelBody->getOperations(),
                               modelBody->begin(),
                               Block::iterator(modelOutput));
  Value hostClock = body->getArgument(0);
  Value state3 = body->getArgument(1);
  Value inst3 = threads > 1 ? body->getArgument(2) : Value();
  DenseMap<Value, Value> replaced;
  for (auto [i, arg] : llvm::enumerate(modelBody->getArguments())) {
    Value to = int(i) == cls.clockInput ? Value(hostClock)
                                        : Value(body->getArgument(inputArg[i]));
    arg.replaceAllUsesWith(to);
    replaced[arg] = to;
  }
  ImplicitLocOpBuilder b(model.getLoc(), body, body->end());
  auto i1 = b.getI1Type();
  auto one = [&]() { return hw::ConstantOp::create(b, i1, 1); };

  // Regions of state, in name order so that a region is a submodule's: each
  // its own copy of the slot's controls, from the copy's, so that no control
  // drives more than about `regionBits` bits of state; a memory larger than
  // that has a region of its own.
  // The controls two host cycles ahead, a copy for each group of
  // `kRegionsPerGroup` regions, so that the ports drive only the groups.
  struct Ahead {
    Value state2, inst2;
    unsigned regions = 0;
  };
  SmallVector<Ahead> aheads;
  auto aheadFor = [&]() -> Ahead & {
    if (aheads.empty() || aheads.back().regions == kRegionsPerGroup) {
      Ahead ahead;
      std::string prefix = ("ahead" + Twine(aheads.size())).str();
      ahead.state2 = keep(seq::FirRegOp::create(
          b, state3, hostClock, b.getStringAttr(prefix + "_state2")));
      if (threads > 1)
        ahead.inst2 = keep(seq::FirRegOp::create(
            b, inst3, hostClock, b.getStringAttr(prefix + "_inst2")));
      aheads.push_back(ahead);
    }
    ++aheads.back().regions;
    return aheads.back();
  };

  struct Region {
    std::string prefix;
    Value inst1, inst, state, instStaged;
  };
  SmallVector<Region> regions;
  uint64_t regionUsed = regionBits;
  auto regionFor = [&](uint64_t bits) -> Region & {
    if (regions.empty() || regionUsed + bits > regionBits) {
      Region region;
      auto &ahead = aheadFor();
      std::string prefix = ("region" + Twine(regions.size())).str();
      region.prefix = prefix;
      Value state1 = keep(seq::FirRegOp::create(
          b, ahead.state2, hostClock, b.getStringAttr(prefix + "_state1")));
      region.state = keep(seq::FirRegOp::create(
          b, state1, hostClock, b.getStringAttr(prefix + "_state")));
      if (threads > 1) {
        region.inst1 = keep(seq::FirRegOp::create(
            b, ahead.inst2, hostClock, b.getStringAttr(prefix + "_inst1")));
        region.inst = keep(seq::FirRegOp::create(
            b, region.inst1, hostClock, b.getStringAttr(prefix + "_inst")));
      }
      regions.push_back(region);
      regionUsed = 0;
    }
    regionUsed += bits;
    return regions.back();
  };

  SmallVector<seq::FirRegOp> registers;
  SmallVector<seq::FirMemOp> memories;
  for (auto &op : *body) {
    if (physical(&op))
      continue;
    if (auto reg = dyn_cast<seq::FirRegOp>(op))
      registers.push_back(reg);
    else if (auto memory = dyn_cast<seq::FirMemOp>(op))
      memories.push_back(memory);
  }
  // State in name order.
  SmallVector<std::pair<std::string, Operation *>> state;
  for (auto reg : registers)
    state.push_back({reg.getName().str(), reg});
  for (auto memory : memories)
    state.push_back({memory.getName().value_or("mem").str(), memory});
  llvm::stable_sort(state, [](auto &a, auto &c) { return a.first < c.first; });

  for (auto &[stateName, op] : state) {
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      b.setLoc(reg.getLoc());
      auto width = hw::getBitWidth(reg.getType());
      Value next = reg.getNext();
      if (reg.getReset())
        next = comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), next);
      auto write = foldsim::peelHolds(b, next, reg.getResult());
      if (!write.holds && !write.enable)
        write.data = next;
      auto &region = regionFor(width);
      Value value;
      if (threads == 1) {
        // One instance: the register, enabled at its state pass.
        auto backedge = BackedgeBuilder(b, reg.getLoc());
        auto nextEdge = backedge.get(reg.getType());
        auto kept = seq::FirRegOp::create(b, nextEdge, hostClock,
                                          reg.getNameAttr());
        Value enable = region.state;
        if (!write.holds && write.enable)
          enable = comb::AndOp::create(b, enable, write.enable);
        nextEdge.setValue(write.holds ? Value(kept)
                                      : Value(comb::MuxOp::create(
                                            b, enable, write.data, kept)));
        value = kept;
      } else {
        auto memType = seq::FirMemType::get(context, threads, width, 1);
        auto memory = seq::FirMemOp::create(
            b, memType, /*readLatency=*/0, /*writeLatency=*/1,
            seq::RUW::Undefined, seq::WUW::Undefined, reg.getNameAttr(),
            hw::InnerSymAttr{}, seq::FirMemInitAttr{}, StringAttr{},
            Attribute{});
        Value data = seq::FirMemReadOp::create(b, memory, region.inst1,
                                               hostClock, one());
        // The read register: the memory's output register.
        auto readEdges = BackedgeBuilder(b, reg.getLoc());
        auto readNext = readEdges.get(data.getType());
        Value read = seq::FirRegOp::create(
            b, readNext, hostClock, b.getStringAttr(reg.getName() + "_read"));
        value = b.createOrFold<hw::BitcastOp>(reg.getType(), read);
        Value loaded = data;
        if (!write.holds) {
          // Written a host cycle after the state pass, from a stage, so that
          // the register's next value ends at a flop, not at the memory's
          // write port.
          Value enable = region.state;
          if (write.enable)
            enable = comb::AndOp::create(b, enable, write.enable);
          if (!region.instStaged)
            region.instStaged = keep(seq::FirRegOp::create(
                b, region.inst, hostClock,
                b.getStringAttr(region.prefix + "_inst_staged")));
          Value stagedEnable = seq::FirRegOp::create(
              b, enable, hostClock, b.getStringAttr(reg.getName() + "_wen"));
          Value stagedData =
              seq::FirRegOp::create(b, asBits(b, write.data), hostClock,
                                    b.getStringAttr(reg.getName() + "_wdata"));
          seq::FirMemWriteOp::create(b, memory, region.instStaged, hostClock,
                                     stagedEnable, stagedData, Value{});
          // The next slot's instance may be written but not yet stored: a
          // staged write lands at the end of this host cycle, and a state pass
          // now writes a host cycle later. The read register takes the newest.
          Value staged = comb::AndOp::create(
              b, stagedEnable,
              comb::ICmpOp::create(b, comb::ICmpPredicate::eq,
                                   region.instStaged, region.inst1));
          loaded = comb::MuxOp::create(b, staged, stagedData, loaded);
          Value now = comb::AndOp::create(
              b, region.state,
              comb::ICmpOp::create(b, comb::ICmpPredicate::eq, region.inst,
                                   region.inst1));
          Value next = asBits(b, write.data);
          if (write.enable)
            next = comb::MuxOp::create(b, write.enable, next, read);
          loaded = comb::MuxOp::create(b, now, next, loaded);
        }
        readNext.setValue(loaded);
      }
      reg.getResult().replaceAllUsesWith(value);
      replaced[reg.getResult()] = value;
      reg.erase();
      continue;
    }
    auto memory = cast<seq::FirMemOp>(op);
    b.setLoc(memory.getLoc());
    auto memType = memory.getType();
    uint64_t bits = uint64_t(memType.getWidth()) * memType.getDepth();
    auto &region = regionFor(bits);
    auto threadedType = seq::FirMemType::get(
        context, memType.getDepth() * threads, memType.getWidth(),
        memType.getMaskWidth());
    auto threadedMemory = seq::FirMemOp::create(
        b, threadedType, memory.getReadLatency(), memory.getWriteLatency(),
        memory.getRuw(), memory.getWuw(), memory.getNameAttr(),
        hw::InnerSymAttr{}, seq::FirMemInitAttr{}, memory.getPrefixAttr(),
        memory.getOutputFileAttr());
    threadedMemory->setDialectAttrs(memory->getDialectAttrs());
    if (folding)
      if (auto it = folding->foldOfStore.find(memory);
          it != folding->foldOfStore.end())
        folding->folds[it->second].threadedStore = threadedMemory;
    unsigned addrWidth = llvm::Log2_64_Ceil(threadedType.getDepth()) - idxWidth;
    // The instance in the low address bits.
    auto addressOf = [&](Value addr) -> Value {
      if (!idxWidth)
        return addr;
      if (!addrWidth)
        return region.inst;
      Value low =
          b.createOrFold<comb::ExtractOp>(b.getIntegerType(addrWidth), addr, 0);
      return b.createOrFold<comb::ConcatOp>(low, region.inst);
    };
    for (auto *user : llvm::make_early_inc_range(memory->getUsers())) {
      if (auto write = dyn_cast<seq::FirMemWriteOp>(user)) {
        Value enable = region.state;
        if (write.getEnable())
          enable = comb::AndOp::create(b, enable, write.getEnable());
        seq::FirMemWriteOp::create(b, threadedMemory,
                                   addressOf(write.getAddress()), hostClock,
                                   enable, write.getData(), write.getMask());
        write.erase();
        continue;
      }
      auto read = cast<seq::FirMemReadOp>(user);
      Value data;
      if (memory.getReadLatency() == 0) {
        data = seq::FirMemReadOp::create(b, threadedMemory,
                                         addressOf(read.getAddress()),
                                         hostClock, read.getEnable());
      } else {
        // A registered read on the state pass; its data, a host cycle later,
        // is the instance's read register's, kept per instance.
        Value enable = region.state;
        if (read.getEnable())
          enable = comb::AndOp::create(b, enable, read.getEnable());
        Value dout = seq::FirMemReadOp::create(
            b, threadedMemory, addressOf(read.getAddress()), hostClock, enable);
        auto readName = (memory.getName().value_or("mem") + "_rdata").str();
        Value capture = seq::FirRegOp::create(
            b, enable, hostClock, b.getStringAttr(readName + "_capture"),
            hw::InnerSymAttr{}, b.getIntegerAttr(i1, 0));
        if (threads == 1) {
          auto backedge = BackedgeBuilder(b, memory.getLoc());
          auto nextEdge = backedge.get(dout.getType());
          auto buffer = seq::FirRegOp::create(b, nextEdge, hostClock,
                                              b.getStringAttr(readName));
          nextEdge.setValue(comb::MuxOp::create(b, capture, dout, buffer));
          data = buffer;
        } else {
          auto width = hw::getBitWidth(dout.getType());
          auto bufferType = seq::FirMemType::get(context, threads, width, 1);
          auto buffer = seq::FirMemOp::create(
              b, bufferType, /*readLatency=*/0, /*writeLatency=*/1,
              seq::RUW::Undefined, seq::WUW::Undefined,
              b.getStringAttr(readName), hw::InnerSymAttr{},
              seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
          Value readInst = seq::FirRegOp::create(
              b, region.inst, hostClock, b.getStringAttr(readName + "_inst"));
          seq::FirMemWriteOp::create(b, buffer, readInst, hostClock, capture,
                                     asBits(b, dout), Value{});
          Value bufferData = seq::FirMemReadOp::create(b, buffer, region.inst1,
                                                       hostClock, one());
          // The next slot's instance's data, out of the memory now, is not
          // yet in the buffer.
          Value fresh = comb::AndOp::create(
              b, capture,
              comb::ICmpOp::create(b, comb::ICmpPredicate::eq, readInst,
                                   region.inst1));
          bufferData = comb::MuxOp::create(b, fresh, asBits(b, dout), bufferData);
          data = seq::FirRegOp::create(b, bufferData, hostClock,
                                       b.getStringAttr(readName + "_read"));
          data = b.createOrFold<hw::BitcastOp>(dout.getType(), data);
        }
      }
      read.getData().replaceAllUsesWith(data);
      replaced[read.getData()] = data;
      read.erase();
    }
    memory.erase();
  }

  // A black box of one instance takes a target edge at its state pass, which
  // its contract makes the only edges it sees.
  for (auto &[op, contract] : boxes) {
    auto inst = cast<hw::InstanceOp>(op);
    b.setLoc(inst.getLoc());
    uint64_t bits = 0;
    for (auto result : inst.getResults())
      bits += hw::getBitWidth(result.getType());
    auto &region = regionFor(bits * contract.latency);
    for (unsigned i = 0, e = inst.getNumOperands(); i < e; ++i) {
      auto name = inst.getInputName(i);
      if (name == contract.clock) {
        auto type = inst.getOperand(i).getType();
        inst.setOperand(i, isa<seq::ClockType>(type)
                               ? hostClock
                               : Value(seq::FromClockOp::create(b, hostClock)));
      } else if (name == contract.enable) {
        inst.setOperand(i, comb::AndOp::create(b, region.state,
                                               inst.getOperand(i)));
      }
    }
  }
  for (auto &op : llvm::make_early_inc_range(*body))
    if (isa<seq::FromClockOp>(op) && op.use_empty())
      op.erase();

  // The folded pipelines' copies run on the host clock and issue at the
  // slot's state pass for its instance; their results reach each store
  // without the state pass's enable, as they come back.
  if (folding) {
    auto &ahead = aheadFor();
    Value state1 = keep(seq::FirRegOp::create(b, ahead.state2, hostClock,
                                              b.getStringAttr("fold_state1")));
    Value state = keep(seq::FirRegOp::create(b, state1, hostClock,
                                             b.getStringAttr("fold_state")));
    Value inst1 = keep(seq::FirRegOp::create(b, ahead.inst2, hostClock,
                                             b.getStringAttr("fold_inst1")));
    Value inst = keep(seq::FirRegOp::create(b, inst1, hostClock,
                                            b.getStringAttr("fold_inst")));
    for (auto &fold : folding->folds)
      seq::FirMemWriteOp::create(
          b, fold.threadedStore,
          b.createOrFold<comb::ConcatOp>(fold.slot, fold.thread), hostClock,
          fold.valid, fold.data, Value{});
    folding->physClock.setValue(hostClock);
    folding->thread.setValue(inst);
    folding->edge.setValue(state);
    folding->reset.setValue(hw::ConstantOp::create(b, i1, 0));
    folding->edges.reset();
  }

  SmallVector<Value> outputs;
  for (auto value : results) {
    auto it = replaced.find(value);
    outputs.push_back(it == replaced.end() ? value : it->second);
  }
  b.setInsertionPointToEnd(body);
  hw::OutputOp::create(b, outputs);
  cls.copy = copy;
  model.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// The simulator
//===----------------------------------------------------------------------===//

/// A table indexed by `index`, of constants of `width` bits.
static Value table(ImplicitLocOpBuilder &b, Value index, ArrayRef<uint64_t> rows,
                   unsigned width) {
  auto type = b.getIntegerType(std::max(width, 1u));
  bool uniform = llvm::all_of(rows, [&](uint64_t v) { return v == rows[0]; });
  if (rows.empty() || uniform)
    return hw::ConstantOp::create(b, type, rows.empty() ? 0 : rows[0]);
  unsigned indexWidth = hw::getBitWidth(index.getType());
  SmallVector<Value> elements;
  for (uint64_t k = (uint64_t(1) << indexWidth); k-- > 0;)
    elements.push_back(
        hw::ConstantOp::create(b, type, k < rows.size() ? rows[k] : 0));
  Value array = hw::ArrayCreateOp::create(b, elements);
  return hw::ArrayGetOp::create(b, array, index);
}

LogicalResult StaticFolder::buildSim() {
  auto *context = top.getContext();
  auto i1 = IntegerType::get(context, 1);

  // Ports: the host clock and reset, a finish bit per class, and a token
  // channel per netlist port but clocks.
  SmallVector<hw::PortInfo> ports;
  size_t numIn = 0, numOut = 0;
  auto addPort = [&](const Twine &portName, Type type, bool isInput) {
    hw::PortInfo port;
    port.name = StringAttr::get(context, portName);
    port.type = type;
    port.dir = isInput ? hw::ModulePort::Direction::Input
                       : hw::ModulePort::Direction::Output;
    port.argNum = isInput ? numIn++ : numOut++;
    ports.push_back(port);
  };
  addPort("hostClock", seq::ClockType::get(context), true);
  addPort("hostReset", i1, true);
  addPort("profile_finish", IntegerType::get(context, classes.size()), false);
  struct InputPorts {
    unsigned valid, bits;
  };
  SmallVector<InputPorts> inputPorts(producers.size());
  for (auto [p, producer] : llvm::enumerate(producers)) {
    if (producer.cls >= 0)
      continue;
    addPort(producer.name + "_ready", i1, false);
    inputPorts[p].valid = numIn;
    addPort(producer.name + "_valid", i1, true);
    inputPorts[p].bits = numIn;
    if (producer.width > 0)
      addPort(producer.name + "_bits", IntegerType::get(context, producer.width),
              true);
  }
  SmallVector<unsigned> outputReady(consumers.size(), 0);
  for (auto [k, consumer] : llvm::enumerate(consumers)) {
    if (consumer.cls >= 0)
      continue;
    outputReady[k] = numIn;
    addPort(consumer.name + "_ready", i1, true);
    addPort(consumer.name + "_valid", i1, false);
    if (consumer.width > 0)
      addPort(consumer.name + "_bits", IntegerType::get(context, consumer.width),
              false);
  }
  OpBuilder tb(netlist);
  auto sim = hw::HWModuleOp::create(tb, netlist.getLoc(),
                                    StringAttr::get(context, simName),
                                    hw::ModulePortInfo(ports));
  auto *body = sim.getBodyBlock();
  body->getTerminator()->erase();
  ImplicitLocOpBuilder b(netlist.getLoc(), body, body->end());
  BackedgeBuilder backedges(b, netlist.getLoc());
  Value hostClock = body->getArgument(0);
  Value hostReset = body->getArgument(1);
  auto constant = [&](unsigned width, uint64_t value) -> Value {
    return hw::ConstantOp::create(b, b.getIntegerType(width), value);
  };
  auto notOf = [&](Value v) { return comb::XorOp::create(b, v, constant(1, 1)); };
  auto reg = [&](Value next, const Twine &regName) -> Value {
    return seq::FirRegOp::create(b, next, hostClock, b.getStringAttr(regName));
  };
  // A copy of a register, kept apart from the register it copies.
  auto copyReg = [&](Value next, const Twine &regName) -> Value {
    return keep(
        seq::FirRegOp::create(b, next, hostClock, b.getStringAttr(regName)));
  };
  auto resetReg = [&](Value next, const Twine &regName, uint64_t init) -> Value {
    auto type = next.getType();
    return seq::FirRegOp::create(
        b, next, hostClock, b.getStringAttr(regName), hostReset,
        hw::ConstantOp::create(b, type, init), hw::InnerSymAttr{},
        /*isAsync=*/false, b.getIntegerAttr(type, init));
  };
  auto eq = [&](Value v, uint64_t k) -> Value {
    return comb::ICmpOp::create(
        b, comb::ICmpPredicate::eq, v,
        hw::ConstantOp::create(b, v.getType(), k));
  };

  // The frame's position three host cycles ahead: a slot, or `ii` between
  // frames; the frame's parity with it. A frame starts when every input has
  // its token and every output room for one.
  unsigned posWidth = llvm::Log2_32_Ceil(ii + 1);
  auto go = backedges.get(i1);
  // A position's next: the next slot, or at the frame's end the next frame's
  // first if it starts, else `ii`.
  auto nextPosition = [&](Value pos) -> Value {
    Value end = comb::OrOp::create(b, eq(pos, ii - 1), eq(pos, ii));
    return comb::MuxOp::create(
        b, end,
        comb::MuxOp::create(b, go, constant(posWidth, 0), constant(posWidth, ii)),
        comb::AddOp::create(b, pos, constant(posWidth, 1)));
  };
  auto pos3Next = backedges.get(b.getIntegerType(posWidth));
  Value pos3 = resetReg(pos3Next, "pos3", ii);
  Value boundary = comb::OrOp::create(b, eq(pos3, ii - 1), eq(pos3, ii));
  Value starting = comb::AndOp::create(b, boundary, go);
  Value rootNext = nextPosition(pos3);
  pos3Next.setValue(rootNext);
  auto par3Next = backedges.get(i1);
  Value par3 = resetReg(par3Next, "parity3", 1);
  par3Next.setValue(comb::MuxOp::create(b, starting, notOf(par3), par3));
  Value pos2 = resetReg(pos3, "pos2", ii), par2 = reg(par3, "parity2");
  Value pos1 = resetReg(pos2, "pos1", ii), par1 = reg(par2, "parity1");
  Value pos0 = resetReg(pos1, "pos0", ii);

  // Each class's slot table, its instance and passes three host cycles ahead
  // (the table's), two, one, then now.
  struct Controls {
    Value inst3, state3;
    Value inst2, state2, ready2, tail2;
    Value inst1, inst, state, ready;
  };
  SmallVector<Controls> controls(classes.size());
  // Nested folding: the sequencer tree. Each class has its own sequencer,
  // the frame's position three host cycles ahead, kept in a register its
  // parent class's sequencer loads with its next position (the root's for a
  // class of no parent): every class's slot table reads a sequencer of its
  // own, a parent's sequencer drives only its children's, and the frame's
  // start (`go`) is the one control all sequencers take.
  SmallVector<Value> sequencer(classes.size());
  if (nest) {
    SmallVector<unsigned> order;
    for (unsigned c = 0; c < classes.size(); ++c)
      order.push_back(c);
    llvm::stable_sort(order, [&](unsigned x, unsigned y) {
      return classes[x].depth < classes[y].depth;
    });
    SmallVector<Value> nextOf(classes.size());
    for (auto c : order) {
      auto &cls = classes[c];
      Value load = rootNext;
      if (cls.parent >= 0) {
        auto &parentNext = nextOf[cls.parent];
        if (!parentNext)
          parentNext = nextPosition(sequencer[cls.parent]);
        load = parentNext;
      }
      // Kept: equivalent to every other sequencer, it must stay a register
      // of its own for its class.
      auto type = load.getType();
      sequencer[c] = keep(seq::FirRegOp::create(
          b, load, hostClock, b.getStringAttr(cls.name + "_seq"), hostReset,
          hw::ConstantOp::create(b, type, ii), hw::InnerSymAttr{},
          /*isAsync=*/false, b.getIntegerAttr(type, ii)));
    }
  }
  // Overlapped frames: a slot past the frame runs in the next frame's, for
  // the frame before it, so none runs until a frame has finished.
  bool overlapped = llvm::any_of(classes, [&](const Class &cls) {
    return llvm::any_of(cls.slot[kState], [&](int s) { return s >= int(ii); }) ||
           llvm::any_of(cls.slot[kReady], [&](int s) { return s >= int(ii); });
  });
  Value primed;
  if (overlapped) {
    auto startedNext = backedges.get(i1);
    Value started = resetReg(startedNext, "started", 0);
    startedNext.setValue(comb::OrOp::create(b, started, starting));
    auto primedNext = backedges.get(i1);
    primed = resetReg(primedNext, "primed", 0);
    primedNext.setValue(comb::OrOp::create(
        b, primed, comb::AndOp::create(b, starting, started)));
  }
  for (auto [c, cls] : llvm::enumerate(classes)) {
    unsigned threads = cls.threads();
    unsigned idxWidth = threads > 1 ? llvm::Log2_32_Ceil(threads) : 0;
    SmallVector<uint64_t> instRow(ii + 1, 0), stateRow(ii + 1, 0),
        readyRow(ii + 1, 0), tailRow(ii + 1, 0);
    for (unsigned i = 0; i < threads; ++i)
      for (auto pass : {kReady, kState}) {
        if (pass == kReady && !cls.hasReady)
          continue;
        unsigned s = cls.slot[pass][i];
        instRow[s % ii] = i;
        (pass == kReady ? readyRow : stateRow)[s % ii] = 1;
        tailRow[s % ii] = s >= ii;
      }
    Value at = nest ? sequencer[c] : pos3;
    auto &ctl = controls[c];
    auto prefix = cls.name;
    ctl.state3 = table(b, at, stateRow, 1);
    Value ready3 = table(b, at, readyRow, 1);
    if (overlapped) {
      // A slot of the frame before runs once a frame has finished.
      Value tail3 = table(b, at, tailRow, 1);
      Value runs = comb::OrOp::create(b, notOf(tail3), primed);
      ctl.state3 = comb::AndOp::create(b, ctl.state3, runs);
      ready3 = comb::AndOp::create(b, ready3, runs);
      ctl.tail2 = reg(tail3, prefix + "_tail2");
    }
    ctl.state2 = reg(ctl.state3, prefix + "_state2");
    ctl.ready2 = reg(ready3, prefix + "_ready2");
    Value state1 = reg(ctl.state2, prefix + "_state1");
    Value ready1 = reg(ctl.ready2, prefix + "_ready1");
    ctl.state = reg(state1, prefix + "_state0");
    ctl.ready = reg(ready1, prefix + "_ready0");
    if (threads > 1) {
      ctl.inst3 = table(b, at, instRow, idxWidth);
      ctl.inst2 = reg(ctl.inst3, prefix + "_inst2");
      ctl.inst1 = reg(ctl.inst2, prefix + "_inst1");
      ctl.inst = reg(ctl.inst1, prefix + "_inst0");
    }
  }

  // The netlist's inputs: each token taken into a holding queue as it comes,
  // and the head moved into the frame's register on the host cycle before the
  // frame's first slot. A frame is decided four host cycles before that slot,
  // so below four host cycles a frame may be decided before the previous one
  // is loaded: the queue then holds a token for each, and a token counts for
  // the frame being decided only beyond those of the decided frames not yet
  // loaded.
  Value load = eq(pos1, 0);
  unsigned holdDepth = 1 + (ii < 4 ? 3 / ii : 0);
  unsigned cntWidth = llvm::Log2_32_Ceil(holdDepth + 2);
  Value decidedUnloaded;
  if (holdDepth > 1) {
    // Frames decided and not yet loaded.
    auto pendingNext = backedges.get(b.getIntegerType(cntWidth));
    decidedUnloaded = resetReg(pendingNext, "frames_unloaded", 0);
    Value plus = comb::AddOp::create(b, decidedUnloaded, constant(cntWidth, 1));
    Value minus = comb::SubOp::create(b, decidedUnloaded, constant(cntWidth, 1));
    pendingNext.setValue(comb::MuxOp::create(
        b, starting, comb::MuxOp::create(b, load, decidedUnloaded, plus),
        comb::MuxOp::create(b, load, minus, decidedUnloaded)));
  }
  DenseMap<unsigned, Value> frameInput, frameInputNext;
  SmallVector<Value> inputValid;
  SmallVector<std::pair<unsigned, Value>> inputReady;
  for (auto [p, producer] : llvm::enumerate(producers)) {
    if (producer.cls >= 0)
      continue;
    Value valid = body->getArgument(inputPorts[p].valid);
    Value bits = producer.width ? body->getArgument(inputPorts[p].bits) : Value();
    Value head;
    if (holdDepth == 1) {
      auto fullNext = backedges.get(i1);
      Value full = resetReg(fullNext, producer.name + "_held", 0);
      Value take = comb::AndOp::create(b, valid, notOf(full));
      fullNext.setValue(comb::MuxOp::create(
          b, take, constant(1, 1),
          comb::MuxOp::create(b, load, constant(1, 0), full)));
      inputValid.push_back(comb::OrOp::create(b, full, valid));
      inputReady.push_back({unsigned(p), notOf(full)});
      if (producer.width) {
        auto heldNext = backedges.get(b.getIntegerType(producer.width));
        head = reg(heldNext, producer.name + "_token");
        heldNext.setValue(comb::MuxOp::create(b, take, bits, head));
      }
    } else {
      auto countNext = backedges.get(b.getIntegerType(cntWidth));
      Value count = resetReg(countNext, producer.name + "_held", 0);
      Value space = comb::ICmpOp::create(b, comb::ICmpPredicate::ult, count,
                                         constant(cntWidth, holdDepth));
      Value take = comb::AndOp::create(b, valid, space);
      Value pop = comb::AndOp::create(
          b, load, comb::ICmpOp::create(b, comb::ICmpPredicate::ne, count,
                                        constant(cntWidth, 0)));
      Value plus = comb::AddOp::create(b, count, constant(cntWidth, 1));
      Value minus = comb::SubOp::create(b, count, constant(cntWidth, 1));
      countNext.setValue(comb::MuxOp::create(
          b, take, comb::MuxOp::create(b, pop, count, plus),
          comb::MuxOp::create(b, pop, minus, count)));
      // Held, or offered and taken now, beyond the decided frames' tokens.
      Value have = comb::AddOp::create(
          b, count, comb::ConcatOp::create(b, constant(cntWidth - 1, 0), take));
      inputValid.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::ugt,
                                                have, decidedUnloaded));
      inputReady.push_back({unsigned(p), space});
      if (producer.width) {
        Value afterPop = comb::MuxOp::create(b, pop, minus, count);
        SmallVector<Backedge> nexts;
        SmallVector<Value> entries;
        for (unsigned e = 0; e < holdDepth; ++e) {
          nexts.push_back(backedges.get(b.getIntegerType(producer.width)));
          entries.push_back(reg(nexts[e], producer.name + "_token" + Twine(e)));
        }
        for (unsigned e = 0; e < holdDepth; ++e) {
          Value kept = e + 1 < holdDepth
                           ? comb::MuxOp::create(b, pop, entries[e + 1], entries[e])
                           : entries[e];
          Value here = comb::AndOp::create(
              b, take, comb::ICmpOp::create(b, comb::ICmpPredicate::eq, afterPop,
                                            constant(cntWidth, e)));
          nexts[e].setValue(comb::MuxOp::create(b, here, bits, kept));
        }
        head = entries[0];
      }
    }
    if (producer.width == 0)
      continue;
    auto next = backedges.get(b.getIntegerType(producer.width));
    Value x = reg(next, producer.name + "_frame");
    Value taken = comb::MuxOp::create(b, load, head, x);
    next.setValue(taken);
    frameInput[p] = x;
    frameInputNext[p] = taken;
  }
  // The frame's input as a class's evaluation a host cycle ahead reads it:
  // in a slot past the frame, the frame before's.
  DenseMap<unsigned, Value> prevFrameInputNext;
  DenseMap<unsigned, Value> classTail1;
  auto frameInputFor = [&](unsigned p, unsigned c) -> Value {
    if (!controls[c].tail2)
      return frameInputNext[p];
    Value &prevNext = prevFrameInputNext[p];
    if (!prevNext) {
      auto next = backedges.get(frameInput[p].getType());
      Value prev = reg(next, producers[p].name + "_prev");
      prevNext = comb::MuxOp::create(b, load, frameInput[p], prev);
      next.setValue(prevNext);
    }
    Value &tail1 = classTail1[c];
    if (!tail1)
      tail1 = reg(controls[c].tail2, classes[c].name + "_tail1");
    return comb::MuxOp::create(b, tail1, prevNext, frameInputNext[p]);
  };
  // A class's own copy of a frame's input (as a reset, it drives much of the
  // class's state).
  std::map<std::pair<unsigned, unsigned>, Value> classFrameInputs;
  auto classFrameInput = [&](unsigned p, unsigned c) -> Value {
    Value &copy = classFrameInputs[{p, c}];
    if (!copy)
      copy = copyReg(frameInputFor(p, c),
                     producers[p].name + "_frame_" + classes[c].name);
    return copy;
  };

  // The copies, their inputs backedges until the links are built.
  SmallVector<hw::InstanceOp> copies(classes.size());
  SmallVector<SmallVector<Backedge>> copyInputs(classes.size());
  for (auto [c, cls] : llvm::enumerate(classes)) {
    SmallVector<Value> operands{hostClock, controls[c].state3};
    if (cls.threads() > 1)
      operands.push_back(controls[c].inst3);
    for (auto [i, type] : llvm::enumerate(cls.inputTypes)) {
      if (int(i) == cls.clockInput)
        continue;
      copyInputs[c].push_back(backedges.get(type));
      operands.push_back(copyInputs[c].back());
    }
    copies[c] = hw::InstanceOp::create(b, cls.copy,
                                       b.getStringAttr(cls.name + "_copy"),
                                       operands);
  }

  // Groups of links, each with its own copy of its producer's controls and of
  // the frame's parity, so that no control drives more than about
  // `regionBits` bits of links.
  struct LinkGroup {
    std::string prefix;
    Value state, ready, inst, bank;
    uint64_t used = 0;
  };
  SmallVector<SmallVector<LinkGroup>> linkGroups(classes.size());
  auto linkGroupFor = [&](unsigned c, uint64_t bits) -> LinkGroup & {
    auto &groups = linkGroups[c];
    if (groups.empty() || groups.back().used + bits > regionBits) {
      LinkGroup group;
      group.prefix = (classes[c].name + "_links" + Twine(groups.size())).str();
      groups.push_back(group);
    }
    groups.back().used += bits;
    return groups.back();
  };

  // A link: a memory per read of a producer's bits, written by the producer
  // at its pass, at its instance (and an early one's next frame's bank), and
  // read at an address the reader gives. A wide link is cut in pieces of at
  // most `regionBits` bits, each in a group and read at its own copy of the
  // address and the bank.
  // With `now`, also what the producer writes in this host cycle and whether
  // that is the entry a reader in this host cycle's slot takes (a late link).
  auto linkRead = [&](unsigned p, unsigned low, unsigned width,
                      function_ref<Value()> address, function_ref<Value()> bank,
                      const Twine &linkName,
                      std::pair<Value, Value> *now = nullptr) -> Value {
    auto &producer = producers[p];
    unsigned c = producer.cls;
    auto &ctl = controls[c];
    unsigned threads = classes[c].threads();
    bool banked = !producer.late;
    Value bits = asBits(b, copies[c].getResult(producer.port));
    std::string baseName = linkName.str();
    SmallVector<Value> pieces, nowPieces;
    for (unsigned at = 0; at < width; at += regionBits) {
      unsigned w = std::min<unsigned>(regionBits, width - at);
      std::string name =
          width > regionBits ? baseName + "_" + std::to_string(at) : baseName;
      Value data = b.createOrFold<comb::ExtractOp>(b.getIntegerType(w), bits,
                                                   low + at);
      auto &group = linkGroupFor(c, w);
      Value &enable = producer.atReady ? group.ready : group.state;
      if (!enable) {
        auto pass = producer.atReady ? "_ready" : "_state";
        enable = copyReg(
            copyReg(producer.atReady ? ctl.ready2 : ctl.state2,
                    group.prefix + pass + "1"),
            group.prefix + pass + "0");
      }
      if (now && !banked) {
        if (!now->first)
          now->first = enable;
        nowPieces.push_back(data);
      }
      if (threads == 1 && !banked) {
        auto next = backedges.get(data.getType());
        Value r = reg(next, name);
        Value written = comb::MuxOp::create(b, enable, data, r);
        next.setValue(written);
        pieces.push_back(oneCycleLinks ? written : r);
        continue;
      }
      unsigned depth = threads * (banked ? 2 : 1);
      auto memType = seq::FirMemType::get(context, depth, w, 1);
      auto memory = seq::FirMemOp::create(
          b, memType, /*readLatency=*/0, /*writeLatency=*/1,
          seq::RUW::Undefined, seq::WUW::Undefined, b.getStringAttr(name),
          hw::InnerSymAttr{}, seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
      // The bank in the high bit, the instance in the low ones.
      SmallVector<Value> writeAddress, readAddress;
      if (banked) {
        if (!group.bank) {
          // The frame the write belongs to: the one before in a slot past it.
          Value parity = copyReg(par1, group.prefix + "_parity0");
          if (ctl.tail2)
            parity = comb::XorOp::create(
                b, parity,
                copyReg(copyReg(ctl.tail2, group.prefix + "_tail1"),
                        group.prefix + "_tail0"));
          group.bank = notOf(parity);
        }
        writeAddress.push_back(group.bank);
        readAddress.push_back(bank());
      }
      if (threads > 1) {
        if (!group.inst)
          group.inst = copyReg(copyReg(ctl.inst2, group.prefix + "_inst1"),
                               group.prefix + "_inst0");
        writeAddress.push_back(group.inst);
        readAddress.push_back(address());
      }
      Value writeAt = b.createOrFold<comb::ConcatOp>(writeAddress);
      Value readAt = b.createOrFold<comb::ConcatOp>(readAddress);
      seq::FirMemWriteOp::create(b, memory, writeAt, hostClock, enable, data,
                                 Value{});
      Value read =
          seq::FirMemReadOp::create(b, memory, readAt, hostClock, constant(1, 1));
      if (now && !banked && nowPieces.size() == 1) {
        // The address is the next slot's reader's; this slot's, a cycle on.
        Value readNow = readAt.getDefiningOp<hw::ConstantOp>()
                            ? readAt
                            : copyReg(readAt, name + "_now");
        now->first = comb::AndOp::create(
            b, enable,
            comb::ICmpOp::create(b, comb::ICmpPredicate::eq, writeAt, readNow));
      }
      if (oneCycleLinks) {
        // A write reaches a read of its entry in the same host cycle.
        Value hit = comb::AndOp::create(
            b, enable,
            comb::ICmpOp::create(b, comb::ICmpPredicate::eq, writeAt, readAt));
        read = comb::MuxOp::create(b, hit, data, read);
      }
      pieces.push_back(read);
    }
    if (now && !nowPieces.empty())
      now->second = nowPieces.size() == 1
                        ? nowPieces[0]
                        : b.createOrFold<comb::ConcatOp>(SmallVector<Value>(
                              nowPieces.rbegin(), nowPieces.rend()));
    if (pieces.empty())
      return constant(0, 0);
    if (pieces.size() == 1)
      return pieces[0];
    return b.createOrFold<comb::ConcatOp>(
        SmallVector<Value>(pieces.rbegin(), pieces.rend()));
  };

  // Each model input: its pieces at positions common to every instance; at
  // each, the sources the instances take it from, each read at the slot's
  // instance's producer instance, chosen by the instance, into the read
  // register. Netlist inputs and shared constants are wired around it.
  for (auto [k, entry] : llvm::enumerate(consumers)) {
    auto &consumer = entry;
    if (consumer.cls < 0)
      continue;
    unsigned c = consumer.cls;
    auto &cls = classes[c];
    auto &ctl = controls[c];
    unsigned threads = cls.threads();
    std::set<unsigned> cuts{consumer.width};
    for (auto &pieces : consumer.pieces) {
      unsigned at = 0;
      for (auto &piece : pieces)
        cuts.insert(at += piece.width);
    }
    // The piece of instance i covering bit `at`, and its offset.
    auto pieceAt = [&](unsigned i, unsigned at) {
      unsigned base = 0;
      for (auto &piece : consumer.pieces[i]) {
        if (at < base + piece.width)
          return std::make_pair(&piece, at - base);
        base += piece.width;
      }
      llvm_unreachable("a bit not covered by a piece");
    };
    // Low bits first.
    SmallVector<Value> parts;
    SmallVector<Value> registered;
    SmallVector<std::pair<unsigned, unsigned>> registeredAt;
    SmallVector<std::pair<Value, Value>> takenAt;
    unsigned at = 0;
    for (unsigned end : cuts) {
      if (end == at)
        continue;
      unsigned width = end - at;
      // The sources: (producer, low), or -1 for a constant.
      std::map<std::pair<int, unsigned>, unsigned> sources;
      SmallVector<std::pair<int, unsigned>> sourceOf(threads);
      SmallVector<uint64_t> producerInst(threads, 0);
      SmallVector<APInt> constants(threads);
      bool frame = false;
      for (unsigned i = 0; i < threads; ++i) {
        auto [piece, offset] = pieceAt(i, at);
        std::pair<int, unsigned> key{-1, 0};
        if (piece->producer >= 0) {
          key = {piece->producer, piece->low + offset};
          producerInst[i] = piece->inst;
          if (producers[piece->producer].cls < 0)
            frame = true;
        } else {
          constants[i] = piece->constant.extractBits(width, offset);
        }
        sources.try_emplace(key, sources.size());
        sourceOf[i] = key;
      }
      // Wired around the read register: the frame's input, or one constant.
      if (sources.size() == 1) {
        auto key = sources.begin()->first;
        if (key.first < 0 &&
            llvm::all_of(constants, [&](const APInt &v) { return v == constants[0]; })) {
          parts.push_back(hw::ConstantOp::create(b, constants[0]));
          at = end;
          continue;
        }
        if (frame) {
          parts.push_back(b.createOrFold<comb::ExtractOp>(
              b.getIntegerType(width), classFrameInput(key.first, c),
              key.second));
          at = end;
          continue;
        }
      }
      SmallVector<Value> sourceData(sources.size());
      SmallVector<std::pair<Value, Value>> sourceNow(sources.size());
      for (auto &[key, index] : sources) {
        if (key.first < 0) {
          SmallVector<uint64_t> rows(threads, 0);
          for (unsigned i = 0; i < threads; ++i)
            if (sourceOf[i] == key)
              rows[i] = constants[i].getZExtValue();
          if (width > 64)
            return netlist.emitError("a constant of over 64 bits differing by "
                                     "instance: ")
                   << consumer.name;
          sourceData[index] =
              threads > 1 ? table(b, ctl.inst1, rows, width)
                          : hw::ConstantOp::create(b, b.getIntegerType(width),
                                                   rows[0]);
          continue;
        }
        auto &producer = producers[key.first];
        if (producer.cls < 0) {
          // The frame's input as the read register will hold it: the token
          // being taken before the frame's first slot.
          sourceData[index] = b.createOrFold<comb::ExtractOp>(
              b.getIntegerType(width), frameInputFor(key.first, c), key.second);
          continue;
        }
        unsigned pThreads = classes[producer.cls].threads();
        SmallVector<uint64_t> rows(threads, 0);
        for (unsigned i = 0; i < threads; ++i)
          if (sourceOf[i] == key)
            rows[i] = producerInst[i];
        unsigned w = llvm::Log2_32_Ceil(pThreads);
        // Each piece of the link reads at its own copy of the address and
        // the bank.
        unsigned addressCopies = 0, bankCopies = 0;
        auto address = [&]() -> Value {
          if (pThreads <= 1)
            return Value();
          if (threads == 1)
            return constant(w, rows[0]);
          return copyReg(table(b, ctl.inst2, rows, w),
                         consumer.name + "_addr" + Twine(at) + "_" +
                             Twine(addressCopies++));
        };
        auto bank = [&]() -> Value {
          std::string name = (consumer.name + "_parity1_" + Twine(at) + "_" +
                              Twine(bankCopies++))
                                 .str();
          Value parity = copyReg(par2, name);
          if (ctl.tail2)
            parity = comb::XorOp::create(b, parity,
                                         copyReg(ctl.tail2, name + "_tail"));
          return parity;
        };
        sourceData[index] =
            linkRead(key.first, key.second, width, address, bank,
                     producer.name + "_to_" + consumer.name + "_" + Twine(at),
                     zeroLink(key.first, c, consumer.port) ? &sourceNow[index]
                                                           : nullptr);
      }
      // Taken around the read register when written in the reader's slot:
      // the source of this slot's instance, a host cycle after the read's.
      std::pair<Value, Value> taken;
      if (llvm::any_of(sourceNow, [](auto &v) { return bool(v.first); })) {
        if (sources.size() == 1) {
          taken = sourceNow[0];
        } else {
          SmallVector<uint64_t> rows(threads, 0);
          for (unsigned i = 0; i < threads; ++i)
            rows[i] = sources[sourceOf[i]];
          unsigned w = llvm::Log2_32_Ceil(sources.size());
          Value select = copyReg(
              copyReg(table(b, ctl.inst2, rows, w),
                      consumer.name + "_select_now1_" + Twine(at)),
              consumer.name + "_select_now0_" + Twine(at));
          SmallVector<Value> hits, lives;
          for (uint64_t s = (uint64_t(1) << w); s-- > 0;) {
            auto &v = sourceNow[std::min<uint64_t>(s, sources.size() - 1)];
            hits.push_back(v.first ? v.first : constant(1, 0));
            lives.push_back(v.second ? v.second : constant(width, 0));
          }
          taken = {hw::ArrayGetOp::create(b, hw::ArrayCreateOp::create(b, hits),
                                          select),
                   hw::ArrayGetOp::create(
                       b, hw::ArrayCreateOp::create(b, lives), select)};
        }
      }
      takenAt.push_back(taken);
      Value data = sourceData[0];
      if (sources.size() > 1) {
        SmallVector<uint64_t> rows(threads, 0);
        for (unsigned i = 0; i < threads; ++i)
          rows[i] = sources[sourceOf[i]];
        unsigned w = llvm::Log2_32_Ceil(sources.size());
        // The choice, in pieces of at most `regionBits` bits, each chosen by
        // its own copy of the select.
        SmallVector<Value> chosen;
        for (unsigned low = 0; low < width; low += regionBits) {
          unsigned n = std::min<unsigned>(regionBits, width - low);
          Value select = copyReg(table(b, ctl.inst2, rows, w),
                                 consumer.name + "_select" + Twine(at) + "_" +
                                     Twine(low));
          SmallVector<Value> elements;
          for (uint64_t s = (uint64_t(1) << w); s-- > 0;)
            elements.push_back(b.createOrFold<comb::ExtractOp>(
                b.getIntegerType(n),
                sourceData[std::min<uint64_t>(s, sources.size() - 1)], low));
          chosen.push_back(hw::ArrayGetOp::create(
              b, hw::ArrayCreateOp::create(b, elements), select));
        }
        data = chosen.size() == 1
                   ? chosen[0]
                   : b.createOrFold<comb::ConcatOp>(
                         SmallVector<Value>(chosen.rbegin(), chosen.rend()));
      }
      registeredAt.push_back({unsigned(parts.size()), width});
      registered.push_back(data);
      parts.push_back({});
      at = end;
    }
    // The read register, of every bit not wired around it.
    if (!registered.empty()) {
      SmallVector<Value> high(registered.rbegin(), registered.rend());
      Value readReg = reg(b.createOrFold<comb::ConcatOp>(high),
                          consumer.name + "_read");
      unsigned low = 0;
      for (auto [r, entry] : llvm::enumerate(registeredAt)) {
        auto [index, width] = entry;
        parts[index] = b.createOrFold<comb::ExtractOp>(b.getIntegerType(width),
                                                       readReg, low);
        if (auto [hit, live] = takenAt[r]; hit)
          parts[index] = comb::MuxOp::create(b, hit, live, parts[index]);
        low += width;
      }
    }
    Value value = parts.size() == 1
                      ? parts[0]
                      : b.createOrFold<comb::ConcatOp>(
                            SmallVector<Value>(parts.rbegin(), parts.rend()));
    unsigned argIndex = 0;
    for (unsigned i = 0; i < consumer.port; ++i)
      if (int(i) != cls.clockInput)
        ++argIndex;
    copyInputs[c][argIndex].setValue(
        b.createOrFold<hw::BitcastOp>(cls.inputTypes[consumer.port], value));
  }

  // The netlist's outputs: a queue of two tokens each, written at its slot
  // from its producers' links (an early output's token of the next target
  // cycle), the first token of an early one there from reset.
  SmallVector<Value> roomTerms;
  SmallVector<Value> portOutputs;
  for (auto [k, consumer] : llvm::enumerate(consumers)) {
    if (consumer.cls >= 0)
      continue;
    SmallVector<Value> bits;
    for (auto &piece : consumer.pieces[0]) {
      if (piece.producer < 0) {
        bits.push_back(hw::ConstantOp::create(b, piece.constant));
        continue;
      }
      auto &producer = producers[piece.producer];
      if (producer.cls < 0) {
        bits.push_back(b.createOrFold<comb::ExtractOp>(
            b.getIntegerType(piece.width), frameInput[piece.producer],
            piece.low));
        continue;
      }
      unsigned pThreads = classes[producer.cls].threads();
      unsigned bankCopies = 0;
      unsigned inst = piece.inst;
      std::string bankName = consumer.name + "_parity0_";
      auto address = [&]() -> Value {
        if (pThreads <= 1)
          return Value();
        return constant(llvm::Log2_32_Ceil(pThreads), inst);
      };
      auto bank = [&]() -> Value {
        return notOf(copyReg(par1, bankName + Twine(bankCopies++)));
      };
      bits.push_back(linkRead(piece.producer, piece.low, piece.width, address,
                              bank, producer.name + "_to_" + consumer.name));
    }
    Value data = bits.empty() ? constant(0, 0)
                 : bits.size() == 1
                     ? bits[0]
                     : b.createOrFold<comb::ConcatOp>(
                           SmallVector<Value>(bits.rbegin(), bits.rend()));
    unsigned width = consumer.width;
    // A token is offered the host cycle it is pushed if the queue is empty,
    // and queued otherwise.
    Value push = eq(pos0, outputSlot[k]);
    Value readyIn = body->getArgument(outputReady[k]);
    uint64_t initCount = outputEarly[k] ? 1 : 0;
    // Overlapped, a frame is decided four host cycles before its first slot,
    // so the pushes of decided frames may still be due: the queue holds one
    // for each besides the two of a frame in turn. Below four host cycles,
    // more than one can be due.
    unsigned dueMax =
        overlapFrames ? std::max<unsigned>(1, (4 + outputSlot[k] + ii - 1) / ii - 1)
                      : 0;
    unsigned capacity = 2 + dueMax;
    unsigned qWidth = llvm::Log2_32_Ceil(capacity + dueMax + 1);
    auto countNext = backedges.get(b.getIntegerType(qWidth));
    Value count = resetReg(countNext, consumer.name + "_count", initCount);
    Value queued = comb::ICmpOp::create(b, comb::ICmpPredicate::ne, count,
                                        constant(qWidth, 0));
    Value valid = comb::OrOp::create(b, queued, push);
    Value pop = comb::AndOp::create(b, readyIn, valid);
    Value popQueued = comb::AndOp::create(b, pop, queued);
    Value pushQueued =
        comb::AndOp::create(b, push, notOf(comb::AndOp::create(b, pop, notOf(queued))));
    Value plus = comb::AddOp::create(b, count, constant(qWidth, 1));
    Value minus = comb::SubOp::create(b, count, constant(qWidth, 1));
    countNext.setValue(comb::MuxOp::create(
        b, pushQueued, comb::MuxOp::create(b, popQueued, count, plus),
        comb::MuxOp::create(b, popQueued, minus, count)));
    Value room = comb::ICmpOp::create(b, comb::ICmpPredicate::ult, count,
                                      constant(qWidth, capacity));
    if (overlapFrames) {
      // The decided frames whose token is not pushed yet: counted up as a
      // frame is decided, down at its push; none is due out of reset. A push
      // this host cycle is no longer due.
      auto dueNext = backedges.get(b.getIntegerType(qWidth));
      Value due = resetReg(dueNext, consumer.name + "_due", 0);
      Value duePlus = comb::AddOp::create(b, due, constant(qWidth, 1));
      Value dueMinus = comb::SubOp::create(b, due, constant(qWidth, 1));
      dueNext.setValue(comb::MuxOp::create(
          b, starting, comb::MuxOp::create(b, push, due, duePlus),
          comb::MuxOp::create(b, push, dueMinus, due)));
      Value dueNow = comb::MuxOp::create(b, push, dueMinus, due);
      room = comb::ICmpOp::create(b, comb::ICmpPredicate::ult,
                                  comb::AddOp::create(b, count, dueNow),
                                  constant(qWidth, capacity));
    }
    roomTerms.push_back(room);
    portOutputs.push_back(valid);
    if (width > 0) {
      // Entries from the head; popping moves them down, and a queued push
      // lands at the first free entry after the pop.
      Value countAfterPop = comb::MuxOp::create(b, popQueued, minus, count);
      SmallVector<Backedge> nexts;
      SmallVector<Value> entries;
      for (unsigned e = 0; e < capacity; ++e) {
        nexts.push_back(backedges.get(b.getIntegerType(width)));
        entries.push_back(e == 0 ? resetReg(nexts[e], consumer.name + "_head", 0)
                                 : reg(nexts[e], consumer.name + "_entry" +
                                                     Twine(e)));
      }
      for (unsigned e = 0; e < capacity; ++e) {
        Value kept = e + 1 < capacity
                         ? comb::MuxOp::create(b, popQueued, entries[e + 1],
                                               entries[e])
                         : entries[e];
        Value here = comb::AndOp::create(b, pushQueued, eq(countAfterPop, e));
        nexts[e].setValue(comb::MuxOp::create(b, here, data, kept));
      }
      portOutputs.push_back(comb::MuxOp::create(b, queued, entries[0], data));
    }
  }

  // The next frame starts when every input has its token, held or offered
  // now (taken this host cycle, so held by the frame's load), and every output
  // room for one.
  SmallVector<Value> goTerms(inputValid.begin(), inputValid.end());
  goTerms.append(roomTerms.begin(), roomTerms.end());
  go.setValue(goTerms.empty() ? constant(1, 1)
                              : comb::AndOp::create(b, goTerms, false));

  // Ports out: finishes, the inputs' readies, then the outputs' channels.
  SmallVector<Value> finishes;
  for (auto &ctl : llvm::reverse(controls))
    finishes.push_back(ctl.state);
  SmallVector<Value> results;
  results.push_back(b.createOrFold<comb::ConcatOp>(finishes));
  for (auto &[p, ready] : inputReady)
    results.push_back(ready);
  results.append(portOutputs.begin(), portOutputs.end());
  hw::OutputOp::create(b, results);

  SmallVector<Attribute> threads;
  for (auto &cls : classes)
    threads.push_back(
        b.getStringAttr(cls.name + "=" + Twine(cls.threads())));
  sim->setAttr("foldsim.threads", b.getArrayAttr(threads));
  sim->setAttr("foldsim.frame", b.getI64IntegerAttr(ii));
  return success();
}

LogicalResult StaticFolder::run() {
  inlineNetlists();
  if (failed(collect()))
    return failure();
  for (auto &cls : classes)
    if (cls.threads() > 1 &&
        failed(foldPipelines(cls.model, cls.threads(), cls.folding)))
      return failure();
  for (auto &cls : classes)
    if (failed(analyze(cls)))
      return failure();
  if (failed(schedule()))
    return failure();
  if (scheduleOnly) {
    // Nothing is built: the folded pipelines' edges stay unresolved.
    for (auto &cls : classes)
      if (cls.folding && cls.folding->edges)
        cls.folding->edges->abandon();
    return success();
  }
  for (auto &cls : classes)
    if (failed(buildCopy(cls)))
      return failure();
  if (failed(buildSim()))
    return failure();
  netlist.erase();
  for (auto *callee : inlined)
    callee->erase();
  return success();
}

void FoldStaticPass::runOnOperation() {
  auto top = getOperation();
  auto netlist = top.lookupSymbol<hw::HWModuleOp>(moduleName);
  if (!netlist) {
    top.emitError("no netlist module '") << moduleName << "'";
    return signalPassFailure();
  }
  StaticFolder folder(top, netlist, simName, regionBits, report, copies,
                      nest, scheduleOnly, oneCycleLinks, overlapFrames,
                      modulo, interleaveCopies, zeroLatencyLinks);
  if (failed(folder.run()))
    return signalPassFailure();
}
