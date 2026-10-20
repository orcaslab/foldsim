//===- Compose.cpp - A simulator of threaded models -----------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Composes the simulator of a netlist of models: the instances of each model
// threaded into one module, and every channel between them in one fabric of
// two-slot token buffers, a latency-insensitive network of the models.
//
//===----------------------------------------------------------------------===//

#include "FAME5.h"
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <map>
#include <set>
#include <numeric>

namespace foldsim {
#define GEN_PASS_DEF_COMPOSE
#include "foldsim/Passes.h.inc"
} // namespace foldsim

#define DEBUG_TYPE "foldsim-compose"

using namespace mlir;
using namespace circt;

namespace {

/// On state a relay or staged tokens add: the side of the channel it belongs
/// to, 1 the producer's and 2 the consumer's. Domains place it by this.
constexpr llvm::StringLiteral kRelaySide = "foldsim.relay_side";

Value onSide(Value value, int side) {
  value.getDefiningOp()->setAttr(
      kRelaySide, IntegerAttr::get(IntegerType::get(value.getContext(), 32), side));
  return value;
}

struct ComposePass
    : public foldsim::impl::ComposeBase<ComposePass> {
  using ComposeBase::ComposeBase;
  void runOnOperation() override;
};

/// Bits of a channel: bits of one thread of a producer, or a constant.
struct Piece {
  /// The producer, or -1 for a constant.
  int producer = -1;
  unsigned thread = 0, low = 0, width = 0;
  APInt constant;
};
using Pieces = SmallVector<Piece>;

/// A model and its instances, one thread each.
struct Class {
  /// The model, erased once threaded, and its name; `base`, the class a copy
  /// was split from, or its own name.
  hw::HWModuleOp model;
  std::string name, base;
  SmallVector<hw::InstanceOp> instances;
  SmallVector<std::string> inputNames, outputNames;
  hw::HWModuleOp threaded;
  /// The threaded module's ports by name: their input or output index.
  llvm::StringMap<unsigned> inputIndex, outputIndex;
  /// For an output wired from an input, that input.
  DenseMap<unsigned, unsigned> bypass;
  /// The producer of each output, or -1.
  SmallVector<int> producerOf;
};

/// A source of tokens: an output of a model, or an input of the netlist.
struct Producer {
  std::string name;
  /// The class, or -1 for an input of the netlist; the output or input.
  int cls = -1;
  unsigned port = 0;
  unsigned threads = 1, width = 0;
  /// Who reads it: the consumer, its thread, and the piece of that thread.
  struct Read {
    unsigned consumer, thread, piece;
  };
  SmallVector<Read> reads;
  bool memory = false, direct = false;
  /// Behind a relay stage: four slots, the token written a host cycle after
  /// it is taken, readers seeing the count a host cycle late and the producer
  /// the readers' counts a host cycle late.
  bool relay = false;
  /// With staged tokens (and no relay): the slot written a host cycle after
  /// the token is taken, from registers, and read through them until then.
  bool staged = false;
  /// Slots per thread, a power of two: two, four behind a relay, or the
  /// channel depth asked for.
  unsigned depth = 2;
  /// Built: each thread's token count, and slots in registers or a memory.
  SmallVector<Value> enq;
  SmallVector<SmallVector<Value>> slots;
  /// The memories of the bits readers read, by low bit and width: each has
  /// copies written alike, read by up to kMaxReadPorts readers.
  struct Memories {
    SmallVector<Value> copies;
    unsigned readPorts = 0;
  };
  std::map<std::tuple<unsigned, unsigned, int>, Memories> mems;
  Value writeAddress, writeEnable, writeData;
  /// Built, with slots in a memory written a host cycle late: the fire and
  /// the address as the token is taken, from which each copy stages its own.
  Value fireNow, addressNow;
  /// Behind a relay: the token as it is taken, from which a reader of some of
  /// its bits gets a staging register of its own bits.
  Value relayBits;
  std::map<std::pair<unsigned, unsigned>, Value> relaySlices;
  /// Built, behind a relay: the staged write, the whole token's bits, and
  /// with slots in registers, each reading class's copies of the slots of the
  /// bits it reads, by class, thread, low bit and width.
  Value relayFire, relayThread, relaySlot, relayStaged;
  std::map<std::tuple<int, unsigned, unsigned, unsigned>, SmallVector<Value>>
      readerSlots;
  /// Built, with a staged write: the token being written, and its thread
  /// and slot.
  Value stagedFire, stagedThread, stagedSlot, stagedBits;
};

/// A sink of tokens: an input of a model, or an output of the netlist.
struct Consumer {
  std::string name;
  /// The class, or -1 for an output of the netlist; the input or output.
  int cls = -1;
  unsigned port = 0;
  unsigned threads = 1, width = 0;
  /// Each thread's bits, the low ones first.
  SmallVector<Pieces> pieces;
  /// Whether every thread reads the same bits of the same producers; or,
  /// `uniform`, bits of the same positions and widths, each from producers
  /// that may differ by thread.
  bool regular = false, direct = false, uniform = false;
  /// Built: for each thread, for each piece, the count of tokens read.
  SmallVector<SmallVector<Value>> deq;
};

/// The options of nested folding (see `foldsim-compose`).
struct NestOptions {
  bool nest = false;
  /// Host cycles from a parent's thread to its children's; negative: the
  /// channels' hop.
  int skew = -1;
  ArrayRef<std::string> offsets, copies;
  bool bankedTokens = false;
};

class Composer {
public:
  Composer(ModuleOp top, hw::HWModuleOp netlist, StringRef simName,
           bool readySchedule, ArrayRef<std::string> relay,
           const foldsim::ThreadOptions &threadOptions, bool stagedTokens,
           unsigned channelDepth, bool domains,
           const NestOptions &nesting = {})
      : top(top), netlist(netlist), simName(simName),
        readySchedule(readySchedule), threadOptions(threadOptions),
        stagedTokens(stagedTokens), channelDepth(channelDepth),
        emitDomainTree(domains), nest(nesting.nest), nestSkew(nesting.skew),
        bankedTokens(nesting.bankedTokens) {
    for (auto &name : relay)
      relayModels.insert(name);
    for (auto &entry : nesting.offsets) {
      auto [name, count] = StringRef(entry).rsplit(':');
      int64_t cycles = 0;
      if (name.empty() || count.getAsInteger(10, cycles) || cycles < 0)
        badOptions.push_back("nest-offsets: '" + entry +
                             "' is not <model>:<host cycles>");
      else
        nestOffsets[name] = cycles;
    }
    for (auto &entry : nesting.copies) {
      auto [name, count] = StringRef(entry).rsplit(':');
      unsigned k = 0;
      if (name.empty() || count.getAsInteger(10, k) || k == 0)
        badOptions.push_back("copies: '" + entry + "' is not <model>:<count>");
      else
        copiesOf[name] = k;
    }
  }
  LogicalResult run();

private:
  void inlineNetlists();
  LogicalResult collect();
  LogicalResult decompose(Value value, unsigned low, unsigned width,
                          Pieces &pieces);
  LogicalResult build();
  Value tokenMemory(Producer &producer, unsigned low, unsigned width,
                    int reader);
  Value relaySlice(OpBuilder &b, Location loc, Producer &producer, unsigned low,
                   unsigned width);

  /// The read ports of a copy of a token memory.
  static constexpr unsigned kMaxReadPorts = 8;
  Value hostClock, hostReset;

  ModuleOp top;
  hw::HWModuleOp netlist;
  StringRef simName;
  bool readySchedule;
  foldsim::ThreadOptions threadOptions;
  bool stagedTokens;
  unsigned channelDepth;
  bool emitDomainTree;
  bool nest;
  int nestSkew;
  bool bankedTokens;
  /// Classes simulated by several copies, each a contiguous range of its
  /// threads (the high digits of the thread index).
  llvm::StringMap<unsigned> copiesOf;
  /// Offsets asked for, and each class's offset, by base name.
  llvm::StringMap<int64_t> nestOffsets, offsetOf;
  SmallVector<std::string> badOptions;
  LogicalResult splitCopies();
  /// With `nest`: each model instance's parent model instance.
  DenseMap<Operation *, Operation *> parentOf;
  /// The host cycles of a turn of each thread of a class of `threads`.
  unsigned phasesOf(unsigned threads) const {
    return threadOptions.idleSlots && threadOptions.registeredState &&
                   !readySchedule
               ? foldsim::turnPhases(threads, threadOptions.stateStages)
               : 1u;
  }
  /// Whether the channels between two models have a relay stage.
  bool relayed(StringRef p, StringRef q) const {
    return relayModels.contains(p) || relayModels.contains(q) ||
           relayModels.contains((p + ":" + q).str()) ||
           relayModels.contains((q + ":" + p).str());
  }
  /// With `nest`: whether every thread current on a host cycle is a child of
  /// the thread its parent's copy is current on, the offsets apart; after
  /// copies split classes.
  LogicalResult verifyNesting();
  /// The netlist instances each instance is inside, as the path of their
  /// names, and the instances of a netlist's own logic.
  DenseMap<Operation *, std::string> containerOf;
  DenseSet<Operation *> logicInstances;
  /// With `nest`: numbers each class's instances so that, every model picked
  /// in turn, the thread current on a host cycle is a child of the parent's
  /// thread current then.
  LogicalResult alignThreads();
  llvm::StringSet<> relayModels;
  hw::HWModuleOp sim;
  /// The netlist each model instance and each netlist instance comes from,
  /// as inlined: the class tree.
  DenseMap<Operation *, std::string> ownerOf;
  llvm::StringMap<std::string> netlistParent;
  /// The domain tree: a netlist's domain holds its model and the domains of
  /// what it instantiates; a class without a netlist is a domain of its own.
  struct Domain {
    std::string name;
    int parent = -1;
    unsigned depth = 0;
    SmallVector<int> children;
  };
  std::vector<Domain> domainTree;
  llvm::StringMap<int> domainIndex;
  SmallVector<int> classDomain;
  LogicalResult buildDomainTree();
  LogicalResult emitDomains();
  llvm::SmallPtrSet<Operation *, 4> inlined;
  std::vector<Class> classes;
  DenseMap<Operation *, std::pair<unsigned, unsigned>> threadOf;
  std::vector<Producer> producers;
  SmallVector<Consumer> consumers;
  /// The producer of each input of the netlist, or -1 for a clock.
  SmallVector<int> inputProducer;
};

} // namespace

void Composer::inlineNetlists() {
  auto *body = netlist.getBodyBlock();
  for (auto instance : body->getOps<hw::InstanceOp>()) {
    ownerOf[instance] = netlist.getModuleName().str();
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
      auto owner = ownerOf.lookup(instance);
      auto calleeName = callee.getModuleName().str();
      if (!netlistParent.count(calleeName))
        netlistParent[calleeName] = owner;
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
          containerOf[inner] =
              containerOf[instance] + "/" + instance.getInstanceName().str();
          if (inner.getInstanceName() == "logic")
            logicInstances.insert(inner);
          inner.setInstanceNameAttr(builder.getStringAttr(
              instance.getInstanceName() + "_" + inner.getInstanceName()));
          ownerOf[inner] = calleeName;
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
      instance.erase();
      changed = true;
    }
  }
}

LogicalResult Composer::collect() {
  auto *body = netlist.getBodyBlock();
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
      cls.base = cls.name;
      for (auto name : instance.getArgNames())
        cls.inputNames.push_back(cast<StringAttr>(name).getValue().str());
      for (auto name : instance.getResultNames())
        cls.outputNames.push_back(cast<StringAttr>(name).getValue().str());
    }
    auto &cls = classes[it->second];
    threadOf[instance] = {it->second, cls.instances.size()};
    cls.instances.push_back(instance);
  }
  if (!badOptions.empty())
    return netlist.emitError(badOptions.front());
  if (nest && readySchedule)
    return netlist.emitError(
        "nest: the nested order is a static order; it needs every model "
        "picked in turn (ready-schedule=false)");
  if (nest && failed(alignThreads()))
    return failure();
  if (failed(splitCopies()))
    return failure();
  if (nest && failed(verifyNesting()))
    return failure();

  // Committing by enable makes a one-thread model take every other host
  // cycle: only where a class of several threads sets the pace.
  auto options = threadOptions;
  options.commitEnable &= llvm::any_of(
      classes, [](auto &cls) { return cls.instances.size() > 1; });
  // Each class threaded, its channels found by name.
  for (auto &cls : classes) {
    auto name = StringAttr::get(top.getContext(), cls.name + "_threaded");
    auto threaded = foldsim::threadModel(cls.model, cls.instances.size(), name,
                                        readySchedule, options);
    if (failed(threaded))
      return failure();
    cls.threaded = *threaded;
    for (auto &port : cls.threaded.getPortList())
      (port.isOutput() ? cls.outputIndex : cls.inputIndex)[port.getName()] =
          port.argNum;
    if (auto bypass = cls.threaded->getAttrOfType<ArrayAttr>("foldsim.bypass")) {
      for (auto entry : bypass.getAsRange<StringAttr>()) {
        auto [output, input] = entry.getValue().split(" = ");
        auto out = llvm::find(cls.outputNames, output);
        auto in = llvm::find(cls.inputNames, input);
        if (out == cls.outputNames.end() || in == cls.inputNames.end())
          return cls.threaded.emitError("bypass '") << entry << "' unknown";
        cls.bypass[out - cls.outputNames.begin()] = in - cls.inputNames.begin();
      }
    }
  }

  // The producers: the netlist's inputs, and the models' output channels.
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
      if (!cls.outputIndex.count(name + "_source_valid")) {
        cls.producerOf.push_back(-1);
        continue;
      }
      cls.producerOf.push_back(producers.size());
      auto &producer = producers.emplace_back();
      producer.name = cls.name + "_" + name;
      producer.cls = c;
      producer.port = o;
      producer.threads = cls.instances.size();
      producer.width = hw::getBitWidth(cls.instances[0].getResult(o).getType());
    }
  }

  // The consumers: the models' input channels, and the netlist's outputs.
  for (auto [c, cls] : llvm::enumerate(classes)) {
    for (auto [i, name] : llvm::enumerate(cls.inputNames)) {
      if (!cls.inputIndex.count(name + "_sink_valid"))
        continue;
      auto &consumer = consumers.emplace_back();
      consumer.name = cls.name + "_" + name;
      consumer.cls = c;
      consumer.port = i;
      consumer.threads = cls.instances.size();
      consumer.width =
          hw::getBitWidth(cls.instances[0].getOperand(i).getType());
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

  // With banked tokens, every thread of a consumer splits its pieces at the
  // boundaries any thread's pieces have, so that a piece position is the
  // same bits for every thread, whatever producer it comes from.
  if (bankedTokens)
    for (auto &consumer : consumers) {
      if (consumer.pieces.size() < 2)
        continue;
      std::set<unsigned> cuts;
      for (auto &pieces : consumer.pieces) {
        unsigned at = 0;
        for (auto &piece : pieces)
          cuts.insert(at += piece.width);
      }
      for (auto &pieces : consumer.pieces) {
        Pieces split;
        unsigned at = 0;
        for (auto &piece : pieces) {
          unsigned end = at + piece.width;
          for (auto it = cuts.upper_bound(at); at < end; ++it) {
            unsigned next = std::min(end, *it);
            Piece part = piece;
            unsigned offset = at - (end - piece.width);
            part.width = next - at;
            if (piece.producer >= 0)
              part.low = piece.low + offset;
            else
              part.constant = piece.constant.extractBits(part.width, offset);
            split.push_back(part);
            at = next;
          }
        }
        pieces = std::move(split);
      }
    }

  // Who reads what.
  for (auto [c, consumer] : llvm::enumerate(consumers))
    for (auto [t, pieces] : llvm::enumerate(consumer.pieces))
      for (auto [j, piece] : llvm::enumerate(pieces))
        if (piece.producer >= 0)
          producers[piece.producer].reads.push_back(
              {unsigned(c), unsigned(t), unsigned(j)});

  // A consumer reads regularly if each piece is, for every thread, the same
  // bits of the same producer, or a constant.
  for (auto &consumer : consumers) {
    auto &first = consumer.pieces[0];
    consumer.regular =
        consumer.threads > 1 &&
        llvm::all_of(consumer.pieces, [&](const Pieces &pieces) {
          if (pieces.size() != first.size())
            return false;
          for (auto [piece, base] : llvm::zip(pieces, first))
            if (piece.producer != base.producer || piece.low != base.low ||
                piece.width != base.width)
              return false;
          return true;
        });
  }

  // An input of the netlist that is exactly one model input of one thread, and
  // an output exactly one model output of one thread, are wired directly.
  auto whole = [&](Consumer &consumer, Producer &producer) {
    auto &pieces = consumer.pieces[0];
    return consumer.threads == 1 && producer.threads == 1 &&
           producer.reads.size() == 1 && pieces.size() == 1 &&
           pieces[0].producer >= 0 &&
           &producers[pieces[0].producer] == &producer && pieces[0].low == 0 &&
           pieces[0].width == producer.width &&
           producer.width == consumer.width;
  };
  for (auto &consumer : consumers) {
    if (consumer.pieces[0].size() != 1 || consumer.pieces[0][0].producer < 0)
      continue;
    auto &producer = producers[consumer.pieces[0][0].producer];
    if ((consumer.cls >= 0) == (producer.cls >= 0) ||
        !whole(consumer, producer))
      continue;
    consumer.direct = producer.direct = true;
  }

  // With banked tokens, a consumer reads uniformly if each piece has, for
  // every thread, the same position and width, from a producer or, for every
  // thread, a constant.
  if (bankedTokens)
    for (auto &consumer : consumers) {
      auto &first = consumer.pieces[0];
      consumer.uniform =
          !consumer.direct &&
          llvm::all_of(consumer.pieces, [&](const Pieces &pieces) {
            if (pieces.size() != first.size())
              return false;
            for (auto [piece, base] : llvm::zip(pieces, first))
              if (piece.width != base.width)
                return false;
            return true;
          });
    }

  // A producer read only regularly, for every thread, keeps its slots in a
  // memory read at the reader's thread; with banked tokens, one read
  // uniformly too, the reader selecting among the memories of its piece's
  // producers.
  for (auto &producer : producers)
    producer.memory = producer.threads > 1 && producer.width > 0 &&
                      llvm::all_of(producer.reads, [&](auto &read) {
                        auto &consumer = consumers[read.consumer];
                        return consumer.regular || consumer.uniform;
                      });
  // A uniform piece reads memories only if every producer of it keeps one.
  for (bool changed = bankedTokens; changed;) {
    changed = false;
    for (auto &consumer : consumers) {
      if (!consumer.uniform || consumer.regular)
        continue;
      for (unsigned j = 0; j < consumer.pieces[0].size(); ++j) {
        bool all = llvm::all_of(consumer.pieces, [&](const Pieces &pieces) {
          return pieces[j].producer < 0 || producers[pieces[j].producer].memory;
        });
        if (all)
          continue;
        for (auto &pieces : consumer.pieces)
          if (pieces[j].producer >= 0 && producers[pieces[j].producer].memory) {
            producers[pieces[j].producer].memory = false;
            changed = true;
          }
      }
    }
  }

  // A relayed model, or each of a pair, is a model of the netlist: a name
  // that is not would leave the channels it means without their relay.
  for (auto &entry : relayModels)
    for (auto name : llvm::split(entry.getKey(), ':'))
      if (llvm::none_of(classes, [&](auto &cls) { return cls.base == name; })) {
        std::string known;
        for (auto &cls : classes)
          known += " " + cls.name;
        return netlist.emitError("relay: '")
               << name << "' is not a model of the netlist (models:" << known
               << ")";
      }

  // A channel between two models, one of them relayed, gets a relay stage;
  // so does one from the netlist's inputs a model named by a relay reads, alone
  // or in a pair, which would otherwise reach it from the top unregistered.
  llvm::StringSet<> relayNamed;
  for (auto &entry : relayModels)
    for (auto name : llvm::split(entry.getKey(), ':'))
      relayNamed.insert(name);
  for (auto &producer : producers) {
    if (producer.direct || producer.reads.empty() ||
        llvm::any_of(producer.reads, [&](auto &read) {
          return consumers[read.consumer].cls < 0;
        }))
      continue;
    bool mine =
        producer.cls >= 0 && relayModels.contains(classes[producer.cls].base);
    producer.relay = llvm::any_of(producer.reads, [&](auto &read) {
      int c = consumers[read.consumer].cls;
      if (c == producer.cls)
        return false;
      auto &q = classes[c].base;
      if (producer.cls < 0)
        return relayNamed.contains(q);
      auto &p = classes[producer.cls].base;
      return mine || relayModels.contains(q) ||
             relayModels.contains(p + ":" + q) ||
             relayModels.contains(q + ":" + p);
    });
  }

  // With staged tokens, every channel with slots stages its writes.
  for (auto &producer : producers) {
    producer.staged = stagedTokens && !producer.direct && !producer.relay &&
                      producer.width > 0 && !producer.reads.empty();
    producer.depth = producer.relay ? 4 : channelDepth;
  }
  return success();
}

LogicalResult Composer::alignThreads() {
  // Each model instance's parent: the logic of the netlist it is in, or, for
  // a netlist's logic, that of the netlist that netlist is in.
  llvm::StringMap<Operation *> logicAt;
  for (auto &cls : classes)
    for (auto instance : cls.instances)
      if (logicInstances.contains(instance))
        logicAt[containerOf[instance]] = instance;
  auto up = [](StringRef path) {
    auto pos = path.rfind('/');
    return pos == StringRef::npos ? StringRef() : path.substr(0, pos);
  };
  DenseMap<Operation *, unsigned> classOfInstance;
  for (auto [c, cls] : llvm::enumerate(classes))
    for (auto instance : cls.instances) {
      classOfInstance[instance] = c;
      StringRef path = containerOf[instance];
      if (logicInstances.contains(instance)) {
        if (path.empty())
          continue;
        path = up(path);
      }
      if (auto *parent = logicAt.lookup(path); parent && parent != instance)
        parentOf[instance] = parent;
    }
  // Each class's parent class, the same for all its instances, and its depth.
  SmallVector<int> parentClass(classes.size(), -1);
  for (auto [c, cls] : llvm::enumerate(classes))
    for (auto instance : cls.instances) {
      auto *parent = parentOf.lookup(instance);
      int pc = parent ? int(classOfInstance[parent]) : -1;
      if (instance != cls.instances.front() && pc != parentClass[c])
        return cls.instances[0].emitError("nest: the instances of '")
               << cls.name << "' have parents of different classes";
      parentClass[c] = pc;
    }
  SmallVector<unsigned> depth(classes.size(), 0), order;
  for (unsigned c = 0; c < classes.size(); ++c) {
    unsigned d = 0;
    for (int p = parentClass[c]; p >= 0; p = parentClass[p])
      if (++d > classes.size())
        return netlist.emitError("nest: classes nested in a cycle");
    depth[c] = d;
    order.push_back(c);
  }
  llvm::stable_sort(order, [&](unsigned x, unsigned y) { return depth[x] < depth[y]; });
  // A class's turn: `phases` host cycles per thread, the first its own.
  DenseMap<Operation *, unsigned> threadNow;
  for (auto c : order) {
    auto &cls = classes[c];
    unsigned n = cls.instances.size();
    int pc = parentClass[c];
    unsigned np = pc >= 0 ? classes[pc].instances.size() : 1;
    auto keep = [&]() {
      for (auto [t, instance] : llvm::enumerate(cls.instances))
        threadNow[instance] = t;
    };
    if (n == 1 || np == 1) {
      keep();
      continue;
    }
    unsigned kc = phasesOf(n), kp = phasesOf(np);
    uint64_t period = std::lcm(uint64_t(n) * kc, uint64_t(np) * kp);
    // The parent's thread each thread must be the child of.
    SmallVector<int> need(n, -1);
    for (uint64_t h = period; h < 2 * period; ++h) {
      if (h % kc || h % kp)
        continue;
      unsigned t = (h / kc) % n, p = (h / kp) % np;
      if (need[t] >= 0 && need[t] != int(p))
        return cls.instances[0].emitError("nest: '")
               << cls.name << "' (" << n << " threads, " << kc
               << " phases) cannot follow its parent's turns (" << np
               << " threads, " << kp << " phases)";
      need[t] = p;
    }
    std::vector<SmallVector<hw::InstanceOp>> byParent(np);
    for (auto instance : cls.instances)
      byParent[threadNow.lookup(parentOf.lookup(instance))].push_back(instance);
    SmallVector<unsigned> next(np, 0);
    SmallVector<hw::InstanceOp> numbered(n);
    for (unsigned t = 0; t < n; ++t) {
      if (need[t] < 0)
        continue;
      auto &kids = byParent[need[t]];
      if (next[need[t]] >= kids.size())
        return cls.instances[0].emitError("nest: '")
               << cls.name << "' has too few instances per parent to follow "
               << "its parent's turns";
      numbered[t] = kids[next[need[t]]++];
    }
    unsigned p = 0;
    for (unsigned t = 0; t < n; ++t) {
      if (numbered[t])
        continue;
      while (next[p] >= byParent[p].size())
        p = (p + 1) % np;
      numbered[t] = byParent[p][next[p]++];
    }
    cls.instances.assign(numbered.begin(), numbered.end());
    for (auto [t, instance] : llvm::enumerate(cls.instances)) {
      threadOf[instance] = {c, unsigned(t)};
      threadNow[instance] = t;
    }
  }
  // Each class current `offset` host cycles after the order above, so the
  // parent of the thread current at h is current at h minus the difference:
  // the offset asked for, or the parent's plus a hop of the channels between
  // them (a host cycle, two behind a relay) rounded up to the class's turn,
  // so that what a parent sends on a host cycle is there when the children
  // are current.
  for (auto name : nestOffsets.keys())
    if (llvm::none_of(classes, [&](auto &cls) { return cls.name == name; }))
      return netlist.emitError("nest-offsets: no model '") << name << "'";
  SmallVector<int64_t> offset(classes.size(), 0);
  for (auto c : order) {
    auto &cls = classes[c];
    int pc = parentClass[c];
    unsigned n = cls.instances.size(), k = phasesOf(n);
    auto it = nestOffsets.find(cls.name);
    if (it != nestOffsets.end()) {
      offset[c] = it->second;
    } else if (pc >= 0) {
      int64_t hop = nestSkew >= 0 ? nestSkew
                    : relayed(classes[pc].name, cls.name) ? 2 : 1;
      offset[c] = llvm::alignTo(offset[pc] + hop, k);
    }
    offsetOf[cls.name] = offset[c];
    if (n == 1)
      continue;
    if (offset[c] % k)
      return cls.instances[0].emitError("nest: the offset of '")
             << cls.name << "' (" << offset[c]
             << ") is not a multiple of its turn's " << k << " host cycles";
    int64_t shift = (offset[c] / k) % n;
    SmallVector<hw::InstanceOp> canonical(cls.instances.begin(),
                                          cls.instances.end());
    for (unsigned t = 0; t < n; ++t) {
      auto instance = canonical[((int64_t(t) - shift) % n + n) % n];
      cls.instances[t] = instance;
      threadOf[instance] = {c, t};
    }
  }
  return success();
}

LogicalResult Composer::splitCopies() {
  for (auto name : copiesOf.keys())
    if (llvm::none_of(classes, [&](auto &cls) { return cls.name == name; }))
      return netlist.emitError("copies: no model '") << name << "'";
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
    classes[c].instances.assign(instances.begin(), instances.begin() + per);
    for (unsigned j = 1; j < k; ++j) {
      // Each copy threads a model of its own.
      auto model = classes[c].model;
      OpBuilder b(model);
      b.setInsertionPointAfter(model);
      auto clone = cast<hw::HWModuleOp>(b.clone(*model));
      std::string name = classes[c].name + "_copy" + std::to_string(j);
      clone.setSymName(name);
      Class copy;
      copy.model = clone;
      copy.name = name;
      copy.base = classes[c].base;
      copy.inputNames = classes[c].inputNames;
      copy.outputNames = classes[c].outputNames;
      unsigned index = classes.size();
      for (unsigned t = 0; t < per; ++t) {
        auto instance = instances[j * per + t];
        copy.instances.push_back(instance);
        threadOf[instance] = {index, t};
      }
      classes.push_back(std::move(copy));
    }
  }
  return success();
}

LogicalResult Composer::verifyNesting() {
  // On every host cycle where a child's copy and its parent's are current,
  // the child's thread is a child of the parent's, their offsets apart.
  auto turn = [&](const Class &cls) {
    return uint64_t(cls.instances.size()) * phasesOf(cls.instances.size());
  };
  uint64_t period = 1;
  for (auto &cls : classes)
    period = std::lcm(period, turn(cls));
  for (auto &cls : classes) {
    unsigned k = phasesOf(cls.instances.size());
    for (uint64_t h = 0; h < period; h += k) {
      auto instance = cls.instances[(h / k) % cls.instances.size()];
      auto *parentInstance = parentOf.lookup(instance);
      if (!parentInstance)
        continue;
      auto [pc, pt] = threadOf.lookup(parentInstance);
      auto &parent = classes[pc];
      unsigned kp = phasesOf(parent.instances.size());
      int64_t delta = offsetOf.lookup(cls.base) - offsetOf.lookup(parent.base);
      int64_t at = ((int64_t(h) - delta) % int64_t(period) + int64_t(period)) %
                   int64_t(period);
      if (at % kp)
        continue;
      if ((at / kp) % parent.instances.size() != pt)
        return cls.instances[0].emitError("nest: the threads of '")
               << cls.name << "' do not follow the turns of '" << parent.name
               << "': copies must split a class at a multiple of its "
                  "parent's turn";
    }
  }
  return success();
}

LogicalResult Composer::decompose(Value value, unsigned low, unsigned width,
                                  Pieces &pieces) {
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
          piece.thread == last.thread && last.low + last.width == piece.low) {
        last.width += piece.width;
        return;
      }
    }
    pieces.push_back(piece);
  };
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    int producer = inputProducer[arg.getArgNumber()];
    if (producer < 0)
      return netlist.emitError("a clock is not a channel");
    append(Piece{producer, 0, low, width, APInt()});
    return success();
  }
  auto *op = value.getDefiningOp();
  if (auto instance = dyn_cast<hw::InstanceOp>(op)) {
    auto [c, thread] = threadOf.lookup(instance);
    auto &cls = classes[c];
    auto output = cast<OpResult>(value).getResultNumber();
    if (auto it = cls.bypass.find(output); it != cls.bypass.end())
      return decompose(instance.getOperand(it->second), low, width, pieces);
    int producer = cls.producerOf[output];
    if (producer < 0)
      return instance.emitError("output '")
             << cls.outputNames[output] << "' is read but has no channel";
    append(Piece{producer, thread, low, width, APInt()});
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

/// A memory of `width` bits of `producer`'s tokens from `low`, with a read
/// port to spare. Readers of different bits read memories of their own bits:
/// distributed RAM has a copy per read port, and a port of the whole token
/// would keep every bit in it. FPGA tools map a memory of up to 16 ports to
/// distributed RAM, so a range with many readers has several copies.
Value Composer::tokenMemory(Producer &producer, unsigned low, unsigned width,
                            int reader) {
  // Behind a relay, each reading class has copies of its own, next to it.
  auto &memories = producer.mems[{low, width, producer.relay ? reader : -1}];
  if (memories.readPorts++ % kMaxReadPorts == 0) {
    OpBuilder b(producer.writeAddress.getContext());
    b.setInsertionPointAfterValue(producer.writeData);
    auto loc = producer.writeAddress.getLoc();
    unsigned threadBits = llvm::Log2_32_Ceil(producer.threads);
    unsigned slots = producer.depth;
    auto type = seq::FirMemType::get(b.getContext(), slots << threadBits,
                                     width, std::nullopt);
    std::string name = producer.name + (producer.relay ? "_rtokens" : "_tokens");
    if (width != producer.width)
      name += "_" + std::to_string(low + width - 1) + "_" + std::to_string(low);
    if (producer.relay)
      name += "_" + classes[reader].name;
    if (!memories.copies.empty())
      name += "_" + std::to_string(memories.copies.size());
    Value mem = seq::FirMemOp::create(
        b, loc, type, /*readLatency=*/0, /*writeLatency=*/1,
        seq::RUW::Undefined, seq::WUW::Undefined, b.getStringAttr(name),
        hw::InnerSymAttr{}, seq::FirMemInitAttr{}, StringAttr{}, StringAttr{});
    if (producer.relay)
      onSide(mem, 2);
    Value data = producer.writeData;
    if (producer.relay) {
      data = relaySlice(b, loc, producer, low, width);
    } else if (width != producer.width) {
      data = b.createOrFold<comb::ExtractOp>(loc, b.getIntegerType(width), data,
                                             low);
    }
    Value address = producer.writeAddress, enable = producer.writeEnable;
    if (producer.fireNow) {
      // Each copy stages its own write, kept: one staged write would drive
      // every copy's write port, and the equal staged threads of every
      // channel of a class merge into one register.
      auto keep = sv::SVAttributeAttr::get(b.getContext(), "keep", R"("true")",
                                           /*emitAsComment=*/false);
      auto stage = [&](Value value, const Twine &stageName) -> Value {
        auto type = value.getType();
        auto reg = seq::FirRegOp::create(
            b, loc, value, hostClock, b.getStringAttr(stageName), hostReset,
            hw::ConstantOp::create(b, loc, type, 0));
        sv::addSVAttributes(reg, {keep});
        return onSide(reg, 1);
      };
      enable = stage(producer.fireNow, name + "_we");
      address = stage(producer.addressNow, name + "_wa");
    }
    seq::FirMemWriteOp::create(b, loc, mem, address, hostClock, enable, data,
                               Value{});
    memories.copies.push_back(mem);
  }
  return memories.copies.back();
}

/// Behind a relay, the staged bits a reader reads: the whole token's, or the
/// reader's bits staged on the producer's side on their own, so the relay
/// carries them, not the whole token.
Value Composer::relaySlice(OpBuilder &b, Location loc, Producer &producer,
                           unsigned low, unsigned width) {
  if (width == producer.width)
    return producer.relayStaged;
  auto &staged = producer.relaySlices[{low, width}];
  if (!staged)
    staged = onSide(
        seq::FirRegOp::create(
            b, loc,
            b.createOrFold<comb::ExtractOp>(loc, b.getIntegerType(width),
                                            producer.relayBits, low),
            hostClock,
            b.getStringAttr(producer.name + "_stg_bits_" +
                            std::to_string(low + width - 1) + "_" +
                            std::to_string(low))),
        1);
  return staged;
}

LogicalResult Composer::build() {
  auto *context = top.getContext();
  auto i1 = IntegerType::get(context, 1);
  LLVM_DEBUG({
    for (auto &consumer : consumers)
      for (auto [t, pieces] : llvm::enumerate(consumer.pieces)) {
        llvm::dbgs() << "foldsim-net " << consumer.name << " " << t << " <-";
        for (auto &piece : pieces)
          if (piece.producer >= 0)
            llvm::dbgs() << " " << producers[piece.producer].name << "@"
                         << piece.thread;
        llvm::dbgs() << "\n";
      }
  });

  // The simulator's ports.
  SmallVector<hw::PortInfo> ports;
  size_t numIn = 0, numOut = 0;
  auto addPort = [&](const Twine &name, Type type, bool isInput) {
    hw::PortInfo port;
    port.name = StringAttr::get(context, name);
    port.type = type;
    port.dir = isInput ? hw::ModulePort::Direction::Input
                       : hw::ModulePort::Direction::Output;
    port.argNum = isInput ? numIn++ : numOut++;
    ports.push_back(port);
    return port.argNum;
  };
  addPort("hostClock", seq::ClockType::get(context), true);
  addPort("hostReset", i1, true);
  struct TopPort {
    unsigned valid, ready, bits;
  };
  SmallVector<TopPort> inputPorts(producers.size()), outputPorts;
  for (auto [p, producer] : llvm::enumerate(producers)) {
    if (producer.cls >= 0)
      continue;
    auto &port = inputPorts[p];
    port.valid = addPort(producer.name + "_valid", i1, true);
    if (producer.width > 0)
      port.bits = addPort(producer.name + "_bits",
                          IntegerType::get(context, producer.width), true);
    port.ready = addPort(producer.name + "_ready", i1, false);
  }
  outputPorts.resize(consumers.size());
  for (auto [c, consumer] : llvm::enumerate(consumers)) {
    if (consumer.cls >= 0)
      continue;
    auto &port = outputPorts[c];
    port.ready = addPort(consumer.name + "_ready", i1, true);
    port.valid = addPort(consumer.name + "_valid", i1, false);
    if (consumer.width > 0)
      port.bits = addPort(consumer.name + "_bits",
                          IntegerType::get(context, consumer.width), false);
  }
  // Bit c: a thread of class c finishes its target cycle.
  auto profilePort = addPort("profile_finish",
                             IntegerType::get(context, classes.size()), false);

  OpBuilder outer(netlist);
  sim = hw::HWModuleOp::create(outer, netlist.getLoc(),
                                    StringAttr::get(context, simName),
                                    hw::ModulePortInfo(ports));
  auto *body = sim.getBodyBlock();
  body->getTerminator()->erase();
  auto b = ImplicitLocOpBuilder::atBlockEnd(netlist.getLoc(), body);
  BackedgeBuilder backedges(b, netlist.getLoc());
  hostClock = body->getArgument(0);
  hostReset = body->getArgument(1);
  SmallVector<Value> outputs(numOut);

  auto constant = [&](const APInt &value) -> Value {
    return hw::ConstantOp::create(b, value);
  };
  auto bit = [&](bool value) { return constant(APInt(1, value)); };
  auto counter = [&](const Twine &name, unsigned width = 2) -> Value {
    auto type = b.getIntegerType(width);
    auto next = backedges.get(type);
    auto reg = seq::FirRegOp::create(b, next, hostClock, b.getStringAttr(name),
                                     hostReset, constant(APInt(width, 0)));
    // The next value is set later, through the register's operand.
    return reg;
  };
  auto setNext = [&](Value reg, Value next) {
    reg.getDefiningOp<seq::FirRegOp>().getNextMutable().assign(next);
  };
  // `values[index]`: one per thread, selected by the thread index.
  auto select = [&](ArrayRef<Value> values, Value index) -> Value {
    if (values.size() == 1)
      return values[0];
    SmallVector<Value> elements(values.rbegin(), values.rend());
    return hw::ArrayGetOp::create(b, hw::ArrayCreateOp::create(b, elements),
                                  index);
  };
  auto isThread = [&](Value index, unsigned thread) -> Value {
    if (!index)
      return bit(true);
    return comb::ICmpOp::create(
        b, comb::ICmpPredicate::eq, index,
        constant(APInt(index.getType().getIntOrFloatBitWidth(), thread)));
  };
  // A copy of `value` a host cycle late.
  auto delayed = [&](Value value, const Twine &name) -> Value {
    unsigned width = value.getType().getIntOrFloatBitWidth();
    return seq::FirRegOp::create(b, value, hostClock, b.getStringAttr(name),
                                 hostReset, constant(APInt(width, 0)));
  };
  auto increment = [&](Value count, Value enable) -> Value {
    auto plusOne = comb::AddOp::create(
        b, count,
        constant(APInt(count.getType().getIntOrFloatBitWidth(), 1)));
    return comb::MuxOp::create(b, enable, plusOne, count);
  };
  auto andAll = [&](ArrayRef<Value> values) -> Value {
    if (values.empty())
      return bit(true);
    return b.createOrFold<comb::AndOp>(values, false);
  };

  // The threaded models, their inputs connected below.
  struct Model {
    hw::InstanceOp instance;
    llvm::StringMap<Backedge> inputs;
    Value thread;
  };
  SmallVector<Model> models(classes.size());
  for (auto [c, cls] : llvm::enumerate(classes)) {
    auto &model = models[c];
    SmallVector<Value> operands = {hostClock, hostReset};
    for (auto &port : cls.threaded.getPortList()) {
      if (port.isOutput() || port.argNum < 2)
        continue;
      auto edge = backedges.get(port.type);
      model.inputs.try_emplace(port.getName(), edge);
      operands.push_back(edge);
    }
    model.instance = hw::InstanceOp::create(
        b, cls.threaded, b.getStringAttr(cls.name + "_i"), operands);
    if (auto it = cls.outputIndex.find("threadIdx");
        it != cls.outputIndex.end())
      model.thread = model.instance.getResult(it->second);
  }
  auto modelOutput = [&](unsigned c, const Twine &name) -> Value {
    auto it = classes[c].outputIndex.find(name.str());
    return it == classes[c].outputIndex.end()
               ? Value()
               : models[c].instance.getResult(it->second);
  };
  auto modelInput = [&](unsigned c, const Twine &name) -> Backedge * {
    auto it = models[c].inputs.find(name.str());
    return it == models[c].inputs.end() ? nullptr : &it->second;
  };

  // The producers' and consumers' handshakes.
  struct Source {
    Value valid, bits, thread;
    std::function<void(Value)> setReady;
  };
  auto sourceOf = [&](unsigned p) {
    auto &producer = producers[p];
    Source source;
    if (producer.cls < 0) {
      auto &port = inputPorts[p];
      source.valid = body->getArgument(port.valid);
      if (producer.width > 0)
        source.bits = body->getArgument(port.bits);
      source.setReady = [&, ready = port.ready](Value value) {
        outputs[ready] = value;
      };
      return source;
    }
    auto &cls = classes[producer.cls];
    auto name = cls.outputNames[producer.port];
    source.valid = modelOutput(producer.cls, name + "_source_valid");
    source.bits = modelOutput(producer.cls, name + "_source_bits");
    source.thread = models[producer.cls].thread;
    auto *ready = modelInput(producer.cls, name + "_source_ready");
    source.setReady = [ready](Value value) { ready->setValue(value); };
    return source;
  };
  struct Sink {
    Value ready, thread;
    std::function<void(Value, Value)> set;
  };
  auto sinkOf = [&](unsigned c) {
    auto &consumer = consumers[c];
    Sink sink;
    if (consumer.cls < 0) {
      auto &port = outputPorts[c];
      sink.ready = body->getArgument(port.ready);
      sink.set = [&, port](Value valid, Value bits) {
        outputs[port.valid] = valid;
        if (bits)
          outputs[port.bits] = bits;
      };
      return sink;
    }
    auto &cls = classes[consumer.cls];
    auto name = cls.inputNames[consumer.port];
    sink.ready = modelOutput(consumer.cls, name + "_sink_ready");
    sink.thread = models[consumer.cls].thread;
    auto *valid = modelInput(consumer.cls, name + "_sink_valid");
    auto *bits = modelInput(consumer.cls, name + "_sink_bits");
    sink.set = [valid, bits](Value v, Value d) {
      valid->setValue(v);
      if (bits)
        bits->setValue(d);
    };
    return sink;
  };

  // Direct connections.
  for (auto [c, consumer] : llvm::enumerate(consumers)) {
    if (!consumer.direct)
      continue;
    auto p = consumer.pieces[0][0].producer;
    auto source = sourceOf(p);
    auto sink = sinkOf(c);
    sink.set(source.valid, source.bits);
    source.setReady(sink.ready);
  }

  // Each producer's token counts and slots, and its readers' counts.
  for (auto [c, consumer] : llvm::enumerate(consumers)) {
    if (consumer.direct)
      continue;
    for (auto [t, pieces] : llvm::enumerate(consumer.pieces)) {
      auto &deq = consumer.deq.emplace_back();
      for (auto [j, piece] : llvm::enumerate(pieces)) {
        if (piece.producer < 0) {
          deq.push_back(Value());
          continue;
        }
        auto &producer = producers[piece.producer];
        Value count = counter(consumer.name +
                                  (producer.relay ? "_rdeq" : "_deq") +
                                  Twine(t) + "_" + Twine(j),
                              llvm::Log2_32(producer.depth) + 1);
        deq.push_back(producer.relay ? onSide(count, 2) : count);
      }
    }
  }
  // `numSlots` slot registers `<name>_<s>`, slot `slot` written with `data`
  // on `write`.
  auto slotRegisters = [&](const Twine &name, unsigned numSlots, Value write,
                           Value slot, Value data) {
    SmallVector<Value> slots;
    unsigned slotBits = llvm::Log2_32(numSlots);
    for (unsigned s = 0; s < numSlots; ++s) {
      auto next = backedges.get(data.getType());
      auto reg = seq::FirRegOp::create(b, next, hostClock,
                                       b.getStringAttr(name + "_" + Twine(s)));
      Value at =
          numSlots > 2
              ? Value(comb::ICmpOp::create(b, comb::ICmpPredicate::eq, slot,
                                           constant(APInt(slotBits, s))))
          : s ? slot
              : Value(comb::XorOp::create(b, slot, bit(true)));
      Value here = comb::AndOp::create(b, write, at);
      next.setValue(comb::MuxOp::create(b, here, data, reg));
      slots.push_back(reg);
    }
    return slots;
  };
  for (auto [p, producer] : llvm::enumerate(producers)) {
    if (producer.direct)
      continue;
    auto source = sourceOf(p);
    unsigned numSlots = producer.depth;
    unsigned countBits = llvm::Log2_32(numSlots) + 1;
    for (unsigned k = 0; k < producer.threads; ++k) {
      Value count =
          counter(producer.name + (producer.relay ? "_renq" : "_enq") + Twine(k),
                  countBits);
      producer.enq.push_back(producer.relay ? onSide(count, 1) : count);
    }
    // Ready while every reader of the thread has read all but one token; with
    // a relay, while the reader's count of a host cycle ago leaves a slot.
    SmallVector<SmallVector<Value>> notFull(producer.threads);
    for (auto &read : producer.reads) {
      auto &consumer = consumers[read.consumer];
      auto &piece = consumer.pieces[read.thread][read.piece];
      Value deq = consumer.deq[read.thread][read.piece];
      if (producer.relay)
        deq = onSide(delayed(deq, consumer.name + "_deqd" + Twine(read.thread) +
                                      "_" + Twine(read.piece)),
                     1);
      Value count = comb::SubOp::create(b, producer.enq[piece.thread], deq);
      notFull[piece.thread].push_back(
          comb::ICmpOp::create(b, comb::ICmpPredicate::ne, count,
                               constant(APInt(countBits, numSlots))));
    }
    SmallVector<Value> readyOf;
    for (auto &terms : notFull)
      readyOf.push_back(andAll(terms));
    Value ready = select(readyOf, source.thread);
    source.setReady(ready);
    Value fire = comb::AndOp::create(b, source.valid, ready);
    for (unsigned k = 0; k < producer.threads; ++k)
      setNext(
          producer.enq[k],
          increment(producer.enq[k],
                    comb::AndOp::create(b, fire, isThread(source.thread, k))));
    if (producer.width == 0)
      continue;
    unsigned slotBits = llvm::Log2_32(numSlots);
    auto slotOf = [&](Value count) {
      return comb::ExtractOp::create(b, b.getIntegerType(slotBits), count, 0);
    };
    // With a relay or staged tokens, the token, its thread and slot are
    // staged a host cycle: the slot's write enable is a register rather than
    // the fire of this cycle.
    Value writeFire = fire, writeThread = source.thread,
          writeBits = source.bits, writeSlot;
    if (producer.memory || producer.relay || producer.staged)
      writeSlot = slotOf(select(producer.enq, source.thread));
    if (producer.memory && (producer.relay || producer.staged)) {
      producer.fireNow = fire;
      producer.addressNow = comb::ConcatOp::create(b, source.thread, writeSlot);
    }
    if (producer.relay || producer.staged) {
      writeFire = onSide(delayed(fire, producer.name + "_stg_fire"), 1);
      if (source.thread)
        writeThread =
            onSide(delayed(source.thread, producer.name + "_stg_thread"), 1);
      writeSlot = onSide(delayed(writeSlot, producer.name + "_stg_slot"), 1);
      writeBits = onSide(seq::FirRegOp::create(
                             b, source.bits, hostClock,
                             b.getStringAttr(producer.name + "_stg_bits")),
                         1);
      if (producer.relay) {
        producer.relayBits = source.bits;
        producer.relayStaged = writeBits;
        producer.relayFire = writeFire;
        producer.relayThread = source.thread ? writeThread : Value();
        producer.relaySlot = writeSlot;
      }
    }
    if (producer.staged) {
      producer.stagedFire = writeFire;
      producer.stagedThread = writeThread;
      producer.stagedSlot = writeSlot;
      producer.stagedBits = writeBits;
    }
    if (producer.memory) {
      producer.writeAddress = comb::ConcatOp::create(b, writeThread, writeSlot);
      producer.writeEnable = writeFire;
      producer.writeData = writeBits;
      continue;
    }
    // Behind a relay, the slots are each reading class's own, built as read.
    if (producer.relay)
      continue;
    for (unsigned k = 0; k < producer.threads; ++k) {
      Value write = comb::AndOp::create(b, writeFire, isThread(writeThread, k));
      Value slot = producer.staged ? writeSlot : slotOf(producer.enq[k]);
      producer.slots.push_back(slotRegisters(producer.name + "_slot" + Twine(k),
                                             numSlots, write, slot, writeBits));
    }
  }

  // Behind a relay, a reading class's slots of thread `k`, of the bits from
  // `low`: written from the staged token, on the reader's side.
  auto readerSlots = [&](Producer &producer, int reader, unsigned k,
                         unsigned low, unsigned width) -> ArrayRef<Value> {
    auto &slots = producer.readerSlots[{reader, k, low, width}];
    if (slots.empty()) {
      Value data = relaySlice(b, b.getLoc(), producer, low, width);
      Value write = comb::AndOp::create(b, producer.relayFire,
                                        isThread(producer.relayThread, k));
      std::string name = producer.name + "_rslot" + std::to_string(k);
      if (width != producer.width)
        name += "_" + std::to_string(low + width - 1) + "_" + std::to_string(low);
      name += "_" + classes[reader].name;
      slots = slotRegisters(name, producer.depth, write, producer.relaySlot,
                            data);
      for (auto reg : slots)
        onSide(reg, 2);
    }
    return slots;
  };

  // With a relay, each reader sees a producer thread's count a host cycle
  // late, from its own copy.
  std::map<std::tuple<unsigned, unsigned, unsigned>, Value> enqCopies;
  auto enqSeen = [&](unsigned c, unsigned p, unsigned k) -> Value {
    auto &producer = producers[p];
    if (!producer.relay)
      return producer.enq[k];
    auto &copy = enqCopies[{c, p, k}];
    if (!copy)
      copy = onSide(delayed(producer.enq[k], consumers[c].name + "_enqd_" +
                                                 producer.name + "_" + Twine(k)),
                    2);
    return copy;
  };

  // A staged token is in its slot a host cycle after it is taken: a reader
  // of that slot until then reads it from the staging registers.
  auto forwarded = [&](Producer &producer, Value thread, Value slot,
                       unsigned low, unsigned width, Value data) -> Value {
    if (!producer.staged)
      return data;
    Value hit = comb::AndOp::create(
        b, producer.stagedFire,
        comb::ICmpOp::create(b, comb::ICmpPredicate::eq, producer.stagedSlot,
                             slot));
    if (producer.stagedThread)
      hit = comb::AndOp::create(
          b, hit,
          comb::ICmpOp::create(b, comb::ICmpPredicate::eq,
                               producer.stagedThread, thread));
    Value bits = b.createOrFold<comb::ExtractOp>(b.getIntegerType(width),
                                                 producer.stagedBits, low);
    return comb::MuxOp::create(b, hit, bits, data);
  };

  // For each class, each thread: whether every input has a token for it.
  std::vector<std::vector<SmallVector<Value>>> readyTerms(classes.size());
  for (auto [c, cls] : llvm::enumerate(classes))
    readyTerms[c].resize(cls.instances.size());

  // Each consumer reads its thread's pieces.
  for (unsigned c = 0; c < consumers.size(); ++c) {
    auto &consumer = consumers[c];
    if (consumer.direct)
      continue;
    auto sink = sinkOf(c);
    auto bitsOf = [&](unsigned t, unsigned j) -> Value {
      auto &piece = consumer.pieces[t][j];
      if (piece.producer < 0)
        return constant(piece.constant);
      auto &producer = producers[piece.producer];
      // A relay's slots are the reader's own, of the bits it reads.
      unsigned low = producer.relay ? 0 : piece.low;
      ArrayRef<Value> slots =
          producer.relay ? readerSlots(producer, consumer.cls, piece.thread,
                                       piece.low, piece.width)
                         : ArrayRef<Value>(producer.slots[piece.thread]);
      Value data;
      if (producer.depth > 2) {
        Value slot = comb::ExtractOp::create(
            b, b.getIntegerType(llvm::Log2_32(producer.depth)),
            consumer.deq[t][j], 0);
        data = select(slots, slot);
        if (producer.staged) {
          Value thread;
          if (producer.stagedThread)
            thread = constant(
                APInt(producer.stagedThread.getType().getIntOrFloatBitWidth(),
                      piece.thread));
          return forwarded(producer, thread, slot, piece.low, piece.width,
                           b.createOrFold<comb::ExtractOp>(
                               b.getIntegerType(piece.width), data, piece.low));
        }
      } else {
        Value slot = comb::ExtractOp::create(b, i1, consumer.deq[t][j], 0);
        data = comb::MuxOp::create(b, slot, slots[1], slots[0]);
        if (producer.staged) {
          Value thread;
          if (producer.stagedThread)
            thread = constant(
                APInt(producer.stagedThread.getType().getIntOrFloatBitWidth(),
                      piece.thread));
          return forwarded(producer, thread, slot, piece.low, piece.width,
                           b.createOrFold<comb::ExtractOp>(
                               b.getIntegerType(piece.width), data, piece.low));
        }
      }
      return b.createOrFold<comb::ExtractOp>(b.getIntegerType(piece.width),
                                             data, low);
    };
    auto concat = [&](SmallVector<Value> parts) -> Value {
      if (parts.empty())
        return {};
      std::reverse(parts.begin(), parts.end());
      return b.createOrFold<comb::ConcatOp>(parts);
    };

    SmallVector<Value> validOf;
    for (auto [t, pieces] : llvm::enumerate(consumer.pieces)) {
      SmallVector<Value> terms;
      for (auto [j, piece] : llvm::enumerate(pieces))
        if (piece.producer >= 0)
          terms.push_back(comb::ICmpOp::create(
              b, comb::ICmpPredicate::ne,
              enqSeen(c, piece.producer, piece.thread), consumer.deq[t][j]));
      validOf.push_back(andAll(terms));
    }
    Value valid = select(validOf, sink.thread);
    if (consumer.cls >= 0)
      for (auto [t, v] : llvm::enumerate(validOf))
        readyTerms[consumer.cls][t].push_back(v);

    Value bits;
    if (consumer.width > 0 && consumer.uniform && !consumer.regular) {
      // Piece by piece: a uniform piece of memory producers reads each of
      // them at the thread and slot the current thread reads it at, and
      // takes the one the current thread reads.
      SmallVector<Value> parts;
      for (auto [j, first] : llvm::enumerate(consumer.pieces[0])) {
        bool fromMemories =
            llvm::any_of(consumer.pieces,
                         [&](const Pieces &pieces) {
                           return pieces[j].producer >= 0;
                         }) &&
            llvm::all_of(consumer.pieces, [&](const Pieces &pieces) {
              return pieces[j].producer < 0 ||
                     producers[pieces[j].producer].memory;
            });
        if (!fromMemories) {
          SmallVector<Value> perThread;
          for (unsigned t = 0; t < consumer.threads; ++t)
            perThread.push_back(bitsOf(t, j));
          parts.push_back(select(perThread, sink.thread));
          continue;
        }
        // A source: a producer's bits from a low bit, or a constant.
        SmallVector<const Piece *> sources;
        SmallVector<unsigned> sourceOf;
        auto same = [](const Piece *x, const Piece &y) {
          if (x->producer != y.producer)
            return false;
          return x->producer >= 0 ? x->low == y.low
                                  : x->constant == y.constant;
        };
        for (auto &pieces : consumer.pieces) {
          auto *it = llvm::find_if(
              sources, [&](const Piece *x) { return same(x, pieces[j]); });
          sourceOf.push_back(it - sources.begin());
          if (it == sources.end())
            sources.push_back(&pieces[j]);
        }
        SmallVector<Value> values;
        for (auto [k, source] : llvm::enumerate(sources)) {
          if (source->producer < 0) {
            values.push_back(constant(source->constant));
            continue;
          }
          auto &producer = producers[source->producer];
          unsigned threadBits = llvm::Log2_32_Ceil(producer.threads);
          unsigned slotBits = llvm::Log2_32(producer.depth);
          SmallVector<Value> threads, slots;
          for (unsigned t = 0; t < consumer.threads; ++t) {
            bool mine = sourceOf[t] == k;
            threads.push_back(constant(
                APInt(threadBits, mine ? consumer.pieces[t][j].thread : 0)));
            slots.push_back(mine ? Value(comb::ExtractOp::create(
                                       b, b.getIntegerType(slotBits),
                                       consumer.deq[t][j], 0))
                                 : constant(APInt(slotBits, 0)));
          }
          Value thread = select(threads, sink.thread);
          Value slot = select(slots, sink.thread);
          Value address = comb::ConcatOp::create(b, thread, slot);
          Value data = seq::FirMemReadOp::create(
              b, tokenMemory(producer, source->low, first.width, consumer.cls),
              address, hostClock, bit(true));
          values.push_back(forwarded(producer, thread, slot, source->low,
                                     first.width, data));
        }
        if (values.size() == 1) {
          parts.push_back(values[0]);
          continue;
        }
        unsigned indexBits = llvm::Log2_32_Ceil(values.size());
        while (values.size() < (1u << indexBits))
          values.push_back(values.back());
        SmallVector<Value> indices;
        for (unsigned t = 0; t < consumer.threads; ++t)
          indices.push_back(constant(APInt(indexBits, sourceOf[t])));
        Value index = select(indices, sink.thread);
        SmallVector<Value> elements(values.rbegin(), values.rend());
        parts.push_back(hw::ArrayGetOp::create(
            b, hw::ArrayCreateOp::create(b, elements), index));
      }
      bits = concat(parts);
    } else if (consumer.width > 0 && consumer.regular) {
      // Piece by piece, at the current thread.
      SmallVector<Value> parts;
      for (auto [j, piece] : llvm::enumerate(consumer.pieces[0])) {
        SmallVector<Value> perThread;
        if (piece.producer >= 0 && producers[piece.producer].memory) {
          auto &producer = producers[piece.producer];
          unsigned threadBits = llvm::Log2_32_Ceil(producer.threads);
          SmallVector<Value> threads, slots;
          for (unsigned t = 0; t < consumer.threads; ++t) {
            threads.push_back(
                constant(APInt(threadBits, consumer.pieces[t][j].thread)));
            slots.push_back(comb::ExtractOp::create(
                b, b.getIntegerType(llvm::Log2_32(producer.depth)),
                consumer.deq[t][j], 0));
          }
          Value thread = select(threads, sink.thread);
          Value slot = select(slots, sink.thread);
          Value address = comb::ConcatOp::create(b, thread, slot);
          Value data = seq::FirMemReadOp::create(
              b, tokenMemory(producer, piece.low, piece.width, consumer.cls),
              address, hostClock, bit(true));
          parts.push_back(forwarded(producer, thread, slot, piece.low,
                                    piece.width, data));
          continue;
        }
        for (unsigned t = 0; t < consumer.threads; ++t)
          perThread.push_back(bitsOf(t, j));
        parts.push_back(select(perThread, sink.thread));
      }
      bits = concat(parts);
    } else if (consumer.width > 0) {
      SmallVector<Value> perThread;
      for (auto [t, pieces] : llvm::enumerate(consumer.pieces)) {
        SmallVector<Value> parts;
        for (unsigned j = 0; j < pieces.size(); ++j)
          parts.push_back(bitsOf(t, j));
        perThread.push_back(concat(parts));
      }
      bits = select(perThread, sink.thread);
    }
    sink.set(valid, bits);

    Value fire = comb::AndOp::create(b, valid, sink.ready);
    for (auto [t, pieces] : llvm::enumerate(consumer.pieces)) {
      Value taken = comb::AndOp::create(b, fire, isThread(sink.thread, t));
      for (auto [j, piece] : llvm::enumerate(pieces))
        if (piece.producer >= 0)
          setNext(consumer.deq[t][j], increment(consumer.deq[t][j], taken));
    }
  }

  for (auto [c, cls] : llvm::enumerate(classes)) {
    auto *ready = modelInput(c, "thread_ready");
    if (!ready)
      continue;
    SmallVector<Value> bits;
    for (unsigned t = cls.instances.size(); t-- > 0;)
      bits.push_back(andAll(readyTerms[c][t]));
    ready->setValue(b.createOrFold<comb::ConcatOp>(bits));
  }

  SmallVector<Value> finishes;
  for (unsigned c = classes.size(); c-- > 0;)
    finishes.push_back(modelOutput(c, "finish"));
  outputs[profilePort] = b.createOrFold<comb::ConcatOp>(finishes);

  for (auto &value : outputs)
    if (!value)
      return netlist.emitError("a simulator output is not driven");
  hw::OutputOp::create(b, outputs);

  // The thread counts, for the shell and the runtime.
  SmallVector<Attribute> threads;
  for (auto &cls : classes)
    threads.push_back(
        b.getStringAttr(cls.name + "=" + Twine(cls.instances.size())));
  sim->setAttr("foldsim.threads", b.getArrayAttr(threads));
  // Nested, which instance each thread simulates, for the runtime's view of
  // the target and for tests.
  if (nest) {
    SmallVector<Attribute> order;
    for (auto &cls : classes) {
      std::string list = cls.name + "=";
      for (auto [t, instance] : llvm::enumerate(cls.instances))
        list += (t ? " " : "") + instance.getInstanceName().str();
      order.push_back(b.getStringAttr(list));
    }
    sim->setAttr("foldsim.thread_order", b.getArrayAttr(order));
  }
  return success();
}


LogicalResult Composer::buildDomainTree() {
  auto addDomain = [&](StringRef name) -> int {
    auto [it, inserted] = domainIndex.try_emplace(name, domainTree.size());
    if (inserted)
      domainTree.push_back({name.str(), -1, 0, {}});
    return it->second;
  };
  std::string topName = netlist.getModuleName().str();
  addDomain(topName);
  // A netlist's domain under the netlist that instantiates it.
  std::function<int(StringRef)> netlistDomain = [&](StringRef name) -> int {
    if (auto it = domainIndex.find(name); it != domainIndex.end())
      return it->second;
    int parent = netlistDomain(netlistParent.lookup(name));
    int d = addDomain(name);
    domainTree[d].parent = parent;
    return d;
  };
  for (auto &cls : classes) {
    // The netlists a class is instantiated in.
    SmallVector<int> owners;
    for (auto instance : cls.instances) {
      auto it = ownerOf.find(instance);
      if (it == ownerOf.end())
        return instance.emitError("a model instance from no netlist");
      int owner = netlistDomain(it->second);
      if (!llvm::is_contained(owners, owner))
        owners.push_back(owner);
    }
    if (owners.size() != 1)
      return cls.instances[0].emitError(
          "a class instantiated by several netlists has no one domain");
    int owner = owners[0];
    if (cls.name == domainTree[owner].name + "_model") {
      classDomain.push_back(owner);
      continue;
    }
    if (domainIndex.count(cls.name))
      return cls.instances[0].emitError("a class named as a netlist");
    int d = addDomain(cls.name);
    domainTree[d].parent = owner;
    classDomain.push_back(d);
  }
  if (domainTree.size() > 64)
    return netlist.emitError("more than 64 domains");
  for (auto [d, domain] : llvm::enumerate(domainTree)) {
    if (domain.parent >= 0)
      domainTree[domain.parent].children.push_back(d);
    for (int p = domain.parent; p >= 0; p = domainTree[p].parent)
      ++domain.depth;
  }
  return success();
}

LogicalResult Composer::emitDomains() {
  auto *context = top.getContext();
  auto *body = sim.getBodyBlock();
  auto *terminator = body->getTerminator();
  unsigned numDomains = domainTree.size();
  using Mask = uint64_t;
  const int topDomain = 0;
  auto bit = [](int d) -> Mask { return Mask(1) << d; };
  auto lca = [&](Mask mask) -> int {
    if (!mask)
      return topDomain;
    int best = -1;
    for (unsigned d = 0; d < numDomains; ++d) {
      if (!(mask & bit(d)))
        continue;
      if (best < 0) {
        best = d;
        continue;
      }
      int x = best, y = d;
      while (domainTree[x].depth > domainTree[y].depth)
        x = domainTree[x].parent;
      while (domainTree[y].depth > domainTree[x].depth)
        y = domainTree[y].parent;
      while (x != y) {
        x = domainTree[x].parent;
        y = domainTree[y].parent;
      }
      best = x;
    }
    return best;
  };

  // The models' domains.
  DenseMap<Operation *, int> domainOf;
  DenseMap<Operation *, unsigned> modelClass;
  for (auto &op : body->without_terminator())
    if (auto instance = dyn_cast<hw::InstanceOp>(op))
      for (auto [c, cls] : llvm::enumerate(classes))
        if (instance.getModuleNameAttr().getAttr() ==
            cls.threaded.getModuleNameAttr()) {
          domainOf[instance] = classDomain[c];
          modelClass[instance] = c;
        }
  auto isModel = [&](Operation *op) { return modelClass.count(op) > 0; };
  auto isConstant = [](Operation *op) {
    return isa<hw::ConstantOp, hw::AggregateConstantOp>(op);
  };
  auto isState = [&](Operation *op) {
    return isModel(op) ||
           isa<seq::FirRegOp, seq::FirMemOp, seq::FirMemReadOp,
               seq::FirMemWriteOp>(op);
  };
  Value clock = body->getArgument(0), reset = body->getArgument(1);
  SmallVector<Operation *> ops;
  for (auto &op : body->without_terminator())
    ops.push_back(&op);

  // The models an op is connected to, through whatever `stops` does not stop.
  auto cones = [&](auto stops, DenseMap<Operation *, Mask> &in,
                   DenseMap<Operation *, Mask> &out) {
    auto sourceMask = [&](Value v, DenseMap<Operation *, Mask> &table) -> Mask {
      if (v == clock || v == reset)
        return 0;
      auto *def = v.getDefiningOp();
      if (!def)
        return bit(topDomain);
      if (isModel(def))
        return bit(domainOf[def]);
      if (isConstant(def) || stops(def))
        return 0;
      return table.lookup(def);
    };
    for (bool changed = true; changed;) {
      changed = false;
      for (auto *op : ops) {
        if (isModel(op) || isConstant(op) || stops(op))
          continue;
        Mask m = in.lookup(op);
        for (auto operand : op->getOperands())
          m |= sourceMask(operand, in);
        if (m != in.lookup(op)) {
          in[op] = m;
          changed = true;
        }
      }
      for (auto *op : llvm::reverse(ops)) {
        if (isModel(op) || isConstant(op) || stops(op))
          continue;
        Mask m = out.lookup(op);
        for (auto *user : op->getUsers()) {
          if (user == terminator)
            m |= bit(topDomain);
          else if (isModel(user))
            m |= bit(domainOf[user]);
          else if (!stops(user))
            m |= out.lookup(user);
        }
        if (m != out.lookup(op)) {
          out[op] = m;
          changed = true;
        }
      }
    }
  };
  // A memory goes with its ports.
  DenseMap<Operation *, Operation *> memoryOf;
  for (auto *op : ops) {
    if (auto read = dyn_cast<seq::FirMemReadOp>(op))
      memoryOf[op] = read.getMemory().getDefiningOp();
    else if (auto write = dyn_cast<seq::FirMemWriteOp>(op))
      memoryOf[op] = write.getMemory().getDefiningOp();
    else if (isa<seq::FirMemOp>(op))
      memoryOf[op] = op;
  }

  // Through the state: a channel's state and logic go to the LCA of the
  // models it connects.
  DenseMap<Operation *, Mask> fanIn, fanOut;
  cones([](Operation *) { return false; }, fanIn, fanOut);
  DenseMap<Operation *, Mask> groupMask;
  for (auto *op : ops)
    if (auto *memory = memoryOf.lookup(op))
      groupMask[memory] |= fanIn.lookup(op) | fanOut.lookup(op);
  for (auto *op : ops) {
    if (isModel(op) || isConstant(op))
      continue;
    if (auto *memory = memoryOf.lookup(op))
      domainOf[op] = lca(groupMask[memory]);
    else
      domainOf[op] = lca(fanIn.lookup(op) | fanOut.lookup(op));
  }

  // Behind a relay, a channel's state is on its producer's or its
  // consumer's side, and logic goes next to what it drives; a staged token's
  // staging is on its producer's.
  auto relaySide = [](Operation *op) -> int {
    auto side = op->getAttrOfType<IntegerAttr>(kRelaySide);
    return side ? side.getInt() : 0;
  };
  // Staged tokens stage their producer's side the same way.
  bool anyRelay =
      llvm::any_of(producers, [](auto &p) { return p.relay || p.staged; });
  if (anyRelay) {
    DenseMap<Operation *, Mask> combIn, combOut;
    cones([&](Operation *op) { return isState(op) && !isa<seq::FirMemReadOp>(op); },
          combIn, combOut);
    auto combSources = [&](Operation *op) {
      Mask m = 0;
      for (auto operand : op->getOperands()) {
        if (operand == clock || operand == reset)
          continue;
        auto *def = operand.getDefiningOp();
        if (!def)
          m |= bit(topDomain);
        else if (isModel(def))
          m |= bit(domainOf[def]);
        else if (!isConstant(def) && !(isState(def) && !isa<seq::FirMemReadOp>(def)))
          m |= combIn.lookup(def);
      }
      return m;
    };
    auto combSinks = [&](Operation *op) {
      Mask m = 0;
      for (auto *user : op->getUsers()) {
        if (user == terminator)
          m |= bit(topDomain);
        else if (isModel(user))
          m |= bit(domainOf[user]);
        else if (!(isState(user) && !isa<seq::FirMemReadOp>(user)))
          m |= combOut.lookup(user);
      }
      return m;
    };
    for (auto *op : ops) {
      auto reg = dyn_cast<seq::FirRegOp>(op);
      if (!reg)
        continue;
      if (int side = relaySide(op))
        domainOf[op] = lca(side == 1 ? combSources(op) : combSinks(op));
    }
    for (auto *op : ops) {
      auto memory = dyn_cast<seq::FirMemOp>(op);
      if (!memory || !relaySide(op))
        continue;
      Mask m = 0;
      for (auto *user : memory->getUsers())
        if (isa<seq::FirMemReadOp>(user))
          m |= combSinks(user);
      int d = lca(m);
      for (auto *other : ops)
        if (memoryOf.lookup(other) == op)
          domainOf[other] = d;
    }
    // Logic between state goes with what it drives, where that state is a
    // relay's: elsewhere it stays at the LCA, so a leaf does not pull in the
    // wide reads of fabric it does not own.
    DenseSet<Operation *> relayState;
    for (auto *op : ops) {
      if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
        if (relaySide(reg))
          relayState.insert(op);
      } else if (auto *memory = memoryOf.lookup(op)) {
        if (relaySide(memory))
          relayState.insert(op);
      }
    }
    DenseSet<Operation *> nearRelay;
    for (bool changed = true; changed;) {
      changed = false;
      for (auto *op : ops) {
        if (isState(op) || isConstant(op) || nearRelay.contains(op))
          continue;
        bool near = false;
        for (auto operand : op->getOperands())
          if (auto *def = operand.getDefiningOp())
            near |= relayState.contains(def) ||
                    (!isState(def) && nearRelay.contains(def));
        for (auto *user : op->getUsers())
          near |= relayState.contains(user) ||
                  (!isState(user) && nearRelay.contains(user));
        if (near) {
          nearRelay.insert(op);
          changed = true;
        }
      }
    }
    DenseMap<Operation *, Mask> anchorIn, anchorOut;
    auto anchor = [&](Operation *op) { return isState(op); };
    for (bool changed = true; changed;) {
      changed = false;
      for (auto *op : ops) {
        if (anchor(op) || isConstant(op))
          continue;
        Mask m = anchorIn.lookup(op);
        for (auto operand : op->getOperands()) {
          if (operand == clock || operand == reset)
            continue;
          auto *def = operand.getDefiningOp();
          if (!def)
            m |= bit(topDomain);
          else if (anchor(def))
            m |= bit(domainOf[def]);
          else if (!isConstant(def))
            m |= anchorIn.lookup(def);
        }
        if (m != anchorIn.lookup(op)) {
          anchorIn[op] = m;
          changed = true;
        }
      }
      for (auto *op : llvm::reverse(ops)) {
        if (anchor(op) || isConstant(op))
          continue;
        Mask m = anchorOut.lookup(op);
        for (auto *user : op->getUsers()) {
          if (user == terminator)
            m |= bit(topDomain);
          else
            m |= anchor(user) ? bit(domainOf[user]) : anchorOut.lookup(user);
        }
        if (m != anchorOut.lookup(op)) {
          anchorOut[op] = m;
          changed = true;
        }
      }
    }
    for (auto *op : ops) {
      if (anchor(op) || isConstant(op) || !nearRelay.contains(op))
        continue;
      Mask out = anchorOut.lookup(op);
      domainOf[op] = lca(out ? out : anchorIn.lookup(op));
    }
  }

  // The values crossing each domain's boundary, in and out.
  auto valueDomain = [&](Value v) -> int {
    auto *def = v.getDefiningOp();
    return def ? domainOf.lookup(def) : topDomain;
  };
  std::vector<llvm::SetVector<Value>> inputs(numDomains), outputs(numDomains);
  auto need = [&](Value v, int e) {
    int dv = valueDomain(v);
    if (dv == e)
      return;
    int l = lca(bit(dv) | bit(e));
    for (int x = dv; x != l; x = domainTree[x].parent)
      outputs[x].insert(v);
    for (int x = e; x != l; x = domainTree[x].parent)
      inputs[x].insert(v);
  };
  for (auto *op : ops) {
    if (isConstant(op))
      continue;
    for (auto operand : op->getOperands()) {
      auto *def = operand.getDefiningOp();
      if (def && isConstant(def))
        continue;
      need(operand, domainOf[op]);
    }
  }
  for (auto operand : terminator->getOperands()) {
    auto *def = operand.getDefiningOp();
    if (!def || !isConstant(def))
      need(operand, topDomain);
  }

  // Port names from what a value is.
  auto nameOf = [&](Value v) -> std::string {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return sim.getInputName(arg.getArgNumber()).str();
    auto *def = v.getDefiningOp();
    if (auto reg = dyn_cast<seq::FirRegOp>(def))
      return reg.getName().str();
    if (auto instance = dyn_cast<hw::InstanceOp>(def))
      return (instance.getInstanceName() + "_" +
              cast<StringAttr>(instance.getResultNames()[cast<OpResult>(v)
                                                              .getResultNumber()])
                  .getValue())
          .str();
    if (auto read = dyn_cast<seq::FirMemReadOp>(def))
      return (read.getMemory().getDefiningOp<seq::FirMemOp>().getName().value_or(
                  "mem") +
              "_rdata")
          .str();
    return "w";
  };

  // A module per domain but the top one, the simulator itself.
  OpBuilder outer(sim);
  SmallVector<hw::HWModuleOp> modules(numDomains);
  std::vector<DenseMap<Value, Value>> local(numDomains);
  for (unsigned d = 1; d < numDomains; ++d) {
    SmallVector<hw::PortInfo> ports;
    llvm::StringSet<> used;
    auto unique = [&](std::string name) {
      std::string candidate = name;
      for (unsigned i = 1; !used.insert(candidate).second; ++i)
        candidate = name + "_" + std::to_string(i);
      return candidate;
    };
    unsigned numIn = 0, numOut = 0;
    for (auto v : inputs[d]) {
      hw::PortInfo port;
      port.name = StringAttr::get(context, unique(nameOf(v)));
      port.type = v.getType();
      port.dir = hw::ModulePort::Direction::Input;
      port.argNum = numIn++;
      ports.push_back(port);
    }
    for (auto v : outputs[d]) {
      hw::PortInfo port;
      port.name = StringAttr::get(context, unique(nameOf(v)));
      port.type = v.getType();
      port.dir = hw::ModulePort::Direction::Output;
      port.argNum = numOut++;
      ports.push_back(port);
    }
    modules[d] = hw::HWModuleOp::create(
        outer, sim.getLoc(),
        StringAttr::get(context, simName + "_" + domainTree[d].name),
        hw::ModulePortInfo(ports));
    modules[d].setPrivate();
    for (auto [i, v] : llvm::enumerate(inputs[d]))
      local[d][v] = modules[d].getBodyBlock()->getArgument(i);
  }
  auto bodyOf = [&](int d) {
    return d == topDomain ? body : modules[d].getBodyBlock();
  };
  // Move each op into its domain.
  for (auto *op : ops) {
    int d = domainOf.lookup(op);
    if (isConstant(op) || d == topDomain)
      continue;
    op->moveBefore(bodyOf(d)->getTerminator());
  }
  // The child domains' instances, with their outputs as the parent's values.
  bool keepRelayed = anyRelay;
  SmallVector<hw::InstanceOp> instances(numDomains);
  for (unsigned d = 1; d < numDomains; ++d) {
    int p = domainTree[d].parent;
    OpBuilder b = OpBuilder::atBlockTerminator(bodyOf(p));
    SmallVector<Value> operands(inputs[d].begin(), inputs[d].end());
    instances[d] = hw::InstanceOp::create(
        b, sim.getLoc(), modules[d],
        b.getStringAttr(domainTree[d].name + "_dom"), operands);
    for (auto [i, v] : llvm::enumerate(outputs[d]))
      local[p][v] = instances[d].getResult(i);
  }
  // A value as a domain sees it.
  std::function<Value(int, Value)> localOf = [&](int d, Value v) -> Value {
    if (valueDomain(v) == d)
      return v;
    auto it = local[d].find(v);
    if (it != local[d].end())
      return it->second;
    return Value();
  };
  DenseMap<std::pair<int, Operation *>, Value> constants;
  auto constantIn = [&](int d, Operation *constant) -> Value {
    if (d == topDomain)
      return constant->getResult(0);
    auto &value = constants[{d, constant}];
    if (!value) {
      OpBuilder b(bodyOf(d), bodyOf(d)->begin());
      value = b.clone(*constant)->getResult(0);
    }
    return value;
  };
  auto rewire = [&](Operation *op, int d) -> LogicalResult {
    for (auto &operand : op->getOpOperands()) {
      Value v = operand.get();
      auto *def = v.getDefiningOp();
      if (def && isConstant(def)) {
        operand.set(constantIn(d, def));
        continue;
      }
      Value mapped = localOf(d, v);
      if (!mapped)
        return op->emitError("a value its domain does not see");
      operand.set(mapped);
    }
    return success();
  };
  for (auto *op : ops)
    if (!isConstant(op) && failed(rewire(op, domainOf.lookup(op))))
      return failure();
  for (unsigned d = 1; d < numDomains; ++d) {
    if (failed(rewire(instances[d], domainTree[d].parent)))
      return failure();
    SmallVector<Value> results;
    for (auto v : outputs[d])
      results.push_back(localOf(d, v));
    if (llvm::any_of(results, [](Value v) { return !v; }))
      return sim.emitError("a domain output it does not see");
    auto *output = modules[d].getBodyBlock()->getTerminator();
    output->setOperands(results);
    // A relayed boundary is kept: nothing crosses it combinationally.
    if (keepRelayed) {
      bool relayed = false;
      for (auto &producer : producers) {
        if (!producer.relay)
          continue;
        int pd = producer.cls < 0 ? topDomain : classDomain[producer.cls];
        for (auto &read : producer.reads) {
          int cd = classDomain[consumers[read.consumer].cls];
          auto inside = [&](int x) {
            for (; x >= 0; x = domainTree[x].parent)
              if (x == int(d))
                return true;
            return false;
          };
          if (inside(pd) != inside(cd))
            relayed = true;
        }
      }
      if (relayed)
        sv::setSVAttributes(instances[d],
                            sv::SVAttributeAttr::get(context, "keep_hierarchy",
                                                     R"("yes")", false));
    }
  }
  if (failed(rewire(terminator, topDomain)))
    return failure();
  // Constants nobody uses any more in the top domain stay for canonicalize.
  return success();
}

LogicalResult Composer::run() {
  inlineNetlists();
  if (failed(collect()) || failed(build()))
    return failure();
  if (emitDomainTree && (failed(buildDomainTree()) || failed(emitDomains())))
    return failure();
  netlist.erase();
  for (auto *op : inlined)
    op->erase();
  top.walk([](Operation *op) { op->removeAttr(kRelaySide); });

  return success();
}

void ComposePass::runOnOperation() {
  auto top = getOperation();
  if (stateStages < 1 || stateStages > 2) {
    top.emitError("a thread's state is written one or two host cycles after "
                  "its edge");
    return signalPassFailure();
  }
  auto netlist = dyn_cast_or_null<hw::HWModuleOp>(
      SymbolTable::lookupSymbolIn(top, moduleName));
  if (!netlist) {
    top.emitError("no netlist '") << moduleName << "'";
    return signalPassFailure();
  }
  foldsim::ThreadOptions threadOptions{regionLoads, registeredState,
                                       stateStages, commitSelect,
                                       stagedRamWrites, pipelineMultipliers,
                                       compactStaging, foldPipelines,
                                       idleSlots, pipelineDepth, blockStateDepth,
                                       commitEnable};
  if (channelDepth < 2 || !llvm::isPowerOf2_32(channelDepth)) {
    top.emitError("a channel's depth is a power of two, at least two");
    return signalPassFailure();
  }
  NestOptions nesting;
  nesting.nest = nest;
  nesting.skew = nestSkew;
  nesting.offsets = nestOffsets;
  nesting.copies = copies;
  nesting.bankedTokens = bankedTokens;
  if (failed(Composer(top, netlist, simName, readySchedule, relayModels,
                          threadOptions, stagedTokens, channelDepth, domains,
                          nesting)
                     .run()))
    signalPassFailure();
}
