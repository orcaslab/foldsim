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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>
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
  /// Host cycles from an instance's state pass to its next evaluation's
  /// first, the fewest over its instances: what the state bypass must cover.
  int stateGap = 0;
  /// Whether its memories include a registered read: its state is then read
  /// a host cycle later after a state pass.
  bool syncReads = false;
  /// Its memories kept outside the simulator (`--virtual-memories`): each
  /// one's row address, data and mask widths, read-during-write order, and
  /// where its request and response ports start among the copy's.
  struct VirtualMemory {
    std::string name;
    unsigned rowWidth, dataWidth, maskWidth;
    bool readFirst;
    unsigned firstInput, firstOutput;
  };
  SmallVector<VirtualMemory> virtualMemories;
  /// With virtual memories, the copy's host reset input (their pending reads
  /// clear with the controllers').
  unsigned resetInput = 0;
  /// The schedule: its ready passes (none, one, or one per level of its late
  /// outputs, `readyLevels`), and each instance's slots: `slot[kState]` its
  /// state pass's, `slot[kReady]` its first ready pass's and `laterReady[l-1]`
  /// its ready pass of level l's.
  bool hasReady = false;
  unsigned readyLevels = 0;
  SmallVector<int> slot[2];
  SmallVector<SmallVector<int>> laterReady;
  /// Each instance's slots of the ready pass of level `level`.
  ArrayRef<int> readySlots(unsigned level) const {
    return level ? ArrayRef<int>(laterReady[level - 1])
                 : ArrayRef<int>(slot[kReady]);
  }
  /// Paged register state (`page`): host cycles an instance's state takes
  /// to move in or out of its page RAM, or 0 if it is not paged.
  unsigned page = 0;
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
  /// For a class output: late, and then whether written at a ready pass, and
  /// at which level's.
  bool late = false, atReady = false;
  unsigned readyLevel = 0;
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
/// A virtual memory's controller reports its free request entries in this
/// many bits; a position runs only with `kVirtualRoom` of them, two requests
/// for each host cycle from a decision to the requests it causes.
static constexpr unsigned kVirtualFreeWidth = 16, kVirtualRoom = 8;

/// One evaluation of a frame: an instance of a class in a pass.
struct Node {
  unsigned cls, inst;
  Pass pass;
  /// A ready pass's level.
  unsigned level = 0;
};

/// Why an evaluation waits for another: the link it reads (the producer, and
/// the consumer's input), or, with no producer, an instance's ready pass
/// before its state pass.
struct Wait {
  int producer = -1;
  unsigned input = 0;
};

/// Which late outputs a choice of ready passes writes at them, and at which
/// pass: one ready pass per class, or (split by level) one per level of its
/// late outputs, each level's inputs written before it in the frame.
struct ReadyWrites {
  /// Per producer: written at a ready pass, and that pass's level.
  SmallVector<bool> atReady;
  SmallVector<unsigned> level;
  /// Per class: its ready passes (0 for none), and the first of the slot
  /// rows of its ready passes past the first.
  SmallVector<unsigned> passes, laterRow;
  unsigned rows = 0;
  /// The row of a class's pass in the slots `place` gives: the state pass's
  /// and the first ready pass's by class, two to a class, then the later
  /// ready passes'.
  unsigned row(unsigned c, Pass pass, unsigned lvl = 0) const {
    return pass == kState || lvl == 0 ? c * 2 + pass : laterRow[c] + lvl - 1;
  }
  /// The pass a producer's output is written at.
  Pass passOf(unsigned p) const { return atReady[p] ? kReady : kState; }
};

/// The evaluations of a frame for a choice of ready passes, and what each
/// waits for within the frame: the evaluation and the host cycles after it,
/// and (in `why`, when kept) the link.
struct FrameGraph {
  std::vector<Node> nodes;
  std::vector<SmallVector<std::pair<unsigned, int>>> preds;
  std::vector<SmallVector<Wait>> why;
};

class StaticFolder {
public:
  StaticFolder(ModuleOp top, hw::HWModuleOp netlist, StringRef simName,
               unsigned regionBits, StringRef report,
               ArrayRef<std::string> copies, bool nest, bool scheduleOnly,
               bool oneCycleLinks, bool overlapFrames, bool modulo,
               bool interleaveCopies, bool zeroLatencyLinks,
               bool zeroLatencyChains, bool shortLead, bool bubbles,
               StringRef virtualMemories, bool stagedEnv, ArrayRef<std::string> specGroups, unsigned specDepth,
               ArrayRef<std::string> envSinks, unsigned registerStateEntries,
               unsigned slrGroups, ArrayRef<std::string> slrSites,
               uint64_t slrCapacity, ArrayRef<std::string> slrAssign,
               ArrayRef<std::string> storage, StringRef readyLevels,
               unsigned forwardAfterLevels, bool shareLinkWrites,
               ArrayRef<std::string> page)
      : top(top), netlist(netlist), simName(simName), regionBits(regionBits),
        reportPath(report), copies(copies), nest(nest),
        specGroupSpecs(specGroups), envSinkNames(envSinks),
        specDepth(specDepth), slrGroups(slrGroups), slrSiteSpecs(slrSites),
        slrCapacity(slrCapacity), slrAssignSpecs(slrAssign),
        scheduleOnly(scheduleOnly), oneCycleLinks(oneCycleLinks),
        overlapFrames(overlapFrames), modulo(modulo),
        interleaveCopies(interleaveCopies), zeroLatencyLinks(zeroLatencyLinks),
        zeroLatencyChains(zeroLatencyChains), shortLead(shortLead),
        bubbles(bubbles), virtualMemoryPattern(virtualMemories.str()),
        readyLevelsMode(readyLevels.str()),
        registerStateEntries(registerStateEntries), storageSpecs(storage),
        forwardAfterLevels(forwardAfterLevels),
        shareLinkWrites(shareLinkWrites), pageSpecs(page),
        envRoundTrip((stagedEnv ? 1 : 0) + 1 + (shortLead ? 3 : 4)),
        linkLatency(oneCycleLinks ? kForwardedLinkLatency : kLinkLatency) {}
  LogicalResult run();

private:
  void inlineNetlists();
  LogicalResult collect();
  LogicalResult resolveEnvironment();
  /// Host cycles from a netlist output's slot to the end of the frame it
  /// must precede: 1 for an early token or one the environment never waits
  /// for, the environment's round trip for a late one, and for a select of
  /// offered pairs the slack of a round trip spread over two frames.
  int envSlack(unsigned k, int period) const;
  int envSlackOf(unsigned k, bool late, int period) const;
  LogicalResult splitCopies();
  LogicalResult nestInstances();
  LogicalResult decompose(Value value, unsigned low, unsigned width,
                          Pieces &pieces);
  LogicalResult analyze(Class &cls);
  LogicalResult schedule();
  /// The shortest frame over the choices of ready passes, with the links
  /// between SLR groups as `slrOf` has them; 0 if every choice is cyclic.
  struct Choice {
    unsigned length = 0;
    SmallVector<bool> ready, outputEarly;
    ReadyWrites writes;
    SmallVector<SmallVector<int>> slots;
    SmallVector<int> outputSlot;
  };
  Choice searchReady();
  /// Assigns the classes to SLR groups (`slrOf`): within the capacity, the
  /// shortest frame, then the fewest link bits between groups.
  LogicalResult chooseGroups();
  /// Bits each class's outputs send to each other class's inputs: per
  /// consumer input, the producer bits it reads (a link of its own).
  void linkBits(SmallVectorImpl<SmallVector<uint64_t>> &bits,
                SmallVectorImpl<SmallVector<unsigned>> &links) const;
  /// Moves each SLR group's logic of the simulator into a module of its
  /// own, instantiated as `slr<g>`.
  LogicalResult extractGroups();
  /// Whether a link from class `p` to class `c` crosses SLR groups: it is
  /// then registered on both sides and taken a host cycle later.
  bool crosses(int p, int c) const {
    return !slrOf.empty() && p >= 0 && c >= 0 && slrOf[p] != slrOf[c];
  }
  int crossDelay(int p, int c) const { return crosses(p, c) ? 1 : 0; }
  /// The evaluations of the given choice of ready passes and what each waits
  /// for; which late outputs are written at ready passes in `writes`.
  void frameGraph(ArrayRef<bool> ready, ReadyWrites &writes,
                  FrameGraph &graph, bool keepWhy);
  /// With `splitReady`: the levels of the late outputs written at ready
  /// passes, as few per class as keep the passes, instance by instance, from
  /// waiting on each other in a cycle. Left at one pass per class where the
  /// outputs themselves wait on each other in a cycle.
  void readyLevels(ArrayRef<bool> ready, ReadyWrites &writes);
  /// Moves to the state pass every late output in `writes.atReady` that
  /// depends on an input a late output not written at a ready pass feeds.
  void settleReady(ReadyWrites &writes);
  /// A late output `q` of instance `a` feeding an input that late output
  /// `p` of instance `b` depends on: {q, a, b}, listed under `p`.
  using Feed = std::tuple<unsigned, unsigned, unsigned>;
  /// The feeds of late output `p`, every instance's; computed once.
  ArrayRef<Feed> lateFeeds(unsigned p);
  std::vector<SmallVector<Feed>> feeds;
  /// Places every evaluation of the given choice of ready passes; II, or 0
  /// if the evaluations depend on each other in a cycle.
  unsigned place(ArrayRef<bool> ready, SmallVectorImpl<SmallVector<int>> &slots,
                 ReadyWrites &writes);
  /// Names a shortest cycle of the evaluations of the given choice of ready
  /// passes, edge by edge, on the diagnostic: which link each waits on, and
  /// whether a ready pass could break it.
  void explainCycle(ArrayRef<bool> ready, InFlightDiagnostic &diag);
  /// Places the evaluations with consecutive frames overlapped: the shortest
  /// frame at which every copy's evaluations fit, modulo the frame, and every
  /// dependence across frames holds; 0 if none does.
  unsigned placeModulo(
      ArrayRef<Node> nodes,
      ArrayRef<SmallVector<std::pair<unsigned, int>>> preds,
      ArrayRef<SmallVector<std::pair<unsigned, int>>> succs,
      ArrayRef<int> height, const ReadyWrites &writes,
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
  ArrayRef<std::string> specGroupSpecs, envSinkNames;
  // Frames of a group's select its offers span, and the values a token holds.
  unsigned specDepth;
  unsigned specValues() const { return 1u << specDepth; }
  /// SLR groups: their most, each class's LUT sites, a group's most sites,
  /// classes pinned; each class's group (empty without groups), and the
  /// simulator built.
  unsigned slrGroups;
  ArrayRef<std::string> slrSiteSpecs;
  uint64_t slrCapacity;
  ArrayRef<std::string> slrAssignSpecs;
  SmallVector<int> slrOf;
  SmallVector<uint64_t> slrSites;
  hw::HWModuleOp simModule;
  /// A select of offered pairs: the netlist output, and the input whose
  /// value in the frame before it is ANDed with.
  struct SpecGroup {
    unsigned select, valid;
  };
  SmallVector<SpecGroup> specGroups;
  /// Per netlist input, its group's index, or -1; per netlist output, the
  /// group it selects for, or -1, and whether it closes no round trip.
  SmallVector<int> specOf, specSelect;
  SmallVector<bool> envSink;
  /// Nested folding: the netlist each instance was inlined from (a path of
  /// instance names), and the instances that are a netlist's own logic.
  DenseMap<Operation *, std::string> containerOf;
  DenseSet<Operation *> logicInstances;
  bool scheduleOnly;
  bool oneCycleLinks, overlapFrames, modulo, interleaveCopies;
  bool zeroLatencyLinks, zeroLatencyChains;
  bool shortLead;
  bool bubbles;
  /// `--virtual-memories`: the names of the memories kept outside.
  std::string virtualMemoryPattern;
  /// When a class may write its ready outputs in more than one ready pass,
  /// by level (`ready-levels`: off, auto or always); and whether
  /// `frameGraph` does so now.
  std::string readyLevelsMode;
  bool splitReady = false;
  unsigned registerStateEntries;
  /// The storage rules (`storage`): a model's stores in a given ram_style.
  struct StorageRule {
    std::string model, store, kind;
    /// A rule for the model's register stores, of at least `minWidth` bits.
    bool regs = false;
    unsigned minWidth = 0;
    bool used = false;
  };
  ArrayRef<std::string> storageSpecs;
  SmallVector<StorageRule> storageRules;
  LogicalResult parseStorage();
  /// The rule for a store of class `cls`: a memory by name, or a register
  /// store of `width` bits; null if none.
  StorageRule *storageFor(const Class &cls, StringRef store, bool reg,
                          unsigned width);
  unsigned forwardAfterLevels;
  bool shareLinkWrites;
  /// `page`: the paged models, as <model>:<p>; each class's `page` from them.
  ArrayRef<std::string> pageSpecs;
  LogicalResult parsePages();
  /// What each paged copy pages, for the report.
  std::string pageReport;
  /// The registers of class `cls` a ready pass reads: the cone of each
  /// output written at a ready pass, through the logic and combinational
  /// reads, up to registers, registered reads and black boxes.
  void readyRegisters(const Class &cls, DenseSet<Operation *> &regs) const;
  /// Whether a paged class has a register to page: written, and read by no
  /// ready pass.
  bool pagesAny(const Class &cls) const;
  /// The fewest host cycles from an instance's state pass to its next
  /// evaluation: the write-back's, and with virtual memories the request's
  /// (a read issued at the state pass is pending before the position whose
  /// read register loads it is decided: 5 host cycles, 4 with a short lead).
  int minStateGap(const Class &cls) const;
  /// Whether class `c`'s state passes use its page port.
  bool paged(unsigned c) const { return classes[c].page > 0; }
  // Host cycles from a late netlist output's slot to the next frame's first.
  int envRoundTrip;
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
    if (k > n)
      return classes[c].instances[0].emitError("copies: ")
             << n << " instances of '" << classes[c].name
             << "' do not make " << k << " copies";
    // Copies of k not dividing n: the first n % k take one instance more.
    auto count = [&](unsigned j) { return n / k + (j < n % k ? 1 : 0); };
    auto start = [&](unsigned j) { return j * (n / k) + std::min(j, n % k); };
    auto instances = classes[c].instances;
    // Copy j's t-th instance: contiguous, or every k-th.
    auto pick = [&](unsigned j, unsigned t) {
      return instances[interleaveCopies ? t * k + j : start(j) + t];
    };
    classes[c].instances.clear();
    for (unsigned t = 0; t < count(0); ++t) {
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
      for (unsigned t = 0; t < count(j); ++t) {
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

/// The most levels on a path from `root` along `edges` (each a successor
/// and the levels it adds), memoized in `memo`: depth first without
/// recursion, since a model's logic may chain far more operations than a
/// thread's stack holds frames. A value on a cycle counts as none.
static int longestPath(
    Value root, DenseMap<Value, int> &memo,
    function_ref<void(Value, SmallVectorImpl<std::pair<Value, int>> &)>
        edges) {
  if (auto it = memo.find(root); it != memo.end())
    return it->second;
  SmallVector<std::pair<Value, bool>> stack{{root, false}};
  SmallVector<std::pair<Value, int>> next;
  while (!stack.empty()) {
    auto [value, expanded] = stack.pop_back_val();
    if (!expanded && memo.count(value))
      continue;
    next.clear();
    edges(value, next);
    if (!expanded) {
      // A placeholder against cycles, then the successors first.
      memo[value] = 0;
      stack.push_back({value, true});
      for (auto [successor, levels] : next)
        if (!memo.count(successor))
          stack.push_back({successor, false});
      continue;
    }
    int levels = 0;
    for (auto [successor, add] : next)
      levels = std::max(levels, add + memo.lookup(successor));
    memo[value] = levels;
  }
  return memo.lookup(root);
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
  // Backward from each output, through the logic, stopping at registers,
  // registered reads and black boxes.
  DenseMap<Value, int> toOutput;
  auto into = [](Value value, SmallVectorImpl<std::pair<Value, int>> &edges) {
    auto *op = value.getDefiningOp();
    if (!op)
      return;
    auto read = dyn_cast<seq::FirMemReadOp>(op);
    auto memory = read ? read.getMemory().getDefiningOp<seq::FirMemOp>()
                       : seq::FirMemOp();
    if (read && (!memory || memory.getReadLatency() == 0))
      edges.push_back({read.getAddress(), 1});
    else if (!read && !isa<seq::FirRegOp, hw::InstanceOp>(op))
      for (auto operand : op->getOperands())
        edges.push_back({operand, logicLevels(op)});
  };
  for (auto operand : output->getOperands())
    cls.outputDepth.push_back(longestPath(operand, toOutput, into));
  // Forward from each input, up to the state it reaches.
  DenseMap<Value, int> toState;
  auto outOf = [](Value value, SmallVectorImpl<std::pair<Value, int>> &edges) {
    for (auto &use : value.getUses()) {
      auto *user = use.getOwner();
      if (isa<seq::FirRegOp, seq::FirMemWriteOp, hw::InstanceOp, hw::OutputOp>(
              user))
        continue;
      if (auto read = dyn_cast<seq::FirMemReadOp>(user)) {
        auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
        if (!memory || memory.getReadLatency() == 0)
          edges.push_back({read.getData(), 1});
        continue;
      }
      for (auto result : user->getResults())
        edges.push_back({result, logicLevels(user)});
    }
  };
  for (auto arg : body->getArguments())
    cls.inputDepth.push_back(longestPath(arg, toState, outOf));
  return success();
}

bool StaticFolder::zeroLink(unsigned p, unsigned c, unsigned input) const {
  auto &producer = producers[p];
  if (!zeroLatencyLinks || producer.cls < 0 || !producer.late)
    return false;
  // Between SLR groups a link is registered on both sides of the crossing.
  if (crosses(producer.cls, c))
    return false;
  if (zeroLatencyChains)
    return true;
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
/// the token is offered at the slot, a staged endpoint takes it the host cycle
/// after (an unstaged one in that host cycle), finishes its cycle and offers
/// its next tokens the host cycle after that, and the next frame starts four
/// host cycles after it is decided on them (three with short-lead).

int StaticFolder::envSlackOf(unsigned k, bool late, int period) const {
  if (envSink[k])
    return 1;
  // An early output's token of a frame is the next cycle's: the environment's
  // tokens it decides load in the frame after that.
  if (!late)
    return std::max(1, envRoundTrip - period);
  if (specSelect[k] >= 0)
    return std::max(2, envRoundTrip - int(specDepth) * period);
  return envRoundTrip;
}

int StaticFolder::envSlack(unsigned k, int period) const {
  return envSlackOf(k, !outputEarly[k], period);
}

/// Host cycles from an instance's state pass to its next evaluation's first:
/// its registers, written at the end of the state pass, and a registered
/// read's data, out of the memory the host cycle after, both reach the next
/// evaluation through a bypass (`buildCopy`).
static int writeBackLatency(const Class &) { return 1; }

/// Whether a copy's instances can keep their registers in a ring: every pass
/// runs inside the frame, and each instance's passes come after the state
/// pass of the instance before it and end with its own. The register an
/// evaluation reads is then the ring's head, and a state pass rotates the
/// ring, so no instance is ever selected.
static bool ringOrder(const Class &cls, unsigned ii) {
  if (cls.threads() < 2)
    return false;
  SmallVector<std::tuple<int, unsigned, unsigned>> passes;
  for (unsigned i = 0; i < cls.threads(); ++i) {
    for (unsigned l = 0; l < cls.readyLevels; ++l) {
      int s = cls.readySlots(l)[i];
      if (s < 0 || s >= int(ii))
        return false;
      passes.push_back({s, unsigned(kReady), i});
    }
    int s = cls.slot[kState][i];
    if (s < 0 || s >= int(ii))
      return false;
    passes.push_back({s, unsigned(kState), i});
  }
  llvm::sort(passes);
  for (unsigned n = 1; n < passes.size(); ++n)
    if (std::get<0>(passes[n]) == std::get<0>(passes[n - 1]))
      return false;
  // Walk the frame from just after its last state pass: each instance's
  // passes must then be one run that its state pass ends.
  unsigned start = 0;
  for (unsigned n = 0; n < passes.size(); ++n)
    if (std::get<1>(passes[n]) == kState)
      start = n + 1;
  int current = -1;
  for (unsigned n = 0; n < passes.size(); ++n) {
    auto [slot, pass, inst] = passes[(start + n) % passes.size()];
    if (current < 0)
      current = inst;
    if (int(inst) != current)
      return false;
    if (pass == kState)
      current = -1;
  }
  return current < 0;
}

void StaticFolder::settleReady(ReadyWrites &writes) {
  // A late output stays at its ready pass while every input it depends on
  // comes from constants, the netlist's inputs, early outputs and late
  // outputs written at ready passes.
  auto &atReady = writes.atReady;
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
}

ArrayRef<StaticFolder::Feed> StaticFolder::lateFeeds(unsigned p) {
  if (feeds.empty()) {
    feeds.resize(producers.size());
    for (auto [o, producer] : llvm::enumerate(producers)) {
      if (producer.cls < 0 || !producer.late)
        continue;
      auto &cls = classes[producer.cls];
      auto &into = feeds[o];
      for (auto input : cls.deps[producer.port].set_bits()) {
        int k = cls.consumerOf[input];
        if (k < 0)
          continue;
        for (auto [b, pieces] : llvm::enumerate(consumers[k].pieces))
          for (auto &piece : pieces)
            if (piece.producer >= 0 && producers[piece.producer].cls >= 0 &&
                producers[piece.producer].late)
              into.push_back({unsigned(piece.producer), piece.inst, unsigned(b)});
      }
      llvm::sort(into);
      into.erase(std::unique(into.begin(), into.end()), into.end());
    }
  }
  return feeds[p];
}

void StaticFolder::readyLevels(ArrayRef<bool> ready, ReadyWrites &writes) {
  auto &atReady = writes.atReady;
  auto &level = writes.level;
  // Each instance's output written at a ready pass, a node: `first[p] + i`.
  // It waits for the instances' outputs that feed an input it depends on,
  // all written at ready passes (`settleReady`).
  SmallVector<unsigned> first(producers.size(), 0);
  unsigned numOuts = 0;
  for (auto [p, at] : llvm::enumerate(atReady))
    if (at) {
      first[p] = numOuts;
      numOuts += classes[producers[p].cls].threads();
    }
  SmallVector<unsigned> waiting(numOuts, 0), outEdges(numOuts + 1, 0);
  SmallVector<std::pair<unsigned, unsigned>> links;
  for (auto [p, at] : llvm::enumerate(atReady)) {
    if (!at)
      continue;
    for (auto [q, a, b] : lateFeeds(p)) {
      assert(atReady[q] && "a ready output fed by one at a state pass");
      links.push_back({first[q] + a, first[p] + b});
      ++waiting[first[p] + b];
    }
  }
  llvm::sort(links);
  for (auto [from, to] : links)
    ++outEdges[from + 1];
  for (unsigned n = 0; n < numOuts; ++n)
    outEdges[n + 1] += outEdges[n];
  // Its rank: the longest chain of such outputs up to it. Outputs that wait
  // on each other in a cycle form a combinational loop through the ports,
  // which no choice of passes schedules: then one ready pass per class, and
  // the frame's cycle is reported as such.
  SmallVector<unsigned> rank(numOuts, 0), queue;
  for (unsigned n = 0; n < numOuts; ++n)
    if (!waiting[n])
      queue.push_back(n);
  unsigned done = 0;
  while (!queue.empty()) {
    unsigned n = queue.pop_back_val();
    ++done;
    for (unsigned e = outEdges[n]; e < outEdges[n + 1]; ++e) {
      unsigned m = links[e].second;
      rank[m] = std::max(rank[m], rank[n] + 1);
      if (!--waiting[m])
        queue.push_back(m);
    }
  }
  if (done != numOuts)
    return;

  // Each of a class's outputs a level of its own, in the order of their
  // highest rank: a pass then has the rank of its output in its instance,
  // and every link raises it, so no pass waits on itself through others.
  // With `rank`, the outputs of the same ranks, instance by instance, share
  // one, for the same reason, and that is all. (Ranked by class instead, a
  // chain through the instances of a class, a core's per-slice CSR written by
  // the slices before, each slice's write read by the next, would be a loop.)
  std::vector<std::map<std::vector<unsigned>, SmallVector<unsigned>>> byRanks(
      classes.size());
  for (auto [p, at] : llvm::enumerate(atReady)) {
    if (!at)
      continue;
    unsigned threads = classes[producers[p].cls].threads();
    std::vector<unsigned> key{0};
    for (unsigned i = 0; i < threads; ++i) {
      key.push_back(rank[first[p] + i]);
      key[0] = std::max(key[0], key.back());
    }
    byRanks[producers[p].cls][key].push_back(p);
  }
  SmallVector<unsigned> levels(classes.size(), 0);
  std::vector<SmallVector<unsigned>> outputs(classes.size());
  bool byRank = readyLevelsMode == "rank";
  for (auto [c, own] : llvm::enumerate(byRanks))
    for (auto &[key, outs] : own) {
      if (byRank) {
        for (unsigned p : outs)
          level[p] = levels[c];
        ++levels[c];
        continue;
      }
      for (unsigned p : outs) {
        level[p] = levels[c]++;
        outputs[c].push_back(p);
      }
    }

  // Then as few levels as keep the passes acyclic, instance by instance:
  // class by class, each output in rank order joins the first level that
  // keeps them so, or starts its own. The classes of the most instances
  // first, while the others' levels are still the finest: a level costs a
  // pass per instance.
  SmallVector<unsigned> passBase(classes.size() + 1, 0);
  auto passOf = [&](unsigned p, unsigned i) {
    unsigned c = producers[p].cls;
    return passBase[c] + i * levels[c] + level[p];
  };
  auto acyclic = [&]() {
    for (auto [c, cls] : llvm::enumerate(classes))
      passBase[c + 1] = passBase[c] + cls.threads() * levels[c];
    unsigned numPasses = passBase.back();
    SmallVector<unsigned> waits(numPasses, 0), edges(numPasses + 1, 0);
    SmallVector<std::pair<unsigned, unsigned>> passLinks;
    for (auto [p, at] : llvm::enumerate(atReady)) {
      if (!at)
        continue;
      for (auto [q, a, b] : lateFeeds(p)) {
        unsigned from = passOf(q, a), to = passOf(p, b);
        if (from == to)
          return false;
        passLinks.push_back({from, to});
      }
    }
    llvm::sort(passLinks);
    passLinks.erase(std::unique(passLinks.begin(), passLinks.end()),
                    passLinks.end());
    for (auto [from, to] : passLinks) {
      ++edges[from + 1];
      ++waits[to];
    }
    for (unsigned n = 0; n < numPasses; ++n)
      edges[n + 1] += edges[n];
    SmallVector<unsigned> sources;
    for (unsigned n = 0; n < numPasses; ++n)
      if (!waits[n])
        sources.push_back(n);
    unsigned placed = 0;
    while (!sources.empty()) {
      unsigned n = sources.pop_back_val();
      ++placed;
      for (unsigned e = edges[n]; e < edges[n + 1]; ++e)
        if (!--waits[passLinks[e].second])
          sources.push_back(passLinks[e].second);
    }
    return placed == numPasses;
  };
  SmallVector<unsigned> byThreads(classes.size());
  for (unsigned c = 0; c < classes.size(); ++c)
    byThreads[c] = c;
  llvm::stable_sort(byThreads, [&](unsigned a, unsigned b) {
    return classes[a].threads() > classes[b].threads();
  });
  for (unsigned c : byThreads) {
    auto &own = outputs[c];
    if (own.size() < 2)
      continue;
    auto place = [&](unsigned g, unsigned l) { level[own[g]] = l; };
    // The outputs not yet placed past every level, each its own: a
    // relabeling of the levels by rank, so acyclic.
    for (unsigned g = 0; g < own.size(); ++g)
      place(g, own.size() + g);
    levels[c] = 2 * own.size();
    unsigned kept = 0;
    for (unsigned g = 0; g < own.size(); ++g) {
      bool joined = false;
      for (unsigned l = 0; l < kept && !joined; ++l) {
        place(g, l);
        joined = acyclic();
      }
      if (!joined)
        place(g, kept++);
    }
    levels[c] = kept;
  }
  for (unsigned c = 0; c < classes.size(); ++c)
    if (ready[c])
      writes.passes[c] = std::max(1u, levels[c]);
}

void StaticFolder::frameGraph(ArrayRef<bool> ready, ReadyWrites &writes,
                              FrameGraph &graph, bool keepWhy) {
  // Which late outputs each class writes at its ready pass: those whose
  // inputs come from constants, the netlist's inputs, early outputs and late
  // outputs written at ready passes; split by level, at which.
  auto &atReady = writes.atReady;
  atReady.assign(producers.size(), false);
  writes.level.assign(producers.size(), 0);
  for (auto [p, producer] : llvm::enumerate(producers))
    if (producer.cls >= 0 && producer.late && ready[producer.cls])
      atReady[p] = true;
  settleReady(writes);
  writes.passes.assign(classes.size(), 0);
  for (unsigned c = 0; c < classes.size(); ++c)
    writes.passes[c] = ready[c] ? 1 : 0;
  if (splitReady)
    readyLevels(ready, writes);
  writes.laterRow.assign(classes.size(), 0);
  writes.rows = classes.size() * 2;
  for (unsigned c = 0; c < classes.size(); ++c) {
    writes.laterRow[c] = writes.rows;
    if (writes.passes[c] > 1)
      writes.rows += writes.passes[c] - 1;
  }

  // The evaluations, and what each waits for within the frame.
  auto &nodes = graph.nodes;
  auto &preds = graph.preds;
  nodes.clear();
  std::map<std::tuple<unsigned, unsigned, unsigned, unsigned>, unsigned>
      nodeIndex;
  for (auto [c, cls] : llvm::enumerate(classes))
    for (unsigned i = 0; i < cls.threads(); ++i)
      for (auto pass : {kReady, kState})
        for (unsigned l = 0; l < (pass == kReady ? writes.passes[c] : 1);
             ++l) {
          nodeIndex[{c, i, pass, l}] = nodes.size();
          nodes.push_back({unsigned(c), i, pass, l});
        }
  preds.assign(nodes.size(), {});
  graph.why.assign(keepWhy ? nodes.size() : 0, {});
  for (auto [n, node] : llvm::enumerate(nodes)) {
    auto &cls = classes[node.cls];
    if (node.pass == kState)
      for (unsigned l = 0; l < writes.passes[node.cls]; ++l) {
        preds[n].push_back({nodeIndex[{node.cls, node.inst, kReady, l}], 1});
        if (keepWhy)
          graph.why[n].push_back({});
      }
    // The inputs this pass reads: all of them for the state pass, those the
    // outputs it writes depend on for a ready pass.
    llvm::BitVector reads(cls.inputNames.size(), node.pass == kState);
    if (node.pass == kReady)
      for (auto [o, late] : llvm::enumerate(cls.late)) {
        unsigned p = cls.producerOf[o];
        if (atReady[p] && writes.level[p] == node.level)
          reads |= cls.deps[o];
      }
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
        auto pass = writes.passOf(piece.producer);
        unsigned l = pass == kReady ? writes.level[piece.producer] : 0;
        preds[n].push_back(
            {nodeIndex[{unsigned(producer.cls), piece.inst, pass, l}],
             zeroLink(piece.producer, node.cls, input)
                 ? 0
                 : linkLatency + crossDelay(producer.cls, node.cls)});
        if (keepWhy)
          graph.why[n].push_back({piece.producer, unsigned(input)});
      }
    }
  }
}

/// The slots of the evaluations `nodes` placed at `slotOf`, by row (`ReadyWrites::row`), each instance's.
static void assignSlots(const ReadyWrites &writes, ArrayRef<Node> nodes,
                        ArrayRef<int> slotOf,
                        SmallVectorImpl<SmallVector<int>> &slots,
                        ArrayRef<Class> classes) {
  slots.assign(writes.rows, {});
  for (auto [c, cls] : llvm::enumerate(classes)) {
    for (auto pass : {kReady, kState})
      slots[c * 2 + pass].assign(cls.threads(), -1);
    for (unsigned l = 1; l < writes.passes[c]; ++l)
      slots[writes.row(c, kReady, l)].assign(cls.threads(), -1);
  }
  for (auto [n, node] : llvm::enumerate(nodes))
    slots[writes.row(node.cls, node.pass, node.level)][node.inst] = slotOf[n];
}

/// An instance's first evaluation of the frame.
static int firstSlot(const ReadyWrites &writes,
                     ArrayRef<SmallVector<int>> slots, unsigned c, unsigned i) {
  int at = slots[c * 2 + kState][i];
  for (unsigned l = 0; l < writes.passes[c]; ++l)
    at = std::min(at, slots[writes.row(c, kReady, l)][i]);
  return at;
}

/// The row of the slots of the pass producer `p` is written at.
static unsigned writeRow(const ReadyWrites &writes, const Producer &producer,
                         unsigned p) {
  return writes.atReady[p]
             ? writes.row(producer.cls, kReady, writes.level[p])
             : writes.row(producer.cls, kState);
}

unsigned StaticFolder::place(ArrayRef<bool> ready,
                             SmallVectorImpl<SmallVector<int>> &slots,
                             ReadyWrites &writes) {
  FrameGraph graph;
  frameGraph(ready, writes, graph, /*keepWhy=*/false);
  auto &nodes = graph.nodes;
  auto &preds = graph.preds;

  // Longest path to a sink, as the priority; a cycle fails. Depth first
  // without recursion: copies make thousands of evaluations.
  SmallVector<int> height(nodes.size(), -1), state(nodes.size(), 0);
  std::vector<SmallVector<std::pair<unsigned, int>>> succs(nodes.size());
  for (unsigned n = 0; n < nodes.size(); ++n)
    for (auto [p, latency] : preds[n])
      succs[p].push_back({n, latency});
  for (unsigned root = 0; root < nodes.size(); ++root) {
    if (state[root])
      continue;
    // Each entry: an evaluation and the next of its successors to visit.
    SmallVector<std::pair<unsigned, unsigned>> stack{{root, 0}};
    state[root] = 1;
    while (!stack.empty()) {
      auto &[n, next] = stack.back();
      if (next < succs[n].size()) {
        unsigned s = succs[n][next++].first;
        if (state[s] == 1)
          return 0;
        if (state[s] == 0) {
          state[s] = 1;
          stack.push_back({s, 0});
        }
        continue;
      }
      int h = 0;
      for (auto [s, latency] : succs[n])
        h = std::max(h, height[s] + latency);
      height[n] = h;
      state[n] = 2;
      stack.pop_back();
    }
  }

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
  // A paged copy's state passes, each at least its page apart: the page
  // port reads an instance's pages in the slots before its state pass.
  std::vector<std::set<int>> pageSlots(classes.size());
  auto pageClash = [&](unsigned c, int t) {
    int p = classes[c].page;
    auto &slots = pageSlots[c];
    auto it = slots.lower_bound(t - p + 1);
    return it != slots.end() && *it < t + p;
  };
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), cmp);
    unsigned n = heap.back();
    heap.pop_back();
    int earliest = 0;
    for (auto [p, latency] : preds[n])
      earliest = std::max(earliest, slotOf[p] + latency);
    unsigned c = nodes[n].cls;
    bool pages = nodes[n].pass == kState && paged(c);
    auto &taken = busy[c];
    while (taken.count(earliest) || (pages && pageClash(c, earliest)))
      ++earliest;
    taken.insert(earliest);
    if (pages)
      pageSlots[c].insert(earliest);
    slotOf[n] = earliest;
    for (auto [s, latency] : succs[n])
      if (!--waiting[s]) {
        heap.push_back(s);
        std::push_heap(heap.begin(), heap.end(), cmp);
      }
  }

  assignSlots(writes, nodes, slotOf, slots, classes);
  int length = 0;
  for (unsigned n = 0; n < nodes.size(); ++n)
    length = std::max(length, slotOf[n] + 1);

  // Across frames: an instance's next read of its state after its write (a
  // threaded class's registers written a host cycle late, from a stage); an
  // early output's reads in the next frame after its write.
  auto first = [&](unsigned c, unsigned i) {
    return firstSlot(writes, slots, c, i);
  };
  for (auto [c, cls] : llvm::enumerate(classes))
    for (unsigned i = 0; i < cls.threads(); ++i) {
      int s = slots[c * 2 + kState][i];
      length = std::max(length, s - first(c, i) + minStateGap(cls));
    }
  // A paged copy: the next frame's first page-in after this frame's last
  // state pass, and an instance's next page-in after its write-back.
  for (auto [c, cls] : llvm::enumerate(classes))
    if (paged(c) && !pageSlots[c].empty())
      length = std::max<int>(
          {length,
           *pageSlots[c].rbegin() - *pageSlots[c].begin() + int(cls.page),
           int(cls.page) + 4});
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
                                      linkLatency +
                                      crossDelay(producer.cls, consumer.cls));
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
      at = std::max(at, slots[writeRow(writes, producer, piece.producer)][piece.inst] +
                            linkLatency - 1);
    }
    outputSlot[k] = at;
    if (envSink[k]) {
      length = std::max(length, at + 1);
    } else if (outputEarly[k]) {
      length = std::max(length, at + 1);
      length = std::max(length, (at + envRoundTrip + 1) / 2);
    } else if (specSelect[k] >= 0) {
      // Captured before the next frame's load, and the environment's round
      // trip within the frames its offers span and the next one.
      length = std::max(length, at + 2);
      length = std::max(length, (at + envRoundTrip + int(specDepth)) /
                                    int(specDepth + 1));
    } else {
      length = std::max(length, at + envRoundTrip);
    }
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
    unsigned period = placeModulo(nodes, preds, succs, height, writes, overlapped);
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
    const ReadyWrites &writes, SmallVectorImpl<SmallVector<int>> &slots) {
  auto &atReady = writes.atReady;
  // No frame is shorter than a copy's evaluations of it.
  SmallVector<unsigned> evaluations(classes.size(), 0);
  for (auto &node : nodes)
    ++evaluations[node.cls];
  unsigned lower = 2;
  for (auto count : evaluations)
    lower = std::max(lower, count);
  // Nor than a paged copy's state passes, each its page apart, nor its
  // page recurrence.
  for (auto [c, cls] : llvm::enumerate(classes))
    if (paged(c) && cls.threads())
      lower = std::max({lower, cls.page * cls.threads(), cls.page + 4});
  auto cmp = [&](unsigned a, unsigned b) {
    if (height[a] != height[b])
      return height[a] < height[b];
    return a > b;
  };
  SmallVector<bool> feedsOutput(producers.size(), false);
  // The netlist outputs each producer feeds, for its slack.
  SmallVector<SmallVector<unsigned>> outputsFed(producers.size());
  for (auto [k, consumer] : llvm::enumerate(consumers))
    if (consumer.cls < 0)
      for (auto &piece : consumer.pieces[0])
        if (piece.producer >= 0) {
          feedsOutput[piece.producer] = true;
          outputsFed[piece.producer].push_back(k);
        }
  auto outputSlack = [&](unsigned port, bool late, int period) {
    int slack = 1;
    for (unsigned k : outputsFed[port])
      slack = std::max(slack, envSlackOf(k, late, period));
    return slack;
  };
  for (unsigned period = lower; period <= 4 * lower + 64; ++period) {
    // List scheduling, a copy's slots taken modulo the frame; an evaluation
    // runs at most one frame late.
    SmallVector<int> slotOf(nodes.size(), -1);
    std::vector<std::set<int>> busy(classes.size());
    // A paged copy's page port, modulo the frame: each state pass holds the
    // slots of its page-in, the page before it and its own.
    std::vector<std::set<int>> pagePort(classes.size());
    auto window = [&](unsigned c, int t, auto &&each) {
      for (int k = 0; k < int(classes[c].page); ++k)
        each(((t - k) % int(period) + int(period)) % int(period));
    };
    auto pageFree = [&](unsigned c, int t) {
      bool free = true;
      window(c, t, [&](int r) { free &= !pagePort[c].count(r); });
      return free;
    };
    auto pageTake = [&](unsigned c, int t, bool take) {
      window(c, t, [&](int r) {
        if (take)
          pagePort[c].insert(r);
        else
          pagePort[c].erase(r);
      });
    };
    auto pages = [&](unsigned n) {
      return nodes[n].pass == kState && paged(nodes[n].cls);
    };
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
      unsigned c = nodes[n].cls;
      auto &taken = busy[c];
      while (earliest < int(2 * period) &&
             (taken.count(earliest % period) ||
              (pages(n) && !pageFree(c, earliest))))
        ++earliest;
      if (earliest >= int(2 * period)) {
        fits = false;
        break;
      }
      taken.insert(earliest % period);
      if (pages(n))
        pageTake(c, earliest, true);
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
        bool at = producer.late && atReady[port]
                      ? node.pass == kReady && node.level == writes.level[port]
                      : node.pass == kState;
        if (!feedsOutput[port] || !at)
          continue;
        latest = std::min<int>(latest, int(period) - linkLatency + 1 -
                                           outputSlack(port, producer.late, period));
      }
      if (latest == INT_MAX)
        continue;
      auto &taken = busy[node.cls];
      taken.erase(slotOf[n] % period);
      if (pages(n))
        pageTake(node.cls, slotOf[n], false);
      int at = slotOf[n];
      for (int t = latest; t > slotOf[n]; --t)
        if (!taken.count(t % period) && (!pages(n) || pageFree(node.cls, t))) {
          at = t;
          break;
        }
      taken.insert(at % period);
      if (pages(n))
        pageTake(node.cls, at, true);
      slotOf[n] = at;
    }

    assignSlots(writes, nodes, slotOf, slots, classes);
    auto first = [&](unsigned c, unsigned i) {
      return firstSlot(writes, slots, c, i);
    };
    auto last = [&](unsigned c, unsigned i) {
      int at = slots[c * 2 + kState][i];
      for (unsigned l = 0; l < writes.passes[c]; ++l)
        at = std::max(at, slots[writes.row(c, kReady, l)][i]);
      return at;
    };
    bool ok = true;
    // An instance's next frame reads its state after this one's write.
    for (auto [c, cls] : llvm::enumerate(classes))
      for (unsigned i = 0; ok && i < cls.threads(); ++i) {
        ok = slots[c * 2 + kState][i] - first(c, i) + minStateGap(cls) <=
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
          if (written - first(consumer.cls, i) + linkLatency +
                      crossDelay(producer.cls, consumer.cls) >
                  int(period) ||
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
        at = std::max(at, slots[writeRow(writes, producer, piece.producer)][piece.inst] +
                              linkLatency - 1);
      }
      outputSlot[k] = at;
      if (at + envSlack(k, period) > int(period))
        ok = false;
    }
    if (ok)
      return period;
  }
  return 0;
}

/// The bits of a model's output `o` that depend combinationally on its input
/// `input`, as `[low, high)` runs: the output split along its wiring
/// (concatenations, extracts, bitcasts, array and struct creation), each
/// piece's logic searched up to registers, registered reads and black boxes.
static SmallVector<std::pair<unsigned, unsigned>>
outputBitsOn(hw::HWModuleOp model, unsigned o, unsigned input) {
  auto *body = model.getBodyBlock();
  Value root = body->getTerminator()->getOperand(o);
  // Whether a value reaches the input: depth first, without recursion.
  DenseMap<Value, bool> reaches;
  auto search = [&](Value start) {
    SmallVector<std::pair<Value, bool>> stack{{start, false}};
    while (!stack.empty()) {
      auto [value, expanded] = stack.pop_back_val();
      if (!expanded && reaches.count(value))
        continue;
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        reaches[value] = arg.getArgNumber() == input;
        continue;
      }
      auto *op = value.getDefiningOp();
      SmallVector<Value> operands;
      if (isa<seq::FirRegOp, hw::ConstantOp, hw::AggregateConstantOp,
              hw::InstanceOp>(op)) {
        reaches[value] = false;
        continue;
      }
      if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
        auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
        if (memory && memory.getReadLatency() > 0) {
          reaches[value] = false;
          continue;
        }
        operands.push_back(read.getAddress());
        if (read.getEnable())
          operands.push_back(read.getEnable());
      } else {
        operands.append(op->operand_begin(), op->operand_end());
      }
      if (!expanded) {
        reaches[value] = false;
        stack.push_back({value, true});
        for (auto operand : operands)
          if (!reaches.count(operand))
            stack.push_back({operand, false});
        continue;
      }
      reaches[value] = llvm::any_of(
          operands, [&](Value operand) { return reaches.lookup(operand); });
    }
    return reaches.lookup(start);
  };
  // The output's pieces: a value's bits [low, low + width) at output bit at.
  SmallVector<std::pair<unsigned, unsigned>> runs;
  struct Item {
    Value value;
    unsigned low, width, at;
  };
  SmallVector<Item> work{
      {root, 0, unsigned(hw::getBitWidth(root.getType())), 0}};
  auto split = [&](const Item &item, ValueRange operands) {
    // The first operand the most significant.
    unsigned base = 0;
    for (auto operand : llvm::reverse(operands)) {
      unsigned w = hw::getBitWidth(operand.getType());
      unsigned lo = std::max(item.low, base);
      unsigned hi = std::min(item.low + item.width, base + w);
      if (lo < hi)
        work.push_back({operand, lo - base, hi - lo, item.at + lo - item.low});
      base += w;
    }
  };
  while (!work.empty()) {
    auto item = work.pop_back_val();
    auto *op = item.value.getDefiningOp();
    if (auto concat = dyn_cast_or_null<comb::ConcatOp>(op)) {
      split(item, concat.getInputs());
      continue;
    }
    if (auto create = dyn_cast_or_null<hw::ArrayCreateOp>(op)) {
      split(item, create.getInputs());
      continue;
    }
    if (auto create = dyn_cast_or_null<hw::StructCreateOp>(op)) {
      split(item, create.getInput());
      continue;
    }
    if (auto extract = dyn_cast_or_null<comb::ExtractOp>(op)) {
      work.push_back({extract.getInput(), item.low + extract.getLowBit(),
                      item.width, item.at});
      continue;
    }
    if (auto bitcast = dyn_cast_or_null<hw::BitcastOp>(op)) {
      work.push_back({bitcast.getInput(), item.low, item.width, item.at});
      continue;
    }
    if (search(item.value))
      runs.push_back({item.at, item.at + item.width});
  }
  llvm::sort(runs);
  SmallVector<std::pair<unsigned, unsigned>> merged;
  for (auto run : runs)
    if (!merged.empty() && merged.back().second >= run.first)
      merged.back().second = std::max(merged.back().second, run.second);
    else
      merged.push_back(run);
  return merged;
}

/// Bit runs as `[hi:lo]` lists, the first few.
static std::string printRuns(ArrayRef<std::pair<unsigned, unsigned>> runs,
                             unsigned width) {
  std::string text;
  llvm::raw_string_ostream os(text);
  if (runs.size() == 1 && runs[0].first == 0 && runs[0].second == width)
    return "all " + std::to_string(width) + " bits";
  unsigned bits = 0;
  for (auto [lo, hi] : runs)
    bits += hi - lo;
  os << bits << " of " << width << " bits:";
  for (auto [n, run] : llvm::enumerate(runs)) {
    if (n == 6) {
      os << " ...";
      break;
    }
    os << " [" << run.second - 1;
    if (run.second - 1 != run.first)
      os << ":" << run.first;
    os << "]";
  }
  return text;
}

void StaticFolder::explainCycle(ArrayRef<bool> ready,
                                InFlightDiagnostic &diag) {
  ReadyWrites writes;
  FrameGraph graph;
  frameGraph(ready, writes, graph, /*keepWhy=*/true);
  auto &atReady = writes.atReady;
  auto &nodes = graph.nodes;
  auto &preds = graph.preds;
  unsigned numNodes = nodes.size();
  // Each edge by its head: (the evaluation that waits, its wait's index).
  std::vector<SmallVector<std::pair<unsigned, unsigned>>> succs(numNodes);
  for (unsigned n = 0; n < numNodes; ++n)
    for (auto [k, pred] : llvm::enumerate(preds[n]))
      succs[pred.first].push_back({n, unsigned(k)});

  // A shortest cycle: breadth first from each evaluation back to itself.
  SmallVector<std::pair<unsigned, unsigned>> cycle;
  SmallVector<unsigned> seen(numNodes, 0), dist(numNodes, 0);
  SmallVector<std::pair<unsigned, unsigned>> via(numNodes);
  for (unsigned root = 0; root < numNodes; ++root) {
    std::deque<unsigned> queue{root};
    seen[root] = root + 1;
    dist[root] = 0;
    bool found = false;
    while (!queue.empty() && !found) {
      unsigned u = queue.front();
      queue.pop_front();
      if (!cycle.empty() && dist[u] + 1 >= cycle.size())
        break;
      for (auto [v, k] : succs[u]) {
        if (v == root) {
          cycle.assign({{v, k}});
          for (unsigned w = u; w != root; w = preds[w][via[w].second].first)
            cycle.push_back(via[w]);
          std::reverse(cycle.begin(), cycle.end());
          found = true;
          break;
        }
        if (seen[v] == root + 1)
          continue;
        seen[v] = root + 1;
        dist[v] = dist[u] + 1;
        via[v] = {v, k};
        queue.push_back(v);
      }
    }
  }
  if (cycle.empty())
    return;

  auto nodeName = [&](unsigned n) {
    auto &node = nodes[n];
    auto &cls = classes[node.cls];
    std::string pass = node.pass == kReady ? "ready pass" : "state pass";
    if (node.pass == kReady && writes.passes[node.cls] > 1)
      pass = "ready pass " + std::to_string(node.level + 1) + " of " +
             std::to_string(writes.passes[node.cls]);
    return (cls.name + " #" + std::to_string(node.inst) + " (" +
            cls.instances[node.inst].getInstanceName() + ") " + pass)
        .str();
  };
  auto outputName = [&](int p) {
    auto &producer = producers[p];
    return "'" + classes[producer.cls].outputNames[producer.port] + "'";
  };
  auto inputName = [&](unsigned c, unsigned input) {
    return "'" + classes[c].inputNames[input] + "'";
  };
  // The first input of a late output fed by a late output not written at a
  // ready pass: what keeps it from its class's ready pass.
  auto blocker = [&](int p) -> std::pair<int, int> {
    auto &producer = producers[p];
    auto &cls = classes[producer.cls];
    for (auto input : cls.deps[producer.port].set_bits()) {
      int k = cls.consumerOf[input];
      if (k < 0)
        continue;
      for (auto &pieces : consumers[k].pieces)
        for (auto &piece : pieces)
          if (piece.producer >= 0 && producers[piece.producer].cls >= 0 &&
              producers[piece.producer].late && !atReady[piece.producer])
            return {int(input), piece.producer};
    }
    return {-1, -1};
  };

  unsigned numReady = llvm::count(ready, true);
  diag << "\n  a shortest cycle, " << cycle.size() << " edge"
       << (cycle.size() == 1 ? "" : "s") << ", with " << numReady << " of "
       << classes.size() << " models given a ready pass (every model with "
       << "late outputs: the most outputs written at ready passes)";
  if (splitReady) {
    diag << ", their ready outputs split by level";
    unsigned shown = 0;
    for (auto [c, cls] : llvm::enumerate(classes))
      if (writes.passes[c] > 1)
        diag << (shown++ ? ", " : " (") << cls.name << ": "
             << writes.passes[c] << " ready passes";
    diag << (shown ? ")" : " (one ready pass each)");
  }
  diag << ":";
  for (auto [e, edge] : llvm::enumerate(cycle)) {
    auto [v, k] = edge;
    unsigned u = preds[v][k].first;
    int latency = preds[v][k].second;
    auto &wait = graph.why[v][k];
    auto &tail = nodes[u];
    auto &head = nodes[v];
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "cycle edge " << e + 1 << " of " << cycle.size() << ": "
       << nodeName(u) << " -> " << nodeName(v) << ": ";
    if (wait.producer < 0) {
      os << "an instance's ready pass precedes its state pass";
      diag.attachNote(classes[head.cls].instances[head.inst].getLoc()) << text;
      continue;
    }
    auto &consumerCls = classes[head.cls];
    os << "output " << outputName(wait.producer) << " -> input "
       << inputName(head.cls, wait.input) << " ("
       << (latency == 0   ? std::string("zero-latency link")
           : latency == 1 ? std::string("1 host cycle")
                          : std::to_string(latency) + " host cycles")
       << ")";
    // Which bits of the input the producer's output drives.
    int k2 = consumerCls.consumerOf[wait.input];
    unsigned bits = 0;
    for (auto &piece : consumers[k2].pieces[head.inst])
      if (piece.producer == wait.producer && piece.inst == tail.inst)
        bits += piece.width;
    os << ", " << bits << " of the input's " << consumers[k2].width
       << " bits.";
    // Why the producer writes it within the frame: what makes it late.
    {
      auto &producerCls = classes[tail.cls];
      auto &deps = producerCls.deps[producers[wait.producer].port];
      os << " " << outputName(wait.producer) << " is late: ";
      if (deps.none()) {
        os << "it depends on a combinational memory read or a black box, no "
              "input.";
      } else {
        os << "it depends combinationally on " << classes[tail.cls].name
           << "'s input" << (deps.count() == 1 ? "" : "s");
        unsigned shown = 0;
        for (auto input : deps.set_bits())
          if (shown++ < 4)
            os << " " << inputName(tail.cls, input);
        if (shown > 4)
          os << " and " << shown - 4 << " more";
        os << ".";
      }
    }
    // Why the consumer's pass reads it.
    if (head.pass == kState) {
      os << " A state pass reads every input.";
    } else {
      os << " The ready pass reads it for its outputs";
      unsigned shown = 0;
      for (auto [o, deps] : llvm::enumerate(consumerCls.deps))
        if (atReady[consumerCls.producerOf[o]] &&
            writes.level[consumerCls.producerOf[o]] == head.level &&
            deps.test(wait.input) && shown++ < 4)
          os << " '" << consumerCls.outputNames[o] << "'";
      if (shown > 4)
        os << " and " << shown - 4 << " more";
      os << ".";
    }
    // Whether the producer's pass could be a ready pass.
    if (tail.pass == kState) {
      if (!ready[tail.cls]) {
        os << " Breakable: " << outputName(wait.producer)
           << " could be written at a ready pass of " << classes[tail.cls].name
           << ", which this choice does not give it.";
      } else {
        auto [input, from] = blocker(wait.producer);
        os << " Not broken by " << classes[tail.cls].name
           << "'s ready pass: " << outputName(wait.producer)
           << " depends combinationally on input "
           << (input >= 0 ? inputName(tail.cls, input) : "?") << ", fed by "
           << (from >= 0 ? classes[producers[from].cls].name + " output " +
                               outputName(from)
                         : "?")
           << ", which is written at a state pass.";
      }
    }
    // Through the consumer: whether the output the cycle leaves by depends
    // on this input.
    auto [nv, nk] = cycle[(e + 1) % cycle.size()];
    auto &next = graph.why[nv][nk];
    if (head.pass == kReady && next.producer >= 0) {
      auto &out = producers[next.producer];
      bool through = consumerCls.deps[out.port].test(wait.input);
      if (through) {
        auto runs = outputBitsOn(consumerCls.model, out.port, wait.input);
        os << " Through " << consumerCls.name << ": its output "
           << outputName(next.producer) << " depends combinationally on "
           << inputName(head.cls, wait.input) << " ("
           << printRuns(runs, out.width)
           << " of the output): a combinational path through the model.";
      } else {
        os << " Through " << consumerCls.name << ": its output "
           << outputName(next.producer) << " does not depend on "
           << inputName(head.cls, wait.input)
           << "; the cycle closes only because one ready pass writes all of "
           << consumerCls.name << "'s ready outputs (a second ready pass "
           << "would break it).";
      }
    }
    diag.attachNote(classes[head.cls].instances[head.inst].getLoc()) << text;
  }

  // Whether ready passes split by output would do: each instance's late
  // output written once the late outputs feeding the inputs it depends on
  // are. A cycle left then is a combinational loop through the ports.
  std::map<std::pair<unsigned, unsigned>, unsigned> outNode;
  SmallVector<std::pair<unsigned, unsigned>> outs;
  for (auto [p, producer] : llvm::enumerate(producers))
    if (producer.cls >= 0 && producer.late)
      for (unsigned i = 0; i < classes[producer.cls].threads(); ++i) {
        outNode[{unsigned(p), i}] = outs.size();
        outs.push_back({unsigned(p), i});
      }
  // Per output node, the output nodes it waits for and through which input.
  std::vector<SmallVector<std::pair<unsigned, unsigned>>> waits(outs.size());
  std::vector<SmallVector<unsigned>> feeds(outs.size());
  for (auto [n, out] : llvm::enumerate(outs)) {
    auto [p, i] = out;
    auto &cls = classes[producers[p].cls];
    std::set<unsigned> from;
    for (auto input : cls.deps[producers[p].port].set_bits()) {
      int k = cls.consumerOf[input];
      if (k < 0)
        continue;
      for (auto &piece : consumers[k].pieces[i])
        if (piece.producer >= 0 && producers[piece.producer].cls >= 0 &&
            producers[piece.producer].late) {
          unsigned m = outNode[{unsigned(piece.producer), piece.inst}];
          if (from.insert(m).second) {
            waits[n].push_back({m, unsigned(input)});
            feeds[m].push_back(n);
          }
        }
    }
  }
  // A cycle, depth first without recursion, then a shortest one through it.
  SmallVector<unsigned> color(outs.size(), 0), parent(outs.size(), 0);
  SmallVector<unsigned> loop;
  for (unsigned root = 0; root < outs.size() && loop.empty(); ++root) {
    if (color[root])
      continue;
    SmallVector<std::pair<unsigned, unsigned>> stack{{root, 0}};
    color[root] = 1;
    while (!stack.empty() && loop.empty()) {
      auto &[n, next] = stack.back();
      if (next < feeds[n].size()) {
        unsigned s = feeds[n][next++];
        if (color[s] == 1) {
          for (unsigned w = n; w != s; w = parent[w])
            loop.push_back(w);
          loop.push_back(s);
        } else if (!color[s]) {
          color[s] = 1;
          parent[s] = n;
          stack.push_back({s, 0});
        }
        continue;
      }
      color[n] = 2;
      stack.pop_back();
    }
  }
  if (loop.empty()) {
    diag.attachNote(netlist.getLoc())
        << "with a ready pass per late output instead (each written once the "
           "outputs feeding the inputs it depends on are), the frame has no "
           "cycle: no path through the ports is combinational, so the netlist "
           "is legal and the schedule needs the models on the cycle to write "
           "their ready outputs in more than one pass";
    return;
  }
  // Shortest, through the found loop's outputs: breadth first back to each.
  SmallVector<unsigned> best;
  SmallVector<int> back(outs.size(), -1);
  for (unsigned root : loop) {
    std::fill(back.begin(), back.end(), -1);
    std::deque<unsigned> queue{root};
    back[root] = root;
    while (!queue.empty()) {
      unsigned u = queue.front();
      queue.pop_front();
      bool closed = false;
      for (unsigned v : feeds[u]) {
        if (v == root) {
          SmallVector<unsigned> path;
          for (unsigned w = u; w != root; w = back[w])
            path.push_back(w);
          path.push_back(root);
          std::reverse(path.begin(), path.end());
          if (best.empty() || path.size() < best.size())
            best = path;
          closed = true;
          break;
        }
        if (back[v] < 0) {
          back[v] = u;
          queue.push_back(v);
        }
      }
      if (closed)
        break;
    }
  }
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "even with a ready pass per late output the frame has a cycle: a "
        "combinational loop through the ports, "
     << best.size() << " output" << (best.size() == 1 ? "" : "s") << ":";
  for (auto [e, n] : llvm::enumerate(best)) {
    unsigned m = best[(e + 1) % best.size()];
    auto [p, i] = outs[n];
    auto &to = classes[producers[outs[m].first].cls];
    unsigned input = 0;
    for (auto [w, in] : waits[m])
      if (w == n)
        input = in;
    os << " " << classes[producers[p].cls].name << " #" << i << " output "
       << outputName(p) << " -> " << to.name << " #" << outs[m].second
       << " input " << inputName(producers[outs[m].first].cls, input)
       << ", on which its output " << outputName(outs[m].first)
       << " depends;";
  }
  diag.attachNote(netlist.getLoc()) << text;
}

StaticFolder::Choice StaticFolder::searchReady() {
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
  Choice best;
  auto tryReady = [&](const SmallVector<bool> &ready) {
    SmallVector<SmallVector<int>> slots;
    ReadyWrites writes;
    unsigned length = place(ready, slots, writes);
    if (!length || (best.length && length >= best.length))
      return false;
    best.length = length;
    best.ready = ready;
    best.slots = std::move(slots);
    best.writes = std::move(writes);
    best.outputSlot = outputSlot;
    best.outputEarly = outputEarly;
    return true;
  };
  auto search = [&] {
    for (uint64_t mask = 0; mask < (uint64_t(1) << candidates.size());
         ++mask) {
      SmallVector<bool> ready(classes.size(), false);
      for (auto [k, c] : llvm::enumerate(candidates))
        ready[c] = (mask >> k) & 1;
      if (alike)
        for (auto [c, cls] : llvm::enumerate(classes))
          if (cls.splitFrom >= 0)
            ready[c] = ready[cls.splitFrom];
      tryReady(ready);
    }
    for (bool shorter = alike && best.length; shorter;) {
      shorter = false;
      for (auto [c, cls] : llvm::enumerate(classes)) {
        if (!late(cls))
          continue;
        auto ready = best.ready;
        ready[c] = !ready[c];
        shorter |= tryReady(ready);
      }
    }
  };
  // With `auto`, only where one ready pass per class leaves a cycle for every
  // choice are a class's ready outputs split by level: a schedule found
  // without is kept as it is (for each assignment of SLR groups alike).
  // `always` and `rank` split them from the start (schedule).
  search();
  if (!best.length && readyLevelsMode == "auto" && !splitReady) {
    splitReady = true;
    search();
    // Split for this choice only; left split when nothing schedules, so the
    // cycle is named for the split passes (explainCycle).
    if (best.length)
      splitReady = false;
  }
  return best;
}


void StaticFolder::linkBits(SmallVectorImpl<SmallVector<uint64_t>> &bits,
                            SmallVectorImpl<SmallVector<unsigned>> &links) const {
  unsigned n = classes.size();
  bits.assign(n, SmallVector<uint64_t>(n, 0));
  links.assign(n, SmallVector<unsigned>(n, 0));
  for (auto &consumer : consumers) {
    if (consumer.cls < 0)
      continue;
    // The producer bits this input reads, of any instance: one link each
    // producer output, holding every producer instance.
    std::map<unsigned, std::set<unsigned>> read;
    for (auto &pieces : consumer.pieces)
      for (auto &piece : pieces)
        if (piece.producer >= 0 && producers[piece.producer].cls >= 0)
          for (unsigned bit = piece.low; bit < piece.low + piece.width; ++bit)
            read[piece.producer].insert(bit);
    for (auto &[p, set] : read) {
      unsigned from = producers[p].cls;
      if (from == unsigned(consumer.cls))
        continue;
      bits[from][consumer.cls] += set.size();
      ++links[from][consumer.cls];
    }
  }
}

LogicalResult StaticFolder::chooseGroups() {
  unsigned n = classes.size();
  auto classNamed = [&](StringRef name) -> int {
    for (auto [c, cls] : llvm::enumerate(classes))
      if (cls.name == name)
        return c;
    return -1;
  };
  slrSites.assign(n, 0);
  for (auto &spec : slrSiteSpecs) {
    auto [name, count] = StringRef(spec).rsplit(':');
    int c = classNamed(name);
    uint64_t sites = 0;
    if (c < 0 || count.getAsInteger(10, sites))
      return netlist.emitError("slr-sites: '") << spec
             << "' is not <class>:<sites> of a class";
    slrSites[c] = sites;
  }
  SmallVector<int> pinned(n, -1);
  for (auto &spec : slrAssignSpecs) {
    auto [name, group] = StringRef(spec).rsplit(':');
    int c = classNamed(name);
    unsigned g = 0;
    if (c < 0 || group.getAsInteger(10, g) || g >= slrGroups)
      return netlist.emitError("slr-assign: '") << spec
             << "' is not <class>:<group> of a class and a group below "
             << slrGroups;
    pinned[c] = g;
  }
  SmallVector<SmallVector<uint64_t>> bits;
  SmallVector<SmallVector<unsigned>> links;
  linkBits(bits, links);

  // The assignments within the capacity with the fewest bits between groups,
  // by branch and bound over the classes, the largest first; a group is
  // opened in order (the groups are alike) unless some class is pinned.
  constexpr unsigned kKept = 64;
  SmallVector<unsigned> order;
  for (unsigned c = 0; c < n; ++c)
    order.push_back(c);
  llvm::stable_sort(order, [&](unsigned x, unsigned y) {
    return slrSites[x] > slrSites[y];
  });
  bool anyPinned = llvm::any_of(pinned, [](int g) { return g >= 0; });
  std::vector<std::pair<uint64_t, SmallVector<int>>> kept;
  SmallVector<int> group(n, -1);
  SmallVector<uint64_t> used(slrGroups, 0);
  uint64_t visited = 0;
  std::function<void(unsigned, unsigned, uint64_t)> visit =
      [&](unsigned k, unsigned opened, uint64_t cut) {
        if (++visited > 20'000'000 ||
            (kept.size() >= kKept && cut >= kept.back().first))
          return;
        if (k == n) {
          auto at = llvm::upper_bound(kept, cut, [](uint64_t v, auto &e) {
            return v < e.first;
          });
          kept.insert(at, {cut, group});
          if (kept.size() > kKept)
            kept.pop_back();
          return;
        }
        unsigned c = order[k];
        unsigned limit = anyPinned ? slrGroups : std::min(slrGroups, opened + 1);
        for (unsigned g = 0; g < limit; ++g) {
          if (pinned[c] >= 0 && int(g) != pinned[c])
            continue;
          if (slrCapacity && used[g] + slrSites[c] > slrCapacity)
            continue;
          uint64_t more = 0;
          for (unsigned j = 0; j < k; ++j) {
            unsigned d = order[j];
            if (group[d] != int(g))
              more += bits[c][d] + bits[d][c];
          }
          group[c] = g;
          used[g] += slrSites[c];
          visit(k + 1, std::max(opened, g + 1), cut + more);
          used[g] -= slrSites[c];
          group[c] = -1;
        }
      };
  visit(0, 0, 0);
  if (kept.empty())
    return netlist.emitError("slr-groups: no assignment of the classes to ")
           << slrGroups << " groups of at most " << slrCapacity << " sites";

  // Of those, the shortest frame, then the fewest bits between groups.
  unsigned bestLength = 0;
  SmallVector<int> best;
  for (auto &entry : kept) {
    slrOf = entry.second;
    unsigned length = searchReady().length;
    if (length && (!bestLength || length < bestLength)) {
      bestLength = length;
      best = entry.second;
    }
  }
  if (!bestLength)
    return netlist.emitError("the evaluations of a frame depend on each other "
                             "in a cycle for every assignment of SLR groups");
  slrOf = best;
  return success();
}

void StaticFolder::readyRegisters(const Class &cls,
                                  DenseSet<Operation *> &regs) const {
  auto *output = const_cast<hw::HWModuleOp &>(cls.model).getBodyBlock()->getTerminator();
  SmallVector<Value> stack;
  DenseSet<Value> seen;
  for (auto [o, operand] : llvm::enumerate(output->getOperands()))
    if (cls.late[o] && producers[cls.producerOf[o]].atReady)
      stack.push_back(operand);
  while (!stack.empty()) {
    Value value = stack.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    auto *op = value.getDefiningOp();
    if (!op || isa<hw::InstanceOp>(op))
      continue;
    if (isa<seq::FirRegOp>(op)) {
      regs.insert(op);
      continue;
    }
    if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
      auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
      if (memory && memory.getReadLatency() > 0)
        continue;
      stack.push_back(read.getAddress());
      if (read.getEnable())
        stack.push_back(read.getEnable());
      continue;
    }
    for (auto operand : op->getOperands())
      stack.push_back(operand);
  }
}

bool StaticFolder::pagesAny(const Class &cls) const {
  DenseSet<Operation *> ready;
  readyRegisters(cls, ready);
  auto *folding = cls.folding.get();
  for (auto reg : const_cast<hw::HWModuleOp &>(cls.model).getBodyBlock()->getOps<seq::FirRegOp>())
    if (!(folding && folding->physical.contains(reg)) &&
        reg.getNext() != reg.getResult() &&
        hw::getBitWidth(reg.getType()) > 0 && !ready.contains(reg))
      return true;
  return false;
}

int StaticFolder::minStateGap(const Class &cls) const {
  int gap = writeBackLatency(cls);
  if (virtualMemoryPattern.empty())
    return gap;
  llvm::Regex pattern(virtualMemoryPattern);
  auto *folding = cls.folding.get();
  for (auto memory :
       const_cast<hw::HWModuleOp &>(cls.model).getBodyBlock()->getOps<seq::FirMemOp>())
    if (!(folding && folding->physical.contains(memory)) &&
        pattern.match(memory.getName().value_or("")))
      return std::max(gap, shortLead ? 4 : 5);
  return gap;
}

LogicalResult StaticFolder::parsePages() {
  for (auto &spec : pageSpecs) {
    auto [model, count] = StringRef(spec).rsplit(':');
    unsigned p = 0;
    if (model.empty() || count.empty() || count.getAsInteger(10, p) || !p)
      return netlist.emitError("page: '")
             << spec << "' is not <model>:<p>, p at least 1";
    bool any = false;
    for (auto &cls : classes) {
      StringRef name = cls.splitFrom >= 0
                           ? StringRef(classes[cls.splitFrom].name)
                           : StringRef(cls.name);
      if (model != "*" && model != name && model != cls.name)
        continue;
      any = true;
      // A copy of one instance keeps its registers in flops: nothing pages.
      if (cls.threads() > 1)
        cls.page = p;
    }
    if (!any && model != "*")
      return netlist.emitError("page: no model '") << model << "'";
  }
  return success();
}

LogicalResult StaticFolder::schedule() {
  auto late = [](const Class &cls) {
    return llvm::any_of(cls.late, [](bool late) { return late; });
  };
  // Past a dozen classes with late outputs, the copies of a model choose
  // alike (searchReady).
  unsigned lateClasses = llvm::count_if(classes, late), lateModels = 0;
  for (auto &cls : classes)
    lateModels += late(cls) && (lateClasses <= 12 || cls.splitFrom < 0);
  if (lateModels > 12)
    return netlist.emitError("too many models with late outputs to choose "
                             "their ready passes: ")
           << lateModels;
  // With `auto`, a class's ready outputs are split by level only where one
  // ready pass per class leaves a cycle for every choice (searchReady).
  // `always` splits them from the start (a class whose levels all merge keeps
  // one ready pass), to exercise the split passes on designs that schedule
  // without; `rank` too, its levels never merged, a pass per rank of a
  // class's outputs.
  if (readyLevelsMode != "off" && readyLevelsMode != "auto" &&
      readyLevelsMode != "always" && readyLevelsMode != "rank")
    return netlist.emitError("ready-levels '")
           << readyLevelsMode << "' is none of off, auto, always and rank";
  splitReady = readyLevelsMode == "always" || readyLevelsMode == "rank";
  if (slrGroups && failed(chooseGroups()))
    return failure();
  Choice best = searchReady();
  if (!best.length) {
    // Named for the choice that writes the most outputs at ready passes:
    // every model with late outputs given one.
    SmallVector<bool> ready(classes.size(), false);
    for (auto [c, cls] : llvm::enumerate(classes))
      ready[c] = late(cls);
    auto diag = netlist.emitError("the evaluations of a frame depend on each "
                                  "other in a cycle for every choice of ready "
                                  "passes");
    explainCycle(ready, diag);
    return diag;
  }
  ii = best.length;
  outputSlot = best.outputSlot;
  outputEarly = best.outputEarly;
  for (auto [p, producer] : llvm::enumerate(producers)) {
    producer.atReady = best.writes.atReady[p];
    producer.readyLevel = best.writes.level[p];
  }
  for (auto [c, cls] : llvm::enumerate(classes)) {
    cls.hasReady = best.ready[c];
    cls.readyLevels = best.writes.passes[c];
    cls.slot[kReady] = best.slots[c * 2 + kReady];
    cls.slot[kState] = best.slots[c * 2 + kState];
    cls.laterReady.clear();
    for (unsigned l = 1; l < cls.readyLevels; ++l)
      cls.laterReady.push_back(best.slots[best.writes.row(c, kReady, l)]);
    cls.stateGap = std::numeric_limits<int>::max();
    for (unsigned i = 0; i < cls.threads(); ++i) {
      int s = cls.slot[kState][i];
      int first = firstSlot(best.writes, best.slots, c, i);
      cls.stateGap = std::min(cls.stateGap, first + int(ii) - s);
    }
  }
  // A paged class whose every register a ready pass reads (or none is
  // written) has nothing to page: scheduled again without its page port.
  bool unpaged = false;
  for (auto &cls : classes)
    if (cls.page && !pagesAny(cls)) {
      llvm::errs() << "foldsim-fold-static: " << cls.name
                   << " has no register to page (its ready passes read them "
                      "all); not paged\n";
      cls.page = 0;
      unpaged = true;
    }
  if (unpaged)
    return schedule();

  std::string text;
  llvm::raw_string_ostream os(text);
  os << "foldsim-fold-static: II = " << ii << "\n";
  if (zeroLatencyLinks) {
    // Each class input fed by a late link, and those taken in its producer's
    // host cycle.
    unsigned links = 0, zero = 0;
    for (auto &consumer : consumers) {
      if (consumer.cls < 0)
        continue;
      std::set<int> fed;
      for (auto &pieces : consumer.pieces)
        for (auto &piece : pieces)
          if (piece.producer >= 0 && producers[piece.producer].cls >= 0 &&
              producers[piece.producer].late)
            fed.insert(piece.producer);
      for (int p : fed) {
        ++links;
        zero += zeroLink(p, consumer.cls, consumer.port);
      }
    }
    os << "  zero-latency links: " << zero << " of " << links << "\n";
  }
  if (!slrOf.empty()) {
    SmallVector<SmallVector<uint64_t>> bits;
    SmallVector<SmallVector<unsigned>> links;
    linkBits(bits, links);
    uint64_t crossBits = 0, allBits = 0;
    unsigned crossLinks = 0, allLinks = 0;
    for (unsigned p = 0; p < classes.size(); ++p)
      for (unsigned c = 0; c < classes.size(); ++c) {
        allBits += bits[p][c];
        allLinks += links[p][c];
        if (crosses(p, c)) {
          crossBits += bits[p][c];
          crossLinks += links[p][c];
        }
      }
    os << "  slr groups: " << slrGroups << " of at most "
       << (slrCapacity ? std::to_string(slrCapacity) : std::string("any"))
       << " sites\n";
    for (unsigned g = 0; g < slrGroups; ++g) {
      uint64_t sites = 0;
      std::string members;
      for (auto [c, cls] : llvm::enumerate(classes))
        if (slrOf[c] == int(g)) {
          sites += slrSites[c];
          members += " " + cls.name;
        }
      if (!members.empty())
        os << "    slr" << g << " (" << sites << " sites):" << members << "\n";
    }
    os << "  slr links registered: " << crossLinks << " of " << allLinks
       << " links, " << crossBits << " of " << allBits << " bits\n";
    for (unsigned p = 0; p < classes.size(); ++p)
      for (unsigned c = 0; c < classes.size(); ++c)
        if (crosses(p, c) && links[p][c])
          os << "    " << classes[p].name << " -> " << classes[c].name << ": "
             << links[p][c] << " links, " << bits[p][c] << " bits\n";
  }
  for (auto &cls : classes) {
    os << "  " << cls.name << " x" << cls.threads()
       << (cls.readyLevels > 1
               ? " (" + std::to_string(cls.readyLevels) + " ready passes)"
           : cls.hasReady ? std::string(" (ready pass)")
                          : std::string())
       << " state gap "
       << cls.stateGap;
    if (cls.page)
      os << " page " << cls.page;
    if (nest)
      os << " depth " << cls.depth
         << (cls.parent >= 0 ? " under " + classes[cls.parent].name : "");
    os << "\n";
    for (unsigned l = 0; l < cls.readyLevels; ++l) {
      os << "    ready" << (l ? std::to_string(l + 1) : "") << ":";
      for (auto s : cls.readySlots(l))
        os << " " << s;
      os << "\n";
    }
    os << "    state:";
    for (auto s : cls.slot[kState])
      os << " " << s;
    os << "\n";
    for (auto [o, late] : llvm::enumerate(cls.late)) {
      auto &producer = producers[cls.producerOf[o]];
      if (!late || !producer.read)
        continue;
      os << "    late " << cls.outputNames[o]
         << (!producer.atReady ? std::string(" @state")
             : producer.readyLevel
                 ? " @ready" + std::to_string(producer.readyLevel + 1)
                 : std::string(" @ready"))
         << " <-";
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

/// A register kept apart from the equal registers it copies; marked as a
/// control, which foldsim-replicate-controls replicates further where it
/// drives more loads than a net should.
static seq::FirRegOp keep(seq::FirRegOp reg) {
  sv::addSVAttributes(reg, {sv::SVAttributeAttr::get(reg.getContext(), "keep",
                                                     R"("true")",
                                                     /*emitAsComment=*/false)});
  reg->setAttr(foldsim::kControlAttrName, UnitAttr::get(reg.getContext()));
  return reg;
}

/// The SLR group an op of the simulator is built in, until `extractGroups`
/// moves it into the group's module.
static constexpr StringLiteral kSlrAttr = "foldsim.slr";

static void tagGroup(Operation *op, int group) {
  if (group >= 0 && !op->hasAttr(kSlrAttr))
    op->setAttr(kSlrAttr, IntegerAttr::get(
                              IntegerType::get(op->getContext(), 32), group));
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

/// Sets an operation's Vivado ram_style, replacing any it has.
static void setRamStyle(Operation *op, StringRef kind) {
  auto *context = op->getContext();
  sv::modifySVAttributes(op, [&](SmallVectorImpl<sv::SVAttributeAttr> &attrs) {
    llvm::erase_if(attrs, [](sv::SVAttributeAttr attr) {
      return attr.getName().getValue() == "ram_style";
    });
    attrs.push_back(sv::SVAttributeAttr::get(
        context, "ram_style", ("\"" + kind + "\"").str(),
        /*emitAsComment=*/false));
  });
}

LogicalResult StaticFolder::parseStorage() {
  for (auto &spec : storageSpecs) {
    auto [model, rest] = StringRef(spec).split(':');
    auto [store, kind] = rest.rsplit('=');
    if (model.empty() || store.empty() || kind.empty() || kind == rest)
      return netlist.emitError("storage: '")
             << spec << "' is not <model>:<store>=<kind>";
    if (!llvm::is_contained({"registers", "distributed", "block", "ultra"},
                            kind))
      return netlist.emitError("storage: '")
             << kind
             << "' is not a ram_style: registers, distributed, block, ultra";
    if (llvm::none_of(classes, [&](auto &cls) {
          return cls.splitFrom < 0 && cls.name == model;
        }))
      return netlist.emitError("storage: no model '") << model << "'";
    StorageRule rule;
    rule.model = model.str();
    rule.store = store.str();
    rule.kind = kind.str();
    if (store == "regs") {
      rule.regs = true;
    } else if (store.consume_front("regs>=")) {
      rule.regs = true;
      if (store.getAsInteger(10, rule.minWidth))
        return netlist.emitError("storage: '")
               << spec << "': regs>= takes a width";
    }
    storageRules.push_back(std::move(rule));
  }
  return success();
}

StaticFolder::StorageRule *StaticFolder::storageFor(const Class &cls,
                                                    StringRef store, bool reg,
                                                    unsigned width) {
  StringRef model =
      cls.splitFrom >= 0 ? StringRef(classes[cls.splitFrom].name) : cls.name;
  StorageRule *best = nullptr;
  for (auto &rule : storageRules) {
    if (rule.model != model || rule.regs != reg)
      continue;
    if (reg ? width >= rule.minWidth &&
                  (!best || rule.minWidth > best->minWidth)
            : rule.store == store)
      best = &rule;
  }
  if (best)
    best->used = true;
  return best;
}

LogicalResult StaticFolder::buildCopy(Class &cls) {
  auto model = cls.model;
  auto *context = model.getContext();
  auto *modelBody = model.getBodyBlock();
  unsigned threads = cls.threads();
  unsigned idxWidth = threads > 1 ? llvm::Log2_32_Ceil(threads) : 0;
  // The entries a per-instance store keeps: every instance index, so a copy
  // of instances not a power of two leaves the last entries unused.
  unsigned entries = 1u << idxWidth;
  // Registers of so few instances, passed in a ring order, are a ring of
  // flops: no store, no read select and no bypass.
  bool ring = !cls.page && threads > 1 && threads <= registerStateEntries &&
              ringOrder(cls, ii);
  // A per-instance store this shallow is flops: Vivado would otherwise spend
  // a distributed RAM LUT's whole depth on a few words, and with two entries
  // the read folds into the bypass mux's LUT.
  auto keepShallowInFlops = [&](seq::FirMemOp store) {
    if (entries <= registerStateEntries)
      sv::addSVAttributes(store, {sv::SVAttributeAttr::get(
                                     context, "ram_style", R"("registers")",
                                     /*emitAsComment=*/false)});
  };
  auto *folding = cls.folding.get();
  auto physical = [&](Operation *op) {
    return folding && folding->physical.contains(op);
  };
  // One instance: a black box with a pipeline contract stays, stepped through
  // its clock enable at the state pass; the clock reaches it, as a bit if its
  // port is one, and nothing else. The boxes are kept in body order, so that
  // the regions their clock enables take, and so the output, are the same run
  // to run.
  llvm::MapVector<Operation *, foldsim::PipelineContract> boxes;
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

  // Memories kept outside the simulator (`--virtual-memories`): a registered
  // read port and at most one write port each. An instance's read is issued
  // at its state pass and must be pending before the sequencer decides the
  // position whose read register loads it, four host cycles before the
  // instance's next evaluation (three with a short lead); so the state gap
  // is at least five (four).
  DenseMap<Operation *, unsigned> virtualOf;
  if (!virtualMemoryPattern.empty()) {
    llvm::Regex pattern(virtualMemoryPattern);
    for (auto &op : *modelBody) {
      auto memory = dyn_cast<seq::FirMemOp>(op);
      if (!memory || physical(&op) ||
          !pattern.match(memory.getName().value_or("")))
        continue;
      if (memory.getReadLatency() != 1)
        return memory.emitError("a virtual memory needs a registered read");
      unsigned reads = 0, writes = 0;
      for (auto *user : memory->getUsers()) {
        if (isa<seq::FirMemReadOp>(user))
          ++reads;
        else if (isa<seq::FirMemWriteOp>(user))
          ++writes;
        else
          return user->emitError("a virtual memory has read and write ports "
                                 "only");
      }
      if (reads != 1 || writes > 1)
        return memory.emitError("a virtual memory has one read port and at "
                                "most one write port");
      int gap = shortLead ? 4 : 5;
      if (cls.stateGap < gap)
        return memory.emitError("a virtual memory needs a state gap of ")
               << gap << " host cycles; class " << cls.name << " has "
               << cls.stateGap;
      auto type = memory.getType();
      unsigned maskWidth = type.getMaskWidth().value_or(1);
      unsigned width = type.getWidth();
      // The controller takes byte enables, or none.
      if (maskWidth > 1 && (width % maskWidth || (width / maskWidth) % 8))
        return memory.emitError("a virtual memory's mask covers whole bytes");
      Class::VirtualMemory vm;
      vm.name = memory.getName().value_or("mem").str();
      vm.rowWidth = std::max(1u, llvm::Log2_64_Ceil(type.getDepth()));
      vm.dataWidth = width;
      vm.maskWidth = maskWidth > 1 ? width / 8 : 1;
      vm.readFirst = memory.getRuw() != seq::RUW::New;
      virtualOf[memory] = cls.virtualMemories.size();
      cls.virtualMemories.push_back(vm);
    }
  }

  // Paged register state: the registers a ready pass reads (the cone of an
  // output written at a ready pass, through the logic and combinational
  // reads, up to registers, registered reads and black boxes) keep their
  // stores, read at any slot; the rest page, read only at the state pass.
  auto *modelOutput = modelBody->getTerminator();
  DenseSet<Operation *> readyRegs;
  if (cls.page)
    readyRegisters(cls, readyRegs);

  // The early outputs' next values, before any register is replaced.
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
  // Paged: the instance whose pages to read, from three host cycles ahead.
  if (cls.page) {
    addPort("fetch3", IntegerType::get(context, 1), true);
    addPort("finst3", IntegerType::get(context, idxWidth), true);
  }
  SmallVector<unsigned> inputArg(cls.inputNames.size(), 0);
  for (auto [i, name] : llvm::enumerate(cls.inputNames)) {
    if (int(i) == cls.clockInput)
      continue;
    inputArg[i] = numIn;
    addPort(name, cls.inputTypes[i], true);
  }
  // A virtual memory's responses in, its requests and pending reads out.
  unsigned instWidth = std::max(idxWidth, 1u);
  auto intType = [&](unsigned width) { return IntegerType::get(context, width); };
  if (!cls.virtualMemories.empty()) {
    cls.resetInput = numIn;
    addPort("hostReset", intType(1), true);
  }
  for (auto [k, vm] : llvm::enumerate(cls.virtualMemories)) {
    std::string name = ("vm" + Twine(k) + "_").str();
    vm.firstInput = numIn;
    addPort(name + "rsp_valid", intType(1), true);
    addPort(name + "rsp_inst", intType(instWidth), true);
    addPort(name + "rsp_data", intType(vm.dataWidth), true);
  }
  for (auto [o, name] : llvm::enumerate(cls.outputNames))
    addPort(name, cls.outputTypes[o], false);
  for (auto [k, vm] : llvm::enumerate(cls.virtualMemories)) {
    std::string name = ("vm" + Twine(k) + "_").str();
    vm.firstOutput = numOut;
    addPort(name + "rd_valid", intType(1), false);
    addPort(name + "rd_inst", intType(instWidth), false);
    addPort(name + "rd_row", intType(vm.rowWidth), false);
    addPort(name + "wr_valid", intType(1), false);
    addPort(name + "wr_inst", intType(instWidth), false);
    addPort(name + "wr_row", intType(vm.rowWidth), false);
    addPort(name + "wr_data", intType(vm.dataWidth), false);
    addPort(name + "wr_mask", intType(vm.maskWidth), false);
    addPort(name + "pending", intType(entries), false);
  }
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
  Value fetch3 = cls.page ? body->getArgument(3) : Value();
  Value finst3 = cls.page ? body->getArgument(4) : Value();
  unsigned pages = cls.page ? unsigned(llvm::PowerOf2Ceil(cls.page)) : 0;
  unsigned pageWidth = pages > 1 ? llvm::Log2_32_Ceil(pages) : 0;
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
    /// Paged: a fetch two host cycles ahead, and the page read address,
    /// {instance, page}: the fetched instance's page 0 two host cycles after
    /// the fetch's `fetch3`, its next pages one a host cycle.
    Value fetch2, readAt;
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
      if (cls.page) {
        ahead.fetch2 = keep(seq::FirRegOp::create(
            b, fetch3, hostClock, b.getStringAttr(prefix + "_fetch2")));
        BackedgeBuilder edges(b, model.getLoc());
        auto instNext = edges.get(finst3.getType());
        Value inst = keep(seq::FirRegOp::create(
            b, instNext, hostClock, b.getStringAttr(prefix + "_page_inst")));
        instNext.setValue(comb::MuxOp::create(b, fetch3, finst3, inst));
        Value readAt = inst;
        if (pageWidth) {
          auto pageNext = edges.get(b.getIntegerType(pageWidth));
          Value page = keep(seq::FirRegOp::create(
              b, pageNext, hostClock, b.getStringAttr(prefix + "_page")));
          pageNext.setValue(comb::MuxOp::create(
              b, fetch3, hw::ConstantOp::create(b, b.getIntegerType(pageWidth), 0),
              comb::AddOp::create(
                  b, page,
                  hw::ConstantOp::create(b, b.getIntegerType(pageWidth), 1))));
          readAt = comb::ConcatOp::create(b, inst, page);
        }
        ahead.readAt = readAt;
      }
      aheads.push_back(ahead);
    }
    ++aheads.back().regions;
    return aheads.back();
  };

  struct Region {
    std::string prefix;
    Value inst2, inst1, inst, state, instStaged;
    /// Paged: its group's fetch two host cycles ahead, and its page read
    /// address.
    Value fetch2, readAt;
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
      region.fetch2 = ahead.fetch2;
      region.readAt = ahead.readAt;
      if (threads > 1) {
        region.inst2 = ahead.inst2;
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

  // A virtual memory: its read and write as requests at the state pass, each
  // instance's read data from its responses, kept like a registered read's
  // (no bypass: the state gap covers the buffer), and the instances whose
  // read is pending, set as it is issued and cleared as its response comes.
  SmallVector<SmallVector<Value>> virtualPorts(cls.virtualMemories.size());
  auto buildVirtualMemory = [&](seq::FirMemOp memory,
                                unsigned k) -> SmallVector<Value> {
    auto &vm = cls.virtualMemories[k];
    auto &region = regionFor(vm.dataWidth);
    Location loc = memory.getLoc();
    BackedgeBuilder edges(b, loc);
    auto zero = [&](unsigned width) -> Value {
      return hw::ConstantOp::create(b, intType(width), 0);
    };
    auto widen = [&](Value value, unsigned width) -> Value {
      unsigned have = hw::getBitWidth(value.getType());
      if (have == width)
        return value;
      if (have > width)
        return b.createOrFold<comb::ExtractOp>(intType(width), value, 0);
      return b.createOrFold<comb::ConcatOp>(zero(width - have), value);
    };
    Value inst = threads > 1 ? region.inst : zero(1);
    Value rspValid = body->getArgument(vm.firstInput);
    Value rspInst = body->getArgument(vm.firstInput + 1);
    Value rspData = body->getArgument(vm.firstInput + 2);
    Value rdValid = zero(1), rdRow = zero(vm.rowWidth);
    Value wrValid = zero(1), wrRow = zero(vm.rowWidth);
    Value wrData = zero(vm.dataWidth), wrMask = zero(vm.maskWidth);
    for (auto *user : llvm::make_early_inc_range(memory->getUsers())) {
      if (auto write = dyn_cast<seq::FirMemWriteOp>(user)) {
        wrValid = region.state;
        if (write.getEnable())
          wrValid = comb::AndOp::create(b, wrValid, write.getEnable());
        wrRow = widen(write.getAddress(), vm.rowWidth);
        wrData = asBits(b, write.getData());
        unsigned maskBits = memory.getType().getMaskWidth().value_or(1);
        if (vm.maskWidth == 1 || !write.getMask()) {
          wrMask = hw::ConstantOp::create(b, APInt::getAllOnes(vm.maskWidth));
        } else {
          // Each mask bit enables its whole bytes.
          unsigned perBit = vm.maskWidth / maskBits;
          SmallVector<Value> bytes;
          for (unsigned i = maskBits; i-- > 0;) {
            Value bit = b.createOrFold<comb::ExtractOp>(intType(1),
                                                        write.getMask(), i);
            bytes.push_back(perBit > 1
                                ? b.createOrFold<comb::ReplicateOp>(
                                      intType(perBit), bit)
                                : bit);
          }
          wrMask = b.createOrFold<comb::ConcatOp>(bytes);
        }
        write.erase();
        continue;
      }
      auto read = cast<seq::FirMemReadOp>(user);
      rdValid = region.state;
      if (read.getEnable())
        rdValid = comb::AndOp::create(b, rdValid, read.getEnable());
      rdRow = widen(read.getAddress(), vm.rowWidth);
      auto readName = vm.name + "_rdata";
      Value data;
      if (threads == 1) {
        auto nextEdge = edges.get(rspData.getType());
        auto buffer = seq::FirRegOp::create(b, nextEdge, hostClock,
                                            b.getStringAttr(readName));
        nextEdge.setValue(comb::MuxOp::create(b, rspValid, rspData, buffer));
        data = buffer;
      } else {
        auto bufferType =
            seq::FirMemType::get(context, entries, vm.dataWidth, 1);
        auto buffer = seq::FirMemOp::create(
            b, bufferType, /*readLatency=*/0, /*writeLatency=*/1,
            seq::RUW::Undefined, seq::WUW::Undefined,
            b.getStringAttr(readName), hw::InnerSymAttr{},
            seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
        keepShallowInFlops(buffer);
        seq::FirMemWriteOp::create(b, buffer, rspInst, hostClock, rspValid,
                                   rspData, Value{});
        Value bufferData = seq::FirMemReadOp::create(b, buffer, region.inst1,
                                                     hostClock, one());
        data = seq::FirRegOp::create(b, bufferData, hostClock,
                                     b.getStringAttr(readName + "_read"));
      }
      data = b.createOrFold<hw::BitcastOp>(read.getData().getType(), data);
      read.getData().replaceAllUsesWith(data);
      replaced[read.getData()] = data;
      read.erase();
    }
    memory.erase();
    auto bitOf = [&](Value index) -> Value {
      return comb::ShlOp::create(b, hw::ConstantOp::create(b, intType(entries), 1),
                                 widen(index, entries));
    };
    Value set = comb::MuxOp::create(b, rdValid, bitOf(inst), zero(entries));
    Value clear = comb::MuxOp::create(b, rspValid, bitOf(rspInst), zero(entries));
    auto pendingNext = edges.get(intType(entries));
    Value pending = seq::FirRegOp::create(
        b, pendingNext, hostClock, b.getStringAttr(vm.name + "_pending"),
        body->getArgument(cls.resetInput), zero(entries), hw::InnerSymAttr{},
        /*isAsync=*/false, b.getIntegerAttr(intType(entries), 0));
    pendingNext.setValue(comb::AndOp::create(
        b, comb::OrOp::create(b, pending, set),
        comb::XorOp::create(b, clear,
                            hw::ConstantOp::create(b, APInt::getAllOnes(entries)))));
    return {rdValid, inst, rdRow, wrValid, inst, wrRow, wrData, wrMask, pending};
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

  // Paged registers, in state order: each read from the fetch buffer (a
  // placeholder until the pages are built), and its write-back register,
  // loaded at the state pass with its next value whole (a held register's
  // old value where its enable is low).
  struct PagedReg {
    Value original;
    Backedge value;
    unsigned width;
    Value writeBack;
    std::string name;
  };
  SmallVector<PagedReg> pagedRegs;
  BackedgeBuilder pageEdges(b, model.getLoc());
  unsigned keptRegs = 0;
  uint64_t keptBits = 0;

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
      if (ring && !write.holds) {
        // The ring: the head is the evaluated instance's state; its state
        // pass moves each register one place toward the head and puts the
        // instance's next value at the tail.
        auto backedge = BackedgeBuilder(b, reg.getLoc());
        SmallVector<Backedge> nextEdges;
        SmallVector<Value> ringRegs;
        for (unsigned i = 0; i < threads; ++i) {
          nextEdges.push_back(backedge.get(reg.getType()));
          ringRegs.push_back(seq::FirRegOp::create(
              b, nextEdges.back(), hostClock,
              b.getStringAttr(reg.getName() + "_ring" + Twine(i))));
        }
        Value head = ringRegs.front();
        Value tail = write.enable ? Value(comb::MuxOp::create(
                                        b, write.enable, write.data, head))
                                  : write.data;
        for (unsigned i = 0; i < threads; ++i) {
          Value in = i + 1 < threads ? ringRegs[i + 1] : tail;
          nextEdges[i].setValue(
              comb::MuxOp::create(b, region.state, in, ringRegs[i]));
        }
        value = head;
      } else if (threads == 1 || ring) {
        // One instance, or a register no instance writes: the register,
        // enabled at its state pass.
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
      } else if (cls.page && !write.holds && width && !readyRegs.contains(reg)) {
        // Paged: read out of the fetch buffer, written whole into its
        // write-back register at the state pass (the pages built below).
        auto placeholder = pageEdges.get(reg.getType());
        Value data = asBits(b, write.data);
        Value next = write.enable ? Value(comb::MuxOp::create(
                                        b, write.enable, data,
                                        asBits(b, placeholder)))
                                  : data;
        BackedgeBuilder backEdges(b, reg.getLoc());
        auto backNext = backEdges.get(data.getType());
        Value back = seq::FirRegOp::create(
            b, backNext, hostClock,
            b.getStringAttr(reg.getName() + "_writeback"));
        backNext.setValue(comb::MuxOp::create(b, region.state, next, back));
        pagedRegs.push_back({reg.getResult(), placeholder, unsigned(width),
                             back, reg.getName().str()});
        value = placeholder;
      } else {
        if (cls.page && !write.holds) {
          ++keptRegs;
          keptBits += width;
        }
        // A store in block or ultra RAM reads a host cycle early, at the
        // instance two host cycles ahead, into the RAM's own register: the
        // same data a host cycle later when no write lands in between, which
        // a state gap of 4 ensures (the staged write lands a host cycle after
        // the state pass).
        auto *rule = storageFor(cls, reg.getName(), /*reg=*/true, width);
        bool block = rule && (rule->kind == "block" || rule->kind == "ultra");
        if (block && cls.stateGap < 4)
          return reg.emitError("storage: '")
                 << rule->kind << "' RAM for the registers of '" << cls.name
                 << "' needs a state gap of 4, not " << cls.stateGap;
        // Read-first: the RAM's output register, as Vivado infers block and
        // ultra RAM (the read never meets a write to its entry).
        auto memType = seq::FirMemType::get(context, entries, width, 1);
        auto memory = seq::FirMemOp::create(
            b, memType, /*readLatency=*/block ? 1 : 0, /*writeLatency=*/1,
            block ? seq::RUW::Old : seq::RUW::Undefined, seq::WUW::Undefined,
            reg.getNameAttr(),
            hw::InnerSymAttr{}, seq::FirMemInitAttr{}, StringAttr{},
            Attribute{});
        keepShallowInFlops(memory);
        if (rule)
          setRamStyle(memory, rule->kind);
        Value data = seq::FirMemReadOp::create(
            b, memory, block ? region.inst2 : region.inst1, hostClock, one());
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
          // now writes a host cycle later. The read register takes the newest;
          // only the bypasses the schedule needs are built.
          if (cls.stateGap < 3) {
            Value staged = comb::AndOp::create(
                b, stagedEnable,
                comb::ICmpOp::create(b, comb::ICmpPredicate::eq,
                                     region.instStaged, region.inst1));
            loaded = comb::MuxOp::create(b, staged, stagedData, loaded);
          }
          if (cls.stateGap < 2) {
            Value now = comb::AndOp::create(
                b, region.state,
                comb::ICmpOp::create(b, comb::ICmpPredicate::eq, region.inst,
                                     region.inst1));
            Value next = asBits(b, write.data);
            if (write.enable)
              next = comb::MuxOp::create(b, write.enable, next, read);
            loaded = comb::MuxOp::create(b, now, next, loaded);
          }
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
    if (auto it = virtualOf.find(memory); it != virtualOf.end()) {
      virtualPorts[it->second] = buildVirtualMemory(memory, it->second);
      continue;
    }
    auto memType = memory.getType();
    uint64_t bits = uint64_t(memType.getWidth()) * memType.getDepth();
    auto &region = regionFor(bits);
    auto threadedType = seq::FirMemType::get(
        context, memType.getDepth() * entries, memType.getWidth(),
        memType.getMaskWidth());
    auto threadedMemory = seq::FirMemOp::create(
        b, threadedType, memory.getReadLatency(), memory.getWriteLatency(),
        memory.getRuw(), memory.getWuw(), memory.getNameAttr(),
        hw::InnerSymAttr{}, seq::FirMemInitAttr{}, memory.getPrefixAttr(),
        memory.getOutputFileAttr());
    threadedMemory->setDialectAttrs(memory->getDialectAttrs());
    if (auto *rule = storageFor(cls, memory.getName().value_or(""),
                                /*reg=*/false, memType.getWidth())) {
      if ((rule->kind == "block" || rule->kind == "ultra") &&
          memory.getReadLatency() == 0)
        return memory.emitError("storage: '")
               << rule->kind << "' RAM for '" << rule->store
               << "', which reads combinationally";
      setRamStyle(threadedMemory, rule->kind);
    }
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
          // The data out of the memory now, before the buffer holds it: read
          // at the state pass just before this slot.
          if (cls.stateGap < 2)
            data = comb::MuxOp::create(b, capture, dout, buffer);
        } else {
          auto width = hw::getBitWidth(dout.getType());
          auto bufferType = seq::FirMemType::get(context, entries, width, 1);
          auto buffer = seq::FirMemOp::create(
              b, bufferType, /*readLatency=*/0, /*writeLatency=*/1,
              seq::RUW::Undefined, seq::WUW::Undefined,
              b.getStringAttr(readName), hw::InnerSymAttr{},
              seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
          keepShallowInFlops(buffer);
          Value readInst = seq::FirRegOp::create(
              b, region.inst, hostClock, b.getStringAttr(readName + "_inst"));
          seq::FirMemWriteOp::create(b, buffer, readInst, hostClock, capture,
                                     asBits(b, dout), Value{});
          Value bufferData = seq::FirMemReadOp::create(b, buffer, region.inst1,
                                                       hostClock, one());
          // The next slot's instance's data, out of the memory now, is not
          // yet in the buffer.
          if (cls.stateGap < 4) {
            Value fresh = comb::AndOp::create(
                b, capture,
                comb::ICmpOp::create(b, comb::ICmpPredicate::eq, readInst,
                                     region.inst1));
            bufferData =
                comb::MuxOp::create(b, fresh, asBits(b, dout), bufferData);
          }
          data = seq::FirRegOp::create(b, bufferData, hostClock,
                                       b.getStringAttr(readName + "_read"));
          // This slot's instance's data, out of the memory now, read at its
          // state pass just before.
          if (cls.stateGap < 2) {
            Value now = comb::AndOp::create(
                b, capture,
                comb::ICmpOp::create(b, comb::ICmpPredicate::eq, readInst,
                                     region.inst));
            data = comb::MuxOp::create(b, now, asBits(b, dout), data);
          }
          data = b.createOrFold<hw::BitcastOp>(dout.getType(), data);
        }
      }
      read.getData().replaceAllUsesWith(data);
      replaced[read.getData()] = data;
      read.erase();
    }
    memory.erase();
  }

  // The paged registers, one vector of W bits a row of `pages` pages of
  // pw = ceil(W / p) bits, the page word cut into columns of whole 72-bit
  // ports (one RAM and one copy of the controls each, at most about
  // `regionBits` bits): pw / 72 ports read a page a host cycle, the
  // bits/ports bound. A column's RAM has a row per {instance, page}:
  // - page-in: the fetch's pages, read one a host cycle from two host cycles
  //   after its `fetch3`, each through the RAM's output register into the
  //   fetch buffer's parts it holds, which the target logic reads: page j
  //   lands at the end of the fetch position's host cycle + j;
  // - write-back: the state pass loads each register's write-back register;
  //   page j is written a host cycle + j after.
  // The schedule spaces a copy's state passes a page apart and fetches each
  // a page before it, so neither buffer is overwritten before it is used.
  if (!pagedRegs.empty()) {
    unsigned p = cls.page;
    BackedgeBuilder edges(b, model.getLoc());
    auto bits = [&](unsigned width) { return b.getIntegerType(width); };
    auto *rule = storageFor(cls, "pages", /*reg=*/false, 0);
    std::string kind = rule ? rule->kind : "block";
    if (kind != "block" && kind != "ultra" && kind != "distributed" &&
        kind != "registers")
      return model.emitError("storage: '") << kind << "' for pages";
    uint64_t pagedBits = 0;
    for (auto &reg : pagedRegs)
      pagedBits += reg.width;
    unsigned pw = unsigned((pagedBits + p - 1) / p);
    unsigned colBits = 72 * std::max(1u, regionBits / 72);
    unsigned columns = (pw + colBits - 1) / colBits;
    struct Column {
      unsigned width;
      Region *region;
      seq::FirMemOp memory;
      Value out, writeAt;
      /// Per page: the host cycle it lands, it is written, and its bits.
      SmallVector<Value> land, writing;
      SmallVector<SmallVector<Value>> words;
    };
    SmallVector<Column> cols(columns);
    // Each column's controls: a region of its own.
    for (unsigned c = 0; c < columns; ++c) {
      unsigned width = std::min(colBits, pw - c * colBits);
      regionFor(regionBits);
      cols[c].width = width;
    }
    for (unsigned c = 0; c < columns; ++c)
      cols[c].region = &regions[regions.size() - columns + c];
    unsigned ports = 0;
    for (auto &col : cols) {
      auto &region = *col.region;
      std::string prefix = region.prefix;
      ports += (col.width + 71) / 72;
      auto memType = seq::FirMemType::get(context, entries * pages, col.width, 1);
      auto memory = seq::FirMemOp::create(
          b, memType, /*readLatency=*/1, /*writeLatency=*/1, seq::RUW::Old,
          seq::WUW::Undefined, b.getStringAttr(prefix + "_pages"),
          hw::InnerSymAttr{}, seq::FirMemInitAttr{}, StringAttr{},
          Attribute{});
      setRamStyle(memory, kind);
      col.memory = memory;
      Value data = seq::FirMemReadOp::create(b, memory, region.readAt,
                                             hostClock, one());
      // The RAM's output register.
      col.out = seq::FirRegOp::create(b, data, hostClock,
                                      b.getStringAttr(prefix + "_page_out"));
      // Page j lands at the end of the host cycle `land[j]` is set.
      Value fetch1 = keep(seq::FirRegOp::create(
          b, region.fetch2, hostClock, b.getStringAttr(prefix + "_fetch1")));
      col.land.push_back(keep(seq::FirRegOp::create(
          b, fetch1, hostClock, b.getStringAttr(prefix + "_land0"))));
      for (unsigned j = 1; j < p; ++j)
        col.land.push_back(keep(seq::FirRegOp::create(
            b, col.land.back(), hostClock,
            b.getStringAttr(prefix + "_land" + Twine(j)))));
      col.words.resize(p);
      // Write-back: page j a host cycle + j after the state pass, at the
      // instance of that pass.
      auto &writing = col.writing;
      writing.push_back(keep(seq::FirRegOp::create(
          b, region.state, hostClock, b.getStringAttr(prefix + "_write0"))));
      for (unsigned j = 1; j < p; ++j)
        writing.push_back(keep(seq::FirRegOp::create(
            b, writing.back(), hostClock,
            b.getStringAttr(prefix + "_write" + Twine(j)))));
      auto instNext = edges.get(region.inst.getType());
      Value writeInst = keep(seq::FirRegOp::create(
          b, instNext, hostClock, b.getStringAttr(prefix + "_write_inst")));
      instNext.setValue(
          comb::MuxOp::create(b, region.state, region.inst, writeInst));
      Value writeAt = writeInst;
      if (pageWidth) {
        auto pageNext = edges.get(bits(pageWidth));
        Value page = keep(seq::FirRegOp::create(
            b, pageNext, hostClock, b.getStringAttr(prefix + "_write_page")));
        pageNext.setValue(comb::MuxOp::create(
            b, region.state, hw::ConstantOp::create(b, bits(pageWidth), 0),
            comb::AddOp::create(b, page,
                                hw::ConstantOp::create(b, bits(pageWidth), 1))));
        writeAt = comb::ConcatOp::create(b, writeInst, page);
      }
      col.writeAt = writeAt;
    }
    // Each register's bits, cut where pages and columns meet: a fetch
    // buffer part loaded as its page lands in its column, and the
    // write-back bits of that page and column, low bits first.
    uint64_t at = 0;
    for (auto &reg : pagedRegs) {
      SmallVector<Value> parts;
      unsigned part = 0;
      for (unsigned lo = 0; lo < reg.width;) {
        uint64_t g = at + lo;
        unsigned j = unsigned(g / pw), k = unsigned(g % pw);
        unsigned c = k / colBits, offset = k % colBits;
        unsigned n = std::min({reg.width - lo, pw - k, colBits - offset});
        auto &col = cols[c];
        std::string name =
            reg.name + (n == reg.width ? std::string()
                                       : "_" + std::to_string(part++));
        auto fetchNext = edges.get(bits(n));
        Value fetched = seq::FirRegOp::create(
            b, fetchNext, hostClock, b.getStringAttr(name + "_fetched"));
        fetchNext.setValue(comb::MuxOp::create(
            b, col.land[j],
            b.createOrFold<comb::ExtractOp>(bits(n), col.out, offset), fetched));
        parts.push_back(fetched);
        col.words[j].push_back(
            b.createOrFold<comb::ExtractOp>(bits(n), reg.writeBack, lo));
        lo += n;
      }
      Value value = parts.size() == 1
                        ? parts[0]
                        : b.createOrFold<comb::ConcatOp>(
                              SmallVector<Value>(parts.rbegin(), parts.rend()));
      value = b.createOrFold<hw::BitcastOp>(Value(reg.value).getType(), value);
      reg.value.setValue(value);
      replaced[reg.original] = value;
      at += reg.width;
    }
    // Each column's write: page j's bits at its host cycle.
    for (auto &col : cols) {
      auto &writing = col.writing;
      Value writeData, writeEnable;
      for (unsigned j = 0; j < p; ++j) {
        unsigned have = 0;
        for (auto word : col.words[j])
          have += hw::getBitWidth(word.getType());
        SmallVector<Value> high;
        if (have < col.width)
          high.push_back(hw::ConstantOp::create(b, bits(col.width - have), 0));
        high.append(col.words[j].rbegin(), col.words[j].rend());
        Value word = high.size() == 1 ? high[0]
                                      : b.createOrFold<comb::ConcatOp>(high);
        if (p > 1)
          word = comb::AndOp::create(
              b, word,
              b.createOrFold<comb::ReplicateOp>(bits(col.width), writing[j]));
        writeData = writeData ? Value(comb::OrOp::create(b, writeData, word))
                              : word;
        writeEnable = writeEnable
                          ? Value(comb::OrOp::create(b, writeEnable, writing[j]))
                          : writing[j];
      }
      seq::FirMemWriteOp::create(b, col.memory, col.writeAt, hostClock,
                                 writeEnable, writeData, Value{});
    }
    std::string text;
    llvm::raw_string_ostream os(text);
    unsigned bound =
        threads * unsigned((pagedBits + 72ull * ports - 1) / (72ull * ports));
    os << "  pages " << cls.name << ": p " << p << ", " << pagedRegs.size()
       << " registers, " << pagedBits << " bits; page RAMs " << entries * pages
       << " rows x " << pw << " bits in " << columns << " columns, " << ports
       << " ports, " << kind << "; kept " << keptRegs << " registers, "
       << keptBits << " bits; state passes " << threads << " x " << p << " = "
       << threads * p << ", ports bound " << bound << "\n";
    pageReport += os.str();
  } else if (cls.page) {
    pageReport += "  pages " + cls.name + ": p " + std::to_string(cls.page) +
                  ", 0 registers; kept " + std::to_string(keptRegs) +
                  " registers, " + std::to_string(keptBits) + " bits\n";
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
  for (auto &ports : virtualPorts)
    outputs.append(ports.begin(), ports.end());
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
    // An offered group input's token holds each of its values.
    if (producer.width > 0)
      addPort(producer.name + "_bits",
              IntegerType::get(context, producer.width *
                                            (specOf[p] >= 0 ? specValues() : 1)),
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
  unsigned advancePort = numIn;
  if (bubbles)
    addPort("advance", i1, true);
  // Each class's virtual memories, numbered over the netlist: a controller's
  // responses and free request entries in, its requests out.
  struct SimVirtual {
    unsigned cls, k, firstInput;
  };
  SmallVector<SimVirtual> simVirtuals;
  for (auto [c, cls] : llvm::enumerate(classes))
    for (auto [k, vm] : llvm::enumerate(cls.virtualMemories)) {
      std::string name = ("vm" + Twine(simVirtuals.size()) + "_").str();
      unsigned instWidth =
          std::max(cls.threads() > 1 ? llvm::Log2_32_Ceil(cls.threads()) : 0u, 1u);
      simVirtuals.push_back({unsigned(c), unsigned(k), unsigned(numIn)});
      addPort(name + "rsp_valid", i1, true);
      addPort(name + "rsp_inst", IntegerType::get(context, instWidth), true);
      addPort(name + "rsp_data", IntegerType::get(context, vm.dataWidth), true);
      addPort(name + "free", IntegerType::get(context, kVirtualFreeWidth), true);
      addPort(name + "rd_valid", i1, false);
      addPort(name + "rd_inst", IntegerType::get(context, instWidth), false);
      addPort(name + "rd_row", IntegerType::get(context, vm.rowWidth), false);
      addPort(name + "wr_valid", i1, false);
      addPort(name + "wr_inst", IntegerType::get(context, instWidth), false);
      addPort(name + "wr_row", IntegerType::get(context, vm.rowWidth), false);
      addPort(name + "wr_data", IntegerType::get(context, vm.dataWidth), false);
      addPort(name + "wr_mask", IntegerType::get(context, vm.maskWidth), false);
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
  // With bubbles, a host cycle `advance` is low the position stays and the
  // slot tables read `ii` the host cycle after (a bubble): every control
  // after the decision is a register chain of it, so the whole frame from
  // there on moves a host cycle later, and no distance shrinks.
  Value advance = bubbles ? body->getArgument(advancePort) : Value();
  // With virtual memories, a position also waits for its instance's read.
  Backedge virtualReady;
  if (!simVirtuals.empty()) {
    virtualReady = backedges.get(i1);
    advance = advance ? Value(comb::AndOp::create(b, advance, virtualReady))
                      : Value(virtualReady);
  }
  // A position's next: the next slot, or at the frame's end the next frame's
  // first if it starts, else `ii`; with bubbles, itself when not advancing.
  auto nextPosition = [&](Value pos) -> Value {
    Value end = comb::OrOp::create(b, eq(pos, ii - 1), eq(pos, ii));
    Value next = comb::MuxOp::create(
        b, end,
        comb::MuxOp::create(b, go, constant(posWidth, 0), constant(posWidth, ii)),
        comb::AddOp::create(b, pos, constant(posWidth, 1)));
    return advance ? comb::MuxOp::create(b, advance, next, pos) : next;
  };
  // The position the slot tables read: `ii` in a bubble. With a short lead
  // the tables read a position as it is decided, so a bubble is this host
  // cycle's; else the decision's register's, so it is the host cycle after.
  auto bubbleOf = [&](const Twine &regName) -> Value {
    return shortLead ? notOf(advance) : resetReg(notOf(advance), regName, 0);
  };
  auto shown = [&](Value pos, Value bubble) -> Value {
    return advance ? comb::MuxOp::create(b, bubble, constant(posWidth, ii), pos)
                   : pos;
  };
  // With a short lead the frame is decided a host cycle later, on the
  // position two host cycles ahead: the one three ahead is its next, which
  // the slot tables read as the decision makes it.
  auto posDNext = backedges.get(b.getIntegerType(posWidth));
  Value posD = resetReg(posDNext, shortLead ? "pos2" : "pos3", ii);
  Value boundary = comb::OrOp::create(b, eq(posD, ii - 1), eq(posD, ii));
  Value starting = comb::AndOp::create(b, boundary, go);
  if (advance)
    starting = comb::AndOp::create(b, starting, advance);
  Value posDN = nextPosition(posD);
  posDNext.setValue(posDN);
  auto parDNext = backedges.get(i1);
  Value parD = resetReg(parDNext, shortLead ? "parity2" : "parity3", 1);
  Value parDN = comb::MuxOp::create(b, starting, notOf(parD), parD);
  parDNext.setValue(parDN);
  Value pos3 = shortLead ? posDN : posD;
  if (advance)
    pos3 = shown(pos3, bubbleOf("bubble3"));
  Value rootNext = posDN;
  // With bubbles and a short lead, the decision's register is not the
  // position shown a host cycle before: in a bubble it holds.
  Value pos2 = shortLead && !advance ? posD : resetReg(pos3, "pos2", ii);
  Value par2 = shortLead ? parD : reg(parD, "parity2");
  Value pos1 = resetReg(pos2, "pos1", ii), par1 = reg(par2, "parity1");
  Value pos0 = resetReg(pos1, "pos0", ii);

  // Each class's slot table, its instance and passes three host cycles ahead
  // (the table's), two, one, then now.
  struct Controls {
    Value inst3, state3;
    /// Paged: the instance whose pages are read, from three host cycles
    /// ahead (a page before its state pass).
    Value fetch3, finst3;
    Value inst2, state2, ready2, tail2;
    Value inst1, inst, state, ready;
    /// The ready passes past the first, two host cycles ahead.
    SmallVector<Value> laterReady2;
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
    auto past = [&](ArrayRef<int> slots) {
      return llvm::any_of(slots, [&](int s) { return s >= int(ii); });
    };
    return past(cls.slot[kState]) || past(cls.slot[kReady]) ||
           llvm::any_of(cls.laterReady, past);
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
    // The ready passes past the first, a row each.
    SmallVector<SmallVector<uint64_t>> laterRows(cls.laterReady.size(),
                                                 SmallVector<uint64_t>(ii + 1, 0));
    for (unsigned i = 0; i < threads; ++i)
      for (auto pass : {kReady, kState})
        for (unsigned l = 0; l < (pass == kReady ? cls.readyLevels : 1); ++l) {
          unsigned s = pass == kState ? cls.slot[kState][i] : cls.readySlots(l)[i];
          instRow[s % ii] = i;
          (pass == kState ? stateRow : l ? laterRows[l - 1] : readyRow)[s % ii] = 1;
          tailRow[s % ii] = s >= ii;
        }
    // A sequencer is the position the decision is made on; with a short
    // lead the tables read its next.
    Value at = !nest ? pos3 : shortLead ? nextPosition(sequencer[c]) : sequencer[c];
    if (nest && advance)
      at = shown(at, bubbleOf(cls.name + "_bubble"));
    auto &ctl = controls[c];
    auto prefix = cls.name;
    ctl.state3 = table(b, at, stateRow, 1);
    Value ready3 = table(b, at, readyRow, 1);
    Value runs;
    if (overlapped) {
      // A slot of the frame before runs once a frame has finished.
      Value tail3 = table(b, at, tailRow, 1);
      runs = comb::OrOp::create(b, notOf(tail3), primed);
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
    if (cls.page) {
      // Each state pass's page-in, a page before it in the position stream
      // (in the frame before for a slot within the page of the frame's start).
      SmallVector<uint64_t> fetchRow(ii + 1, 0), finstRow(ii + 1, 0);
      for (unsigned i = 0; i < threads; ++i) {
        int f = cls.slot[kState][i] - int(cls.page);
        unsigned pos = unsigned(((f % int(ii)) + int(ii)) % int(ii));
        fetchRow[pos] = 1;
        finstRow[pos] = i;
      }
      ctl.fetch3 = table(b, at, fetchRow, 1);
      ctl.finst3 = table(b, at, finstRow, idxWidth);
    }
    for (auto [l, row] : llvm::enumerate(laterRows)) {
      Value later3 = table(b, at, row, 1);
      if (overlapped)
        later3 = comb::AndOp::create(b, later3, runs);
      ctl.laterReady2.push_back(
          reg(later3, prefix + "_ready_l" + Twine(l + 1) + "_2"));
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
  // Each group's select: its valid input and select output, both of the
  // frame before the one loading; at depth 2, also the one of the frame
  // before that, kept from the load before.
  SmallVector<Backedge> groupSelect;
  SmallVector<Value> groupIndex;
  for (unsigned g = 0; g < specGroups.size(); ++g) {
    groupSelect.push_back(backedges.get(i1));
    Value index = groupSelect.back();
    if (specDepth == 2) {
      auto olderNext = backedges.get(i1);
      Value older = resetReg(olderNext, consumers[specGroups[g].select].name +
                                            "_select_older", 0);
      olderNext.setValue(
          comb::MuxOp::create(b, load, groupSelect.back(), older));
      index = comb::ConcatOp::create(b, older, groupSelect.back());
    }
    groupIndex.push_back(index);
  }
  SmallVector<Value> inputValid;
  SmallVector<std::pair<unsigned, Value>> inputReady;
  for (auto [p, producer] : llvm::enumerate(producers)) {
    if (producer.cls >= 0)
      continue;
    Value valid = body->getArgument(inputPorts[p].valid);
    Value bits = producer.width ? body->getArgument(inputPorts[p].bits) : Value();
    unsigned tokenWidth = producer.width * (specOf[p] >= 0 ? specValues() : 1);
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
        auto heldNext = backedges.get(b.getIntegerType(tokenWidth));
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
          nexts.push_back(backedges.get(b.getIntegerType(tokenWidth)));
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
    // An offered group input: the value its group's select chooses, the
    // high one for a select set in the frame before.
    if (specOf[p] >= 0) {
      unsigned w = producer.width;
      SmallVector<Value> values;
      for (unsigned v = specValues(); v-- > 0;)
        values.push_back(
            comb::ExtractOp::create(b, b.getIntegerType(w), head, v * w));
      if (specDepth == 1)
        head = comb::MuxOp::create(b, groupIndex[specOf[p]], values[0],
                                   values[1]);
      else
        head = hw::ArrayGetOp::create(b, hw::ArrayCreateOp::create(b, values),
                                      groupIndex[specOf[p]]);
    }
    Value taken = comb::MuxOp::create(b, load, head, x);
    next.setValue(taken);
    frameInput[p] = x;
    frameInputNext[p] = taken;
  }
  // The frame's input as a class's evaluation a host cycle ahead reads it:
  // in a slot past the frame, the frame before's.
  std::map<std::pair<unsigned, int>, Value> prevFrameInputNext;
  DenseMap<unsigned, Value> classTail1;
  auto frameInputFor = [&](unsigned p, unsigned c) -> Value {
    if (!controls[c].tail2)
      return frameInputNext[p];
    // One per SLR group, in the group.
    Value &prevNext = prevFrameInputNext[{p, slrOf.empty() ? -1 : slrOf[c]}];
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
    if (cls.page) {
      operands.push_back(controls[c].fetch3);
      operands.push_back(controls[c].finst3);
    }
    for (auto [i, type] : llvm::enumerate(cls.inputTypes)) {
      if (int(i) == cls.clockInput)
        continue;
      copyInputs[c].push_back(backedges.get(type));
      operands.push_back(copyInputs[c].back());
    }
    if (!cls.virtualMemories.empty())
      operands.push_back(hostReset);
    for (auto &v : simVirtuals)
      if (v.cls == c)
        for (unsigned j = 0; j < 3; ++j)
          operands.push_back(body->getArgument(v.firstInput + j));
    copies[c] = hw::InstanceOp::create(b, cls.copy,
                                       b.getStringAttr(cls.name + "_copy"),
                                       operands);
    if (!slrOf.empty())
      tagGroup(copies[c], slrOf[c]);
  }

  // Groups of links, each with its own copy of its producer's controls and of
  // the frame's parity, so that no control drives more than about
  // `regionBits` bits of links.
  // With SLR groups, each group reading a class's links has groups of its
  // own (`to`: the reading group, -1 for the netlist's outputs), in it; one
  // in another group than the producer's has its controls a host cycle
  // later, with the producer's bits registered in the producer's group.
  struct LinkGroup {
    std::string prefix;
    Value state, ready, inst, bank;
    /// The enables of the ready passes past the first.
    SmallVector<Value> laterReady;
    // A host cycle later, for the writes of registered bits (the ready
    // passes' by level).
    Value stateLate, instLate, bankLate;
    SmallVector<Value> readyLate;
    uint64_t used = 0;
  };
  std::map<std::pair<unsigned, int>, SmallVector<LinkGroup>> linkGroups;
  auto linkGroupFor = [&](unsigned c, int to, uint64_t bits) -> LinkGroup & {
    auto &groups = linkGroups[{c, to}];
    if (groups.empty() || groups.back().used + bits > regionBits) {
      LinkGroup group;
      group.prefix = (classes[c].name +
                      (to >= 0 ? "_slr" + Twine(to) : Twine()) + "_links" +
                      Twine(groups.size()))
                         .str();
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
  // The producer's bits a reading group in another SLR group takes, registered
  // in the producer's group: one register for every link of that group.
  std::map<std::tuple<unsigned, unsigned, unsigned, int>, Value> crossData;
  // Without `forward`, a write is not forwarded to a read of its entry in
  // the same host cycle: the schedule never reads it there. With `after`,
  // it is forwarded past the read register instead of into it: each piece's
  // hit and written bits are given there (at the piece's offset) for the
  // reader to register and choose behind its read register, and the read
  // is the memory's alone.
  // With `late`, the link is written a host cycle after the pass, from a
  // register of the producer's bits that every such link of them shares, so
  // the producer's path into the links ends at one register, not at every
  // reader's memory; with `forwardLate`, that write is forwarded to the read
  // of its host cycle (from the register: shallow).
  struct AfterForward {
    unsigned offset, width;
    Value hit, data;
  };
  // A producer's bits registered (the producer, its low bit and width).
  std::map<std::tuple<int, unsigned, unsigned>, Value> writtenBits;
  auto writtenOf = [&](unsigned p, unsigned low, Value data) -> Value {
    unsigned w = data.getType().getIntOrFloatBitWidth();
    Value &written = writtenBits[{int(p), low, w}];
    if (!written)
      written = reg(data, producers[p].name + "_written_" + Twine(low));
    return written;
  };
  auto linkRead = [&](unsigned p, unsigned low, unsigned width,
                      function_ref<Value()> address, function_ref<Value()> bank,
                      const Twine &linkName, int to,
                      std::pair<Value, Value> *now = nullptr,
                      bool forward = true,
                      SmallVectorImpl<AfterForward> *after = nullptr,
                      bool late = false, bool forwardLate = false) -> Value {
    auto &producer = producers[p];
    unsigned c = producer.cls;
    auto &ctl = controls[c];
    unsigned threads = classes[c].threads();
    bool banked = !producer.late;
    bool crossing = to >= 0 && !slrOf.empty() && slrOf[c] != to;
    assert(!(crossing && now) && "a link between SLR groups is never zero");
    Value bits = asBits(b, copies[c].getResult(producer.port));
    std::string baseName = linkName.str();
    SmallVector<Value> pieces, nowPieces;
    for (unsigned at = 0; at < width; at += regionBits) {
      unsigned w = std::min<unsigned>(regionBits, width - at);
      std::string name =
          width > regionBits ? baseName + "_" + std::to_string(at) : baseName;
      Value data = b.createOrFold<comb::ExtractOp>(b.getIntegerType(w), bits,
                                                   low + at);
      if (crossing) {
        Value &registered = crossData[{p, low + at, w, to}];
        if (!registered) {
          if (auto *op = data.getDefiningOp())
            tagGroup(op, slrOf[c]);
          registered = copyReg(data, producer.name + "_slr" + Twine(to) + "_" +
                                         Twine(low + at));
          tagGroup(registered.getDefiningOp(), slrOf[c]);
        }
        data = registered;
      }
      auto &group = linkGroupFor(c, to, w);
      unsigned level = producer.atReady ? producer.readyLevel : 0;
      if (level && group.laterReady.size() < level)
        group.laterReady.resize(level);
      Value &enable = !producer.atReady ? group.state
                      : level           ? group.laterReady[level - 1]
                                        : group.ready;
      if (!enable) {
        std::string pass = !producer.atReady ? "_state"
                           : level ? "_ready_l" + std::to_string(level) + "_"
                                   : "_ready";
        enable = copyReg(
            copyReg(!producer.atReady ? ctl.state2
                    : level           ? ctl.laterReady2[level - 1]
                                      : ctl.ready2,
                    group.prefix + pass + "1"),
            group.prefix + pass + "0");
        if (crossing)
          enable = copyReg(enable, group.prefix + pass + "_late");
      }
      if (now && !banked) {
        if (!now->first)
          now->first = enable;
        nowPieces.push_back(data);
      }
      // What the link is written with, and when.
      Value writeData = data, writeEnable = enable;
      if (late) {
        writeData = writtenOf(p, low + at, data);
        if (producer.atReady && group.readyLate.size() <= level)
          group.readyLate.resize(level + 1);
        Value &lateEnable =
            producer.atReady ? group.readyLate[level] : group.stateLate;
        if (!lateEnable)
          lateEnable = copyReg(
              enable, group.prefix +
                          (!producer.atReady ? "_state"
                           : level ? "_ready_l" + std::to_string(level)
                                   : std::string("_ready")) +
                          "_late");
        writeEnable = lateEnable;
      }
      if (threads == 1 && !banked) {
        auto next = backedges.get(data.getType());
        Value r = reg(next, name);
        Value written = comb::MuxOp::create(b, writeEnable, writeData, r);
        next.setValue(written);
        Value value = r;
        if (late && forwardLate)
          value = written;
        if (oneCycleLinks && forward) {
          if (after)
            after->push_back({at, w, enable, data});
          else
            value = late ? comb::MuxOp::create(b, enable, data, value).getResult()
                         : written;
        }
        pieces.push_back(value);
        continue;
      }
      // An early link's address is its bank above the instance's index, so
      // the bank of 1 starts past every index: with instances not a power of
      // two, beyond the last instance.
      unsigned indices = threads > 1 ? llvm::PowerOf2Ceil(threads) : 1;
      unsigned depth = banked ? 2 * indices : threads;
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
          if (crossing)
            group.bank = copyReg(group.bank, group.prefix + "_bank_late");
        }
        writeAddress.push_back(group.bank);
        readAddress.push_back(bank());
      }
      if (threads > 1) {
        if (!group.inst) {
          group.inst = copyReg(copyReg(ctl.inst2, group.prefix + "_inst1"),
                               group.prefix + "_inst0");
          if (crossing)
            group.inst = copyReg(group.inst, group.prefix + "_inst_late");
        }
        writeAddress.push_back(group.inst);
        readAddress.push_back(address());
      }
      Value writeAt = b.createOrFold<comb::ConcatOp>(writeAddress);
      Value readAt = b.createOrFold<comb::ConcatOp>(readAddress);
      Value lateAt;
      if (late) {
        SmallVector<Value> lateAddress;
        if (banked) {
          if (!group.bankLate)
            group.bankLate = copyReg(group.bank, group.prefix + "_bank_late");
          lateAddress.push_back(group.bankLate);
        }
        if (threads > 1) {
          if (!group.instLate)
            group.instLate = copyReg(group.inst, group.prefix + "_inst_late");
          lateAddress.push_back(group.instLate);
        }
        lateAt = b.createOrFold<comb::ConcatOp>(lateAddress);
      }
      seq::FirMemWriteOp::create(b, memory, late ? lateAt : writeAt, hostClock,
                                 writeEnable, writeData, Value{});
      Value read =
          seq::FirMemReadOp::create(b, memory, readAt, hostClock, constant(1, 1));
      if (late && forwardLate) {
        // The registered write lands a host cycle on: forwarded from the
        // register to the read of its host cycle.
        Value hit = comb::AndOp::create(
            b, writeEnable,
            comb::ICmpOp::create(b, comb::ICmpPredicate::eq, lateAt, readAt));
        read = comb::MuxOp::create(b, hit, writeData, read);
      }
      if (now && !banked && nowPieces.size() == 1) {
        // The address is the next slot's reader's; this slot's, a cycle on.
        Value readNow = readAt.getDefiningOp<hw::ConstantOp>()
                            ? readAt
                            : copyReg(readAt, name + "_now");
        now->first = comb::AndOp::create(
            b, enable,
            comb::ICmpOp::create(b, comb::ICmpPredicate::eq, writeAt, readNow));
      }
      if (oneCycleLinks && forward) {
        // A write reaches a read of its entry in the same host cycle.
        Value hit = comb::AndOp::create(
            b, enable,
            comb::ICmpOp::create(b, comb::ICmpPredicate::eq, writeAt, readAt));
        if (after)
          after->push_back({at, w, hit, data});
        else
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

  // What each class evaluates at each position of the frame: its instance,
  // its passes, and whether the slot is the frame before's. Every control is
  // a fixed delay of the position, so a link's forwarding and its reader's
  // taking around the read register are needed only where the schedule
  // brings a write and a read of its entry together; elsewhere they are
  // logic on the producer's path into the read register that never decides
  // a value (and, across a zero-latency link into the producer, a path from
  // one class through another back into the first).
  // A ready pass is one of the class's levels (`readyLevels`), each at its
  // own slots, as the slot tables have them.
  struct Evaluation {
    unsigned inst = 0;
    /// The ready passes evaluated, by level.
    uint64_t ready = 0;
    bool state = false, tail = false;
    bool runs() const { return ready || state; }
  };
  SmallVector<SmallVector<Evaluation>> frameAt(classes.size());
  for (auto [c, cls] : llvm::enumerate(classes)) {
    frameAt[c].resize(ii);
    for (unsigned i = 0; i < cls.threads(); ++i)
      for (auto pass : {kReady, kState})
        for (unsigned l = 0; l < (pass == kReady ? cls.readyLevels : 1); ++l) {
          unsigned s =
              pass == kState ? cls.slot[kState][i] : cls.readySlots(l)[i];
          auto &at = frameAt[c][s % ii];
          at.inst = i;
          if (pass == kState)
            at.state = true;
          else
            at.ready |= uint64_t(1) << l;
          at.tail = s >= ii;
        }
  }
  // Whether a producer writes its links at position `at`.
  auto writesAt = [&](const Producer &producer, unsigned at) {
    auto &e = frameAt[producer.cls][at];
    return producer.atReady ? bool((e.ready >> producer.readyLevel) & 1)
                            : e.state;
  };
  // Whether a read of producer `p`'s link by class `c`, for the instances
  // `takes` holds, at producer instance `from[i]`, ever needs a write
  // forwarded: the write lands `lands` host cycles after its position's
  // (one, or two registered), and the read register loaded in the host cycle
  // it lands is used in the next one, by such an instance at its entry (for
  // an early link, in the bank that read takes: the write's frame's next).
  // Between frames the position may idle, so each host cycle after a
  // frame's last position is the next frame's first, or idle. With bubbles
  // (an `advance` input, or virtual memories) any host cycle may idle: the
  // read `lands` host cycles on is then of a position 1 to `lands` steps on.
  bool stalls = bool(advance);
  auto needsForward = [&](unsigned p, unsigned c, ArrayRef<bool> takes,
                          ArrayRef<uint64_t> from, unsigned lands = 1) {
    auto &producer = producers[p];
    bool banked = !producer.late;
    bool threaded = classes[producer.cls].threads() > 1;
    for (unsigned at = 0; at < ii; ++at) {
      if (!writesAt(producer, at))
        continue;
      auto &write = frameAt[producer.cls][at];
      for (unsigned steps = stalls ? 1 : lands; steps <= lands; ++steps) {
      // The positions `steps` host cycles on (`ii` idle), and the frames
      // they are past the write's.
      SmallVector<std::pair<unsigned, int>> reach{{at, 0}};
      for (unsigned step = 0; step < steps; ++step) {
        SmallVector<std::pair<unsigned, int>> next;
        for (auto [pos, frames] : reach) {
          if (pos + 1 < ii) {
            next.push_back({pos + 1, frames});
            continue;
          }
          next.push_back({0, frames + 1});
          next.push_back({ii, frames});
        }
        reach = std::move(next);
      }
      for (auto [pos, frames] : reach) {
        if (pos >= ii)
          continue;
        auto &read = frameAt[c][pos];
        if (!read.runs() || !takes[read.inst])
          continue;
        if (threaded && from[read.inst] != write.inst)
          continue;
        // The write's token is of the frame after its own, the read's of
        // its own.
        int written = 1 - int(write.tail);
        int wanted = frames - int(read.tail);
        if (banked && (wanted - written) % 2 != 0)
          continue;
        return true;
      }
      }
    }
    return false;
  };
  // Whether a reader of class `c` ever takes producer `p`'s write around its
  // read register: a write at a position that evaluates an instance
  // `takes` holds, at its entry.
  auto takesNow = [&](unsigned p, unsigned c, ArrayRef<bool> takes,
                      ArrayRef<uint64_t> from) {
    auto &producer = producers[p];
    bool threaded = classes[producer.cls].threads() > 1;
    for (unsigned at = 0; at < ii; ++at) {
      if (!writesAt(producer, at))
        continue;
      auto &read = frameAt[c][at];
      if (!read.runs() || !takes[read.inst])
        continue;
      if (threaded && from[read.inst] != frameAt[producer.cls][at].inst)
        continue;
      return true;
    }
    return false;
  };
  // Whether the hit of such a take (a write at the entry the evaluated
  // instance's row names) comes only at positions evaluating an instance
  // `takes` holds: then no other source's instance needs it masked.
  auto hitsOnlyTakes = [&](unsigned p, unsigned c, ArrayRef<bool> takes,
                           ArrayRef<uint64_t> from) {
    auto &producer = producers[p];
    bool threaded = classes[producer.cls].threads() > 1;
    for (unsigned at = 0; at < ii; ++at) {
      auto &read = frameAt[c][at];
      if (!writesAt(producer, at) || !read.runs() || takes[read.inst])
        continue;
      if (!threaded || from[read.inst] == frameAt[producer.cls][at].inst)
        return false;
    }
    return true;
  };

  // The link reads built: those forwarded (past the read register), those
  // taken in the producer's host cycle, and those written from a shared
  // register (forwarded the host cycle the write lands), for the report.
  unsigned linkReads = 0, forwardedReads = 0, forwardedAfter = 0, nowReads = 0,
           lateReads = 0, forwardedLate = 0;
  // The class readers of each producer: a producer's bits read by several
  // are written to their links from one register of them.
  SmallVector<unsigned> classReaders(producers.size(), 0);
  for (auto &consumer : consumers) {
    if (consumer.cls < 0)
      continue;
    std::set<int> read;
    for (auto &pieces : consumer.pieces)
      for (auto &piece : pieces)
        if (piece.producer >= 0 && producers[piece.producer].cls >= 0)
          read.insert(piece.producer);
    for (int p : read)
      ++classReaders[p];
  }

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
    int to = slrOf.empty() ? -1 : slrOf[c];
    // What this input builds is in its class's SLR group, but for the
    // producers' registers of links from other groups.
    Operation *mark = body->empty() ? nullptr : &body->back();
    llvm::scope_exit tagBuilt([&] {
      if (to < 0)
        return;
      for (auto *op = mark ? mark->getNextNode() : &body->front(); op;
           op = op->getNextNode())
        if (!isa<hw::ConstantOp>(op))
          tagGroup(op, to);
    });
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
    SmallVector<SmallVector<AfterForward>> afterAt;
    // A read forwarded past its register where the reader's input stays
    // shallow behind it and reaches no output: the producer's path ends at a
    // register of its written bits, not at the read register through the
    // choice.
    // (Not with SLR groups: the registers are the producer's, shared by
    // readers in every group.)
    bool afterForward =
        forwardAfterLevels > 0 && slrOf.empty() &&
        !cls.inputToOutput.test(consumer.port) &&
        cls.inputDepth[consumer.port] + 1 < int(forwardAfterLevels);
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
      SmallVector<bool> nowAlone(sources.size(), false);
      SmallVector<SmallVector<AfterForward>> sourceAfter(sources.size());
      // A source's producer instance by instance, and the instances taking it.
      auto rowsOf = [&](std::pair<int, unsigned> key,
                        SmallVectorImpl<uint64_t> &rows,
                        SmallVectorImpl<bool> &takes) {
        rows.assign(threads, 0);
        takes.assign(threads, false);
        for (unsigned i = 0; i < threads; ++i)
          if (sourceOf[i] == key) {
            rows[i] = producerInst[i];
            takes[i] = true;
          }
      };
      // Forwarded past the read register only where no source is taken in
      // its producer's host cycle: a taken write is chosen there too, and the
      // late value then passes one choice, not two.
      bool after = afterForward;
      for (auto &[key, index] : sources) {
        if (!after || key.first < 0 || producers[key.first].cls < 0)
          continue;
        SmallVector<uint64_t> rows;
        SmallVector<bool> takes;
        rowsOf(key, rows, takes);
        if (zeroLink(key.first, c, consumer.port) &&
            takesNow(key.first, c, takes, rows))
          after = false;
      }
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
        SmallVector<uint64_t> rows;
        SmallVector<bool> takes;
        rowsOf(key, rows, takes);
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
        bool now = zeroLink(key.first, c, consumer.port) &&
                   takesNow(key.first, c, takes, rows);
        nowAlone[index] = now && hitsOnlyTakes(key.first, c, takes, rows);
        // A link between SLR groups is written a host cycle later, from
        // registered bits in the producer's group: forwarded wherever links
        // are, and never from a register shared with other groups' readers.
        bool crossing = to >= 0 && slrOf[producer.cls] != to;
        bool forward = oneCycleLinks &&
                       (crossing || needsForward(key.first, c, takes, rows));
        // Registered for the link writes where several classes read the
        // bits, or where a forward past the read register registers them.
        bool late = shareLinkWrites && slrOf.empty() &&
                    (classReaders[key.first] > 1 || (forward && after));
        bool forwardLate = late && needsForward(key.first, c, takes, rows, 2);
        ++linkReads;
        forwardedReads += forward;
        forwardedAfter += forward && after;
        nowReads += now;
        lateReads += late;
        forwardedLate += forwardLate;
        sourceData[index] =
            linkRead(key.first, key.second, width, address, bank,
                     producer.name + "_to_" + consumer.name + "_" + Twine(at),
                     to, now ? &sourceNow[index] : nullptr, forward,
                     after ? &sourceAfter[index] : nullptr, late, forwardLate);
      }
      // Taken around the read register when written in the reader's slot:
      // the source of this slot's instance, a host cycle after the read's.
      std::pair<Value, Value> taken;
      unsigned nowSources =
          llvm::count_if(sourceNow, [](auto &v) { return bool(v.first); });
      if (nowSources) {
        auto alone = llvm::find(nowAlone, true);
        if (sources.size() == 1) {
          taken = sourceNow[0];
        } else if (nowSources == 1 && alone != nowAlone.end() &&
                   sourceNow[alone - nowAlone.begin()].first) {
          // One source is ever taken, and its hit only where it is the
          // instance's: no select of the source.
          taken = sourceNow[alone - nowAlone.begin()];
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
      // Forwarded past the read register: each piece's hit, of this source
      // where the instance's source is chosen, registered with the read, and
      // the written bits a host cycle on, one register per producer's piece
      // whatever reads it.
      SmallVector<AfterForward> afterRead;
      for (auto &[key, index] : sources) {
        if (sourceAfter[index].empty())
          continue;
        Value mine;
        if (sources.size() > 1) {
          SmallVector<uint64_t> rows(threads, 0);
          for (unsigned i = 0; i < threads; ++i)
            rows[i] = sources[sourceOf[i]];
          unsigned w = llvm::Log2_32_Ceil(sources.size());
          mine = comb::ICmpOp::create(
              b, comb::ICmpPredicate::eq,
              copyReg(table(b, ctl.inst2, rows, w),
                      consumer.name + "_select_after" + Twine(at) + "_" +
                          Twine(index)),
              constant(w, index));
        }
        for (auto &piece : sourceAfter[index]) {
          Value hit = mine ? comb::AndOp::create(b, piece.hit, mine) : piece.hit;
          Value written =
              writtenOf(key.first, key.second + piece.offset, piece.data);
          afterRead.push_back(
              {piece.offset, piece.width,
               reg(hit, consumer.name + "_hit_" + Twine(at + piece.offset) +
                            "_" + Twine(index)),
               written});
        }
      }
      afterAt.push_back(std::move(afterRead));
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
        for (auto &piece : afterAt[r]) {
          // The piece's bits chosen behind the register, the rest kept.
          Value bits = parts[index];
          SmallVector<Value> high;
          if (piece.offset + piece.width < width)
            high.push_back(b.createOrFold<comb::ExtractOp>(
                b.getIntegerType(width - piece.offset - piece.width), bits,
                piece.offset + piece.width));
          high.push_back(comb::MuxOp::create(
              b, piece.hit, piece.data,
              b.createOrFold<comb::ExtractOp>(b.getIntegerType(piece.width),
                                              bits, piece.offset)));
          if (piece.offset)
            high.push_back(b.createOrFold<comb::ExtractOp>(
                b.getIntegerType(piece.offset), bits, 0));
          parts[index] = high.size() == 1
                             ? high[0]
                             : b.createOrFold<comb::ConcatOp>(high);
        }
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

  {
    std::string text;
    llvm::raw_string_ostream os(text);
    os << pageReport;
    os << "  link reads: " << linkReads << ", forwarded " << forwardedReads
       << " (past the read register " << forwardedAfter << "), taken now "
       << nowReads << "; written from a shared register " << lateReads
       << " (forwarded " << forwardedLate << ")\n";
    llvm::errs() << os.str();
    if (!reportPath.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream file(reportPath, ec, llvm::sys::fs::OF_Append);
      if (ec)
        return netlist.emitError("report: ") << ec.message();
      file << os.str();
    }
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
                              bank, producer.name + "_to_" + consumer.name,
                              /*to=*/-1));
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
    // A group's select output, kept from its push to the next frame's load:
    // set with the group's valid input of the same frame, it chooses the high
    // values of the pairs that frame loads. None is set out of reset.
    if (int g = specSelect[k]; g >= 0) {
      auto selNext = backedges.get(i1);
      Value sel = resetReg(selNext, consumer.name + "_select", 0);
      selNext.setValue(comb::MuxOp::create(b, push, data, sel));
      groupSelect[g].setValue(comb::AndOp::create(
          b, frameInput[specGroups[g].valid], sel));
    }
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

  // A position runs only when, for each class of virtual memories, the
  // instance it loads has no read pending, and each controller has room for
  // the requests the decisions in flight may issue. It is the position after
  // the decision's, without a bubble: the one a bubble would delay.
  if (virtualReady) {
    Value end = comb::OrOp::create(b, eq(posD, ii - 1), eq(posD, ii));
    Value next = comb::MuxOp::create(
        b, end,
        comb::MuxOp::create(b, go, constant(posWidth, 0), constant(posWidth, ii)),
        comb::AddOp::create(b, posD, constant(posWidth, 1)));
    SmallVector<Value> terms;
    for (auto [c, cls] : llvm::enumerate(classes)) {
      if (cls.virtualMemories.empty())
        continue;
      unsigned threads = cls.threads();
      unsigned idxWidth = threads > 1 ? llvm::Log2_32_Ceil(threads) : 0;
      SmallVector<uint64_t> useRow(ii + 1, 0), instRow(ii + 1, 0);
      for (unsigned i = 0; i < threads; ++i)
        for (auto pass : {kReady, kState}) {
          if (pass == kReady && !cls.hasReady)
            continue;
          unsigned s = cls.slot[pass][i];
          useRow[s % ii] = 1;
          instRow[s % ii] = i;
        }
      Value use = table(b, next, useRow, 1);
      Value inst = threads > 1 ? table(b, next, instRow, idxWidth) : Value();
      for (auto &vm : cls.virtualMemories) {
        Value pending = copies[c].getResult(vm.firstOutput + 8);
        Value bit = pending;
        if (inst) {
          unsigned entries = 1u << idxWidth;
          Value shift = b.createOrFold<comb::ConcatOp>(
              constant(entries - idxWidth, 0), inst);
          bit = b.createOrFold<comb::ExtractOp>(
              i1, comb::ShrUOp::create(b, pending, shift), 0);
        }
        terms.push_back(comb::OrOp::create(b, notOf(use), notOf(bit)));
      }
    }
    for (auto &v : simVirtuals)
      terms.push_back(comb::ICmpOp::create(
          b, comb::ICmpPredicate::uge, body->getArgument(v.firstInput + 3),
          constant(kVirtualFreeWidth, kVirtualRoom)));
    virtualReady.setValue(comb::AndOp::create(b, terms, false));
  }

  // Ports out: finishes, the inputs' readies, then the outputs' channels.
  SmallVector<Value> finishes;
  for (auto &ctl : llvm::reverse(controls))
    finishes.push_back(ctl.state);
  SmallVector<Value> results;
  results.push_back(b.createOrFold<comb::ConcatOp>(finishes));
  for (auto &[p, ready] : inputReady)
    results.push_back(ready);
  results.append(portOutputs.begin(), portOutputs.end());
  for (auto &v : simVirtuals) {
    auto &vm = classes[v.cls].virtualMemories[v.k];
    for (unsigned j = 0; j < 8; ++j)
      results.push_back(copies[v.cls].getResult(vm.firstOutput + j));
  }
  hw::OutputOp::create(b, results);

  SmallVector<Attribute> threads;
  for (auto &cls : classes)
    threads.push_back(
        b.getStringAttr(cls.name + "=" + Twine(cls.threads())));
  sim->setAttr("foldsim.threads", b.getArrayAttr(threads));
  // The virtual memories, for the kernel: name, class, instance index, row
  // address, data and mask widths, read-first, instances.
  if (!simVirtuals.empty()) {
    SmallVector<Attribute> virtuals;
    for (auto &v : simVirtuals) {
      auto &cls = classes[v.cls];
      auto &vm = cls.virtualMemories[v.k];
      unsigned instWidth =
          std::max(cls.threads() > 1 ? llvm::Log2_32_Ceil(cls.threads()) : 0u, 1u);
      virtuals.push_back(b.getStringAttr(
          vm.name + ":" + cls.name + ":" + Twine(instWidth) + ":" +
          Twine(vm.rowWidth) + ":" + Twine(vm.dataWidth) + ":" +
          Twine(vm.maskWidth) + ":" + Twine(vm.readFirst ? 1 : 0) + ":" +
          Twine(cls.threads())));
    }
    sim->setAttr("foldsim.vmems", b.getArrayAttr(virtuals));
  }
  sim->setAttr("foldsim.frame", b.getI64IntegerAttr(ii));
  simModule = sim;
  return success();
}

LogicalResult StaticFolder::extractGroups() {
  auto *context = top.getContext();
  auto *body = simModule.getBodyBlock();
  SmallVector<SmallVector<Operation *>> opsOf(slrGroups);
  for (auto &op : body->without_terminator())
    if (auto attr = op.getAttrOfType<IntegerAttr>(kSlrAttr)) {
      opsOf[attr.getInt()].push_back(&op);
      op.removeAttr(kSlrAttr);
    }
  // A name for the port of a value: its op's name, if any.
  auto nameOf = [&](Value value) -> std::string {
    if (auto arg = dyn_cast<BlockArgument>(value))
      return simModule.getInputName(arg.getArgNumber()).str();
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
  for (unsigned g = 0; g < slrGroups; ++g) {
    auto &ops = opsOf[g];
    if (ops.empty())
      continue;
    DenseSet<Operation *> inGroup(ops.begin(), ops.end());
    // Inputs: values from outside but constants, which are copied in;
    // outputs: values read outside.
    SmallVector<Value> inputs, outputs;
    DenseSet<Value> isInput;
    SetVector<Operation *> constants;
    for (auto *op : ops)
      for (auto operand : op->getOperands()) {
        auto *def = operand.getDefiningOp();
        if (def && inGroup.contains(def))
          continue;
        if (def && isa<hw::ConstantOp>(def)) {
          constants.insert(def);
          continue;
        }
        if (isInput.insert(operand).second)
          inputs.push_back(operand);
      }
    for (auto *op : ops)
      for (auto result : op->getResults())
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !inGroup.contains(user);
            }))
          outputs.push_back(result);

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
    for (auto [i, value] : llvm::enumerate(inputs))
      addPort(value, true, i);
    for (auto [i, value] : llvm::enumerate(outputs))
      addPort(value, false, i);
    OpBuilder mb(simModule);
    auto module = hw::HWModuleOp::create(
        mb, simModule.getLoc(),
        StringAttr::get(context, simName + "_slr" + Twine(g)),
        hw::ModulePortInfo(ports));
    module.setPrivate();
    auto *groupBody = module.getBodyBlock();
    groupBody->getTerminator()->erase();
    IRMapping map;
    for (auto [value, arg] : llvm::zip(inputs, groupBody->getArguments()))
      map.map(value, arg);
    OpBuilder gb = OpBuilder::atBlockEnd(groupBody);
    for (auto *constant : constants)
      gb.clone(*constant, map);
    SmallVector<Operation *> clones;
    for (auto *op : ops)
      clones.push_back(gb.clone(*op, map));
    // A graph region: an operand may be defined further down.
    for (auto *clone : clones)
      for (auto &operand : clone->getOpOperands())
        if (auto mapped = map.lookupOrNull(operand.get()))
          operand.set(mapped);
    SmallVector<Value> results;
    for (auto value : outputs)
      results.push_back(map.lookup(value));
    hw::OutputOp::create(gb, simModule.getLoc(), results);

    OpBuilder ib(body->getTerminator());
    auto instance = hw::InstanceOp::create(
        ib, simModule.getLoc(), module,
        StringAttr::get(context, "foldsim_slr" + Twine(g)), inputs);
    for (auto [value, result] : llvm::zip(outputs, instance.getResults()))
      value.replaceUsesWithIf(result, [&](OpOperand &use) {
        return !inGroup.contains(use.getOwner());
      });
    for (auto *op : ops)
      op->dropAllUses();
    for (auto *op : ops)
      op->erase();
  }
  return success();
}

LogicalResult StaticFolder::resolveEnvironment() {
  specOf.assign(producers.size(), -1);
  specSelect.assign(consumers.size(), -1);
  envSink.assign(consumers.size(), false);
  if (specDepth < 1 || specDepth > 2)
    return netlist.emitError("spec-depth: ") << specDepth << " is not 1 or 2";
  auto output = [&](StringRef name) -> int {
    for (auto [k, consumer] : llvm::enumerate(consumers))
      if (consumer.cls < 0 && consumer.name == name)
        return k;
    return -1;
  };
  auto input = [&](StringRef name) -> int {
    for (auto [p, producer] : llvm::enumerate(producers))
      if (producer.cls < 0 && producer.name == name)
        return p;
    return -1;
  };
  for (auto &name : envSinkNames) {
    int k = output(name);
    if (k < 0)
      return netlist.emitError("env-sinks: no netlist output '") << name << "'";
    envSink[k] = true;
  }
  for (auto &spec : specGroupSpecs) {
    auto [selectName, rest] = StringRef(spec).split('=');
    SmallVector<StringRef> names;
    rest.split(names, ':', -1, false);
    int select = output(selectName);
    if (select < 0 || names.empty())
      return netlist.emitError("spec-groups: bad group '") << spec << "'";
    if (consumers[select].width != 1)
      return netlist.emitError("spec-groups: select '")
             << selectName << "' is not one bit";
    unsigned g = specGroups.size();
    int valid = -1;
    for (auto name : names) {
      int p = input(name);
      if (p < 0)
        return netlist.emitError("spec-groups: no netlist input '") << name << "'";
      if (specOf[p] >= 0)
        return netlist.emitError("spec-groups: input '") << name
               << "' in two groups";
      specOf[p] = g;
      if (valid < 0)
        valid = p;
    }
    if (producers[valid].width != 1)
      return netlist.emitError("spec-groups: valid '")
             << names.front() << "' is not one bit";
    specSelect[select] = g;
    specGroups.push_back({unsigned(select), unsigned(valid)});
  }
  return success();
}

LogicalResult StaticFolder::run() {
  inlineNetlists();
  if (failed(collect()) || failed(resolveEnvironment()))
    return failure();
  for (auto &cls : classes)
    if (cls.threads() > 1 &&
        failed(foldPipelines(cls.model, cls.threads(), cls.folding)))
      return failure();
  for (auto &cls : classes)
    if (failed(analyze(cls)))
      return failure();
  // Without a schedule, or with one only, nothing is built: the folded
  // pipelines' edges stay unresolved.
  auto abandonFolds = [&] {
    for (auto &cls : classes)
      if (cls.folding && cls.folding->edges)
        cls.folding->edges->abandon();
  };
  if (failed(parsePages()) || failed(schedule()) || failed(parseStorage())) {
    abandonFolds();
    return failure();
  }
  if (scheduleOnly) {
    abandonFolds();
    return success();
  }
  for (auto &cls : classes)
    if (failed(buildCopy(cls)))
      return failure();
  for (auto &rule : storageRules)
    if (!rule.regs && !rule.used)
      return netlist.emitError("storage: no memory '")
             << rule.store << "' in model '" << rule.model << "'";
  if (failed(buildSim()))
    return failure();
  if (!slrOf.empty() && failed(extractGroups()))
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
                      modulo, interleaveCopies, zeroLatencyLinks, zeroLatencyChains,
                      shortLead, bubbles, virtualMemories, stagedEnv,
                      specGroups, specDepth,
                      envSinks, registerStateEntries, slrGroups, slrSites,
                      slrCapacity, slrAssign, storage, readyLevels,
                      forwardAfterLevels, shareLinkWrites, page);
  if (failed(folder.run()))
    return signalPassFailure();
}
