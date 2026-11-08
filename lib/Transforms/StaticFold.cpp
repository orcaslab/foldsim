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

#include <chrono>
#include "FAME5.h"
#include "PassSplit.h"
#include "Pipelines.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "foldsim/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
#include "llvm/Support/Format.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>
#include <optional>
#include <map>
#include <numeric>
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
  /// Pure logic: no registers, memories or black boxes, and so it needs no
  /// clock. Its outputs are late or constant; its state pass writes no state
  /// and a ready pass computes the same outputs a slot of its own earlier,
  /// which the search for ready passes weighs as for any class.
  bool stateless = false;
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
  /// Staged evaluation (`stage-levels`): the stage registers its logic is
  /// cut by. Its state and inputs are read this many positions before its
  /// state pass, whose writes are as without stages; its state gap counts
  /// from the state pass to the next evaluation's read. 0: none.
  unsigned stages = 0;
  /// Multicycle (`multicycle`): host cycles from the capture registers to
  /// the end registers, the stages less one (0: none, or stage registers).
  unsigned multicycle = 0;
  /// Multicycle: the bits of its capture and end registers.
  uint64_t multicycleCaps = 0, multicycleEnds = 0;
  /// Pass splitting (`pass-split`): the passes of an evaluation (0: none)
  /// and the host cycles of each (the multicycle span without splitting).
  unsigned passSplit = 0, passSpan = 0;
  /// Pass splitting: host cycles from the state pass to the deferred
  /// registers' writes, (passSplit + 1) x passSpan (0: none).
  unsigned deferSpan = 0;
  uint64_t passHoldBits = 0;
  /// With --vmem-skip-held-reads: a virtual memory of it reads at a held
  /// address (heldReadRow), so its requests leave from registers a host
  /// cycle after the state pass, given a state gap of the lead + 2.
  bool heldReads = false;
  /// Whether its memories include a registered read: its state is then read
  /// a host cycle later after a state pass.
  bool syncReads = false;
  /// Its memories kept outside the simulator (`--virtual-memories`): each
  /// one's row address, data and mask widths, read-during-write order, and
  /// where its request and response ports start among the copy's.
  /// With --vmem-merge-lanes, the arrays split from one target RAM (the same
  /// read address and enable, the same write address) are one memory: its
  /// lanes, each at a whole-byte offset of the row, and enabled by its own
  /// write enable (byte enables).
  struct VirtualMemory {
    std::string name;
    unsigned rowWidth, dataWidth, maskWidth;
    bool readFirst;
    unsigned firstInput, firstOutput;
    SmallVector<seq::FirMemOp> lanes;
    SmallVector<unsigned> laneOffset;
    /// With --vmem-skip-held-reads, where the read's address is a registered
    /// one (heldReadRow): an op in the model computing the row the previous
    /// target cycle read, else null.
    Value heldRow;
    /// Its requests leave a host cycle after the state pass, from registers
    /// (the skip computed there): the room and the state gap take a host
    /// cycle more.
    bool delayed = false;
    /// A column of a page word kept off chip (`pages=hbm`): its instances
    /// are {instance, page} (`instWidth` bits, `instances` of them), a row
    /// each, read at the fetch's positions and waited for by the state pass
    /// alone; 0 for a memory of the model (the class's instances).
    bool pages = false;
    unsigned instWidth = 0, instances = 0;
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
  /// to move in or out of its page RAM, or 0 if it is not paged; whether
  /// its pages are kept off chip (`<model>:pages=hbm`), behind virtual
  /// memories; and the slots from an instance's first page read to its
  /// state pass: the page, and with pages off chip at least 2, so that the
  /// read is pending before the state pass is decided.
  unsigned page = 0;
  bool pagesOffChip = false;
  /// Paged links (`link-kinds` with a page): the most host cycles a link
  /// it reads or writes moves an entry in, page by page; every pass of the
  /// copy then holds its link port that many slots, so its passes are at
  /// least that far apart (0: none).
  unsigned linkPage = 0;
  /// Off chip with a window (`page-hbm-window`): its slots (0: none), and
  /// each instance's slot.
  unsigned pageWindow = 0;
  SmallVector<unsigned> windowSlot;
  /// Off chip on engines with `page-write-gate`: each page's write-back bits
  /// through a gated register, the column's write data their OR; its page
  /// writes two host cycles after the state pass at the earliest.
  bool pageGate = false;
  /// The fewest host cycles from a state pass to the instance's next
  /// evaluation its stores allow (StaticFolder::minStateGap), once known.
  mutable int minGap = -1;
  unsigned pageLead() const {
    return pagesOffChip && page ? std::max(page, 2u) : page;
  }
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
/// many bits (VMEM_INTERFACE.md, guarantee 2: StaticFolder::virtualRoom).
static constexpr unsigned kVirtualFreeWidth = 16;
/// The control stages from a slot table's read to its pass: the state and
/// ready passes' two staged registers and the one that makes them now.
static constexpr int kControlStages = 3;

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

/// What a class evaluates at a position of the frame: its instance, its
/// passes, and whether the slot is the frame before's.
struct Evaluation {
  unsigned inst = 0;
  /// The ready passes evaluated, by level.
  uint64_t ready = 0;
  bool state = false, tail = false;
  bool runs() const { return ready || state; }
};

class StaticFolder {
public:
  /// The modulo search's periods tried one at a time before it skips ahead
  /// (pass option modulo-linear).
  unsigned moduloLinear = 16;
  /// `pass-split`: <glob>:<passes>.
  SmallVector<std::string> passSplitSpecs;
  /// `link-hbm-lead`: the most positions a stream's request precedes its
  /// landing.
  unsigned linkHbmLead = kLinkHbmLead;
  /// `link-hbm-shift`: the most positions a stream's request moves earlier,
  /// or its write later, to share a page engine (0: none).
  unsigned linkHbmShift = 0;
  /// `fold-ports` / `fold-ports-reset`: the port bundles folded into record
  /// streams (empty: none).
  std::string foldPortsSpec, foldPortsReset = "reset";
  /// `link-hbm-fit-slots`: a stream on the page engines takes the fewest
  /// landing slots its requests need.
  bool linkHbmFitSlots = false;
  /// `page-write-gate`: off-chip pages on engines written back through
  /// per-page gated registers and an OR (choosePageGates, Class::pageGate).
  bool pageWriteGate = false;
  /// `link-hbm-min-lead`: the least room (positions from a row's write to its
  /// landing) a member is streamed with (0: the decision lead + 6). A
  /// member with less room is built as an on-chip link: its landing would
  /// wait for the memory's round trip every frame.
  unsigned linkHbmMinLeadOpt = 0;
  /// `link-hbm-row-bits`: the most bits of an HBM link stream's row (its
  /// controller's DW): a wider word is cut into rows (pages) of this many.
  unsigned linkHbmRowBits = 512;
  /// `min-ii` (with modulo): the shortest frame the search starts from; the
  /// positions beyond the schedule's own stay idle. A longer frame is a
  /// longer prefetch distance for every request a pass issues (for studies
  /// of latency hiding, or a host rate matched to another simulator's).
  unsigned minII = 0;
  /// `slr-serialize`: <glob>:<k> or <k> entries.
  std::vector<std::string> slrSerializeSpecs;
  /// `ready-choice`: the classes with a ready pass, given instead of
  /// searched ("split:" first: ready outputs split by level); empty: search.
  std::string readyChoice;
  /// `staged-link-memories` (with `staged-links`): see stagedMemories.
  bool stagedLinkMemories = false;
  StaticFolder(ModuleOp top, hw::HWModuleOp netlist, StringRef simName,
               unsigned regionBits, StringRef report,
               ArrayRef<std::string> copies, bool nest, bool scheduleOnly,
               bool oneCycleLinks, bool overlapFrames, bool modulo,
               bool interleaveCopies, bool zeroLatencyLinks,
               bool zeroLatencyChains, bool shortLead, bool bubbles,
               StringRef virtualMemories, bool mergeLanes, bool skipHeldReads,
               bool stagedEnv, ArrayRef<std::string> specGroups, unsigned specDepth,
               ArrayRef<std::string> envSinks, unsigned registerStateEntries,
               unsigned slrGroups, ArrayRef<std::string> slrSites,
               uint64_t slrCapacity, ArrayRef<std::string> slrCapacities,
               ArrayRef<std::string> slrAssign, bool sllEstimate,
               uint64_t sllBudget,
               ArrayRef<std::string> storage, StringRef readyLevels,
               unsigned forwardAfterLevels, bool shareLinkWrites,
               unsigned zeroChainHops, ArrayRef<std::string> page,
               bool pageMasks, unsigned pageHbmColumnBits,
               unsigned pageHbmWindow, unsigned pageEngines,
               unsigned pageMemoryBits, StringRef pageMemories, StringRef pageRows, bool compactLinks,
               ArrayRef<std::string> registeredLinks, bool shareLinks,
               bool stagedLinks, unsigned linkBlockDepth,
               unsigned linkUltraDepth, StringRef linkKindSpec,
               unsigned stageLevels, ArrayRef<std::string> multicycle)
      : top(top), netlist(netlist), simName(simName), regionBits(regionBits),
        reportPath(report), copies(copies), nest(nest),
        specGroupSpecs(specGroups), envSinkNames(envSinks),
        specDepth(specDepth), slrGroups(slrGroups), slrSiteSpecs(slrSites),
        slrCapacity(slrCapacity), slrCapacitySpecs(slrCapacities),
        slrAssignSpecs(slrAssign), sllEstimate(sllEstimate),
        sllBudget(sllBudget),
        scheduleOnly(scheduleOnly), oneCycleLinks(oneCycleLinks),
        overlapFrames(overlapFrames), modulo(modulo),
        interleaveCopies(interleaveCopies), zeroLatencyLinks(zeroLatencyLinks),
        zeroLatencyChains(zeroLatencyChains), shortLead(shortLead),
        bubbles(bubbles), virtualMemoryPattern(virtualMemories.str()),
        mergeLanes(mergeLanes), skipHeldReads(skipHeldReads),
        readyLevelsMode(readyLevels.str()),
        registerStateEntries(registerStateEntries), storageSpecs(storage),
        forwardAfterLevels(forwardAfterLevels),
        shareLinkWrites(shareLinkWrites), zeroChainHops(zeroChainHops),
        pageSpecs(page), pageMasks(pageMasks),
        pageHbmColumnBits(pageHbmColumnBits), pageHbmWindow(pageHbmWindow),
        pageEngineCap(pageEngines),
        pageMemoryBits(pageMemoryBits), pageMemories(pageMemories.str()),
        pageRows(pageRows.str()), compactLinks(compactLinks),
        registeredLinkSpecs(registeredLinks), shareLinks(shareLinks),
        stagedLinks(stagedLinks), linkBlockDepth(linkBlockDepth),
        linkUltraDepth(linkUltraDepth), linkKindSpec(linkKindSpec.str()),
        stageLevels(stageLevels), multicycleSpecs(multicycle),
        envRoundTrip((stagedEnv ? 1 : 0) + 1 + decisionLead()),
        linkLatency(oneCycleLinks ? kForwardedLinkLatency : kLinkLatency) {}
  LogicalResult run();

private:
  /// Host cycles from the decision on a position (the sequencer's `advance`,
  /// taken on what the controllers' `free` shows) to the passes it runs: the
  /// decision's register (none with short-lead, whose slot tables read the
  /// position as it is decided), then the control stages. A pass's virtual
  /// requests leave in that host cycle.
  int decisionLead() const { return (shortLead ? 0 : 1) + kControlStages; }
  /// The free request entries a position needs to run: a `free` the
  /// sequencer sees does not yet count the requests of the decisions still in
  /// flight (one a host cycle, decisionLead() of them) nor those of its own
  /// decision, each at most a read and a write; a memory whose requests leave
  /// a host cycle after the pass (`delayed`) has one decision more in flight.
  unsigned virtualRoom(bool delayed = false) const {
    return 2 * (decisionLead() + 1 + (delayed ? 1 : 0));
  }

  void inlineNetlists();
  LogicalResult collect();
  void propagateConstants();
  LogicalResult resolveEnvironment();
  /// `fold-ports`: the netlist's port bundles. A bundle's outputs of a
  /// target cycle leave as one request record and its inputs come as one
  /// response record, each on a stream shared by the bundles and tagged with
  /// the bundle; the fields of a record in netlist order, low first.
  struct PortFold {
    bool on = false;
    unsigned ports = 0, chanWidth = 1, reqWidth = 0, rspWidth = 0;
    struct Field {
      std::string name;
      unsigned width = 0, lo = 0;
    };
    SmallVector<Field> req, rsp;
    /// The consumer of (bundle, request field), the producer of (bundle,
    /// response field).
    SmallVector<SmallVector<unsigned>> reqOf, rspOf;
    /// Each consumer's and producer's bundle and field (-1: not folded).
    SmallVector<int> consumerPort, consumerField, producerPort, producerField;
    /// The netlist input whose frame token rides on each request record.
    int reset = -1;
  } portFold;
  LogicalResult resolvePortFold();
  /// The pushes of a folded field, one bundle a host cycle: each bundle's
  /// at its own slot or the first after the previous bundle's.
  void staggerPortPushes();
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
    /// The late outputs moved to their state pass (`stateOnly`).
    DenseSet<unsigned> moved;
    /// The modulo search of this choice: its lower bound, periods tried.
    unsigned lower = 0, tried = 0;
  };
  Choice searchReady();
  /// Assigns the classes to SLR groups (`slrOf`): within the capacity, the
  /// shortest frame, then the fewest link bits between groups.
  LogicalResult chooseGroups();
  LogicalResult parseSlrSpecs(unsigned groups);
  LogicalResult estimateSlls();
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
  /// `slr-serialize`: the host cycles a link of class `c` between groups
  /// takes, its bits sent a k-th at a time (1: unserialized).
  SmallVector<unsigned> serialK;
  bool slrSerialize = false;
  LogicalResult parseSerialize();
  unsigned serialOf(int c) const {
    return c >= 0 && unsigned(c) < serialK.size() ? serialK[c] : 1;
  }
  /// The wires (SLLs) `bits` of class `p`'s links take between groups.
  uint64_t serialWires(int p, uint64_t bits) const {
    unsigned k = serialOf(p);
    return (bits + k - 1) / k;
  }
  /// The host cycles a link of class `c` takes between groups: 1 (its
  /// register), or with slr-serialize k + 1 (the pass's bits held, then k
  /// chunks sent, each in a register fed only by registers).
  unsigned crossLatency(int c) const {
    unsigned k = serialOf(c);
    return k > 1 ? k + 1 : 1;
  }
  int crossDelay(int p, int c) const {
    return crosses(p, c) ? int(crossLatency(p)) : 0;
  }
  /// `slr-serialize`: the spacing of each class's passes of one kind (its
  /// k, where a link of it crosses groups and k > 1; else 0).
  SmallVector<unsigned> serialSpacing() const {
    SmallVector<unsigned> space(classes.size(), 0);
    if (!slrSerialize || slrOf.empty())
      return space;
    for (auto &consumer : consumers) {
      if (consumer.cls < 0)
        continue;
      for (auto &pieces : consumer.pieces)
        for (auto &piece : pieces) {
          if (piece.producer < 0)
            continue;
          int p = producers[piece.producer].cls;
          if (p >= 0 && crosses(p, consumer.cls) && serialOf(p) > 1)
            space[p] = serialOf(p);
        }
    }
    return space;
  }
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
                 ReadyWrites &writes, unsigned below = 0);
  /// The evaluations of the last cycle `place` found, for the search for
  /// ready passes and the error.
  std::string lastCycle;
  SmallVector<Node> lastCycleNodes;
  /// The search for ready passes: the choices placed, and those not placed
  /// as no shorter than the best frame by their bound.
  unsigned searchPlaced = 0, searchPruned = 0;
  /// What each instance of each class reads from late outputs, input by
  /// input in order, with the link's latency: the same for every choice of
  /// ready passes, so built once for a search (`frameGraph`); cleared where
  /// the links' latencies change (chain cuts, SLR groups).
  struct LateRead {
    unsigned input, producer, inst;
    int latency;
  };
  std::vector<std::vector<SmallVector<LateRead, 4>>> lateReads;
  /// Late outputs written at their class's state pass even with a ready
  /// pass: what the search for ready passes moved out of a cycle.
  DenseSet<unsigned> stateOnly;
  /// The outputs written at the ready pass of the k-th evaluation of a cycle
  /// that read the evaluation before it: what moving to the state pass cuts.
  SmallVector<unsigned> readersOf(ArrayRef<Node> cycle, unsigned k,
                                  const ReadyWrites &writes);
  /// Names a shortest cycle of the evaluations of the given choice of ready
  /// passes, edge by edge, on the diagnostic: which link each waits on, and
  /// whether a ready pass could break it.
  void explainCycle(ArrayRef<bool> ready, InFlightDiagnostic &diag);
  /// Places the evaluations with consecutive frames overlapped: the shortest
  /// frame at which every copy's evaluations fit, modulo the frame, and every
  /// dependence across frames holds, shorter than `below`; 0 if none does.
  unsigned placeModulo(
      ArrayRef<Node> nodes,
      ArrayRef<SmallVector<std::pair<unsigned, int>>> preds,
      ArrayRef<SmallVector<std::pair<unsigned, int>>> succs,
      ArrayRef<int> height, const ReadyWrites &writes,
      SmallVectorImpl<SmallVector<int>> &slots, unsigned below);
  /// The last modulo search: its lower bound and the periods it tried.
  unsigned moduloLower = 0, moduloTried = 0, moduloTotal = 0;
  LogicalResult buildCopy(Class &cls);
  LogicalResult stageLogic(Class &cls, Block *body, Operation *held,
                           Value hostClock, Value live1, Value pre1,
                           Value rinst, bool keepOld,
                           DenseMap<Operation *, Value> &readInst);
  /// Multicycle: checks that the capture registers of `copy` reach only its
  /// end registers through logic, and the end registers are reached only
  /// from its capture registers and constants: the multicycle paths are
  /// exactly the cone's.
  LogicalResult checkMulticycle(Class &cls);
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
  /// Each group's most sites, if the groups differ (`slr-capacities`: group g
  /// is the g-th SLR of the region, which the floorplan names); else every
  /// group's is slrCapacity and the groups are alike.
  ArrayRef<std::string> slrCapacitySpecs;
  SmallVector<uint64_t> slrCaps;
  ArrayRef<std::string> slrAssignSpecs;
  /// SLLs (super long lines between adjacent SLRs): with `sll-estimate`, the
  /// fewest link bits any assignment of the classes to the region's SLRs
  /// (slr-capacities, in order: adjacent SLRs adjacent) must carry over its
  /// worst boundary, reported; with `slr-sll-budget`, the most link bits an SLR
  /// group assignment may put over any one boundary.
  bool sllEstimate;
  uint64_t sllBudget;
  std::string sllReport;
  SmallVector<int> slrOf;
  SmallVector<uint64_t> slrSites;
  /// Further resources of each class and SLR beyond the LUT sites (slr-sites
  /// <class>:<sites>/<r1>/..., slr-capacities <sites>/<r1>/...: a copy's
  /// flops, URAM, block RAM...), each kept within its SLR's like the sites,
  /// in sll-estimate.
  SmallVector<SmallVector<uint64_t>> slrMore, slrCapsMore;
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
  /// `--vmem-merge-lanes`: the arrays split from one target RAM are one.
  bool mergeLanes;
  /// `--vmem-skip-held-reads`: a virtual memory's read is not issued while
  /// its address register holds and its row was not written since its last
  /// read; the instance keeps its read data.
  bool skipHeldReads;
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
  /// `zero-chain-hops`: the most zero-latency links a path may take within
  /// a host cycle (0: unbounded), and the links cut to keep it (chainCuts),
  /// each as (producer, consumer class, input).
  unsigned zeroChainHops;
  /// `compact-links`: a link memory holds only the producer instances its
  /// reader takes.
  bool compactLinks;
  /// `registered-links`: the classes whose links are never zero-latency,
  /// as globs, and whether each class (a copy by its class) matches.
  ArrayRef<std::string> registeredLinkSpecs;
  SmallVector<bool> registeredLinks;
  LogicalResult parseRegisteredLinks();
  /// `share-links`: outputs of a model driven by one value are one producer.
  bool shareLinks;
  unsigned sharedOutputs = 0;
  /// Outputs taken by their readers as constants (propagateConstants).
  unsigned constantOutputs = 0;
  /// One link memory (a read of a producer's bits by a class input, a run of
  /// bits with the same sources): the producer, its bits, and the producer
  /// instances the reader takes, sorted and distinct.
  struct LinkUse {
    unsigned producer, low, width, reader;
    std::string input;
    SmallVector<uint64_t> rows;
    /// The reader's instances that take it, and the producer instance each
    /// takes.
    SmallVector<unsigned> readers;
    SmallVector<uint64_t> readerRows;
    /// The reader's input, and the sources of its bits there (this one,
    /// other producers, frame inputs, constants).
    unsigned port = 0, sources = 0;
    /// The first bit of the reader's input it is.
    unsigned at = 0;
  };
  /// `staged-links`: an early link of one producer instance kept in two
  /// registers, written at the producer's pass and copied to the one its
  /// readers read at a position of the frame between the last read of the
  /// frame before and the first of this one (no bank select, no memory):
  /// that position (0 to II-1), or -1 where the schedule leaves none.
  int stagedCopy(unsigned p, unsigned inst, unsigned reader,
                 ArrayRef<unsigned> readers) const;
  /// The host cycles stagedCopy may copy at: [low, high] before reducing
  /// modulo II (high - low < II), or none.
  std::optional<std::pair<int, int>>
  stagedWindow(unsigned p, unsigned inst, unsigned reader,
               ArrayRef<unsigned> readers) const;
  /// `staged-link-memories`: the staged links of one producer port's bits
  /// (producer, low, width) to one reader class, two or more, keep their
  /// first registers as one memory, an entry per producer instance they take
  /// (`insts`, sorted), written at the producer's passes as the registers
  /// were; each link's second register loads the memory's read at its own
  /// copy position, the positions of a group distinct (one read port). A
  /// copy position of the window that stagedCopy chose from, so the second
  /// register takes what the first held there: the frame's values are as
  /// with registers, cycle for cycle.
  struct StagedGroup {
    unsigned producer, low, width, reader;
    SmallVector<uint64_t> insts;
    /// Each member's copy position and its entry, by position.
    std::map<int, unsigned> entryAt;
  };
  struct StagedMember {
    unsigned group, entry;
    int copyAt;
  };
  std::vector<StagedGroup> stagedGroups;
  /// By (reader class, reader input, its bit, producer, low).
  std::map<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned>,
           StagedMember>
      stagedMembers;
  /// The II the groups were planned at (-1: not yet): replanned if it moves.
  int stagedPlannedII = -1;
  SmallVector<int> stagedPlannedSlr;
  void planStagedMemories();
  const StagedMember *stagedMemberOf(unsigned reader, unsigned port,
                                     unsigned at, unsigned p,
                                     unsigned low) const {
    auto it = stagedMembers.find({reader, port, at, p, low});
    return it == stagedMembers.end() ? nullptr : &it->second;
  }
  bool stagedLinks;
  /// `link-block-depth`, `link-ultra-depth`: a link memory at least this
  /// deep, the only source of its reader's bits and never forwarded or
  /// taken in its producer's host cycle, in block (ultra) RAM, read a host
  /// cycle on into the RAM's output register (0: none).
  unsigned linkBlockDepth, linkUltraDepth;
  /// The passes a frame of class `c` (a copy): its instances' state and
  /// ready passes.
  unsigned passesOf(unsigned c) const {
    return classes[c].threads() * (1 + classes[c].readyLevels);
  }
  /// A class's model: a copy's class's name.
  StringRef modelOf(unsigned c) const {
    return classes[c].splitFrom >= 0 ? StringRef(classes[classes[c].splitFrom].name)
                                     : StringRef(classes[c].name);
  }
  /// `link-kinds`: <producer model>:<reader model>:<kind> rules (kind
  /// distributed, block or ultra), parsed; a rule overrides the depths for
  /// the links between those models that block or ultra RAM may hold.
  std::string linkKindSpec;
  /// Whether the links from producer `p` to class `reader` stream through
  /// HBM (`link-kinds` kind `hbm`).
  bool linkHbmRule(unsigned p, unsigned reader) const {
    if (linkKinds.empty())
      return false;
    auto it = linkKinds.find({modelOf(producers[p].cls).str(), modelOf(reader).str()});
    return it != linkKinds.end() && it->second.kind == "hbm";
  }
  struct LinkKind {
    std::string kind;
    unsigned page = 0;
  };
  std::map<std::pair<std::string, std::string>, LinkKind> linkKinds;
  /// HBM link streams (`link-kinds` kind `hbm`): the early links from one
  /// producer class to one reader class, written at the same pass, kept in
  /// device memory behind one virtual memory (a controller of the kernel, a
  /// row of the image a single address: INSTANCES 1). The producer's bits
  /// they read (`runs`, a word of `wordBits`) are held at its pass and written
  /// a row (`dw` bits, `pages` rows an instance) a host cycle over the pages
  /// after it, at {bank, instance, page}; the reader's passes read the rows
  /// they need (`needs`: producer instance and page, per reader instance) in
  /// slot order, each requested kLinkHbmLead positions before the host cycle
  /// it lands in its read registers, the request's tag a slot of a landing
  /// buffer the response fills. A position whose landing slot has no data
  /// waits in a bubble; the controller keeps the RAM's order, so a read is
  /// exact whenever its write was requested before it (the schedule's lead,
  /// checked: a link whose write is not is built as the rule's fallback).
  struct StreamRun {
    unsigned producer, low, width, offset;
  };
  struct LinkStream {
    unsigned pcls = 0, rcls = 0;
    bool atReady = false;
    unsigned level = 0;
    SmallVector<StreamRun> runs;
    unsigned wordBits = 0, dw = 0, pages = 0;
    SmallVector<SmallVector<std::pair<uint64_t, unsigned>>> needs;
    unsigned maxNeed = 0;
    /// After the schedule: the members kept and dropped (too close a read).
    unsigned kept = 0, dropped = 0;
    /// A schedule kept none of its members: it builds nothing, spaces no
    /// pass (layoutStreams' ports), and keeps no member of a later one.
    bool excluded = false;
    /// Per position (ii + 1 rows, the last a bubble's): the landing of a row
    /// (its reader instance, producer instance, page) and its request
    /// (producer instance, page, the frame's flip for the bank).
    SmallVector<uint64_t> popValid, popInst, popRow, popPage;
    SmallVector<uint64_t> reqValid, reqRow, reqPage, reqFlip;
    uint64_t readRows = 0;
    /// The simulator's virtual memory serving it (port vm<k>), once built;
    /// or, carried by a page engine (`page-engines`), that engine and its
    /// member index there.
    unsigned vm = 0;
    int engine = -1;
    unsigned member = 0;
    /// Per position (ii + 1): a write of a page of the pass before (its
    /// valid and page), `pages` positions after each producer pass.
    SmallVector<uint64_t> writeValid, writePage;
    /// Positions from a request to its landing: kLinkHbmLead, or less where
    /// the kept links' writes come later.
    unsigned lead = 0;
    /// The kept members' least room (positions from the row's write, the
    /// frame before's, to the request: `lead` at most), so `room - lead`
    /// is what a request may move earlier and a write later together
    /// (`link-hbm-shift`); and the most a request and a write were moved.
    unsigned room = 0, shift = 0, writeShift = 0;
    /// Each landing (by position q of the frame, from the reader's first
    /// pass: possibly negative) and its row, page and reader instance, in
    /// order; each producer pass (its writes' positions follow it).
    struct Landing {
      int q;
      unsigned inst;
      uint64_t row, page;
    };
    SmallVector<Landing> landings;
    SmallVector<int> writePasses;
    /// Per position (ii + 1): a landing's positions back to its request
    /// (`lead`, more where it was moved).
    SmallVector<uint64_t> popLead;
    unsigned instBits() const {
      return std::max(1u, llvm::Log2_32_Ceil(slots()));
    }
    /// The controller's view (foldsim_vmem TAGW): one instance (an index
    /// bit, always 0) under a tag of instBits() + 1 bits: a read's tag is its
    /// landing slot, a write's has the top bit set, so no write ever carries
    /// the tag of an outstanding read (the controller's guarantees hold per
    /// tag and instance). The port's width is both.
    unsigned tagBits() const { return instBits() + 1; }
    unsigned portInstBits() const { return tagBits() + 1; }
    /// `link-hbm-fit-slots`: the landing slots the placement needs (0: the
    /// most a lead takes, pow2(lead + 8)).
    unsigned slotCount = 0;
    unsigned leadSlots() const { return llvm::PowerOf2Ceil(lead + 8); }
    unsigned slots() const { return slotCount ? slotCount : leadSlots(); }
  };
  /// The most positions a stream's request precedes its landing (an HBM
  /// round trip at the kernel's clock and a margin); the fewest are those
  /// from a position's decision to its run and a margin, so that a landing's
  /// request is issued, and pending, before the decision that waits for it.
  static constexpr unsigned kLinkHbmLead = 48;
  unsigned linkHbmMinLead() const {
    return std::max(unsigned(decisionLead() + 6), linkHbmMinLeadOpt);
  }
  std::vector<LinkStream> linkStreams;
  /// Each candidate member by (reader class, input, its bit, producer, low):
  /// its stream, its bits' offset in the stream's word, and whether the
  /// schedule kept it.
  struct StreamMember {
    unsigned stream, offset;
    bool kept = false;
  };
  std::map<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned>,
           StreamMember>
      streamMembers;
  bool streamsPlanned = false;
  /// The streams as laid out (before planStreams narrows their needs to
  /// the kept members'), and each class's link page without them (the
  /// `link-kinds` pages).
  std::vector<LinkStream> laidOutStreams;
  SmallVector<unsigned> linkPageKinds;
  /// The streams' layouts and the ports they need (before the schedule).
  void layoutStreams();
  /// Each class's link page: its `link-kinds` page, raised by the ports of
  /// the streams not excluded.
  void streamPages();
  /// A reader instance's passes as its links are read: a staged class reads
  /// its stages before each state pass.
  SmallVector<int> streamReadPasses(const Class &cls, unsigned i) const;
  /// A stream member's use and its room for the request's lead.
  struct StreamRoom {
    decltype(streamMembers)::iterator member;
    int most, least, late;
    const LinkUse *use;
  };
  /// The members the schedule keeps (streamMembers' kept; each stream's
  /// lead, kept and dropped), from the laid-out needs; `uses` (linkUses')
  /// back `rooms`.
  void keepStreamMembers(const std::vector<LinkUse> &uses,
                         std::vector<StreamRoom> &rooms);
  /// The layouts of the streams of the links `pick` takes (of those a
  /// stream may carry), into `out`, their members into `members` if given.
  void layoutStreamsOf(function_ref<bool(const LinkUse &)> pick,
                       std::vector<LinkStream> &out,
                       decltype(streamMembers) *members) const;
  /// The members the schedule keeps, and each stream's landings and
  /// requests by position (after it).
  LogicalResult planStreams();
  /// A stream's tables from its requests (a landing's, in `landings`'
  /// order, by position: before the frame's first one or past it where
  /// the request is a frame before or after) and its writes (each pass's
  /// pages in turn).
  void setStreamTables(LinkStream &st, ArrayRef<int> requests,
                       ArrayRef<int> writes, bool &collide);
  /// `link-hbm-shift`: a stream's requests and writes placed at positions an
  /// engine leaves free (`reads`, `writes`), each at most `maxShift` from its
  /// own; false if none.
  bool placeStream(const LinkStream &st, ArrayRef<uint64_t> reads,
                   ArrayRef<uint64_t> writes, unsigned maxShift, unsigned slots,
                   SmallVectorImpl<int> &requests,
                   SmallVectorImpl<int> &writeAt) const;
  /// The landing slots a stream's requests need (`link-hbm-fit-slots`): the
  /// fewest, a power of two, with each request at least 8 positions after
  /// the landing of its slot's last use, as `lead + 8` slots give at its
  /// lead.
  unsigned fitSlots(const LinkStream &st, ArrayRef<int> requests) const;
  /// Off chip (pages=hbm): each position's page read [valid, instance, page]
  /// and page write [valid, page] of `cls` (ii + 1 entries each, the last a
  /// bubble's), as its controls' tables have them.
  void offChipRows(const Class &cls, SmallVectorImpl<uint64_t> &readRow,
                   SmallVectorImpl<uint64_t> &instRow,
                   SmallVectorImpl<uint64_t> &pageRow,
                   SmallVectorImpl<uint64_t> &writeRow,
                   SmallVectorImpl<uint64_t> &wpageRow) const;
  /// HBM page streams (`page-engines`): off-chip page columns carried by
  /// shared engines. An engine is one virtual memory of one image
  /// (INSTANCES 1) whose rows are {member, instance, row}; a request's tag
  /// (foldsim_vmem TAGW) is {member, instance}, so the controller's
  /// guarantees per tag are each column's per instance (a read outstanding
  /// per instance and page at most; no write of one whose read is
  /// outstanding: its state pass waited for it). Its members never request
  /// at the same position (a read and a write a position at most, checked):
  /// the frame's static order multiplexes them, without FIFOs or arbiters.
  struct PageEngine {
    /// (class, virtual memory) of each column it carries, then the HBM link
    /// streams it carries (their members after the columns).
    SmallVector<std::pair<unsigned, unsigned>> members;
    SmallVector<unsigned> streams;
    /// Positions with a request: a read, a write (ii + 1 each).
    SmallVector<uint64_t> reads, writes;
    /// A member's tag bits (a column's instance-and-page, a stream's
    /// landing tag) and address bits (a column's {instance, row}, a stream's
    /// row), the widest.
    unsigned dataWidth = 0, instWidth = 0, addrWidth = 0;
    bool readFirst = false;
    unsigned count() const { return members.size() + streams.size(); }
    unsigned memberBits() const {
      return std::max(1u, llvm::Log2_32_Ceil(count()));
    }
    unsigned tagBits() const { return memberBits() + instWidth; }
    unsigned rowBits() const { return memberBits() + addrWidth; }
  };
  std::vector<PageEngine> pageEngines;
  /// With engines: each off-chip paged class's page reads and writes at the
  /// positions the engines' assignment chose (offChipRows' rows).
  struct PageRows {
    SmallVector<uint64_t> read, inst, page, write, wpage;
  };
  std::map<unsigned, PageRows> pageRowsOf;
  /// Each carried column's engine and member index.
  std::map<std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned>>
      pageEngineOf;
  LogicalResult assignPageEngines();
  void reportPageEngines(raw_ostream &os);
  unsigned streamRowBits(const LinkStream &s) const {
    unsigned threads = classes[s.pcls].threads();
    return 1 + (threads > 1 ? llvm::Log2_32_Ceil(threads) : 0) +
           (s.pages > 1 ? llvm::Log2_32_Ceil(s.pages) : 0);
  }
  /// The page of the links from producer `p` to class `reader` (0: none).
  unsigned linkPageRule(unsigned p, unsigned reader) const {
    if (linkKinds.empty())
      return 0;
    auto it = linkKinds.find({modelOf(producers[p].cls).str(), modelOf(reader).str()});
    return it == linkKinds.end() || it->second.kind == "distributed" ? 0
                                                                     : it->second.page;
  }
  /// Whether a link that block or ultra RAM may hold, the input's only
  /// source, is paged: its rule's page p, where the schedule writes its
  /// entries and reads them p host cycles apart and each entry is written
  /// 2p + 3 host cycles before a read of it (the write's p host cycles, the
  /// read's p + 1 and the RAM's); early links only. Returns p, or 0.
  unsigned linkPaged(const LinkUse &use, bool banked, unsigned entries) const;
  LogicalResult parseLinkKinds();
  /// A link memory's RAM style: by `link-kinds`, else by depth.
  StringRef linkStyle(unsigned p, unsigned reader, uint64_t depth,
                      unsigned width) const {
    if (!linkKinds.empty()) {
      auto it = linkKinds.find({modelOf(producers[p].cls).str(),
                                modelOf(reader).str()});
      if (it != linkKinds.end())
        return it->second.kind == "distributed" || it->second.kind == "hbm"
                   ? StringRef()
                   : StringRef(it->second.kind);
    }
    return syncLinkStyle(depth, width);
  }
  /// `stage-levels`: the most estimated logic levels a class's state cone
  /// spans in one host cycle (0: no staged evaluation), and the classes'
  /// stages from it (`Class::stages`).
  unsigned stageLevels;
  /// `multicycle`: the classes (globs) staged without stage registers.
  ArrayRef<std::string> multicycleSpecs;
  LogicalResult chooseStages();
  /// Multicycle: the positions a copy's state passes are kept apart by the
  /// schedule (its page lead, or its multicycle span), 0 for none.
  unsigned passSpacing(unsigned c) const {
    auto &cls = classes[c];
    unsigned defer = cls.deferSpan ? cls.deferSpan + 1 + cls.page : 0u;
    return std::max({cls.pageLead(), cls.multicycle, defer});
  }
  /// `pass-split`: each named class's deferred lanes, planned on its model.
  LogicalResult planPassSplits();
  StringRef syncLinkStyle(uint64_t depth, unsigned width) const {
    if (linkUltraDepth && depth >= linkUltraDepth)
      return "ultra";
    if (linkBlockDepth && depth >= linkBlockDepth)
      return "block";
    return {};
  }
  /// A link memory's depth: an entry per producer instance (kept), doubled
  /// by frame parity for an early one.
  static uint64_t linkDepth(unsigned entries, bool banked) {
    unsigned indices = entries > 1 ? llvm::PowerOf2Ceil(entries) : 1;
    return banked ? 2 * indices : entries;
  }
  /// Every link memory the simulator builds, as buildSim cuts them.
  void linkUses(std::vector<LinkUse> &uses) const;
  /// A link's accesses in the schedule: the fewest host cycles between the
  /// writes of two of its entries (`writeGap`), between two reads of
  /// different entries (`readGap`: a read of the entry the read before took
  /// needs none), and from an entry's write to a read of it (`lead`: an
  /// early link's reads take the frame before's write); and the reads of a
  /// new entry a frame (`reads`). The frame (II) where there is no second.
  struct LinkTiming {
    int writeGap, readGap, lead;
    unsigned reads;
    /// The fewest host cycles between two reads, of any entries.
    int eventGap;
  };
  LinkTiming linkTiming(const LinkUse &use) const;
  /// The schedule report's lines on the links' storage.
  void reportLinks(raw_ostream &os);
  /// What each class evaluates at each position (`frameAt`), and whether
  /// any host cycle may idle (bubbles); then whether a link's reads ever
  /// need its producer's write forwarded, or taken around the read register
  /// in its host cycle (buildSim's links, and the report's choice of their
  /// storage).
  void buildFrameTable(bool stalls);
  SmallVector<SmallVector<Evaluation>> frameAt;
  /// What each class reads at each position: its evaluations' reads, the
  /// stages before their state passes (`frameAt` for a class of no stages).
  /// A link's reader is looked up here, its producer in `frameAt`.
  SmallVector<SmallVector<Evaluation>> readFrameAt;
  bool stalls = false;
  bool writesAt(const Producer &producer, unsigned at) const;
  bool needsForward(unsigned p, unsigned c, ArrayRef<bool> takes,
                    ArrayRef<uint64_t> from, unsigned lands = 1) const;
  bool takesNow(unsigned p, unsigned c, ArrayRef<bool> takes,
                ArrayRef<uint64_t> from) const;
  bool hitsOnlyTakes(unsigned p, unsigned c, ArrayRef<bool> takes,
                     ArrayRef<uint64_t> from) const;
  /// Whether a link of producer `p` whose reader takes `rows` is compacted.
  bool compacted(unsigned p, size_t rows) const {
    unsigned threads = classes[producers[p].cls].threads();
    return compactLinks && threads > 1 && rows < threads;
  }
  std::set<std::tuple<unsigned, unsigned, unsigned>> cutLinks;
  /// Counts, instance by instance, the zero-latency links on the paths into
  /// each late output and cuts the links that would pass `zeroChainHops`.
  void chainCuts();
  /// `page`: the paged models, as <model>:<p>; each class's `page` from them.
  ArrayRef<std::string> pageSpecs;
  bool pageMasks;
  /// The most bits of an off-chip page word in one virtual memory's row (a
  /// controller per column, so wide: 8 beats of the 512-bit AXI port).
  unsigned pageHbmColumnBits;
  /// The most instance slots of an off-chip page window (0: none).
  unsigned pageHbmWindow;
  /// The most shared engines carrying off-chip page columns (0: a
  /// controller per column; `page-engines`).
  unsigned pageEngineCap;
  /// The most bits of a combinationally read memory paged as rows, and a
  /// regex of the memories that may (empty: any).
  unsigned pageMemoryBits;
  std::string pageMemories;
  /// `page-rows`: <model>:<memory> names that page as rows whatever their
  /// size (empty: none).
  std::string pageRows;
  /// A paged class's combinationally read memories of at most
  /// `pageMemoryBits` bits that no ready pass reads, as a register per row:
  /// the reads a choice of the rows, the writes their next values. Returns
  /// the memories and bits so turned.
  std::pair<unsigned, uint64_t>
  memoriesAsRows(Class &cls, const DenseSet<Operation *> &ready,
                 const DenseMap<Operation *, unsigned> &virtualOf);
  /// Chooses each off-chip paged copy's window from its schedule.
  void chooseWindows();
  void choosePageGates();
  LogicalResult parsePages();
  /// Marks the paged classes whose pages a storage rule keeps off chip.
  LogicalResult pagesOffChip();
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

/// Whether a model is pure logic: no registers, memories or black boxes.
static bool isStateless(hw::HWModuleOp model) {
  for (auto &op : *model.getBodyBlock())
    if (isa<seq::FirRegOp, seq::FirMemOp, seq::CompRegOp, hw::InstanceOp>(op))
      return false;
  return true;
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
      // A stateless model (no registers, memories or black boxes) is pure
      // logic and needs no clock: its outputs are all late.
      cls.stateless = isStateless(model);
      if (cls.clockInput < 0 && !cls.stateless)
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
  propagateConstants();
  // Outputs of a model driven by the same value (a bus broadcast to several
  // children: a crossbar's data to each of its port arbiters) are one
  // producer: every reader of them takes the first, so they share links.
  if (shareLinks) {
    SmallVector<int> canonical(producers.size());
    std::iota(canonical.begin(), canonical.end(), 0);
    for (auto &cls : classes) {
      DenseMap<Value, unsigned> first;
      auto *output = cls.model.getBodyBlock()->getTerminator();
      for (auto [o, value] : llvm::enumerate(output->getOperands())) {
        auto [it, inserted] = first.try_emplace(value, o);
        if (!inserted) {
          canonical[cls.producerOf[o]] = cls.producerOf[it->second];
          ++sharedOutputs;
        }
      }
    }
    for (auto &consumer : consumers)
      for (auto &pieces : consumer.pieces)
        for (auto &piece : pieces)
          if (piece.producer >= 0)
            piece.producer = canonical[piece.producer];
  }
  for (auto &consumer : consumers)
    for (auto &pieces : consumer.pieces)
      for (auto &piece : pieces)
        if (piece.producer >= 0)
          producers[piece.producer].read = true;
  return success();
}

/// Outputs of a model that are constants of each instance -- functions of
/// its static inputs (driven by constants alone in every instance, as an
/// identity a parent passes on: `base + i`) and of constants, through no
/// register, memory, black box or other input -- are taken by their readers
/// as the constants they are. A link would hold the value too, but an early
/// link's storage is written a frame ahead, so the first frame would read
/// it unwritten, while the target has the value from its first cycle (and
/// may use it in reset: a memory written then keeps it). Iterated: an input
/// that becomes constant in every instance is static in turn, and a reader
/// may pass it on. An output whose logic does not fold to a constant stays
/// a link.
void StaticFolder::propagateConstants() {
  // Evaluates `root` with the static inputs' values, on clones of the cone's
  // operations (folding may rewrite an operation in place).
  auto evaluate = [&](Value root, DenseMap<Value, Attribute> &values) -> Attribute {
    SmallVector<std::pair<Value, bool>> stack{{root, false}};
    while (!stack.empty()) {
      auto [value, expanded] = stack.pop_back_val();
      if (values.count(value))
        continue;
      auto *op = value.getDefiningOp();
      if (!op || op->getNumRegions() || !isMemoryEffectFree(op) ||
          isa<seq::FirRegOp, seq::CompRegOp, seq::FirMemReadOp, hw::InstanceOp>(op))
        return {};
      if (!expanded) {
        stack.push_back({value, true});
        for (auto operand : op->getOperands())
          if (!values.count(operand))
            stack.push_back({operand, false});
        continue;
      }
      SmallVector<Attribute> operands;
      for (auto operand : op->getOperands()) {
        auto attr = values.lookup(operand);
        if (!attr)
          return {};
        operands.push_back(attr);
      }
      auto *clone = op->clone();
      SmallVector<OpFoldResult> results;
      bool folded = succeeded(clone->fold(operands, results)) &&
                    results.size() == op->getNumResults();
      SmallVector<Attribute> attrs;
      if (folded)
        for (auto [r, result] : llvm::enumerate(results)) {
          Attribute attr;
          if (auto a = dyn_cast<Attribute>(result))
            attr = a;
          else if (auto v = dyn_cast<Value>(result)) {
            // An operand of the clone: the original's at that index.
            for (auto [k, operand] : llvm::enumerate(clone->getOperands()))
              if (operand == v)
                attr = operands[k];
          }
          if (!isa_and_nonnull<IntegerAttr>(attr)) {
            folded = false;
            break;
          }
          attrs.push_back(attr);
        }
      clone->destroy();
      if (!folded)
        return {};
      for (auto [r, attr] : llvm::enumerate(attrs))
        values[op->getResult(r)] = attr;
    }
    return values.lookup(root);
  };
  // A value of every instance's pieces, if all are constants.
  auto constantOf = [](const Pieces &pieces) -> std::optional<APInt> {
    std::optional<APInt> value;
    for (auto &piece : pieces) {
      if (piece.producer >= 0)
        return std::nullopt;
      value = value ? piece.constant.concat(*value) : piece.constant;
    }
    return value;
  };
  DenseSet<unsigned> replaced;
  for (bool changed = true; changed;) {
    changed = false;
    for (auto [c, cls] : llvm::enumerate(classes)) {
      auto *body = cls.model.getBodyBlock();
      unsigned numInputs = cls.inputNames.size();
      // Each instance's static inputs' values.
      SmallVector<SmallVector<std::optional<APInt>>> statics(
          cls.threads(), SmallVector<std::optional<APInt>>(numInputs));
      for (unsigned i = 0; i < numInputs; ++i) {
        int k = cls.consumerOf[i];
        if (k < 0 || !isa<IntegerType>(cls.inputTypes[i]))
          continue;
        SmallVector<std::optional<APInt>> each;
        for (auto &pieces : consumers[k].pieces)
          each.push_back(constantOf(pieces));
        if (llvm::all_of(each, [](auto &v) { return v.has_value(); }))
          for (auto [inst, v] : llvm::enumerate(each))
            statics[inst][i] = v;
      }
      auto *output = body->getTerminator();
      for (auto [o, value] : llvm::enumerate(output->getOperands())) {
        unsigned p = cls.producerOf[o];
        if (!isa<IntegerType>(value.getType()) || replaced.contains(p))
          continue;
        SmallVector<APInt> each;
        for (unsigned inst = 0; inst < cls.threads(); ++inst) {
          DenseMap<Value, Attribute> values;
          for (unsigned i = 0; i < numInputs; ++i)
            if (statics[inst][i])
              values[body->getArgument(i)] =
                  IntegerAttr::get(cls.inputTypes[i], *statics[inst][i]);
          auto attr = evaluate(value, values);
          if (!attr)
            break;
          each.push_back(cast<IntegerAttr>(attr).getValue());
        }
        if (each.size() != cls.threads())
          continue;
        for (auto &consumer : consumers)
          for (auto &pieces : consumer.pieces) {
            if (llvm::none_of(pieces, [&](const Piece &q) {
                  return q.producer == int(p);
                }))
              continue;
            Pieces rebuilt;
            for (auto &q : pieces) {
              Piece piece = q;
              if (q.producer == int(p))
                piece = Piece{-1, 0, 0, q.width,
                              each[q.inst].extractBits(q.width, q.low)};
              if (!rebuilt.empty() && piece.producer < 0 &&
                  rebuilt.back().producer < 0) {
                auto &last = rebuilt.back();
                last.constant = piece.constant.concat(last.constant);
                last.width += piece.width;
                continue;
              }
              rebuilt.push_back(piece);
            }
            pieces = std::move(rebuilt);
          }
        replaced.insert(p);
        ++constantOutputs;
        changed = true;
      }
    }
  }
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
      copy.stateless = classes[c].stateless;
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

static bool findHeldRead(seq::FirMemReadOp read, comb::MuxOp &found,
                         bool &qOnTrue);
static int stageOpFullLevels(Operation *op);

/// The logic levels a link taken in its producer's host cycle may span: the
/// Beacon's gate.
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

/// An op's logic levels for staged evaluation, in half levels, by width as the
/// logic screen counts them on the V80's fabric (foldsim-estimate-timing): an
/// adder or an ordering compare a LUT and a carry chain of 8-bit blocks, an
/// equality compare 3-bit matches ANDed in LUT6 trees, a wide AND/OR/XOR a
/// LUT6 tree; a choice or a gate of two or three inputs half a level (a LUT6
/// takes two of them chained); otherwise as `logicLevels`.
static int stageOpLevels(Operation *op) {
  if (isa<comb::MuxOp, comb::AndOp, comb::OrOp, comb::XorOp>(op) &&
      op->getNumOperands() <= 3)
    return 1;
  return 2 * stageOpFullLevels(op);
}

static int stageOpFullLevels(Operation *op) {
  auto lut6 = [](int64_t inputs) {
    int levels = 1;
    for (int64_t n = 6; n < inputs; n *= 6)
      ++levels;
    return levels;
  };
  if (auto cmp = dyn_cast<comb::ICmpOp>(op)) {
    int64_t w = hw::getBitWidth(cmp.getLhs().getType());
    switch (cmp.getPredicate()) {
    case comb::ICmpPredicate::eq:
    case comb::ICmpPredicate::ne:
    case comb::ICmpPredicate::ceq:
    case comb::ICmpPredicate::cne:
    case comb::ICmpPredicate::weq:
    case comb::ICmpPredicate::wne:
      return w <= 3 ? 1 : 1 + lut6((w + 2) / 3);
    default:
      return 2 + int(w / 8);
    }
  }
  if (isa<comb::AddOp, comb::SubOp>(op))
    return 2 + int(hw::getBitWidth(op->getResult(0).getType()) / 8);
  if (isa<comb::AndOp, comb::OrOp, comb::XorOp>(op))
    return lut6(op->getNumOperands());
  return logicLevels(op);
}

/// The levels a value adds on the paths into it: through the logic and
/// combinational reads (a level each), stopping at registers (stage
/// registers too), registered reads, black boxes and the model's inputs.
static void depthEdges(Value value,
                       SmallVectorImpl<std::pair<Value, int>> &edges) {
  auto *op = value.getDefiningOp();
  if (!op)
    return;
  auto read = dyn_cast<seq::FirMemReadOp>(op);
  auto memory = read ? read.getMemory().getDefiningOp<seq::FirMemOp>()
                     : seq::FirMemOp();
  if (read) {
    if (memory && memory.getReadLatency() > 0)
      return;
    edges.push_back({read.getAddress(), 2});
    if (read.getEnable())
      edges.push_back({read.getEnable(), 2});
    return;
  }
  if (isa<seq::FirRegOp, seq::CompRegClockEnabledOp, seq::FirMemOp,
          hw::InstanceOp>(op))
    return;
  int levels = stageOpLevels(op);
  for (auto operand : op->getOperands())
    if (!isa<seq::ClockType>(operand.getType()))
      edges.push_back({operand, levels});
}

/// A port of a state op that the state pass writes: none of its clock or
/// memory handle.
static bool statePort(Value operand) {
  return !isa<seq::ClockType, seq::FirMemType>(operand.getType());
}

/// A value's place in staged evaluation: its stage, and the levels from the
/// stage's registers to it (-1: a constant, at any stage).
struct Staging {
  int stage = 0, levels = 0;
};

/// The staging of `root` and the values it depends on, into `memo`, soonest
/// first: an op joins the latest stage of its operands unless its levels
/// there would pass `limit`, and then starts the next stage, its operands
/// taken through the stage registers; so no stage spans more than `limit`
/// levels (or one op's). Depth first without recursion.
static Staging stageOf(Value root, int limit, DenseMap<Value, Staging> &memo) {
  SmallVector<std::pair<Value, bool>> stack{{root, false}};
  SmallVector<std::pair<Value, int>> next;
  while (!stack.empty()) {
    auto [value, expanded] = stack.pop_back_val();
    if (!expanded && memo.count(value))
      continue;
    if (auto *op = value.getDefiningOp();
        op && isa<hw::ConstantOp, hw::AggregateConstantOp>(op)) {
      memo[value] = {-1, 0};
      continue;
    }
    next.clear();
    depthEdges(value, next);
    if (!expanded) {
      // A placeholder against cycles, then the operands first.
      memo[value] = {0, 0};
      stack.push_back({value, true});
      for (auto [operand, levels] : next)
        if (!memo.count(operand))
          stack.push_back({operand, false});
      continue;
    }
    if (next.empty()) {
      memo[value] = {0, 0};
      continue;
    }
    int stage = 0, levels = next.front().second, at = 0;
    for (auto [operand, add] : next)
      stage = std::max(stage, memo.lookup(operand).stage);
    for (auto [operand, add] : next)
      if (auto s = memo.lookup(operand); s.stage == stage)
        at = std::max(at, s.levels);
    if (at > 0 && at + levels > limit) {
      ++stage;
      at = 0;
    }
    memo[value] = {stage, at + levels};
  }
  return memo.lookup(root);
}

/// The stage a write of `value` through `extra` more levels ends in.
static int sinkStage(Value value, int extra, int limit,
                     DenseMap<Value, Staging> &memo) {
  if (!value)
    return 0;
  auto s = stageOf(value, limit, memo);
  if (s.stage < 0)
    return 0;
  return s.stage + (s.levels > 0 && s.levels + extra > limit ? 1 : 0);
}

LogicalResult StaticFolder::chooseStages() {
  std::optional<llvm::Regex> pattern;
  if (!virtualMemoryPattern.empty())
    pattern.emplace(virtualMemoryPattern);
  // The classes whose virtual memories read at held addresses: the schedule
  // keeps their state gap at the lead + 2, so that the requests leave from
  // registers (the skip decided there), not from the target's address logic.
  if (pattern && skipHeldReads)
    for (auto &cls : classes) {
      StringRef modelName = cls.splitFrom >= 0
                                ? StringRef(classes[cls.splitFrom].name)
                                : cls.name;
      auto *folding = cls.folding.get();
      for (auto memory : cls.model.getBodyBlock()->getOps<seq::FirMemOp>()) {
        StringRef memName = memory.getName().value_or("");
        if ((folding && folding->physical.contains(memory)) ||
            memory.getReadLatency() != 1 ||
            (!pattern->match(memName) &&
             !pattern->match((modelName + ":" + memName).str())))
          continue;
        for (auto *user : memory->getUsers())
          if (auto read = dyn_cast<seq::FirMemReadOp>(user)) {
            comb::MuxOp mux;
            bool onTrue;
            cls.heldReads |= findHeldRead(read, mux, onTrue);
          }
      }
    }
  if (!stageLevels)
    return success();
  for (auto &cls : classes) {
    // Not for a class with a late output (a ready pass, or a state pass that
    // writes what its readers take in the frame), folded pipelines or black
    // boxes, or virtual memories read at held addresses (their requests leave
    // from the skip's registers, decided on the unstaged logic). Virtual
    // memories otherwise stage as the stores do: a request leaves at the
    // state pass with the last stage's row and data, and its response is
    // waited for by the next evaluation's read position (the state gap counts
    // from the state pass to that read, firstSlot). Pages off chip stage as
    // on chip: an instance's pages are fetched (and waited for) by its
    // evaluation's read position, the stages before its state pass, and
    // written back after the state pass (offChipRows).
    if (llvm::any_of(cls.late, [](bool late) { return late; }) ||
        cls.heldReads || (cls.folding && !cls.folding->folds.empty()))
      continue;
    auto *body = cls.model.getBodyBlock();
    bool eligible = true;
    for (auto &op : *body)
      if (isa<hw::InstanceOp>(op))
        eligible = false;
    if (!eligible)
      continue;
    // The last stage of a register's next value (its reset a choice more),
    // a memory's write or registered read, and an early output's next value
    // (its logic over the registers' next values, stages of it more).
    DenseMap<Value, Staging> memo;
    int limit = 2 * int(stageLevels), next = 0, last = 0;
    for (auto &op : *body) {
      if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
        int add = reg.getReset() ? 1 : 0;
        next = std::max({next, sinkStage(reg.getNext(), 2 * add, limit, memo),
                         sinkStage(reg.getReset(), 2 * add, limit, memo)});
      } else if (isa<seq::FirMemWriteOp>(op)) {
        for (auto operand : op.getOperands())
          if (statePort(operand))
            last = std::max(last, sinkStage(operand, 0, limit, memo));
      } else if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
        auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
        if (memory && memory.getReadLatency() > 0)
          last = std::max({last, sinkStage(read.getAddress(), 0, limit, memo),
                           sinkStage(read.getEnable(), 0, limit, memo)});
      }
    }
    last = std::max(last, next);
    for (int out : cls.outputDepth)
      last = std::max(last, next + (out + limit - 1) / limit);
    cls.stages = unsigned(last);

    // Multicycle: the cone's depth over the levels a host cycle, as host
    // cycles from the capture registers to the end registers (at least 1);
    // the reads a position before the first, the state pass a position after
    // the last.
    StringRef baseName = cls.splitFrom >= 0 ? StringRef(classes[cls.splitFrom].name)
                                            : StringRef(cls.name);
    bool listed = false;
    unsigned given = 0;
    for (auto &spec : multicycleSpecs) {
      auto [pattern, cycles] = StringRef(spec).split(':');
      if (auto glob = llvm::GlobPattern::create(pattern);
          glob && glob->match(baseName)) {
        listed = true;
        if (!cycles.empty())
          cycles.getAsInteger(10, given);
      }
    }
    if (!listed || !cls.stages)
      continue;
    bool combReads = false;
    for (auto read : body->getOps<seq::FirMemReadOp>()) {
      auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
      combReads |= !memory || memory.getReadLatency() == 0;
    }
    if (combReads) {
      llvm::errs() << "foldsim-fold-static: " << cls.name
                   << " reads a memory combinationally: staged, not multicycle\n";
      continue;
    }
    DenseMap<Value, Staging> depths;
    constexpr int kUnbounded = std::numeric_limits<int>::max() / 4;
    auto depthOf = [&](Value value, int extra) {
      if (!value)
        return 0;
      auto s = stageOf(value, kUnbounded, depths);
      return s.stage < 0 ? 0 : s.levels + extra;
    };
    int deepest = 0, nextDepth = 0;
    for (auto &op : *body) {
      if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
        int add = reg.getReset() ? 1 : 0;
        nextDepth = std::max({nextDepth, depthOf(reg.getNext(), 2 * add),
                              depthOf(reg.getReset(), 2 * add)});
      } else if (isa<seq::FirMemWriteOp>(op)) {
        for (auto operand : op.getOperands())
          if (statePort(operand))
            deepest = std::max(deepest, depthOf(operand, 0));
      } else if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
        deepest = std::max({deepest, depthOf(read.getAddress(), 0),
                            depthOf(read.getEnable(), 0)});
      }
    }
    deepest = std::max(deepest, nextDepth);
    for (int out : cls.outputDepth)
      deepest = std::max(deepest, nextDepth + 2 * out);
    cls.multicycle = unsigned(std::max(1, (deepest + limit - 1) / limit));
    // Given (<glob>:<cycles>, from the logic screen's count of the cone): the
    // span, whatever the estimate.
    if (given)
      cls.multicycle = given;
    cls.stages = cls.multicycle + 1;
    llvm::errs() << "foldsim-fold-static: " << cls.name << " multicycle: "
                 << deepest << " half levels estimated, over "
                 << cls.multicycle << " host cycles ("
                 << (given ? "given" : "estimated at " + std::to_string(limit) +
                                           " half levels a host cycle")
                 << "), state passes " << cls.multicycle
                 << " positions apart\n";
  }
  for (auto &spec : multicycleSpecs) {
    auto [pattern, cycles] = StringRef(spec).split(':');
    unsigned n = 0;
    if (!llvm::GlobPattern::create(pattern) ||
        (!cycles.empty() && (cycles.getAsInteger(10, n) || n < 1)))
      return netlist.emitError("multicycle: '")
             << spec << "' is not <glob> or <glob>:<host cycles>";
  }
  return success();
}

LogicalResult StaticFolder::planPassSplits() {
  for (auto &spec : passSplitSpecs) {
    SmallVector<StringRef> parts;
    StringRef(spec).split(parts, ':');
    unsigned n = 0, span = 1;
    if (parts.size() < 2 || parts.size() > 3 ||
        !llvm::GlobPattern::create(parts[0]) || parts[1].getAsInteger(10, n) ||
        n < 2 || (parts.size() == 3 && (parts[2].getAsInteger(10, span) || !span)))
      return netlist.emitError("pass-split: '")
             << spec << "' is not <glob>:<passes>[:<span>], passes at least 2";
  }
  for (auto &cls : classes) {
    StringRef base = cls.splitFrom >= 0 ? StringRef(classes[cls.splitFrom].name)
                                        : StringRef(cls.name);
    unsigned passes = 0, span = 1;
    for (auto &spec : passSplitSpecs) {
      SmallVector<StringRef> parts;
      StringRef(spec).split(parts, ':');
      if (auto glob = llvm::GlobPattern::create(parts[0]);
          glob && glob->match(base)) {
        parts[1].getAsInteger(10, passes);
        span = 1;
        if (parts.size() == 3)
          parts[2].getAsInteger(10, span);
      }
    }
    if (passes < 2)
      continue;
    auto refuse = [&](StringRef why) {
      llvm::errs() << "foldsim-fold-static: " << cls.name << " not split: " << why
                   << "\n";
    };
    if (cls.stages) {
      refuse("staged");
      continue;
    }
    // Held reads keep their address registers: the read address uses the
    // hold's choice, so it is never logic only next state uses.
    // The next values an early output's next value reads (buildCopy's
    // nextValue): the registers in an early output's cone, through logic.
    auto *body = cls.model.getBodyBlock();
    auto *output = body->getTerminator();
    SmallVector<Value> pinned;
    DenseSet<Operation *> seen;
    SmallVector<Value> work;
    for (auto [o, value] : llvm::enumerate(output->getOperands()))
      if (!cls.late[o])
        work.push_back(value);
    while (!work.empty()) {
      Value v = work.pop_back_val();
      auto *def = v.getDefiningOp();
      if (!def || !seen.insert(def).second)
        continue;
      if (auto reg = dyn_cast<seq::FirRegOp>(def)) {
        pinned.push_back(reg.getNext());
        if (reg.getReset()) {
          pinned.push_back(reg.getReset());
          pinned.push_back(reg.getResetValue());
        }
        continue;
      }
      if (isa<seq::FirMemOp, seq::FirMemReadOp, hw::InstanceOp>(def))
        continue;
      for (auto operand : def->getOperands())
        work.push_back(operand);
    }
    // A folded pipeline's registers are written as its results come back.
    if (auto *folding = cls.folding.get())
      for (auto *op : folding->physical)
        if (auto reg = dyn_cast<seq::FirRegOp>(op))
          pinned.push_back(reg.getNext());
    auto plan = foldsim::planDeferred(body, pinned, passes);
    if (!plan.stats.groups) {
      refuse("no lanes in the logic only next state uses");
      continue;
    }
    cls.passSplit = passes;
    cls.passSpan = span;
    cls.deferSpan = (passes + 1) * span;
    llvm::errs() << "foldsim-fold-static: " << cls.name << " pass-split: "
                 << passes << " passes of " << span << " host cycles, "
                 << plan.regs.size() << " registers written "
                 << cls.deferSpan << " host cycles after the state pass; planned "
                 << plan.stats.groups << " lane groups, "
                 << llvm::format("%.0f", plan.stats.before) << " -> "
                 << llvm::format("%.0f", plan.stats.after) << " LUTs estimated\n";
  }
  return success();
}

LogicalResult StaticFolder::parseRegisteredLinks() {
  registeredLinks.assign(classes.size(), false);
  for (auto &spec : registeredLinkSpecs) {
    auto glob = llvm::GlobPattern::create(spec);
    if (!glob)
      return netlist.emitError("registered-links: '") << spec
                                                      << "' is not a glob";
    bool any = false;
    for (auto [c, cls] : llvm::enumerate(classes)) {
      int base = cls.splitFrom >= 0 ? cls.splitFrom : int(c);
      if (glob->match(classes[base].name)) {
        registeredLinks[c] = any = true;
      }
    }
    if (!any)
      return netlist.emitError("registered-links: no class matches '")
             << spec << "'";
  }
  return success();
}

LogicalResult StaticFolder::parseSerialize() {
  serialK.assign(classes.size(), 1);
  for (auto &spec : slrSerializeSpecs) {
    auto [pattern, count] = StringRef(spec).rsplit(':');
    if (count.empty())
      std::swap(pattern, count);
    unsigned k = 0;
    if (count.getAsInteger(10, k) || k < 1 || k > 64)
      return netlist.emitError("slr-serialize: '")
             << spec << "' is not <glob>:<k> or <k>, k from 1 to 64";
    auto glob = llvm::GlobPattern::create(pattern.empty() ? "*" : pattern);
    if (!glob)
      return netlist.emitError("slr-serialize: '") << spec << "' is not a glob";
    bool any = false;
    for (auto [c, cls] : llvm::enumerate(classes)) {
      int base = cls.splitFrom >= 0 ? cls.splitFrom : int(c);
      if (glob->match(classes[base].name) || glob->match(cls.name)) {
        serialK[c] = k;
        any = true;
      }
    }
    if (!any)
      return netlist.emitError("slr-serialize: no class matches '") << spec << "'";
  }
  slrSerialize = llvm::any_of(serialK, [](unsigned k) { return k > 1; });
  return success();
}

LogicalResult StaticFolder::parseLinkKinds() {
  if (linkHbmRowBits < 8 || linkHbmRowBits > 4096)
    return netlist.emitError("link-hbm-row-bits: ")
           << linkHbmRowBits << " is not 8 to 4096";
  if (linkKindSpec.empty())
    return success();
  std::set<std::string> models;
  for (unsigned c = 0; c < classes.size(); ++c)
    models.insert(modelOf(c).str());
  SmallVector<StringRef> rules;
  StringRef(linkKindSpec).split(rules, ',', -1, false);
  for (StringRef rule : rules) {
    SmallVector<StringRef> parts;
    rule.trim().split(parts, ':');
    unsigned page = 0;
    if ((parts.size() != 3 && parts.size() != 4) ||
        !llvm::is_contained({"distributed", "block", "ultra", "hbm"}, parts[2]) ||
        (parts.size() == 4 &&
         (parts[3].getAsInteger(10, page) || page < 2 || page > 64 ||
          parts[2] == "distributed" || parts[2] == "hbm")))
      return netlist.emitError("link-kinds: '")
             << rule
             << "' is not <producer model>:<reader model>:"
                "distributed|block|ultra[:<page 2..64>]|hbm";
    for (StringRef m : {parts[0], parts[1]})
      if (!models.count(m.str()))
        return netlist.emitError("link-kinds: no class model '") << m << "'";
    // A pair of models takes one kind: a second, different rule for it is
    // an error (the last would win the storage while the first's page still
    // spaced the passes).
    auto [kit, fresh] = linkKinds.try_emplace({parts[0].str(), parts[1].str()},
                                              LinkKind{parts[2].str(), page});
    if (!fresh && (kit->second.kind != parts[2] || kit->second.page != page))
      return netlist.emitError("link-kinds: '")
             << rule << "' conflicts with an earlier rule for " << parts[0]
             << ":" << parts[1];
    // Both models' copies space their passes by the page.
    if (page)
      for (unsigned c = 0; c < classes.size(); ++c)
        if (modelOf(c) == parts[0] || modelOf(c) == parts[1])
          classes[c].linkPage = std::max(classes[c].linkPage, page);
  }
  return success();
}

void StaticFolder::layoutStreams() {
  if (!llvm::any_of(linkKinds, [](auto &k) { return k.second.kind == "hbm"; }))
    return;
  layoutStreamsOf([&](const LinkUse &use) { return linkHbmRule(use.producer, use.reader); },
                  linkStreams, &streamMembers);
  laidOutStreams = linkStreams;
  linkPageKinds.clear();
  for (auto &cls : classes)
    linkPageKinds.push_back(cls.linkPage);
  streamPages();
}

void StaticFolder::streamPages() {
  // Ports: the producer's passes a write's pages apart, the reader's its
  // landings and two more apart (the last lands two host cycles before
  // the slot; the next pass's first after it).
  for (unsigned c = 0; c < classes.size(); ++c)
    classes[c].linkPage = linkPageKinds[c];
  for (auto &s : laidOutStreams) {
    if (s.excluded)
      continue;
    classes[s.pcls].linkPage = std::max(classes[s.pcls].linkPage, s.pages);
    classes[s.rcls].linkPage = std::max(classes[s.rcls].linkPage, s.maxNeed + 2);
  }
}

void StaticFolder::layoutStreamsOf(function_ref<bool(const LinkUse &)> pick,
                                   std::vector<LinkStream> &out,
                                   decltype(streamMembers) *members) const {
  // Every link a stream may carry: early (banked), from a producer class of
  // several instances, more than one entry kept.
  std::vector<LinkUse> uses;
  linkUses(uses);
  std::map<std::tuple<unsigned, unsigned, bool, unsigned>, unsigned> streamOf;
  std::vector<SmallVector<const LinkUse *>> usesOf;
  for (auto &use : uses) {
    auto &producer = producers[use.producer];
    if (producer.cls < 0 || producer.late || classes[producer.cls].threads() < 2 ||
        use.rows.size() < 2 || !slrOf.empty() || shortLead || !pick(use))
      continue;
    auto key = std::make_tuple(unsigned(producer.cls), use.reader, producer.atReady,
                               producer.atReady ? producer.readyLevel : 0u);
    auto [it, fresh] = streamOf.try_emplace(key, out.size());
    if (fresh) {
      LinkStream s;
      s.pcls = producer.cls;
      s.rcls = use.reader;
      s.atReady = producer.atReady;
      s.level = std::get<3>(key);
      out.push_back(s);
      usesOf.emplace_back();
    }
    usesOf[it->second].push_back(&use);
  }
  for (auto [k, s] : llvm::enumerate(out)) {
    // The word: each producer's bits the members read, as runs of
    // overlapping or adjacent bits, by producer and low bit.
    std::map<unsigned, std::vector<std::pair<unsigned, unsigned>>> spans;
    for (auto *use : usesOf[k])
      spans[use->producer].push_back({use->low, use->low + use->width});
    unsigned offset = 0;
    for (auto &[p, list] : spans) {
      llvm::sort(list);
      unsigned lo = list[0].first, hi = list[0].second;
      auto flush = [&] {
        s.runs.push_back({p, lo, hi - lo, offset});
        offset += hi - lo;
      };
      for (auto [a, e] : list) {
        if (a > hi) {
          flush();
          lo = a;
          hi = e;
        } else {
          hi = std::max(hi, e);
        }
      }
      flush();
    }
    s.wordBits = offset;
    s.dw = std::min(linkHbmRowBits, offset);
    s.pages = (offset + s.dw - 1) / s.dw;
    auto offsetOf = [&](unsigned p, unsigned low) {
      for (auto &run : s.runs)
        if (run.producer == p && low >= run.low && low < run.low + run.width)
          return run.offset + low - run.low;
      llvm_unreachable("a member's bits in no run");
    };
    // What each reader instance's passes read: the rows and pages its
    // members' bits are in.
    unsigned rthreads = classes[s.rcls].threads();
    std::vector<std::set<std::pair<uint64_t, unsigned>>> need(rthreads);
    for (auto *use : usesOf[k]) {
      unsigned off = offsetOf(use->producer, use->low);
      if (members)
        (*members)[{use->reader, use->port, use->at, use->producer, use->low}] =
            StreamMember{unsigned(k), off, false};
      for (auto [i, row] : llvm::zip(use->readers, use->readerRows))
        for (unsigned j = off / s.dw; j <= (off + use->width - 1) / s.dw; ++j)
          need[i].insert({row, j});
    }
    s.needs.resize(rthreads);
    for (unsigned i = 0; i < rthreads; ++i) {
      s.needs[i].assign(need[i].begin(), need[i].end());
      s.maxNeed = std::max<unsigned>(s.maxNeed, need[i].size());
      s.readRows += need[i].size() * (1 + classes[s.rcls].readyLevels);
      s.kept += !need[i].empty();
    }
  }
}

SmallVector<int> StaticFolder::streamReadPasses(const Class &cls,
                                                unsigned i) const {
  SmallVector<int> slots{cls.slot[kState][i] - int(cls.stages)};
  for (unsigned l = 0; l < cls.readyLevels; ++l)
    slots.push_back(cls.readySlots(l)[i]);
  return slots;
}

void StaticFolder::keepStreamMembers(const std::vector<LinkUse> &uses,
                                     std::vector<StreamRoom> &rooms) {
  int frame = int(ii);
  // Each candidate's room for the request's lead: its requests after its
  // entry's write (the frame before's, every page, the write's register a
  // host cycle on, and a margin: `most`), and its last before the next
  // write of its bank (`least`). A stream's lead is kLinkHbmLead or its
  // kept members' least room; one with less than linkHbmMinLead() is built
  // as the rule's fallback.
  // And no landing after the next write of its row (its bank's, a frame
  // on): the controller orders a read after an earlier write of its row, not
  // a write after an earlier read, so a read must be answered (its landing
  // run) before the row's next write is requested (`late` <= 0).
  rooms.clear();
  for (auto &use : uses) {
    auto it = streamMembers.find({use.reader, use.port, use.at, use.producer, use.low});
    if (it == streamMembers.end())
      continue;
    auto &st = linkStreams[it->second.stream];
    auto &producer = producers[use.producer];
    auto &pcls = classes[producer.cls];
    int most = std::numeric_limits<int>::max(), least = 0,
        late = std::numeric_limits<int>::min();
    for (auto [i, row] : llvm::zip(use.readers, use.readerRows)) {
      int w = producer.atReady ? pcls.readySlots(producer.readyLevel)[row]
                               : pcls.slot[kState][row];
      int n = int(st.needs[i].size());
      for (int s : streamReadPasses(classes[use.reader], i)) {
        most = std::min(most, (s - n - 1) - (w + int(st.pages) + 4 - frame) - 1);
        least = std::max(least, (s - 2) - (w + frame) + 1);
        late = std::max(late, (s - 2) - (w + frame));
      }
    }
    rooms.push_back({it, most, least, late, &use});
  }
  for (auto &st : linkStreams) {
    st.lead = linkHbmLead;
    st.kept = 0;
    st.dropped = 0;
    st.readRows = 0;
  }
  for (auto &r : rooms)
    if (r.most >= int(linkHbmMinLead()) && r.least <= r.most && r.late <= 0) {
      auto &st = linkStreams[r.member->second.stream];
      st.lead = std::min<unsigned>(st.lead, r.most);
    }
  // A member (a reader input's bits from a producer output) may stand for
  // several uses (other readers and rows of the same wiring): kept only if
  // every one of them is.
  for (auto &r : rooms)
    r.member->second.kept = true;
  for (auto &r : rooms) {
    auto &st = linkStreams[r.member->second.stream];
    bool ok = r.most >= int(st.lead) && r.least <= int(st.lead) &&
              r.late <= 0 && st.lead >= linkHbmMinLead() && !st.excluded;
    if (!ok)
      r.member->second.kept = false;
  }
  std::set<decltype(streamMembers)::iterator::pointer> counted;
  for (auto &r : rooms)
    if (counted.insert(&*r.member).second) {
      auto &st = linkStreams[r.member->second.stream];
      ++(r.member->second.kept ? st.kept : st.dropped);
    }
}

LogicalResult StaticFolder::planStreams() {
  if (streamsPlanned)
    return success();
  streamsPlanned = true;
  int frame = int(ii);
  auto mod = [&](int x) { return ((x % frame) + frame) % frame; };
  std::vector<LinkUse> uses;
  std::vector<StreamRoom> rooms;
  linkUses(uses);
  keepStreamMembers(uses, rooms);
  // Only the kept members' rows are requested and land: a dropped member's
  // row may be written after the reads the stream would request of it (its
  // reason to be dropped), and its links are built as before.
  {
    std::vector<std::vector<std::set<std::pair<uint64_t, unsigned>>>> need(linkStreams.size());
    for (auto [k, st] : llvm::enumerate(linkStreams))
      need[k].resize(classes[st.rcls].threads());
    for (auto &r : rooms) {
      if (!r.member->second.kept)
        continue;
      unsigned k = r.member->second.stream, off = r.member->second.offset;
      auto &st = linkStreams[k];
      for (auto [i, row] : llvm::zip(r.use->readers, r.use->readerRows))
        for (unsigned j = off / st.dw; j <= (off + r.use->width - 1) / st.dw; ++j)
          need[k][i].insert({row, j});
    }
    for (auto [k, st] : llvm::enumerate(linkStreams)) {
      st.maxNeed = 0;
      for (unsigned i = 0; i < st.needs.size(); ++i) {
        st.needs[i].assign(need[k][i].begin(), need[k][i].end());
        st.maxNeed = std::max<unsigned>(st.maxNeed, st.needs[i].size());
      }
    }
  }
  // The least room of the kept members (at least the lead, the least of
  // every member's room or kLinkHbmLead).
  for (auto &st : linkStreams)
    st.room = st.lead;
  {
    std::vector<int> least(linkStreams.size(), std::numeric_limits<int>::max());
    for (auto &r : rooms)
      if (r.member->second.kept)
        least[r.member->second.stream] = std::min(least[r.member->second.stream], r.most);
    for (auto [k, st] : llvm::enumerate(linkStreams))
      if (st.kept && least[k] != std::numeric_limits<int>::max())
        st.room = std::max<unsigned>(st.lead, unsigned(least[k]));
  }
  for (auto &st : linkStreams) {
    if (!st.kept)
      continue;
    auto &rcls = classes[st.rcls];
    // Each pass's landings: its rows over the host cycles before it, the
    // last two before its slot.
    st.landings.clear();
    for (unsigned i = 0; i < rcls.threads(); ++i) {
      unsigned n = st.needs[i].size();
      if (!n)
        continue;
      for (int s : streamReadPasses(rcls, i))
        for (unsigned k = 0; k < n; ++k)
          st.landings.push_back({s - int(n) - 1 + int(k), i, st.needs[i][k].first,
                                 st.needs[i][k].second});
    }
    // In the frame's order (by position; a landing keeps its q, whose frame
    // gives its request's bank flip).
    llvm::stable_sort(st.landings,
                      [&](auto &a, auto &b) { return mod(a.q) < mod(b.q); });
    st.readRows = st.landings.size();
    // Each producer pass (its position in the frame): its pages written
    // after it.
    st.writePasses.clear();
    auto &pc = classes[st.pcls];
    for (unsigned i = 0; i < pc.threads(); ++i)
      st.writePasses.push_back(mod(!st.atReady ? pc.slot[kState][i]
                                               : pc.readySlots(st.level)[i]));
    llvm::sort(st.writePasses);
    // Each landing requested `lead` positions earlier, page j written at the
    // j-th position after each producer pass (assignPageEngines may move
    // them, `link-hbm-shift`).
    SmallVector<int> requests, writes;
    for (auto &l : st.landings)
      requests.push_back(l.q - int(st.lead));
    for (int w : st.writePasses)
      for (unsigned j = 0; j < st.pages; ++j)
        writes.push_back(w + 1 + int(j));
    bool collide = false;
    setStreamTables(st, requests, writes, collide);
    if (collide)
      return netlist.emitError("link streams: ")
             << modelOf(st.pcls) << " -> " << modelOf(st.rcls)
             << ": two landings in one position (the reader's link port)";
  }
  return success();
}

void StaticFolder::setStreamTables(LinkStream &st, ArrayRef<int> requests,
                                   ArrayRef<int> writes, bool &collide) {
  int frame = int(ii);
  auto mod = [&](int x) { return unsigned(((x % frame) + frame) % frame); };
  auto floorDiv = [&](int x) {
    return x >= 0 ? x / frame : -((-x + frame - 1) / frame);
  };
  for (auto *v : {&st.popValid, &st.popInst, &st.popRow, &st.popPage, &st.popLead,
                  &st.reqValid, &st.reqRow, &st.reqPage, &st.reqFlip,
                  &st.writeValid, &st.writePage})
    v->assign(ii + 1, 0);
  st.shift = 0;
  for (auto [l, t] : llvm::zip(st.landings, requests)) {
    unsigned pos = mod(l.q);
    if (st.popValid[pos])
      collide = true;
    st.popValid[pos] = 1;
    st.popInst[pos] = l.inst;
    st.popRow[pos] = l.row;
    st.popPage[pos] = l.page;
    st.popLead[pos] = unsigned(l.q - t);
    st.shift = std::max(st.shift, unsigned(l.q - t) - st.lead);
    unsigned tpos = mod(t);
    st.reqValid[tpos] = 1;
    st.reqRow[tpos] = l.row;
    st.reqPage[tpos] = l.page;
    st.reqFlip[tpos] = floorDiv(t) & 1;
  }
  unsigned x = 0;
  st.writeShift = 0;
  for (size_t p = 0; p < st.writePasses.size(); ++p)
    for (unsigned j = 0; j < st.pages; ++j) {
      st.writeShift = std::max(
          st.writeShift, unsigned(writes[x] - (st.writePasses[p] + 1 + int(j))));
      unsigned pos = mod(writes[x++]);
      st.writeValid[pos] = 1;
      st.writePage[pos] = j;
    }
}

bool StaticFolder::placeStream(const LinkStream &st, ArrayRef<uint64_t> reads,
                               ArrayRef<uint64_t> writes, unsigned maxShift,
                               unsigned slotCount, SmallVectorImpl<int> &requests,
                               SmallVectorImpl<int> &writeAt) const {
  // Requests in landing order (a landing takes the next slot: the landing
  // buffer's pointers count both in order), placed from the last: each at
  // the latest position free on the engine at most `maxShift` before its own
  // (`lead` before its landing) and before the next request, and at least 8
  // positions after the landing `slots` before it (its slot's last use, the
  // margin the landing slots have at `lead`: the slots are not grown); the
  // first after the last of the frame before. Then each write in order at
  // the earliest position free within the room the requests left after its
  // own (`pages` after the pass, page j at j), at most at the producer's
  // next pass (the held word is the pass's until then). A request `d`
  // earlier and a write `e` later keep a read after the frame before's
  // write when d + e <= room - lead (the room is measured to the write of
  // the last page at its own). (Order is the frame's: landings by
  // position, each request compared at its landing's position less its
  // lead, the last a frame before the first.)
  int frame = int(ii);
  auto mod = [&](int x) { return int(((x % frame) + frame) % frame); };
  unsigned slack = std::min(maxShift, st.room - st.lead);
  requests.clear();
  writeAt.clear();
  int n = int(st.landings.size()), slots = int(slotCount);
  // A landing's position in the frame of the landing `k` (k < 0: the frame
  // before's).
  auto landingAt = [&](int k) {
    int back = k < 0 ? (-k + n - 1) / n : 0;
    return mod(st.landings[k + back * n].q) - back * frame;
  };
  SmallVector<int> at(n);
  unsigned used = 0;
  for (int k = n - 1; k >= 0; --k) {
    int own = landingAt(k) - int(st.lead);
    int hi = k == n - 1 ? own : std::min(own, at[k + 1] - 1);
    int lo = own - int(slack);
    lo = std::max(lo, landingAt(k - slots) + 8);
    int t = hi;
    while (t >= lo && reads[mod(t)])
      --t;
    if (t < lo)
      return false;
    at[k] = t;
    used = std::max(used, unsigned(own - t));
  }
  if (n && at[0] <= at[n - 1] - frame)
    return false;
  for (int k = 0; k < n; ++k)
    requests.push_back(st.landings[k].q - int(st.lead) -
                       ((landingAt(k) - int(st.lead)) - at[k]));
  unsigned wslack = std::min(maxShift, slack - used);
  size_t np = st.writePasses.size();
  for (size_t p = 0; p < np; ++p) {
    int w = st.writePasses[p];
    int next = p + 1 < np ? st.writePasses[p + 1] : st.writePasses[0] + frame;
    int last = writeAt.empty() ? std::numeric_limits<int>::min() : writeAt.back();
    for (unsigned j = 0; j < st.pages; ++j) {
      int own = w + 1 + int(j);
      int lo = std::max(own, last + 1),
          hi = std::min(own + int(wslack), next - int(st.pages - 1 - j));
      int q = lo;
      while (q <= hi && writes[mod(q)])
        ++q;
      if (q > hi)
        return false;
      writeAt.push_back(q);
      last = q;
    }
  }
  return true;
}

unsigned StaticFolder::fitSlots(const LinkStream &st, ArrayRef<int> requests) const {
  int frame = int(ii), n = int(st.landings.size());
  auto mod = [&](int x) { return int(((x % frame) + frame) % frame); };
  auto landingAt = [&](int k) {
    int back = k < 0 ? (-k + n - 1) / n : 0;
    return mod(st.landings[k + back * n].q) - back * frame;
  };
  // (The slots ahead of the next landing the decisions check: a few more
  // than the decisions in flight.)
  unsigned least = std::max(8u, unsigned(llvm::PowerOf2Ceil(decisionLead() + 3)));
  for (unsigned slots = least; slots < st.leadSlots(); slots *= 2) {
    bool ok = true;
    for (int k = 0; k < n && ok; ++k) {
      // The request in its landing's frame.
      int at = landingAt(k) - (st.landings[k].q - requests[k]);
      ok = at >= landingAt(k - int(slots)) + 8;
    }
    if (ok)
      return slots;
  }
  return st.leadSlots();
}

void StaticFolder::offChipRows(const Class &cls,
                           SmallVectorImpl<uint64_t> &readRow,
                           SmallVectorImpl<uint64_t> &instRow,
                           SmallVectorImpl<uint64_t> &pageRow,
                           SmallVectorImpl<uint64_t> &writeRow,
                           SmallVectorImpl<uint64_t> &wpageRow) const {
  if (auto it = pageRowsOf.find(unsigned(&cls - &classes[0]));
      it != pageRowsOf.end()) {
    auto &r = it->second;
    readRow.assign(r.read.begin(), r.read.end());
    instRow.assign(r.inst.begin(), r.inst.end());
    pageRow.assign(r.page.begin(), r.page.end());
    writeRow.assign(r.write.begin(), r.write.end());
    wpageRow.assign(r.wpage.begin(), r.wpage.end());
    return;
  }
  unsigned threads = cls.threads();
  auto wrap = [&](int pos) { return unsigned(((pos % int(ii)) + int(ii)) % int(ii)); };
  for (auto *row : {&readRow, &instRow, &pageRow, &writeRow, &wpageRow})
    row->assign(ii + 1, 0);
  // Each fetch as early as the fetch buffer allows: at the copy's state
  // pass before it (the one before the first in the frame before), so
  // its pages' round trip hides in the slots between; the schedule keeps
  // them at least the lead apart.
  // With a window, each instance's pages prefetched at the copy's state
  // pass `pageWindow` before its own, into its slot, and fetched from
  // the window a lead before its state pass, as on chip (staged, both at
  // the read positions, the stages before: the slot's last fetch from the
  // window is a lead before the prefetch; chooseWindows).
  // Staged, the fetch buffer is read at the evaluation's read position, the
  // stages before its state pass: each direct fetch as early as the copy's
  // read position before it, and a lead before its own; but after the frame
  // before's write-back of each page (at the state pass + 1 + page), which
  // the schedule's state gap leaves room for (minStateGap). Unstaged, that
  // bound never binds (the copy's other passes are a lead apart).
  SmallVector<int> states(cls.slot[kState].begin(), cls.slot[kState].end());
  llvm::sort(states);
  unsigned back = cls.pageWindow ? cls.pageWindow : 1;
  auto fetchOf = [&](int s) {
    int k = int(llvm::lower_bound(states, s) - states.begin()) - int(back);
    int frames = k < 0 ? -1 : 0;
    int n = int(states.size());
    return states[((k % n) + n) % n] + frames * int(ii);
  };
  for (unsigned i = 0; i < threads; ++i) {
    int s = cls.slot[kState][i];
    int stages = int(cls.stages);
    int f = cls.pageWindow
                ? fetchOf(s) - stages
                : std::max(std::min(fetchOf(s), s - int(cls.pageLead())) - stages,
                           s - int(ii) + 2);
    for (unsigned j = 0; j < cls.page; ++j) {
      unsigned pos = wrap(f + int(j));
      readRow[pos] = 1;
      instRow[pos] = i;
      pageRow[pos] = j;
      // Pass splitting: the write-back is complete deferSpan on.
      unsigned wpos = wrap(s + 1 + int(cls.deferSpan) + int(j));
      writeRow[wpos] = 1;
      wpageRow[wpos] = j;
    }
  }
}

LogicalResult StaticFolder::assignPageEngines() {
  pageEngines.clear();
  pageEngineOf.clear();
  pageRowsOf.clear();
  if (!pageEngineCap)
    return success();
  // Each engine takes a read and a write a position. The HBM link streams
  // first, at their own positions (requests a lead before their landings,
  // writes after their producer passes), each on the first engine free
  // there, else a new one. Then each off-chip paged copy, on `columns`
  // consecutive engines (its columns move together: column k on the k-th),
  // the first run of engines (new ones past the last) on which its page
  // reads and writes can be placed at positions free on all of them,
  // earliest first:
  //  - page j of the instance of state pass s read within
  //    [min(previous state pass, s - lead), s - 1], page 0 by s - 2 (its
  //    read sets the instance's pending read, which the state pass's
  //    decision must see), the pages in order (one fetch at a time: the
  //    fetch buffer holds the instance before until its state pass), after
  //    the frame before's write of the page (the controller orders a read
  //    after an earlier write of its row);
  //  - page j written within [s + 1, next state pass] (the write-back buffer
  //    holds it until then), in order;
  //  - with an on-chip window (page-hbm-window), its prefetches at their own
  //    positions (each instance's pages from the copy's state pass a window
  //    before, in a row: its slot's counter expects them so), nothing moved.
  // So copies and streams whose traffic falls in other positions share
  // engines.
  int frame = int(ii);
  auto wrap = [&](int pos) { return unsigned(((pos % frame) + frame) % frame); };
  auto newEngine = [&]() -> PageEngine & {
    pageEngines.emplace_back();
    pageEngines.back().reads.assign(ii + 1, 0);
    pageEngines.back().writes.assign(ii + 1, 0);
    return pageEngines.back();
  };
  for (auto [k, st] : llvm::enumerate(linkStreams)) {
    if (!st.kept)
      continue;
    // At its own positions, or within `link-hbm-shift` of them where an
    // engine leaves them taken; on a new engine at its own.
    // With `link-hbm-fit-slots`, the fewest landing slots its own positions
    // need, else (to share an engine) as many as its lead takes.
    SmallVector<int> requests, writes;
    for (auto &l : st.landings)
      requests.push_back(l.q - int(st.lead));
    unsigned own = linkHbmFitSlots ? fitSlots(st, requests) : st.leadSlots();
    unsigned slots = own;
    unsigned e = 0;
    for (; e < pageEngines.size(); ++e)
      if (placeStream(st, pageEngines[e].reads, pageEngines[e].writes, linkHbmShift,
                      own, requests, writes))
        break;
    if (e == pageEngines.size() && own != st.leadSlots())
      for (e = 0, slots = st.leadSlots(); e < pageEngines.size(); ++e)
        if (placeStream(st, pageEngines[e].reads, pageEngines[e].writes, linkHbmShift,
                        slots, requests, writes))
          break;
    if (e == pageEngines.size()) {
      slots = own;
      if (e >= pageEngineCap)
        return netlist.emitError("page engines: ")
               << pageEngineCap << " engines cannot carry the link stream "
               << modelOf(st.pcls) << " -> " << modelOf(st.rcls);
      newEngine();
      requests.clear();
      writes.clear();
      for (auto &l : st.landings)
        requests.push_back(l.q - int(st.lead));
      for (int w : st.writePasses)
        for (unsigned j = 0; j < st.pages; ++j)
          writes.push_back(w + 1 + int(j));
    }
    bool collide = false;
    setStreamTables(st, requests, writes, collide);
    st.slotCount = linkHbmFitSlots ? slots : 0;
    auto &eng = pageEngines[e];
    for (unsigned q = 0; q < ii; ++q) {
      eng.reads[q] |= st.reqValid[q];
      eng.writes[q] |= st.writeValid[q];
    }
    eng.streams.push_back(unsigned(k));
    eng.dataWidth = std::max(eng.dataWidth, st.dw);
    eng.instWidth = std::max(eng.instWidth, st.tagBits());
    eng.addrWidth = std::max(eng.addrWidth, streamRowBits(st));
  }
  for (auto [c, cls] : llvm::enumerate(classes)) {
    if (!cls.page || !cls.pagesOffChip)
      continue;
    unsigned columns = 0;
    for (auto &vm : cls.virtualMemories)
      columns += vm.pages;
    if (!columns)
      continue;
    unsigned p = cls.page, n = cls.threads();
    // The state passes in the frame's circular order: a modulo schedule may
    // put a slot past the frame (it runs in the next frame's), so a copy's
    // state passes can span more than a frame of absolute positions; each
    // instance's page traffic lies between its circular neighbours.
    SmallVector<std::pair<int, unsigned>> states;
    for (unsigned i = 0; i < n; ++i)
      states.push_back({int(wrap(cls.slot[kState][i])), i});
    llvm::sort(states);
    PageRows fixed;
    if (cls.pageWindow)
      offChipRows(cls, fixed.read, fixed.inst, fixed.page, fixed.write, fixed.wpage);
    PageRows rows;
    unsigned first = 0;
    for (;; ++first) {
      if (first + columns > pageEngineCap)
        return netlist.emitError("page engines: ")
               << pageEngineCap << " engines cannot carry the page columns of "
               << cls.name << " (" << columns << " columns)";
      // The run's positions taken (engines past the last: none).
      SmallVector<uint64_t> reads(ii, 0), writes(ii, 0);
      for (unsigned e = first; e < first + columns && e < pageEngines.size(); ++e)
        for (unsigned q = 0; q < ii; ++q) {
          reads[q] |= pageEngines[e].reads[q];
          writes[q] |= pageEngines[e].writes[q];
        }
      if (cls.pageWindow) {
        bool clash = false;
        for (unsigned q = 0; q < ii && !clash; ++q)
          clash = (fixed.read[q] && reads[q]) || (fixed.write[q] && writes[q]);
        if (clash)
          continue;
        rows = fixed;
        break;
      }
      for (auto *row : {&rows.read, &rows.inst, &rows.page, &rows.write, &rows.wpage})
        row->assign(ii + 1, 0);
      bool fits = true;
      // Writes first (each read must follow the frame before's write).
      SmallVector<SmallVector<int>> writeAt(n, SmallVector<int>(p, 0));
      int lastWrite = std::numeric_limits<int>::min();
      for (unsigned x = 0; x < n && fits; ++x) {
        auto [st, i] = states[x];
        int next = x + 1 < n ? states[x + 1].first : states[0].first + frame;
        // Pass splitting: the write-back is complete deferSpan on.
        int q = std::max(st + 1 + int(cls.deferSpan), lastWrite + 1);
        for (unsigned j = 0; j < p && fits; ++j, ++q) {
          while (q <= next && writes[wrap(q)])
            ++q;
          if (q > next || int(p - 1 - j) > next - q) {
            fits = false;
            break;
          }
          writes[wrap(q)] = 1;
          rows.write[wrap(q)] = 1;
          rows.wpage[wrap(q)] = j;
          writeAt[i][j] = q;
          lastWrite = q;
        }
      }
      // Staged, the reads are placed by the evaluations' read positions, the
      // stages before the state passes (the fetch buffer is read there).
      int lastRead = std::numeric_limits<int>::min();
      for (unsigned x = 0; x < n && fits; ++x) {
        auto [sp, i] = states[x];
        int st = sp - int(cls.stages);
        int prev = (x ? states[x - 1].first : states[n - 1].first - frame) -
                   int(cls.stages);
        int q = std::max(std::min(prev, st - int(cls.pageLead())), lastRead + 1);
        for (unsigned j = 0; j < p && fits; ++j, ++q) {
          q = std::max(q, writeAt[i][j] - frame + 1);
          while (q < st && reads[wrap(q)])
            ++q;
          // Page 0's read sets the instance's pending read at its decision:
          // at least two positions before the state pass, whose decision
          // must see it (the lead's reason).
          if (q >= st || int(p - 1 - j) > st - 1 - q || (j == 0 && q > st - 2)) {
            fits = false;
            break;
          }
          reads[wrap(q)] = 1;
          rows.read[wrap(q)] = 1;
          rows.inst[wrap(q)] = i;
          rows.page[wrap(q)] = j;
          lastRead = q;
        }
      }
      if (fits)
        break;
    }
    pageRowsOf[unsigned(c)] = rows;
    while (pageEngines.size() < first + columns)
      newEngine();
    unsigned column = 0;
    for (auto [k, vm] : llvm::enumerate(cls.virtualMemories)) {
      if (!vm.pages)
        continue;
      auto &eng = pageEngines[first + column++];
      for (unsigned q = 0; q < ii; ++q) {
        eng.reads[q] |= rows.read[q];
        eng.writes[q] |= rows.write[q];
      }
      eng.members.push_back({unsigned(c), unsigned(k)});
      eng.dataWidth = std::max(eng.dataWidth, vm.dataWidth);
      eng.instWidth = std::max(eng.instWidth, vm.instWidth);
      eng.addrWidth = std::max(eng.addrWidth, vm.instWidth + vm.rowWidth);
      eng.readFirst = true;
    }
  }
  // An engine's row is whole bytes (foldsim_vmem: a byte mask of DW/8 bits
  // covers DW only when DW is a multiple of 8): a link stream's row of any
  // width, and the columns' masks (a bit a byte of theirs), fit under it.
  for (auto &eng : pageEngines)
    eng.dataWidth = unsigned(llvm::alignTo(eng.dataWidth, 8));
  // Member numbers: a column's its place among the engine's columns, a
  // stream's after them.
  for (auto [e, eng] : llvm::enumerate(pageEngines)) {
    for (auto [m, member] : llvm::enumerate(eng.members))
      pageEngineOf[member] = {unsigned(e), unsigned(m)};
    for (auto [x, k] : llvm::enumerate(eng.streams)) {
      linkStreams[k].engine = int(e);
      linkStreams[k].member = eng.members.size() + x;
    }
  }
  return success();
}

void StaticFolder::reportPageEngines(raw_ostream &os) {
  // HBM page streams: each engine's columns (class:column), width, tag.
  if (pageEngines.empty())
    return;
  unsigned columns = 0;
  for (auto &eng : pageEngines)
    columns += eng.members.size();
  os << "  page engines: " << pageEngines.size() << " engines, " << columns
     << " page columns\n";
  for (auto [e, eng] : llvm::enumerate(pageEngines)) {
    os << "    page engine " << e << " " << eng.dataWidth << " bits, tag "
       << eng.tagBits() << ", row " << eng.rowBits() << ":";
    for (auto [c, k] : eng.members)
      os << " " << classes[c].name << ":" << classes[c].virtualMemories[k].name;
    for (unsigned k : eng.streams)
      os << " link:" << modelOf(linkStreams[k].pcls) << ">"
         << modelOf(linkStreams[k].rcls);
    os << "\n";
  }
  if (pageWriteGate) {
    os << "  page write gate:";
    for (auto &cls : classes)
      if (cls.pageGate)
        os << " " << cls.name;
    os << "\n";
  }
  if (linkHbmShift) {
    unsigned moved = 0, most = 0, mostWrite = 0;
    for (auto &st : linkStreams)
      if (st.kept && st.engine >= 0) {
        moved += st.shift || st.writeShift;
        most = std::max(most, st.shift);
        mostWrite = std::max(mostWrite, st.writeShift);
      }
    os << "  page engine shifts: " << moved << " streams moved, requests up to "
       << most << ", writes up to " << mostWrite << " positions\n";
  }
  if (linkHbmFitSlots) {
    // Each built stream's landing slots, in the order of its `link stream`
    // line.
    os << "  link stream slots:";
    for (auto &st : linkStreams)
      if (st.kept)
        os << " " << st.slots();
    os << "\n";
  }
}

bool StaticFolder::zeroLink(unsigned p, unsigned c, unsigned input) const {
  auto &producer = producers[p];
  if (!zeroLatencyLinks || producer.cls < 0 || !producer.late)
    return false;
  if (!registeredLinks.empty() && registeredLinks[producer.cls])
    return false;
  // Between SLR groups a link is registered on both sides of the crossing.
  if (crosses(producer.cls, c))
    return false;
  // A staged reader reads its inputs from their read registers, its stages
  // before its state pass.
  if (classes[c].stages)
    return false;
  if (zeroLatencyChains)
    return !cutLinks.count({p, c, input});
  auto &cls = classes[c];
  if (cls.inputToOutput.test(input))
    return false;
  // The link's choice of source and the read register's bypass, between.
  return classes[producer.cls].outputDepth[producer.port] +
             cls.inputDepth[input] + 3 <=
         kZeroLinkLevels;
}

void StaticFolder::chainCuts() {
  lateReads.clear();
  cutLinks.clear();
  if (!zeroLatencyLinks || !zeroLatencyChains || !zeroChainHops)
    return;
  // Instance by instance: a late output's hops, the most zero-latency links
  // on a path into it within its host cycle (each adds its reader's choice
  // of source to the path); a link from it takes one more. A link that
  // would take a chain past `zeroChainHops` is cut: read a host cycle later
  // from its register, where the chain starts anew.
  struct Edge {
    unsigned p, pinst, c, cinst, input;
  };
  std::vector<Edge> edges;
  for (auto &consumer : consumers) {
    if (consumer.cls < 0)
      continue;
    for (auto [ci, pieces] : llvm::enumerate(consumer.pieces))
      for (auto &piece : pieces)
        if (piece.producer >= 0 && producers[piece.producer].cls >= 0 &&
            producers[piece.producer].late)
          edges.push_back({unsigned(piece.producer), piece.inst,
                           unsigned(consumer.cls), unsigned(ci),
                           consumer.port});
  }
  llvm::sort(edges, [](const Edge &a, const Edge &b) {
    return std::tie(a.p, a.pinst, a.c, a.cinst, a.input) <
           std::tie(b.p, b.pinst, b.c, b.cinst, b.input);
  });
  edges.erase(std::unique(edges.begin(), edges.end(),
                          [](const Edge &a, const Edge &b) {
                            return std::tie(a.p, a.pinst, a.c, a.cinst,
                                            a.input) ==
                                   std::tie(b.p, b.pinst, b.c, b.cinst,
                                            b.input);
                          }),
              edges.end());
  // Hops per (producer, instance), to a fixed point, held at the bound plus
  // one where port-level dependences make a loop.
  int cap = int(zeroChainHops) + 1;
  std::map<std::pair<unsigned, unsigned>, int> hops;
  auto settle = [&] {
    hops.clear();
    std::map<std::tuple<unsigned, unsigned, unsigned>, int> in;
    for (bool changed = true; changed;) {
      changed = false;
      for (auto &e : edges) {
        if (cutLinks.count({e.p, e.c, e.input}))
          continue;
        int at = hops[{e.p, e.pinst}] + 1;
        int &slot = in[{e.c, e.cinst, e.input}];
        slot = std::max(slot, at);
      }
      for (auto [p, producer] : llvm::enumerate(producers)) {
        if (producer.cls < 0 || !producer.late)
          continue;
        auto &cls = classes[producer.cls];
        for (unsigned i = 0; i < cls.threads(); ++i) {
          int at = 0;
          for (auto input : cls.deps[producer.port].set_bits())
            if (auto it = in.find({unsigned(producer.cls), i, unsigned(input)});
                it != in.end())
              at = std::max(at, it->second);
          at = std::min(at, cap);
          int &h = hops[{unsigned(p), i}];
          if (at > h) {
            h = at;
            changed = true;
          }
        }
      }
    }
  };
  // Cut the links where a chain first passes the bound, then again, until
  // none does.
  for (;;) {
    settle();
    std::set<std::tuple<unsigned, unsigned, unsigned>> cut;
    for (auto &e : edges)
      if (!cutLinks.count({e.p, e.c, e.input}) &&
          hops[{e.p, e.pinst}] + 1 > int(zeroChainHops) &&
          hops[{e.p, e.pinst}] == int(zeroChainHops))
        cut.insert({e.p, e.c, e.input});
    if (cut.empty())
      break;
    cutLinks.insert(cut.begin(), cut.end());
    if (getenv("FOLDSIM_CHAIN_DEBUG"))
      for (auto &[p, c, input] : cut)
        llvm::errs() << "chain cut " << classes[producers[p].cls].name << "."
                     << classes[producers[p].cls].outputNames[producers[p].port]
                     << " -> " << classes[c].name << "."
                     << classes[c].inputNames[input] << "\n";
  }
}

void StaticFolder::linkUses(std::vector<LinkUse> &uses) const {
  // As buildSim cuts a class input: at every bit where any instance's piece
  // starts or ends, each run of bits read from each class producer (and the
  // low bit there) its own link, at the producer instances the reader's
  // instances take.
  for (auto &consumer : consumers) {
    if (consumer.cls < 0)
      continue;
    unsigned threads = classes[consumer.cls].threads();
    std::set<unsigned> cuts{consumer.width};
    for (auto &pieces : consumer.pieces) {
      unsigned at = 0;
      for (auto &piece : pieces)
        cuts.insert(at += piece.width);
    }
    unsigned at = 0;
    for (unsigned end : cuts) {
      if (end == at)
        continue;
      struct Source {
        std::set<uint64_t> rows;
        SmallVector<unsigned> readers;
        SmallVector<uint64_t> readerRows;
      };
      std::map<std::pair<unsigned, unsigned>, Source> sources;
      // Every source as buildSim keys them: a producer's bits, or one key
      // for all constants.
      std::set<std::pair<int, unsigned>> keys;
      for (unsigned i = 0; i < threads; ++i) {
        unsigned base = 0;
        for (auto &piece : consumer.pieces[i]) {
          if (at < base + piece.width) {
            if (piece.producer >= 0) {
              keys.insert({piece.producer, piece.low + at - base});
              if (producers[piece.producer].cls >= 0) {
                auto &source =
                    sources[{unsigned(piece.producer), piece.low + at - base}];
                source.rows.insert(piece.inst);
                source.readers.push_back(i);
                source.readerRows.push_back(piece.inst);
              }
            } else {
              keys.insert({-1, 0});
            }
            break;
          }
          base += piece.width;
        }
      }
      for (auto &[key, source] : sources) {
        LinkUse use{key.first, key.second, end - at, unsigned(consumer.cls),
                    classes[consumer.cls].inputNames[consumer.port],
                    SmallVector<uint64_t>(source.rows.begin(),
                                          source.rows.end()),
                    source.readers, source.readerRows};
        use.port = consumer.port;
        use.sources = keys.size();
        use.at = at;
        uses.push_back(std::move(use));
      }
      at = end;
    }
  }
}

StaticFolder::LinkTiming
StaticFolder::linkTiming(const LinkUse &use) const {
  int frame = int(ii);
  LinkTiming t{frame, frame, frame, 0, frame};
  if (frame <= 0)
    return t;
  auto &producer = producers[use.producer];
  auto &pcls = classes[producer.cls];
  auto writeAt = [&](uint64_t inst) {
    return producer.atReady ? pcls.readySlots(producer.readyLevel)[inst]
                            : pcls.slot[kState][inst];
  };
  auto mod = [&](int s) { return ((s % frame) + frame) % frame; };
  // Writes: the entries' slots in the frame.
  SmallVector<int> writes;
  for (uint64_t r : use.rows)
    writes.push_back(mod(writeAt(r)));
  llvm::sort(writes);
  for (size_t k = 0; k < writes.size() && writes.size() > 1; ++k) {
    int next = k + 1 < writes.size() ? writes[k + 1] : writes[0] + frame;
    t.writeGap = std::min(t.writeGap, next - writes[k]);
  }
  // Reads: each reader instance's passes, with the entry each takes.
  auto &rcls = classes[use.reader];
  SmallVector<std::pair<int, uint64_t>> reads;
  for (auto [i, row] : llvm::zip(use.readers, use.readerRows)) {
    auto take = [&](int s) {
      reads.push_back({mod(s), row});
      // Host cycles from the entry's write to this read (an early link
      // reads the write of the frame before).
      int lead = s - writeAt(row) + (producer.late ? 0 : frame);
      t.lead = std::min(t.lead, lead);
    };
    take(rcls.slot[kState][i]);
    for (unsigned l = 0; l < rcls.readyLevels; ++l)
      take(rcls.readySlots(l)[i]);
  }
  llvm::sort(reads);
  for (size_t k = 0; k < reads.size() && reads.size() > 1; ++k) {
    int gap = reads[(k + 1) % reads.size()].first - reads[k].first;
    t.eventGap = std::min(t.eventGap, gap > 0 ? gap : gap + frame);
  }
  for (size_t k = 0; k < reads.size(); ++k) {
    auto &before = reads[(k + reads.size() - 1) % reads.size()];
    if (reads.size() > 1 && before.second == reads[k].second)
      continue;
    ++t.reads;
    if (reads.size() > 1) {
      int gap = reads[k].first - before.first;
      t.readGap = std::min(t.readGap, gap > 0 ? gap : gap + frame);
    }
  }
  return t;
}

unsigned StaticFolder::linkPaged(const LinkUse &use, bool banked,
                                 unsigned entries) const {
  unsigned p = linkPageRule(use.producer, use.reader);
  if (!p || ii == 0 || !banked || entries < 2 || use.sources != 1 ||
      !slrOf.empty() || shortLead)
    return 0;
  auto t = linkTiming(use);
  if (t.writeGap < int(p) || t.eventGap < int(p) || t.lead < int(2 * p + 3))
    return 0;
  return p;
}

int StaticFolder::stagedCopy(unsigned p, unsigned inst, unsigned reader,
                             ArrayRef<unsigned> readers) const {
  auto window = stagedWindow(p, inst, reader, readers);
  if (!window)
    return -1;
  int high = window->second;
  return ((high % int(ii)) + int(ii)) % int(ii);
}

std::optional<std::pair<int, int>>
StaticFolder::stagedWindow(unsigned p, unsigned inst, unsigned reader,
                           ArrayRef<unsigned> readers) const {
  if (!stagedLinks || ii == 0)
    return std::nullopt;
  auto &producer = producers[p];
  if (producer.late)
    return std::nullopt;
  // With SLR groups, only a link within one group: one between groups is
  // registered at its producer's group and read into its reader's.
  if (!slrOf.empty() && slrOf[producer.cls] != slrOf[reader])
    return std::nullopt;
  auto &pcls = classes[producer.cls];
  // The producer's write: at the end of its pass's slot, visible after it.
  int written = producer.atReady
                    ? pcls.readySlots(producer.readyLevel)[inst]
                    : pcls.slot[kState][inst];
  // Each reader instance reads at each of its passes, its read register
  // loaded the host cycle before the slot.
  auto &rcls = classes[reader];
  int first = std::numeric_limits<int>::max(), last = 0;
  for (unsigned i : readers) {
    auto take = [&](int s) {
      first = std::min(first, s);
      last = std::max(last, s);
    };
    take(rcls.slot[kState][i] - int(rcls.stages));
    for (unsigned l = 0; l < rcls.readyLevels; ++l)
      take(rcls.readySlots(l)[i]);
  }
  if (readers.empty())
    return std::nullopt;
  // The copy at the end of a host cycle t of frame f (t past the frame's
  // first, II earlier for the frame before) takes the producer's write of
  // frame f-1: after it lands (t >= written + 1 - II), before the write of
  // frame f lands (t <= written), after the last read of frame f-1
  // (t >= last - 1 - II), and in time for the first read of frame f
  // (t <= first - 2).
  int low = std::max(written + 1, last - 1) - int(ii);
  int high = std::min(written, first - 2);
  if (low > high)
    return std::nullopt;
  return std::make_pair(low, high);
}

void StaticFolder::planStagedMemories() {
  if (stagedPlannedII == int(ii) && stagedPlannedSlr == slrOf)
    return;
  stagedPlannedII = int(ii);
  stagedPlannedSlr = slrOf;
  stagedGroups.clear();
  stagedMembers.clear();
  if (!stagedLinkMemories || !stagedLinks || ii == 0)
    return;
  // The staged links, as the report and buildSim find them: an early link
  // of one producer instance with a copy position.
  std::vector<LinkUse> uses;
  linkUses(uses);
  struct Candidate {
    unsigned port, at;
    uint64_t inst;
    int low, high;
  };
  std::map<std::tuple<unsigned, unsigned, unsigned, unsigned>,
           SmallVector<Candidate>>
      byGroup;
  for (auto &use : uses) {
    auto &producer = producers[use.producer];
    if (producer.late)
      continue;
    unsigned threads = classes[producer.cls].threads();
    bool compact = compacted(use.producer, use.rows.size());
    unsigned entries = compact ? use.rows.size() : threads;
    if (entries != 1)
      continue;
    uint64_t inst = compact ? use.rows[0] : 0;
    auto window = stagedWindow(use.producer, unsigned(inst), use.reader,
                               use.readers);
    if (!window)
      continue;
    byGroup[{use.producer, use.low, use.width, use.reader}].push_back(
        {use.port, use.at, inst, window->first, window->second});
  }
  int frame = int(ii);
  auto mod = [&](int t) { return ((t % frame) + frame) % frame; };
  for (auto &[key, candidates] : byGroup) {
    if (candidates.size() < 2)
      continue;
    // Distinct copy positions, one read port: the tightest windows first,
    // each at the latest free position of its window (stagedCopy's choice
    // where free).
    SmallVector<unsigned> order(candidates.size());
    for (unsigned k = 0; k < order.size(); ++k)
      order[k] = k;
    llvm::stable_sort(order, [&](unsigned x, unsigned y) {
      auto &a = candidates[x], &b = candidates[y];
      return std::make_tuple(a.high - a.low, a.port, a.at) <
             std::make_tuple(b.high - b.low, b.port, b.at);
    });
    std::set<int> taken;
    SmallVector<std::pair<unsigned, int>> placed;
    for (unsigned k : order) {
      auto &cand = candidates[k];
      for (int t = cand.high; t >= cand.low && t > cand.high - frame; --t)
        if (!taken.count(mod(t))) {
          taken.insert(mod(t));
          placed.push_back({k, mod(t)});
          break;
        }
    }
    if (placed.size() < 2)
      continue;
    StagedGroup group{std::get<0>(key), std::get<1>(key), std::get<2>(key),
                      std::get<3>(key), {}, {}};
    for (auto [k, at] : placed)
      group.insts.push_back(candidates[k].inst);
    llvm::sort(group.insts);
    group.insts.erase(std::unique(group.insts.begin(), group.insts.end()),
                      group.insts.end());
    unsigned g = stagedGroups.size();
    for (auto [k, at] : placed) {
      auto &cand = candidates[k];
      unsigned entry =
          llvm::lower_bound(group.insts, cand.inst) - group.insts.begin();
      group.entryAt[at] = entry;
      stagedMembers[{group.reader, cand.port, cand.at, group.producer,
                     group.low}] = {g, entry, at};
    }
    stagedGroups.push_back(std::move(group));
  }
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
    // The links by source, bucketed in linear time (this test runs for every
    // output and level tried, every placement: sorting all the links each
    // time was most of the ready-pass search at 128 SMs). Duplicates stay:
    // each counts once in its target's wait and is released once, so the
    // topological count, and the answer, are those of the deduplicated set.
    for (auto [from, to] : passLinks) {
      ++edges[from + 1];
      ++waits[to];
    }
    for (unsigned n = 0; n < numPasses; ++n)
      edges[n + 1] += edges[n];
    SmallVector<unsigned> fill(edges.begin(), edges.end() - 1),
        targets(passLinks.size());
    for (auto [from, to] : passLinks)
      targets[fill[from]++] = to;
    SmallVector<unsigned> sources;
    for (unsigned n = 0; n < numPasses; ++n)
      if (!waits[n])
        sources.push_back(n);
    unsigned placed = 0;
    while (!sources.empty()) {
      unsigned n = sources.pop_back_val();
      ++placed;
      for (unsigned e = edges[n]; e < edges[n + 1]; ++e)
        if (!--waits[targets[e]])
          sources.push_back(targets[e]);
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
    if (producer.cls >= 0 && producer.late && ready[producer.cls] &&
        !stateOnly.contains(p))
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

  // What each instance reads from late outputs, once for the search.
  if (lateReads.empty()) {
    lateReads.resize(classes.size());
    for (auto [c, cls] : llvm::enumerate(classes)) {
      lateReads[c].resize(cls.threads());
      for (unsigned i = 0; i < cls.threads(); ++i)
        for (unsigned input = 0; input < cls.inputNames.size(); ++input) {
          int k = cls.consumerOf[input];
          if (k < 0)
            continue;
          for (auto &piece : consumers[k].pieces[i]) {
            if (piece.producer < 0)
              continue;
            auto &producer = producers[piece.producer];
            if (producer.cls < 0 || !producer.late)
              continue;
            // A staged reader reads its stages before its state pass.
            lateReads[c][i].push_back(
                {input, unsigned(piece.producer), piece.inst,
                 zeroLink(piece.producer, c, input)
                     ? 0
                     : int(linkLatency + crossDelay(producer.cls, c) +
                           cls.stages)});
          }
        }
    }
  }

  // The evaluations, and what each waits for within the frame: each
  // instance's ready passes by level, then its state pass.
  auto &nodes = graph.nodes;
  auto &preds = graph.preds;
  nodes.clear();
  SmallVector<unsigned> base(classes.size());
  for (auto [c, cls] : llvm::enumerate(classes)) {
    base[c] = nodes.size();
    for (unsigned i = 0; i < cls.threads(); ++i)
      for (auto pass : {kReady, kState})
        for (unsigned l = 0; l < (pass == kReady ? writes.passes[c] : 1); ++l)
          nodes.push_back({unsigned(c), i, pass, l});
  }
  auto nodeIndex = [&](unsigned c, unsigned i, Pass pass, unsigned l) {
    return base[c] + i * (writes.passes[c] + 1) +
           (pass == kReady ? l : writes.passes[c]);
  };
  preds.assign(nodes.size(), {});
  graph.why.assign(keepWhy ? nodes.size() : 0, {});
  for (auto [n, node] : llvm::enumerate(nodes)) {
    auto &cls = classes[node.cls];
    if (node.pass == kState)
      for (unsigned l = 0; l < writes.passes[node.cls]; ++l) {
        preds[n].push_back({nodeIndex(node.cls, node.inst, kReady, l), 1});
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
    for (auto &read : lateReads[node.cls][node.inst]) {
      if (!reads.test(read.input))
        continue;
      auto &producer = producers[read.producer];
      auto pass = writes.passOf(read.producer);
      unsigned l = pass == kReady ? writes.level[read.producer] : 0;
      preds[n].push_back(
          {nodeIndex(unsigned(producer.cls), read.inst, pass, l), read.latency});
      if (keepWhy)
        graph.why[n].push_back({int(read.producer), read.input});
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

/// An instance's first evaluation of the frame, as the position it reads
/// at: a staged class's state pass reads its stages before it.
static int firstSlot(const ReadyWrites &writes,
                     ArrayRef<SmallVector<int>> slots, unsigned c, unsigned i,
                     unsigned stages) {
  int at = slots[c * 2 + kState][i] - int(stages);
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

/// The first slot from `at` on a copy's slots `taken` leave free, the slots
/// taken modulo `period` (0: not), and every residue not taken.
static int firstFree(const llvm::BitVector &taken, int at, unsigned period) {
  if (!period) {
    if (at >= int(taken.size()))
      return at;
    int free = taken.find_first_unset_in(at, taken.size());
    return free < 0 ? int(taken.size()) : free;
  }
  unsigned r = at % period;
  int free = taken.find_first_unset_in(r, period);
  if (free < 0)
    return at + (period - r) + taken.find_first_unset();
  return at + (free - r);
}

/// Takes slot `at` of a copy's slots (modulo `period`, 0: not).
static void takeSlot(llvm::BitVector &taken, int at, unsigned period) {
  unsigned bit = period ? at % period : at;
  if (bit >= taken.size())
    taken.resize(std::max<unsigned>(bit + 1, taken.size() * 2));
  taken.set(bit);
}

unsigned StaticFolder::place(ArrayRef<bool> ready,
                             SmallVectorImpl<SmallVector<int>> &slots,
                             ReadyWrites &writes, unsigned below) {
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
        if (state[s] == 1) {
          // The cycle, for the search and the error: its evaluations in
          // order, from `s` to the top of the stack.
          auto at = llvm::find_if(stack, [&](auto &e) { return e.first == s; });
          lastCycleNodes.clear();
          lastCycle.clear();
          llvm::raw_string_ostream os(lastCycle);
          for (auto it = at; it != stack.end(); ++it) {
            auto &node = nodes[it->first];
            lastCycleNodes.push_back(node);
            os << (it == at ? "" : " -> ") << classes[node.cls].name << "["
               << node.inst << "]" << (node.pass == kReady ? " ready" : "");
          }
          os << " -> " << classes[nodes[s].cls].name << "[" << nodes[s].inst
             << "]";
          return 0;
        }
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
  std::vector<llvm::BitVector> busy(classes.size());
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
  // port reads an instance's pages in the slots before its state pass. A
  // multicycle copy's, its multicycle span apart: one evaluation at a time.
  std::vector<std::set<int>> pageSlots(classes.size());
  // A copy with paged links: every pass at least its link page apart.
  std::vector<std::set<int>> linkSlots(classes.size());
  // slr-serialize: a copy whose link crosses groups, each kind of pass (the
  // state pass, a ready pass of each level) its k apart: a pass's bits are
  // sent over the k host cycles after it.
  auto serial = serialSpacing();
  std::map<std::pair<unsigned, unsigned>, std::set<int>> serialSlots;
  auto serialKey = [&](unsigned n) {
    return std::make_pair(nodes[n].cls, nodes[n].pass == kState
                                            ? 0u
                                            : 1u + nodes[n].level);
  };
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), cmp);
    unsigned n = heap.back();
    heap.pop_back();
    // A staged state pass reads its stages before it, within the frame.
    int earliest = nodes[n].pass == kState ? int(classes[nodes[n].cls].stages) : 0;
    for (auto [p, latency] : preds[n])
      earliest = std::max(earliest, slotOf[p] + latency);
    unsigned c = nodes[n].cls;
    bool pages = nodes[n].pass == kState && passSpacing(c);
    auto &taken = busy[c];
    bool spaced = classes[c].linkPage > 1;
    earliest = firstFree(taken, earliest, 0);
    // A slot clashing with a state pass x of the copy (|t - x| < p) clashes
    // for every t up to x + p - 1: the search resumes at x + p, past the
    // window, instead of one slot at a time (the same slot is found; a copy
    // of thousands of paged instances otherwise rescanned its whole span).
    for (;;) {
      int next = earliest;
      auto past = [&](std::set<int> &placed, int p) {
        auto it = placed.lower_bound(earliest - p + 1);
        if (it != placed.end() && *it < earliest + p)
          next = std::max(next, *it + p);
      };
      if (pages)
        past(pageSlots[c], passSpacing(c));
      if (spaced)
        past(linkSlots[c], int(classes[c].linkPage));
      if (serial[c] > 1)
        past(serialSlots[serialKey(n)], int(serial[c]));
      if (next == earliest)
        break;
      earliest = firstFree(taken, next, 0);
    }
    takeSlot(taken, earliest, 0);
    if (pages)
      pageSlots[c].insert(earliest);
    if (spaced)
      linkSlots[c].insert(earliest);
    if (serial[c] > 1)
      serialSlots[serialKey(n)].insert(earliest);
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
    return firstSlot(writes, slots, c, i, classes[c].stages);
  };
  for (auto [c, cls] : llvm::enumerate(classes))
    for (unsigned i = 0; i < cls.threads(); ++i) {
      int s = slots[c * 2 + kState][i];
      length = std::max(length, s - first(c, i) + minStateGap(cls));
    }
  // A paged copy: the next frame's first page-in after this frame's last
  // state pass, and an instance's next page-in after its write-back.
  for (auto [c, cls] : llvm::enumerate(classes))
    if (passSpacing(c) && !pageSlots[c].empty())
      length = std::max<int>(
          {length,
           *pageSlots[c].rbegin() - *pageSlots[c].begin() + int(passSpacing(c)),
           paged(c) ? int(cls.pageLead()) + 4 : 0});
  // Serialized links: the next frame's first pass of a kind k after the last.
  for (auto &[key, placed] : serialSlots)
    if (!placed.empty())
      length = std::max<int>(length, *placed.rbegin() - *placed.begin() +
                                         int(serial[key.first]));
  // Paged links: the next frame's first pass a link page after the last.
  for (auto [c, cls] : llvm::enumerate(classes))
    if (cls.linkPage > 1 && !linkSlots[c].empty())
      length = std::max<int>(length, *linkSlots[c].rbegin() -
                                         *linkSlots[c].begin() +
                                         int(cls.linkPage));
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
  }
  // Folded port bundles: a field's pushes one bundle a host cycle.
  staggerPortPushes();
  for (auto [k, consumer] : llvm::enumerate(consumers)) {
    if (consumer.cls >= 0)
      continue;
    // (A folded bundle's request record leaves a host cycle after its last
    // push, from the record memories.)
    int at = outputSlot[k] + (portFold.on && portFold.consumerPort[k] >= 0 ? 1 : 0);
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
    unsigned period =
        placeModulo(nodes, preds, succs, height, writes, overlapped,
                    below ? std::min<unsigned>(below, length) : length);
    if (period && period < unsigned(length)) {
      slots = std::move(overlapped);
      return period;
    }
    outputSlot = aloneSlot;
    outputEarly = aloneEarly;
  }
  // `min-ii`: the frame alone, its positions past the schedule's idle.
  return std::max(length, int(minII));
}

SmallVector<unsigned> StaticFolder::readersOf(ArrayRef<Node> cycle, unsigned k,
                                              const ReadyWrites &writes) {
  SmallVector<unsigned> moved;
  auto &q = cycle[k];
  auto &p = cycle[(k + cycle.size() - 1) % cycle.size()];
  if (q.pass != kReady)
    return moved;
  // q's ready pass (of its level) waits for p: the outputs it writes that
  // read p's class.
  auto &cls = classes[q.cls];
  for (auto [o, late] : llvm::enumerate(cls.late)) {
    unsigned out = cls.producerOf[o];
    if (!late || !writes.atReady[out] || writes.level[out] != q.level ||
        stateOnly.contains(out))
      continue;
    bool reads = false;
    for (auto input : cls.deps[o].set_bits()) {
      int c = cls.consumerOf[input];
      if (c < 0)
        continue;
      for (auto &piece : consumers[c].pieces[q.inst])
        reads |= piece.producer >= 0 &&
                 producers[piece.producer].cls == int(p.cls) &&
                 producers[piece.producer].late;
    }
    if (reads)
      moved.push_back(out);
  }
  return moved;
}

unsigned StaticFolder::placeModulo(
    ArrayRef<Node> nodes, ArrayRef<SmallVector<std::pair<unsigned, int>>> preds,
    ArrayRef<SmallVector<std::pair<unsigned, int>>> succs, ArrayRef<int> height,
    const ReadyWrites &writes, SmallVectorImpl<SmallVector<int>> &slots,
    unsigned below) {
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
    if (passSpacing(c) && cls.threads())
      lower = std::max({lower, passSpacing(c) * cls.threads(),
                        paged(c) ? cls.pageLead() + 4 : 0u});
  // Nor than a copy with paged links' passes, each its link page apart.
  for (auto [c, cls] : llvm::enumerate(classes))
    if (cls.linkPage > 1)
      lower = std::max(lower, cls.linkPage * evaluations[c]);
  // Nor than a serialized copy's passes of a kind, each its k apart.
  auto serial = serialSpacing();
  for (auto [c, cls] : llvm::enumerate(classes))
    if (serial[c] > 1)
      lower = std::max(lower, serial[c] * cls.threads());
  // Nor than `min-ii`.
  lower = std::max(lower, minII);
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
  // A frame no shorter than `below` is of no use: the frames apart are.
  // The periods tried: one at a time from the lower bound for the first
  // kLinear, then skipping ahead by a doubling step until one schedules,
  // then halving the interval between the last that did not and the first
  // that did, down to adjacent periods (a modulo schedule's feasibility is
  // not strictly monotone in the period, so the frame found may be a few
  // periods above the least, never below one that failed). Each period is
  // the same attempt; the one returned is re-run last when the search went
  // past it, since an attempt leaves its slots in place.
  const unsigned kLinear = std::max(1u, moduloLinear);
  unsigned top = std::min(4 * lower + 64, below ? below - 1 : 0u);
  unsigned failed = lower - 1, scheduled = 0, step = 1;
  auto advance = [&](unsigned period, bool fits) -> unsigned {
    if (fits)
      scheduled = period;
    else
      failed = std::max(failed, period);
    if (!scheduled) {
      if (period >= top)
        return 0;
      if (period - lower + 1 < kLinear)
        return period + 1;
      unsigned next = std::min(top, period + step);
      step *= 2;
      return next;
    }
    if (scheduled - failed <= 1)
      return scheduled;
    return failed + (scheduled - failed) / 2;
  };
  // With a frame to beat (`below`), the longest period of use is tried
  // first: a choice it does not place is dropped without the search below it,
  // on the premise the doubling and halving steps above already rest on
  // (feasibility monotone in the period). The ready-pass search places most
  // choices against a frame found before (searchReady); where paged copies'
  // spacing sets the lower bound a few percent under that frame, each such
  // choice otherwise ran the whole search (16 linear periods, then doubling up
  // to the frame) to fail. A choice the probe places is then searched from
  // the lower bound as before (the same period found).
  bool probing = below && top > lower;
  auto next = [&](unsigned period, bool fits) -> unsigned {
    if (!probing)
      return advance(period, fits);
    probing = false;
    if (!fits)
      return 0;
    failed = lower - 1;
    scheduled = 0;
    step = 1;
    return lower;
  };
  bool fitsNow = false;
  moduloLower = lower;
  moduloTried = 0;
  static const char *searchLog = getenv("FOLDSIM_SEARCH_LOG");
  auto started = std::chrono::steady_clock::now();
  auto logged = [&](unsigned result) {
    if (searchLog)
      llvm::errs() << "foldsim-fold-static: modulo search lower " << lower
                   << " top " << top << " tried " << moduloTried << " -> "
                   << result << " in "
                   << std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - started)
                          .count()
                   << " ms\n";
    return result;
  };
  for (unsigned period = probing ? top : lower; period && period <= top;
       period = next(period, fitsNow)) {
    fitsNow = false;
    ++moduloTried;
    ++moduloTotal;
    // List scheduling, a copy's slots taken modulo the frame; an evaluation
    // runs at most one frame late.
    SmallVector<int> slotOf(nodes.size(), -1);
    std::vector<llvm::BitVector> busy(classes.size(),
                                      llvm::BitVector(period, false));
    // A paged copy's page port, modulo the frame: each state pass holds the
    // slots of its page-in, the page before it and its own: the residues
    // (t - lead, t], as at most two runs of bits.
    std::vector<llvm::BitVector> pagePort(classes.size());
    // A copy with paged links' link port alike: each pass holds the link
    // page's slots up to it, (t - page, t].
    std::vector<llvm::BitVector> linkPort(classes.size());
    auto windowOf = [&](int lead, int t, auto &&each) {
      lead = std::min<int>(lead, period);
      int start = ((t - lead + 1) % int(period) + int(period)) % int(period);
      if (start + lead <= int(period)) {
        each(start, start + lead);
      } else {
        each(start, int(period));
        each(0, start + lead - int(period));
      }
    };
    // A multicycle copy's evaluation holds its span's (passSpacing).
    auto window = [&](unsigned c, int t, auto &&each) {
      windowOf(passSpacing(c), t, each);
    };
    auto linkFree = [&](unsigned c, int t) {
      auto &port = linkPort[c];
      if (port.empty())
        return true;
      bool free = true;
      windowOf(classes[c].linkPage, t,
               [&](int lo, int hi) { free &= port.find_first_in(lo, hi) < 0; });
      return free;
    };
    auto linkTake = [&](unsigned c, int t, bool take) {
      auto &port = linkPort[c];
      if (port.empty())
        port.resize(period);
      windowOf(classes[c].linkPage, t, [&](int lo, int hi) {
        if (take)
          port.set(lo, hi);
        else
          port.reset(lo, hi);
      });
    };
    auto spaced = [&](unsigned n) { return classes[nodes[n].cls].linkPage > 1; };
    // slr-serialize: each kind of pass of a copy whose link crosses groups
    // holds the k slots up to it, (t - k, t], modulo the frame.
    std::map<std::pair<unsigned, unsigned>, llvm::BitVector> serialPort;
    auto serialKey = [&](unsigned n) {
      return std::make_pair(nodes[n].cls, nodes[n].pass == kState
                                              ? 0u
                                              : 1u + nodes[n].level);
    };
    auto serialed = [&](unsigned n) { return serial[nodes[n].cls] > 1; };
    auto serialFree = [&](unsigned n, int t) {
      auto it = serialPort.find(serialKey(n));
      if (it == serialPort.end())
        return true;
      bool free = true;
      windowOf(serial[nodes[n].cls], t, [&](int lo, int hi) {
        free &= it->second.find_first_in(lo, hi) < 0;
      });
      return free;
    };
    auto serialTake = [&](unsigned n, int t, bool take) {
      auto &port = serialPort[serialKey(n)];
      if (port.empty())
        port.resize(period);
      windowOf(serial[nodes[n].cls], t, [&](int lo, int hi) {
        if (take)
          port.set(lo, hi);
        else
          port.reset(lo, hi);
      });
    };
    auto pageFree = [&](unsigned c, int t) {
      auto &port = pagePort[c];
      if (port.empty())
        return true;
      bool free = true;
      window(c, t, [&](int lo, int hi) { free &= port.find_first_in(lo, hi) < 0; });
      return free;
    };
    auto pageTake = [&](unsigned c, int t, bool take) {
      auto &port = pagePort[c];
      if (port.empty())
        port.resize(period);
      window(c, t, [&](int lo, int hi) {
        if (take)
          port.set(lo, hi);
        else
          port.reset(lo, hi);
      });
    };
    auto pages = [&](unsigned n) {
      return nodes[n].pass == kState && passSpacing(nodes[n].cls) > 0;
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
      int earliest =
          nodes[n].pass == kState ? int(classes[nodes[n].cls].stages) : 0;
      for (auto [p, latency] : preds[n])
        earliest = std::max(earliest, slotOf[p] + latency);
      unsigned c = nodes[n].cls;
      auto &taken = busy[c];
      earliest = firstFree(taken, earliest, period);
      while (earliest < int(2 * period) &&
             ((pages(n) && !pageFree(c, earliest)) ||
              (spaced(n) && !linkFree(c, earliest)) ||
              (serialed(n) && !serialFree(n, earliest))))
        earliest = firstFree(taken, earliest + 1, period);
      if (earliest >= int(2 * period)) {
        fits = false;
        break;
      }
      takeSlot(taken, earliest, period);
      if (pages(n))
        pageTake(c, earliest, true);
      if (spaced(n))
        linkTake(c, earliest, true);
      if (serialed(n))
        serialTake(n, earliest, true);
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
      // The latest slot in (slotOf[n], latest] the copy leaves free, by
      // blocks of residues from `latest` down.
      auto &taken = busy[node.cls];
      taken.reset(slotOf[n] % period);
      if (pages(n))
        pageTake(node.cls, slotOf[n], false);
      if (spaced(n))
        linkTake(node.cls, slotOf[n], false);
      if (serialed(n))
        serialTake(n, slotOf[n], false);
      int at = slotOf[n];
      if (!pages(n) && !spaced(n) && !serialed(n)) {
        for (int hi = latest; hi > slotOf[n];) {
          int r = hi % int(period);
          int lo = std::max(slotOf[n] + 1, hi - r);
          int free = taken.find_last_unset_in(lo % int(period), r + 1);
          if (free >= 0) {
            at = hi - (r - free);
            break;
          }
          hi = lo - 1;
        }
      } else {
        for (int t = latest; t > slotOf[n]; --t)
          if (!taken.test(t % period) && (!pages(n) || pageFree(node.cls, t)) &&
              (!spaced(n) || linkFree(node.cls, t)) &&
              (!serialed(n) || serialFree(n, t))) {
            at = t;
            break;
          }
      }
      taken.set(at % period);
      if (pages(n))
        pageTake(node.cls, at, true);
      if (spaced(n))
        linkTake(node.cls, at, true);
      if (serialed(n))
        serialTake(n, at, true);
      slotOf[n] = at;
    }

    assignSlots(writes, nodes, slotOf, slots, classes);
    auto first = [&](unsigned c, unsigned i) {
      return firstSlot(writes, slots, c, i, classes[c].stages);
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
    }
    staggerPortPushes();
    for (auto [k, consumer] : llvm::enumerate(consumers))
      if (consumer.cls < 0 &&
          outputSlot[k] + (portFold.on && portFold.consumerPort[k] >= 0 ? 1 : 0) +
                  envSlack(k, period) > int(period))
        ok = false;
    // The least period not ruled out: done; otherwise search below it.
    if (ok && period <= failed + 1 && !probing)
      return logged(period);
    fitsNow = ok;
  }
  return logged(0);
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
  // gets shorter. Past a dozen models (a crossbar's port arbiters as classes
  // of their own, say), the search starts from no ready pass and from a ready
  // pass for every model instead, with up to two models flipped until one
  // places, and flips from there while the frame gets shorter.
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
  stateOnly.clear();
  lateReads.clear();
  // A frame is at least as long as the evaluations of its busiest copy: a
  // slot for each instance's state pass, and one for its ready pass. A
  // choice whose bound is no shorter than the best frame found cannot better
  // it, and is not placed (the result is the same; the search is not).
  auto bound = [&](const SmallVector<bool> &ready) {
    unsigned b = 0;
    for (auto [c, cls] : llvm::enumerate(classes))
      b = std::max(b, cls.threads() * (ready[c] ? 2u : 1u));
    return b;
  };
  auto tryReady = [&](const SmallVector<bool> &ready, bool orEqual = false) {
    if (best.length &&
        (orEqual ? bound(ready) > best.length : bound(ready) >= best.length)) {
      ++searchPruned;
      return false;
    }
    ++searchPlaced;
    SmallVector<SmallVector<int>> slots;
    ReadyWrites writes;
    unsigned length = place(ready, slots, writes,
                            best.length ? best.length + (orEqual ? 1 : 0) : 0);
    if (!length ||
        (best.length && (orEqual ? length > best.length : length >= best.length)))
      return false;
    best.length = length;
    best.ready = ready;
    best.slots = std::move(slots);
    best.writes = std::move(writes);
    best.outputSlot = outputSlot;
    best.outputEarly = outputEarly;
    best.moved = stateOnly;
    best.lower = moduloLower;
    best.tried = moduloTried;
    return true;
  };
  // The choice of a ready pass for candidate k: bit k of `mask`, flipped for
  // the candidates in `flips`.
  auto choice = [&](uint64_t mask, ArrayRef<unsigned> flips = {}) {
    SmallVector<bool> ready(classes.size(), false);
    for (auto [k, c] : llvm::enumerate(candidates))
      ready[c] = (k < 64 ? (mask >> k) & 1 : mask == ~uint64_t(0)) ^
                 llvm::is_contained(flips, unsigned(k));
    if (alike)
      for (auto [c, cls] : llvm::enumerate(classes))
        if (cls.splitFrom >= 0)
          ready[c] = ready[cls.splitFrom];
    return ready;
  };
  // Shorter frames from the best choice: single classes flipped; past a dozen
  // models also a model with all its copies, and a model's ready pass dropped
  // where the frame is no longer, so that the frame can shorten once the
  // passes that set it are gone.
  auto shorten = [&](bool greedy) {
    for (bool shorter = (alike || greedy) && best.length; shorter;) {
      shorter = false;
      for (auto [c, cls] : llvm::enumerate(classes)) {
        if (!late(cls))
          continue;
        auto ready = best.ready;
        ready[c] = !ready[c];
        shorter |= tryReady(ready);
      }
      for (auto [c, cls] : llvm::enumerate(classes)) {
        if (!greedy || !late(cls) || cls.splitFrom >= 0)
          continue;
        auto from = best.ready;
        auto ready = from;
        for (auto [d, other] : llvm::enumerate(classes))
          if (d == c || other.splitFrom == int(c))
            ready[d] = !from[c];
        unsigned before = best.length;
        if (tryReady(ready, /*orEqual=*/!ready[c]))
          shorter |= best.length < before || !ready[c];
      }
    }
  };
  auto search = [&] {
    bool greedy = candidates.size() > 12;
    if (greedy) {
      // No ready pass, a ready pass for every model, then those with one or
      // two models flipped, until a choice places.
      unsigned n = candidates.size();
      for (uint64_t seed : {uint64_t(0), ~uint64_t(0)})
        tryReady(choice(seed));
      for (uint64_t seed : {~uint64_t(0), uint64_t(0)})
        for (unsigned k = 0; k < n && !best.length; ++k)
          tryReady(choice(seed, {k}));
      for (uint64_t seed : {~uint64_t(0), uint64_t(0)})
        for (unsigned k = 0; k < n && !best.length; ++k)
          for (unsigned j = k + 1; j < n && !best.length; ++j)
            tryReady(choice(seed, {k, j}));
    } else {
      for (uint64_t mask = 0; mask < (uint64_t(1) << candidates.size());
           ++mask)
        tryReady(choice(mask));
    }
    shorten(greedy);
  };
  // Every late output a ready pass can compute is written there, which can
  // make a ready pass both feed a class and read it (a crossbar's port
  // arbiter between its parent's request and ready logic). Then, from a ready
  // pass for every model, the outputs that close the cycle found are moved to
  // their state pass, one cycle at a time (a parent's ready pass first, depth
  // first within a budget of placements), until a choice places.
  auto moveOut = [&] {
    auto ready = choice(~uint64_t(0));
    unsigned budget = 256;
    std::function<bool()> attempt = [&]() {
      if (!budget--)
        return false;
      SmallVector<SmallVector<int>> slots;
      ReadyWrites writes;
      if (place(ready, slots, writes))
        return tryReady(ready);
      auto cycle = lastCycleNodes;
      SmallVector<unsigned> order(cycle.size());
      std::iota(order.begin(), order.end(), 0);
      llvm::stable_sort(order, [&](unsigned x, unsigned y) {
        return classes[cycle[x].cls].depth < classes[cycle[y].cls].depth;
      });
      for (auto k : order) {
        auto moved = readersOf(cycle, k, writes);
        if (moved.empty())
          continue;
        for (auto out : moved)
          stateOnly.insert(out);
        if (attempt())
          return true;
        for (auto out : moved)
          stateOnly.erase(out);
      }
      return false;
    };
    if (attempt())
      shorten(/*greedy=*/true);
  };
  // A choice given (ready-choice): placed once, as the search would place it
  // with no frame to beat; the classes' ready outputs split by level for that
  // placement only when asked, as the `auto` search leaves them.
  if (!readyChoice.empty()) {
    StringRef spec = readyChoice;
    bool split = spec.consume_front("split:");
    SmallVector<StringRef> names;
    spec.split(names, ',', -1, /*KeepEmpty=*/false);
    SmallVector<bool> ready(classes.size(), false);
    for (StringRef name : names) {
      auto it = llvm::find_if(classes, [&](const Class &cls) {
        return cls.name == name;
      });
      if (it == classes.end()) {
        netlist.emitError("ready-choice: no class '") << name << "'";
        return best;
      }
      if (!late(*it)) {
        netlist.emitError("ready-choice: class '")
            << name << "' has no late output for a ready pass";
        return best;
      }
      ready[it - classes.begin()] = true;
    }
    bool before = splitReady;
    splitReady = split || splitReady;
    tryReady(ready);
    splitReady = before;
    stateOnly = best.moved;
    return best;
  }
  // Where no choice of one ready pass per class places: outputs moved out of
  // the cycles, and (with `auto`) a class's ready outputs split by level; the
  // shorter frame is kept, the split passes on a tie (no output written
  // later than its inputs allow). A schedule found
  // without either is kept as it is (for each assignment of SLR groups
  // alike). `always` and `rank` split them from the start (schedule).
  search();
  if (best.length)
    return best;
  moveOut();
  if (readyLevelsMode == "auto" && !splitReady) {
    Choice moved = std::move(best);
    best = Choice();
    stateOnly.clear();
    splitReady = true;
    search();
    // Split for this choice only; left split when nothing schedules, so the
    // cycle is named for the split passes (explainCycle).
    if (best.length || moved.length)
      splitReady = false;
    if (moved.length && (!best.length || moved.length < best.length))
      best = std::move(moved);
  }
  stateOnly = best.moved;
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

/// Each class's LUT sites (slr-sites) and each of `groups` SLRs' capacity
/// (slr-capacities, else slr-capacity for each).
LogicalResult StaticFolder::parseSlrSpecs(unsigned groups) {
  unsigned n = classes.size();
  slrSites.assign(n, 0);
  slrMore.assign(n, {});
  for (auto &spec : slrSiteSpecs) {
    auto [name, count] = StringRef(spec).rsplit(':');
    int c = -1;
    for (auto [i, cls] : llvm::enumerate(classes))
      if (cls.name == name)
        c = i;
    SmallVector<StringRef> parts;
    count.split(parts, '/');
    SmallVector<uint64_t> values(parts.size(), 0);
    bool bad = c < 0;
    for (auto [k, part] : llvm::enumerate(parts))
      bad |= part.getAsInteger(10, values[k]);
    if (bad)
      return netlist.emitError("slr-sites: '") << spec
             << "' is not <class>:<sites>[/<resource>...] of a class";
    slrSites[c] = values[0];
    slrMore[c].assign(values.begin() + 1, values.end());
  }
  slrCaps.assign(groups, slrCapacity);
  slrCapsMore.assign(groups, {});
  if (!slrCapacitySpecs.empty()) {
    if (slrCapacitySpecs.size() != groups)
      return netlist.emitError("slr-capacities: ")
             << slrCapacitySpecs.size() << " capacities for " << groups
             << " groups";
    for (auto [g, spec] : llvm::enumerate(slrCapacitySpecs)) {
      SmallVector<StringRef> parts;
      StringRef(spec).split(parts, '/');
      bool bad = parts[0].getAsInteger(10, slrCaps[g]);
      for (StringRef part : ArrayRef(parts).drop_front()) {
        uint64_t v = 0;
        bad |= part.getAsInteger(10, v);
        slrCapsMore[g].push_back(v);
      }
      if (bad)
        return netlist.emitError("slr-capacities: '") << spec
               << "' is not a number of sites[/<resource>...]";
    }
  }
  // The further resources: as many for every class as the capacities give
  // (a class that names fewer takes none of the rest).
  size_t more = 0;
  for (auto &caps : slrCapsMore)
    more = std::max(more, caps.size());
  for (auto &caps : slrCapsMore)
    if (caps.size() != more)
      return netlist.emitError("slr-capacities: every group needs ")
             << more << " further resources";
  for (auto [c, values] : llvm::enumerate(slrMore)) {
    if (values.size() > more)
      return netlist.emitError("slr-sites: ")
             << classes[c].name << " names more resources than the capacities";
    values.resize(more, 0);
  }
  return success();
}

/// The SLLs the design needs at the least: over the assignments of the
/// classes (each copy whole) to the region's SLRs in order, within each SLR's
/// capacity, the smallest worst boundary of link bits (a link between SLRs i
/// and j crosses every boundary between them), by branch and bound over the
/// classes, the largest first. A class larger than every SLR cannot be placed
/// whole: its own cut is not known here, and it is named.
LogicalResult StaticFolder::estimateSlls() {
  unsigned n = classes.size();
  unsigned groups = slrCapacitySpecs.size();
  if (!groups)
    return netlist.emitError("sll-estimate needs slr-capacities");
  if (failed(parseSlrSpecs(groups)))
    return failure();
  SmallVector<SmallVector<uint64_t>> bits;
  SmallVector<SmallVector<unsigned>> links;
  linkBits(bits, links);
  // A serialized class's links take a k-th of their bits in wires.
  for (unsigned p = 0; p < bits.size(); ++p)
    for (auto &b : bits[p])
      b = serialWires(p, b);
  SmallVector<unsigned> order;
  for (unsigned c = 0; c < n; ++c)
    order.push_back(c);
  llvm::stable_sort(order, [&](unsigned x, unsigned y) {
    return slrSites[x] > slrSites[y];
  });
  uint64_t largestCap = *llvm::max_element(slrCaps);
  size_t more = slrCapsMore.empty() ? 0 : slrCapsMore[0].size();
  SmallVector<uint64_t> largestMore(more, 0);
  for (auto &caps : slrCapsMore)
    for (size_t r = 0; r < more; ++r)
      largestMore[r] = std::max(largestMore[r], caps[r]);
  // Larger than every SLR in some resource: it cannot be placed whole.
  auto tooBig = [&](unsigned c) {
    if (slrSites[c] > largestCap)
      return true;
    for (size_t r = 0; r < more; ++r)
      if (slrMore[c][r] > largestMore[r])
        return true;
    return false;
  };
  std::string tooLarge;
  for (unsigned c = 0; c < n; ++c)
    if (tooBig(c))
      tooLarge += " " + classes[c].name + " (" + std::to_string(slrSites[c]) +
                  " sites)";
  // Branch over the classes that matter (the 10 largest by sites or by link
  // bits); the rest, small and lightly linked, are placed greedily at each
  // leaf, each where it adds least to the worst boundary.
  SmallVector<uint64_t> linkTotal(n, 0);
  for (unsigned c = 0; c < n; ++c)
    for (unsigned d = 0; d < n; ++d)
      linkTotal[c] += bits[c][d] + bits[d][c];
  llvm::SmallDenseSet<unsigned> branch;
  {
    SmallVector<unsigned> byLinks(order.begin(), order.end());
    llvm::stable_sort(byLinks, [&](unsigned x, unsigned y) {
      return linkTotal[x] > linkTotal[y];
    });
    for (unsigned k = 0; k < std::min<unsigned>(n, 10); ++k) {
      branch.insert(order[k]);
      branch.insert(byLinks[k]);
    }
  }
  SmallVector<unsigned> big, small;
  for (unsigned c : order)
    (branch.contains(c) ? big : small).push_back(c);
  llvm::stable_sort(small, [&](unsigned x, unsigned y) {
    return linkTotal[x] > linkTotal[y];
  });
  SmallVector<int> group(n, -1), best;
  SmallVector<uint64_t> used(groups, 0), cut(groups ? groups - 1 : 0, 0),
      bestCut;
  SmallVector<SmallVector<uint64_t>> usedMore(groups,
                                              SmallVector<uint64_t>(more, 0));
  auto place = [&](unsigned c, unsigned g, int sign) {
    used[g] += sign * int64_t(slrSites[c]);
    for (size_t r = 0; r < more; ++r)
      usedMore[g][r] += sign * int64_t(slrMore[c][r]);
  };
  uint64_t bestWorst = ~0ull, bestTotal = ~0ull, visited = 0;
  // The boundaries a class adds to in group g, given the classes placed.
  auto adds = [&](unsigned c, unsigned g, SmallVectorImpl<uint64_t> &added) {
    added.assign(cut.size(), 0);
    for (unsigned d = 0; d < n; ++d) {
      if (d == c || group[d] < 0 || unsigned(group[d]) == g)
        continue;
      unsigned h = group[d];
      uint64_t w = bits[c][d] + bits[d][c];
      for (unsigned b = std::min(g, h); b < std::max(g, h); ++b)
        added[b] += w;
    }
  };
  auto fitsIn = [&](unsigned c, unsigned g) {
    if (tooBig(c))
      return slrCaps[g] == largestCap;
    if (used[g] + slrSites[c] > slrCaps[g])
      return false;
    for (size_t r = 0; r < more; ++r)
      if (usedMore[g][r] + slrMore[c][r] > slrCapsMore[g][r])
        return false;
    return true;
  };
  std::function<void(unsigned)> visit = [&](unsigned k) {
    if (++visited > 20'000'000)
      return;
    uint64_t worst = cut.empty() ? 0 : *llvm::max_element(cut);
    if (worst > bestWorst)
      return;
    if (k == big.size()) {
      // The small classes, greedily; undone after.
      SmallVector<std::pair<unsigned, SmallVector<uint64_t>>> placed;
      bool ok = true;
      for (unsigned c : small) {
        int bestG = -1;
        uint64_t bestW = ~0ull;
        SmallVector<uint64_t> bestAdd, added;
        for (unsigned g = 0; g < groups; ++g) {
          if (!fitsIn(c, g))
            continue;
          adds(c, g, added);
          uint64_t w = 0;
          for (unsigned b = 0; b < cut.size(); ++b)
            w = std::max(w, cut[b] + added[b]);
          if (w < bestW) {
            bestW = w;
            bestG = g;
            bestAdd = added;
          }
        }
        if (bestG < 0) {
          ok = false;
          break;
        }
        for (unsigned b = 0; b < cut.size(); ++b)
          cut[b] += bestAdd[b];
        group[c] = bestG;
        place(c, bestG, 1);
        placed.push_back({c, bestAdd});
      }
      if (ok) {
        uint64_t w = cut.empty() ? 0 : *llvm::max_element(cut), total = 0;
        for (auto x : cut)
          total += x;
        if (w < bestWorst || (w == bestWorst && total < bestTotal)) {
          bestWorst = w;
          bestTotal = total;
          best = group;
          bestCut.assign(cut.begin(), cut.end());
        }
      }
      for (auto &[c, added] : llvm::reverse(placed)) {
        for (unsigned b = 0; b < cut.size(); ++b)
          cut[b] -= added[b];
        place(c, group[c], -1);
        group[c] = -1;
      }
      return;
    }
    unsigned c = big[k];
    SmallVector<uint64_t> added;
    for (unsigned g = 0; g < groups; ++g) {
      if (!fitsIn(c, g))
        continue;
      adds(c, g, added);
      for (unsigned b = 0; b < cut.size(); ++b)
        cut[b] += added[b];
      group[c] = g;
      place(c, g, 1);
      visit(k + 1);
      place(c, g, -1);
      group[c] = -1;
      for (unsigned b = 0; b < cut.size(); ++b)
        cut[b] -= added[b];
    }
  };
  visit(0);
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "  sll estimate: " << groups << " SLRs of ";
  for (auto [g, cap] : llvm::enumerate(slrCaps))
    os << (g ? "," : "") << cap;
  os << " sites";
  if (sllBudget)
    os << ", budget " << sllBudget << " a boundary";
  os << "\n";
  if (best.empty()) {
    os << "    no assignment within the SLRs' capacities"
       << (visited > 20'000'000 ? " found (search capped)" : "") << "\n";
  } else {
    os << "    boundaries:";
    for (auto [b, x] : llvm::enumerate(bestCut))
      os << " " << b << "-" << b + 1 << " " << x;
    os << " link bits (worst " << bestWorst << ", total " << bestTotal << ")"
       << (visited > 20'000'000 ? " (search capped: an upper bound)" : "")
       << (sllBudget && bestWorst > sllBudget ? ": OVER BUDGET" : "") << "\n";
    for (unsigned g = 0; g < groups; ++g) {
      uint64_t sites = 0;
      std::string members;
      for (unsigned c = 0; c < n; ++c)
        if (best[c] == int(g)) {
          sites += slrSites[c];
          members += " " + classes[c].name;
        }
      os << "    slr" << g << " (" << sites << " sites):" << members << "\n";
    }
  }
  if (!tooLarge.empty())
    os << "    larger than every SLR (its own cut unknown):" << tooLarge << "\n";
  sllReport = text;
  return success();
}

LogicalResult StaticFolder::chooseGroups() {
  unsigned n = classes.size();
  auto classNamed = [&](StringRef name) -> int {
    for (auto [c, cls] : llvm::enumerate(classes))
      if (cls.name == name)
        return c;
    return -1;
  };
  if (failed(parseSlrSpecs(slrGroups)))
    return failure();
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
  // opened in order (the groups are alike) unless some class is pinned or
  // the groups' capacities differ.
  constexpr unsigned kKept = 64;
  SmallVector<unsigned> order;
  for (unsigned c = 0; c < n; ++c)
    order.push_back(c);
  llvm::stable_sort(order, [&](unsigned x, unsigned y) {
    return slrSites[x] > slrSites[y];
  });
  bool anyPinned = llvm::any_of(pinned, [](int g) { return g >= 0; }) ||
                   !slrCapacitySpecs.empty();
  std::vector<std::pair<uint64_t, SmallVector<int>>> kept;
  SmallVector<int> group(n, -1);
  SmallVector<uint64_t> used(slrGroups, 0);
  // With slr-capacities the groups are the region's SLRs in order: a link
  // between groups i and j crosses every boundary between them, and with
  // slr-sll-budget no boundary may carry more link bits than the budget.
  bool budgeted = sllBudget && !slrCapacitySpecs.empty();
  SmallVector<uint64_t> boundary(slrGroups ? slrGroups - 1 : 0, 0);
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
          if (slrCaps[g] && used[g] + slrSites[c] > slrCaps[g])
            continue;
          // A link between groups takes a k-th of its bits in wires
          // (slr-serialize).
          auto wires = [&](unsigned x, unsigned y) {
            return serialWires(x, bits[x][y]) + serialWires(y, bits[y][x]);
          };
          uint64_t more = 0;
          for (unsigned j = 0; j < k; ++j) {
            unsigned d = order[j];
            if (group[d] != int(g))
              more += wires(c, d);
          }
          SmallVector<uint64_t> added(boundary.size(), 0);
          if (budgeted) {
            bool over = false;
            for (unsigned j = 0; j < k; ++j) {
              unsigned d = order[j];
              unsigned h = group[d];
              if (h == g)
                continue;
              for (unsigned b = std::min(g, h); b < std::max(g, h); ++b)
                added[b] += wires(c, d);
            }
            for (unsigned b = 0; b < boundary.size(); ++b)
              over |= boundary[b] + added[b] > sllBudget;
            if (over)
              continue;
            for (unsigned b = 0; b < boundary.size(); ++b)
              boundary[b] += added[b];
          }
          group[c] = g;
          used[g] += slrSites[c];
          visit(k + 1, std::max(opened, g + 1), cut + more);
          used[g] -= slrSites[c];
          group[c] = -1;
          for (unsigned b = 0; b < boundary.size(); ++b)
            boundary[b] -= added[b];
        }
      };
  visit(0, 0, 0);
  if (kept.empty()) {
    std::string caps = std::to_string(slrCapacity);
    if (!slrCapacitySpecs.empty()) {
      caps.clear();
      for (auto [g, cap] : llvm::enumerate(slrCaps))
        caps += (g ? "," : "") + std::to_string(cap);
    }
    uint64_t largest = 0;
    for (auto sites : slrSites)
      largest = std::max(largest, sites);
    return netlist.emitError("slr-groups: no assignment of the classes to ")
           << slrGroups << " groups of at most " << caps
           << " sites (the largest class copy: " << largest << " sites)"
           << (budgeted ? " with at most " + std::to_string(sllBudget) +
                              " link bits over each SLR boundary"
                        : std::string());
  }

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
      // A memory read here (its read register, or combinationally): read at
      // a ready pass too, so it keeps its store.
      if (memory)
        regs.insert(memory);
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

std::pair<unsigned, uint64_t>
StaticFolder::memoriesAsRows(Class &cls, const DenseSet<Operation *> &ready,
                             const DenseMap<Operation *, unsigned> &virtualOf) {
  auto *body = cls.model.getBodyBlock();
  auto *folding = cls.folding.get();
  Value clock = body->getArgument(cls.clockInput);
  SmallVector<seq::FirMemOp> memories;
  llvm::Regex only(pageMemories.empty() ? ".*" : pageMemories);
  llvm::Regex chosenRx(pageRows);
  StringRef model =
      cls.splitFrom >= 0 ? StringRef(classes[cls.splitFrom].name) : StringRef(cls.name);
  for (auto memory : body->getOps<seq::FirMemOp>()) {
    auto type = memory.getType();
    StringRef memName = memory.getName().value_or("");
    // Named by page-rows: whatever its size.
    bool chosen = !pageRows.empty() && chosenRx.match((model + ":" + memName).str());
    if (!chosen && (!pageMemoryBits || !only.match(memName) ||
                    uint64_t(type.getDepth()) * type.getWidth() > pageMemoryBits))
      continue;
    // A folded pipeline's result store is the pipeline's, not the model's.
    if ((folding && (folding->physical.contains(memory) ||
                     folding->foldOfStore.count(memory))) ||
        ready.contains(memory) || virtualOf.count(memory) ||
        memory.getReadLatency() > 1 || memory.getInit())
      continue;
    if (llvm::all_of(memory->getUsers(), [&](Operation *user) {
          if (isa<seq::FirMemReadOp>(user))
            return true;
          auto write = dyn_cast<seq::FirMemWriteOp>(user);
          return write && write.getClk() == clock;
        }))
      memories.push_back(memory);
  }
  uint64_t bits = 0;
  for (auto memory : memories) {
    auto type = memory.getType();
    unsigned depth = type.getDepth(), width = type.getWidth();
    bits += uint64_t(depth) * width;
    ImplicitLocOpBuilder b(memory.getLoc(), memory);
    auto rowType = b.getIntegerType(width);
    BackedgeBuilder edges(b, memory.getLoc());
    SmallVector<Backedge> nexts;
    SmallVector<Value> rows;
    std::string name = memory.getName().value_or("mem").str();
    for (unsigned e = 0; e < depth; ++e) {
      nexts.push_back(edges.get(rowType));
      rows.push_back(seq::FirRegOp::create(b, nexts.back(), clock,
                                           b.getStringAttr(name + "_row" + Twine(e))));
    }
    SmallVector<Value> next(rows.begin(), rows.end());
    // A registered read, as the target's: a memory reading old data
    // registers the row read where enabled; any other registers the address
    // where enabled and reads the rows at it (firtool's lowering of a
    // latency-1 read), so a write after the read shows, as in the target's
    // write-first RAM. Either register pages with the rows.
    bool registered = memory.getReadLatency() == 1;
    bool readNew = memory.getRuw() != seq::RUW::Old;
    auto rowAt = [&](ArrayRef<Value> source, Value addr) -> Value {
      if (depth == 1)
        return source[0];
      SmallVector<Value> elements(source.rbegin(), source.rend());
      return hw::ArrayGetOp::create(b, hw::ArrayCreateOp::create(b, elements), addr);
    };
    SmallVector<std::tuple<seq::FirMemReadOp, Backedge, Value>> outputs;
    for (auto *user : llvm::make_early_inc_range(memory->getUsers())) {
      if (auto read = dyn_cast<seq::FirMemReadOp>(user)) {
        b.setInsertionPoint(read);
        Value data;
        if (registered && readNew) {
          auto addrNext = edges.get(read.getAddress().getType());
          Value addr = seq::FirRegOp::create(b, addrNext, clock,
                                             b.getStringAttr(name + "_raddr"));
          outputs.push_back({read, addrNext, addr});
          data = rowAt(rows, addr);
        } else if (registered) {
          auto outNext = edges.get(rowType);
          data = seq::FirRegOp::create(b, outNext, clock,
                                       b.getStringAttr(name + "_rdata"));
          outputs.push_back({read, outNext, data});
        } else {
          if (depth == 1) {
            data = rows[0];
          } else {
            SmallVector<Value> elements(rows.rbegin(), rows.rend());
            data = hw::ArrayGetOp::create(b, hw::ArrayCreateOp::create(b, elements),
                                          read.getAddress());
          }
          // A combinational read not enabled reads nothing (firtool's x, as
          // the threaded memory's read gives it): zero.
          if (read.getEnable())
            data = comb::MuxOp::create(b, read.getEnable(), data,
                                       hw::ConstantOp::create(b, rowType, 0));
        }
        read.getData().replaceAllUsesWith(data);
        if (!registered)
          read.erase();
        continue;
      }
      auto write = cast<seq::FirMemWriteOp>(user);
      b.setInsertionPoint(write);
      Value enable = write.getEnable();
      unsigned addrWidth = hw::getBitWidth(write.getAddress().getType());
      Value data = write.getData();
      unsigned lanes = write.getMask() ? hw::getBitWidth(write.getMask().getType()) : 1;
      for (unsigned e = 0; e < depth; ++e) {
        Value hit = addrWidth
                        ? Value(comb::ICmpOp::create(
                              b, comb::ICmpPredicate::eq, write.getAddress(),
                              hw::ConstantOp::create(b, b.getIntegerType(addrWidth), e)))
                        : Value(hw::ConstantOp::create(b, b.getI1Type(), 1));
        if (enable)
          hit = comb::AndOp::create(b, hit, enable);
        Value value = data;
        if (lanes > 1) {
          // Each mask bit its lane of the row.
          unsigned lane = width / lanes;
          SmallVector<Value> parts;
          for (unsigned l = lanes; l-- > 0;) {
            Value on = b.createOrFold<comb::ExtractOp>(b.getI1Type(), write.getMask(), l);
            parts.push_back(comb::MuxOp::create(
                b, on, b.createOrFold<comb::ExtractOp>(b.getIntegerType(lane), data, l * lane),
                b.createOrFold<comb::ExtractOp>(b.getIntegerType(lane), next[e], l * lane)));
          }
          value = b.createOrFold<comb::ConcatOp>(parts);
        }
        next[e] = comb::MuxOp::create(b, hit, value, next[e]);
      }
      write.erase();
    }
    for (auto &[read, outNext, self] : outputs) {
      b.setInsertionPoint(read);
      // The address (new) or the row read as of the cycle's start (old).
      Value value = readNew ? read.getAddress() : rowAt(rows, read.getAddress());
      // The register holds where the read is not enabled.
      Value loaded = read.getEnable() ? Value(comb::MuxOp::create(b, read.getEnable(), value,
                                                                  self))
                                      : value;
      read.erase();
      outNext.setValue(loaded);
    }
    for (unsigned e = 0; e < depth; ++e)
      nexts[e].setValue(next[e]);
    memory.erase();
  }
  return {unsigned(memories.size()), bits};
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
  // Computed once a class: the scheduler asks for every instance of every
  // choice it places.
  if (cls.minGap >= 0)
    return cls.minGap;
  int gap = writeBackLatency(cls);
  // A staged paged class: an instance's write-back, its page a host cycle
  // after the state pass, ends before its next fetch, a page before the
  // next evaluation's read.
  if (cls.stages && cls.page)
    gap = std::max(gap, 2 * int(cls.pageLead()) + 2);
  // A staged class's writes reach its next evaluation's reads through its
  // stores alone, no bypass.
  if (cls.stages)
    gap = std::max(gap, 4);
  // Pass splitting: the deferred writes land before the next evaluation's
  // reads (a block RAM store's a host cycle early), no bypass.
  if (cls.deferSpan)
    gap = std::max(gap, int(cls.deferSpan) + 4 +
                            (cls.page ? 2 * int(cls.page) + int(cls.pageLead()) : 0));
  if (!virtualMemoryPattern.empty()) {
    llvm::Regex pattern(virtualMemoryPattern);
    StringRef model = cls.splitFrom >= 0 ? StringRef(classes[cls.splitFrom].name)
                                         : StringRef(cls.name);
    auto *folding = cls.folding.get();
    // As buildCopy finds its virtual memories: by name or <class model>:<name>.
    // A held-read class's requests leave a host cycle after the state pass:
    // a host cycle more.
    for (auto memory :
         const_cast<hw::HWModuleOp &>(cls.model).getBodyBlock()->getOps<seq::FirMemOp>()) {
      StringRef name = memory.getName().value_or("");
      if (!(folding && folding->physical.contains(memory)) &&
          (pattern.match(name) || pattern.match((model + ":" + name).str()))) {
        gap = std::max(gap, decisionLead() + (cls.heldReads ? 2 : 1));
        break;
      }
    }
  }
  cls.minGap = gap;
  return gap;
}

void StaticFolder::choosePageGates() {
  // `page-write-gate`: an off-chip paged class on engines, of more than one
  // page and without a window.
  for (auto &cls : classes)
    cls.pageGate = pageWriteGate && pageEngineCap && cls.page >= 2 &&
                   cls.pagesOffChip && !cls.pageWindow;
}

void StaticFolder::chooseWindows() {
  for (auto &cls : classes) {
    cls.pageWindow = 0;
    cls.windowSlot.clear();
    if (!cls.page || !cls.pagesOffChip || !pageHbmWindow || cls.threads() < 2)
      continue;
    // The copy's state passes in order, and each instance's place in it.
    unsigned n = cls.threads();
    SmallVector<unsigned> order(n);
    for (unsigned i = 0; i < n; ++i)
      order[i] = i;
    llvm::sort(order, [&](unsigned x, unsigned y) {
      return cls.slot[kState][x] < cls.slot[kState][y];
    });
    auto stateAt = [&](int k) {
      // The k-th state pass of the cycle (negative: the frame before).
      int f = k < 0 ? -1 : 0;
      int r = ((k % int(n)) + int(n)) % int(n);
      return cls.slot[kState][order[r]] + f * int(ii);
    };
    int lead = cls.pageLead();
    for (unsigned w = std::min(pageHbmWindow, n - 1); w >= 1; --w) {
      if (n % w)
        continue;
      bool ok = true;
      for (int k = 0; ok && k < int(n); ++k) {
        // Staged, the prefetch is at the read position w before (its slot's
        // last fetch from the window is a lead before it).
        int s = stateAt(k), prefetch = stateAt(k - int(w)) - int(cls.stages);
        // Its pages pending before its fetch is decided (a lead before its
        // evaluation's read, the stages before its state pass), and the next
        // frame's prefetch after this frame's write-back.
        ok = prefetch + 2 <= s - int(cls.stages) - lead &&
             prefetch + int(ii) >= s + 2;
      }
      if (!ok)
        continue;
      cls.pageWindow = w;
      cls.windowSlot.assign(n, 0);
      for (unsigned k = 0; k < n; ++k)
        cls.windowSlot[order[k]] = k % w;
      break;
    }
    llvm::errs() << "foldsim-fold-static: " << cls.name << " off-chip page window "
                 << cls.pageWindow
                 << (cls.pageWindow ? "" : " (none fits the schedule: fetched directly)")
                 << "\n";
  }
}

LogicalResult StaticFolder::pagesOffChip() {
  for (auto &cls : classes) {
    StringRef model =
        cls.splitFrom >= 0 ? StringRef(classes[cls.splitFrom].name) : cls.name;
    for (auto &rule : storageRules)
      if (rule.model == model && !rule.regs && rule.store == "pages" &&
          rule.kind == "hbm")
        cls.pagesOffChip = cls.page > 0;
  }
  return success();
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

void StaticFolder::reportLinks(raw_ostream &os) {
  // The links' storage, as buildSim makes it: a register for a late link of
  // one entry, two (staged) for an early one where the schedule allows, else
  // a memory of an entry per producer instance (per instance read, with
  // compact links), doubled by frame parity for an early one, in block or
  // ultra RAM where deep enough and never forwarded, else distributed RAM.
  // Costs by the cost model's rules (foldsim_cost.mem_cost's distributed
  // RAM, foldsim_storage's RAMB36 and URAM288 shapes); a line per kind
  // (`link kinds`) for the fold-degree model to price.
  std::vector<LinkUse> uses;
  linkUses(uses);
  bool haveSchedule = !frameAt.empty();
  auto lutram = [](uint64_t depth, unsigned width) {
    if (depth <= 32)
      return width * 0.57;
    uint64_t blocks = (depth + 63) / 64;
    return width * blocks * 1.14 +
           (depth > 64 ? width * (blocks - 1) / 3.0 : 0.0);
  };
  // RAMB36 holding depth x width (one port pair): 72x512, 36x1k, 18x2k,
  // 9x4k, 4x8k, 2x16k, 1x32k; an 18K half where it fits one.
  auto bram36 = [](uint64_t depth, unsigned width) {
    static const std::pair<unsigned, unsigned> shapes36[] = {
        {512, 72}, {1024, 36}, {2048, 18}, {4096, 9},
        {8192, 4}, {16384, 2}, {32768, 1}};
    static const std::pair<unsigned, unsigned> shapes18[] = {
        {512, 36}, {1024, 18}, {2048, 9}, {4096, 4},
        {8192, 2}, {16384, 1}};
    double best = 1e18;
    for (auto [d, w] : shapes36)
      best = std::min(best, double((width + w - 1) / w) *
                                double((depth + d - 1) / d));
    for (auto [d, w] : shapes18)
      best = std::min(best, 0.5 * double((width + w - 1) / w) *
                                double((depth + d - 1) / d));
    return best;
  };
  auto uram = [](uint64_t depth, unsigned width) {
    return double((width + 71) / 72) * double((depth + 4095) / 4096);
  };
  // Each producer's reading classes: several share a register of its bits.
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
  struct Kind {
    unsigned memories = 0;
    uint64_t bits = 0;
    double cost = 0;
  };
  Kind distributed, block, ultra, registers, staged, paged, hbm;
  std::vector<std::string> useLines;
  if (haveSchedule)
    planStagedMemories();
  // `staged-link-memories`: each group's memory, once (distributed RAM, an
  // entry per producer instance), and its links' second registers.
  Kind stagedMemories;
  std::set<unsigned> stagedGroupsSeen;
  if (haveSchedule && !linkStreams.empty())
    (void)planStreams();
  auto streamKept = [&](const LinkUse &use) {
    auto it = streamMembers.find(
        {use.reader, use.port, use.at, use.producer, use.low});
    return it != streamMembers.end() && it->second.kept;
  };
  uint64_t compactSaved = 0;
  unsigned compactMemories = 0;
  // By producer and reading class: (LUTRAM, bits).
  std::map<std::pair<unsigned, unsigned>, std::pair<double, uint64_t>> byPair;
  std::unique_ptr<llvm::raw_fd_ostream> dump;
  if (const char *path = getenv("FOLDSIM_LINK_DUMP")) {
    std::error_code ec;
    dump = std::make_unique<llvm::raw_fd_ostream>(path, ec);
    if (ec)
      dump.reset();
  }
  for (auto &use : uses) {
    auto &producer = producers[use.producer];
    bool banked = !producer.late;
    unsigned threads = classes[producer.cls].threads();
    bool compact = compacted(use.producer, use.rows.size());
    unsigned entries = compact ? use.rows.size() : threads;
    uint64_t full = linkDepth(threads, banked);
    StringRef kind;
    uint64_t depth = 0;
    bool syncOk = false;
    unsigned page = 0;
    if (entries == 1 && !banked) {
      kind = "register";
      depth = 1;
      registers.memories++;
      registers.bits += use.width;
    } else if (entries == 1 && banked && haveSchedule &&
               stagedCopy(use.producer, compact ? unsigned(use.rows[0]) : 0,
                          use.reader, use.readers) >= 0) {
      kind = "staged";
      depth = 2;
      staged.memories++;
      if (auto *member = stagedMemberOf(use.reader, use.port, use.at,
                                        use.producer, use.low)) {
        staged.bits += use.width;
        if (stagedGroupsSeen.insert(member->group).second) {
          auto &group = stagedGroups[member->group];
          uint64_t rows = group.insts.size();
          stagedMemories.memories++;
          stagedMemories.bits += rows * use.width;
          stagedMemories.cost += lutram(rows, use.width);
          distributed.memories++;
          distributed.bits += rows * use.width;
          distributed.cost += lutram(rows, use.width);
          auto &pair = byPair[{unsigned(producer.cls), use.reader}];
          pair.first += lutram(rows, use.width);
          pair.second += rows * use.width;
          // For the fold-degree model's link tiers: a distributed RAM
          // only (read at each link's copy position, not the reader's).
          std::string line;
          llvm::raw_string_ostream ls(line);
          ls << "    link use " << modelOf(producer.cls) << " "
             << modelOf(use.reader) << " " << use.width << " " << rows << " "
             << rows << " distributed 0 " << int(ii) << " " << int(ii) << " "
             << int(ii) << " " << group.entryAt.size() << " 0 1 "
             << passesOf(producer.cls) << " " << passesOf(use.reader)
             << " 0\n";
          useLines.push_back(std::move(line));
        }
      } else {
        staged.bits += 2 * use.width;
      }
    } else if (haveSchedule && streamKept(use)) {
      // In device memory, behind its stream's virtual memory.
      kind = "hbm";
      depth = linkDepth(entries, banked);
      syncOk = true;
      hbm.memories++;
      hbm.bits += depth * use.width;
    } else {
      depth = linkDepth(entries, banked);
      StringRef style;
      if (haveSchedule && entries != 1 && slrOf.empty() &&
          (use.sources == 1 || !shortLead)) {
        // As buildSim decides: never forwarded, taken now or forwarded
        // from a shared register.
        unsigned rthreads = classes[use.reader].threads();
        SmallVector<bool> takes(rthreads, false);
        SmallVector<uint64_t> rows(rthreads, 0);
        for (auto [i, r] : llvm::zip(use.readers, use.readerRows)) {
          takes[i] = true;
          rows[i] = r;
        }
        bool forward =
            oneCycleLinks && needsForward(use.producer, use.reader, takes, rows);
        bool now = zeroLink(use.producer, use.reader, use.port) &&
                   takesNow(use.producer, use.reader, takes, rows);
        bool late = shareLinkWrites && classReaders[use.producer] > 1;
        bool forwardLate =
            late && needsForward(use.producer, use.reader, takes, rows, 2);
        // The only source: read into the RAM's output register, which is
        // the read register. One of several: read a host cycle ahead, where
        // no write of an entry it reads lands in the host cycles between.
        if (use.sources == 1 ? !forward && !now && !forwardLate
                             : !forward && !now &&
                                   !needsForward(use.producer, use.reader,
                                                 takes, rows, 2) &&
                                   !(late && needsForward(use.producer,
                                                          use.reader, takes,
                                                          rows, 3)))
          syncOk = true;
      }
      if (syncOk)
        style = linkStyle(use.producer, use.reader, depth, use.width);
      // A paged link: p rows an entry of a p-th of its bits.
      if (!style.empty())
        page = linkPaged(use, banked, entries);
      uint64_t rows = page ? depth * llvm::PowerOf2Ceil(page) : depth;
      unsigned cols = page ? (use.width + page - 1) / page : use.width;
      if (page) {
        paged.memories++;
        paged.bits += 2 * use.width;
      }
      if (style == "ultra") {
        kind = "ultra";
        ultra.memories++;
        ultra.bits += rows * cols;
        ultra.cost += uram(rows, cols);
      } else if (style == "block") {
        kind = "block";
        block.memories++;
        block.bits += rows * cols;
        block.cost += bram36(rows, cols);
      } else {
        kind = "distributed";
        distributed.memories++;
        distributed.bits += depth * use.width;
        distributed.cost += lutram(depth, use.width);
        auto &pair = byPair[{unsigned(producer.cls), use.reader}];
        pair.first += lutram(depth, use.width);
        pair.second += depth * use.width;
      }
    }
    if (entries < threads) {
      ++compactMemories;
      compactSaved += (full - depth) * use.width;
    }
    if (haveSchedule && kind != "register" && kind != "staged") {
      // For the fold-degree model's link tiers: the link's models, shape,
      // kind, whether block or ultra RAM may hold it, and its accesses.
      auto timing = linkTiming(use);
      std::string line;
      llvm::raw_string_ostream ls(line);
      ls << "    link use " << modelOf(producer.cls) << " "
         << modelOf(use.reader) << " " << use.width << " " << depth << " "
         << entries << " " << kind << " " << (syncOk ? 1 : 0) << " "
         << timing.writeGap << " " << timing.readGap << " " << timing.lead
         << " " << timing.reads << " " << (banked ? 1 : 0) << " "
         << use.sources << " " << passesOf(producer.cls) << " "
         << passesOf(use.reader) << " " << page << "\n";
      useLines.push_back(std::move(line));
    }
    if (dump) {
      auto &pc = classes[producer.cls];
      *dump << pc.name << " " << pc.outputNames[producer.port] << " "
            << use.low << " " << use.width << " " << threads << " "
            << use.rows.size() << " " << (banked ? 1 : 0) << " "
            << classes[use.reader].name << " "
            << classes[use.reader].threads() << " " << use.input << " "
            << kind << " " << depth << " " << producer.name << "_to_"
            << classes[use.reader].name << "_" << use.input << "_" << use.at;
      if (haveSchedule) {
        auto timing = linkTiming(use);
        *dump << " " << timing.writeGap << " " << timing.readGap << " "
              << timing.lead << " " << timing.reads;
      }
      *dump << "\n";
    }
  }
  if (constantOutputs)
    os << "  constant outputs: " << constantOutputs
       << " (taken by their readers as constants)\n";
  os << "  link storage: " << distributed.memories + block.memories +
                                  ultra.memories
     << " memories, "
     << distributed.bits + block.bits + ultra.bits << " bits, "
     << uint64_t(distributed.cost) << " LUTRAM LUTs; " << registers.bits
     << " register bits";
  if (shareLinks)
    os << "; shared outputs " << sharedOutputs;
  if (stagedLinks)
    os << "; staged: " << staged.memories << " links, " << staged.bits
       << " register bits";
  if (stagedLinkMemories)
    os << "; staged memories: " << stagedMemories.memories << " ("
       << stagedMembers.size() << " links), " << stagedMemories.bits
       << " bits, " << uint64_t(stagedMemories.cost) << " LUTRAM LUTs";
  if (compactLinks)
    os << "; compact: " << compactMemories << " links, " << compactSaved
       << " bits saved";
  if (paged.memories)
    os << "; paged: " << paged.memories << " links, " << paged.bits
       << " flops";
  os << "\n";
  // For the fold-degree model: memories, bits and cost per kind (LUTRAM
  // LUTs, RAMB36, URAM288; flip-flops for registers).
  os << "  link kinds: distributed " << distributed.memories << " "
     << distributed.bits << " " << uint64_t(std::ceil(distributed.cost))
     << "; block " << block.memories << " " << block.bits << " "
     << llvm::format("%.1f", block.cost) << "; ultra " << ultra.memories
     << " " << ultra.bits << " " << llvm::format("%.0f", ultra.cost)
     << "; registers "
     << registers.memories + staged.memories << " "
     << registers.bits + staged.bits << "\n";
  for (auto &line : useLines)
    os << line;
  // HBM link streams: for the fold-degree model, each one's word, rows and
  // traffic a frame (rows written: every producer instance's pages; read:
  // the landings).
  if (haveSchedule && !linkStreams.empty()) {
    unsigned built = 0;
    for (auto &st : linkStreams)
      built += st.kept > 0;
    os << "  link streams: " << built << " streams, " << hbm.memories
       << " links, " << hbm.bits << " bits\n";
    for (auto &st : linkStreams)
      if (st.kept)
        os << "    link stream " << modelOf(st.pcls) << " " << modelOf(st.rcls)
           << " " << st.kept << " " << st.dropped << " " << st.wordBits << " "
           << st.dw << " " << st.pages << " "
           << uint64_t(classes[st.pcls].threads()) * st.pages << " "
           << st.readRows << " " << st.maxNeed << " " << st.lead << "\n";
  }
  // For the fold-degree model's link tier: every pair a stream could carry
  // (rule or not): its word, row, rows an instance, the most rows a reader
  // pass lands, and rows read and written a frame.
  if (haveSchedule) {
    std::vector<LinkStream> cands;
    layoutStreamsOf([](const LinkUse &) { return true; }, cands, nullptr);
    for (auto &st : cands)
      os << "    link pair " << modelOf(st.pcls) << " " << modelOf(st.rcls) << " "
         << st.wordBits << " " << st.dw << " " << st.pages << " " << st.maxNeed
         << " " << st.readRows << " "
         << uint64_t(classes[st.pcls].threads()) * st.pages << " "
         << passesOf(st.pcls) << " " << passesOf(st.rcls) << "\n";
  }
  // The largest in distributed RAM, by the classes they join.
  SmallVector<std::pair<std::pair<unsigned, unsigned>, std::pair<double, uint64_t>>>
      largest(byPair.begin(), byPair.end());
  llvm::sort(largest, [](auto &x, auto &y) {
    return x.second.first > y.second.first;
  });
  for (auto &[pair, cost] : ArrayRef(largest).take_front(8))
    os << "    links " << classes[pair.first].name << " -> "
       << classes[pair.second].name << ": " << uint64_t(cost.first)
       << " LUTRAM LUTs, " << cost.second << " bits\n";
}

/// The inputs named on a late output's line of the schedule report.
static constexpr unsigned kNamedDeps = 8;

/// A class's slots on the report: each run of three or more consecutive
/// slots as `first..last` (foldsim_degree.parse_classes reads both forms).
static void printSlots(raw_ostream &os, ArrayRef<int> slots) {
  for (size_t i = 0; i < slots.size();) {
    size_t j = i;
    while (j + 1 < slots.size() && slots[j + 1] == slots[j] + 1)
      ++j;
    if (j - i >= 2) {
      os << " " << slots[i] << ".." << slots[j];
      i = j + 1;
    } else {
      os << " " << slots[i];
      ++i;
    }
  }
}

LogicalResult StaticFolder::schedule() {
  chainCuts();
  auto late = [](const Class &cls) {
    return llvm::any_of(cls.late, [](bool late) { return late; });
  };
  // Past a dozen classes with late outputs, the copies of a model choose
  // alike; past a dozen models, the search is greedy (searchReady).
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
  if (sllEstimate && failed(estimateSlls()))
    return failure();
  if (slrGroups && failed(chooseGroups()))
    return failure();
  Choice best = searchReady();
  if (best.tried) {
    // The report gives the chosen schedule's search (the last one placed may
    // be another choice's).
    moduloLower = best.lower;
    moduloTried = best.tried;
  }
  if (!best.length) {
    // Named for the choice that writes the most outputs at ready passes:
    // every model with late outputs given one.
    SmallVector<bool> ready(classes.size(), false);
    for (auto [c, cls] : llvm::enumerate(classes))
      ready[c] = late(cls);
    auto diag = netlist.emitError("the evaluations of a frame depend on each "
                                  "other in a cycle for every choice of ready "
                                  "passes");
    if (!lastCycle.empty())
      diag << " (last tried: " << lastCycle << ")";
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
      int first = firstSlot(best.writes, best.slots, c, i, cls.stages);
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
      cls.pagesOffChip = false;
      unpaged = true;
    }
  if (unpaged)
    return schedule();
  // An HBM link stream the schedule keeps no member of builds nothing (its
  // links are on chip), so the spacing its ports asked for is dropped: such
  // streams are excluded and the frame scheduled again without them. Once
  // excluded a stream stays so (none of its members kept), so the streams
  // excluded only grow and this ends. The members kept are planned anew on
  // the schedule that stays (planStreams, from the streams as laid out).
  if (!laidOutStreams.empty()) {
    linkStreams = laidOutStreams;
    streamsPlanned = false;
    std::vector<LinkUse> uses;
    std::vector<StreamRoom> rooms;
    linkUses(uses);
    keepStreamMembers(uses, rooms);
    bool excluded = false;
    for (auto [k, st] : llvm::enumerate(linkStreams))
      if (!st.excluded && !st.kept) {
        llvm::errs() << "foldsim-fold-static: link stream " << modelOf(st.pcls)
                     << " -> " << modelOf(st.rcls) << " keeps no member at II "
                     << ii << "; scheduled again without its spacing\n";
        laidOutStreams[k].excluded = true;
        excluded = true;
      }
    if (excluded) {
      streamPages();
      return schedule();
    }
  }

  std::string text;
  llvm::raw_string_ostream os(text);
  os << "foldsim-fold-static: II = " << ii << "\n";
  if (moduloTried)
    os << "modulo search: lower bound " << moduloLower << ", " << moduloTried
       << " periods tried (" << moduloTotal << " over the ready-pass search)\n";
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
    if (searchPruned)
      os << "  ready-pass search: " << searchPlaced << " choices placed, "
         << searchPruned << " bounded out\n";
    if (zeroChainHops && zeroLatencyChains)
      os << "  chains cut past " << zeroChainHops << " hops: "
         << cutLinks.size() << " links\n";
  }
  os << sllReport;
  if (!slrOf.empty()) {
    SmallVector<SmallVector<uint64_t>> bits;
    SmallVector<SmallVector<unsigned>> links;
    linkBits(bits, links);
    uint64_t crossBits = 0, allBits = 0, crossWires = 0;
    unsigned crossLinks = 0, allLinks = 0;
    for (unsigned p = 0; p < classes.size(); ++p)
      for (unsigned c = 0; c < classes.size(); ++c) {
        allBits += bits[p][c];
        allLinks += links[p][c];
        if (crosses(p, c)) {
          crossBits += bits[p][c];
          crossLinks += links[p][c];
          crossWires += serialWires(p, bits[p][c]);
        }
      }
    os << "  slr groups: " << slrGroups << " of at most ";
    if (slrCapacitySpecs.empty())
      os << (slrCapacity ? std::to_string(slrCapacity) : std::string("any"));
    else
      for (auto [g, cap] : llvm::enumerate(slrCaps))
        os << (g ? "," : "") << cap;
    os << " sites\n";
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
    if (slrSerialize)
      os << "  slr links serialized: "
         << crossWires << " wires for " << crossBits << " bits\n";
    // Groups that are the region's SLRs in order: the link bits over each
    // boundary, the SLLs they take.
    if (!slrCapacitySpecs.empty() && slrGroups > 1) {
      SmallVector<uint64_t> over(slrGroups - 1, 0);
      for (unsigned p = 0; p < classes.size(); ++p)
        for (unsigned c = 0; c < classes.size(); ++c)
          if (slrOf[p] >= 0 && slrOf[c] >= 0 && slrOf[p] != slrOf[c])
            for (int b = std::min(slrOf[p], slrOf[c]);
                 b < std::max(slrOf[p], slrOf[c]); ++b)
              over[b] += serialWires(p, bits[p][c]);
      os << "  slr boundaries:";
      for (auto [b, x] : llvm::enumerate(over))
        os << " " << b << "-" << b + 1 << " " << x;
      os << (slrSerialize ? " link wires" : " link bits")
         << (sllBudget ? " (budget " + std::to_string(sllBudget) + ")" : std::string()) << "\n";
    }
    for (unsigned p = 0; p < classes.size(); ++p)
      for (unsigned c = 0; c < classes.size(); ++c)
        if (crosses(p, c) && links[p][c])
          os << "    " << classes[p].name << " -> " << classes[c].name << ": "
             << links[p][c] << " links, " << bits[p][c] << " bits"
             << (serialOf(p) > 1 ? ", " + std::to_string(serialOf(p)) +
                                       " host cycles on " +
                                       std::to_string(serialWires(p, bits[p][c])) +
                                       " wires"
                                 : std::string())
             << "\n";
  }
  for (auto &cls : classes) {
    os << "  " << cls.name << " x" << cls.threads()
       << (cls.readyLevels > 1
               ? " (" + std::to_string(cls.readyLevels) + " ready passes)"
           : cls.hasReady ? std::string(" (ready pass)")
                          : std::string())
       << " state gap "
       << cls.stateGap;
    if (cls.stages)
      os << " stages " << cls.stages;
    if (cls.multicycle)
      os << " multicycle " << cls.multicycle;
    if (cls.passSplit)
      os << " pass-split " << cls.passSplit;
    if (cls.page)
      os << " page " << cls.page << (cls.pagesOffChip ? " off chip" : "");
    if (nest)
      os << " depth " << cls.depth
         << (cls.parent >= 0 ? " under " + classes[cls.parent].name : "");
    if (cls.stateless)
      os << " stateless";
    os << "\n";
    for (unsigned l = 0; l < cls.readyLevels; ++l) {
      os << "    ready" << (l ? std::to_string(l + 1) : "") << ":";
      printSlots(os, cls.readySlots(l));
      os << "\n";
    }
    os << "    state:";
    printSlots(os, cls.slot[kState]);
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
         << (stateOnly.contains(cls.producerOf[o]) ? " (moved)" : "")
         << " <-";
      // The first inputs by name, then their count: a model's late output
      // may depend on thousands (the top's AXI outputs at 16 SMs).
      unsigned named = 0, inputs = cls.deps[o].count();
      for (auto input : cls.deps[o].set_bits()) {
        if (named == kNamedDeps && inputs > kNamedDeps + 1) {
          os << " ... (" << inputs - named << " more)";
          break;
        }
        os << " " << cls.inputNames[input];
        ++named;
      }
      os << "\n";
    }
  }
  buildFrameTable(bubbles || !virtualMemoryPattern.empty());
  reportLinks(os);
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

/// A slot table: `rows[index]` (`width` bits; zero past the rows).
static Value table(ImplicitLocOpBuilder &b, Value index, ArrayRef<uint64_t> rows,
                   unsigned width);

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

namespace {
/// Whether two values carry the same bits in every target cycle: the same
/// value, or the same side-effect-free operation of such operands. Two
/// registers are taken as such a pair (coinductively) once their clocks,
/// resets and starting values agree, if their next values are such a pair
/// under that assumption: they then agree forever, as a memory macro's
/// per-lane copies of one read-address pipeline do. A query that fails says
/// nothing; one object answers one query.
struct SameBits {
  static constexpr unsigned kMaxSteps = 4096;
  DenseSet<std::pair<Value, Value>> pairs;
  unsigned steps = 0;

  static bool sameAttrs(Operation *a, Operation *b) {
    auto kept = [](Operation *op) {
      SmallVector<NamedAttribute> attrs;
      for (auto attr : op->getAttrs())
        if (!llvm::is_contained({"name", "inner_sym", "sv.namehint"},
                                attr.getName().getValue()))
          attrs.push_back(attr);
      return attrs;
    };
    return kept(a) == kept(b);
  }

  bool operator()(Value a, Value b) {
    if (a == b)
      return true;
    if (a.getType() != b.getType() || ++steps > kMaxSteps)
      return false;
    auto key = a.getAsOpaquePointer() < b.getAsOpaquePointer()
                   ? std::make_pair(a, b)
                   : std::make_pair(b, a);
    if (pairs.contains(key))
      return true;
    auto ra = dyn_cast<OpResult>(a), rb = dyn_cast<OpResult>(b);
    if (!ra || !rb || ra.getResultNumber() != rb.getResultNumber())
      return false;
    Operation *oa = ra.getOwner(), *ob = rb.getOwner();
    if (oa->getName() != ob->getName() ||
        oa->getNumOperands() != ob->getNumOperands() ||
        oa->getNumRegions() || ob->getNumRegions() || !sameAttrs(oa, ob))
      return false;
    if (isa<seq::FirRegOp>(oa))
      pairs.insert(key);
    else if (!isMemoryEffectFree(oa))
      return false;
    for (auto [x, y] : llvm::zip(oa->getOperands(), ob->getOperands()))
      if (!(*this)(x, y))
        return false;
    pairs.insert(key);
    return true;
  }
};
} // namespace

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

/// A registered read's previous row (--vmem-skip-held-reads): the row the
/// read took in the previous target cycle. Found where the address is a
/// function f of one value M alone (side-effect-free combinational ops and
/// constants), M = mux(c, A, q) or mux(c, q, A), q a register without reset
/// on the read's clock whose next value is M: q is the previous cycle's M, so
/// f(q) is the previous cycle's address. This is a registered read address
/// with an enable, as front ends emit SRAMs and `if (en) raddr_q <= raddr;
/// rdata = mem[raddr_q]` imports (c low: the address holds). The address
/// cone is cloned with q in place of the mux, next to it, so that registers
/// replaced later reach it; it is a register and a few levels, off the path
/// that computes c and A. Null where the pattern is not found.
/// The mux of such a held address, and whether q is its true value; false
/// where the pattern is not found. Changes nothing.
static bool findHeldRead(seq::FirMemReadOp read, comb::MuxOp &found,
                         bool &qOnTrue) {
  found = {};
  qOnTrue = false;
  SmallVector<Value> work{read.getAddress()};
  DenseSet<Value> seen;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    if (seen.size() > 256)
      return false;
    auto *op = value.getDefiningOp();
    if (!op)
      return false;
    if (isa<hw::ConstantOp>(op))
      continue;
    if (auto mux = dyn_cast<comb::MuxOp>(op)) {
      bool matched = false;
      for (bool onTrue : {true, false}) {
        Value q = onTrue ? mux.getTrueValue() : mux.getFalseValue();
        auto reg = q.getDefiningOp<seq::FirRegOp>();
        if (!reg || reg.getReset() || reg.getNext() != value ||
            reg.getClk() != read.getClk())
          continue;
        if (found && found != mux)
          return false;
        found = mux;
        qOnTrue = onTrue;
        matched = true;
        break;
      }
      if (matched)
        continue;
    }
    if (!isa<comb::CombDialect>(op->getDialect()) && !isa<hw::BitcastOp>(op))
      return false;
    if (op->getNumRegions() || !isMemoryEffectFree(op))
      return false;
    for (auto operand : op->getOperands())
      work.push_back(operand);
  }
  return bool(found);
}

static Value heldReadRow(seq::FirMemReadOp read) {
  comb::MuxOp found;
  bool qOnTrue;
  if (!findHeldRead(read, found, qOnTrue))
    return {};
  OpBuilder b(found->getContext());
  b.setInsertionPointAfter(found);
  // The address cone again, q in place of the mux.
  IRMapping mapping;
  mapping.map(found.getResult(),
              qOnTrue ? found.getTrueValue() : found.getFalseValue());
  std::function<Value(Value)> rebuild = [&](Value value) -> Value {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    auto *op = value.getDefiningOp();
    if (isa<hw::ConstantOp>(op))
      return value;
    for (auto operand : op->getOperands())
      rebuild(operand);
    b.clone(*op, mapping);
    return mapping.lookup(value);
  };
  return rebuild(read.getAddress());
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
    // `hbm`: a paged model's page word kept off chip (`pages` only).
    if (kind == "hbm" && store != "pages")
      return netlist.emitError("storage: 'hbm' is for a model's pages, not '")
             << store << "'";
    if (!llvm::is_contained({"registers", "distributed", "block", "ultra", "hbm"},
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

/// Staged evaluation (`stage-levels`): cuts the target logic of a copy (the
/// model's ops spliced into `body`, before its state is replaced) into
/// `cls.stages + 1` segments by stage registers, so that no segment spans
/// more than about `stageLevels` estimated levels. Each op goes to the stage
/// its depth from the state and inputs falls in; an operand from an earlier
/// stage is taken through the stage registers between. The state pass's
/// writes (registers' next values, memories' writes and registered reads, the
/// copy's outputs, `held`'s operands) take the last stage's values.
///
/// The stage registers step with the positions (`live1`, a host cycle before
/// the copy's `now`: whether the position then is one, not idle), so that an
/// evaluation read at a position reaches its state pass `cls.stages`
/// positions on, with bubbles or idle host cycles between as anywhere.
/// A register's next value is split into its write enable and data first
/// (peelHolds) and its hold left at the write, so that the old value need
/// not cross the stages: a store's write is enabled, a kept register holds
/// its own value until its state pass. With `keepOld` (a dense page word,
/// which writes the old value back) the whole next value crosses them.
/// A combinational read at stage k reads the instance the stage holds:
/// `readInst` maps each to `rinst` (the evaluated instance at its read)
/// through k stage registers.
LogicalResult StaticFolder::stageLogic(Class &cls, Block *body,
                                       Operation *held, Value hostClock,
                                       Value live1, Value pre1, Value rinst,
                                       bool keepOld,
                                       DenseMap<Operation *, Value> &readInst) {
  unsigned stages = cls.stages;
  ImplicitLocOpBuilder b(cls.model.getLoc(), body, body->end());

  // Registers: each replaced by one whose next value is set below, from its
  // write enable and data at the last stage.
  struct RegWrite {
    seq::FirRegOp reg;
    Backedge next;
    Value enable, data;
    bool holds = false;
  };
  BackedgeBuilder edges(b, cls.model.getLoc());
  SmallVector<RegWrite> writes;
  SmallVector<seq::FirRegOp> regs;
  for (auto &op : *body)
    if (auto reg = dyn_cast<seq::FirRegOp>(op))
      regs.push_back(reg);
  for (auto reg : regs) {
    b.setLoc(reg.getLoc());
    Value next = reg.getNext();
    if (reg.getReset())
      next = comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), next);
    RegWrite write;
    if (keepOld) {
      write.data = next;
    } else {
      auto peeled = foldsim::peelHolds(b, next, reg.getResult());
      write.holds = peeled.holds;
      write.enable = peeled.enable;
      write.data = peeled.holds ? Value() : peeled.data ? peeled.data : next;
    }
    write.next = edges.get(reg.getType());
    write.reg = seq::FirRegOp::create(b, write.next, reg.getClk(),
                                      reg.getNameAttr());
    reg.getResult().replaceAllUsesWith(write.reg.getResult());
    reg.erase();
    writes.push_back(write);
  }
  // The registers' old next values, now dead; the writes' values held as
  // uses meanwhile (a value of the old next value may have had the register
  // as its only user).
  SmallVector<Value> kept;
  for (auto &write : writes) {
    if (write.enable)
      kept.push_back(write.enable);
    if (write.data)
      kept.push_back(write.data);
  }
  auto *keeper = UnrealizedConversionCastOp::create(b, TypeRange{}, kept)
                     .getOperation();
  for (bool erased = true; erased;) {
    erased = false;
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body)))
      if (&op != held && &op != keeper &&
          (isa<comb::CombDialect>(op.getDialect()) ||
           isa<hw::ConstantOp, hw::AggregateConstantOp, hw::BitcastOp,
               hw::ArrayCreateOp, hw::ArrayGetOp, hw::ArraySliceOp,
               hw::ArrayConcatOp, hw::ArrayInjectOp, hw::StructCreateOp,
               hw::StructExtractOp, hw::StructInjectOp, hw::StructExplodeOp,
               seq::FirMemReadOp>(op)) &&
          isOpTriviallyDead(&op)) {
        op.erase();
        erased = true;
      }
  }
  {
    unsigned k = 0;
    for (auto &write : writes) {
      if (write.enable)
        write.enable = keeper->getOperand(k++);
      if (write.data)
        write.data = keeper->getOperand(k++);
    }
    keeper->erase();
  }

  // Multicycle: no stage registers. The logic's leaves (the state, the
  // inputs, registered reads' data) are taken into capture registers at the
  // read position (`live1`: cap, a host cycle ahead), the writes' values
  // (registers' enables and data, memory ports, outputs) into end registers
  // at the position before the state pass (`pre1`), which writes from them.
  // Between the two the logic has the multicycle span, and the schedule
  // takes no other read of the copy in it (its state passes that far apart).
  if (cls.multicycle) {
    auto isConstant = [](Value value) {
      auto *op = value.getDefiningOp();
      return op && isa<hw::ConstantOp, hw::AggregateConstantOp>(op);
    };
    DenseSet<Operation *> logic;
    SmallVector<Operation *> ops;
    for (auto &op : *body) {
      if (&op == held || op.getNumResults() == 0 ||
          isa<seq::FirRegOp, seq::CompRegClockEnabledOp, seq::FirMemOp,
              seq::FirMemReadOp, hw::ConstantOp, hw::AggregateConstantOp,
              UnrealizedConversionCastOp, hw::InstanceOp>(op))
        continue;
      logic.insert(&op);
      ops.push_back(&op);
    }
    std::string prefix = "fsmc_" + cls.name + "__";
    Value one = hw::ConstantOp::create(b, b.getI1Type(), 1);
    // The enables, a copy per 512 registers.
    struct Enable {
      Value ahead, now;
      unsigned uses = 512, made = 0;
      const char *name;
    } capEn{live1, Value(), 512, 0, "cap0_"}, endEn{pre1, Value(), 512, 0, "pre0_"};
    auto enableOf = [&](Enable &en) {
      if (en.uses == 512) {
        en.now = seq::CompRegClockEnabledOp::create(
            b, b.getI1Type(), en.ahead, hostClock, one,
            b.getStringAttr(Twine(en.name) + Twine(en.made++)), Value(), Value(),
            Value(), hw::InnerSymAttr{});
        en.uses = 0;
      }
      ++en.uses;
      return en.now;
    };
    unsigned caps = 0, ends = 0;
    uint64_t capBits = 0, endBits = 0;
    // DONT_TOUCH: the timing constraints name them, so synthesis must not
    // retime, merge or absorb them (into a DSP's or ultra RAM's registers).
    auto dontTouch = sv::SVAttributeAttr::get(b.getContext(), "DONT_TOUCH",
                                              R"("true")",
                                              /*emitAsComment=*/false);
    auto take = [&](Value value, Enable &en, const Twine &name) -> Value {
      Value bits = asBits(b, value);
      auto regOp = seq::CompRegClockEnabledOp::create(
          b, bits.getType(), bits, hostClock, enableOf(en), b.getStringAttr(name),
          Value(), Value(), Value(), hw::InnerSymAttr{});
      sv::addSVAttributes(regOp, {dontTouch});
      Value reg = regOp;
      if (reg.getType() != value.getType())
        reg = b.createOrFold<hw::BitcastOp>(value.getType(), reg);
      return reg;
    };
    DenseMap<Value, Value> captured, ended;
    auto capture = [&](Value value) -> Value {
      if (isConstant(value) || logic.contains(value.getDefiningOp()))
        return value;
      Value &known = captured[value];
      if (!known) {
        known = take(value, capEn, prefix + "cap" + Twine(caps++));
        capBits += std::max<int64_t>(1, hw::getBitWidth(value.getType()));
      }
      return known;
    };
    auto ending = [&](Value value) -> Value {
      if (!value || isConstant(value))
        return value;
      Value &known = ended[value];
      if (!known) {
        known = take(capture(value), endEn, prefix + "end" + Twine(ends++));
        endBits += std::max<int64_t>(1, hw::getBitWidth(value.getType()));
      }
      return known;
    };
    // The logic reads the captures.
    for (auto *op : ops)
      for (auto &operand : op->getOpOperands())
        if (statePort(operand.get()))
          if (Value at = capture(operand.get()); at != operand.get())
            operand.set(at);
    // The writes, from the end registers.
    for (auto &write : writes) {
      b.setLoc(write.reg.getLoc());
      if (write.holds) {
        write.next.setValue(write.reg.getResult());
        continue;
      }
      Value data = ending(write.data);
      write.next.setValue(
          write.enable ? Value(comb::MuxOp::create(b, ending(write.enable), data,
                                                   write.reg.getResult()))
                       : data);
    }
    for (auto &op : *body) {
      bool sink = &op == held || isa<seq::FirMemWriteOp>(op);
      if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
        auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
        sink = memory && memory.getReadLatency() > 0;
      }
      if (!sink)
        continue;
      for (auto &operand : op.getOpOperands())
        if (statePort(operand.get()))
          if (Value at = ending(operand.get()); at != operand.get())
            operand.set(at);
    }
    // The schedule's side: the copy's state passes, modulo the frame, at
    // least the span apart (no capture before the end registers are taken),
    // and the reads the span and a position before them.
    SmallVector<int> passes;
    for (int s : cls.slot[kState])
      passes.push_back(s % int(ii));
    llvm::sort(passes);
    for (unsigned k = 0; k < passes.size(); ++k) {
      int gap = k + 1 < passes.size() ? passes[k + 1] - passes[k]
                                      : passes[0] + int(ii) - passes[k];
      if (passes.size() > 1 && gap < int(cls.multicycle))
        return cls.model.emitError("multicycle: ")
               << cls.name << "'s state passes " << gap
               << " positions apart, under its span " << cls.multicycle;
    }
    if (int(ii) < int(cls.multicycle) * int(passes.size()))
      return cls.model.emitError("multicycle: a frame of ")
             << ii << " under " << cls.name << "'s evaluations";
    // In simulation, the timing the constraints assume, checked on every
    // run: no capture in an open evaluation (but at its end registers'
    // host cycle), and from a capture to its end registers at least the span
    // in host cycles.
    {
      auto i1 = b.getI1Type();
      unsigned width = llvm::Log2_32_Ceil(cls.multicycle + 2);
      auto cntType = b.getIntegerType(width);
      auto top = cls.model->getParentOfType<ModuleOp>();
      if (!top.lookupSymbol<sv::MacroDeclOp>("SYNTHESIS")) {
        OpBuilder tb(top.getBody(), top.getBody()->begin());
        sv::MacroDeclOp::create(tb, cls.model.getLoc(), "SYNTHESIS");
      }
      Value capNow = seq::CompRegClockEnabledOp::create(
          b, i1, live1, hostClock, one, b.getStringAttr("mc_check_cap"), Value(),
          Value(), Value(), hw::InnerSymAttr{});
      Value preNow = seq::CompRegClockEnabledOp::create(
          b, i1, pre1, hostClock, one, b.getStringAttr("mc_check_pre"), Value(),
          Value(), Value(), hw::InnerSymAttr{});
      BackedgeBuilder checks(b, cls.model.getLoc());
      auto seenNext = checks.get(i1), pendingNext = checks.get(i1),
           countNext = checks.get(cntType);
      Value seen = seq::CompRegClockEnabledOp::create(
          b, i1, seenNext, hostClock, one, b.getStringAttr("mc_check_seen"),
          Value(), Value(), Value(), hw::InnerSymAttr{});
      Value pending = seq::CompRegClockEnabledOp::create(
          b, i1, pendingNext, hostClock, one, b.getStringAttr("mc_check_pending"),
          Value(), Value(), Value(), hw::InnerSymAttr{});
      Value count = seq::CompRegClockEnabledOp::create(
          b, cntType, countNext, hostClock, one, b.getStringAttr("mc_check_count"),
          Value(), Value(), Value(), hw::InnerSymAttr{});
      Value cntMax = hw::ConstantOp::create(b, cntType, cls.multicycle);
      Value notPre = comb::XorOp::create(b, preNow, one);
      seenNext.setValue(comb::OrOp::create(b, seen, capNow));
      pendingNext.setValue(comb::OrOp::create(
          b, capNow, comb::AndOp::create(b, pending, notPre)));
      Value full = comb::ICmpOp::create(b, comb::ICmpPredicate::uge, count, cntMax);
      Value inc = comb::AddOp::create(b, count, hw::ConstantOp::create(b, cntType, 1));
      countNext.setValue(comb::MuxOp::create(
          b, capNow, hw::ConstantOp::create(b, cntType, 0),
          comb::MuxOp::create(b, full, count, inc)));
      Value early = comb::AndOp::create(
          b, capNow, comb::AndOp::create(b, pending, notPre));
      Value short_ = comb::ICmpOp::create(
          b, comb::ICmpPredicate::ult, count,
          hw::ConstantOp::create(b, cntType, cls.multicycle - 1));
      Value orphan = comb::XorOp::create(b, pending, one);
      Value late = comb::AndOp::create(
          b, comb::AndOp::create(b, preNow, seen),
          comb::OrOp::create(b, orphan, short_));
      Value clock = seq::FromClockOp::create(b, hostClock);
      std::string what = ("foldsim: multicycle " + cls.name +
                          ": an evaluation's capture and end registers under " +
                          Twine(cls.multicycle) + " host cycles apart, or "
                          "another capture between")
                             .str();
      // Off in synthesis, unless FOLDSIM_CHECKS is defined (a gate that
      // simulates the synthesis view, -DSYNTHESIS, keeps it).
      if (!top.lookupSymbol<sv::MacroDeclOp>("FOLDSIM_CHECKS")) {
        OpBuilder tb(top.getBody(), top.getBody()->begin());
        sv::MacroDeclOp::create(tb, cls.model.getLoc(), "FOLDSIM_CHECKS");
      }
      Value violation = comb::OrOp::create(b, early, late);
      auto monitor = [&] {
        sv::AlwaysOp::create(b, ArrayRef{sv::EventControl::AtPosEdge},
                             ArrayRef{clock}, [&] {
          sv::IfOp::create(b, violation, [&] {
            sv::FatalProceduralOp::create(b, 1, b.getStringAttr(what));
          });
        });
      };
      sv::IfDefOp::create(b, "FOLDSIM_CHECKS", monitor, [&] {
        sv::IfDefOp::create(b, "SYNTHESIS", [] {}, monitor);
      });
    }
    cls.multicycleCaps = capBits;
    cls.multicycleEnds = endBits;
    llvm::errs() << "foldsim-fold-static: " << cls.name << " multicycle "
                 << cls.multicycle << ": " << capBits << " capture bits ("
                 << caps << " registers), " << endBits << " end bits (" << ends
                 << " registers)\n";
    return success();
  }

  // Each value's stage, soonest (stageOf). The writes are at the last stage;
  // where the cone here (its enables and data split from the registers' next
  // values) needs more stages than the schedule gave, the levels of a stage
  // grow until it fits.
  SmallVector<Value> sinks;
  for (auto &write : writes) {
    if (write.enable)
      sinks.push_back(write.enable);
    if (write.data)
      sinks.push_back(write.data);
  }
  for (auto &op : *body) {
    auto read = dyn_cast<seq::FirMemReadOp>(op);
    auto memory = read ? read.getMemory().getDefiningOp<seq::FirMemOp>()
                       : seq::FirMemOp();
    if (&op == held || (read && memory && memory.getReadLatency() > 0) ||
        isa<seq::FirMemWriteOp>(op))
      for (auto operand : op.getOperands())
        if (statePort(operand))
          sinks.push_back(operand);
  }
  DenseMap<Value, Staging> memo;
  int limit = std::max<int>(2, 2 * stageLevels);
  for (;; ++limit) {
    memo.clear();
    int last = 0;
    for (auto sink : sinks)
      last = std::max(last, sinkStage(sink, 0, limit, memo));
    if (last <= int(stages))
      break;
  }
  for (auto &op : *body)
    for (auto result : op.getResults())
      if (!memo.count(result))
        stageOf(result, limit, memo);

  // The logic's ops, and the same cut latest first (each op in the latest
  // stage its readers allow, the writes at the last): of the two, the one
  // whose values cross fewer stage register bits.
  auto isNode = [&](Operation *op) {
    if (!op || op->getBlock() != body || op == held || op->getNumResults() == 0 ||
        isa<seq::FirRegOp, seq::CompRegClockEnabledOp, seq::FirMemOp, hw::ConstantOp,
            hw::AggregateConstantOp, UnrealizedConversionCastOp>(op))
      return false;
    if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
      auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
      return !(memory && memory.getReadLatency() > 0);
    }
    return true;
  };
  DenseSet<Value> sinkSet(sinks.begin(), sinks.end());
  // Readers before their operands' producers: a depth-first post order on
  // the operands, reversed.
  SmallVector<Operation *> post;
  {
    DenseSet<Operation *> seen;
    for (auto &root : *body) {
      if (!isNode(&root) || seen.contains(&root))
        continue;
      SmallVector<std::pair<Operation *, unsigned>> stack{{&root, 0}};
      seen.insert(&root);
      while (!stack.empty()) {
        auto &[op, next] = stack.back();
        if (next < op->getNumOperands()) {
          auto *def = op->getOperand(next++).getDefiningOp();
          if (isNode(def) && seen.insert(def).second)
            stack.push_back({def, 0});
          continue;
        }
        post.push_back(op);
        stack.pop_back();
      }
    }
  }
  DenseMap<Operation *, Staging> late;
  int lateLast = 0;
  for (auto *op : llvm::reverse(post)) {
    int levels = 0;
    if (auto read = dyn_cast<seq::FirMemReadOp>(op))
      levels = 2;
    else
      levels = stageOpLevels(op);
    int stage = 0, at = 0;
    bool used = false;
    for (auto result : op->getResults()) {
      if (sinkSet.contains(result))
        used = true;
      for (auto *user : result.getUsers())
        if (auto it = late.find(user); it != late.end()) {
          used = true;
          stage = std::max(stage, it->second.stage);
        }
    }
    for (auto result : op->getResults())
      for (auto *user : result.getUsers())
        if (auto it = late.find(user); it != late.end() && it->second.stage == stage)
          at = std::max(at, it->second.levels);
    (void)used;
    if (at > 0 && at + levels > limit) {
      ++stage;
      at = 0;
    }
    late[op] = {stage, at + levels};
    lateLast = std::max(lateLast, stage);
  }
  // Stage register bits: each value from its stage to its latest reader's
  // (a write's at the last).
  auto crossing = [&](function_ref<int(Value)> at) {
    uint64_t bits = 0;
    DenseSet<Value> counted;
    auto count = [&](Value value, int reader) {
      int from = at(value);
      if (from < 0 || reader <= from)
        return;
      (void)counted;
      bits += uint64_t(std::max<int64_t>(1, hw::getBitWidth(value.getType()))) *
              uint64_t(reader - from);
    };
    // The latest reader of each value.
    DenseMap<Value, int> latest;
    for (auto *op : post)
      for (auto operand : op->getOperands())
        if (statePort(operand))
          latest[operand] = std::max(latest.lookup(operand), at(op->getResult(0)));
    for (auto sink : sinks)
      latest[sink] = int(stages);
    for (auto [value, reader] : latest)
      count(value, reader);
    return bits;
  };
  auto constantAt = [](Value value) {
    auto *op = value.getDefiningOp();
    return op && isa<hw::ConstantOp, hw::AggregateConstantOp>(op);
  };
  auto soonest = [&](Value value) -> int {
    if (constantAt(value))
      return -1;
    return std::min<int>(memo.lookup(value).stage, stages);
  };
  auto latest = [&](Value value) -> int {
    if (constantAt(value))
      return -1;
    auto it = late.find(value.getDefiningOp());
    if (it == late.end())
      return 0;  // the state, the inputs, registered reads
    return int(stages) - it->second.stage;
  };
  bool useLate = false;
  if (lateLast <= int(stages)) {
    uint64_t early = crossing(soonest), lateBits = crossing(latest);
    useLate = lateBits < early;
    // (FOLDSIM_STAGE_CUT=soonest|latest forces one, for the tests.)
    if (const char *force = getenv("FOLDSIM_STAGE_CUT"))
      useLate = StringRef(force) == "latest";
    llvm::errs() << "foldsim-fold-static: " << cls.name << " staged: " << stages
                 << " stages, soonest " << early << " stage register bits, latest "
                 << lateBits << (useLate ? " (latest)" : " (soonest)") << "\n";
  }
  auto stageOf = [&](Value value) -> int {
    return useLate ? latest(value) : soonest(value);
  };

  // A value at a later stage: through the stage registers, as bits. The
  // registers' enable a copy per 512 of them.
  DenseMap<std::pair<Value, unsigned>, Value> delays;
  Value one = hw::ConstantOp::create(b, b.getI1Type(), 1);
  Value live;
  unsigned onLive = 0, made = 0;
  std::function<Value(Value, unsigned)> delayed = [&](Value value,
                                                      unsigned k) -> Value {
    int from = stageOf(value);
    if (from < 0 || from >= int(k))
      return value;
    if (Value known = delays.lookup({value, k}))
      return known;
    Value before = delayed(value, k - 1);
    if (!live || onLive == 512) {
      live = seq::CompRegClockEnabledOp::create(
          b, live1.getType(), live1, hostClock, one,
          b.getStringAttr("stage_live" + Twine(made / 512)), Value(), Value(),
          Value(), hw::InnerSymAttr{});
      onLive = 0;
    }
    ++onLive;
    ++made;
    Value bits = asBits(b, before);
    Value staged = seq::CompRegClockEnabledOp::create(
        b, bits.getType(), bits, hostClock, live,
        b.getStringAttr("stage" + Twine(k)), Value(), Value(), Value(),
        hw::InnerSymAttr{});
    if (staged.getType() != value.getType())
      staged = b.createOrFold<hw::BitcastOp>(value.getType(), staged);
    delays[{value, k}] = staged;
    return staged;
  };

  // The logic: each op's operands from its stage.
  SmallVector<Operation *> nodes;
  for (auto &op : *body) {
    // (The copy's own control registers, made before the cut, are not the
    // target's logic.)
    if (&op == held || op.getNumResults() == 0 ||
        isa<seq::FirRegOp, seq::CompRegClockEnabledOp, seq::FirMemOp,
            hw::ConstantOp, hw::AggregateConstantOp>(op))
      continue;
    if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
      auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
      if (memory && memory.getReadLatency() > 0)
        continue;
    }
    // The registers' next values, placeholders until set below.
    if (isa<UnrealizedConversionCastOp>(op))
      continue;
    nodes.push_back(&op);
  }
  for (auto *op : nodes) {
    unsigned k = std::max(0, stageOf(op->getResult(0)));
    if (isa<seq::FirMemReadOp>(op) && rinst)
      readInst[op] = delayed(rinst, k);
    for (auto &operand : op->getOpOperands())
      if (statePort(operand.get()))
        if (Value at = delayed(operand.get(), k); at != operand.get())
          operand.set(at);
  }
  // The writes: at the last stage.
  for (auto &write : writes) {
    b.setLoc(write.reg.getLoc());
    if (write.holds) {
      write.next.setValue(write.reg.getResult());
      continue;
    }
    Value data = delayed(write.data, stages);
    write.next.setValue(
        write.enable ? Value(comb::MuxOp::create(b, delayed(write.enable, stages),
                                                 data, write.reg.getResult()))
                     : data);
  }
  for (auto &op : *body) {
    bool sink = &op == held || isa<seq::FirMemWriteOp>(op);
    if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
      auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
      sink = memory && memory.getReadLatency() > 0;
    }
    if (!sink)
      continue;
    for (auto &operand : op.getOpOperands())
      if (statePort(operand.get()))
        if (Value at = delayed(operand.get(), stages); at != operand.get())
          operand.set(at);
  }
  return success();
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
  bool ring = !cls.stages && !cls.page && !cls.deferSpan && threads > 1 &&
              threads <= registerStateEntries &&
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
      if (cls.clockInput < 0 ||
          reg.getClk() != modelBody->getArgument(cls.clockInput))
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
  // The regex is searched in a memory's name and in its name qualified by
  // its class's model, `<model>:<name>` (a copy's by the class it was split
  // from), so that one class's memories can be named apart from another's
  // with the same path, as the storage rules name them.
  DenseMap<Operation *, unsigned> virtualOf;
  SmallVector<seq::FirMemOp> candidates;
  if (!virtualMemoryPattern.empty()) {
    llvm::Regex pattern(virtualMemoryPattern);
    StringRef model =
        cls.splitFrom >= 0 ? StringRef(classes[cls.splitFrom].name) : cls.name;
    for (auto &op : *modelBody) {
      auto memory = dyn_cast<seq::FirMemOp>(op);
      if (!memory || physical(&op))
        continue;
      StringRef memName = memory.getName().value_or("");
      if (!pattern.match(memName) &&
          !pattern.match((model + ":" + memName).str()))
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
      // The pending bit its state pass sets must be there for the decision
      // on the instance's next evaluation.
      int gap = decisionLead() + 1;
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
      candidates.push_back(memory);
    }
  }
  // Each memory alone, or (--vmem-merge-lanes) with the later ones split
  // from the same target RAM: the same depth, read-during-write order and
  // clock, a read address and enable that carry the same bits in every
  // cycle, and (all or none written) the same write address. A lane takes
  // whole bytes of the row, as many as its bits need, and its write enable
  // (with its mask) enables them; the row stays within the controller's 4096
  // bits.
  auto portsOf = [](seq::FirMemOp memory) {
    std::pair<seq::FirMemReadOp, seq::FirMemWriteOp> ports;
    for (auto *user : memory->getUsers()) {
      if (auto read = dyn_cast<seq::FirMemReadOp>(user))
        ports.first = read;
      else
        ports.second = cast<seq::FirMemWriteOp>(user);
    }
    return ports;
  };
  auto sameRam = [&](seq::FirMemOp a, seq::FirMemOp c) {
    if (a.getType().getDepth() != c.getType().getDepth() ||
        a.getRuw() != c.getRuw())
      return false;
    auto [ra, wa] = portsOf(a);
    auto [rc, wc] = portsOf(c);
    if (ra.getClk() != rc.getClk() || !SameBits()(ra.getAddress(), rc.getAddress()) ||
        bool(ra.getEnable()) != bool(rc.getEnable()) ||
        (ra.getEnable() && !SameBits()(ra.getEnable(), rc.getEnable())))
      return false;
    if (bool(wa) != bool(wc))
      return false;
    return !wa || (wa.getClk() == wc.getClk() &&
                   SameBits()(wa.getAddress(), wc.getAddress()));
  };
  auto bytesOf = [](seq::FirMemOp memory) {
    return (memory.getType().getWidth() + 7) / 8;
  };
  DenseSet<Operation *> merged;
  for (auto [i, memory] : llvm::enumerate(candidates)) {
    if (merged.contains(memory))
      continue;
    auto type = memory.getType();
    unsigned maskWidth = type.getMaskWidth().value_or(1);
    Class::VirtualMemory vm;
    vm.name = memory.getName().value_or("mem").str();
    vm.rowWidth = std::max(1u, llvm::Log2_64_Ceil(type.getDepth()));
    vm.dataWidth = type.getWidth();
    vm.maskWidth = maskWidth > 1 ? type.getWidth() / 8 : 1;
    vm.readFirst = memory.getRuw() != seq::RUW::New;
    vm.lanes.push_back(memory);
    vm.laneOffset.push_back(0);
    if (mergeLanes) {
      unsigned bytes = bytesOf(memory);
      for (auto other : ArrayRef(candidates).drop_front(i + 1)) {
        if (merged.contains(other) || (bytes + bytesOf(other)) * 8 > 4096 ||
            !sameRam(memory, other))
          continue;
        vm.lanes.push_back(other);
        vm.laneOffset.push_back(bytes * 8);
        bytes += bytesOf(other);
        merged.insert(other);
      }
      if (vm.lanes.size() > 1) {
        vm.name += "_x" + std::to_string(vm.lanes.size());
        vm.dataWidth = bytes * 8;
        vm.maskWidth = bytes;
      }
    }
    // The lanes share the first's read address (sameRam), whose row the
    // requests carry.
    // The skip's request leaves a host cycle later: the pending bit it sets
    // must still be there for the decision on the instance's next evaluation.
    if (skipHeldReads && cls.stateGap >= decisionLead() + 2) {
      vm.heldRow = heldReadRow(portsOf(memory).first);
      vm.delayed = bool(vm.heldRow);
    }
    for (auto lane : vm.lanes)
      virtualOf[lane] = cls.virtualMemories.size();
    cls.virtualMemories.push_back(std::move(vm));
  }

  // Paged register state: the registers a ready pass reads (the cone of an
  // output written at a ready pass, through the logic and combinational
  // reads, up to registers, registered reads and black boxes) keep their
  // stores, read at any slot; the rest page, read only at the state pass.
  auto *modelOutput = modelBody->getTerminator();
  DenseSet<Operation *> readyRegs;
  std::pair<unsigned, uint64_t> rowMemories{0, 0};
  if (cls.page) {
    readyRegisters(cls, readyRegs);
    if ((pageMemoryBits || !pageRows.empty()) && threads > 1 && !cls.stages)
      rowMemories = memoriesAsRows(cls, readyRegs, virtualOf);
  }

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
  // Off chip, each position's page read (instance, page) and page write
  // (page) instead.
  unsigned pageBits = 0;
  if (cls.page)
    pageBits = std::max(1u, llvm::Log2_32_Ceil(unsigned(llvm::PowerOf2Ceil(cls.page))));
  // With a window, the fetch from it too, by slot.
  unsigned windowWidth =
      cls.pageWindow > 1 ? llvm::Log2_32_Ceil(cls.pageWindow) : 1;
  if (cls.page && !cls.pagesOffChip) {
    addPort("fetch3", IntegerType::get(context, 1), true);
    addPort("finst3", IntegerType::get(context, idxWidth), true);
  } else if (cls.page && cls.pageWindow) {
    addPort("fetch3", IntegerType::get(context, 1), true);
    addPort("finst3", IntegerType::get(context, windowWidth), true);
  }
  if (cls.page && cls.pagesOffChip) {
    addPort("pread3", IntegerType::get(context, 1), true);
    addPort("pinst3", IntegerType::get(context, idxWidth), true);
    addPort("ppage3", IntegerType::get(context, pageBits), true);
    addPort("pwrite3", IntegerType::get(context, 1), true);
    addPort("pwpage3", IntegerType::get(context, pageBits), true);
  }
  // Staged: the instance whose evaluation reads at the position three host
  // cycles ahead (its state pass the stages on), and whether that position
  // is one (not idle): the stages step with the positions.
  // Multicycle: whether the position three host cycles ahead is a read
  // (`cap3`, in place of `live3`) or the one before a state pass (`pre3`).
  unsigned rinstArg = 0, liveArg = 0, preArg = 0;
  if (cls.stages) {
    if (threads > 1) {
      rinstArg = numIn;
      addPort("rinst3", IntegerType::get(context, idxWidth), true);
    }
    liveArg = numIn;
    addPort(cls.multicycle ? "cap3" : "live3", IntegerType::get(context, 1), true);
    if (cls.multicycle) {
      preArg = numIn;
      addPort("pre3", IntegerType::get(context, 1), true);
    }
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
  bool offChip = cls.page && cls.pagesOffChip;
  bool window = offChip && cls.pageWindow;
  bool fetches = cls.page && (!offChip || window);
  Value fetch3 = fetches ? body->getArgument(3) : Value();
  Value finst3 = fetches ? body->getArgument(4) : Value();
  // Off chip: [read, instance, page, write, write page], three ahead.
  SmallVector<Value> offChip3;
  if (offChip)
    for (unsigned k = 0; k < 5; ++k)
      offChip3.push_back(body->getArgument((fetches ? 5 : 3) + k));
  unsigned pages = cls.page ? unsigned(llvm::PowerOf2Ceil(cls.page)) : 0;
  unsigned pageWidth = pages > 1 ? llvm::Log2_32_Ceil(pages) : 0;
  // The copy's outputs are held as the operands of placeholder ops until its
  // hw.output is built: registers, memory reads and paged registers are
  // replaced and erased below, and only a use follows a replacement (a
  // Value kept aside, or a map keyed by it, outlives the op it names).
  auto hold = [&](ValueRange values) -> Operation * {
    auto builder = OpBuilder::atBlockEnd(body);
    return UnrealizedConversionCastOp::create(builder, model.getLoc(),
                                              TypeRange{}, values);
  };
  auto release = [](Operation *held, SmallVectorImpl<Value> &into) {
    into.append(held->operand_begin(), held->operand_end());
    held->erase();
  };
  Operation *heldResults = hold(results);
  for (auto [i, arg] : llvm::enumerate(modelBody->getArguments())) {
    Value to = int(i) == cls.clockInput ? Value(hostClock)
                                        : Value(body->getArgument(inputArg[i]));
    arg.replaceAllUsesWith(to);
  }
  ImplicitLocOpBuilder b(model.getLoc(), body, body->end());
  auto i1 = b.getI1Type();
  auto one = [&]() { return hw::ConstantOp::create(b, i1, 1); };

  // Pass splitting (deferred lanes): the logic only some registers' next
  // values use, its leaves captured at the state pass, its lanes over the
  // passes after it; those registers written deferSpan host cycles after
  // their state pass (at the instance captured there), before the next
  // evaluation reads them (the state gap) and the next capture (the pass
  // spacing). Nothing else reads that logic, so the state pass is unchanged.
  DenseMap<Operation *, Value> deferredNext;
  // Its controls: registers of the copy, not target state.
  DenseSet<Operation *> deferControls;
  Value deferWen, deferInst;
  if (cls.deferSpan) {
    auto chain = [&](Value v, unsigned n, const Twine &name, bool preset) {
      for (unsigned k = 0; k < n; ++k) {
        auto reg = preset ? seq::FirRegOp::create(
                                b, v, hostClock, b.getStringAttr(name + Twine(k)),
                                hw::InnerSymAttr{}, b.getIntegerAttr(i1, 0))
                          : seq::FirRegOp::create(
                                b, v, hostClock, b.getStringAttr(name + Twine(k)));
        deferControls.insert(reg);
        v = reg;
      }
      return v;
    };
    std::string prefix = "fsps_" + cls.name + "__";
    SmallVector<Value> pinned;
    if (folding)
      for (auto *op : folding->physical)
        if (auto reg = dyn_cast<seq::FirRegOp>(op))
          pinned.push_back(reg.getNext());
    auto plan = foldsim::planDeferred(body, pinned, cls.passSplit);
    // The state pass, as the regions see it (three host cycles on).
    Value stateNow = chain(state3, 3, prefix + "state", true);
    auto st = foldsim::applyDeferred(b, plan, hostClock, stateNow,
                                     cls.passSplit, cls.passSpan, prefix,
                                     deferredNext);
    cls.passHoldBits = st.holdBits;
    deferWen = chain(stateNow, cls.deferSpan, prefix + "wen", true);
    if (threads > 1)
      deferInst = seq::CompRegClockEnabledOp::create(
          b, inst3.getType(), chain(inst3, 3, prefix + "inst", false),
          hostClock, stateNow, b.getStringAttr(prefix + "winst"), Value(),
          Value(), Value(), hw::InnerSymAttr{});
    uint64_t deferredBits = 0;
    for (auto *op : plan.regs)
      deferredBits += hw::getBitWidth(op->getResult(0).getType());
    llvm::errs() << "foldsim-fold-static: " << cls.name << " pass-split "
                 << cls.passSplit << ": " << plan.regs.size()
                 << " deferred registers (" << deferredBits << " bits), " << st.groups << " lane groups, "
                 << st.replicas << " replicas on " << st.lanes << " lanes, "
                 << llvm::format("%.0f", st.before) << " -> "
                 << llvm::format("%.0f", st.after) << " LUTs estimated, "
                 << st.captureBits << " capture bits, " << st.holdBits
                 << " hold bits, " << st.selectBits << " chosen leaf bits (groups";
    for (unsigned n : st.sizes)
      llvm::errs() << " x" << n;
    llvm::errs() << ")\n";
  }

  // Staged evaluation: the target logic cut into stages; its reads (the
  // state's read registers, the inputs, combinational reads) at the read
  // side's instance, `rinst3` on, its writes at the state pass's.
  Value readInst3 = inst3;
  DenseMap<Operation *, Value> readInst;
  if (cls.stages) {
    auto chain = [&](Value value, unsigned regs, const Twine &name) {
      for (unsigned k = 0; k < regs; ++k)
        value = seq::CompRegClockEnabledOp::create(
            b, value.getType(), value, hostClock, one(),
            b.getStringAttr(name + Twine(regs - 1 - k)), Value(), Value(),
            Value(), hw::InnerSymAttr{});
      return value;
    };
    if (threads > 1)
      readInst3 = body->getArgument(rinstArg);
    Value live1 = chain(body->getArgument(liveArg), 2,
                        cls.multicycle ? "cap" : "live");
    Value pre1 = cls.multicycle ? chain(body->getArgument(preArg), 2, "pre")
                                : Value();
    Value rinst = threads > 1 ? chain(readInst3, 3, "rinst") : Value();
    if (failed(stageLogic(cls, body, heldResults, hostClock, live1, pre1, rinst,
                          /*keepOld=*/cls.page && !pageMasks, readInst)))
      return failure();
  }

  // Regions of state, in name order so that a region is a submodule's: each
  // its own copy of the slot's controls, from the copy's, so that no control
  // drives more than about `regionBits` bits of state; a memory larger than
  // that has a region of its own.
  // The controls two host cycles ahead, a copy for each group of
  // `kRegionsPerGroup` regions, so that the ports drive only the groups.
  struct Ahead {
    Value state2, inst2, winst2;
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
      if (threads > 1) {
        ahead.inst2 = keep(seq::FirRegOp::create(
            b, readInst3, hostClock, b.getStringAttr(prefix + "_inst2")));
        // Staged: the state pass's instance apart from the read's.
        ahead.winst2 = readInst3 == inst3
                           ? ahead.inst2
                           : keep(seq::FirRegOp::create(
                                 b, inst3, hostClock,
                                 b.getStringAttr(prefix + "_winst2")));
      }
      if (fetches) {
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
            b,
            ahead.winst2 == ahead.inst2
                ? region.inst1
                : Value(keep(seq::FirRegOp::create(
                      b, ahead.winst2, hostClock,
                      b.getStringAttr(prefix + "_winst1")))),
            hostClock, b.getStringAttr(prefix + "_inst")));
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
  SmallVector<Operation *> virtualPorts(cls.virtualMemories.size(), nullptr);
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
    // Each instance's read data, from the responses.
    auto readBuffer = [&]() -> Value {
      auto readName = vm.name + "_rdata";
      Value data;
      if (threads == 1) {
        auto nextEdge = edges.get(rspData.getType());
        auto buffer = seq::FirRegOp::create(b, nextEdge, hostClock,
                                            b.getStringAttr(readName));
        nextEdge.setValue(comb::MuxOp::create(b, rspValid, rspData, buffer));
        data = buffer;
      } else {
        // Read a host cycle early, at the instance two host cycles ahead,
        // into the RAM's own output register, so that a wide or deep buffer
        // may be block RAM (Vivado's choice by size) and not only
        // distributed RAM. Exact: no response for the instance lands between
        // that read and its pass, since a pass is decided only once the
        // instance has no read pending (virtualReady), a decision lead
        // (>= 3 host cycles) before it.
        auto bufferType =
            seq::FirMemType::get(context, entries, vm.dataWidth, 1);
        auto buffer = seq::FirMemOp::create(
            b, bufferType, /*readLatency=*/1, /*writeLatency=*/1,
            seq::RUW::Old, seq::WUW::Undefined,
            b.getStringAttr(readName), hw::InnerSymAttr{},
            seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
        keepShallowInFlops(buffer);
        seq::FirMemWriteOp::create(b, buffer, rspInst, hostClock, rspValid,
                                   rspData, Value{});
        Value bufferData = seq::FirMemReadOp::create(b, buffer, region.inst2,
                                                     hostClock, one());
        data = seq::FirRegOp::create(b, bufferData, hostClock,
                                     b.getStringAttr(readName + "_read"));
      }
      return data;
    };
    if (vm.lanes.size() > 1) {
      // The lanes of one target RAM: one read and one write of the whole row,
      // each lane's bytes enabled by its own write enable.
      SmallVector<Value> datas, masks;
      Value anyWrite;
      for (auto lane : vm.lanes) {
        seq::FirMemReadOp read;
        seq::FirMemWriteOp write;
        for (auto *user : lane->getUsers()) {
          if (auto r = dyn_cast<seq::FirMemReadOp>(user))
            read = r;
          else
            write = cast<seq::FirMemWriteOp>(user);
        }
        unsigned width = lane.getType().getWidth();
        unsigned bytes = (width + 7) / 8;
        if (lane == vm.lanes.front()) {
          rdValid = region.state;
          if (read.getEnable())
            rdValid = comb::AndOp::create(b, rdValid, read.getEnable());
          rdRow = widen(read.getAddress(), vm.rowWidth);
          if (write)
            wrRow = widen(write.getAddress(), vm.rowWidth);
        }
        if (write) {
          Value enable = write.getEnable() ? Value(write.getEnable()) : Value(one());
          anyWrite = anyWrite ? Value(comb::OrOp::create(b, anyWrite, enable))
                              : enable;
          datas.push_back(widen(asBits(b, write.getData()), bytes * 8));
          unsigned maskBits = lane.getType().getMaskWidth().value_or(1);
          SmallVector<Value> laneMask;
          for (unsigned i = maskBits; i-- > 0;) {
            Value bit = enable;
            if (maskBits > 1 && write.getMask())
              bit = comb::AndOp::create(
                  b, bit,
                  b.createOrFold<comb::ExtractOp>(intType(1), write.getMask(), i));
            unsigned perBit = bytes / maskBits;
            laneMask.push_back(perBit > 1 ? b.createOrFold<comb::ReplicateOp>(
                                                intType(perBit), bit)
                                          : bit);
          }
          masks.push_back(b.createOrFold<comb::ConcatOp>(laneMask));
          write.erase();
        } else {
          datas.push_back(zero(bytes * 8));
          masks.push_back(zero(bytes));
        }
      }
      if (anyWrite)
        wrValid = comb::AndOp::create(b, region.state, anyWrite);
      std::reverse(datas.begin(), datas.end());
      std::reverse(masks.begin(), masks.end());
      wrData = b.createOrFold<comb::ConcatOp>(datas);
      wrMask = b.createOrFold<comb::ConcatOp>(masks);
      Value data = readBuffer();
      for (auto [lane, offset] : llvm::zip(vm.lanes, vm.laneOffset)) {
        seq::FirMemReadOp read;
        for (auto *user : lane->getUsers())
          read = cast<seq::FirMemReadOp>(user);
        unsigned width = lane.getType().getWidth();
        Value slice = b.createOrFold<comb::ExtractOp>(intType(width), data, offset);
        slice = b.createOrFold<hw::BitcastOp>(read.getData().getType(), slice);
        read.getData().replaceAllUsesWith(slice);
        read.erase();
      }
      for (auto lane : llvm::drop_begin(vm.lanes))
        lane.erase();
    }
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
      Value data = readBuffer();
      data = b.createOrFold<hw::BitcastOp>(read.getData().getType(), data);
      read.getData().replaceAllUsesWith(data);
      read.erase();
    }
    memory.erase();
    auto bitOf = [&](Value index) -> Value {
      return comb::ShlOp::create(b, hw::ConstantOp::create(b, intType(entries), 1),
                                 widen(index, entries));
    };
    if (vm.delayed) {
      // A held read (--vmem-skip-held-reads). The requests leave a host cycle
      // after the state pass, from registers, and the skip is decided there
      // from registers alone: no request valid waits for the target's paths
      // to the read's enable or the write's address.
      // `fresh` has a bit per instance: its read data is the row its last read
      // took, as a read of that row would see it now. Set at a state pass that
      // issued (or skipped) its read, unless (read first) the same pass writes
      // the row; cleared at a pass with the read's enable low; all clear after
      // a reset. A read of a fresh instance's last row (the held row: the
      // read's address register, so whenever its enable is low) is skipped,
      // unless (write first) the same pass writes that row.
      Value hostReset = body->getArgument(cls.resetInput);
      auto stage = [&](Value value, StringRef suffix, bool reset) -> Value {
        if (isa_and_nonnull<hw::ConstantOp>(value.getDefiningOp()))
          return value;
        auto name = b.getStringAttr(vm.name + "_q" + suffix);
        if (!reset)
          return seq::FirRegOp::create(b, value, hostClock, name);
        return seq::FirRegOp::create(
            b, value, hostClock, name, hostReset,
            hw::ConstantOp::create(b, value.getType(), 0), hw::InnerSymAttr{},
            /*isAsync=*/false, b.getIntegerAttr(value.getType(), 0));
      };
      Value stateQ = stage(region.state, "_state", true);
      Value instQ = stage(inst, "_inst", false);
      Value rdValidQ = stage(rdValid, "_rd_valid", true);
      Value rdRowQ = stage(rdRow, "_rd_row", false);
      Value heldRowQ = stage(widen(vm.heldRow, vm.rowWidth), "_held_row", false);
      Value wrValidQ = stage(wrValid, "_wr_valid", true);
      Value wrRowQ = stage(wrRow, "_wr_row", false);
      Value wrDataQ = stage(wrData, "_wr_data", false);
      Value wrMaskQ = stage(wrMask, "_wr_mask", false);
      Value ones = hw::ConstantOp::create(b, APInt::getAllOnes(entries));
      auto freshNext = edges.get(intType(entries));
      Value fresh = seq::FirRegOp::create(
          b, freshNext, hostClock, b.getStringAttr(vm.name + "_fresh"),
          hostReset, zero(entries), hw::InnerSymAttr{},
          /*isAsync=*/false, b.getIntegerAttr(intType(entries), 0));
      Value isFresh = b.createOrFold<comb::ExtractOp>(
          intType(1), comb::ShrUOp::create(b, fresh, widen(instQ, entries)), 0);
      // The pass writes the row it reads.
      Value writesRow = comb::AndOp::create(
          b, wrValidQ,
          comb::ICmpOp::create(b, comb::ICmpPredicate::eq, wrRowQ, rdRowQ));
      Value notWrites = comb::XorOp::create(b, writesRow, one());
      Value skip = comb::AndOp::create(
          b, isFresh,
          comb::ICmpOp::create(b, comb::ICmpPredicate::eq, rdRowQ, heldRowQ));
      if (!vm.readFirst)
        skip = comb::AndOp::create(b, skip, notWrites);
      Value current = vm.readFirst
                          ? Value(comb::AndOp::create(b, rdValidQ, notWrites))
                          : rdValidQ;
      Value others = comb::AndOp::create(
          b, fresh, comb::XorOp::create(b, bitOf(instQ), ones));
      Value updated = comb::OrOp::create(
          b, others, comb::MuxOp::create(b, current, bitOf(instQ), zero(entries)));
      freshNext.setValue(comb::MuxOp::create(b, stateQ, updated, fresh));
      rdValid = comb::AndOp::create(b, rdValidQ,
                                    comb::XorOp::create(b, skip, one()));
      inst = instQ;
      rdRow = rdRowQ;
      wrValid = wrValidQ;
      wrRow = wrRowQ;
      wrData = wrDataQ;
      wrMask = wrMaskQ;
    }
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
    if (physical(&op) || deferControls.contains(&op))
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
    /// With page masks, the enable it is written under, registered at the
    /// state pass (null: written every state pass).
    Value enable;
  };
  SmallVector<PagedReg> pagedRegs;
  // Off chip, each page column's virtual memory and its outputs (requests
  // and pending reads), appended to the copy's ports once its body is built.
  SmallVector<std::pair<unsigned, Operation *>> pageRequests;
  // With page masks, each enable's register, one per enable.
  DenseMap<std::pair<Value, unsigned>, Value> pagedEnableRegs;
  BackedgeBuilder pageEdges(b, model.getLoc());
  unsigned keptRegs = 0;
  uint64_t keptBits = 0;

  // A virtual memory's later lanes go with its first.
  DenseSet<Operation *> laneDone;
  for (auto &[stateName, op] : state) {
    if (laneDone.contains(op))
      continue;
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      b.setLoc(reg.getLoc());
      auto width = hw::getBitWidth(reg.getType());
      Value next = reg.getNext();
      if (reg.getReset())
        next = comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), next);
      Value value;
      // Pass splitting: written deferSpan host cycles after the state pass,
      // the whole next value, at the instance captured there. A paged one
      // (pages on chip) loads its write-back register then instead; the
      // copy's page writes start then (below).
      bool deferLoad = false;
      auto deferred = deferredNext.find(reg);
      if (deferred != deferredNext.end() && cls.page &&
          !(cls.pagesOffChip && cls.pageWindow) && threads > 1 && width &&
          !readyRegs.contains(reg)) {
        deferLoad = true;
        next = deferred->second;
      } else if (deferred != deferredNext.end()) {
        auto it = deferred;
        auto &region = regionFor(width);
        Value data = asBits(b, it->second);
        if (threads == 1) {
          BackedgeBuilder backedge(b, reg.getLoc());
          auto nextEdge = backedge.get(data.getType());
          Value kept = seq::FirRegOp::create(b, nextEdge, hostClock,
                                             reg.getNameAttr());
          nextEdge.setValue(comb::MuxOp::create(b, deferWen, data, kept));
          value = b.createOrFold<hw::BitcastOp>(reg.getType(), kept);
        } else {
          auto *rule = storageFor(cls, reg.getName(), /*reg=*/true, width);
          bool block = rule && (rule->kind == "block" || rule->kind == "ultra");
          auto memType = seq::FirMemType::get(context, entries, width, 1);
          auto memory = seq::FirMemOp::create(
              b, memType, /*readLatency=*/block ? 1 : 0, /*writeLatency=*/1,
              block ? seq::RUW::Old : seq::RUW::Undefined, seq::WUW::Undefined,
              reg.getNameAttr(), hw::InnerSymAttr{}, seq::FirMemInitAttr{},
              StringAttr{}, Attribute{});
          keepShallowInFlops(memory);
          if (rule)
            setRamStyle(memory, rule->kind);
          Value rdata = seq::FirMemReadOp::create(
              b, memory, block ? region.inst2 : region.inst1, hostClock, one());
          Value read = seq::FirRegOp::create(
              b, rdata, hostClock, b.getStringAttr(reg.getName() + "_read"));
          seq::FirMemWriteOp::create(b, memory, deferInst, hostClock, deferWen,
                                     data, Value{});
          value = b.createOrFold<hw::BitcastOp>(reg.getType(), read);
        }
        reg.getResult().replaceAllUsesWith(value);
        reg.erase();
        continue;
      }
      auto write = deferLoad ? foldsim::HeldWrite{} :
                   foldsim::peelHolds(b, next, reg.getResult());
      if (!write.holds && !write.enable)
        write.data = next;
      auto &region = regionFor(width);
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
        // Under masks the enable is the write mask's: the next value goes
        // straight into the write-back register.
        Value enable;
        if (write.enable && pageMasks) {
          Value &kept = pagedEnableRegs[{write.enable, 0u}];
          if (!kept) {
            BackedgeBuilder enableEdges(b, reg.getLoc());
            auto enableNext = enableEdges.get(b.getI1Type());
            kept = seq::FirRegOp::create(
                b, enableNext, hostClock,
                b.getStringAttr(reg.getName() + "_wen"));
            enableNext.setValue(
                comb::MuxOp::create(b, region.state, write.enable, kept));
          }
          enable = kept;
        }
        Value next = write.enable && !pageMasks
                         ? Value(comb::MuxOp::create(b, write.enable, data,
                                                     asBits(b, placeholder)))
                         : data;
        BackedgeBuilder backEdges(b, reg.getLoc());
        auto backNext = backEdges.get(data.getType());
        Value back = seq::FirRegOp::create(
            b, backNext, hostClock,
            b.getStringAttr(reg.getName() + "_writeback"));
        backNext.setValue(comb::MuxOp::create(b, deferLoad ? deferWen : region.state,
                                              next, back));
        pagedRegs.push_back({reg.getResult(), placeholder, unsigned(width),
                             back, reg.getName().str(), enable});
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
      reg.erase();
      continue;
    }
    auto memory = cast<seq::FirMemOp>(op);
    b.setLoc(memory.getLoc());
    if (auto it = virtualOf.find(memory); it != virtualOf.end()) {
      auto &lanes = cls.virtualMemories[it->second].lanes;
      for (auto lane : lanes)
        laneDone.insert(lane);
      virtualPorts[it->second] =
          hold(buildVirtualMemory(lanes.front(), it->second));
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
    // Staged, a combinational read takes the instance of its stage.
    auto addressOf = [&](Value addr, Operation *port = nullptr) -> Value {
      Value inst = region.inst;
      if (Value staged = port ? readInst.lookup(port) : Value())
        inst = staged;
      if (!idxWidth)
        return addr;
      if (!addrWidth)
        return inst;
      Value low =
          b.createOrFold<comb::ExtractOp>(b.getIntegerType(addrWidth), addr, 0);
      return b.createOrFold<comb::ConcatOp>(low, inst);
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
                                         addressOf(read.getAddress(), read),
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
        kind != "registers" && kind != "hbm")
      return model.emitError("storage: '") << kind << "' for pages";
    uint64_t pagedBits = 0;
    for (auto &reg : pagedRegs)
      pagedBits += reg.width;
    // The page word's layout: each register's offset in the vector. With
    // masks, the registers written every state pass first, then those of
    // each enable together, each run from a byte boundary and padded to
    // whole bytes (the RAM's write-enable grain), so a byte is one
    // enable's; the page word is whole bytes too.
    bool masked = llvm::any_of(pagedRegs, [](auto &r) { return bool(r.enable); });
    // Off chip, a virtual memory's write mask is bytes: always built.
    bool bytesMasked = masked || offChip;
    unsigned grain = offChip ? 8 : masked ? (kind == "ultra" ? 8 : 9) : 1;
    SmallVector<uint64_t> bitAt(pagedRegs.size(), 0);
    llvm::MapVector<Value, SmallVector<unsigned>> groups;
    for (auto [i, reg] : llvm::enumerate(pagedRegs))
      groups[reg.enable].push_back(i);
    uint64_t layoutBits = 0;
    // The always-written run first (the null enable, if any).
    SmallVector<std::pair<Value, SmallVector<unsigned>>> runs;
    if (auto it = groups.find(Value()); it != groups.end())
      runs.push_back(*it);
    for (auto &[enable, members] : groups)
      if (enable)
        runs.push_back({enable, members});
    // Each byte's enable (null: always written), for the masks.
    SmallVector<Value> byteEnable;
    for (auto &[enable, members] : runs) {
      layoutBits = (layoutBits + grain - 1) / grain * grain;
      for (unsigned i : members) {
        bitAt[i] = layoutBits;
        layoutBits += pagedRegs[i].width;
      }
      uint64_t end = (layoutBits + grain - 1) / grain * grain;
      while (byteEnable.size() * grain < end)
        byteEnable.push_back(enable);
    }
    unsigned pw = unsigned((layoutBits + p - 1) / p);
    pw = (pw + grain - 1) / grain * grain;
    while (bytesMasked && byteEnable.size() * grain < uint64_t(pw) * p)
      byteEnable.push_back(Value());
    // Off chip a column is a virtual memory's row: wide, few controllers.
    unsigned colBits =
        offChip ? std::max(8u, pageHbmColumnBits / 8 * 8)
                : 72 * std::max(1u, regionBits / 72);
    unsigned columns = (pw + colBits - 1) / colBits;
    struct Column {
      unsigned width;
      Region *region;
      seq::FirMemOp memory;
      Value out, writeAt;
      /// The page being written, binary (p > 1).
      Value writeIndex;
      /// Per page: the host cycle it lands, it is written, and its bits,
      /// at their offsets in the column.
      SmallVector<Value> land, writing;
      SmallVector<SmallVector<std::pair<unsigned, Value>>> parts;
      /// Off chip: its virtual memory, its write's valid, and its requests
      /// and pending reads, in the order of the copy's vm ports.
      unsigned vm = 0, vmWidth = 0;
      Value writeValid;
      SmallVector<Value> requests;
      /// Gated (Class::pageGate): per page, the synchronous reset of its
      /// write-back register (low the host cycle before the page's write).
      SmallVector<Value> gateReset;
    };
    std::vector<Column> cols(columns);
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
      if (offChip) {
        // A virtual memory of {instance, page} rows: each position's page
        // read and page write requests, three host cycles from the tables;
        // the responses into the fetch buffer as they come, by page; the
        // instance's read pending from its fetch's decision until its last
        // page is in.
        auto chain = [&](Value v3, const Twine &what) -> Value {
          Value v2 = keep(seq::FirRegOp::create(b, v3, hostClock,
                                                b.getStringAttr(prefix + "_" + what + "2")));
          Value v1 = keep(seq::FirRegOp::create(b, v2, hostClock,
                                                b.getStringAttr(prefix + "_" + what + "1")));
          return keep(seq::FirRegOp::create(b, v1, hostClock,
                                            b.getStringAttr(prefix + "_" + what + "0")));
        };
        Value inst0 = chain(offChip3[1], "pinst");
        Value page0 = chain(offChip3[2], "ppage");
        Value wpage0 = chain(offChip3[4], "pwpage");
        auto low = [&](Value v) -> Value {
          return b.createOrFold<comb::ExtractOp>(bits(pageWidth), v, 0);
        };
        auto isPage = [&](Value v, unsigned j) -> Value {
          if (!pageWidth)
            return one();
          return comb::ICmpOp::create(b, comb::ICmpPredicate::eq, low(v),
                                      hw::ConstantOp::create(b, bits(pageWidth), j));
        };
        auto virtualInst = [&](Value inst, Value page) -> Value {
          return pageWidth ? b.createOrFold<comb::ConcatOp>(inst, low(page)) : inst;
        };
        Class::VirtualMemory vm;
        vm.name = prefix + "_pages";
        vm.rowWidth = 1;
        // A row of a power of two of bits (of at least 32), as the
        // controller lays rows out: the last column's padding is never
        // written (its mask bytes are off).
        col.vmWidth = std::max(32u, unsigned(llvm::PowerOf2Ceil(col.width)));
        vm.dataWidth = col.vmWidth;
        vm.maskWidth = col.vmWidth / 8;
        vm.readFirst = true;
        vm.pages = true;
        vm.instWidth = idxWidth + pageWidth;
        vm.instances = entries * pages;
        if (cls.virtualMemories.empty()) {
          auto [name, arg] = copy.appendInput("hostReset", b.getI1Type());
          cls.resetInput = arg.getArgNumber();
        }
        Value hostReset = body->getArgument(cls.resetInput);
        // The requests' valids from reset registers: no request from the
        // host reset's state (a stray read's answer would end a fetch).
        auto chainR = [&](Value v3, const Twine &what) -> Value {
          Value v = v3;
          for (unsigned d = 3; d-- > 0;) {
            auto type = v.getType();
            v = keep(seq::FirRegOp::create(
                b, v, hostClock, b.getStringAttr(prefix + "_" + what + Twine(d)),
                hostReset, hw::ConstantOp::create(b, type, 0), hw::InnerSymAttr{},
                /*isAsync=*/false, b.getIntegerAttr(type, 0)));
          }
          return v;
        };
        Value write0 = chainR(offChip3[3], "pwrite");
        // A page read only within a fetch that started (its page 0 read):
        // a fetch that wraps into the frame before is partial in the first
        // frame, and its other pages' reads would go out unaccounted.
        auto activeNext = edges.get(b.getI1Type());
        Value active = seq::FirRegOp::create(
            b, activeNext, hostClock, b.getStringAttr(prefix + "_fetching"),
            hostReset,
            hw::ConstantOp::create(b, b.getI1Type(), 0), hw::InnerSymAttr{},
            /*isAsync=*/false, b.getIntegerAttr(b.getI1Type(), 0));
        Value first3 = comb::AndOp::create(b, offChip3[0], isPage(offChip3[2], 0));
        Value last3 = comb::AndOp::create(b, offChip3[0], isPage(offChip3[2], p - 1));
        activeNext.setValue(comb::MuxOp::create(
            b, last3, hw::ConstantOp::create(b, b.getI1Type(), 0),
            comb::OrOp::create(b, first3, active)));
        Value read3 = comb::AndOp::create(
            b, offChip3[0], comb::OrOp::create(b, first3, active));
        Value read0 = chainR(read3, "pread");
        std::string vmName = ("vm" + Twine(cls.virtualMemories.size()) + "_").str();
        auto rspValid = copy.appendInput(vmName + "rsp_valid", b.getI1Type()).second;
        vm.firstInput = rspValid.getArgNumber();
        auto rspInst = copy.appendInput(vmName + "rsp_inst", bits(vm.instWidth)).second;
        auto rspData = copy.appendInput(vmName + "rsp_data", bits(col.vmWidth)).second;
        auto resetReg = [&](Value next, const Twine &name) -> Value {
          auto type = next.getType();
          return seq::FirRegOp::create(b, next, hostClock, b.getStringAttr(name),
                                       hostReset, hw::ConstantOp::create(b, type, 0),
                                       hw::InnerSymAttr{}, /*isAsync=*/false,
                                       b.getIntegerAttr(type, 0));
        };
        // The responses, registered: directly, page j's parts load as it
        // comes; with a window, into the window's row {slot, page}, which
        // the fetch reads as from an on-chip page RAM.
        Value valid = resetReg(rspValid, prefix + "_rsp_valid");
        Value rspAt = seq::FirRegOp::create(b, rspInst, hostClock,
                                            b.getStringAttr(prefix + "_rsp_inst"));
        Value rspBits = b.createOrFold<comb::ExtractOp>(
            bits(col.width),
            seq::FirRegOp::create(b, rspData, hostClock,
                                  b.getStringAttr(prefix + "_rsp_data")),
            0);
        // The instance a response is of, and its slot.
        Value rspInstance = pageWidth ? b.createOrFold<comb::ExtractOp>(
                                            bits(idxWidth), rspAt, pageWidth)
                                      : rspAt;
        Value rspSlot;
        if (window) {
          SmallVector<uint64_t> slots(entries, 0);
          for (unsigned i = 0; i < threads; ++i)
            slots[i] = cls.windowSlot[i];
          rspSlot = table(b, rspInstance, slots, windowWidth);
          unsigned slotsP2 = 1u << windowWidth;
          auto memType = seq::FirMemType::get(context, slotsP2 * pages, col.width, 1);
          auto memory = seq::FirMemOp::create(
              b, memType, /*readLatency=*/1, /*writeLatency=*/1, seq::RUW::Old,
              seq::WUW::Undefined, b.getStringAttr(prefix + "_window"),
              hw::InnerSymAttr{}, seq::FirMemInitAttr{}, StringAttr{},
              Attribute{});
          setRamStyle(memory, "block");
          Value writeAt = pageWidth ? b.createOrFold<comb::ConcatOp>(rspSlot, low(rspAt))
                                    : rspSlot;
          seq::FirMemWriteOp::create(b, memory, writeAt, hostClock, valid, rspBits,
                                     Value{});
          Value data = seq::FirMemReadOp::create(b, memory, region.readAt,
                                                 hostClock, one());
          col.out = seq::FirRegOp::create(b, data, hostClock,
                                          b.getStringAttr(prefix + "_page_out"));
          Value fetch1 = keep(seq::FirRegOp::create(
              b, region.fetch2, hostClock, b.getStringAttr(prefix + "_fetch1")));
          col.land.push_back(keep(seq::FirRegOp::create(
              b, fetch1, hostClock, b.getStringAttr(prefix + "_land0"))));
          for (unsigned j = 1; j < p; ++j)
            col.land.push_back(keep(seq::FirRegOp::create(
                b, col.land.back(), hostClock,
                b.getStringAttr(prefix + "_land" + Twine(j)))));
        } else {
          col.out = rspBits;
          for (unsigned j = 0; j < p; ++j)
            col.land.push_back(comb::AndOp::create(b, valid, isPage(rspAt, j)));
        }
        // Pages in: counted from the fetch's decision (its page 0's read,
        // three ahead); the last clears the pending read. Directly one fetch
        // is in flight at a time; with a window one per slot, each counted
        // in its slot's counter.
        unsigned cw = llvm::Log2_32_Ceil(p + 1);
        Value start = comb::AndOp::create(b, offChip3[0], isPage(offChip3[2], 0));
        Value done;
        unsigned counters = window ? cls.pageWindow : 1;
        Value startSlot, doneTerms;
        if (window) {
          SmallVector<uint64_t> slots(entries, 0);
          for (unsigned i = 0; i < threads; ++i)
            slots[i] = cls.windowSlot[i];
          startSlot = table(b, offChip3[1], slots, windowWidth);
        }
        SmallVector<Value> doneOf;
        for (unsigned w = 0; w < counters; ++w) {
          auto isSlot = [&](Value slot) -> Value {
            if (!window)
              return one();
            return comb::ICmpOp::create(b, comb::ICmpPredicate::eq, slot,
                                        hw::ConstantOp::create(b, bits(windowWidth), w));
          };
          Value mineStart = comb::AndOp::create(b, start, isSlot(startSlot));
          Value mineIn = comb::AndOp::create(b, valid, isSlot(rspSlot));
          auto countNext = edges.get(bits(cw));
          Value count = resetReg(countNext, prefix + "_pages_in" +
                                                (window ? "_" + std::to_string(w) : ""));
          doneOf.push_back(comb::AndOp::create(
              b, mineIn,
              comb::ICmpOp::create(b, comb::ICmpPredicate::eq, count,
                                   hw::ConstantOp::create(b, bits(cw), p - 1))));
          countNext.setValue(comb::MuxOp::create(
              b, mineStart, hw::ConstantOp::create(b, bits(cw), 0),
              comb::MuxOp::create(
                  b, mineIn,
                  comb::AddOp::create(b, count, hw::ConstantOp::create(b, bits(cw), 1)),
                  count)));
        }
        done = doneOf.size() == 1 ? doneOf[0] : comb::OrOp::create(b, doneOf, false);
        auto widen = [&](Value v) -> Value {
          unsigned have = hw::getBitWidth(v.getType());
          return have >= entries
                     ? b.createOrFold<comb::ExtractOp>(bits(entries), v, 0)
                     : b.createOrFold<comb::ConcatOp>(
                           hw::ConstantOp::create(b, bits(entries - have), 0), v);
        };
        Value bit = comb::ShlOp::create(b, hw::ConstantOp::create(b, bits(entries), 1),
                                        widen(offChip3[1]));
        auto pendingNext = edges.get(bits(entries));
        Value pending = resetReg(pendingNext, prefix + "_pages_pending");
        Value all = hw::ConstantOp::create(b, APInt::getAllOnes(entries));
        // Directly, the one fetch in flight is done: no read pending; with a
        // window, the done slot's instance's (the response's).
        Value doneBit = window ? Value(comb::ShlOp::create(
                                     b, hw::ConstantOp::create(b, bits(entries), 1),
                                     widen(rspInstance)))
                               : all;
        pendingNext.setValue(comb::OrOp::create(
            b,
            comb::AndOp::create(
                b, pending,
                comb::XorOp::create(
                    b, comb::MuxOp::create(b, done, doneBit,
                                           hw::ConstantOp::create(b, bits(entries), 0)),
                    all)),
            comb::MuxOp::create(b, start, bit,
                                hw::ConstantOp::create(b, bits(entries), 0))));
        // Write-back: a position's page write, of the last state pass's
        // instance, while that pass's pages are not all written (a state pass
        // that did not run writes none).
        auto wbNext = edges.get(b.getI1Type());
        Value wb = resetReg(wbNext, prefix + "_writeback_valid");
        Value last = comb::AndOp::create(b, write0, isPage(wpage0, p - 1));
        wbNext.setValue(comb::MuxOp::create(
            b, region.state, one(),
            comb::MuxOp::create(b, last, hw::ConstantOp::create(b, b.getI1Type(), 0), wb)));
        auto instNext = edges.get(region.inst.getType());
        Value writeInst = keep(seq::FirRegOp::create(
            b, instNext, hostClock, b.getStringAttr(prefix + "_write_inst")));
        instNext.setValue(
            comb::MuxOp::create(b, region.state, region.inst, writeInst));
        for (unsigned j = 0; j < p; ++j)
          col.writing.push_back(comb::AndOp::create(b, write0, isPage(wpage0, j)));
        if (pageWidth)
          col.writeIndex = low(wpage0);
        if (cls.pageGate) {
          // Each page's gated register loads the host cycle before the
          // page's write: the write tables a register earlier.
          auto earlier = [](Value v) {
            return v.getDefiningOp<seq::FirRegOp>().getNext();
          };
          Value write1 = earlier(write0), wpage1 = earlier(wpage0);
          for (unsigned j = 0; j < p; ++j)
            col.gateReset.push_back(comb::XorOp::create(
                b, comb::AndOp::create(b, write1, isPage(wpage1, j)), one()));
        }
        col.writeValid = comb::AndOp::create(b, write0, wb);
        Value zero1 = hw::ConstantOp::create(b, b.getI1Type(), 0);
        col.requests = {read0, virtualInst(inst0, page0), zero1, col.writeValid,
                        virtualInst(writeInst, wpage0), zero1, Value(), Value(), pending};
        col.parts.resize(p);
        col.vm = cls.virtualMemories.size();
        cls.virtualMemories.push_back(vm);
        continue;
      }
      ports += (col.width + 71) / 72;
      auto memType = seq::FirMemType::get(
          context, entries * pages, col.width,
          masked ? std::optional<size_t>(col.width / grain) : std::optional<size_t>(1));
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
      col.parts.resize(p);
      // Write-back: page j a host cycle + j after the state pass, at the
      // instance of that pass.
      // Pass splitting: the writes start once the deferred registers'
      // write-back parts are loaded, deferSpan host cycles on (the schedule
      // keeps the copy's state passes and the instance's next fetch apart
      // from them).
      Value writeStart = cls.deferSpan ? deferWen : region.state;
      auto &writing = col.writing;
      writing.push_back(keep(seq::FirRegOp::create(
          b, writeStart, hostClock, b.getStringAttr(prefix + "_write0"))));
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
            b, writeStart, hw::ConstantOp::create(b, bits(pageWidth), 0),
            comb::AddOp::create(b, page,
                                hw::ConstantOp::create(b, bits(pageWidth), 1))));
        writeAt = comb::ConcatOp::create(b, writeInst, page);
        col.writeIndex = page;
      }
      col.writeAt = writeAt;
    }
    // Each register's bits, cut where pages and columns meet: a fetch
    // buffer part loaded as its page lands in its column, and the
    // write-back bits of that page and column, low bits first.
    for (auto [r, reg] : llvm::enumerate(pagedRegs)) {
      uint64_t at = bitAt[r];
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
        Value back = b.createOrFold<comb::ExtractOp>(bits(n), reg.writeBack, lo);
        if (!col.gateReset.empty()) {
          // Gated: zero but the host cycle of page j's write (its
          // synchronous reset, a FF's SR pin), so the column ORs its pages.
          // Pages are written in order from the state pass + 1, so only page
          // 0 may load at the state pass itself: from what the write-back
          // register loads then (its next value), the others from it.
          Value from = j ? back
                         : b.createOrFold<comb::ExtractOp>(
                               bits(n),
                               reg.writeBack.getDefiningOp<seq::FirRegOp>().getNext(),
                               lo);
          auto zero = hw::ConstantOp::create(b, bits(n), 0);
          back = seq::FirRegOp::create(
              b, from, hostClock, b.getStringAttr(name + "_gate" + Twine(j)),
              col.gateReset[j], zero, hw::InnerSymAttr{}, /*isAsync=*/false,
              b.getIntegerAttr(bits(n), 0));
        }
        col.parts[j].push_back({offset, back});
        lo += n;
      }
      Value value = parts.size() == 1
                        ? parts[0]
                        : b.createOrFold<comb::ConcatOp>(
                              SmallVector<Value>(parts.rbegin(), parts.rend()));
      value = b.createOrFold<hw::BitcastOp>(Value(reg.value).getType(), value);
      reg.value.setValue(value);
    }
    // Each column's write: page j's bits at its host cycle, and with masks
    // its bytes' enables. The page is chosen by its binary index (the write's
    // page counter, `writeIndex`), a p:1 multiplexer a bit (a LUT6 and the
    // slice's wide-mux cells: about ceil(p / 4) LUTs a bit, two bits a LUT at
    // p 2), not by AND-ing each page with its one-hot strobe and OR-ing them
    // (two inputs a page). The strobes still enable the write.
    for (auto [c, col] : llvm::enumerate(cols)) {
      auto &writing = col.writing;
      Value writeData, writeEnable, writeMask;
      unsigned bytes = bytesMasked ? col.width / grain : 0;
      SmallVector<Value> words, masks;
      for (unsigned j = 0; j < p; ++j) {
        // The page's bits in order, zeros between.
        auto &parts = col.parts[j];
        llvm::sort(parts, [](auto &x, auto &y) { return x.first < y.first; });
        SmallVector<Value> low;
        unsigned have = 0;
        for (auto &[off, value] : parts) {
          if (off > have)
            low.push_back(hw::ConstantOp::create(b, bits(off - have), 0));
          low.push_back(value);
          have = off + hw::getBitWidth(value.getType());
        }
        if (have < col.width)
          low.push_back(hw::ConstantOp::create(b, bits(col.width - have), 0));
        Value word = low.size() == 1
                         ? low[0]
                         : b.createOrFold<comb::ConcatOp>(
                               SmallVector<Value>(low.rbegin(), low.rend()));
        words.push_back(word);
        if (bytesMasked) {
          // Byte k of this page and column: the vector's byte at its offset.
          uint64_t first = (uint64_t(j) * pw + uint64_t(c) * colBits) / grain;
          SmallVector<Value> high;
          for (unsigned k = bytes; k-- > 0;) {
            Value enable = byteEnable[first + k];
            high.push_back(enable ? enable : one());
          }
          masks.push_back(high.size() == 1 ? high[0]
                                           : b.createOrFold<comb::ConcatOp>(high));
        }
        writeEnable = writeEnable
                          ? Value(comb::OrOp::create(b, writeEnable, writing[j]))
                          : writing[j];
      }
      // The page of the write: index j while page j writes (pages past p,
      // when p is not a power of two, never write: zeros).
      auto select = [&](SmallVector<Value> &of, unsigned width) -> Value {
        if (p == 1)
          return of[0];
        SmallVector<Value> elems;
        for (unsigned j = 1u << pageWidth; j-- > 0;)
          elems.push_back(j < p ? of[j] : hw::ConstantOp::create(b, bits(width), 0));
        return hw::ArrayGetOp::create(b, hw::ArrayCreateOp::create(b, elems),
                                      col.writeIndex);
      };
      // Gated: every page's word zero but the one being written: their OR.
      writeData = !col.gateReset.empty()
                      ? Value(comb::OrOp::create(b, words, false))
                      : select(words, col.width);
      if (bytesMasked)
        writeMask = select(masks, bytes);
      if (offChip) {
        unsigned pad = col.vmWidth - col.width;
        if (pad) {
          writeData = b.createOrFold<comb::ConcatOp>(
              hw::ConstantOp::create(b, bits(pad), 0), writeData);
          writeMask = b.createOrFold<comb::ConcatOp>(
              hw::ConstantOp::create(b, bits(pad / 8), 0), writeMask);
        }
        col.requests[6] = writeData;
        col.requests[7] = writeMask;
        pageRequests.push_back({col.vm, hold(col.requests)});
        continue;
      }
      // A column of one byte has a memory of mask width 1, which is no
      // mask (lowering ignores it): its byte's enable is the write's.
      if (writeMask && bytes == 1) {
        writeEnable = comb::AndOp::create(b, writeEnable, writeMask);
        writeMask = Value();
      }
      seq::FirMemWriteOp::create(b, col.memory, col.writeAt, hostClock,
                                 writeEnable, writeData, writeMask);
    }
    std::string text;
    llvm::raw_string_ostream os(text);
    unsigned bound = ports ? threads * unsigned((pagedBits + 72ull * ports - 1) /
                                                (72ull * ports))
                           : 0;
    os << "  pages " << cls.name << ": p " << p << ", " << pagedRegs.size()
       << " registers, " << pagedBits << " bits";
    if (rowMemories.first)
      os << " (with " << rowMemories.first << " memories' rows, "
         << rowMemories.second << " bits)";
    if (masked)
      os << " (" << layoutBits << " in " << grain << "-bit bytes, "
         << groups.size() - (groups.count(Value()) ? 1 : 0) << " enables)";
    if (offChip)
      os << "; off chip: " << columns << " virtual memories of "
         << entries * pages << " rows x " << pw << " bits (vm"
         << cols.front().vm << (columns > 1 ? "-vm" + std::to_string(cols.back().vm) : "")
         << "), lead " << cls.pageLead() << "; kept " << keptRegs << " registers, "
         << keptBits << " bits; state passes " << threads << " x " << cls.pageLead()
         << " = " << threads * cls.pageLead() << "\n";
    else
      os << "; page RAMs " << entries * pages
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
  release(heldResults, outputs);
  for (auto *held : virtualPorts)
    if (held)
      release(held, outputs);
  b.setInsertionPointToEnd(body);
  hw::OutputOp::create(b, outputs);
  for (auto &[k, held] : pageRequests) {
    SmallVector<Value> values;
    release(held, values);
    auto &vm = cls.virtualMemories[k];
    vm.firstOutput = copy.getNumOutputPorts();
    std::string name = ("vm" + Twine(k) + "_").str();
    SmallVector<std::pair<StringAttr, Value>> ports;
    for (auto [what, value] :
         llvm::zip(ArrayRef<StringRef>{"rd_valid", "rd_inst", "rd_row", "wr_valid",
                                       "wr_inst", "wr_row", "wr_data", "wr_mask",
                                       "pending"},
                   values))
      ports.push_back({b.getStringAttr(name + what), value});
    copy.appendOutputs(ports);
  }
  cls.copy = copy;
  model.erase();
  if (cls.multicycle && failed(checkMulticycle(cls)))
    return failure();
  return success();
}

/// Logic between registers: what a multicycle path may pass through.
static bool isPathLogic(Operation *op) {
  return isa<comb::CombDialect>(op->getDialect()) ||
         isa<hw::ConstantOp, hw::AggregateConstantOp, hw::BitcastOp,
             hw::ArrayCreateOp, hw::ArrayGetOp, hw::ArraySliceOp,
             hw::ArrayConcatOp, hw::ArrayInjectOp, hw::StructCreateOp,
             hw::StructExtractOp, hw::StructInjectOp, hw::StructExplodeOp>(op);
}

LogicalResult StaticFolder::checkMulticycle(Class &cls) {
  std::string prefix = "fsmc_" + cls.name + "__";
  auto *body = cls.copy.getBodyBlock();
  // Pass splitting's registers (fsps_<class>__): the holds a lane loads at
  // the end of a replica's pass, and the pass counters (their controls).
  std::string psPrefix = "fsps_" + cls.name + "__";
  DenseSet<Operation *> caps, ends, holds, controls;
  for (auto reg : body->getOps<seq::CompRegClockEnabledOp>()) {
    StringRef name = reg.getName().value_or("");
    if (name.starts_with(psPrefix)) {
      (name.substr(psPrefix.size()).starts_with("hold") ? holds : controls)
          .insert(reg.getOperation());
      continue;
    }
    if (!name.starts_with(prefix))
      continue;
    (name.substr(prefix.size()).starts_with("cap") ? caps : ends)
        .insert(reg.getOperation());
  }
  // Forward: every path out of a capture register ends at an end
  // register's data.
  SmallVector<Value> work;
  DenseSet<Operation *> seen;
  for (auto *cap : caps)
    work.push_back(cap->getResult(0));
  while (!work.empty()) {
    Value value = work.pop_back_val();
    for (auto &use : value.getUses()) {
      auto *user = use.getOwner();
      if (auto reg = dyn_cast<seq::CompRegClockEnabledOp>(user);
          reg && (ends.contains(user) || holds.contains(user)) &&
          use.get() == reg.getInput())
        continue;
      if (!isPathLogic(user))
        return user->emitError("multicycle: a path from ")
               << cls.name << "'s capture registers ends outside its end "
                  "registers";
      if (seen.insert(user).second)
        for (auto result : user->getResults())
          work.push_back(result);
    }
  }
  // Backward: every path into an end register starts at a capture register
  // or a constant (or a hold, whose own input starts there or at the pass
  // counters).
  seen.clear();
  for (auto *end : ends)
    work.push_back(cast<seq::CompRegClockEnabledOp>(end).getInput());
  for (auto *hold : holds)
    work.push_back(cast<seq::CompRegClockEnabledOp>(hold).getInput());
  while (!work.empty()) {
    Value value = work.pop_back_val();
    auto *def = value.getDefiningOp();
    if (def && (caps.contains(def) || holds.contains(def) ||
                controls.contains(def)))
      continue;
    if (!def || !isPathLogic(def))
      return cls.copy.emitError("multicycle: a path into ")
             << cls.name << "'s end registers starts outside its capture "
                "registers";
    if (seen.insert(def).second)
      for (auto operand : def->getOperands())
        work.push_back(operand);
  }
  // And the enables: the capture registers' at the reads, the end
  // registers' at the positions before the state passes, from the copy's
  // ports alone through registers (no logic between).
  auto enableFrom = [&](Operation *reg, StringRef port) -> bool {
    Value en = cast<seq::CompRegClockEnabledOp>(reg).getClockEnable();
    for (int k = 0; k < 3; ++k) {
      auto r = en.getDefiningOp<seq::CompRegClockEnabledOp>();
      if (!r)
        return false;
      en = r.getInput();
    }
    auto arg = dyn_cast<BlockArgument>(en);
    return arg && cls.copy.getInputName(arg.getArgNumber()) == port;
  };
  for (auto *cap : caps)
    if (!enableFrom(cap, "cap3"))
      return cap->emitError("multicycle: a capture register not taken at the reads");
  for (auto *end : ends)
    if (!enableFrom(end, "pre3"))
      return end->emitError("multicycle: an end register not taken before the state passes");
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

void StaticFolder::buildFrameTable(bool stalls) {
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
  frameAt.assign(classes.size(), {});
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
  // A staged class reads its stages before each state pass (which `place`
  // puts at least that far into the frame).
  readFrameAt = frameAt;
  for (auto [c, cls] : llvm::enumerate(classes)) {
    if (!cls.stages)
      continue;
    readFrameAt[c].assign(ii, Evaluation());
    for (unsigned i = 0; i < cls.threads(); ++i) {
      int r = cls.slot[kState][i] - int(cls.stages);
      assert(r >= 0 && "a staged state pass reads within the frame");
      auto &at = readFrameAt[c][r % ii];
      at.inst = i;
      at.state = true;
      at.tail = r >= int(ii);
    }
  }
  this->stalls = stalls;
}

/// Whether a producer writes its links at position `at`.
bool StaticFolder::writesAt(const Producer &producer, unsigned at) const {
  auto &e = frameAt[producer.cls][at];
  return producer.atReady ? bool((e.ready >> producer.readyLevel) & 1)
                          : e.state;
}

/// Whether a read of producer `p`'s link by class `c`, for the instances
/// `takes` holds, at producer instance `from[i]`, ever needs a write
/// forwarded: the write lands `lands` host cycles after its position's (one,
/// or two registered), and the read register loaded in the host cycle it
/// lands is used in the next one, by such an instance at its entry (for an
/// early link, in the bank that read takes: the write's frame's next).
/// Between frames the position may idle, so each host cycle after a frame's
/// last position is the next frame's first, or idle. With bubbles (an
/// `advance` input, or virtual memories) any host cycle may idle: the read
/// `lands` host cycles on is then of a position 1 to `lands` steps on.
bool StaticFolder::needsForward(unsigned p, unsigned c, ArrayRef<bool> takes,
                                ArrayRef<uint64_t> from, unsigned lands) const {
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
      auto &read = readFrameAt[c][pos];
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
}

/// Whether a reader of class `c` ever takes producer `p`'s write around its
/// read register: a write at a position that evaluates an instance `takes`
/// holds, at its entry.
bool StaticFolder::takesNow(unsigned p, unsigned c, ArrayRef<bool> takes,
                            ArrayRef<uint64_t> from) const {
  auto &producer = producers[p];
  bool threaded = classes[producer.cls].threads() > 1;
  for (unsigned at = 0; at < ii; ++at) {
    if (!writesAt(producer, at))
      continue;
    auto &read = readFrameAt[c][at];
    if (!read.runs() || !takes[read.inst])
      continue;
    if (threaded && from[read.inst] != frameAt[producer.cls][at].inst)
      continue;
    return true;
  }
  return false;
}

/// Whether the hit of such a take (a write at the entry the evaluated
/// instance's row names) comes only at positions evaluating an instance
/// `takes` holds: then no other source's instance needs it masked.
bool StaticFolder::hitsOnlyTakes(unsigned p, unsigned c, ArrayRef<bool> takes,
                                 ArrayRef<uint64_t> from) const {
  auto &producer = producers[p];
  bool threaded = classes[producer.cls].threads() > 1;
  for (unsigned at = 0; at < ii; ++at) {
    auto &read = readFrameAt[c][at];
    if (!writesAt(producer, at) || !read.runs() || takes[read.inst])
      continue;
    if (!threaded || from[read.inst] == frameAt[producer.cls][at].inst)
      return false;
  }
  return true;
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
  auto &pf = portFold;
  auto folded = [&](unsigned p) { return pf.on && pf.producerPort[p] >= 0; };
  auto foldedOut = [&](unsigned k) { return pf.on && pf.consumerPort[k] >= 0; };
  for (auto [p, producer] : llvm::enumerate(producers)) {
    if (producer.cls >= 0 || folded(p))
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
  // Folded port bundles: the response records in, a record a host cycle.
  unsigned rspPort = 0;
  if (pf.on && pf.rspWidth) {
    addPort("rsp_ready", i1, false);
    rspPort = numIn;
    addPort("rsp_valid", i1, true);
    addPort("rsp_chan", IntegerType::get(context, pf.chanWidth), true);
    addPort("rsp_bits", IntegerType::get(context, pf.rspWidth), true);
  }
  SmallVector<unsigned> outputReady(consumers.size(), 0);
  for (auto [k, consumer] : llvm::enumerate(consumers)) {
    if (consumer.cls >= 0 || foldedOut(k))
      continue;
    outputReady[k] = numIn;
    addPort(consumer.name + "_ready", i1, true);
    addPort(consumer.name + "_valid", i1, false);
    if (consumer.width > 0)
      addPort(consumer.name + "_bits", IntegerType::get(context, consumer.width),
              false);
  }
  // And the request records out.
  unsigned reqPort = 0;
  if (pf.on) {
    reqPort = numIn;
    addPort("req_ready", i1, true);
    addPort("req_valid", i1, false);
    addPort("req_chan", IntegerType::get(context, pf.chanWidth), false);
    addPort("req_treset", i1, false);
    addPort("req_bits", IntegerType::get(context, pf.reqWidth), false);
  }
  unsigned advancePort = numIn;
  if (bubbles)
    addPort("advance", i1, true);
  // Each class's virtual memories, numbered over the netlist: a controller's
  // responses and free request entries in, its requests out.
  struct SimVirtual {
    unsigned cls, k, firstInput;
    /// An HBM link stream's (`k` its index), not a class's.
    bool stream = false;
    /// An HBM page stream's engine (`k` its index), not a class's.
    bool engine = false;
  };
  SmallVector<SimVirtual> simVirtuals;
  if (!linkStreams.empty() && failed(planStreams()))
    return failure();
  if (failed(assignPageEngines()))
    return failure();
  // Each class virtual memory's, unless an engine carries it.
  std::map<std::pair<unsigned, unsigned>, unsigned> simOf;
  for (auto [c, cls] : llvm::enumerate(classes))
    for (auto [k, vm] : llvm::enumerate(cls.virtualMemories)) {
      if (pageEngineOf.count({unsigned(c), unsigned(k)}))
        continue;
      simOf[{unsigned(c), unsigned(k)}] = simVirtuals.size();
      std::string name = ("vm" + Twine(simVirtuals.size()) + "_").str();
      unsigned instWidth =
          vm.instWidth ? vm.instWidth
                       : std::max(cls.threads() > 1 ? llvm::Log2_32_Ceil(cls.threads()) : 0u,
                                  1u);
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
  // Each HBM link stream's: its rows a single address (instances 1), the
  // request's tag a landing slot.
  for (auto [k, st] : llvm::enumerate(linkStreams)) {
    if (!st.kept || st.engine >= 0)
      continue;
    std::string name = ("vm" + Twine(simVirtuals.size()) + "_").str();
    st.vm = simVirtuals.size();
    simVirtuals.push_back({unsigned(k), unsigned(k), unsigned(numIn), true});
    auto iw = IntegerType::get(context, st.portInstBits());
    auto aw = IntegerType::get(context, streamRowBits(st));
    auto dw = IntegerType::get(context, st.dw);
    addPort(name + "rsp_valid", i1, true);
    addPort(name + "rsp_inst", iw, true);
    addPort(name + "rsp_data", dw, true);
    addPort(name + "free", IntegerType::get(context, kVirtualFreeWidth), true);
    addPort(name + "rd_valid", i1, false);
    addPort(name + "rd_inst", iw, false);
    addPort(name + "rd_row", aw, false);
    addPort(name + "wr_valid", i1, false);
    addPort(name + "wr_inst", iw, false);
    addPort(name + "wr_row", aw, false);
    addPort(name + "wr_data", dw, false);
    addPort(name + "wr_mask", i1, false);
  }
  // Each page engine's: {tag, instance 0} requests, {member, instance, row}
  // rows, byte masks.
  SmallVector<unsigned> engineSim;
  for (auto [e, eng] : llvm::enumerate(pageEngines)) {
    std::string name = ("vm" + Twine(simVirtuals.size()) + "_").str();
    engineSim.push_back(simVirtuals.size());
    simVirtuals.push_back({unsigned(e), unsigned(e), unsigned(numIn), false, true});
    auto iw = IntegerType::get(context, eng.tagBits() + 1);
    auto aw = IntegerType::get(context, eng.rowBits());
    auto dw = IntegerType::get(context, eng.dataWidth);
    auto mw = IntegerType::get(context, eng.dataWidth / 8);
    addPort(name + "rsp_valid", i1, true);
    addPort(name + "rsp_inst", iw, true);
    addPort(name + "rsp_data", dw, true);
    addPort(name + "free", IntegerType::get(context, kVirtualFreeWidth), true);
    addPort(name + "rd_valid", i1, false);
    addPort(name + "rd_inst", iw, false);
    addPort(name + "rd_row", aw, false);
    addPort(name + "wr_valid", i1, false);
    addPort(name + "wr_inst", iw, false);
    addPort(name + "wr_row", aw, false);
    addPort(name + "wr_data", dw, false);
    addPort(name + "wr_mask", mw, false);
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
    /// ahead (a page before its state pass). Off chip: each position's page
    /// read [valid, instance, page] and page write [valid, page].
    Value fetch3, finst3;
    SmallVector<Value> offChip3;
    /// The frame's position three host cycles ahead, as the tables read it.
    Value at3;
    Value inst2, state2, ready2, tail2;
    /// Whether the slot is the frame before's, three host cycles ahead.
    Value tail3;
    Value inst1, inst, state, ready;
    /// The ready passes past the first, two host cycles ahead.
    SmallVector<Value> laterReady2;
    /// Staged (read side): whether the position three host cycles ahead is
    /// one, not idle; multicycle, whether it is a read, and (`pre3`) whether
    /// it is the one before a state pass.
    Value live3, pre3;
  };
  SmallVector<Controls> controls(classes.size());
  // Staged classes: what each reads at a position, its stages before each
  // state pass (its links' reads, the copy's reads of its state); the
  // writes are `controls`'.
  SmallVector<Controls> readControls(classes.size());
  auto readCtl = [&](unsigned c) -> Controls & {
    return classes[c].stages ? readControls[c] : controls[c];
  };
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
    ctl.at3 = at;
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
      ctl.tail3 = tail3;
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
    auto wrap = [&](int pos) { return unsigned(((pos % int(ii)) + int(ii)) % int(ii)); };
    if (cls.page && !cls.pagesOffChip) {
      // Each state pass's page-in, a page before it in the position stream
      // (in the frame before for a slot within the page of the frame's start).
      SmallVector<uint64_t> fetchRow(ii + 1, 0), finstRow(ii + 1, 0);
      // Staged, a page before the evaluation's read, its stages before.
      for (unsigned i = 0; i < threads; ++i) {
        unsigned pos = wrap(cls.slot[kState][i] - int(cls.pageLead()) -
                            int(cls.stages));
        fetchRow[pos] = 1;
        finstRow[pos] = i;
      }
      ctl.fetch3 = table(b, at, fetchRow, 1);
      ctl.finst3 = table(b, at, finstRow, idxWidth);
    } else if (cls.page) {
      // Off chip: page j of each state pass's instance read at the fetch's
      // j-th position, and written at the j-th after the state pass.
      unsigned pageBits = std::max(
          1u, llvm::Log2_32_Ceil(unsigned(llvm::PowerOf2Ceil(cls.page))));
      SmallVector<uint64_t> readRow, instRow, pageRow, writeRow, wpageRow;
      offChipRows(cls, readRow, instRow, pageRow, writeRow, wpageRow);
      if (cls.pageWindow) {
        // The fetch from the window, a lead before the evaluation's read
        // (staged, its stages before the state pass), as on chip.
        SmallVector<uint64_t> fetchRow(ii + 1, 0), fslotRow(ii + 1, 0);
        for (unsigned i = 0; i < threads; ++i) {
          unsigned pos = wrap(cls.slot[kState][i] - int(cls.pageLead()) -
                              int(cls.stages));
          fetchRow[pos] = 1;
          fslotRow[pos] = cls.windowSlot[i];
        }
        ctl.fetch3 = table(b, at, fetchRow, 1);
        ctl.finst3 = table(b, at, fslotRow,
                           cls.pageWindow > 1 ? llvm::Log2_32_Ceil(cls.pageWindow) : 1);
      }
      ctl.offChip3 = {table(b, at, readRow, 1), table(b, at, instRow, idxWidth),
                      table(b, at, pageRow, pageBits), table(b, at, writeRow, 1),
                      table(b, at, wpageRow, pageBits)};
    }
    for (auto [l, row] : llvm::enumerate(laterRows)) {
      Value later3 = table(b, at, row, 1);
      if (overlapped)
        later3 = comb::AndOp::create(b, later3, runs);
      ctl.laterReady2.push_back(
          reg(later3, prefix + "_ready_l" + Twine(l + 1) + "_2"));
    }
    // Staged: the read side, the stages before each state pass (within the
    // frame, or past it as the state pass may be).
    if (cls.stages) {
      auto &rctl = readControls[c];
      SmallVector<uint64_t> rinstRow(ii + 1, 0), rtailRow(ii + 1, 0),
          liveRow(ii + 1, 1), preRow(ii + 1, 0);
      liveRow[ii] = 0;
      // Multicycle: the capture registers taken at the reads alone, the end
      // registers at the positions before the state passes.
      if (cls.multicycle)
        llvm::fill(liveRow, 0);
      for (unsigned i = 0; i < threads; ++i) {
        int r = cls.slot[kState][i] - int(cls.stages);
        rinstRow[r % ii] = i;
        rtailRow[r % ii] = r >= int(ii);
        if (cls.multicycle) {
          liveRow[r % ii] = 1;
          preRow[(cls.slot[kState][i] - 1) % ii] = 1;
        }
      }
      rctl.at3 = at;
      rctl.live3 = table(b, at, liveRow, 1);
      if (cls.multicycle)
        rctl.pre3 = table(b, at, preRow, 1);
      if (threads > 1) {
        rctl.inst3 = table(b, at, rinstRow, idxWidth);
        rctl.inst2 = reg(rctl.inst3, prefix + "_rinst2");
        rctl.inst1 = reg(rctl.inst2, prefix + "_rinst1");
        rctl.inst = reg(rctl.inst1, prefix + "_rinst0");
      }
      if (overlapped) {
        rctl.tail3 = table(b, at, rtailRow, 1);
        rctl.tail2 = reg(rctl.tail3, prefix + "_rtail2");
      }
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
  Value rspReadyOut;
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
  // Folded port bundles: a response record is taken whole, by the bundle it
  // names, when every field of that bundle has room (the fields' queues move
  // together: a token each, taken and loaded at once).
  SmallVector<Backedge> rspReady;
  SmallVector<Value> rspFor;
  SmallVector<SmallVector<Value>> rspSpace;
  if (pf.on && pf.rspWidth) {
    Value chan = body->getArgument(rspPort + 1);
    for (unsigned port = 0; port < pf.ports; ++port) {
      rspReady.push_back(backedges.get(i1));
      rspFor.push_back(comb::AndOp::create(
          b, body->getArgument(rspPort),
          comb::AndOp::create(
              b, comb::ICmpOp::create(b, comb::ICmpPredicate::eq, chan,
                                      constant(pf.chanWidth, port)),
              rspReady.back())));
    }
    rspSpace.resize(pf.ports);
  }
  for (auto [p, producer] : llvm::enumerate(producers)) {
    if (producer.cls >= 0)
      continue;
    Value valid, bits;
    if (folded(p)) {
      auto &field = pf.rsp[pf.producerField[p]];
      valid = rspFor[pf.producerPort[p]];
      if (producer.width)
        bits = b.createOrFold<comb::ExtractOp>(
            b.getIntegerType(field.width), body->getArgument(rspPort + 2), field.lo);
    } else {
      valid = body->getArgument(inputPorts[p].valid);
      bits = producer.width ? body->getArgument(inputPorts[p].bits) : Value();
    }
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
  // A bundle's record taken when all its fields have room; the stream's
  // ready that of the bundle it names.
  if (pf.on && pf.rspWidth) {
    SmallVector<std::pair<unsigned, Value>> kept;
    for (auto &[p, ready] : inputReady) {
      if (folded(p))
        rspSpace[pf.producerPort[p]].push_back(ready);
      else
        kept.push_back({p, ready});
    }
    inputReady = std::move(kept);
    SmallVector<Value> any;
    Value chan = body->getArgument(rspPort + 1);
    for (unsigned port = 0; port < pf.ports; ++port) {
      Value ready = comb::AndOp::create(b, rspSpace[port], false);
      rspReady[port].setValue(ready);
      any.push_back(comb::AndOp::create(
          b, ready, comb::ICmpOp::create(b, comb::ICmpPredicate::eq, chan,
                                         constant(pf.chanWidth, port))));
    }
    rspReadyOut = comb::OrOp::create(b, any, false);
  }
  // The frame's input as a class's evaluation a host cycle ahead reads it:
  // in a slot past the frame, the frame before's.
  std::map<std::pair<unsigned, int>, Value> prevFrameInputNext;
  DenseMap<unsigned, Value> classTail1;
  auto frameInputFor = [&](unsigned p, unsigned c) -> Value {
    if (!readCtl(c).tail2)
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
      tail1 = reg(readCtl(c).tail2, classes[c].name + "_tail1");
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
    if (cls.page && (!cls.pagesOffChip || cls.pageWindow)) {
      operands.push_back(controls[c].fetch3);
      operands.push_back(controls[c].finst3);
    }
    if (cls.page && cls.pagesOffChip)
      operands.append(controls[c].offChip3.begin(), controls[c].offChip3.end());
    if (cls.stages) {
      if (cls.threads() > 1)
        operands.push_back(readControls[c].inst3);
      operands.push_back(readControls[c].live3);
      if (cls.multicycle)
        operands.push_back(readControls[c].pre3);
    }
    for (auto [i, type] : llvm::enumerate(cls.inputTypes)) {
      if (int(i) == cls.clockInput)
        continue;
      copyInputs[c].push_back(backedges.get(type));
      operands.push_back(copyInputs[c].back());
    }
    if (!cls.virtualMemories.empty())
      operands.push_back(hostReset);
    // Each virtual memory's responses, in the class's order: its own, or its
    // engine's where the tag names it (the instance its low bits).
    for (auto [k, vm] : llvm::enumerate(cls.virtualMemories)) {
      auto carried = pageEngineOf.find({unsigned(c), unsigned(k)});
      if (carried == pageEngineOf.end()) {
        auto &v = simVirtuals[simOf.at({unsigned(c), unsigned(k)})];
        for (unsigned j = 0; j < 3; ++j)
          operands.push_back(body->getArgument(v.firstInput + j));
        continue;
      }
      auto [e, m] = carried->second;
      auto &eng = pageEngines[e];
      auto &v = simVirtuals[engineSim[e]];
      Value tag = b.createOrFold<comb::ExtractOp>(
          b.getIntegerType(eng.tagBits()), body->getArgument(v.firstInput + 1), 1);
      Value member = b.createOrFold<comb::ExtractOp>(
          b.getIntegerType(eng.memberBits()), tag, eng.instWidth);
      operands.push_back(comb::AndOp::create(
          b, body->getArgument(v.firstInput), eq(member, m)));
      operands.push_back(b.createOrFold<comb::ExtractOp>(
          b.getIntegerType(vm.instWidth), tag, 0));
      operands.push_back(b.createOrFold<comb::ExtractOp>(
          b.getIntegerType(vm.dataWidth), body->getArgument(v.firstInput + 2), 0));
    }
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
  // slr-serialize: each class's pass enable, by pass kind (0 the state pass,
  // 1 + l a ready pass of level l), at the pass's host cycle.
  std::map<std::pair<unsigned, unsigned>, Value> serialEnable;
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
  // A compact link's write side (`compact-links`), per link group and set of
  // producer instances kept: each instance's entry and whether it is kept, by
  // tables of the slot's instance registered as the group's instance is (and
  // a host cycle later for a late write).
  struct CompactWrite {
    Value inst, member, instLate, memberLate;
  };
  std::map<std::pair<std::string, std::vector<uint64_t>>, CompactWrite>
      compactWrites;
  // Paged links' gathers, per reader class and page: tables of the frame's
  // position (the class's `at3`) give, for each host cycle of the p before
  // a pass, the page read and the instance whose entry it reads (and
  // whether that position is in another frame than the pass, for the bank);
  // registered as the class's controls are, the page is the read address's
  // low bits a host cycle ahead and selects the register it loads.
  struct LinkGather {
    Value inst2, flip2, page1;
    SmallVector<Value> load0;
  };
  std::map<std::pair<unsigned, unsigned>, LinkGather> linkGathers;
  auto gatherFor = [&](unsigned c, unsigned page) -> const LinkGather & {
    auto &g = linkGathers[{c, page}];
    if (g.page1)
      return g;
    auto &cls = classes[c];
    auto &ctl = controls[c];
    unsigned threads = cls.threads();
    unsigned idxWidth = threads > 1 ? llvm::Log2_32_Ceil(threads) : 1;
    unsigned pageBits = std::max(1u, llvm::Log2_32(unsigned(llvm::PowerOf2Ceil(page))));
    SmallVector<uint64_t> valid(ii + 1, 0), pageRow(ii + 1, 0), inst(ii + 1, 0),
        flip(ii + 1, 0);
    auto floorDiv = [&](int x) {
      return x >= 0 ? x / int(ii) : -((-x + int(ii) - 1) / int(ii));
    };
    for (unsigned i = 0; i < threads; ++i) {
      // (A staged reader reads its stages before its state pass.)
      SmallVector<int> slots{cls.slot[kState][i] - int(cls.stages)};
      for (unsigned l = 0; l < cls.readyLevels; ++l)
        slots.push_back(cls.readySlots(l)[i]);
      for (int s : slots)
        for (unsigned k = 0; k < page; ++k) {
          int q = s - int(page) + int(k);
          unsigned row = unsigned(((q % int(ii)) + int(ii)) % int(ii));
          valid[row] = 1;
          pageRow[row] = k;
          inst[row] = i;
          flip[row] = floorDiv(q) & 1;
        }
    }
    std::string prefix = (cls.name + "_gather" + Twine(page)).str();
    Value valid2 = reg(table(b, ctl.at3, valid, 1), prefix + "_valid2");
    Value page2 = reg(table(b, ctl.at3, pageRow, pageBits), prefix + "_page2");
    g.inst2 = threads > 1 ? reg(table(b, ctl.at3, inst, idxWidth), prefix + "_inst2")
                          : Value();
    g.flip2 = reg(table(b, ctl.at3, flip, 1), prefix + "_flip2");
    g.page1 = reg(page2, prefix + "_page1");
    for (unsigned k = 0; k < page; ++k) {
      Value load2 = comb::AndOp::create(
          b, valid2,
          comb::ICmpOp::create(b, comb::ICmpPredicate::eq, page2,
                               constant(pageBits, k)));
      g.load0.push_back(reg(reg(load2, prefix + "_load1_" + Twine(k)),
                            prefix + "_load0_" + Twine(k)));
    }
    return g;
  };
  // With `compact` (the producer instances the reader takes, sorted), the
  // link keeps an entry for each of those alone, at its index among them;
  // the other instances' writes are masked, and so are their forwards and
  // takes (a masked write's entry may be a kept one's). The reader's address
  // is then that index.
  auto linkRead = [&](unsigned p, unsigned low, unsigned width,
                      function_ref<Value()> address, function_ref<Value()> bank,
                      const Twine &linkName, int to,
                      std::pair<Value, Value> *now = nullptr,
                      bool forward = true,
                      SmallVectorImpl<AfterForward> *after = nullptr,
                      bool late = false, bool forwardLate = false,
                      ArrayRef<uint64_t> compact = {},
                      StringRef syncStyle = {}, unsigned page = 0,
                      const LinkGather *gather = nullptr) -> Value {
    assert((!page || (gather && !late && !now && !forward && !after)) &&
           "a paged link is never forwarded, taken now or written late");
    auto &producer = producers[p];
    unsigned c = producer.cls;
    auto &ctl = controls[c];
    unsigned threads =
        compact.empty() ? classes[c].threads() : unsigned(compact.size());
    bool banked = !producer.late;
    bool crossing = to >= 0 && !slrOf.empty() && slrOf[c] != to;
    assert(!(crossing && now) && "a link between SLR groups is never zero");
    // slr-serialize: the host cycles the crossing takes; every register of
    // the write (enable, entry, bank) is that many host cycles late.
    unsigned serial = crossing ? serialOf(c) : 1;
    unsigned crossLate = crossing ? crossLatency(c) : 1;
    auto lateBy = [&](Value v, const Twine &lateName) {
      std::string n0 = lateName.str();
      v = copyReg(v, n0);
      for (unsigned j = 1; j < crossLate; ++j)
        v = copyReg(v, n0 + std::to_string(j + 1));
      return v;
    };
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
          std::string crossName = producer.name + "_slr" + std::to_string(to) +
                                  "_" + std::to_string(low + at);
          if (serial <= 1) {
            registered = copyReg(data, crossName);
            tagGroup(registered.getDefiningOp(), slrOf[c]);
          } else {
            // slr-serialize: the producer's group holds the bits of its
            // pass (host cycle T) and sends chunk j of them in a register
            // over T + 2 + j, every chunk from the held bits: the crossing
            // register is fed by registers alone (a mux of the hold), never
            // by the pass's own logic, so a crossing adds no logic level to
            // the producer's paths. The reader's group shifts the chunks
            // along, so at T + k + 1 the last chunk and the shifted ones are
            // the bits.
            Operation *mark = &body->back();
            unsigned cw = (w + serial - 1) / serial;
            Value &e0 = serialEnable[{c, producer.atReady ? 1 + producer.readyLevel : 0u}];
            if (!e0) {
              Value base = !producer.atReady ? ctl.state2
                           : producer.readyLevel ? ctl.laterReady2[producer.readyLevel - 1]
                                             : ctl.ready2;
              std::string en = classes[c].name + "_serial" +
                               (!producer.atReady ? std::string("_state")
                                : producer.readyLevel ? "_ready_l" + std::to_string(producer.readyLevel)
                                        : std::string("_ready"));
              e0 = copyReg(copyReg(base, en + "1"), en + "0");
            }
            SmallVector<Value> step{e0};
            for (unsigned j = 1; j <= serial; ++j)
              step.push_back(copyReg(step.back(), crossName + "_step" +
                                                      std::to_string(j)));
            Value hold = seq::CompRegClockEnabledOp::create(
                b, data, hostClock, e0, b.getStringAttr(crossName + "_hold"),
                Value(), Value(), Value(), hw::InnerSymAttr());
            auto chunk = [&](Value from, unsigned j) -> Value {
              unsigned lo = std::min(j * cw, w);
              unsigned n = std::min(cw, w - lo);
              if (!n)
                return constant(cw, 0);
              Value v = b.createOrFold<comb::ExtractOp>(b.getIntegerType(n),
                                                        from, lo);
              return n < cw ? b.createOrFold<comb::ConcatOp>(
                                  constant(cw - n, 0), v)
                            : v;
            };
            Value next = constant(cw, 0);
            for (unsigned j = serial; j-- > 0;)
              next = comb::MuxOp::create(b, step[j + 1], chunk(hold, j), next);
            Value sent = copyReg(next, crossName);
            for (auto *op = mark->getNextNode(); op; op = op->getNextNode())
              if (!isa<hw::ConstantOp>(op))
                tagGroup(op, slrOf[c]);
            // The reader's side (tagged with the reader's group by the
            // input): chunk j shifted k - 1 - j times.
            SmallVector<Value> parts{sent};
            for (unsigned j = 1; j < serial; ++j)
              parts.push_back(copyReg(parts.back(), crossName + "_shift" +
                                                        std::to_string(j)));
            Value all = b.createOrFold<comb::ConcatOp>(parts);
            registered =
                b.createOrFold<comb::ExtractOp>(b.getIntegerType(w), all, 0);
          }
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
          enable = lateBy(enable, group.prefix + pass + "_late");
      }
      // A compact link's entry and mask, registered as the group's.
      Value member, memberLate, compactInst, compactInstLate;
      if (!compact.empty()) {
        auto &cw = compactWrites[{group.prefix,
                                  std::vector<uint64_t>(compact.begin(),
                                                        compact.end())}];
        if (!cw.member) {
          unsigned producerThreads = classes[c].threads();
          SmallVector<uint64_t> index(producerThreads, 0),
              kept(producerThreads, 0);
          for (auto [k, r] : llvm::enumerate(compact)) {
            index[r] = k;
            kept[r] = 1;
          }
          std::string prefix =
              group.prefix + "_compact" + std::to_string(compactWrites.size());
          cw.member = copyReg(copyReg(table(b, ctl.inst2, kept, 1),
                                      prefix + "_member1"),
                              prefix + "_member0");
          if (crossing)
            cw.member = lateBy(cw.member, prefix + "_member_late");
          if (threads > 1) {
            cw.inst = copyReg(
                copyReg(table(b, ctl.inst2, index, llvm::Log2_32_Ceil(threads)),
                        prefix + "_inst1"),
                prefix + "_inst0");
            if (crossing)
              cw.inst = lateBy(cw.inst, prefix + "_inst_late");
          }
        }
        member = cw.member;
        compactInst = cw.inst;
        if (late) {
          if (!cw.memberLate)
            cw.memberLate = copyReg(cw.member, group.prefix + "_member_late");
          memberLate = cw.memberLate;
          if (cw.inst && !cw.instLate)
            cw.instLate = copyReg(cw.inst, group.prefix + "_cinst_late");
          compactInstLate = cw.instLate;
        }
      }
      // The pass's enable, masked to the kept instances.
      Value kept = member ? Value(comb::AndOp::create(b, enable, member)) : enable;
      if (now && !banked) {
        if (!now->first)
          now->first = kept;
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
      if (member)
        writeEnable = comb::AndOp::create(b, writeEnable,
                                          late ? memberLate : member);
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
            after->push_back({at, w, kept, data});
          else
            value = late ? comb::MuxOp::create(b, kept, data, value).getResult()
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
      // A block or ultra RAM link (`syncStyle`) reads at the same registered
      // address a host cycle on, into its output register, which is the read
      // register: read-first, as a distributed RAM's read register takes the
      // entry before a write landing at the same edge.
      bool sync = !syncStyle.empty();
      auto makeMemory = [&](uint64_t rows, unsigned cols) {
        auto memory = seq::FirMemOp::create(
            b, seq::FirMemType::get(context, rows, cols, 1),
            /*readLatency=*/sync ? 1 : 0, /*writeLatency=*/1,
            sync ? seq::RUW::Old : seq::RUW::Undefined, seq::WUW::Undefined,
            b.getStringAttr(name), hw::InnerSymAttr{}, seq::FirMemInitAttr{},
            StringAttr{}, Attribute{});
        if (sync)
          setRamStyle(memory, syncStyle);
        return memory;
      };
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
            group.bank = lateBy(group.bank, group.prefix + "_bank_late");
        }
        writeAddress.push_back(group.bank);
        readAddress.push_back(bank());
      }
      if (threads > 1) {
        if (compact.empty() && !group.inst) {
          group.inst = copyReg(copyReg(ctl.inst2, group.prefix + "_inst1"),
                               group.prefix + "_inst0");
          if (crossing)
            group.inst = lateBy(group.inst, group.prefix + "_inst_late");
        }
        writeAddress.push_back(compact.empty() ? group.inst : compactInst);
        readAddress.push_back(address());
      }
      if (page) {
        // A paged link (`link-kinds` with a page p): an entry in p rows of
        // a p-th of its bits, the page in the low address bits. The write
        // holds the producer's bits and the entry at its pass and writes a
        // page a host cycle over the p after it; the reader's gather (the
        // class's gather controls) reads its entry's pages over the p + 1
        // host cycles before its slot into a register each, which together
        // are the read register.
        unsigned pages = llvm::PowerOf2Ceil(page);
        unsigned pageBits = std::max(1u, llvm::Log2_32(pages));
        unsigned cols = (w + page - 1) / page;
        auto memory = makeMemory(uint64_t(depth) * pages, cols);
        Value writeAt = b.createOrFold<comb::ConcatOp>(writeAddress);
        Value heldAt = seq::CompRegClockEnabledOp::create(
            b, writeAt, hostClock, writeEnable,
            b.getStringAttr(name + "_page_at"), Value(), Value(), Value(),
            hw::InnerSymAttr());
        Value held = seq::CompRegClockEnabledOp::create(
            b, writeData, hostClock, writeEnable,
            b.getStringAttr(name + "_page_data"), Value(), Value(), Value(),
            hw::InnerSymAttr());
        // The write's page: a host cycle each after the pass.
        SmallVector<Value> step;
        Value previous = writeEnable;
        for (unsigned k = 0; k < page; ++k)
          step.push_back(previous = reg(previous, name + "_page_we" + Twine(k)));
        Value we = step[0];
        for (unsigned k = 1; k < page; ++k)
          we = comb::OrOp::create(b, we, step[k]);
        SmallVector<Value> indexBits;
        for (unsigned bit = 0; bit < pageBits; ++bit) {
          Value v = constant(1, 0);
          for (unsigned k = 0; k < page; ++k)
            if (k >> bit & 1)
              v = comb::OrOp::create(b, v, step[k]);
          indexBits.push_back(v);
        }
        Value index = b.createOrFold<comb::ConcatOp>(
            SmallVector<Value>(indexBits.rbegin(), indexBits.rend()));
        SmallVector<Value> chunks;
        for (unsigned k = pages; k-- > 0;) {
          unsigned lo = std::min(k * cols, w);
          unsigned n = std::min(cols, w - lo);
          Value chunk = n ? b.createOrFold<comb::ExtractOp>(b.getIntegerType(n),
                                                          held, lo)
                          : constant(cols, 0);
          if (n && n < cols)
            chunk = b.createOrFold<comb::ConcatOp>(constant(cols - n, 0), chunk);
          chunks.push_back(chunk);
        }
        Value pageData = hw::ArrayGetOp::create(
            b, hw::ArrayCreateOp::create(b, chunks), index);
        seq::FirMemWriteOp::create(
            b, memory, b.createOrFold<comb::ConcatOp>(heldAt, index), hostClock,
            we, pageData, Value{});
        // The gather: the read address the caller gave (bank and entry,
        // registered) with the page of the class's gather controls.
        readAddress.push_back(gather->page1);
        Value readAt = b.createOrFold<comb::ConcatOp>(readAddress);
        Value out = seq::FirMemReadOp::create(b, memory, readAt, hostClock,
                                             constant(1, 1));
        SmallVector<Value> gathered;
        for (unsigned k = page; k-- > 0;)
          gathered.push_back(seq::CompRegClockEnabledOp::create(
              b, out, hostClock, gather->load0[k],
              b.getStringAttr(name + "_page" + Twine(k)), Value(), Value(),
              Value(), hw::InnerSymAttr()));
        Value all = b.createOrFold<comb::ConcatOp>(gathered);
        pieces.push_back(
            b.createOrFold<comb::ExtractOp>(b.getIntegerType(w), all, 0));
        continue;
      }
      auto memory = makeMemory(depth, w);
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
          if (compact.empty() && !group.instLate)
            group.instLate = copyReg(group.inst, group.prefix + "_inst_late");
          lateAddress.push_back(compact.empty() ? group.instLate
                                                : compactInstLate);
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
            b, kept,
            comb::ICmpOp::create(b, comb::ICmpPredicate::eq, writeAt, readNow));
      }
      if (oneCycleLinks && forward) {
        // A write reaches a read of its entry in the same host cycle.
        Value hit = comb::AndOp::create(
            b, kept,
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

  buildFrameTable(bool(advance));

  // The link reads built: those forwarded (past the read register), those
  // taken in the producer's host cycle, and those written from a shared
  // register (forwarded the host cycle the write lands), for the report.
  unsigned linkReads = 0, forwardedReads = 0, forwardedAfter = 0, nowReads = 0,
           lateReads = 0, forwardedLate = 0, compactReads = 0, stagedReads = 0,
           syncReads = 0, earlyReads = 0, pagedReads = 0, streamReads = 0;
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

  // HBM link streams, built on their first member: the landings (tables of
  // the reader's position, three host cycles ahead, registered to now) read
  // the landing buffer at the pop pointer; the requests (tables likewise)
  // issue at the request pointer, a landing slot each, pending until its
  // response fills it; the producer's word held at its pass and written a
  // row a host cycle. A landing whose request would have been issued before
  // the first frame (a frame not run: `cross` frames before, more than have
  // run) takes no slot and loads zero, the link's content before any write.
  struct StreamBuilt {
    Value pop1, inst1, row1, page1, data, popPtr, pending;
    SmallVector<Value> outs;
  };
  std::vector<StreamBuilt> streamBuilt(linkStreams.size());
  auto streamFor = [&](unsigned k) -> StreamBuilt & {
    auto &sb = streamBuilt[k];
    if (sb.data)
      return sb;
    auto &st = linkStreams[k];
    auto &rctl = controls[st.rcls];
    auto &pctl = controls[st.pcls];
    std::string prefix = ("lks" + Twine(k)).str();
    unsigned rthreads = classes[st.rcls].threads();
    unsigned pthreads = classes[st.pcls].threads();
    unsigned rIdx = rthreads > 1 ? llvm::Log2_32_Ceil(rthreads) : 0;
    unsigned pIdx = pthreads > 1 ? llvm::Log2_32_Ceil(pthreads) : 0;
    unsigned pgBits = st.pages > 1 ? llvm::Log2_32_Ceil(st.pages) : 0;
    unsigned ib = st.instBits(), slots = st.slots();
    // A response's landing slot: its tag's low bits, above the instance bit
    // (carried by a page engine: the engine's answer whose member is this
    // stream, the slot its member tag's low bits).
    Value rsp, rspInst, rspData;
    if (st.engine < 0) {
      rsp = body->getArgument(simVirtuals[st.vm].firstInput);
      rspInst = b.createOrFold<comb::ExtractOp>(
          b.getIntegerType(ib), body->getArgument(simVirtuals[st.vm].firstInput + 1), 1);
      rspData = body->getArgument(simVirtuals[st.vm].firstInput + 2);
    } else {
      auto &eng = pageEngines[st.engine];
      auto &v = simVirtuals[engineSim[st.engine]];
      Value tag = b.createOrFold<comb::ExtractOp>(
          b.getIntegerType(eng.tagBits()), body->getArgument(v.firstInput + 1), 1);
      Value member = b.createOrFold<comb::ExtractOp>(
          b.getIntegerType(eng.memberBits()), tag, eng.instWidth);
      rsp = comb::AndOp::create(b, body->getArgument(v.firstInput),
                                eq(member, st.member));
      rspInst = b.createOrFold<comb::ExtractOp>(b.getIntegerType(ib), tag, 0);
      rspData = b.createOrFold<comb::ExtractOp>(
          b.getIntegerType(st.dw), body->getArgument(v.firstInput + 2), 0);
    }
    auto chain3 = [&](Value v, const Twine &n) {
      return reg(reg(reg(v, n + "2"), n + "1"), n + "0");
    };
    // Enables from reset registers: no request, write or landing from the
    // host reset's random state.
    auto chain3r = [&](Value v, const Twine &n) {
      return resetReg(resetReg(resetReg(v, n + "2", 0), n + "1", 0), n + "0", 0);
    };
    // Frames run since the reset (saturating), and each landing's frames
    // back to its request.
    int frame = int(ii);
    auto floorDiv = [&](int x) {
      return x >= 0 ? x / frame : -((-x + frame - 1) / frame);
    };
    SmallVector<uint64_t> cross(ii + 1, 0), last(ii + 1, 0);
    last[ii - 1] = 1;
    unsigned maxCross = 0;
    auto &rcls = classes[st.rcls];
    for (unsigned i = 0; i < rthreads; ++i) {
      unsigned n = st.needs[i].size();
      // (A staged reader reads its stages before its state pass.)
      SmallVector<int> passSlots{rcls.slot[kState][i] - int(rcls.stages)};
      for (unsigned l = 0; l < rcls.readyLevels; ++l)
        passSlots.push_back(rcls.readySlots(l)[i]);
      for (int sl : n ? passSlots : SmallVector<int>{})
        for (unsigned j = 0; j < n; ++j) {
          int q = sl - int(n) - 1 + int(j);
          unsigned c = floorDiv(q) -
                       floorDiv(q - int(st.popLead[((q % frame) + frame) % frame]));
          cross[((q % frame) + frame) % frame] = c;
          maxCross = std::max(maxCross, c);
        }
    }
    unsigned cb = std::max(1u, llvm::Log2_32_Ceil(maxCross + 1));
    Value last1 = resetReg(resetReg(table(b, rctl.at3, last, 1), prefix + "_last2", 0),
                           prefix + "_last1", 0);
    auto ranNext = backedges.get(b.getIntegerType(cb));
    Value ran = resetReg(ranNext, prefix + "_ran", 0);
    Value full = comb::ICmpOp::create(b, comb::ICmpPredicate::uge, ran,
                                      constant(cb, maxCross));
    Value last0 = resetReg(last1, prefix + "_last0", 0);
    ranNext.setValue(comb::MuxOp::create(
        b, comb::AndOp::create(b, last0, notOf(full)),
        comb::AddOp::create(b, ran, constant(cb, 1)), ran));
    // The landings, a host cycle ahead.
    auto t1 = [&](ArrayRef<uint64_t> rows, unsigned w, const Twine &n) -> Value {
      if (!w)
        return Value();
      return reg(reg(table(b, rctl.at3, rows, w), prefix + "_" + n + "2"),
                 prefix + "_" + n + "1");
    };
    sb.pop1 = resetReg(resetReg(table(b, rctl.at3, st.popValid, 1), prefix + "_pop2", 0),
                       prefix + "_pop1", 0);
    sb.inst1 = t1(st.popInst, rIdx, "inst");
    sb.row1 = t1(st.popRow, pIdx, "row");
    sb.page1 = t1(st.popPage, pgBits, "page");
    // Missing: requested in a frame before the first (cross > the frames
    // run, which `ran` counts as the landing's host cycle runs).
    Value skip0 = constant(1, 0);
    if (maxCross)
      skip0 = comb::ICmpOp::create(
          b, comb::ICmpPredicate::ugt,
          reg(t1(cross, cb, "cross"), prefix + "_cross0"), ran);
    Value pop0 = resetReg(sb.pop1, prefix + "_pop0", 0);
    Value take0 = comb::AndOp::create(b, pop0, notOf(skip0));
    auto popNext = backedges.get(b.getIntegerType(ib));
    sb.popPtr = resetReg(popNext, prefix + "_popptr", 0);
    popNext.setValue(comb::AddOp::create(
        b, sb.popPtr, b.createOrFold<comb::ConcatOp>(constant(ib - 1, 0), take0)));
    auto landing = seq::FirMemOp::create(
        b, seq::FirMemType::get(context, slots, st.dw, 1), /*readLatency=*/0,
        /*writeLatency=*/1, seq::RUW::Undefined, seq::WUW::Undefined,
        b.getStringAttr(prefix + "_landing"), hw::InnerSymAttr{},
        seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
    seq::FirMemWriteOp::create(b, landing, rspInst, hostClock, rsp, rspData, Value{});
    Value read = seq::FirMemReadOp::create(b, landing, sb.popPtr, hostClock,
                                          constant(1, 1));
    sb.data = comb::MuxOp::create(b, skip0, constant(st.dw, 0), read);
    // The requests, now.
    Value req0 = chain3r(table(b, rctl.at3, st.reqValid, 1), prefix + "_req");
    Value flip0 = chain3(table(b, rctl.at3, st.reqFlip, 1), prefix + "_flip");
    Value parity0 = reg(par1, prefix + "_parity0");
    SmallVector<Value> rowBits{comb::XorOp::create(b, parity0, flip0)};
    if (pIdx)
      rowBits.push_back(chain3(table(b, rctl.at3, st.reqRow, pIdx), prefix + "_rrow"));
    if (pgBits)
      rowBits.push_back(chain3(table(b, rctl.at3, st.reqPage, pgBits), prefix + "_rpage"));
    auto reqNext = backedges.get(b.getIntegerType(ib));
    Value reqPtr = resetReg(reqNext, prefix + "_reqptr", 0);
    reqNext.setValue(comb::AddOp::create(
        b, reqPtr, b.createOrFold<comb::ConcatOp>(constant(ib - 1, 0), req0)));
    // Pending slots: set by a request, cleared by its response.
    auto pendNext = backedges.get(b.getIntegerType(slots));
    sb.pending = resetReg(pendNext, prefix + "_pending", 0);
    auto bitAt = [&](Value on, Value at) -> Value {
      Value one = b.createOrFold<comb::ConcatOp>(constant(slots - 1, 0), on);
      return comb::ShlOp::create(
          b, one, b.createOrFold<comb::ConcatOp>(constant(slots - ib, 0), at));
    };
    Value set = comb::OrOp::create(b, sb.pending, bitAt(req0, reqPtr));
    pendNext.setValue(comb::AndOp::create(
        b, set, comb::XorOp::create(b, bitAt(rsp, rspInst), constant(slots, ~0ull))));
    // The writes: the word held at the producer's pass, a row a host cycle
    // over the pages after it, at {bank, instance, page}.
    Value pass2 = !st.atReady ? pctl.state2
                  : st.level  ? pctl.laterReady2[st.level - 1]
                              : pctl.ready2;
    Value we0 = resetReg(resetReg(pass2, prefix + "_we1", 0), prefix + "_we0", 0);
    Value wparity = reg(par1, prefix + "_wparity0");
    if (pctl.tail2)
      wparity = comb::XorOp::create(
          b, wparity, reg(reg(pctl.tail2, prefix + "_tail1"), prefix + "_tail0"));
    SmallVector<Value> at{notOf(wparity)};
    if (pIdx)
      at.push_back(reg(reg(pctl.inst2, prefix + "_winst1"), prefix + "_winst0"));
    SmallVector<Value> wordParts;
    for (auto &run : llvm::reverse(st.runs))
      wordParts.push_back(b.createOrFold<comb::ExtractOp>(
          b.getIntegerType(run.width),
          asBits(b, copies[st.pcls].getResult(producers[run.producer].port)),
          run.low));
    Value word = b.createOrFold<comb::ConcatOp>(wordParts);
    Value held = seq::CompRegClockEnabledOp::create(
        b, word, hostClock, we0, b.getStringAttr(prefix + "_word"), Value(),
        Value(), Value(), hw::InnerSymAttr());
    Value heldAt = seq::CompRegClockEnabledOp::create(
        b, b.createOrFold<comb::ConcatOp>(at), hostClock, we0,
        b.getStringAttr(prefix + "_word_at"), Value(), Value(), Value(),
        hw::InnerSymAttr());
    // Page j written at the j-th position after the pass (the producer's
    // tables, as every request: bubbles stop it, and other requesters of an
    // engine may take the positions between), while that pass's pages are
    // not all written (a pass that did not run writes none). The next pass
    // is `pages` positions on at the earliest, so the held word is the
    // pass's until its last page.
    Value wv0 = chain3r(table(b, pctl.at3, st.writeValid, 1), prefix + "_wv");
    Value wp0 = pgBits ? chain3(table(b, pctl.at3, st.writePage, pgBits), prefix + "_wp")
                       : Value();
    auto armNext = backedges.get(b.getI1Type());
    Value armed = resetReg(armNext, prefix + "_armed", 0);
    Value lastW = pgBits ? Value(comb::AndOp::create(
                               b, wv0,
                               comb::ICmpOp::create(b, comb::ICmpPredicate::eq, wp0,
                                                    constant(pgBits, st.pages - 1))))
                         : wv0;
    armNext.setValue(comb::MuxOp::create(
        b, we0, constant(1, 1), comb::MuxOp::create(b, lastW, constant(1, 0), armed)));
    Value we = comb::AndOp::create(b, wv0, armed);
    Value wrRow = heldAt, wrData;
    unsigned padded = st.dw * st.pages;
    Value wide = padded > st.wordBits
                     ? b.createOrFold<comb::ConcatOp>(
                           constant(padded - st.wordBits, 0), held)
                     : held;
    if (pgBits) {
      Value index = wp0;
      SmallVector<Value> chunks;
      for (unsigned j = 1u << pgBits; j-- > 0;)
        chunks.push_back(j < st.pages ? b.createOrFold<comb::ExtractOp>(
                                            b.getIntegerType(st.dw), wide, j * st.dw)
                                      : constant(st.dw, 0));
      wrData = hw::ArrayGetOp::create(b, hw::ArrayCreateOp::create(b, chunks), index);
      wrRow = b.createOrFold<comb::ConcatOp>(heldAt, index);
    } else {
      wrData = wide;
    }
    // Requests' {tag, instance}: a read's tag its slot, a write's the top
    // bit alone; instance 0.
    sb.outs = {req0,
               b.createOrFold<comb::ConcatOp>(
                   ValueRange{constant(1, 0), reqPtr, constant(1, 0)}),
               b.createOrFold<comb::ConcatOp>(rowBits), we,
               b.createOrFold<comb::ConcatOp>(
                   ValueRange{constant(1, 1), constant(ib, 0), constant(1, 0)}),
               wrRow, wrData, constant(1, 1)};
    return sb;
  };

  // `staged-link-memories`: a group's memory (StagedGroup), built at its
  // first link: written at the producer's passes of the instances it keeps,
  // as their first registers were, and read at each link's copy position.
  planStagedMemories();
  SmallVector<Value> stagedMemReads(stagedGroups.size());
  auto stagedMemoryRead = [&](const StagedMember &member, Value data) -> Value {
    Value &read = stagedMemReads[member.group];
    if (read)
      return read;
    auto &group = stagedGroups[member.group];
    auto &producer = producers[group.producer];
    auto &pctl = controls[producer.cls];
    std::string name =
        (producer.name + "_staged_mem" + Twine(member.group)).str();
    unsigned depth = std::max<unsigned>(group.insts.size(), 2);
    unsigned aw = llvm::Log2_32_Ceil(depth);
    unsigned pThreads = classes[producer.cls].threads();
    Value pass3 = !producer.atReady ? pctl.state2
                  : producer.readyLevel
                      ? pctl.laterReady2[producer.readyLevel - 1]
                      : pctl.ready2;
    Value write = copyReg(copyReg(pass3, name + "_we1"), name + "_we0");
    Value writeAt = constant(aw, 0);
    if (pThreads > 1) {
      SmallVector<uint64_t> kept(pThreads, 0), index(pThreads, 0);
      for (auto [k, r] : llvm::enumerate(group.insts)) {
        kept[r] = 1;
        index[r] = k;
      }
      write = comb::AndOp::create(
          b, write,
          copyReg(copyReg(table(b, pctl.inst2, kept, 1), name + "_mine1"),
                  name + "_mine0"));
      writeAt = copyReg(copyReg(table(b, pctl.inst2, index, aw), name + "_at1"),
                        name + "_at0");
    }
    auto memory = seq::FirMemOp::create(
        b, seq::FirMemType::get(context, depth, group.width, 1),
        /*readLatency=*/0, /*writeLatency=*/1, seq::RUW::Undefined,
        seq::WUW::Undefined, b.getStringAttr(name), hw::InnerSymAttr{},
        seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
    seq::FirMemWriteOp::create(b, memory, writeAt, hostClock, write, data,
                               Value{});
    // The entry each copy position loads, as the copies' enables are timed.
    SmallVector<uint64_t> entryRow(ii + 1, 0);
    for (auto [pos, entry] : group.entryAt)
      entryRow[pos] = entry;
    Value readAt = copyReg(
        copyReg(copyReg(table(b, pctl.at3, entryRow, aw), name + "_ra2"),
                name + "_ra1"),
        name + "_ra0");
    read = seq::FirMemReadOp::create(b, memory, readAt, hostClock,
                                     constant(1, 1));
    return read;
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
    // A staged class reads its inputs at its read side.
    auto &ctl = readCtl(c);
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
      bool syncPart = false;
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
          // Each instance's constant, by instance: a table of up to 64 bits
          // a slice, the slices joined (high first).
          SmallVector<Value> slices;
          for (unsigned lo = 0; lo < width; lo += 64) {
            unsigned w = std::min(64u, width - lo);
            SmallVector<uint64_t> rows(threads, 0);
            for (unsigned i = 0; i < threads; ++i)
              if (sourceOf[i] == key)
                rows[i] = constants[i].extractBits(w, lo).getZExtValue();
            slices.push_back(threads > 1 ? table(b, ctl.inst1, rows, w)
                                         : hw::ConstantOp::create(
                                               b, b.getIntegerType(w), rows[0]));
          }
          std::reverse(slices.begin(), slices.end());
          sourceData[index] = slices.size() == 1
                                  ? slices[0]
                                  : b.createOrFold<comb::ConcatOp>(slices);
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
        unsigned pThreadsAll = pThreads;
        SmallVector<uint64_t> rows;
        SmallVector<bool> takes;
        rowsOf(key, rows, takes);
        // A compact link: the producer instances taken, and each instance's
        // index among them for its address.
        SmallVector<uint64_t> kept;
        for (unsigned i = 0; i < threads; ++i)
          if (takes[i])
            kept.push_back(rows[i]);
        llvm::sort(kept);
        kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
        SmallVector<uint64_t> entryRows = rows;
        if (compacted(key.first, kept.size())) {
          ++compactReads;
          pThreads = kept.size();
          for (unsigned i = 0; i < threads; ++i)
            if (takes[i])
              entryRows[i] = llvm::lower_bound(kept, rows[i]) - kept.begin();
        } else {
          kept.clear();
        }
        unsigned w = llvm::Log2_32_Ceil(pThreads);
        // Each piece of the link reads at its own copy of the address and
        // the bank.
        unsigned addressCopies = 0, bankCopies = 0;
        auto address = [&]() -> Value {
          if (pThreads <= 1)
            return Value();
          if (threads == 1)
            return constant(w, entryRows[0]);
          return copyReg(table(b, ctl.inst2, entryRows, w),
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
        // The read's address and bank a host cycle earlier, for a block or
        // ultra RAM link read a host cycle ahead (several sources: its
        // output register, then the choice of source, then the read
        // register).
        auto address2 = [&]() -> Value {
          if (pThreads <= 1)
            return Value();
          if (threads == 1)
            return constant(w, entryRows[0]);
          return copyReg(table(b, ctl.inst3, entryRows, w),
                         consumer.name + "_addr2_" + Twine(at) + "_" +
                             Twine(addressCopies++));
        };
        auto bank2 = [&]() -> Value {
          std::string name = (consumer.name + "_parity2_" + Twine(at) + "_" +
                              Twine(bankCopies++))
                                 .str();
          Value parity = copyReg(parD, name);
          if (ctl.tail3)
            parity = comb::XorOp::create(b, parity,
                                         copyReg(ctl.tail3, name + "_tail"));
          return parity;
        };
        // A staged link (`staged-links`): an early link of one producer
        // instance in two registers, the first written at the producer's
        // pass, the second, which the reader reads, loaded from it at a
        // position of the frame between the reads of consecutive frames.
        if (!producer.late && pThreads == 1) {
          SmallVector<unsigned> readers;
          for (unsigned i = 0; i < threads; ++i)
            if (takes[i])
              readers.push_back(i);
          unsigned inst = kept.empty() ? 0 : unsigned(kept[0]);
          int copyAt = stagedCopy(key.first, inst, c, readers);
          // `staged-link-memories`: the group's memory in place of the
          // first register, read at this link's own copy position.
          const StagedMember *gathered =
              copyAt >= 0 ? stagedMemberOf(c, consumer.port, at, key.first,
                                           key.second)
                          : nullptr;
          if (gathered)
            copyAt = gathered->copyAt;
          if (copyAt >= 0) {
            ++stagedReads;
            auto &pctl = controls[producer.cls];
            std::string name = (producer.name + "_to_" + consumer.name +
                                "_" + Twine(at))
                                   .str();
            Value data = b.createOrFold<comb::ExtractOp>(
                b.getIntegerType(width),
                asBits(b, copies[producer.cls].getResult(producer.port)),
                key.second);
            // The pass's enable, of this instance alone where the producer
            // has several.
            Value pass3 = !producer.atReady ? pctl.state2
                          : producer.readyLevel
                              ? pctl.laterReady2[producer.readyLevel - 1]
                              : pctl.ready2;
            Value write = copyReg(copyReg(pass3, name + "_we1"), name + "_we0");
            if (classes[producer.cls].threads() > 1) {
              SmallVector<uint64_t> mine(classes[producer.cls].threads(), 0);
              mine[inst] = 1;
              write = comb::AndOp::create(
                  b, write,
                  copyReg(copyReg(table(b, pctl.inst2, mine, 1),
                                  name + "_mine1"),
                          name + "_mine0"));
            }
            SmallVector<uint64_t> row(ii + 1, 0);
            row[copyAt] = 1;
            Value copy = copyReg(
                copyReg(copyReg(table(b, pctl.at3, row, 1), name + "_copy2"),
                        name + "_copy1"),
                name + "_copy0");
            // Clock enables, as the flops take them: no select on the
            // producer's path into the first register. Named like the other
            // link registers, without an inner symbol: the names repeat
            // where one producer port feeds several instances of a reader
            // (a bit each), which symbols may not.
            auto enabledReg = [&](Value next, Value enable,
                                  const Twine &regName) -> Value {
              return seq::CompRegClockEnabledOp::create(
                  b, next.getType(), next, hostClock, enable,
                  b.getStringAttr(regName), Value(), Value(), Value(),
                  hw::InnerSymAttr{});
            };
            if (gathered) {
              Value read = stagedMemoryRead(*gathered, data);
              sourceData[index] = enabledReg(read, copy, name + "_held");
              continue;
            }
            Value stageA = enabledReg(data, write, name + "_staged");
            Value stageB = enabledReg(stageA, copy, name + "_held");
            sourceData[index] = stageB;
            continue;
          }
        }
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
        // The input's only source, never forwarded or taken around its
        // read register: a deep link in block or ultra RAM, whose output
        // register is the read register.
        StringRef style;
        if (sources.size() == 1 && !now && !forward && !(late && forwardLate) &&
            to < 0 && !(pThreads == 1 && !producer.late) &&
            !(pThreads == 1 && producer.late))
          style = linkStyle(key.first, c, linkDepth(pThreads, !producer.late),
                            width);
        if (!style.empty()) {
          syncPart = true;
          ++syncReads;
        }
        // A paged link (`link-kinds` with a page), where the schedule lets it.
        unsigned page = 0;
        if (!style.empty()) {
          LinkUse use{unsigned(key.first), key.second, width, unsigned(c), std::string(), {}, {}, {}};
          std::set<uint64_t> distinct;
          for (unsigned i = 0; i < threads; ++i)
            if (takes[i]) {
              distinct.insert(rows[i]);
              use.readers.push_back(i);
              use.readerRows.push_back(rows[i]);
            }
          use.rows.assign(distinct.begin(), distinct.end());
          use.sources = sources.size();
          use.port = consumer.port;
          page = linkPaged(use, !producer.late, pThreads);
          pagedReads += page > 0;
        }
        // One of several sources: read a host cycle ahead, where no write
        // of an entry it reads lands in the two host cycles before its read
        // register loads (three for a write from a shared register).
        bool early = false;
        if (style.empty() && sources.size() > 1 && !now && !forward &&
            !shortLead && to < 0 && pThreads != 1 &&
            !needsForward(key.first, c, takes, rows, 2) &&
            !(late && needsForward(key.first, c, takes, rows, 3))) {
          style = linkStyle(key.first, c, linkDepth(pThreads, !producer.late),
                            width);
          early = !style.empty();
          if (early) {
            ++syncReads;
            ++earlyReads;
          }
        }
        // A member of an HBM link stream: its read registers loaded from
        // the landings, page by page, each where its row lands.
        auto member = streamMembers.find(
            {c, consumer.port, at, unsigned(key.first), key.second});
        if (member != streamMembers.end() && member->second.kept &&
            linkStreams[member->second.stream].kept) {
          auto &st = linkStreams[member->second.stream];
          auto &sb = streamFor(member->second.stream);
          unsigned off = member->second.offset;
          unsigned pIdx = pThreadsAll > 1 ? llvm::Log2_32_Ceil(pThreadsAll) : 0;
          unsigned pgBits = st.pages > 1 ? llvm::Log2_32_Ceil(st.pages) : 0;
          std::string name = (producer.name + "_to_" + consumer.name + "_" +
                              Twine(at) + "_hbm")
                                 .str();
          Value mine1 = sb.pop1;
          if (threads > 1) {
            SmallVector<uint64_t> takeRow(threads, 0);
            for (unsigned i = 0; i < threads; ++i)
              takeRow[i] = takes[i];
            mine1 = comb::AndOp::create(b, mine1, table(b, sb.inst1, takeRow, 1));
          }
          if (pIdx) {
            Value want = threads > 1 ? table(b, sb.inst1, rows, pIdx)
                                     : constant(pIdx, rows[0]);
            mine1 = comb::AndOp::create(
                b, mine1,
                comb::ICmpOp::create(b, comb::ICmpPredicate::eq, sb.row1, want));
          }
          SmallVector<Value> slices;
          for (unsigned j = off / st.dw; j <= (off + width - 1) / st.dw; ++j) {
            unsigned lo = std::max(off, j * st.dw);
            unsigned hi = std::min(off + width, (j + 1) * st.dw);
            Value en1 = mine1;
            if (pgBits)
              en1 = comb::AndOp::create(
                  b, en1,
                  comb::ICmpOp::create(b, comb::ICmpPredicate::eq, sb.page1,
                                       constant(pgBits, j)));
            Value en0 = resetReg(en1, name + "_en" + Twine(j), 0);
            slices.push_back(seq::CompRegClockEnabledOp::create(
                b,
                b.createOrFold<comb::ExtractOp>(b.getIntegerType(hi - lo),
                                                sb.data, lo - j * st.dw),
                hostClock, en0, b.getStringAttr(name + Twine(j)), Value(),
                Value(), Value(), hw::InnerSymAttr()));
          }
          std::reverse(slices.begin(), slices.end());
          sourceData[index] = b.createOrFold<comb::ConcatOp>(slices);
          ++streamReads;
          // The only source: the read register (valid from the host cycle
          // before the slot); one of several: as an early read's output.
          if (sources.size() == 1)
            syncPart = true;
          continue;
        }
        if (page) {
          auto &g = gatherFor(c, page);
          auto addressP = [&]() -> Value {
            if (pThreads <= 1)
              return Value();
            if (threads == 1)
              return constant(w, entryRows[0]);
            return copyReg(table(b, g.inst2, entryRows, w),
                           consumer.name + "_paddr" + Twine(at) + "_" +
                               Twine(addressCopies++));
          };
          auto bankP = [&]() -> Value {
            std::string name = (consumer.name + "_pparity1_" + Twine(at) + "_" +
                                Twine(bankCopies++))
                                   .str();
            return comb::XorOp::create(b, copyReg(par2, name),
                                       copyReg(g.flip2, name + "_flip"));
          };
          sourceData[index] = linkRead(
              key.first, key.second, width, addressP, bankP,
              producer.name + "_to_" + consumer.name + "_" + Twine(at), to,
              nullptr, false, nullptr, false, false, kept, style, page, &g);
          continue;
        }
        sourceData[index] =
            early ? linkRead(key.first, key.second, width, address2, bank2,
                             producer.name + "_to_" + consumer.name + "_" +
                                 Twine(at),
                             to, nullptr, false, nullptr, late, false, kept,
                             style)
                  : linkRead(key.first, key.second, width, address, bank,
                             producer.name + "_to_" + consumer.name + "_" +
                                 Twine(at),
                             to, now ? &sourceNow[index] : nullptr, forward,
                             after ? &sourceAfter[index] : nullptr, late,
                             forwardLate, kept, style);
      }
      if (syncPart) {
        // Registered already: wired around the read register.
        parts.push_back(sourceData[0]);
        at = end;
        continue;
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
       << " (forwarded " << forwardedLate << ")";
    if (compactLinks)
      os << "; compact " << compactReads;
    if (stagedLinks)
      os << "; staged " << stagedReads;
    if (linkBlockDepth || linkUltraDepth)
      os << "; block/ultra " << syncReads << " (read ahead " << earlyReads
         << ")";
    if (pagedReads)
      os << "; paged " << pagedReads;
    if (streamReads)
      os << "; hbm " << streamReads;
    os << "\n";
    reportPageEngines(os);
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
  // Folded port bundles: each output's token pushed into its field's
  // record memory (a row a target cycle a bundle), its queue only counted
  // here; a bundle's record leaves whole (`reqSent`), a host cycle after
  // the last of its pushes.
  SmallVector<Backedge> reqSent;
  struct FoldOut {
    Value push, data, queued;
    unsigned capacity = 0;
  };
  std::map<unsigned, FoldOut> foldOut;
  for (unsigned port = 0; pf.on && port < pf.ports; ++port)
    reqSent.push_back(backedges.get(i1));
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
    Value readyIn = foldedOut(k) ? Value(reqSent[pf.consumerPort[k]])
                                 : body->getArgument(outputReady[k]);
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
    Value valid = foldedOut(k) ? queued : comb::OrOp::create(b, queued, push);
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
    if (foldedOut(k)) {
      foldOut[k] = {push, data, queued, capacity};
      continue;
    }
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

  // Folded port bundles: each field's tokens in a memory of a row per bundle
  // and queue entry (the bundle's row in turn: a write pointer per output),
  // written at the field's pushes, one bundle a host cycle (staggered). A
  // bundle's record is sendable once every field holds a token; the lowest
  // such bundle is offered, its fields read at its read pointer, an early
  // field's first token (the queue's from reset) zero, and the frame's reset
  // token stored beside the record at a late field's push.
  Value reqValidOut, reqChanOut, reqTresetOut, reqBitsOut;
  if (pf.on) {
    unsigned P = pf.ports, cw = pf.chanWidth;
    unsigned cap = 2;
    for (auto &[k, fo] : foldOut)
      cap = std::max(cap, fo.capacity);
    unsigned rows = unsigned(llvm::PowerOf2Ceil(cap)), rw = llvm::Log2_32_Ceil(rows);
    // The field whose push carries the frame's reset token: a late one.
    int resetField = -1;
    for (unsigned f = 0; f < pf.req.size() && resetField < 0; ++f)
      if (!outputEarly[pf.reqOf[0][f]])
        resetField = f;
    if (resetField < 0)
      return netlist.emitError("fold-ports: the bundles have no late output to "
                               "carry the frame's reset token");
    // Lowest sendable bundle: a tree of (any, index) over the bundles.
    SmallVector<Value> sendable;
    for (unsigned port = 0; port < P; ++port) {
      SmallVector<Value> all;
      for (unsigned k : pf.reqOf[port])
        all.push_back(foldOut[k].queued);
      sendable.push_back(comb::AndOp::create(b, all, false));
    }
    std::function<std::pair<Value, Value>(unsigned, unsigned)> lowest =
        [&](unsigned lo, unsigned n) -> std::pair<Value, Value> {
      if (n == 1)
        return {sendable[lo], Value()};
      unsigned half = unsigned(llvm::PowerOf2Ceil(n)) / 2;
      auto [la, li] = lowest(lo, half);
      if (n == half)
        return {la, li};
      auto [ra, ri] = lowest(lo + half, n - half);
      unsigned bits = llvm::Log2_32(half);
      auto pad = [&](Value v) -> Value {
        if (!bits)
          return constant(0, 0);
        if (!v)
          return constant(bits, 0);
        unsigned have = hw::getBitWidth(v.getType());
        return have == bits ? v
                            : b.createOrFold<comb::ConcatOp>(constant(bits - have, 0), v);
      };
      Value idx = bits ? b.createOrFold<comb::ConcatOp>(
                             notOf(la), comb::MuxOp::create(b, la, pad(li), pad(ri)))
                       : notOf(la);
      return {comb::OrOp::create(b, la, ra), idx};
    };
    auto [any, low] = lowest(0, P);
    Value chan = !low ? constant(cw, 0)
                 : hw::getBitWidth(low.getType()) == cw
                     ? low
                     : b.createOrFold<comb::ConcatOp>(
                           constant(cw - hw::getBitWidth(low.getType()), 0), low);
    Value sent = comb::AndOp::create(b, any, body->getArgument(reqPort));
    // Per bundle: read pointer and whether its first record is still to go.
    SmallVector<Value> rp, first;
    for (unsigned port = 0; port < P; ++port) {
      Value mine = comb::AndOp::create(
          b, sent, comb::ICmpOp::create(b, comb::ICmpPredicate::eq, chan,
                                        constant(cw, port)));
      reqSent[port].setValue(mine);
      auto rpNext = backedges.get(b.getIntegerType(rw));
      Value r = resetReg(rpNext, "ports_rp" + Twine(port), 0);
      rpNext.setValue(comb::MuxOp::create(
          b, mine, comb::AddOp::create(b, r, constant(rw, 1)), r));
      rp.push_back(r);
      auto firstNext = backedges.get(i1);
      Value fs = resetReg(firstNext, "ports_first" + Twine(port), 1);
      firstNext.setValue(comb::MuxOp::create(b, mine, constant(1, 0), fs));
      first.push_back(fs);
    }
    auto pick = [&](ArrayRef<Value> perPort) -> Value {
      SmallVector<Value> elems;
      unsigned n = 1u << cw;
      for (unsigned j = n; j-- > 0;)
        elems.push_back(perPort[std::min(j, P - 1)]);
      return hw::ArrayGetOp::create(b, hw::ArrayCreateOp::create(b, elems), chan);
    };
    Value rpSel = pick(rp), firstSel = pick(first);
    Value readAddr = b.createOrFold<comb::ConcatOp>(chan, rpSel);
    unsigned depth = (1u << cw) * rows;
    auto memory = [&](const Twine &name, unsigned width,
                      ArrayRef<std::pair<Value, Value>> writes,
                      ArrayRef<Value> pointers) -> Value {
      // One write a host cycle (the pushes are staggered): its bundle's row
      // at its pointer.
      SmallVector<Value> en, addr, data;
      for (auto [port, w] : llvm::enumerate(writes)) {
        en.push_back(w.first);
        Value a = b.createOrFold<comb::ConcatOp>(constant(cw, port), pointers[port]);
        addr.push_back(comb::MuxOp::create(b, w.first, a, constant(cw + rw, 0)));
        data.push_back(comb::MuxOp::create(b, w.first, w.second, constant(width, 0)));
      }
      auto mem = seq::FirMemOp::create(
          b, seq::FirMemType::get(context, depth, width, 1), /*readLatency=*/0,
          /*writeLatency=*/1, seq::RUW::Undefined, seq::WUW::Undefined,
          b.getStringAttr(name), hw::InnerSymAttr{}, seq::FirMemInitAttr{},
          StringAttr{}, Attribute{});
      seq::FirMemWriteOp::create(b, mem, comb::OrOp::create(b, addr, false), hostClock,
                                 comb::OrOp::create(b, en, false),
                                 comb::OrOp::create(b, data, false), Value{});
      return seq::FirMemReadOp::create(b, mem, readAddr, hostClock, constant(1, 1));
    };
    SmallVector<Value> fieldBits;
    Value resetBits;
    for (auto [f, field] : llvm::enumerate(pf.req)) {
      SmallVector<std::pair<Value, Value>> writes;
      SmallVector<Value> pointers;
      for (unsigned port = 0; port < P; ++port) {
        unsigned k = pf.reqOf[port][f];
        auto &fo = foldOut[k];
        writes.push_back({fo.push, fo.data});
        // The output's write pointer: past its reset token for an early one.
        auto wpNext = backedges.get(b.getIntegerType(rw));
        Value wp = resetReg(wpNext, consumers[k].name + "_wp", outputEarly[k] ? 1 : 0);
        wpNext.setValue(comb::MuxOp::create(
            b, fo.push, comb::AddOp::create(b, wp, constant(rw, 1)), wp));
        pointers.push_back(wp);
      }
      Value got = memory("ports_" + field.name, field.width, writes, pointers);
      if (outputEarly[pf.reqOf[0][f]])
        got = comb::MuxOp::create(b, firstSel, constant(field.width, 0), got);
      fieldBits.push_back(got);
      if (int(f) == resetField) {
        SmallVector<std::pair<Value, Value>> resets;
        for (auto &w : writes)
          resets.push_back({w.first, frameInput[pf.reset]});
        resetBits = memory("ports_treset", 1, resets, pointers);
      }
    }
    reqValidOut = any;
    reqChanOut = chan;
    reqTresetOut = resetBits;
    reqBitsOut = b.createOrFold<comb::ConcatOp>(
        SmallVector<Value>(fieldBits.rbegin(), fieldBits.rend()));
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
  // The next position is taken as the frame going on (at a frame's end, the
  // next frame's first): with no next frame no position runs, so its wait is
  // only a bubble more, and the readiness does not chain after the frame's
  // go (every output's room) into the position's next.
  if (virtualReady) {
    Value end = comb::OrOp::create(b, eq(posD, ii - 1), eq(posD, ii));
    Value next = comb::MuxOp::create(
        b, end, constant(posWidth, 0),
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
          // Staged, the read data is used at the evaluation's read position,
          // the stages before its state pass.
          int s = cls.slot[pass][i] - (pass == kState ? int(cls.stages) : 0);
          unsigned at = unsigned(((s % int(ii)) + int(ii)) % int(ii));
          useRow[at] = 1;
          instRow[at] = i;
        }
      Value use = table(b, next, useRow, 1);
      Value inst = threads > 1 ? table(b, next, instRow, idxWidth) : Value();
      Value classInst = inst;
      // A page word off chip is waited for by the state pass alone (staged,
      // by its evaluation's read position, the stages before it, where the
      // fetch buffer is read). With a window, by the fetch from it, a lead
      // before.
      Value stateUse, pageInst = inst;
      if (cls.pagesOffChip && !cls.pageWindow) {
        // (The instance a state pass loads is the slot's: instRow names it
        // at every pass, staged at its read position.)
        SmallVector<uint64_t> stateRow(ii + 1, 0);
        for (unsigned i = 0; i < threads; ++i) {
          int r = cls.slot[kState][i] - int(cls.stages);
          stateRow[unsigned(((r % int(ii)) + int(ii)) % int(ii))] = 1;
        }
        stateUse = table(b, next, stateRow, 1);
      } else if (cls.pagesOffChip) {
        SmallVector<uint64_t> fetchRow(ii + 1, 0), fetchInst(ii + 1, 0);
        for (unsigned i = 0; i < threads; ++i) {
          int f = cls.slot[kState][i] - int(cls.pageLead()) - int(cls.stages);
          unsigned pos = unsigned(((f % int(ii)) + int(ii)) % int(ii));
          fetchRow[pos] = 1;
          fetchInst[pos] = i;
        }
        stateUse = table(b, next, fetchRow, 1);
        pageInst = table(b, next, fetchInst, idxWidth);
      }
      for (auto &vm : cls.virtualMemories) {
        Value used = vm.pages ? stateUse : use;
        Value pending = copies[c].getResult(vm.firstOutput + 8);
        Value bit = pending;
        Value inst = vm.pages ? pageInst : classInst;
        if (inst) {
          unsigned entries = 1u << idxWidth;
          Value shift = b.createOrFold<comb::ConcatOp>(
              constant(entries - idxWidth, 0), inst);
          bit = b.createOrFold<comb::ExtractOp>(
              i1, comb::ShrUOp::create(b, pending, shift), 0);
        }
        terms.push_back(comb::OrOp::create(b, notOf(used), notOf(bit)));
      }
    }
    for (auto &v : simVirtuals) {
      // (An HBM link stream's page writes are at positions too, one a
      // position: no room past the decisions' in flight. Before they were,
      // its `pages` writes after a pass needed that many more, which a
      // stream of more pages than the controller's queue could never get.)
      terms.push_back(comb::ICmpOp::create(
          b, comb::ICmpPredicate::uge, body->getArgument(v.firstInput + 3),
          constant(kVirtualFreeWidth,
                   virtualRoom(!v.stream && !v.engine &&
                               classes[v.cls].virtualMemories[v.k].delayed))));
    }
    // An HBM link stream's landing at the position: the next slots to land
    // (as many as the landings decided and not yet run) answered.
    for (auto [k, st] : llvm::enumerate(linkStreams)) {
      if (!st.kept)
        continue;
      auto &sb = streamFor(k);
      unsigned ib = st.instBits(), slots = st.slots();
      SmallVector<Value> clear;
      for (unsigned d = 0; d <= unsigned(decisionLead()) + 2; ++d) {
        Value at = comb::AddOp::create(b, sb.popPtr, constant(ib, d));
        Value bit = b.createOrFold<comb::ExtractOp>(
            i1,
            comb::ShrUOp::create(
                b, sb.pending,
                b.createOrFold<comb::ConcatOp>(constant(slots - ib, 0), at)),
            0);
        clear.push_back(notOf(bit));
      }
      terms.push_back(comb::OrOp::create(
          b, notOf(table(b, next, st.popValid, 1)),
          comb::AndOp::create(b, clear, false)));
    }
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
  if (rspReadyOut)
    results.push_back(rspReadyOut);
  results.append(portOutputs.begin(), portOutputs.end());
  if (pf.on) {
    results.push_back(reqValidOut);
    results.push_back(reqChanOut);
    results.push_back(reqTresetOut);
    results.push_back(reqBitsOut);
  }
  for (auto &v : simVirtuals) {
    if (v.stream) {
      auto &outs = streamFor(v.k).outs;
      results.append(outs.begin(), outs.end());
      continue;
    }
    if (v.engine) {
      // A page engine's requests: its members', at most one read and one
      // write a host cycle (their positions differ), each tagged
      // {member, instance} and at the row {member, instance, row}.
      auto &eng = pageEngines[v.k];
      unsigned dw = eng.dataWidth;
      auto zext = [&](Value x, unsigned w) -> Value {
        unsigned have = hw::getBitWidth(x.getType());
        return have == w ? x
                         : b.createOrFold<comb::ConcatOp>(constant(w - have, 0), x);
      };
      SmallVector<Value> rdValid, rdInst, rdRow, wrValid, wrInst, wrRow, wrData, wrMask;
      for (auto [m, member] : llvm::enumerate(eng.members)) {
        auto [c, k] = member;
        auto &vm = classes[c].virtualMemories[k];
        auto out = [&](unsigned j) { return copies[c].getResult(vm.firstOutput + j); };
        auto tagOf = [&](Value inst) -> Value {
          return b.createOrFold<comb::ConcatOp>(
              constant(eng.memberBits(), m), zext(inst, eng.instWidth));
        };
        auto when = [&](Value valid, Value x) -> Value {
          return comb::MuxOp::create(b, valid, x,
                                     constant(hw::getBitWidth(x.getType()), 0));
        };
        Value rv = out(0), wv = out(3);
        rdValid.push_back(rv);
        wrValid.push_back(wv);
        auto rowOf = [&](Value inst, Value row) -> Value {
          return b.createOrFold<comb::ConcatOp>(
              constant(eng.memberBits(), m),
              zext(b.createOrFold<comb::ConcatOp>(inst, row), eng.addrWidth));
        };
        rdInst.push_back(when(rv, b.createOrFold<comb::ConcatOp>(
                                      tagOf(out(1)), constant(1, 0))));
        rdRow.push_back(when(rv, rowOf(out(1), out(2))));
        wrInst.push_back(when(wv, b.createOrFold<comb::ConcatOp>(
                                      tagOf(out(4)), constant(1, 0))));
        wrRow.push_back(when(wv, rowOf(out(4), out(5))));
        wrData.push_back(when(wv, zext(out(6), dw)));
        wrMask.push_back(when(wv, zext(out(7), dw / 8)));
      }
      // Its link streams': a read's member tag its slot, a write's the top
      // bit (the stream's own {tag, instance} without the instance bit); its
      // rows its own; a whole row's bytes written.
      for (auto [x, k] : llvm::enumerate(eng.streams)) {
        auto &st = linkStreams[k];
        unsigned m = eng.members.size() + x;
        auto &outs = streamFor(k).outs;
        auto tagOf = [&](Value portInst) -> Value {
          Value t = b.createOrFold<comb::ExtractOp>(
              b.getIntegerType(st.tagBits()), portInst, 1);
          return b.createOrFold<comb::ConcatOp>(constant(eng.memberBits(), m),
                                                zext(t, eng.instWidth));
        };
        auto when = [&](Value valid, Value x) -> Value {
          return comb::MuxOp::create(b, valid, x,
                                     constant(hw::getBitWidth(x.getType()), 0));
        };
        auto rowOf = [&](Value row) -> Value {
          return b.createOrFold<comb::ConcatOp>(constant(eng.memberBits(), m),
                                                zext(row, eng.addrWidth));
        };
        Value rv = outs[0], wv = outs[3];
        rdValid.push_back(rv);
        wrValid.push_back(wv);
        rdInst.push_back(when(rv, b.createOrFold<comb::ConcatOp>(tagOf(outs[1]),
                                                                  constant(1, 0))));
        rdRow.push_back(when(rv, rowOf(outs[2])));
        wrInst.push_back(when(wv, b.createOrFold<comb::ConcatOp>(tagOf(outs[4]),
                                                                  constant(1, 0))));
        wrRow.push_back(when(wv, rowOf(outs[5])));
        wrData.push_back(when(wv, zext(outs[6], dw)));
        wrMask.push_back(when(wv, zext(constant((st.dw + 7) / 8, ~0ull), dw / 8)));
      }
      auto any = [&](SmallVector<Value> &xs) -> Value {
        return xs.size() == 1 ? xs[0] : Value(comb::OrOp::create(b, xs, false));
      };
      results.append({any(rdValid), any(rdInst), any(rdRow), any(wrValid),
                      any(wrInst), any(wrRow), any(wrData), any(wrMask)});
      continue;
    }
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
      if (v.engine) {
        auto &eng = pageEngines[v.k];
        virtuals.push_back(b.getStringAttr(
            "page_engine" + Twine(v.k) + ":__pages" + Twine(v.k) + ":" +
            Twine(eng.tagBits() + 1) + ":" + Twine(eng.rowBits()) + ":" +
            Twine(eng.dataWidth) + ":" + Twine(eng.dataWidth / 8) + ":" +
            Twine(eng.readFirst ? 1 : 0) + ":1:" + Twine(eng.tagBits())));
        continue;
      }
      if (v.stream) {
        auto &st = linkStreams[v.k];
        virtuals.push_back(b.getStringAttr(
            "link_" + modelOf(st.pcls) + "_" + modelOf(st.rcls) + ":__link" +
            Twine(v.k) + ":" + Twine(st.portInstBits()) + ":" +
            Twine(streamRowBits(st)) + ":" + Twine(st.dw) + ":1:0:1:" +
            Twine(st.tagBits())));
        continue;
      }
      auto &cls = classes[v.cls];
      auto &vm = cls.virtualMemories[v.k];
      unsigned instWidth =
          vm.instWidth ? vm.instWidth
                       : std::max(cls.threads() > 1 ? llvm::Log2_32_Ceil(cls.threads()) : 0u,
                                  1u);
      virtuals.push_back(b.getStringAttr(
          vm.name + ":" + cls.name + ":" + Twine(instWidth) + ":" +
          Twine(vm.rowWidth) + ":" + Twine(vm.dataWidth) + ":" +
          Twine(vm.maskWidth) + ":" + Twine(vm.readFirst ? 1 : 0) + ":" +
          Twine(vm.instances ? vm.instances : cls.threads()) +
          // A page column (its read and write of a host cycle may be of
          // two instances: never in a controller group, whose FIFO entry
          // keeps one instance for both).
          (vm.pages ? ":0:pages" : "")));
    }
    sim->setAttr("foldsim.vmems", b.getArrayAttr(virtuals));
  }
  sim->setAttr("foldsim.frame", b.getI64IntegerAttr(ii));
  // Folded port bundles, for the kernel's header: the record layouts
  // ("REQ:<field>:<lo>:<width>", "RSP:...") after "ports:<n>:<chan bits>".
  if (portFold.on) {
    SmallVector<Attribute> fields{b.getStringAttr(
        ("ports:" + Twine(portFold.ports) + ":" + Twine(portFold.chanWidth)).str())};
    for (auto &f : portFold.req)
      fields.push_back(b.getStringAttr(
          ("REQ:" + f.name + ":" + Twine(f.lo) + ":" + Twine(f.width)).str()));
    for (auto &f : portFold.rsp)
      fields.push_back(b.getStringAttr(
          ("RSP:" + f.name + ":" + Twine(f.lo) + ":" + Twine(f.width)).str()));
    sim->setAttr("foldsim.ports", b.getArrayAttr(fields));
  }
  // Multicycle copies, for the flow's timing constraints: the registers'
  // name prefix (fsmc_<class>__, then cap<k> or end<k>), the host cycles
  // from capture to end, and the bits of each.
  SmallVector<Attribute> multicycles;
  for (auto &cls : classes)
    if (cls.multicycle)
      multicycles.push_back(b.getStringAttr(
          "fsmc_" + cls.name + "__:" + Twine(cls.multicycle) + ":" +
          Twine(cls.multicycleCaps) + ":" + Twine(cls.multicycleEnds)));
  if (!multicycles.empty())
    sim->setAttr("foldsim.multicycle", b.getArrayAttr(multicycles));
  // Pass-split copies: each pass's paths (captures and pass counters to the
  // holds, holds to the end registers), its span of host cycles.
  SmallVector<Attribute> splits;
  for (auto &cls : classes)
    if (cls.passSplit && cls.passHoldBits)
      splits.push_back(b.getStringAttr(
          "fsps_" + cls.name + "__:" + Twine(cls.passSpan) + ":" +
          Twine(cls.passSplit) + ":" + Twine(cls.passHoldBits)));
  if (!splits.empty())
    sim->setAttr("foldsim.pass_split", b.getArrayAttr(splits));
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

LogicalResult StaticFolder::resolvePortFold() {
  auto &pf = portFold;
  pf.consumerPort.assign(consumers.size(), -1);
  pf.consumerField.assign(consumers.size(), -1);
  pf.producerPort.assign(producers.size(), -1);
  pf.producerField.assign(producers.size(), -1);
  if (foldPortsSpec.empty())
    return success();
  // A spec without a group is a prefix: <prefix><bundle>_<field>.
  std::string spec = StringRef(foldPortsSpec).contains('(')
                         ? foldPortsSpec
                         : llvm::Regex::escape(foldPortsSpec) + "([0-9]+)_(.*)";
  llvm::Regex re("^(" + spec + ")$");
  std::string error;
  if (!re.isValid(error))
    return netlist.emitError("fold-ports: ") << error;
  // (The regex is wrapped in a group of its own: its captures are 2 and 3.)
  auto split = [&](StringRef name, unsigned &port, std::string &field) {
    SmallVector<StringRef> m;
    if (!re.match(name, &m) || m.size() < 4)
      return false;
    if (m[2].getAsInteger(10, port))
      return false;
    field = m[3].str();
    return true;
  };
  // Each bundle's fields, in netlist order (the consumers' and producers'
  // order of the netlist's ports).
  std::map<unsigned, SmallVector<std::pair<std::string, unsigned>>> outs, ins;
  for (auto [k, consumer] : llvm::enumerate(consumers)) {
    unsigned port;
    std::string field;
    if (consumer.cls >= 0 || !split(consumer.name, port, field))
      continue;
    if (envSink[k])
      return netlist.emitError("fold-ports: '") << consumer.name
             << "' is an env-sinks output";
    outs[port].push_back({field, unsigned(k)});
  }
  for (auto [p, producer] : llvm::enumerate(producers)) {
    unsigned port;
    std::string field;
    if (producer.cls >= 0 || !split(producer.name, port, field))
      continue;
    ins[port].push_back({field, unsigned(p)});
  }
  if (outs.empty())
    return netlist.emitError("fold-ports: no netlist output matches '")
           << foldPortsSpec << "'";
  pf.ports = outs.size();
  if (outs.rbegin()->first + 1 != pf.ports || ins.size() != (ins.empty() ? 0 : pf.ports) ||
      (!ins.empty() && ins.rbegin()->first + 1 != pf.ports))
    return netlist.emitError("fold-ports: the bundles are not numbered 0 to ")
           << pf.ports - 1 << " with outputs and inputs in each";
  pf.chanWidth = std::max(1u, llvm::Log2_32_Ceil(pf.ports));
  auto tokenWidth = [&](unsigned p) {
    return producers[p].width * (specOf[p] >= 0 ? specValues() : 1);
  };
  for (auto &[name, k] : outs.begin()->second) {
    pf.req.push_back({name, consumers[k].width, pf.reqWidth});
    pf.reqWidth += consumers[k].width;
  }
  if (!ins.empty())
    for (auto &[name, p] : ins.begin()->second) {
      pf.rsp.push_back({name, tokenWidth(p), pf.rspWidth});
      pf.rspWidth += tokenWidth(p);
    }
  pf.reqOf.assign(pf.ports, {});
  pf.rspOf.assign(pf.ports, {});
  for (unsigned port = 0; port < pf.ports; ++port) {
    auto &o = outs[port];
    if (o.size() != pf.req.size())
      return netlist.emitError("fold-ports: bundle ") << port << " has "
             << o.size() << " outputs, bundle 0 " << pf.req.size();
    for (auto [f, entry] : llvm::enumerate(o)) {
      auto &[name, k] = entry;
      if (name != pf.req[f].name || consumers[k].width != pf.req[f].width)
        return netlist.emitError("fold-ports: output '") << consumers[k].name
               << "' is not field " << f << " ('" << pf.req[f].name << "', "
               << pf.req[f].width << " bits) of bundle 0";
      pf.reqOf[port].push_back(k);
      pf.consumerPort[k] = port;
      pf.consumerField[k] = f;
    }
    if (ins.empty())
      continue;
    auto &i = ins[port];
    if (i.size() != pf.rsp.size())
      return netlist.emitError("fold-ports: bundle ") << port << " has "
             << i.size() << " inputs, bundle 0 " << pf.rsp.size();
    for (auto [f, entry] : llvm::enumerate(i)) {
      auto &[name, p] = entry;
      if (name != pf.rsp[f].name || tokenWidth(p) != pf.rsp[f].width)
        return netlist.emitError("fold-ports: input '") << producers[p].name
               << "' is not field " << f << " ('" << pf.rsp[f].name << "', "
               << pf.rsp[f].width << " bits) of bundle 0";
      pf.rspOf[port].push_back(p);
      pf.producerPort[p] = port;
      pf.producerField[p] = f;
    }
  }
  for (auto [p, producer] : llvm::enumerate(producers))
    if (producer.cls < 0 && producer.name == foldPortsReset)
      pf.reset = p;
  if (pf.reset < 0 || pf.producerPort[pf.reset] >= 0 || producers[pf.reset].width != 1)
    return netlist.emitError("fold-ports-reset: no one-bit netlist input '")
           << foldPortsReset << "' outside the bundles";
  pf.on = true;
  return success();
}

void StaticFolder::staggerPortPushes() {
  if (!portFold.on)
    return;
  for (unsigned f = 0; f < portFold.req.size(); ++f) {
    int prev = std::numeric_limits<int>::min();
    for (unsigned port = 0; port < portFold.ports; ++port) {
      int &at = outputSlot[portFold.reqOf[port][f]];
      at = std::max(at, prev + 1);
      prev = at;
    }
  }
}

LogicalResult StaticFolder::run() {
  inlineNetlists();
  if (failed(collect()) || failed(resolveEnvironment()) || failed(resolvePortFold()) ||
      failed(parseRegisteredLinks()) || failed(parseSerialize()) ||
      failed(parseLinkKinds()))
    return failure();
  for (auto &cls : classes)
    if (cls.threads() > 1 &&
        failed(foldPipelines(cls.model, cls.threads(), cls.folding)))
      return failure();
  for (auto &cls : classes)
    if (failed(analyze(cls)))
      return failure();
  // The HBM link streams' layouts, once the analysis has told which
  // producers are late (a late output is no stream's: its readers take it
  // in the frame it is written).
  layoutStreams();
  // The links' storage alone, before any schedule (it does not depend on
  // one): for a quick look at a large design.
  if (getenv("FOLDSIM_LINKS_ONLY")) {
    reportLinks(llvm::errs());
    for (auto &cls : classes)
      if (cls.folding && cls.folding->edges)
        cls.folding->edges->abandon();
    return failure();
  }
  // Without a schedule, or with one only, nothing is built: the folded
  // pipelines' edges stay unresolved.
  auto abandonFolds = [&] {
    for (auto &cls : classes)
      if (cls.folding && cls.folding->edges)
        cls.folding->edges->abandon();
  };
  if (failed(parsePages()) || failed(parseStorage()) || failed(pagesOffChip()) ||
      failed(chooseStages()) || failed(planPassSplits()) ||
      failed(schedule())) {
    abandonFolds();
    return failure();
  }
  chooseWindows();
  choosePageGates();
  if (scheduleOnly) {
    // The page engines the build would take (and the link streams they
    // carry), in the report too: a cost model prices a schedule as built.
    if (pageEngineCap) {
      if ((!linkStreams.empty() && failed(planStreams())) || failed(assignPageEngines())) {
        abandonFolds();
        return failure();
      }
      std::string text;
      llvm::raw_string_ostream os(text);
      reportPageEngines(os);
      llvm::errs() << os.str();
      if (!reportPath.empty() && !text.empty()) {
        std::error_code ec;
        llvm::raw_fd_ostream file(reportPath, ec, llvm::sys::fs::OF_Append);
        if (ec) {
          abandonFolds();
          return netlist.emitError("report: ") << ec.message();
        }
        file << text;
      }
    }
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
                      shortLead, bubbles, virtualMemories, vmemMergeLanes,
                      vmemSkipHeldReads, stagedEnv,
                      specGroups, specDepth,
                      envSinks, registerStateEntries, slrGroups, slrSites,
                      slrCapacity, slrCapacities, slrAssign, sllEstimate,
                      slrSllBudget, storage, readyLevels,
                      forwardAfterLevels, shareLinkWrites, zeroChainHops,
                      page, pageMasks, pageHbmColumnBits, pageHbmWindow,
                      pageEngines, pageMemoryBits, pageMemories, pageRows, compactLinks,
                      registeredLinks, shareLinks, stagedLinks, linkBlockDepth,
                      linkUltraDepth, linkKinds, stageLevels, multicycle);
  folder.moduloLinear = moduloLinear;
  folder.passSplitSpecs.assign(passSplit.begin(), passSplit.end());
  folder.linkHbmLead = std::max(1u, unsigned(linkHbmLead));
  folder.linkHbmMinLeadOpt = linkHbmMinLead;
  folder.linkHbmShift = linkHbmShift;
  folder.foldPortsSpec = foldPorts;
  folder.foldPortsReset = foldPortsReset;
  folder.linkHbmFitSlots = linkHbmFitSlots;
  folder.pageWriteGate = pageWriteGate;
  folder.linkHbmRowBits = linkHbmRowBits;
  folder.minII = minII;
  folder.slrSerializeSpecs.assign(slrSerialize.begin(), slrSerialize.end());
  folder.readyChoice = readyChoice;
  folder.stagedLinkMemories = stagedLinks && stagedLinkMemories;
  if (failed(folder.run()))
    return signalPassFailure();
}
