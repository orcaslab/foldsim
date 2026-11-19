//===- FAME5.cpp - FAME-5 multithreading of a flat model ------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Channelizes a model (FAME-1: every port a token channel, an output that
// depends on no input fired eagerly) and multithreads it (FAME-5: one copy
// holds the state of every instance, each register a memory with a word per
// instance, and simulates one instance per host cycle), on HW/Comb/Seq IR.
//
//===----------------------------------------------------------------------===//

#include "FAME5.h"
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
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <numeric>
#include <optional>

namespace foldsim {
#define GEN_PASS_DEF_FAME5
#include "foldsim/Passes.h.inc"
} // namespace foldsim

#define DEBUG_TYPE "foldsim-fame5"

using namespace mlir;
using namespace circt;

namespace {

struct FAME5Pass : public foldsim::impl::FAME5Base<FAME5Pass> {
  using FAME5Base::FAME5Base;
  void runOnOperation() override;
};

/// Sets of inputs, each stored once: a model's bits depend on few distinct
/// sets, and a set per bit of every value does not fit in memory for a large
/// model. Set 0 is the empty set.
class DepSets {
public:
  explicit DepSets(unsigned numInputs) : numInputs(numInputs) {
    intern(llvm::BitVector(numInputs));
  }
  /// The set of only `input`.
  unsigned single(unsigned input) {
    llvm::BitVector set(numInputs);
    set.set(input);
    return intern(std::move(set));
  }
  /// The union of two sets.
  unsigned unite(unsigned a, unsigned b) {
    if (a == b || b == 0)
      return a;
    if (a == 0)
      return b;
    auto key = std::make_pair(std::min(a, b), std::max(a, b));
    if (auto it = unions.find(key); it != unions.end())
      return it->second;
    llvm::BitVector set = sets[a];
    set |= sets[b];
    return unions[key] = intern(std::move(set));
  }
  const llvm::BitVector &get(unsigned id) const { return sets[id]; }

private:
  unsigned intern(llvm::BitVector set) {
    auto words = set.getData();
    auto hash = llvm::hash_combine_range(words.begin(), words.end());
    auto &ids = byHash[hash];
    for (auto id : ids)
      if (sets[id] == set)
        return id;
    ids.push_back(sets.size());
    sets.push_back(std::move(set));
    return sets.size() - 1;
  }

  unsigned numInputs;
  std::vector<llvm::BitVector> sets;
  DenseMap<llvm::hash_code, SmallVector<unsigned, 1>> byHash;
  DenseMap<std::pair<unsigned, unsigned>, unsigned> unions;
};

/// The inputs each bit of a value depends on, as sets of DepSets.
using BitDeps = SmallVector<unsigned>;

/// A target port of the model and the channel it becomes.
struct Channel {
  StringAttr name;
  /// A channel of several ports carries them as fields, the first one in the
  /// high bits of its type, each on a bits port `<name>_{sink,source}_<field>`.
  struct Field {
    StringAttr name;
    unsigned argNum;
    Type type;
  };
  SmallVector<Field> fields;
  Type type;
  /// For an input, its block argument in the model; for an output, its index.
  unsigned argNum = 0;
  /// The bits of the port the channel carries, if only some.
  bool sliced = false;
  unsigned low = 0;
  /// For an output, the value the model drives it with.
  Value current;
  /// The inputs an output depends on combinationally.
  llvm::BitVector deps;
  bool eager = false;
  Value next;
};

/// Builds the threaded module of one model.
class Threader {
public:
  using Slices = llvm::StringMap<SmallVector<std::pair<unsigned, unsigned>>>;
  /// The ports that are fields of a channel: port -> (channel, field, rank).
  using Fields =
      llvm::StringMap<std::tuple<std::string, std::string, unsigned>>;

  /// Without a `gate`, the target state is written on the host clock, with
  /// the target clock's enable.
  Threader(hw::HWModuleOp model, unsigned threads, StringAttr name,
           hw::HWModuleExternOp gate, const Slices &slices,
           const Fields &fields, StringRef clockChannel, bool threadIndexPort,
           bool tickFromFirst = false, bool readySchedule = false,
           const foldsim::ThreadOptions &options = {})
      : model(model), threads(threads), name(name), gate(gate), slices(slices),
        fields(fields), clockChannel(clockChannel),
        threadIndexPort(threadIndexPort), tickFromFirst(tickFromFirst),
        readySchedule(readySchedule), regionLoads(options.regionLoads),
        registeredState(options.registeredState),
        stateStages(options.stateStages), commitSelect(options.commitSelect),
        stagedRamWrites(options.stagedRamWrites),
        pipelineMultipliers(options.pipelineMultipliers),
        pipelineDepth(options.pipelineDepth),
        memoryLoads(options.memoryLoads),
        readPortDepth(options.readPortDepth),
        unreverseAddresses(options.unreverseAddresses),
        compactStaging(options.compactStaging),
        foldPipelines(options.foldPipelines),
        blockStateDepth(options.registeredState ? options.blockStateDepth : 0),
        externalSchedule(options.externalSchedule),
        startGiven(options.externalSchedule || options.localSchedule),
        scheduleStart(options.scheduleStart),
        phases(threads == 1
                   ? (options.commitEnable && options.registeredState &&
                              options.commitSelect && !readySchedule
                          ? 2
                          : 1)
               : options.idleSlots && options.registeredState && !readySchedule
                   ? turnPhases(threads, options.stateStages)
                   : 1) {}

  LogicalResult run();

private:
  LogicalResult checkModel();
  /// With `foldPipelines`: replaces each fixed-latency pipeline of the model
  /// by its target state (a slot counter, a store of results, the last layer
  /// if it stays) and an unthreaded copy on the host clock, whose host-side
  /// placeholders `finishFolds` connects once the threaded module exists.
  LogicalResult prepareFolds();
  LogicalResult finishFolds(ImplicitLocOpBuilder &b, Value edge,
                            Value hostReset);
  void computeDeps();
  const BitDeps &getDeps(Value value);
  BitDeps computeBitDeps(Value value);
  Value getNext(Value value, ImplicitLocOpBuilder &b);
  /// Finds the values computed from the target state, and those computed
  /// from a memory's read data.
  void computeStateDeps();
  bool dependsOnState(Value value) { return fromState.contains(value); }
  /// A memory of one entry per thread, read at the current thread; a register
  /// for a single thread, with the `preset` value if any.
  Value createThreadState(ImplicitLocOpBuilder &b, StringRef stateName,
                          Type type, Value &memory, IntegerAttr preset = {});
  void writeThreadState(ImplicitLocOpBuilder &b, Value memory, Value clock,
                        Value data, Value enable = {});
  /// How a register is written: never (it only holds its value), or with
  /// `data` when `enable` (always, if null).
  using Write = foldsim::HeldWrite;
  /// Moves everything on the target clock to the host clock, written only
  /// when `enable`.
  LogicalResult enableOnHostClock(ImplicitLocOpBuilder &b, Value targetClock,
                                  Value enable);

  /// With regions, the ports of the target's state and memories, whose
  /// thread index and write control come from their region's copies. `low` is
  /// a memory port's address without its thread bits; a write's `term` is its
  /// enable without the target edge, which writes on the `edge`, and with
  /// `preset` also while the threads are initialized. A write that is not
  /// `staged` keeps its enable and data and lands on the edge itself. A
  /// registered read of a memory whose writes are staged has its `enable` in
  /// the target, without the edge.
  struct RegionRead {
    seq::FirMemReadOp op;
    Value low;
    bool registered = false;
    Value enable;
  };
  struct RegionWrite {
    seq::FirMemWriteOp op;
    Value low, term;
    bool edge, preset, staged = true;
  };
  struct RegionMemory {
    Value memory;
    SmallVector<RegionRead> reads;
    SmallVector<RegionWrite> writes;
    /// The thread state of a register, read ahead with registered state.
    bool state = false;
    /// Where its staging is located with compact staging: the register or
    /// memory it threads.
    std::optional<Location> loc;
  };
  /// With registered state, a register of a single thread: committed by
  /// select, its `candidate` taken on its enable (`wen` on the edge, and with
  /// `preset` while initialized). Both are registered as soon as the register
  /// is threaded: as operands, they follow the later replacement of what they
  /// read (another register, a memory's read data).
  struct SingleRegister {
    seq::FirRegOp reg;
    Value candidate, wen;
    bool preset;
  };
  SmallVector<SingleRegister> singleRegisters;
  /// Commits each single thread's register by select: the candidate and the
  /// enable are registered every host cycle and the state is the committed
  /// value or the candidate, by the registered enable.
  void commitBySelect(ImplicitLocOpBuilder &b, Value edge, Value initializing);
  RegionMemory &regionMemory(Value memory);
  /// Splits each recorded memory of more than `memoryLoads` loads by bits into
  /// memories of at most that many, in regions of their own.
  void splitWideMemories(ImplicitLocOpBuilder &b);
  /// Splits the recorded state into regions of at most `regionLoads` loads,
  /// in name order so that a region is a submodule's state, and gives each
  /// its control.
  void buildRegions(ImplicitLocOpBuilder &b, Value edge, Value initializing,
                    Value hostReset);

  hw::HWModuleOp model;
  unsigned threads;
  StringAttr name;
  hw::HWModuleExternOp gate;
  const Slices &slices;
  const Fields &fields;
  /// The channel the target clock comes in on, if any.
  StringRef clockChannel;
  bool threadIndexPort;
  /// Whether the target clock ticks from the first finish after reset, rather
  /// than from the second.
  bool tickFromFirst;
  /// Whether the next thread is one whose inputs are all there, by a
  /// `thread_ready` input, rather than the next in turn.
  bool readySchedule;
  /// The most loads a region's copy of the thread index or of the target edge
  /// drives; 0 for no regions.
  unsigned regionLoads;
  /// Whether the state is read a host cycle ahead into registers, the next
  /// thread chosen a cycle earlier (Phase 2), and a single thread's state
  /// committed by select.
  bool registeredState;
  /// With registered state and at least three threads: the host cycles from
  /// a thread's edge to its state's write, so that synthesis can retime the
  /// stages into the logic computing it (C-slow on the state's update).
  unsigned stateStages;
  /// Whether a single thread's registers commit by select.
  bool commitSelect;
  /// With regions: whether a memory with a registered read is written a
  /// host cycle after the edge too, its read forwarding the staged write
  /// where it reads new data under a write.
  bool stagedRamWrites;
  /// With state written two host cycles late: whether a state computed by
  /// a multiplier takes its first stage at its next value's inputs.
  bool pipelineMultipliers;
  /// With two state stages: the logic levels of a state cone above which it
  /// is split over the two host cycles, or 0.
  unsigned pipelineDepth;
  /// With regions: the loads above which a memory is split by bits, or 0.
  unsigned memoryLoads;
  /// The levels of a registered read's address or enable above which its
  /// cone on registers alone is computed a host cycle ahead, or 0.
  unsigned readPortDepth;
  /// Whether memories addressed by every port as `~x` are addressed by `x`.
  bool unreverseAddresses;
  /// With regions: whether staging takes the location of the state it
  /// stages, and a staged write enable has no preset value.
  bool compactStaging;
  /// Whether fixed-latency pipelines are simulated by an unthreaded copy and
  /// a per-thread store of their results rather than threaded.
  bool foldPipelines;
  /// Thread state of at least this many threads asks for block RAM.
  unsigned blockStateDepth;
  /// Whether the turns come in on `schedule_next`, and the position on the
  /// host cycle after reset.
  bool externalSchedule;
  /// Whether the position on the host cycle after reset is `scheduleStart`,
  /// the model's turns brought from outside or counted from it.
  bool startGiven;
  uint64_t scheduleStart;
  /// The bits of a position: the thread index's, then the phase's.
  unsigned positionWidth() const {
    return (threads > 1 ? llvm::Log2_32_Ceil(threads) : 0) +
           (phases > 1 ? llvm::Log2_32(phases) : 0);
  }
  /// A folded pipeline's store and its write from the unthreaded copy.
  struct Fold {
    seq::FirMemOp store;
    Value threadedStore;
    Value slot, thread, valid, data;
  };
  /// With `idleSlots`: the host cycles of each thread's turn, the first its
  /// own and the others idle, so that the threads' turns are at least as many
  /// host cycles as a thread's write takes to land and one more.
  unsigned phases;
  static unsigned turnPhases(unsigned threads, unsigned stateStages) {
    return foldsim::turnPhases(threads, stateStages);
  }
  /// With idle slots, the host cycle of the current thread's turn, and that
  /// of the next host cycle; and whether this one is idle.
  Value phase, aheadPhase, idle;
  /// The reset value of `aheadIdx`: the thread of the host cycle after the
  /// first; and of `threadIdx`.
  uint64_t aheadReset = 1, idxReset = 0, aheadPhaseReset = 1;
  SmallVector<Fold> folds;
  DenseMap<Operation *, unsigned> foldOfStore;
  /// The ops of the unthreaded copies: on the host clock, not target state.
  DenseSet<Operation *> physical;
  /// In a model of one thread, the black boxes with a pipeline contract: they
  /// stay instances, on the host clock and enabled on the target's edges.
  SetVector<Operation *> keptBoxes;
  std::unique_ptr<BackedgeBuilder> foldEdges;
  Backedge physClock, foldThread, foldEdge, foldReset;
  /// Whether the scheduler built never makes a thread that finishes current
  /// on the next host cycle, which folding relies on. Each scheduler that
  /// keeps it says so.
  bool separatesFinishes = false;
  /// The combinational cone of a next value, within the model.
  static bool inCone(Operation *op) {
    if (!op || op->getNumRegions() || isa<hw::InstanceOp, hw::OutputOp>(op))
      return false;
    auto ns = op->getDialect()->getNamespace();
    return ns == "comb" || ns == "hw";
  }
  bool coneHasMultiplier(Value value);
  /// The logic levels of `op` as synthesized, roughly: wiring takes none.
  static unsigned levelsOf(Operation *op) {
    if (isa<hw::ConstantOp, hw::AggregateConstantOp, hw::BitcastOp,
            comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp,
            hw::ArrayCreateOp, hw::ArrayConcatOp, hw::StructCreateOp,
            hw::StructExtractOp>(op))
      return 0;
    return isa<comb::MulOp>(op) ? 4 : 1;
  }
  /// The logic levels of each op of a cone, from its inputs, memoized.
  DenseMap<Operation *, unsigned> coneLevelsOf;
  unsigned coneLevels(Value value);
  Value pipelinedCone(ImplicitLocOpBuilder &b, Value value);
  /// The logic levels of `op` mapped to LUTs, roughly, for `readPortDepth`:
  /// a carry chain by its width, a comparison or a wide gate by its LUT
  /// tree, a LUT RAM read by its depth.
  static unsigned mappedLevelsOf(Operation *op);
  /// With `readPortDepth`: computes the register-only part of each deep
  /// registered read's address and enable a host cycle ahead.
  void pipelineReadPorts(ImplicitLocOpBuilder &b, hw::HWModuleOp threaded);
  /// With `unreverseAddresses`: addresses the model's memories by `x` where
  /// each of their ports takes `~x` or a constant.
  void unreverseMemoryAddresses();
  /// The cone inputs registered for `pipelineMultipliers` and
  /// `pipelineDepth`, and the values computed from them on the second host
  /// cycle, each shared by cones: a value is the same whichever cut computes
  /// it.
  DenseMap<Value, Value> coneInputs, coneLate;
  bool deepState() const {
    return registeredState && stateStages > 1 && threads * phases >= 3;
  }
  /// Whether the state is split into regions with their own control.
  bool hasRegions() const {
    return (regionLoads || registeredState) && threads > 1 && !gate;
  }
  /// With registered state, the thread after the current one.
  Value aheadIdx;
  SmallVector<RegionMemory> regionMemories;
  /// With registered state, each state read whose uses its read-ahead
  /// register took.
  DenseMap<Value, Value> readAhead;
  DenseMap<Value, unsigned> regionMemoryIndex;
  /// A single thread's state: the registers not written yet, with their next
  /// value and clock to be.
  DenseMap<Value, std::pair<Backedge, Backedge>> unwritten;
  std::unique_ptr<BackedgeBuilder> registers1;

  unsigned clockArg = 0;
  SmallVector<Channel> inputs, outputs;
  /// The outputs that are inputs wired through, as `output = input`.
  SmallVector<Attribute> bypasses;
  SmallVector<seq::FirRegOp> registers;
  SmallVector<seq::FirMemOp> memories;
  /// Whether `value` is computed from a memory's read data, other than
  /// through a register. The next value of such an output is not computed, so
  /// it is not fired eagerly.
  bool dependsOnMemory(Value value) { return fromMemory.contains(value); }
  /// For each bit of an input port, the channel carrying it.
  DenseMap<Value, SmallVector<unsigned>> inputOfArg;
  /// The channels of fields, by name, as their index plus one.
  llvm::StringMap<unsigned> inputBundles, outputBundles;
  std::map<std::pair<unsigned, bool>, SmallVector<unsigned>> fieldRanks;
  DenseMap<Value, std::unique_ptr<BitDeps>> deps;
  /// The union of a value's bits' sets, once they are final.
  DenseMap<Value, unsigned> allDeps;
  /// The values whose sets are not final yet, each with its index in the
  /// order they were reached and the lowest index of the values it reaches
  /// that are not final: a combinational cycle is a strongly connected
  /// component of them, whose sets are iterated until they do not change.
  DenseMap<Value, std::pair<unsigned, unsigned>> depsOpen;
  SmallVector<Value> depsStack, depsCalls;
  DenseSet<Value> depsSelfCycles;
  unsigned depsReached = 0;
  std::unique_ptr<DepSets> depSets;
  DenseMap<Value, Value> nexts;
  /// The next values of a combinational cycle, used before they are built.
  DenseSet<Value> nextsInProgress;
  DenseMap<Value, Backedge> nextsAhead;
  std::unique_ptr<BackedgeBuilder> nextCycles;
  DenseSet<Value> fromState, fromMemory;
  Value threadIdx, hostClock;
};

} // namespace

/// The input whose bits `value` is, in order, wired through the ops that only
/// move bits around, or null.
static BlockArgument getWiredInput(Value value) {
  auto width = hw::getBitWidth(value.getType());
  // Where bit `bit` of `v` comes from, if from an input.
  std::function<std::pair<BlockArgument, int64_t>(Value, int64_t)> source =
      [&](Value v, int64_t bit) -> std::pair<BlockArgument, int64_t> {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return {arg, bit};
    auto *op = v.getDefiningOp();
    APInt index;
    if (isa_and_nonnull<hw::BitcastOp>(op))
      return source(op->getOperand(0), bit);
    if (auto extract = dyn_cast_or_null<comb::ExtractOp>(op))
      return source(extract.getInput(), extract.getLowBit() + bit);
    if (isa_and_nonnull<comb::ConcatOp, hw::ArrayCreateOp, hw::StructCreateOp>(
            op)) {
      // Operands are listed from the most significant one.
      for (auto operand : llvm::reverse(op->getOperands())) {
        auto w = hw::getBitWidth(operand.getType());
        if (bit < w)
          return source(operand, bit);
        bit -= w;
      }
      return {};
    }
    if (auto get = dyn_cast_or_null<hw::ArrayGetOp>(op);
        get && matchPattern(get.getIndex(), m_ConstantInt(&index)))
      return source(get.getInput(),
                    index.getZExtValue() * hw::getBitWidth(get.getType()) +
                        bit);
    if (auto extract = dyn_cast_or_null<hw::StructExtractOp>(op)) {
      auto elements =
          hw::type_cast<hw::StructType>(extract.getInput().getType())
              .getElements();
      for (auto &element : elements.drop_front(extract.getFieldIndex() + 1))
        bit += hw::getBitWidth(element.type);
      return source(extract.getInput(), bit);
    }
    return {};
  };
  BlockArgument input;
  for (int64_t bit = 0; bit < width; ++bit) {
    auto [arg, argBit] = source(value, bit);
    if (!arg || argBit != bit || (input && arg != input))
      return {};
    input = arg;
  }
  if (!input || hw::getBitWidth(input.getType()) != width)
    return {};
  return input;
}

LogicalResult Threader::checkModel() {
  auto *body = model.getBodyBlock();
  bool hasClock = false;
  for (auto [i, port] : llvm::enumerate(model.getPortList())) {
    if (isa<hw::InOutType>(port.type))
      return model.emitError("inout port '") << port.name << "' in a model";
    if (isa<seq::ClockType>(port.type)) {
      if (port.dir != hw::ModulePort::Direction::Input || hasClock)
        return model.emitError("a model has a single clock input");
      hasClock = true;
      clockArg = port.argNum;
      continue;
    }
    bool isInput = port.dir == hw::ModulePort::Direction::Input;
    // A port of no bits is a channel of tokens only. An input of bits the model
    // does not read would only hold its threads back.
    auto width = hw::getBitWidth(port.type);
    if (width < 0)
      return model.emitError("port '") << port.name << "' has no bit width";
    if (auto field = fields.find(port.name.getValue()); field != fields.end()) {
      auto &[channelName, fieldName, rank] = field->second;
      auto &channels = isInput ? inputs : outputs;
      auto &index = (isInput ? inputBundles : outputBundles)[channelName];
      if (!index) {
        Channel channel;
        channel.name = StringAttr::get(model.getContext(), channelName);
        channels.push_back(channel);
        index = channels.size();
      }
      channels[index - 1].fields.push_back(
          {StringAttr::get(model.getContext(), fieldName),
           unsigned(port.argNum), port.type});
      fieldRanks[{index - 1, isInput}].push_back(rank);
      if (isInput)
        inputOfArg[body->getArgument(port.argNum)] =
            SmallVector<unsigned>(width, index - 1);
      continue;
    }
    if (isInput && width > 0 && body->getArgument(port.argNum).use_empty())
      continue;
    // An output that is an input, wired through, is not a channel of the
    // model: the input's tokens go to it around the model.
    if (!isInput && width > 0) {
      auto output = body->getTerminator()->getOperand(port.argNum);
      if (auto input = getWiredInput(output)) {
        bypasses.push_back(StringAttr::get(
            model.getContext(), port.name.getValue() + " = " +
                                    model.getInputName(input.getArgNumber())));
        continue;
      }
    }
    auto &channels = isInput ? inputs : outputs;
    SmallVector<unsigned> bitChannels(width, channels.size());

    // A port may be split into channels of some of its bits each. They must
    // cover an input exactly; an output's other bits are not exported.
    auto it = slices.find(port.name.getValue());
    if (it == slices.end()) {
      Channel channel;
      channel.name = port.name;
      channel.type = port.type;
      channel.argNum = port.argNum;
      channels.push_back(channel);
    } else {
      auto portSlices = it->second;
      llvm::sort(portSlices,
                 [](auto a, auto b) { return a.second < b.second; });
      unsigned next = 0;
      for (auto [hi, lo] : portSlices) {
        if (lo < next || (isInput && lo != next) || hi < lo ||
            hi >= unsigned(width))
          return model.emitError("slices of port '")
                 << port.name << "' do not cover it exactly";
        next = hi + 1;
        for (unsigned bit = lo; bit <= hi; ++bit)
          bitChannels[bit] = channels.size();
        Channel channel;
        channel.name = StringAttr::get(
            model.getContext(),
            port.name.getValue() + "_" + Twine(hi) +
                (hi == lo ? Twine("_") : "_" + Twine(lo) + "_"));
        channel.type = IntegerType::get(model.getContext(), hi - lo + 1);
        channel.argNum = port.argNum;
        channel.sliced = true;
        channel.low = lo;
        channels.push_back(channel);
      }
      if (isInput && next != unsigned(width))
        return model.emitError("slices of port '")
               << port.name << "' do not cover it exactly";
    }
    if (isInput)
      inputOfArg[body->getArgument(port.argNum)] = bitChannels;
  }
  // A channel's fields in the order given, its type their bits together.
  for (auto isInput : {true, false}) {
    auto &channels = isInput ? inputs : outputs;
    for (auto [index, channel] : llvm::enumerate(channels)) {
      if (channel.fields.empty())
        continue;
      auto &ranks = fieldRanks[{index, isInput}];
      SmallVector<unsigned> order(ranks.size());
      std::iota(order.begin(), order.end(), 0);
      llvm::sort(order, [&](auto a, auto b) { return ranks[a] < ranks[b]; });
      SmallVector<Channel::Field> sorted;
      unsigned width = 0;
      for (auto i : order) {
        sorted.push_back(channel.fields[i]);
        width += hw::getBitWidth(channel.fields[i].type);
      }
      channel.fields = sorted;
      channel.type = IntegerType::get(model.getContext(), width);
    }
  }
  if (!hasClock)
    return model.emitError("a model has a clock input");

  for (auto &op : *body) {
    if (physical.contains(&op))
      continue;
    if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
      auto *module = SymbolTable::lookupNearestSymbolFrom(
          inst, inst.getReferencedModuleNameAttr());
      if (module && foldsim::isBlackBox(module)) {
        auto contract = foldsim::getPipelineContract(module);
        if (failed(contract))
          return failure();
        if (!*contract)
          return inst.emitError("black box '")
                 << inst.getReferencedModuleName()
                 << "' in a threaded model has no pipeline contract: its "
                    "state cannot be threaded, and without a contract it "
                    "cannot be folded";
        // One thread has no state to thread: the box is its own state, as
        // its contract says, on the target's edges.
        if (threads == 1 && !gate) {
          auto ports = cast<hw::HWModuleLike>(module).getPortList();
          for (auto &port : ports) {
            if (port.dir != hw::ModulePort::Direction::Input)
              continue;
            auto operand = inst.getOperand(port.argNum);
            if (port.getName() == (*contract)->clock) {
              auto conv = operand.getDefiningOp<seq::FromClockOp>();
              Value clock = conv ? conv.getInput() : operand;
              if (clock != body->getArgument(clockArg))
                return inst.emitError("black box '")
                       << inst.getInstanceName()
                       << "' not on the model's clock";
            }
          }
          keptBoxes.insert(inst);
          continue;
        }
        return inst.emitError("black box '")
               << inst.getReferencedModuleName()
               << "' in a threaded model: "
               << (foldPipelines
                       ? "it is not in a pipeline that can be folded (see "
                         "foldsim-find-pipelines)"
                       : "it is simulated only as a folded pipeline "
                         "(fold-pipelines)");
      }
    }
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      if (reg.getIsAsync())
        return reg.emitError("asynchronous reset in a threaded model");
      if (reg.getClk() != body->getArgument(clockArg))
        return reg.emitError("register not on the model's clock");
      if (hw::getBitWidth(reg.getType()) <= 0)
        return reg.emitError("register has no bit width");
      registers.push_back(reg);
      continue;
    }
    if (auto memory = dyn_cast<seq::FirMemOp>(op)) {
      if (memory.getInit())
        return memory.emitError("initialized memory in a threaded model");
      if (memory.getReadLatency() > 1 || memory.getWriteLatency() != 1)
        return memory.emitError(
            "a threaded memory has a read latency of 0 or 1 and a write "
            "latency of 1");
      if (!llvm::isPowerOf2_32(threads))
        return memory.emitError("threading a memory needs a power of two of "
                                "threads");
      memories.push_back(memory);
      continue;
    }
    // A clock as a bit is checked with the clock's uses below.
    if (isa<hw::OutputOp, seq::FirMemReadOp, seq::FirMemWriteOp,
            seq::FromClockOp>(op))
      continue;
    if (isa<hw::HWInstanceLike>(op) || op.getDialect()->getNamespace() == "seq")
      return op.emitError("a threaded model must be flat and keep its state "
                          "in registers and memories");
  }
  for (auto &use : body->getArgument(clockArg).getUses()) {
    auto *owner = use.getOwner();
    if (isa<seq::FirRegOp, seq::FirMemReadOp, seq::FirMemWriteOp>(owner) ||
        keptBoxes.contains(owner))
      continue;
    // A black box's clock input of one bit.
    if (isa<seq::FromClockOp>(owner) &&
        llvm::all_of(owner->getUsers(),
                     [&](Operation *user) { return keptBoxes.contains(user); }))
      continue;
    return owner->emitError("clock used other than by a register or memory");
  }
  return success();
}

/// The inputs each bit of `value` depends on combinationally. Bits are followed
/// through the ops that only move bits around, so that a signal packed with
/// data into an aggregate does not take on the data's dependences; any other
/// op makes each result bit depend on every operand bit.
const BitDeps &Threader::getDeps(Value value) {
  if (auto it = deps.find(value); it != deps.end()) {
    // A value reached again before its sets are final is on a cycle: its
    // sets so far are read, and the cycle iterated once it is all reached.
    if (auto open = depsOpen.find(value); open != depsOpen.end()) {
      auto &caller = depsOpen[depsCalls.back()].second;
      caller = std::min(caller, open->second.first);
      if (depsCalls.back() == value)
        depsSelfCycles.insert(value);
    }
    return *it->second;
  }
  auto width = std::max<int64_t>(hw::getBitWidth(value.getType()), 0);
  if (!depSets)
    depSets = std::make_unique<DepSets>(inputs.size());
  unsigned index = depsReached++;
  depsOpen[value] = {index, index};
  depsStack.push_back(value);
  deps[value] = std::make_unique<BitDeps>(width, 0);
  depsCalls.push_back(value);
  *deps[value] = computeBitDeps(value);
  depsCalls.pop_back();
  unsigned low = depsOpen[value].second;
  if (!depsCalls.empty()) {
    auto &caller = depsOpen[depsCalls.back()].second;
    caller = std::min(caller, low);
  }
  if (low != index)
    return *deps[value];

  // The first value reached of a cycle: the sets of the cycle's values only
  // grow, from none, until they hold every input any of them depends on.
  SmallVector<Value> cycle;
  do
    cycle.push_back(depsStack.pop_back_val());
  while (cycle.back() != value);
  if (cycle.size() > 1 || depsSelfCycles.contains(value)) {
    for (bool changed = true; changed;) {
      changed = false;
      for (auto member : cycle) {
        depsCalls.push_back(member);
        auto bits = computeBitDeps(member);
        depsCalls.pop_back();
        if (bits != *deps[member]) {
          *deps[member] = std::move(bits);
          changed = true;
        }
      }
    }
  }
  for (auto member : cycle)
    depsOpen.erase(member);
  return *deps[value];
}

/// The sets of `value`'s bits from those of its operands, as they are now.
BitDeps Threader::computeBitDeps(Value value) {
  auto width = std::max<int64_t>(hw::getBitWidth(value.getType()), 0);
  auto &sets = *depSets;
  BitDeps bits(width, 0);
  auto all = [&](Value operand) -> unsigned {
    if (auto it = allDeps.find(operand); it != allDeps.end())
      return it->second;
    unsigned any = 0;
    auto operandBits = getDeps(operand);
    for (auto bit : operandBits)
      any = sets.unite(any, bit);
    if (depsOpen.contains(operand))
      return any;
    return allDeps[operand] = any;
  };
  auto copyBits = [&](Value operand, unsigned from, unsigned to,
                      unsigned count) {
    auto operandBits = getDeps(operand);
    for (unsigned i = 0; i < count; ++i)
      bits[to + i] = sets.unite(bits[to + i], operandBits[from + i]);
  };
  auto *op = value.getDefiningOp();
  APInt index;
  if (auto it = inputOfArg.find(value); it != inputOfArg.end()) {
    for (auto [bit, channel] : llvm::zip(bits, it->second))
      bit = sets.single(channel);
  } else if (auto read = dyn_cast_or_null<seq::FirMemReadOp>(op)) {
    // A registered read is state; a combinational one follows its address.
    auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
    if (!memory || memory.getReadLatency() == 0) {
      unsigned any = all(read.getAddress());
      if (read.getEnable())
        any = sets.unite(any, all(read.getEnable()));
      for (auto &bit : bits)
        bit = any;
    }
  } else if (!op ||
             isa<seq::FirRegOp, hw::ConstantOp, hw::AggregateConstantOp>(op) ||
             keptBoxes.contains(op)) {
    // Inputs of the threaded module's state only.
  } else if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
    unsigned at = 0;
    for (auto operand : llvm::reverse(concat.getInputs())) {
      auto w = hw::getBitWidth(operand.getType());
      copyBits(operand, 0, at, w);
      at += w;
    }
  } else if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
    copyBits(extract.getInput(), extract.getLowBit(), 0, width);
  } else if (auto replicate = dyn_cast<comb::ReplicateOp>(op)) {
    auto w = hw::getBitWidth(replicate.getInput().getType());
    for (unsigned at = 0; at + w <= unsigned(width); at += w)
      copyBits(replicate.getInput(), 0, at, w);
  } else if (isa<hw::BitcastOp>(op)) {
    copyBits(op->getOperand(0), 0, 0, width);
  } else if (isa<hw::ArrayCreateOp, hw::ArrayConcatOp>(op)) {
    // Operands are listed from the most significant one.
    unsigned at = 0;
    for (auto operand : llvm::reverse(op->getOperands())) {
      auto w = hw::getBitWidth(operand.getType());
      copyBits(operand, 0, at, w);
      at += w;
    }
  } else if (auto get = dyn_cast<hw::ArrayGetOp>(op)) {
    auto arrayType = hw::type_cast<hw::ArrayType>(get.getInput().getType());
    auto size = arrayType.getNumElements();
    if (matchPattern(get.getIndex(), m_ConstantInt(&index))) {
      if (index.getZExtValue() < size)
        copyBits(get.getInput(), index.getZExtValue() * width, 0, width);
    } else {
      for (unsigned e = 0; e < size; ++e)
        copyBits(get.getInput(), e * width, 0, width);
      auto indexDeps = all(get.getIndex());
      for (auto &bit : bits)
        bit = sets.unite(bit, indexDeps);
    }
  } else if (auto slice = dyn_cast<hw::ArraySliceOp>(op);
             slice &&
             matchPattern(slice.getLowIndex(), m_ConstantInt(&index))) {
    auto elementWidth = hw::getBitWidth(
        hw::type_cast<hw::ArrayType>(slice.getType()).getElementType());
    copyBits(slice.getInput(), index.getZExtValue() * elementWidth, 0, width);
  } else if (auto create = dyn_cast<hw::StructCreateOp>(op)) {
    // The first field is the most significant.
    unsigned at = 0;
    for (auto operand : llvm::reverse(create.getInput())) {
      auto w = hw::getBitWidth(operand.getType());
      copyBits(operand, 0, at, w);
      at += w;
    }
  } else if (auto extract = dyn_cast<hw::StructExtractOp>(op)) {
    auto structType =
        hw::type_cast<hw::StructType>(extract.getInput().getType());
    unsigned at = 0;
    auto elements = structType.getElements();
    for (auto i = elements.size(); i-- > extract.getFieldIndex() + 1;)
      at += hw::getBitWidth(elements[i].type);
    copyBits(extract.getInput(), at, 0, width);
  } else if (isa<comb::AndOp, comb::OrOp, comb::XorOp>(op)) {
    for (auto operand : op->getOperands())
      copyBits(operand, 0, 0, width);
  } else if (auto mux = dyn_cast<comb::MuxOp>(op)) {
    copyBits(mux.getTrueValue(), 0, 0, width);
    copyBits(mux.getFalseValue(), 0, 0, width);
    auto condDeps = all(mux.getCond());
    for (auto &bit : bits)
      bit = sets.unite(bit, condDeps);
  } else {
    unsigned any = 0;
    for (auto operand : op->getOperands())
      any = sets.unite(any, all(operand));
    for (auto &bit : bits)
      bit = any;
  }
  return bits;
}

void Threader::computeDeps() {
  auto output = cast<hw::OutputOp>(model.getBodyBlock()->getTerminator());
  ImplicitLocOpBuilder b(model.getLoc(), output);
  for (auto &channel : outputs) {
    if (!channel.fields.empty()) {
      // The fields' bits together, the first one high.
      SmallVector<Value> parts;
      for (auto &field : channel.fields)
        parts.push_back(b.createOrFold<hw::BitcastOp>(
            b.getIntegerType(hw::getBitWidth(field.type)),
            output.getOperand(field.argNum)));
      channel.current = b.createOrFold<comb::ConcatOp>(parts);
      channel.deps = llvm::BitVector(inputs.size());
      for (auto bit : getDeps(channel.current))
        channel.deps |= depSets->get(bit);
      continue;
    }
    auto value = output.getOperand(channel.argNum);
    channel.current = value;
    channel.deps = llvm::BitVector(inputs.size());
    auto width = hw::getBitWidth(channel.type);
    for (auto bit : ArrayRef(getDeps(value)).slice(channel.low, width))
      channel.deps |= depSets->get(bit);
  }
}

/// Marks the values computed from `sources`, through the ops `through` takes.
static void markUsers(ArrayRef<Value> sources, DenseSet<Value> &marked,
                      function_ref<bool(Operation *)> through) {
  SmallVector<Value> worklist;
  for (auto source : sources)
    if (marked.insert(source).second)
      worklist.push_back(source);
  while (!worklist.empty())
    for (auto *user : worklist.pop_back_val().getUsers())
      if (through(user))
        for (auto result : user->getResults())
          if (marked.insert(result).second)
            worklist.push_back(result);
}

/// Forward from the state rather than back from each value: a combinational
/// cycle then needs no special care.
void Threader::computeStateDeps() {
  SmallVector<Value> registerValues, readData;
  for (auto reg : registers)
    registerValues.push_back(reg.getResult());
  for (auto memory : memories)
    for (auto *user : memory->getUsers())
      if (auto read = dyn_cast<seq::FirMemReadOp>(user))
        readData.push_back(read.getData());
  // A kept black box's outputs change on the target's edges like a memory's
  // registered read, its next value not computed here.
  for (auto *box : keptBoxes)
    for (auto result : box->getResults())
      readData.push_back(result);
  markUsers(registerValues, fromState, [](Operation *) { return true; });
  markUsers(readData, fromMemory,
            [](Operation *op) { return !isa<seq::FirRegOp>(op); });
}

/// The value `value` takes in the next target cycle: its logic with every
/// register replaced by the value it is about to take. Only values that do
/// not depend on an input combinationally have one.
Value Threader::getNext(Value value, ImplicitLocOpBuilder &b) {
  if (auto it = nexts.find(value); it != nexts.end())
    return it->second;
  // On a combinational cycle, a value's next value is used before it is
  // built.
  if (nextsInProgress.contains(value)) {
    if (auto it = nextsAhead.find(value); it != nextsAhead.end())
      return it->second;
    if (!nextCycles)
      nextCycles = std::make_unique<BackedgeBuilder>(b, value.getLoc());
    return nextsAhead[value] = nextCycles->get(value.getType());
  }
  Value result;
  if (auto reg = value.getDefiningOp<seq::FirRegOp>()) {
    result = reg.getNext();
    if (reg.getReset())
      result =
          comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), result);
  } else if (!dependsOnState(value)) {
    result = value;
  } else {
    auto *op = value.getDefiningOp();
    for (auto result : op->getResults())
      nextsInProgress.insert(result);
    IRMapping mapping;
    for (auto operand : op->getOperands())
      mapping.map(operand, getNext(operand, b));
    auto *clone = b.clone(*op, mapping);
    for (auto [old, cloned] : llvm::zip(op->getResults(), clone->getResults())) {
      nextsInProgress.erase(old);
      nexts[old] = cloned;
      if (auto ahead = nextsAhead.find(old); ahead != nextsAhead.end()) {
        ahead->second.setValue(cloned);
        nextsAhead.erase(ahead);
      }
    }
    return nexts[value];
  }
  return nexts[value] = result;
}

Value Threader::createThreadState(ImplicitLocOpBuilder &b, StringRef stateName,
                                  Type type, Value &memory,
                                  IntegerAttr preset) {
  auto width = hw::getBitWidth(type);
  if (threads == 1) {
    // Its write gives the register its next value and clock.
    auto next = registers1->get(b.getIntegerType(width));
    auto clock = registers1->get(seq::ClockType::get(b.getContext()));
    if (preset)
      preset = b.getIntegerAttr(b.getIntegerType(width),
                                preset.getValue().zextOrTrunc(width));
    memory = seq::FirRegOp::create(b, next, clock, b.getStringAttr(stateName),
                                   hw::InnerSymAttr{}, preset);
    unwritten.try_emplace(memory, next, clock);
    return b.createOrFold<hw::BitcastOp>(type, memory);
  }
  auto memType = seq::FirMemType::get(b.getContext(), threads, width, 1);
  memory = seq::FirMemOp::create(
      b, memType, /*readLatency=*/0, /*writeLatency=*/1, seq::RUW::Undefined,
      seq::WUW::Undefined, b.getStringAttr(stateName), hw::InnerSymAttr{},
      seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
  // Read at a registered index into the read-ahead register, deep thread
  // state is a block RAM read synchronously.
  if (blockStateDepth && threads >= blockStateDepth)
    sv::addSVAttributes(memory.getDefiningOp(),
                        {sv::SVAttributeAttr::get(b.getContext(), "ram_style",
                                                  R"("block")",
                                                  /*emitAsComment=*/false)});
  Value data =
      seq::FirMemReadOp::create(b, memory, threadIdx, hostClock,
                                hw::ConstantOp::create(b, b.getI1Type(), 1));
  return b.createOrFold<hw::BitcastOp>(type, data);
}

void Threader::writeThreadState(ImplicitLocOpBuilder &b, Value memory,
                                Value clock, Value data, Value enable) {
  if (threads == 1) {
    auto it = unwritten.find(memory);
    auto [next, regClock] = it->second;
    data = b.createOrFold<hw::BitcastOp>(memory.getType(), data);
    next.setValue(enable ? b.createOrFold<comb::MuxOp>(enable, data, memory)
                         : data);
    regClock.setValue(clock);
    unwritten.erase(it);
    return;
  }
  auto width = cast<seq::FirMemType>(memory.getType()).getWidth();
  data = b.createOrFold<hw::BitcastOp>(b.getIntegerType(width), data);
  if (!enable)
    enable = hw::ConstantOp::create(b, b.getI1Type(), 1);
  seq::FirMemWriteOp::create(b, memory, threadIdx, clock, enable, data,
                             Value{});
}

/// Bits `[low, low + width)` of the integer `value`, looking through wiring.
static Value sliceBits(ImplicitLocOpBuilder &b, Value value, unsigned low,
                       unsigned width) {
  if (low == 0 && hw::getBitWidth(value.getType()) == width)
    return value;
  if (auto extract = value.getDefiningOp<comb::ExtractOp>())
    return sliceBits(b, extract.getInput(), extract.getLowBit() + low, width);
  if (auto concat = value.getDefiningOp<comb::ConcatOp>()) {
    // The operands are most significant first.
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

/// The concatenation of `pieces`, adjacent slices of one value joined.
static Value joinBits(ImplicitLocOpBuilder &b, ArrayRef<Value> pieces) {
  SmallVector<Value> joined;
  for (auto piece : pieces) {
    auto extract = piece.getDefiningOp<comb::ExtractOp>();
    auto last = joined.empty() ? comb::ExtractOp()
                               : joined.back().getDefiningOp<comb::ExtractOp>();
    if (extract && last && last.getInput() == extract.getInput() &&
        extract.getLowBit() + hw::getBitWidth(extract.getType()) ==
            last.getLowBit()) {
      joined.back() = sliceBits(b, extract.getInput(), extract.getLowBit(),
                                hw::getBitWidth(extract.getType()) +
                                    hw::getBitWidth(last.getType()));
      continue;
    }
    joined.push_back(piece);
  }
  return joined.size() == 1 ? joined[0]
                            : b.createOrFold<comb::ConcatOp>(joined);
}

/// `value` as `type`, looking through bitcasts.
static Value castBits(ImplicitLocOpBuilder &b, Value value, Type type) {
  if (value.getType() == type)
    return value;
  if (auto cast = value.getDefiningOp<hw::BitcastOp>())
    return castBits(b, cast.getInput(), type);
  return b.createOrFold<hw::BitcastOp>(type, value);
}

/// `value` with the mux below its wiring (bitcasts, extracts and concats)
/// moved to the top, or `value` if there is none: a mux of the slices of the
/// register is the register.
static Value hoistMux(ImplicitLocOpBuilder &b, Value value) {
  auto *op = value.getDefiningOp();
  if (!op)
    return value;
  if (auto cast = dyn_cast<hw::BitcastOp>(op)) {
    auto mux = hoistMux(b, cast.getInput()).getDefiningOp<comb::MuxOp>();
    if (!mux)
      return value;
    return comb::MuxOp::create(
        b, mux.getCond(), castBits(b, mux.getTrueValue(), value.getType()),
        castBits(b, mux.getFalseValue(), value.getType()));
  }
  if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
    auto mux = hoistMux(b, extract.getInput()).getDefiningOp<comb::MuxOp>();
    if (!mux)
      return value;
    unsigned low = extract.getLowBit(),
             width = hw::getBitWidth(extract.getType());
    return comb::MuxOp::create(b, mux.getCond(),
                               sliceBits(b, mux.getTrueValue(), low, width),
                               sliceBits(b, mux.getFalseValue(), low, width));
  }
  if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
    SmallVector<Value> pieces;
    Value cond;
    for (auto operand : concat.getInputs()) {
      pieces.push_back(hoistMux(b, operand));
      auto mux = pieces.back().getDefiningOp<comb::MuxOp>();
      if (!mux)
        continue;
      // Pieces under different conditions have no one write enable; hoisting
      // them one condition at a time would split the register's next value
      // into a branch per combination of them.
      if (cond && mux.getCond() != cond)
        return value;
      cond = mux.getCond();
    }
    if (!cond)
      return value;
    // The pieces that are not muxes are the same on both sides.
    SmallVector<Value> onTrue, onFalse;
    for (auto piece : pieces) {
      auto mux = piece.getDefiningOp<comb::MuxOp>();
      bool split = mux && mux.getCond() == cond;
      onTrue.push_back(split ? mux.getTrueValue() : piece);
      onFalse.push_back(split ? mux.getFalseValue() : piece);
    }
    return comb::MuxOp::create(b, cond, joinBits(b, onTrue),
                               joinBits(b, onFalse));
  }
  return value;
}

foldsim::HeldWrite foldsim::peelHolds(ImplicitLocOpBuilder &b, Value next,
                                      Value reg) {
  if (next == reg)
    return {/*holds=*/true, {}, {}};
  next = hoistMux(b, next);
  if (next == reg)
    return {/*holds=*/true, {}, {}};
  auto mux = next.getDefiningOp<comb::MuxOp>();
  if (!mux)
    return {false, {}, next};
  auto onTrue = peelHolds(b, mux.getTrueValue(), reg);
  auto onFalse = peelHolds(b, mux.getFalseValue(), reg);
  if (onTrue.holds && onFalse.holds)
    return {true, {}, {}};
  auto one = [&]() { return hw::ConstantOp::create(b, b.getI1Type(), 1); };
  auto cond = mux.getCond();
  if (onTrue.holds)
    return {false,
            onFalse.enable
                ? b.createOrFold<comb::AndOp>(
                      b.createOrFold<comb::XorOp>(cond, one()), onFalse.enable)
                : b.createOrFold<comb::XorOp>(cond, one()),
            onFalse.data};
  if (onFalse.holds)
    return {false,
            onTrue.enable ? b.createOrFold<comb::AndOp>(cond, onTrue.enable)
                          : cond,
            onTrue.data};
  Value enable;
  if (onTrue.enable || onFalse.enable)
    enable =
        b.createOrFold<comb::MuxOp>(cond, onTrue.enable ? onTrue.enable : one(),
                                    onFalse.enable ? onFalse.enable : one());
  Value data =
      onTrue.data == mux.getTrueValue() && onFalse.data == mux.getFalseValue()
          ? next
          : b.createOrFold<comb::MuxOp>(cond, onTrue.data, onFalse.data);
  return {false, enable, data};
}

LogicalResult Threader::enableOnHostClock(ImplicitLocOpBuilder &b,
                                          Value targetClock, Value enable) {
  auto andEnable = [&](Value other) -> Value {
    return other ? comb::AndOp::create(b, other, enable).getResult() : enable;
  };
  // Setting an operand that is optional may move the op's operands, so the
  // users are collected first and their clock set through the op.
  SmallVector<Operation *> users;
  for (auto *user : targetClock.getUsers())
    if (!llvm::is_contained(users, user))
      users.push_back(user);
  for (auto *user : users) {
    if (auto reg = dyn_cast<seq::FirRegOp>(user)) {
      // A synchronous reset only acts on an edge, so it is enabled too.
      Value next = reg.getNext();
      if (reg.hasReset()) {
        next =
            comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), next);
        reg.getResetMutable().clear();
        reg.getResetValueMutable().clear();
      }
      reg.getNextMutable().assign(
          comb::MuxOp::create(b, enable, next, reg.getResult()));
      reg.getClkMutable().assign(hostClock);
    } else if (auto write = dyn_cast<seq::FirMemWriteOp>(user)) {
      write.getEnableMutable().assign(andEnable(write.getEnable()));
      write.getClkMutable().assign(hostClock);
    } else if (auto read = dyn_cast<seq::FirMemReadOp>(user)) {
      read.getEnableMutable().assign(andEnable(read.getEnable()));
      read.getClkMutable().assign(hostClock);
    } else {
      return user->emitError("on the target clock of a threaded model");
    }
  }
  return success();
}

Threader::RegionMemory &Threader::regionMemory(Value memory) {
  auto [it, inserted] =
      regionMemoryIndex.try_emplace(memory, regionMemories.size());
  if (inserted)
    regionMemories.push_back({memory, {}, {}, false, std::nullopt});
  return regionMemories[it->second];
}

bool Threader::coneHasMultiplier(Value value) {
  SmallVector<Operation *> stack;
  DenseSet<Operation *> seen;
  if (auto *op = value.getDefiningOp(); inCone(op))
    stack.push_back(op);
  while (!stack.empty()) {
    auto *op = stack.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    if (isa<comb::MulOp>(op))
      return true;
    for (auto operand : op->getOperands())
      if (auto *def = operand.getDefiningOp(); inCone(def))
        stack.push_back(def);
  }
  return false;
}

unsigned Threader::coneLevels(Value value) {
  auto *root = value.getDefiningOp();
  if (!inCone(root))
    return 0;
  if (auto it = coneLevelsOf.find(root); it != coneLevelsOf.end())
    return it->second;
  // Post-order, operands first.
  SmallVector<std::pair<Operation *, unsigned>> stack = {{root, 0}};
  while (!stack.empty()) {
    auto &[op, next] = stack.back();
    if (next < op->getNumOperands()) {
      auto *def = op->getOperand(next++).getDefiningOp();
      if (inCone(def) && !coneLevelsOf.count(def))
        stack.push_back({def, 0});
      continue;
    }
    unsigned levels = 0;
    for (auto operand : op->getOperands())
      if (auto *def = operand.getDefiningOp(); inCone(def))
        levels = std::max(levels, coneLevelsOf.lookup(def));
    coneLevelsOf[op] = levels + levelsOf(op);
    stack.pop_back();
  }
  return coneLevelsOf.lookup(root);
}

/// `value`'s cone computed a host cycle late, cut in two: the logic after the
/// cut is cloned on registers of the values it reads that are not (the cone's
/// inputs among them), each registered once for every cone, and computed on
/// the second host cycle. The rest of the cone is computed on the host cycle
/// before, as the logic that feeds a block's input registers.
/// - With `pipelineMultipliers`, the cut is where the multipliers start: the
///   multipliers and what reads them are after it.
/// - With `pipelineDepth`, a cone deeper than that is cut at a level: the ops
///   deeper than it are after it. The level leaves neither host cycle more
///   than `pipelineDepth` levels if it can, and of those registers the fewest
///   bits; a cone too deep for that is cut in the middle.
/// Both make the cut the union of the two.
Value Threader::pipelinedCone(ImplicitLocOpBuilder &b, Value value) {
  auto input = [&](Value leaf) -> Value {
    auto &reg = coneInputs[leaf];
    if (!reg) {
      std::string leafName = "cone_in";
      if (auto r = leaf.getDefiningOp<seq::FirRegOp>())
        leafName = (r.getName() + "_in").str();
      else if (auto read = leaf.getDefiningOp<seq::FirMemReadOp>())
        leafName = (read.getMemory().getDefiningOp<seq::FirMemOp>().getName()
                        .value_or("mem") +
                    "_in")
                       .str();
      else if (auto arg = dyn_cast<BlockArgument>(leaf))
        leafName = (cast<hw::HWModuleOp>(arg.getOwner()->getParentOp())
                        .getInputName(arg.getArgNumber()) +
                    "_in")
                       .str();
      reg = seq::FirRegOp::create(b, leaf, hostClock,
                                  b.getStringAttr(leafName));
    }
    return reg;
  };
  auto *root = value.getDefiningOp();
  if (!inCone(root))
    return input(value);
  // Post-order over the cone, then clones in that order.
  SmallVector<Operation *> order;
  DenseSet<Operation *> seen;
  SmallVector<std::pair<Operation *, unsigned>> stack = {{root, 0}};
  seen.insert(root);
  while (!stack.empty()) {
    auto &[op, next] = stack.back();
    if (next < op->getNumOperands()) {
      auto *def = op->getOperand(next++).getDefiningOp();
      if (inCone(def) && seen.insert(def).second)
        stack.push_back({def, 0});
      continue;
    }
    order.push_back(op);
    stack.pop_back();
  }
  // The ops computed from a multiplier's result, operands before users.
  DenseSet<Operation *> multiplied;
  if (pipelineMultipliers)
    for (auto *op : order)
      if (isa<comb::MulOp>(op) ||
          llvm::any_of(op->getOperands(), [&](Value operand) {
            return multiplied.contains(operand.getDefiningOp());
          }))
        multiplied.insert(op);
  // After a cut at level `cut`: the multiplied ops and those deeper than it.
  // A user is never shallower than its operands, so both are closed under
  // users, and so is their union.
  unsigned levels = coneLevels(value);
  auto isAfter = [&](Operation *op, unsigned cut) {
    return op && (multiplied.contains(op) ||
                  (cut && coneLevelsOf.lookup(op) > cut));
  };
  unsigned cut = 0;
  if (pipelineDepth && levels > pipelineDepth) {
    // The bits a cut registers that no other cone has registered already.
    auto cost = [&](unsigned at) {
      uint64_t bits = 0;
      DenseSet<Value> counted;
      for (auto *op : order) {
        if (!isAfter(op, at))
          continue;
        for (auto operand : op->getOperands())
          if (!isAfter(operand.getDefiningOp(), at) &&
              !operand.getDefiningOp<hw::ConstantOp>() &&
              !coneInputs.count(operand) && counted.insert(operand).second)
            bits += hw::getBitWidth(operand.getType());
      }
      return bits;
    };
    unsigned lo = levels > 2 * pipelineDepth ? (levels + 1) / 2
                                             : std::max(1u, levels - pipelineDepth);
    unsigned hi = levels > 2 * pipelineDepth ? (levels + 1) / 2
                                             : std::min(levels - 1, pipelineDepth);
    // The cheapest, and of those the nearest the middle.
    uint64_t best = UINT64_MAX;
    for (unsigned at = lo; at <= hi; ++at) {
      uint64_t bits = cost(at);
      auto offMiddle = [&](unsigned x) {
        return std::abs(int(2 * x) - int(levels));
      };
      if (bits < best || (bits == best && offMiddle(at) < offMiddle(cut))) {
        best = bits;
        cut = at;
      }
    }
  }
  if (!isAfter(root, cut))
    return input(value);
  IRMapping map;
  for (auto *op : order) {
    if (!isAfter(op, cut))
      continue;
    for (auto operand : op->getOperands())
      if (!isAfter(operand.getDefiningOp(), cut) && !map.contains(operand))
        map.map(operand, operand.getDefiningOp<hw::ConstantOp>()
                             ? operand
                             : input(operand));
    // Computed on the second host cycle by another cone already.
    if (llvm::all_of(op->getResults(),
                     [&](Value result) { return coneLate.count(result); })) {
      for (auto result : op->getResults())
        map.map(result, coneLate.lookup(result));
      continue;
    }
    auto *clone = b.clone(*op, map);
    for (auto [result, late] : llvm::zip(op->getResults(), clone->getResults()))
      coneLate[result] = late;
  }
  return map.lookup(value);
}

unsigned Threader::mappedLevelsOf(Operation *op) {
  auto log = [](uint64_t n, unsigned base) {
    unsigned levels = 0;
    for (uint64_t m = 1; m < n; m *= base)
      ++levels;
    return levels;
  };
  auto width = [](Value value) -> uint64_t {
    int64_t w = hw::getBitWidth(value.getType());
    return w > 0 ? w : 1;
  };
  if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
    auto depth = cast<seq::FirMemType>(read.getMemory().getType()).getDepth();
    return 1 + log(llvm::divideCeil(depth, 64), 4);
  }
  if (!levelsOf(op))
    return 0;
  if (isa<comb::AddOp, comb::SubOp>(op))
    return 1 + width(op->getResult(0)) / 8;
  if (auto cmp = dyn_cast<comb::ICmpOp>(op)) {
    uint64_t w = width(cmp.getLhs());
    auto predicate = cmp.getPredicate();
    if (predicate == comb::ICmpPredicate::eq ||
        predicate == comb::ICmpPredicate::ne ||
        predicate == comb::ICmpPredicate::ceq ||
        predicate == comb::ICmpPredicate::cne ||
        predicate == comb::ICmpPredicate::weq ||
        predicate == comb::ICmpPredicate::wne)
      return 1 + log(llvm::divideCeil(w, 3), 6);
    return 1 + w / 8;
  }
  if (isa<comb::AndOp, comb::OrOp, comb::XorOp>(op))
    return std::max(1u, log(op->getNumOperands(), 6));
  return levelsOf(op);
}

/// A registered read of a memory takes its address and enable on the host
/// clock's edge that ends the host cycle computing them. The part of their
/// cone that depends on registers (and constants) alone can be computed a
/// host cycle earlier, from those registers' next values -- with registered
/// state, the next thread's state as it is read ahead -- and registered: the
/// register holds on every host cycle what the cone computes on it, so the
/// port sees the same values on the same host cycles, and its own host cycle
/// computes only the rest, from the registers and the other values the cone
/// reads (ports, memory reads). What is computed ahead: from the port down,
/// the first values on registers alone at least `readPortDepth` estimated
/// levels deep and wider than a LUT's inputs -- registering a narrower value
/// would hide its few inputs from the logic that reads it, which merges such
/// values (decodes of one state, say) into its own LUTs. A register's next
/// value includes its synchronous reset; a register with an asynchronous
/// reset or on another clock is not computed ahead; a value registered ahead
/// starts from the preset its cone computes from its registers' presets, and
/// is not computed ahead where those presets do not fold.
void Threader::pipelineReadPorts(ImplicitLocOpBuilder &b,
                                 hw::HWModuleOp threaded) {
  auto *body = threaded.getBodyBlock();
  b.setInsertionPoint(body->getTerminator());
  auto latencyOf = [](seq::FirMemReadOp read) -> std::optional<unsigned> {
    if (auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>())
      return memory.getReadLatency();
    return std::nullopt;
  };
  // The cone: combinational logic, and LUT RAM reads (a read of latency 0)
  // as logic on their address and enable that is never computed ahead.
  auto isConeOp = [&](Operation *op) {
    if (auto read = dyn_cast_or_null<seq::FirMemReadOp>(op))
      return latencyOf(read) == 0u;
    return inCone(op);
  };
  auto coneOperands = [&](Operation *op) {
    SmallVector<Value> operands;
    if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
      operands.push_back(read.getAddress());
      if (read.getEnable())
        operands.push_back(read.getEnable());
    } else {
      operands.append(op->operand_begin(), op->operand_end());
    }
    return operands;
  };
  // Post-order over a cone, each value once: `visit` runs on a value whose
  // cone operands were all visited before, unless `leaf` says the cone stops
  // at it.
  auto postOrder = [&](Value root, auto &&seen, auto &&visit,
                       auto &&leaf) {
    SmallVector<std::pair<Value, bool>> stack = {{root, false}};
    while (!stack.empty()) {
      auto [value, expanded] = stack.pop_back_val();
      if (seen(value))
        continue;
      auto *op = value.getDefiningOp();
      if (!expanded && isConeOp(op) && !leaf(value)) {
        stack.push_back({value, true});
        for (auto operand : coneOperands(op))
          if (!seen(operand))
            stack.push_back({operand, false});
        continue;
      }
      visit(value);
    }
  };
  // Whether a value is a function of at most a LUT's inputs (6 bits of
  // registers, ports or other leaves): such a value is one level, and is
  // never computed ahead -- a register of it would hide its few inputs from
  // the logic that reads it, which merges such values (decodes of one
  // state, say) into its own LUTs.
  constexpr unsigned lutInputs = 6;
  struct Support {
    SmallVector<Value, 4> leaves;
    bool wide = false;
  };
  DenseMap<Value, Support> supports;
  std::function<const Support &(Value)> supportOf =
      [&](Value v) -> const Support & {
    if (auto it = supports.find(v); it != supports.end())
      return it->second;
    Support support;
    auto *op = v.getDefiningOp();
    auto bitsOf = [](const Support &s) {
      uint64_t bits = 0;
      for (auto leaf : s.leaves)
        bits += std::max<int64_t>(hw::getBitWidth(leaf.getType()), 1);
      return bits;
    };
    if (isa_and_nonnull<hw::ConstantOp>(op)) {
    } else if (op && inCone(op)) {
      for (auto operand : op->getOperands()) {
        const Support &s = supportOf(operand);
        if (s.wide) {
          support.wide = true;
          break;
        }
        for (auto leaf : s.leaves)
          if (!llvm::is_contained(support.leaves, leaf))
            support.leaves.push_back(leaf);
        if (bitsOf(support) > lutInputs) {
          support.wide = true;
          break;
        }
      }
    } else {
      support.leaves.push_back(v);
      support.wide = bitsOf(support) > lutInputs;
    }
    if (support.wide)
      support.leaves.clear();
    return supports[v] = std::move(support);
  };
  auto isNarrow = [&](Value v) { return !supportOf(v).wide; };
  // Levels from the cone's leaves (registers, ports, other ops).
  DenseMap<Value, unsigned> levels;
  auto levelsOfValue = [&](Value root) {
    postOrder(
        root, [&](Value v) { return levels.count(v); },
        [&](Value v) {
          auto *op = v.getDefiningOp();
          unsigned l = 0;
          if (isConeOp(op)) {
            for (auto operand : coneOperands(op))
              l = std::max(l, levels.lookup(operand));
            l += mappedLevelsOf(op);
            if (l > 1 && isa<comb::CombDialect>(op->getDialect()) &&
                isNarrow(v))
              l = 1;
          }
          levels[v] = l;
        },
        [](Value) { return false; });
    return levels.lookup(root);
  };
  // Whether a value depends on registers of the host clock and constants
  // alone.
  DenseMap<Value, bool> early;
  auto isEarly = [&](Value root) {
    postOrder(
        root, [&](Value v) { return early.count(v); },
        [&](Value v) {
          auto *op = v.getDefiningOp();
          bool e = false;
          if (auto reg = dyn_cast_or_null<seq::FirRegOp>(op))
            e = reg.getClk() == hostClock && !reg.getIsAsync();
          else if (isa_and_nonnull<seq::FirMemReadOp>(op))
            e = false;
          else if (isConeOp(op))
            e = llvm::all_of(op->getOperands(),
                             [&](Value o) { return early.lookup(o); });
          early[v] = e;
        },
        [](Value) { return false; });
    return early.lookup(root);
  };
  // A register's next value, its synchronous reset included.
  DenseMap<Value, Value> nextOf;
  auto nextValue = [&](seq::FirRegOp reg) -> Value {
    auto &next = nextOf[reg];
    if (!next) {
      next = reg.getNext();
      if (reg.getReset())
        next = comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(),
                                   next);
    }
    return next;
  };
  // The levels of an early value computed a host cycle ahead: from its
  // registers' next values.
  DenseMap<Value, unsigned> aheadLevels;
  std::function<unsigned(Value)> levelsAhead = [&](Value v) -> unsigned {
    if (auto it = aheadLevels.find(v); it != aheadLevels.end())
      return it->second;
    unsigned l = 0;
    auto *op = v.getDefiningOp();
    if (auto reg = dyn_cast_or_null<seq::FirRegOp>(op)) {
      l = levelsOfValue(reg.getNext()) + (reg.getReset() ? 1 : 0);
    } else if (isConeOp(op)) {
      for (auto operand : op->getOperands())
        l = std::max(l, levelsAhead(operand));
      l += mappedLevelsOf(op);
    }
    return aheadLevels[v] = l;
  };
  // A value's preset from its registers' presets, if they fold; null if
  // they do not, or if no register of its cone has one.
  DenseMap<Value, Attribute> presets;
  DenseMap<Value, bool> presetted;
  std::function<bool(Value)> hasPreset = [&](Value v) -> bool {
    if (auto it = presetted.find(v); it != presetted.end())
      return it->second;
    bool p = false;
    auto *op = v.getDefiningOp();
    if (auto reg = dyn_cast_or_null<seq::FirRegOp>(op))
      p = reg.hasPresetValue();
    else if (isConeOp(op) && !isa<hw::ConstantOp>(op))
      p = llvm::any_of(op->getOperands(), hasPreset);
    return presetted[v] = p;
  };
  std::function<Attribute(Value)> presetOf = [&](Value v) -> Attribute {
    if (auto it = presets.find(v); it != presets.end())
      return it->second;
    Attribute preset;
    auto *op = v.getDefiningOp();
    if (auto reg = dyn_cast_or_null<seq::FirRegOp>(op)) {
      auto type = dyn_cast<IntegerType>(v.getType());
      if (auto attr = reg.getPresetAttr(); attr && type)
        preset = IntegerAttr::get(
            type, cast<IntegerAttr>(attr).getValue().zextOrTrunc(
                      type.getWidth()));
    } else if (auto constant = dyn_cast_or_null<hw::ConstantOp>(op)) {
      preset = constant.getValueAttr();
    } else if (op && isConeOp(op) && op->getNumResults() == 1) {
      SmallVector<Attribute> operands;
      for (auto operand : op->getOperands())
        operands.push_back(presetOf(operand));
      SmallVector<OpFoldResult> results;
      if (llvm::all_of(operands, [](Attribute a) { return !!a; }) &&
          succeeded(op->fold(operands, results)) && results.size() == 1) {
        if (auto attr = dyn_cast<Attribute>(results[0]))
          preset = attr;
        else
          for (auto [operand, attr] :
               llvm::zip(op->getOperands(), operands))
            if (operand == cast<Value>(results[0]))
              preset = attr;
      }
    }
    return presets[v] = preset;
  };

  DenseMap<Value, Value> aheadClone, aheadReg;
  std::function<Value(Value)> cloneAhead = [&](Value v) -> Value {
    if (auto it = aheadClone.find(v); it != aheadClone.end())
      return it->second;
    Value result;
    auto *op = v.getDefiningOp();
    if (auto reg = dyn_cast_or_null<seq::FirRegOp>(op)) {
      result = nextValue(reg);
    } else if (isa<hw::ConstantOp>(op)) {
      result = v;
    } else {
      IRMapping map;
      for (auto operand : op->getOperands())
        map.map(operand, cloneAhead(operand));
      result = b.clone(*op, map)->getResult(cast<OpResult>(v).getResultNumber());
    }
    return aheadClone[v] = result;
  };
  auto registerAhead = [&](Value v, StringRef memoryName) -> Value {
    auto &reg = aheadReg[v];
    if (!reg) {
      Attribute preset = hasPreset(v) ? presetOf(v) : Attribute();
      reg = seq::FirRegOp::create(b, cloneAhead(v), hostClock,
                                  b.getStringAttr(memoryName + "_ahead"),
                                  hw::InnerSymAttr{}, preset);
    }
    return reg;
  };
  // The cone above the values computed ahead, cloned on their registers; a
  // LUT RAM read keeps its port, its address and enable set in place to the
  // same values.
  DenseSet<Value> cut;
  DenseMap<Value, Value> lateClone;
  std::function<Value(Value, StringRef)> cloneLate =
      [&](Value v, StringRef memoryName) -> Value {
    if (cut.contains(v))
      return registerAhead(v, memoryName);
    auto *op = v.getDefiningOp();
    if (!isConeOp(op))
      return v;
    if (auto it = lateClone.find(v); it != lateClone.end())
      return it->second;
    Value result = v;
    if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
      Value address = cloneLate(read.getAddress(), memoryName);
      if (address != read.getAddress())
        read.getAddressMutable().assign(address);
      if (auto enable = read.getEnable()) {
        Value e = cloneLate(enable, memoryName);
        if (e != enable)
          read.getEnableMutable().assign(e);
      }
    } else {
      IRMapping map;
      bool changed = false;
      for (auto operand : op->getOperands()) {
        Value mapped = cloneLate(operand, memoryName);
        changed |= mapped != operand;
        map.map(operand, mapped);
      }
      if (changed)
        result = b.clone(*op, map)->getResult(
            cast<OpResult>(v).getResultNumber());
    }
    return lateClone[v] = result;
  };

  SmallVector<seq::FirMemReadOp> ports;
  for (auto read : body->getOps<seq::FirMemReadOp>())
    if (latencyOf(read).value_or(0) >= 1 && read.getClk() == hostClock)
      ports.push_back(read);
  for (auto read : ports) {
    auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
    std::string memoryName = memory.getName().value_or("mem").str();
    for (unsigned which = 0; which < 2; ++which) {
      Value root = which ? read.getEnable() : read.getAddress();
      if (!root || levelsOfValue(root) < readPortDepth)
        continue;
      // The values computed ahead: from the port down, the first values on
      // registers alone at least `readPortDepth` levels deep, wider than a
      // LUT's inputs, and whose presets fold where their registers have
      // them; the cone of any other value is searched on.
      SmallVector<Value> values;
      uint64_t bits = 0;
      unsigned ahead = 0;
      DenseSet<Value> seen;
      SmallVector<Value> stack = {root};
      while (!stack.empty()) {
        Value v = stack.pop_back_val();
        auto *op = v.getDefiningOp();
        if (!seen.insert(v).second || !isConeOp(op))
          continue;
        bool early = isEarly(v);
        if (early && levelsOfValue(v) < readPortDepth)
          continue;
        if (early && !isNarrow(v) && (!hasPreset(v) || presetOf(v))) {
          if (cut.insert(v).second || !llvm::is_contained(values, v)) {
            values.push_back(v);
            bits += std::max<int64_t>(hw::getBitWidth(v.getType()), 1);
            ahead = std::max(ahead, levelsAhead(v));
          }
          continue;
        }
        for (auto operand : coneOperands(op))
          stack.push_back(operand);
      }
      if (values.empty())
        continue;
      Value pipelined =
          cloneLate(root, memoryName + (which ? "_re" : "_ra"));
      if (which)
        read.getEnableMutable().assign(pipelined);
      else
        read.getAddressMutable().assign(pipelined);
      read.emitRemark() << "read-port-depth: " << memoryName
                        << (which ? " enable" : " address") << " of "
                        << levelsOfValue(root) << " estimated levels: "
                        << values.size() << " values (" << bits
                        << " bits) computed a host cycle ahead, at most "
                        << ahead << " levels from the registers' next values";
    }
  }
}

/// An array declared [0:N-1] imports with its rows reversed: each port of its
/// memory addresses it by `(N-1) - i`, the complement of the index, an adder
/// on every address path. Where every port of a memory takes a complement
/// (`all ones - x` or `x ^ all ones`) or a constant, each takes `x` (the
/// constant's complement) instead: the same accesses to the same rows, but
/// rows reversed, which no port can tell, the memory having no initial
/// contents (`checkModel`).
void Threader::unreverseMemoryAddresses() {
  auto complementOf = [](Value address) -> std::optional<Value> {
    auto *op = address.getDefiningOp();
    auto isOnes = [](Value v) {
      APInt c;
      return matchPattern(v, m_ConstantInt(&c)) && c.isAllOnes();
    };
    if (auto sub = dyn_cast_or_null<comb::SubOp>(op))
      if (isOnes(sub.getLhs()))
        return sub.getRhs();
    if (auto xorOp = dyn_cast_or_null<comb::XorOp>(op))
      if (xorOp.getNumOperands() == 2) {
        if (isOnes(xorOp.getOperand(1)))
          return xorOp.getOperand(0);
        if (isOnes(xorOp.getOperand(0)))
          return xorOp.getOperand(1);
      }
    return std::nullopt;
  };
  for (auto memory : model.getBodyBlock()->getOps<seq::FirMemOp>()) {
    SmallVector<OpOperand *> addresses;
    bool reversed = true, any = false;
    for (auto &use : memory->getUses()) {
      auto *user = use.getOwner();
      Value address;
      if (auto read = dyn_cast<seq::FirMemReadOp>(user))
        address = read.getAddress();
      else if (auto write = dyn_cast<seq::FirMemWriteOp>(user))
        address = write.getAddress();
      else {
        reversed = false;
        break;
      }
      APInt c;
      if (matchPattern(address, m_ConstantInt(&c))) {
      } else if (complementOf(address)) {
        any = true;
      } else {
        reversed = false;
        break;
      }
      addresses.push_back(isa<seq::FirMemReadOp>(user)
                              ? &cast<seq::FirMemReadOp>(user)
                                     .getAddressMutable()
                              : &cast<seq::FirMemWriteOp>(user)
                                     .getAddressMutable());
    }
    if (!reversed || !any)
      continue;
    OpBuilder b(memory);
    for (auto *operand : addresses) {
      Value address = operand->get();
      APInt c;
      if (matchPattern(address, m_ConstantInt(&c))) {
        b.setInsertionPoint(operand->getOwner());
        operand->set(hw::ConstantOp::create(b, address.getLoc(), ~c));
      } else {
        operand->set(*complementOf(address));
      }
    }
  }
}

void Threader::splitWideMemories(ImplicitLocOpBuilder &b) {
  auto keep = sv::SVAttributeAttr::get(b.getContext(), "keep", R"("true")",
                                       /*emitAsComment=*/false);
  // A port the regions do not control keeps its address and enable, each
  // part a copy of the gates and registers that drive them last, so that no
  // one of them drives every part. A register is copied only with a preset:
  // the copy then holds its value on every host cycle.
  std::function<Value(Value, unsigned)> copyOf = [&](Value value,
                                                     unsigned depth) -> Value {
    auto *op = value ? value.getDefiningOp() : nullptr;
    if (!op || !depth || op->getNumResults() != 1)
      return value;
    auto reg = dyn_cast<seq::FirRegOp>(op);
    if (reg ? !reg.getPreset() || reg.getInnerSymAttr()
            : !isa<comb::CombDialect>(op->getDialect()))
      return value;
    auto *clone = b.clone(*op);
    if (reg) {
      sv::addSVAttributes(clone, {keep});
    } else {
      for (auto &operand : clone->getOpOperands())
        operand.set(copyOf(operand.get(), depth - 1));
    }
    return clone->getResult(0);
  };
  SmallVector<RegionMemory> split;
  for (auto &state : regionMemories) {
    auto memory = state.memory.getDefiningOp<seq::FirMemOp>();
    auto type = memory.getType();
    uint64_t width = type.getWidth();
    // The bits of a mask bit.
    uint64_t grain = type.getMaskWidth() && *type.getMaskWidth()
                         ? width / *type.getMaskWidth()
                         : 1;
    // The widest part of at most `memoryLoads` loads: whole mask bits, or
    // the bits of one.
    uint64_t part = std::max<uint64_t>(
        1, memoryLoads / llvm::divideCeil(type.getDepth(), 32));
    if (grain <= part)
      part = part / grain * grain;
    if (width <= part || memory.getInitAttr() ||
        !llvm::all_of(memory->getUsers(), [](Operation *user) {
          return isa<seq::FirMemReadOp, seq::FirMemWriteOp>(user);
        })) {
      split.push_back(std::move(state));
      continue;
    }
    auto loc = b.getLoc();
    b.setLoc(memory.getLoc());
    struct Part {
      uint64_t low, width;
      RegionMemory state;
    };
    SmallVector<Part> parts;
    for (uint64_t low = 0, bits; low < width; low += bits) {
      bits = std::min(part, width - low);
      if (grain > part)
        bits = std::min(bits, (low / grain + 1) * grain - low);
      auto partType = seq::FirMemType::get(
          b.getContext(), type.getDepth(), bits,
          type.getMaskWidth()
              ? std::optional<uint32_t>(std::max<uint64_t>(1, bits / grain))
              : std::nullopt);
      auto partMemory = seq::FirMemOp::create(
          b, partType, memory.getReadLatency(), memory.getWriteLatency(),
          memory.getRuw(), memory.getWuw(),
          b.getStringAttr(memory.getName().value_or("mem") + "_p" +
                          Twine(parts.size())),
          hw::InnerSymAttr{}, seq::FirMemInitAttr{}, memory.getPrefixAttr(),
          memory.getOutputFileAttr());
      partMemory->setDialectAttrs(memory->getDialectAttrs());
      parts.push_back({low, bits, {partMemory, {}, {}, state.state, state.loc}});
    }
    auto readOf = [&](seq::FirMemReadOp op) {
      return llvm::find_if(state.reads,
                           [&](auto &read) { return read.op == op; });
    };
    auto writeOf = [&](seq::FirMemWriteOp op) {
      return llvm::find_if(state.writes,
                           [&](auto &write) { return write.op == op; });
    };
    // In the order of the ports, which the parts' ports keep.
    SmallVector<Operation *> users(memory->getUsers());
    llvm::sort(users, [](Operation *x, Operation *y) {
      return x->isBeforeInBlock(y);
    });
    for (auto *user : users) {
      if (auto read = dyn_cast<seq::FirMemReadOp>(user)) {
        auto recorded = readOf(read);
        bool own = recorded == state.reads.end();
        SmallVector<Value> data;
        for (auto [i, p] : llvm::enumerate(parts)) {
          auto port = seq::FirMemReadOp::create(
              b, p.state.memory,
              own && i ? copyOf(read.getAddress(), 2) : read.getAddress(),
              read.getClk(),
              own && i ? copyOf(read.getEnable(), 2) : read.getEnable());
          if (!own)
            p.state.reads.push_back(
                {port, recorded->low, recorded->registered, recorded->enable});
          data.push_back(port);
        }
        std::reverse(data.begin(), data.end());
        read.getData().replaceAllUsesWith(
            b.createOrFold<comb::ConcatOp>(data));
        read.erase();
        continue;
      }
      auto write = cast<seq::FirMemWriteOp>(user);
      auto recorded = writeOf(write);
      bool own = recorded == state.writes.end();
      for (auto [i, p] : llvm::enumerate(parts)) {
        Value mask = write.getMask();
        if (mask)
          mask = b.createOrFold<comb::ExtractOp>(
              b.getIntegerType(std::max<uint64_t>(1, p.width / grain)), mask,
              p.low / grain);
        auto port = seq::FirMemWriteOp::create(
            b, p.state.memory,
            own && i ? copyOf(write.getAddress(), 2) : write.getAddress(),
            write.getClk(),
            own && i ? copyOf(write.getEnable(), 2) : write.getEnable(),
            b.createOrFold<comb::ExtractOp>(b.getIntegerType(p.width),
                                            write.getData(), p.low),
            mask);
        if (!own)
          p.state.writes.push_back({port, recorded->low, recorded->term,
                                    recorded->edge, recorded->preset,
                                    recorded->staged});
      }
      write.erase();
    }
    memory.erase();
    for (auto &p : parts)
      split.push_back(std::move(p.state));
    b.setLoc(loc);
  }
  regionMemories = std::move(split);
  regionMemoryIndex.clear();
  for (auto [i, state] : llvm::enumerate(regionMemories))
    regionMemoryIndex[state.memory] = i;
}

void Threader::buildRegions(ImplicitLocOpBuilder &b, Value edge,
                            Value initializing, Value hostReset) {
  if (memoryLoads)
    splitWideMemories(b);
  auto *context = b.getContext();
  auto i1 = b.getI1Type();
  auto idxType = threadIdx.getType();
  auto idxNext = threadIdx.getDefiningOp<seq::FirRegOp>().getNext();
  auto keep = sv::SVAttributeAttr::get(context, "keep", R"("true")",
                                       /*emitAsComment=*/false);
  auto nameOf = [&](const RegionMemory &state) {
    return state.memory.getDefiningOp<seq::FirMemOp>().getName().value_or("");
  };
  // A memory's port drives a LUT's address pins per bit and 32 entries.
  auto loadsOf = [&](const RegionMemory &state) {
    auto type = cast<seq::FirMemType>(state.memory.getType());
    return uint64_t(type.getWidth()) * llvm::divideCeil(type.getDepth(), 32);
  };
  SmallVector<unsigned> order(regionMemories.size());
  std::iota(order.begin(), order.end(), 0);
  // The state written here first, then the thread state only read here, in
  // regions of their own.
  auto readOnly = [&](unsigned i) { return regionMemories[i].writes.empty(); };
  llvm::stable_sort(order, [&](unsigned x, unsigned y) {
    if (readOnly(x) != readOnly(y))
      return readOnly(y);
    return nameOf(regionMemories[x]) < nameOf(regionMemories[y]);
  });
  SmallVector<SmallVector<unsigned>> regions;
  uint64_t loads = 0;
  for (auto i : order) {
    auto stateLoads = loadsOf(regionMemories[i]);
    if (regions.empty() || readOnly(i) != readOnly(regions.back().front()) ||
        (regionLoads && loads && loads + stateLoads > regionLoads)) {
      regions.emplace_back();
      loads = 0;
    }
    regions.back().push_back(i);
    loads += stateLoads;
  }

  // Each region's copies are registers with the same next value as the one
  // they copy, so they hold its value on the same host cycle. The thread
  // index has its copy for reads and, a cycle later, for writes; the target
  // edge and the initialization a copy a cycle late. Kept: equal registers
  // are otherwise merged back into one.
  auto copy = [&](Value next, const Twine &copyName, Value reset,
                  Value resetValue, uint64_t initial = 0) {
    auto type = next.getType();
    auto preset = b.getIntegerAttr(type, initial);
    seq::FirRegOp reg =
        reset ? seq::FirRegOp::create(b, next, hostClock,
                                      b.getStringAttr(copyName), reset,
                                      resetValue, hw::InnerSymAttr{},
                                      /*isAsync=*/false, preset)
              : seq::FirRegOp::create(b, next, hostClock,
                                      b.getStringAttr(copyName),
                                      hw::InnerSymAttr{}, preset);
    sv::addSVAttributes(reg, {keep});
    return Value(reg);
  };
  // A write lands a host cycle after its thread's edge: the next thread is
  // never the one that just finished, nor, after the initialization, the
  // last one initialized, so no read sees it late.
  auto staged = [&](Value value, const Twine &stagedName) -> Value {
    if (!value)
      return value;
    return seq::FirRegOp::create(b, value, hostClock,
                                 b.getStringAttr(stagedName));
  };
  // In turn, the thread index is a function of the host cycles since reset: a
  // region counts it itself rather than copying the model's, so nothing far
  // feeds its index and it is placed by the memories it addresses.
  // A kept register reset to `initial`, its next value set by the caller.
  auto counter = [&](Type type, const Twine &counterName, uint64_t initial) {
    Value init = hw::ConstantOp::create(b, type, initial);
    seq::FirRegOp reg = seq::FirRegOp::create(
        b, init, hostClock, b.getStringAttr(counterName), hostReset, init,
        hw::InnerSymAttr{}, /*isAsync=*/false, b.getIntegerAttr(type, initial));
    sv::addSVAttributes(reg, {keep});
    return reg;
  };
  auto setNext = [](seq::FirRegOp reg, Value next) {
    reg.getNextMutable().assign(next);
  };
  auto inTurnAfter = [&](Value index) -> Value {
    return comb::MuxOp::create(
        b,
        comb::ICmpOp::create(b, comb::ICmpPredicate::eq, index,
                             hw::ConstantOp::create(b, idxType, threads - 1)),
        hw::ConstantOp::create(b, idxType, 0),
        comb::AddOp::create(b, index, hw::ConstantOp::create(b, idxType, 1)));
  };
  auto countTurns = [&](unsigned r, Value &current, Value &ahead) {
    if (!aheadIdx) {
      auto idx = counter(idxType, "threadIdx_r" + Twine(r), idxReset);
      setNext(idx, inTurnAfter(idx));
      current = idx;
      return;
    }
    auto idx = counter(idxType, "threadIdx_a" + Twine(r), aheadReset);
    Value next = inTurnAfter(idx);
    if (aheadPhase) {
      auto phaseType = aheadPhase.getType();
      auto phaseReg = counter(phaseType, "phase_a" + Twine(r), aheadPhaseReset);
      setNext(phaseReg, comb::AddOp::create(
                            b, phaseReg, hw::ConstantOp::create(b, phaseType, 1)));
      next = comb::MuxOp::create(
          b,
          comb::ICmpOp::create(b, comb::ICmpPredicate::eq, phaseReg,
                               hw::ConstantOp::create(b, phaseType, phases - 1)),
          next, idx);
    }
    setNext(idx, next);
    ahead = idx;
    auto now = counter(idxType, "threadIdx_r" + Twine(r), idxReset);
    setNext(now, idx);
    current = now;
  };
  for (auto [r, region] : llvm::enumerate(regions)) {
    bool readsState = aheadIdx && llvm::any_of(region, [&](unsigned i) {
                        return regionMemories[i].state;
                      });
    Value readIdx, aheadCopy;
    if (!readySchedule) {
      countTurns(r, readIdx, aheadCopy);
      if (!readsState)
        aheadCopy = {};
    } else {
      readIdx = copy(idxNext, "threadIdx_r" + Twine(r), hostReset,
                     hw::ConstantOp::create(b, idxType, idxReset),
                     /*initial=*/idxReset);
      // With registered state, the thread state is read for the next thread.
      if (readsState)
        aheadCopy =
            copy(aheadIdx.getDefiningOp<seq::FirRegOp>().getNext(),
                 "threadIdx_a" + Twine(r), hostReset,
                 hw::ConstantOp::create(b, idxType, aheadReset),
                 /*initial=*/aheadReset);
    }
    // Only a region with writes has their control.
    Value writeIdx, edgeLate;
    if (!readOnly(region.front())) {
      writeIdx = copy(readIdx, "threadIdx_w" + Twine(r), {}, {});
      edgeLate = copy(edge, "edge_r" + Twine(r), {}, {});
    }
    Value writeIdx2;
    Value initLate;
    for (auto i : region) {
      auto &state = regionMemories[i];
      auto memoryName = nameOf(state);
      // A port's address or enable may be a state read whose uses an earlier
      // region already moved to its read-ahead register: it is that register.
      auto fresh = [&](Value &value) {
        if (auto it = readAhead.find(value); value && it != readAhead.end())
          value = it->second;
      };
      for (auto &read : state.reads) {
        fresh(read.low);
        fresh(read.enable);
      }
      for (auto &write : state.writes) {
        fresh(write.low);
        fresh(write.term);
      }
      // Staging located where the state is, rather than at the whole model:
      // merged locations of thousands of registers grow the Verilog.
      if (compactStaging)
        b.setLoc(state.loc ? *state.loc : state.memory.getLoc());
      Value stateIdx = aheadIdx && state.state ? aheadCopy : readIdx;
      for (auto &read : state.reads)
        read.op.getAddressMutable().assign(
            read.low ? b.createOrFold<comb::ConcatOp>(read.low, stateIdx)
                     : stateIdx);
      // The write landing on this cycle's edge.
      Value pendingData, pendingEnable;
      // A registered read on an edge that writes its address reads the
      // write's data if the memory reads new data under a write: the write
      // lands a host cycle later, so its staged data is forwarded then.
      auto memoryOp = state.memory.getDefiningOp<seq::FirMemOp>();
      bool readsNew = memoryOp.getRuw() == seq::RUW::New;
      struct Forward {
        Value hit, data, mask;
      };
      SmallVector<SmallVector<Forward>> forwards(state.reads.size());
      for (auto &write : state.writes) {
        if (!write.staged) {
          write.op.getAddressMutable().assign(
              write.low ? b.createOrFold<comb::ConcatOp>(write.low, readIdx)
                        : readIdx);
          continue;
        }
        if (readsNew)
          for (auto [k, read] : llvm::enumerate(state.reads)) {
            if (!read.registered)
              continue;
            SmallVector<Value> terms;
            for (auto term : {write.term, read.enable})
              if (term && !matchPattern(term, m_One()))
                terms.push_back(term);
            if (write.low)
              terms.push_back(comb::ICmpOp::create(
                  b, comb::ICmpPredicate::eq, write.low, read.low));
            Value hit = terms.empty()
                            ? Value(hw::ConstantOp::create(b, i1, 1))
                            : b.createOrFold<comb::AndOp>(terms, false);
            forwards[k].push_back(
                {seq::FirRegOp::create(b, hit, hostClock,
                                       b.getStringAttr(memoryName + "_fwd"),
                                       hw::InnerSymAttr{},
                                       b.getIntegerAttr(i1, 0)),
                 {}, {}});
          }
        Value address = writeIdx;
        if (write.low)
          address = comb::ConcatOp::create(
              b, staged(write.low, memoryName + "_waddr"), writeIdx);
        Value unstagedData = write.op.getData();
        bool deep = deepState() && state.state && !write.low;
        // A multiplier's cone takes the first stage at its inputs: Vivado
        // cannot retime the stage into the DSPs that compute it.
        bool pipelined =
            deep && ((pipelineMultipliers && coneHasMultiplier(unstagedData)) ||
                     (pipelineDepth && coneLevels(unstagedData) > pipelineDepth));
        Value data = pipelined ? pipelinedCone(b, unstagedData)
                               : staged(unstagedData, memoryName + "_wdata");
        Value mask = staged(write.op.getMask(), memoryName + "_wmask");
        Value enable;
        if (write.edge) {
          enable = edgeLate;
          if (write.term)
            enable = comb::AndOp::create(
                b,
                // With compact staging, without a preset: the edge's copy is
                // zero until the first edge it copies.
                compactStaging
                    ? seq::FirRegOp::create(b, write.term, hostClock,
                                            b.getStringAttr(memoryName + "_wen"))
                    : seq::FirRegOp::create(
                          b, write.term, hostClock,
                          b.getStringAttr(memoryName + "_wen"),
                          hw::InnerSymAttr{}, b.getIntegerAttr(i1, 0)),
                edgeLate);
        }
        if (write.preset) {
          if (!initLate)
            initLate = copy(initializing, "init_r" + Twine(r), {}, {});
          enable = enable ? Value(comb::OrOp::create(b, initLate, enable))
                          : initLate;
        }
        // The state's write a second host cycle late: the next thread but
        // one is never one finishing now either.
        if (deep) {
          if (!writeIdx2)
            writeIdx2 = copy(writeIdx, "threadIdx_v" + Twine(r), {}, {});
          address = writeIdx2;
          data = staged(data, memoryName + "_wdata2");
          mask = staged(mask, memoryName + "_wmask2");
          enable = seq::FirRegOp::create(b, enable, hostClock,
                                         b.getStringAttr(memoryName + "_wen2"),
                                         hw::InnerSymAttr{},
                                         b.getIntegerAttr(i1, 0));
        }
        write.op.getAddressMutable().assign(address);
        write.op.getDataMutable().assign(data);
        if (mask)
          write.op.getMaskMutable().assign(mask);
        write.op.getEnableMutable().assign(enable);
        pendingData = data;
        pendingEnable = enable;
        for (auto &forward : forwards)
          if (!forward.empty() && !forward.back().data) {
            forward.back().data = data;
            forward.back().mask = mask;
          }
      }
      for (auto [k, read] : llvm::enumerate(state.reads)) {
        if (!read.registered)
          continue;
        // Its data is used only on the host cycle after an edge, so it reads
        // on every host cycle: its enable is the target's alone.
        if (read.enable)
          read.op.getEnableMutable().assign(read.enable);
        else
          read.op.getEnableMutable().clear();
        Value dout = read.op.getData();
        SmallVector<OpOperand *> uses;
        for (auto &use : dout.getUses())
          uses.push_back(&use);
        Value data = dout;
        auto width = hw::getBitWidth(dout.getType());
        // The writes in port order: the last one that hits wins.
        for (auto &forward : forwards[k]) {
          unsigned granules =
              forward.mask ? hw::getBitWidth(forward.mask.getType()) : 1;
          unsigned granule = width / granules;
          SmallVector<Value> pieces;
          for (unsigned g = granules; g-- > 0;) {
            Value hit = forward.hit;
            if (forward.mask)
              hit = comb::AndOp::create(
                  b, hit,
                  b.createOrFold<comb::ExtractOp>(i1, forward.mask, g));
            auto type = b.getIntegerType(granule);
            pieces.push_back(comb::MuxOp::create(
                b, hit,
                b.createOrFold<comb::ExtractOp>(type, forward.data,
                                                g * granule),
                b.createOrFold<comb::ExtractOp>(type, data, g * granule)));
          }
          data = pieces.size() == 1 ? pieces[0]
                                    : b.createOrFold<comb::ConcatOp>(pieces);
        }
        for (auto *use : uses)
          use->set(data);
      }
      if (!aheadIdx || !state.state)
        continue;
      // The next thread's state, read a host cycle ahead into registers. The
      // write landing on this cycle's edge is the only one not in the memory
      // yet: it is forwarded when it is that thread's. In turn, a thread is
      // current once every `threads` host cycles, its state read ahead
      // `threads - 1` after its edge, by when the write is in the memory (from
      // the second host cycle after the edge, the third with two stages)
      // unless there are fewer threads and idle host cycles in their turns:
      // nothing is forwarded. Picked by readiness, a thread may run again
      // sooner.
      bool mayHit =
          readySchedule || threads * phases < (deepState() ? 4u : 3u);
      Value hit;
      if (pendingEnable && mayHit)
        hit = comb::AndOp::create(
            b, pendingEnable,
            comb::ICmpOp::create(b, comb::ICmpPredicate::eq,
                                 writeIdx2 ? writeIdx2 : writeIdx, aheadCopy));
      for (auto &read : state.reads) {
        Value data = read.op.getData();
        auto ahead = seq::FirRegOp::create(
            b, data, hostClock, b.getStringAttr(memoryName + "_state"));
        data.replaceAllUsesExcept(ahead, ahead);
        readAhead[data] = ahead;
        if (hit)
          ahead.getNextMutable().assign(
              comb::MuxOp::create(b, hit, pendingData, data));
      }
    }
  }
}

void Threader::commitBySelect(ImplicitLocOpBuilder &b, Value edge,
                              Value initializing) {
  auto i1 = b.getI1Type();
  auto keep = sv::SVAttributeAttr::get(b.getContext(), "keep", R"("true")",
                                       /*emitAsComment=*/false);
  // Registers in name order, a copy of the enables per `regionLoads` bits.
  llvm::stable_sort(singleRegisters, [](const SingleRegister &x,
                                       const SingleRegister &y) {
    return seq::FirRegOp(x.reg).getName() < seq::FirRegOp(y.reg).getName();
  });
  Value edgeLate, initLate;
  uint64_t loads = 0;
  unsigned region = 0;
  for (auto &single : singleRegisters) {
    auto reg = single.reg;
    auto type = reg.getType();
    uint64_t width = hw::getBitWidth(type);
    if (!edgeLate || (regionLoads && loads && loads + width > regionLoads)) {
      auto late = [&](Value value, const Twine &lateName) {
        auto copy = seq::FirRegOp::create(b, value, hostClock,
                                          b.getStringAttr(lateName),
                                          hw::InnerSymAttr{},
                                          b.getIntegerAttr(i1, 0));
        sv::addSVAttributes(copy, {keep});
        return Value(copy);
      };
      edgeLate = late(edge, "edge_r" + Twine(region));
      initLate = late(initializing, "init_r" + Twine(region));
      ++region;
      loads = 0;
    }
    loads += width;
    if (compactStaging)
      b.setLoc(reg.getLoc());
    Value enable = edgeLate;
    if (single.wen)
      enable = comb::AndOp::create(b, single.wen, edgeLate);
    if (single.preset)
      enable = comb::OrOp::create(b, initLate, enable);
    // The committed value, and the state: it or the candidate committed now.
    auto committed = seq::FirRegOp::create(
        b, reg.getResult(), hostClock, reg.getNameAttr(), hw::InnerSymAttr{},
        reg.getPresetAttr());
    Value value = comb::MuxOp::create(b, enable, single.candidate, committed);
    // Idle on the host cycle after an edge, the model reads the committed
    // register itself: the enable is the register's clock enable.
    reg.getResult().replaceAllUsesWith(phases > 1 ? Value(committed) : value);
    committed.getNextMutable().assign(value);
    reg.erase();
  }
}

/// Bits `[low, low + width)` of `value` as an integer.
static Value bitsOf(ImplicitLocOpBuilder &b, Value value, unsigned low,
                    unsigned width) {
  auto all = hw::getBitWidth(value.getType());
  Value bits = castBits(b, value, b.getIntegerType(all));
  return sliceBits(b, bits, low, width);
}

/// The value of `width` bits made of runs of bits of other values: (value,
/// low bit, count), the least significant run first.
static Value fromRuns(ImplicitLocOpBuilder &b,
                      ArrayRef<std::tuple<Value, unsigned, unsigned>> runs) {
  SmallVector<Value> pieces;
  for (auto &[value, low, count] : llvm::reverse(runs))
    pieces.push_back(bitsOf(b, value, low, count));
  return joinBits(b, pieces);
}

/// Appends bit `bit` of `value` to `runs`, extending the last run if it is
/// the next bit of the same value.
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

// Folding a pipeline. Its unthreaded copy runs on the host clock, every host
// cycle, on whatever thread is current; a thread's issue (its target edge with
// the pipeline enabled) enters it tagged with the thread and the thread's slot,
// its count of enabled edges modulo the store's depth S. The copy's result
// comes out c host cycles later (c = S) and is written, a host cycle later
// still, into the store at (slot, thread): readable c + 2 host cycles after
// the issue. The thread reads it, at its own slot, on the edge S enabled edges
// later (last layer kept: c = S = L - 1) or on any host cycle once it has made
// S of them (last layer stored: c = S = L).
//
// A thread advances at most one target cycle per host cycle, and every
// scheduler of this threader with two threads or more never makes a thread
// that finishes current on the next host cycle. So the edges of one thread are
// two host cycles apart at least: S enabled edges after an issue at host cycle
// h, the thread's edge is at h + 2S at the earliest, and with the last layer
// stored, its next host cycle at h + 2S + 2. The result is readable at
// h + S + 2: in time when S >= 2. A slot is overwritten by the issue S enabled
// edges later, after its last read. A thread's slot counter is target state:
// it is threaded like any register, so any scheduler order keeps it right.
LogicalResult Threader::prepareFolds() {
  auto *body = model.getBodyBlock();
  // Where folding cannot be exact: with one thread, it is current on every
  // host cycle; a clock gate is the platform's, and its edges too.
  StringRef unfit;
  if (threads < 2)
    unfit = "its one thread is current on every host cycle, so its edges "
            "may be on consecutive host cycles";
  else if (gate)
    unfit = "its target clock is a clock gate of the platform's";
  else if (!llvm::isPowerOf2_32(threads))
    unfit = "a threaded store needs a power of two of threads";
  ImplicitLocOpBuilder b(model.getLoc(), body->getTerminator());
  foldsim::PipelineAnalysis analysis(model);
  if (!unfit.empty()) {
    // Only reported when there is something to fold; the model is left as
    // it is (a black box in it is an error of `checkModel`).
    auto copy = cast<hw::HWModuleOp>(OpBuilder(model).clone(*model));
    ImplicitLocOpBuilder cb(copy.getLoc(), copy.getBodyBlock()->getTerminator());
    foldsim::PipelineAnalysis probe(copy);
    bool any = succeeded(probe.run(cb)) && !probe.pipelines.empty();
    copy.erase();
    if (any)
      model.emitRemark("pipelines of '")
          << model.getName() << "' not folded: " << unfit;
    return success();
  }
  if (failed(analysis.run(b)))
    return failure();
  if (analysis.pipelines.empty())
    return success();
  auto *context = model.getContext();
  auto i1 = b.getI1Type();
  Value modelClock;
  for (auto [i, port] : llvm::enumerate(model.getPortList()))
    if (isa<seq::ClockType>(port.type) &&
        port.dir == hw::ModulePort::Direction::Input)
      modelClock = body->getArgument(port.argNum);
  if (!modelClock)
    return model.emitError("a model has a clock input");
  foldEdges = std::make_unique<BackedgeBuilder>(b, model.getLoc());
  physClock = foldEdges->get(seq::ClockType::get(context));
  auto idxType = b.getIntegerType(llvm::Log2_32_Ceil(threads));
  foldThread = foldEdges->get(idxType);
  foldEdge = foldEdges->get(i1);
  foldReset = foldEdges->get(i1);
  auto one = hw::ConstantOp::create(b, i1, 1);
  auto hostReg = [&](Value next, const Twine &regName,
                     bool cleared = false) -> Value {
    auto type = next.getType();
    seq::FirRegOp reg =
        cleared ? seq::FirRegOp::create(
                      b, next, physClock, b.getStringAttr(regName), foldReset,
                      hw::ConstantOp::create(b, type, 0), hw::InnerSymAttr{},
                      /*isAsync=*/false, b.getIntegerAttr(type, 0))
                : seq::FirRegOp::create(b, next, physClock,
                                        b.getStringAttr(regName));
    physical.insert(reg);
    return reg;
  };

  // The stored bits of each pipeline: a register bit of the last layer (its
  // next value if the last layer stays, else its value), and a black box's
  // outputs in the last layer.
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
    // When the result is readable, and when it is read at the earliest.
    // The copy registers its issue first: the leaves of the first layer's
    // logic (state, inputs), so that the thread's state reaches registers
    // through no logic. Its result comes out a host cycle later for it.
    unsigned out = depth + 1;
    unsigned readable = out + 2;
    unsigned firstRead = stored ? 2 * layers : 2 * (layers - 1);
    if (readable > firstRead)
      return model.emitError("folded pipeline ")
             << pi << " of " << layers << " layers: its results are back "
             << readable << " host cycles after issue, after their first "
             << "read at " << firstRead;

    // The thread's slot: its enabled edges modulo the store's depth.
    auto slotType = b.getIntegerType(llvm::Log2_32_Ceil(depth));
    auto slotNext = foldEdges->get(slotType);
    Value slot = seq::FirRegOp::create(b, slotNext, modelClock,
                                       b.getStringAttr(prefix + "_slot"),
                                       hw::InnerSymAttr{},
                                       b.getIntegerAttr(slotType, 0));
    Value last = comb::ICmpOp::create(b, comb::ICmpPredicate::eq, slot,
                                      hw::ConstantOp::create(b, slotType,
                                                             depth - 1));
    Value step = comb::MuxOp::create(
        b, last, hw::ConstantOp::create(b, slotType, 0),
        comb::AddOp::create(b, slot, hw::ConstantOp::create(b, slotType, 1)));
    slotNext.setValue(p.enable ? Value(comb::MuxOp::create(b, p.enable, step,
                                                           slot))
                               : step);

    // The store, read at the thread's slot.
    unsigned width = words[pi].size();
    auto memType = seq::FirMemType::get(context, depth, width, 1);
    auto store = seq::FirMemOp::create(
        b, memType, /*readLatency=*/0, /*writeLatency=*/1, seq::RUW::Undefined,
        seq::WUW::Undefined, b.getStringAttr(prefix + "_store"),
        hw::InnerSymAttr{}, seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
    storeRead[pi] = seq::FirMemReadOp::create(b, store, slot, modelClock, one);
    foldOfStore[store] = folds.size();

    // The unthreaded copy: a host register per register of the pipeline, a
    // black box per black box, on the values the current thread gives its
    // first layer.
    DenseMap<Value, Value> copies;
    SmallVector<std::pair<unsigned, Backedge>> copyNexts;
    for (auto ri : p.registers) {
      auto &reg = registers[ri];
      // A last layer that stays needs no copy.
      bool copied = false;
      for (unsigned bit = 0; bit < reg.layer.size(); ++bit)
        if (reg.layer[bit] && reg.pipeline[bit] == pi &&
            (stored || reg.layer[bit] < layers))
          copied = true;
      if (!copied)
        continue;
      auto type = b.getIntegerType(reg.layer.size());
      auto next = foldEdges->get(type);
      auto copy = hostReg(next, reg.reg.getName() + "_fold");
      copies[reg.reg.getResult()] =
          castBits(b, copy, reg.reg.getResult().getType());
      copyNexts.push_back({ri, next});
    }
    SmallVector<std::pair<unsigned, hw::InstanceOp>> boxCopies;
    for (auto bi : p.boxes)
      boxCopies.push_back({bi, {}});
    // `value` computed from the copy's registers and black boxes: its logic
    // cloned where it depends on the pipeline, the value itself elsewhere.
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
    // The issue, registered: the logic of the first layer cloned on its
    // leaves, each registered once.
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
    // Black boxes first: registers after them read their outputs.
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
                        ? Value(physClock)
                        : (clockConv = seq::FromClockOp::create(b, physClock));
        else if (name == box.contract.enable)
          operand = hw::ConstantOp::create(b, port.type, 1);
        else if (llvm::is_contained(box.contract.dataIn, name))
          operand = box.inLayer > 0
                        ? Value() // set below, once registers have copies
                        : issueCopy(operand);
        operands.push_back(operand);
      }
      // Inputs from the pipeline are cloned once every copy exists.
      for (auto &operand : operands)
        if (!operand)
          operand = hw::ConstantOp::create(b, b.getI1Type(), 0);
      auto *module = SymbolTable::lookupNearestSymbolFrom(
          box.inst, box.inst.getReferencedModuleNameAttr());
      copy = hw::InstanceOp::create(
          b, module, b.getStringAttr(box.inst.getInstanceName() + "_fold"),
          operands);
      physical.insert(copy);
      if (clockConv)
        physical.insert(clockConv.getDefiningOp());
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
    // Each register's copy takes its next value: the thread's own, issued,
    // for its first layer, the copy's for the later ones.
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
               later ? copiedData(ri)
                     : first ? issueCopy(reg.data) : reg.data,
               bit);
      }
      next.setValue(fromRuns(b, runs));
    }
    if (cloneFailed)
      return failure();

    // The stored word, from the copy.
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
    Value valid = p.enable ? Value(comb::AndOp::create(b, foldEdge, p.enable))
                           : foldEdge;
    Value slotTag = slot, threadTag = foldThread;
    for (unsigned k = 1; k <= out + 1; ++k) {
      valid = hostReg(valid, prefix + "_valid" + Twine(k), /*cleared=*/true);
      slotTag = hostReg(slotTag, prefix + "_slot" + Twine(k));
      threadTag = hostReg(threadTag, prefix + "_thread" + Twine(k));
    }
    Value data = hostReg(word, prefix + "_wdata");
    folds.push_back({store, {}, slotTag, threadTag, valid, data});
  }

  // The target: each register of a pipeline keeps its bits that stay (the
  // last layer, now taking its next value from the store, and bits of no
  // pipeline); a stored last layer is read from the store; the other layers
  // are gone.
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
      auto next = foldEdges->get(data.getType());
      keptReg = seq::FirRegOp::create(b, next, reg.reg.getClk(),
                                      b.getStringAttr(name));
      next.setValue(reg.enable
                        ? Value(comb::MuxOp::create(b, reg.enable, data,
                                                    keptReg))
                        : data);
    }
    if (!other.empty()) {
      SmallVector<std::tuple<Value, unsigned, unsigned>> runs;
      for (auto bit : other)
        addBit(runs, reg.reg.getNext(), bit);
      auto next = foldEdges->get(b.getIntegerType(other.size()));
      otherReg = seq::FirRegOp::create(b, next, reg.reg.getClk(),
                                       b.getStringAttr(name));
      next.setValue(fromRuns(b, runs));
    }
    // The register's value, from the bits that stay, the store and zeros
    // for the bits only the pipeline read.
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

  // What only the pipelines' registers used, and the analysis' peeled
  // writes, are dead.
  for (bool changed = true; changed;) {
    changed = false;
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
      if (physical.contains(&op) || !op.use_empty() || op.getNumResults() == 0)
        continue;
      if (inCone(&op) || isa<seq::FromClockOp>(op)) {
        op.erase();
        changed = true;
      }
    }
  }
  return success();
}

LogicalResult Threader::finishFolds(ImplicitLocOpBuilder &b, Value edge,
                                    Value hostReset) {
  if (!separatesFinishes)
    return model.emitError("fold-pipelines: the scheduler of '")
           << model.getName()
           << "' may make a thread that finishes current on the next host "
              "cycle, which folded pipelines cannot follow";
  for (auto &fold : folds) {
    auto port = seq::FirMemWriteOp::create(
        b, fold.threadedStore,
        b.createOrFold<comb::ConcatOp>(fold.slot, fold.thread), hostClock,
        fold.valid, fold.data, Value{});
    (void)port;
  }
  physClock.setValue(hostClock);
  foldThread.setValue(threadIdx);
  foldEdge.setValue(edge);
  foldReset.setValue(hostReset);

  // The folding is exact because a thread that finishes is never current on
  // the next host cycle: checked in simulation, so that a scheduler that
  // breaks it does not go unnoticed.
  auto i1 = b.getI1Type();
  auto top = model->getParentOfType<ModuleOp>();
  if (!top.lookupSymbol<sv::MacroDeclOp>("SYNTHESIS")) {
    OpBuilder tb(top.getBody(), top.getBody()->begin());
    sv::MacroDeclOp::create(tb, model.getLoc(), "SYNTHESIS");
  }
  auto zero = b.getIntegerAttr(i1, 0);
  Value lastEdge = seq::FirRegOp::create(
      b, edge, hostClock, b.getStringAttr("fold_last_edge"), hostReset,
      hw::ConstantOp::create(b, i1, 0), hw::InnerSymAttr{}, false, zero);
  Value lastThread = seq::FirRegOp::create(
      b, threadIdx, hostClock, b.getStringAttr("fold_last_thread"));
  Value again = comb::AndOp::create(
      b, lastEdge,
      comb::ICmpOp::create(b, comb::ICmpPredicate::eq, lastThread, threadIdx));
  // An idle host cycle of the thread's turn neither finishes nor uses a
  // result.
  if (idle)
    again = comb::AndOp::create(
        b, again,
        comb::XorOp::create(b, idle, hw::ConstantOp::create(b, i1, 1)));
  Value clock = seq::FromClockOp::create(b, hostClock);
  sv::IfDefOp::create(
      b, "SYNTHESIS", [] {},
      [&] {
        sv::AlwaysOp::create(b, ArrayRef{sv::EventControl::AtPosEdge},
                             ArrayRef{clock}, [&] {
          sv::IfOp::create(b, again, [&] {
            sv::FatalProceduralOp::create(
                b, 1,
                b.getStringAttr("foldsim: a thread is current on the host "
                                "cycle after its edge; folded pipelines "
                                "need it never to be"));
          });
        });
      });
  return success();
}

LogicalResult Threader::run() {
  if (foldPipelines && failed(prepareFolds()))
    return failure();
  if (failed(checkModel()))
    return failure();
  if (unreverseAddresses)
    unreverseMemoryAddresses();
  computeDeps();
  computeStateDeps();

  // The threaded module's ports: the host clock and reset, then a decoupled
  // channel per target port, inputs first, its bits flat.
  auto *context = model.getContext();
  auto i1 = IntegerType::get(context, 1);
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
  // The target clock's channel carries whether each cycle's edge is enabled,
  // and is always valid.
  if (!clockChannel.empty()) {
    addPort(clockChannel + "_sink_ready", i1, false);
    addPort(clockChannel + "_sink_bits", i1, true);
  }
  // The bits ports of a channel: one, or one per field.
  auto addBits = [&](Channel &channel, StringRef dir, bool isInput) {
    if (channel.fields.empty()) {
      if (hw::getBitWidth(channel.type) > 0)
        addPort(channel.name.getValue() + "_" + dir + "_bits",
                IntegerType::get(context, hw::getBitWidth(channel.type)),
                isInput);
      return;
    }
    for (auto &field : channel.fields)
      addPort(channel.name.getValue() + "_" + dir + "_" + field.name.getValue(),
              IntegerType::get(context, hw::getBitWidth(field.type)), isInput);
  };
  for (auto &channel : inputs) {
    addPort(channel.name.getValue() + "_sink_ready", i1, false);
    addPort(channel.name.getValue() + "_sink_valid", i1, true);
    addBits(channel, "sink", true);
  }
  for (auto &channel : outputs) {
    addPort(channel.name.getValue() + "_source_ready", i1, true);
    addPort(channel.name.getValue() + "_source_valid", i1, false);
    addBits(channel, "source", false);
  }
  if (threadIndexPort && threads > 1)
    addPort("threadIdx", IntegerType::get(context, llvm::Log2_32_Ceil(threads)),
            false);
  if (readySchedule && threads > 1)
    addPort("thread_ready", IntegerType::get(context, threads), true);
  // Whether the current thread finishes its target cycle, for the simulator's
  // own accounting.
  if (threadIndexPort)
    addPort("finish", i1, false);
  bool schedulePorts = externalSchedule && positionWidth() > 0;
  if (schedulePorts) {
    addPort("schedule", IntegerType::get(context, positionWidth()), false);
    addPort("schedule_next", IntegerType::get(context, positionWidth()), true);
  }

  OpBuilder top(model);
  auto threaded = hw::HWModuleOp::create(top, model.getLoc(), name,
                                         hw::ModulePortInfo(ports));
  auto *body = threaded.getBodyBlock();
  body->getTerminator()->erase();
  hostClock = body->getArgument(0);
  Value hostReset = body->getArgument(1);
  SmallVector<Value> sinkValid, sinkBits, sourceReady;
  unsigned arg = 2;
  Value clockBits;
  if (!clockChannel.empty())
    clockBits = body->getArgument(arg++);
  ImplicitLocOpBuilder top0(model.getLoc(), body, body->begin());
  for (auto &channel : inputs) {
    sinkValid.push_back(body->getArgument(arg++));
    if (!channel.fields.empty()) {
      SmallVector<Value> parts;
      for (unsigned i = 0; i < channel.fields.size(); ++i)
        parts.push_back(body->getArgument(arg++));
      sinkBits.push_back(top0.createOrFold<comb::ConcatOp>(parts));
      continue;
    }
    sinkBits.push_back(
        hw::getBitWidth(channel.type) > 0
            ? Value(body->getArgument(arg++))
            : hw::ConstantOp::create(top0, top0.getIntegerType(0), 0));
  }
  for (unsigned i = 0; i < outputs.size(); ++i)
    sourceReady.push_back(body->getArgument(arg++));
  Value threadReady;
  if (readySchedule && threads > 1)
    threadReady = body->getArgument(arg++);
  Value scheduleNext;
  if (schedulePorts)
    scheduleNext = body->getArgument(arg++);
  // Where the turns come from outside, the position on the host cycle after
  // reset, and on the one after that.
  uint64_t positions = uint64_t(threads) * phases;
  uint64_t startNow = scheduleStart % positions,
           startNext = (scheduleStart + 1) % positions;
  auto threadOfPosition = [&](uint64_t position) { return position / phases; };
  auto phaseOfPosition = [&](uint64_t position) { return position % phases; };

  // The target logic moves into the threaded module as is, reading the
  // input channels' bits.
  auto *modelBody = model.getBodyBlock();
  auto modelOutput = modelBody->getTerminator();
  body->getOperations().splice(body->end(), modelBody->getOperations(),
                               modelBody->begin(),
                               Block::iterator(modelOutput));
  ImplicitLocOpBuilder b(model.getLoc(), body, body->end());
  BackedgeBuilder backedges(b, model.getLoc());
  registers1 = std::make_unique<BackedgeBuilder>(b, model.getLoc());
  auto targetClock = backedges.get(seq::ClockType::get(context));
  modelBody->getArgument(clockArg).replaceAllUsesWith(targetClock);
  // Values kept aside for the outputs may be replaced below.
  DenseMap<Value, Value> replaced;
  // An input split into channels is their bits put back together.
  for (auto [i, channel] : llvm::enumerate(inputs)) {
    if (!channel.fields.empty()) {
      // Each field's bits, the first one high.
      unsigned high = hw::getBitWidth(channel.type);
      for (auto &field : channel.fields) {
        auto arg = modelBody->getArgument(field.argNum);
        auto width = hw::getBitWidth(field.type);
        high -= width;
        Value value = b.createOrFold<comb::ExtractOp>(b.getIntegerType(width),
                                                      sinkBits[i], high);
        value = b.createOrFold<hw::BitcastOp>(arg.getType(), value);
        arg.replaceAllUsesWith(value);
        replaced[arg] = value;
      }
      continue;
    }
    auto arg = modelBody->getArgument(channel.argNum);
    if (replaced.count(arg))
      continue;
    Value value = sinkBits[i];
    if (channel.sliced) {
      SmallVector<Value> parts;
      for (auto [j, other] : llvm::enumerate(inputs))
        if (other.argNum == channel.argNum)
          parts.push_back(sinkBits[j]);
      // Channels of a port are in ascending bit order; concat wants the most
      // significant first.
      std::reverse(parts.begin(), parts.end());
      value = b.createOrFold<comb::ConcatOp>(parts);
    }
    value = b.createOrFold<hw::BitcastOp>(arg.getType(), value);
    arg.replaceAllUsesWith(value);
    replaced[arg] = value;
  }

  // The current thread, advancing every host cycle: to the next in turn, or,
  // scheduling by readiness, to the first after it whose inputs are all there.
  Backedge idxNext;
  // The threads' state is initialized on reset and, with more than one
  // thread, on the host cycles after it, one thread each, before any thread
  // runs: a reset of one cycle initializes them all.
  Value initializing = hostReset, sweeping;
  if (threads > 1) {
    auto idxType = b.getIntegerType(llvm::Log2_32_Ceil(threads));
    idxNext = backedges.get(idxType);
    idxReset = startGiven ? threadOfPosition(startNow) : 0;
    threadIdx = seq::FirRegOp::create(
        b, idxNext, hostClock, b.getStringAttr("threadIdx"), hostReset,
        hw::ConstantOp::create(b, idxType, idxReset), hw::InnerSymAttr{},
        /*isAsync=*/false, b.getIntegerAttr(idxType, idxReset));
    auto sweepNext = backedges.get(i1);
    sweeping = seq::FirRegOp::create(
        b, sweepNext, hostClock, b.getStringAttr("init_sweep"), hostReset,
        hw::ConstantOp::create(b, i1, 1), hw::InnerSymAttr{},
        /*isAsync=*/false, b.getIntegerAttr(i1, 1));
    // Every thread once, from the first after reset.
    Value last = comb::ICmpOp::create(
        b, comb::ICmpPredicate::eq, threadIdx,
        hw::ConstantOp::create(b, idxType,
                               (idxReset + threads - 1) % threads));
    sweepNext.setValue(comb::AndOp::create(
        b, sweeping,
        comb::XorOp::create(b, last, hw::ConstantOp::create(b, i1, 1))));
    initializing = comb::OrOp::create(b, hostReset, sweeping);
  }
  // With registered state the next thread is chosen a host cycle earlier: the
  // thread after the current one is a register too.
  Backedge aheadNext, aheadPhaseNext;
  // A model of one thread committing by enable is idle every other host
  // cycle, from the first after reset: its state, written on the host cycle
  // after the edge, is read on the one after that.
  if (threads == 1 && phases > 1) {
    auto phaseNext = backedges.get(i1);
    // Outside turns must start idle too: the state's preset is committed on
    // the host cycle after reset.
    if (startGiven && phaseOfPosition(startNow) != 1) {
      model.emitError("a one-thread model committing by enable starts its "
                      "turns idle (schedule start ")
          << scheduleStart << ")";
      return failure();
    }
    if (scheduleNext) {
      // Outside turns: the next host cycle's phase in a register of its own,
      // as a model of several threads has.
      aheadPhase = seq::FirRegOp::create(
          b, scheduleNext, hostClock, b.getStringAttr("phase_ahead"),
          hostReset, hw::ConstantOp::create(b, i1, 0), hw::InnerSymAttr{},
          /*isAsync=*/false, b.getIntegerAttr(i1, 0));
      phase = seq::FirRegOp::create(
          b, aheadPhase, hostClock, b.getStringAttr("phase"), hostReset,
          hw::ConstantOp::create(b, i1, 1), hw::InnerSymAttr{},
          /*isAsync=*/false, b.getIntegerAttr(i1, 1));
    } else {
      phase = seq::FirRegOp::create(
          b, phaseNext, hostClock, b.getStringAttr("phase"), hostReset,
          hw::ConstantOp::create(b, i1, 1), hw::InnerSymAttr{},
          /*isAsync=*/false, b.getIntegerAttr(i1, 1));
      phaseNext.setValue(
          comb::XorOp::create(b, phase, hw::ConstantOp::create(b, i1, 1)));
    }
    idle = phase;
  }
  if (threads > 1 && registeredState) {
    auto idxType = b.getIntegerType(llvm::Log2_32_Ceil(threads));
    // With idle slots the host cycle after the first is the first thread's
    // too, idle.
    aheadReset = startGiven ? threadOfPosition(startNext)
                 : phases > 1     ? 0
                                  : 1;
    aheadNext = backedges.get(idxType);
    aheadIdx = seq::FirRegOp::create(
        b, aheadNext, hostClock, b.getStringAttr("threadIdx_ahead"), hostReset,
        hw::ConstantOp::create(b, idxType, aheadReset), hw::InnerSymAttr{},
        /*isAsync=*/false, b.getIntegerAttr(idxType, aheadReset));
    if (phases > 1) {
      auto phaseType = b.getIntegerType(llvm::Log2_32(phases));
      aheadPhaseNext = backedges.get(phaseType);
      aheadPhaseReset = startGiven ? phaseOfPosition(startNext) : 1;
      uint64_t phaseReset = startGiven ? phaseOfPosition(startNow) : 0;
      aheadPhase = seq::FirRegOp::create(
          b, aheadPhaseNext, hostClock, b.getStringAttr("phase_ahead"),
          hostReset, hw::ConstantOp::create(b, phaseType, aheadPhaseReset),
          hw::InnerSymAttr{}, /*isAsync=*/false,
          b.getIntegerAttr(phaseType, aheadPhaseReset));
      phase = seq::FirRegOp::create(
          b, aheadPhase, hostClock, b.getStringAttr("phase"), hostReset,
          hw::ConstantOp::create(b, phaseType, phaseReset), hw::InnerSymAttr{},
          /*isAsync=*/false, b.getIntegerAttr(phaseType, phaseReset));
      idle = comb::ICmpOp::create(b, comb::ICmpPredicate::ne, phase,
                                  hw::ConstantOp::create(b, phaseType, 0));
    }
  }

  // An output that depends on no input fires eagerly, with the next cycle's
  // value once its thread has finished.
  for (auto &channel : outputs) {
    // With a clock channel, whether a cycle's finish makes an edge, and so
    // what the next value is, is only known from the clock token.
    channel.eager = clockChannel.empty() && channel.deps.none() &&
                    !dependsOnMemory(channel.current);
    if (channel.eager)
      channel.next = getNext(channel.current, b);
    LLVM_DEBUG({
      llvm::dbgs() << "foldsim-deps " << name << " " << channel.name.getValue()
                   << (channel.eager ? " eager" : "") << " <-";
      for (auto dep : channel.deps.set_bits())
        llvm::dbgs() << " " << inputs[dep].name.getValue();
      llvm::dbgs() << "\n";
    });
  }

  // Each register keeps one value per thread. The branches of its next value
  // that hold it become the write enable of its memory rather than logic that
  // writes it back.
  // A register with a preset value takes it while the threads are initialized,
  // each thread's entry when that thread is current: its target edges are then
  // those of the host clock it is written on.
  auto targetEdge = backedges.get(i1);
  bool regions = hasRegions();
  for (auto reg : registers) {
    Value next = reg.getNext();
    if (reg.getReset())
      next = comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), next);
    auto write = foldsim::peelHolds(b, next, reg.getResult());
    // Written every cycle: the next value as it is.
    if (!write.holds && !write.enable)
      write.data = next;
    Value memory;
    auto value = createThreadState(b, reg.getName(), reg.getType(), memory,
                                   reg.getPresetAttr());
    reg.getResult().replaceAllUsesWith(value);
    replaced[reg.getResult()] = value;
    if (auto preset = reg.getPreset()) {
      auto width = hw::getBitWidth(reg.getType());
      Value initial = hw::ConstantOp::create(b, preset->zextOrTrunc(width));
      Value data = initial, enable = initializing;
      if (!write.holds) {
        data = comb::MuxOp::create(
            b, initializing, initial,
            b.createOrFold<hw::BitcastOp>(b.getIntegerType(width),
                                          write.data));
        Value edge = write.enable
                         ? Value(comb::AndOp::create(b, targetEdge,
                                                     write.enable))
                         : Value(targetEdge);
        enable = comb::OrOp::create(b, initializing, edge);
      }
      writeThreadState(b, memory, hostClock, data, enable);
    } else if (!write.holds) {
      writeThreadState(b, memory, targetClock, write.data, write.enable);
    }
    if (commitSelect && threads == 1 && !write.holds && !gate) {
      Value data = write.data;
      if (reg.getPreset()) {
        auto width = hw::getBitWidth(reg.getType());
        data = comb::MuxOp::create(
            b, initializing,
            hw::ConstantOp::create(b, reg.getPreset()->zextOrTrunc(width)),
            b.createOrFold<hw::BitcastOp>(b.getIntegerType(width), write.data));
      }
      auto single = memory.getDefiningOp<seq::FirRegOp>();
      auto loc = b.getLoc();
      if (compactStaging)
        b.setLoc(single.getLoc());
      // The candidate and its enable, taken every host cycle.
      Value candidate = seq::FirRegOp::create(
          b, b.createOrFold<hw::BitcastOp>(single.getType(), data), hostClock,
          b.getStringAttr(single.getName() + "_next"));
      Value wen;
      if (write.enable)
        wen = seq::FirRegOp::create(b, write.enable, hostClock,
                                    b.getStringAttr(single.getName() + "_wen"),
                                    hw::InnerSymAttr{},
                                    b.getIntegerAttr(i1, 0));
      b.setLoc(loc);
      singleRegisters.push_back(
          {single, candidate, wen, bool(reg.getPreset())});
    }
    if (regions) {
      auto &state = regionMemory(memory);
      state.state = true;
      state.loc = reg.getLoc();
      for (auto *user : memory.getUsers()) {
        if (auto read = dyn_cast<seq::FirMemReadOp>(user))
          state.reads.push_back({read, {}, /*registered=*/false, {}});
        else if (auto port = dyn_cast<seq::FirMemWriteOp>(user))
          state.writes.push_back({port, {}, write.enable, !write.holds,
                                  bool(reg.getPreset()), /*staged=*/true});
      }
    }
    reg.erase();
  }

  // Each memory holds every thread's contents, the thread in the low address
  // bits, and is written on the target clock. A registered read reads on the
  // target clock too, and its data is kept per thread in a buffer filled on
  // the next host cycle, so that the memory stays a block RAM.
  // Each memory sees the target's edges by registers of its own, kept, as
  // it counts its read's thread: one model-wide pair would drive every
  // buffer's write enable.
  auto keepReg = [&](seq::FirRegOp reg) {
    sv::addSVAttributes(reg, {sv::SVAttributeAttr::get(
                                 b.getContext(), "keep", R"("true")",
                                 /*emitAsComment=*/false)});
    return reg;
  };
  auto getEdgeSeen = [&](const Twine &prefix) -> Value {
    auto zero = b.getIntegerAttr(i1, 0);
    auto toggleNext = backedges.get(i1);
    auto toggle = keepReg(seq::FirRegOp::create(
        b, toggleNext, targetClock, b.getStringAttr(prefix + "_toggle"),
        hw::InnerSymAttr{}, zero));
    toggleNext.setValue(
        comb::XorOp::create(b, toggle, hw::ConstantOp::create(b, i1, 1)));
    auto seen = keepReg(seq::FirRegOp::create(
        b, toggle, hostClock, b.getStringAttr(prefix + "_toggle_seen"),
        hw::InnerSymAttr{}, zero));
    return comb::XorOp::create(b, toggle, seen);
  };
  auto idxWidth = threads > 1 ? threadIdx.getType().getIntOrFloatBitWidth() : 0;
  for (auto memory : memories) {
    auto memType = memory.getType();
    auto depth = memType.getDepth() * threads;
    auto threadedType = seq::FirMemType::get(
        b.getContext(), depth, memType.getWidth(), memType.getMaskWidth());
    auto threadedMemory = seq::FirMemOp::create(
        b, threadedType, memory.getReadLatency(), memory.getWriteLatency(),
        memory.getRuw(), memory.getWuw(), memory.getNameAttr(),
        hw::InnerSymAttr{}, seq::FirMemInitAttr{}, memory.getPrefixAttr(),
        memory.getOutputFileAttr());
    threadedMemory->setDialectAttrs(memory->getDialectAttrs());
    if (auto fold = foldOfStore.find(memory); fold != foldOfStore.end())
      folds[fold->second].threadedStore = threadedMemory;
    unsigned addrWidth = llvm::Log2_64_Ceil(depth) - idxWidth;
    Value lastLow;
    auto addressOf = [&](Value addr) -> Value {
      lastLow = {};
      if (!idxWidth)
        return addr;
      if (!addrWidth)
        return threadIdx;
      Value low =
          b.createOrFold<comb::ExtractOp>(b.getIntegerType(addrWidth), addr, 0);
      lastLow = low;
      return b.createOrFold<comb::ConcatOp>(low, threadIdx);
    };
    for (auto *user : llvm::make_early_inc_range(memory->getUsers())) {
      if (auto write = dyn_cast<seq::FirMemWriteOp>(user)) {
        auto port = seq::FirMemWriteOp::create(
            b, threadedMemory, addressOf(write.getAddress()), targetClock,
            write.getEnable(), write.getData(), write.getMask());
        // Without staged RAM writes, a registered read on the edge of a write
        // to its address reads what the memory's read-under-write says, so
        // such a memory's writes land on the edge.
        bool staged = memory.getReadLatency() == 0 || stagedRamWrites;
        if (regions)
          regionMemory(threadedMemory).loc = memory.getLoc();
        if (regions)
          regionMemory(threadedMemory)
              .writes.push_back({port, lastLow, write.getEnable(),
                                 /*edge=*/true, /*preset=*/false, staged});
        write.erase();
        continue;
      }
      auto read = cast<seq::FirMemReadOp>(user);
      Value data;
      if (memory.getReadLatency() == 0) {
        auto port = seq::FirMemReadOp::create(b, threadedMemory,
                                              addressOf(read.getAddress()),
                                              hostClock, read.getEnable());
        if (regions)
          regionMemory(threadedMemory)
              .reads.push_back({port, lastLow, /*registered=*/false, {}});
        data = port;
      } else {
        // A registered read's data is only there on the host cycle after the
        // target edge that read it, even for one thread: once the target
        // clock runs on the host clock with an enable, so does the read. The
        // thread keeps it until its next read.
        auto doutPort = seq::FirMemReadOp::create(
            b, threadedMemory, addressOf(read.getAddress()), targetClock,
            read.getEnable());
        if (regions)
          regionMemory(threadedMemory)
              .reads.push_back({doutPort, lastLow,
                                /*registered=*/stagedRamWrites,
                                read.getEnable()});
        Value dout = doutPort;
        Value enable = read.getEnable()
                           ? read.getEnable()
                           : Value(hw::ConstantOp::create(b, i1, 1));
        auto readEdge = seq::FirRegOp::create(b, enable, targetClock,
                                              b.getStringAttr("read_edge"));
        Value buffer;
        data = createThreadState(
            b, (memory.getName().value_or("mem") + "_rdata").str(),
            dout.getType(), buffer);
        std::string readName = (memory.getName().value_or("mem") + "_read").str();
        Value capture =
            comb::AndOp::create(b, getEdgeSeen(readName), readEdge);
        if (threads == 1) {
          // The thread runs again on the host cycle that fills the buffer, so
          // it reads the data as it comes. With more threads, that cycle is
          // another thread's: the next thread is never one that just finished.
          data = comb::MuxOp::create(b, capture, dout, data);
          writeThreadState(b, buffer, hostClock, data);
        } else {
          // In turn, with registered state, the buffer is read a host cycle
          // ahead too: filled on the host cycle after its thread's edge, it
          // holds the data from the second, and its thread is read ahead
          // `threads * phases - 1` host cycles after the edge.
          if (regions && aheadIdx && !readySchedule && threads * phases >= 3) {
            auto &state = regionMemory(buffer);
            state.state = true;
            state.loc = memory.getLoc();
            for (auto *user : buffer.getUsers())
              if (auto port = dyn_cast<seq::FirMemReadOp>(user))
                state.reads.push_back({port, {}, /*registered=*/false, {}});
          }
          // With a preset, so that a split memory's parts can copy it.
          auto readThread = keepReg(seq::FirRegOp::create(
              b, threadIdx, targetClock,
              b.getStringAttr(readName + "_thread"), hw::InnerSymAttr{},
              b.getIntegerAttr(threadIdx.getType(), 0)));
          seq::FirMemWriteOp::create(b, buffer, readThread, hostClock, capture,
                                     dout, Value{});
        }
      }
      read.getData().replaceAllUsesWith(data);
      replaced[read.getData()] = data;
      read.erase();
    }
    memory.erase();
  }

  auto resolve = [&](Value value) {
    auto it = replaced.find(value);
    return it == replaced.end() ? value : it->second;
  };
  // A channel of some bits of an output carries just those.
  auto sliceOf = [&](Channel &channel, Value value) -> Value {
    if (!value || !channel.sliced)
      return value;
    auto width = hw::getBitWidth(value.getType());
    Value bits = b.createOrFold<hw::BitcastOp>(b.getIntegerType(width), value);
    return b.createOrFold<comb::ExtractOp>(channel.type, bits, channel.low);
  };
  for (auto &channel : outputs) {
    channel.current = sliceOf(channel, resolve(channel.current));
    channel.next = sliceOf(channel, resolve(channel.next));
  }

  // The channel flags of the current thread.
  auto one = hw::ConstantOp::create(b, i1, 1);
  auto zero = hw::ConstantOp::create(b, i1, 0);
  auto notOf = [&](Value v) { return comb::XorOp::create(b, v, one); };
  SmallVector<Value> inFired, inMem, outFired, outMem;
  for (auto &channel : inputs) {
    inMem.emplace_back();
    inFired.push_back(createThreadState(
        b, (channel.name.getValue() + "_fired").str(), i1, inMem.back()));
  }
  for (auto &channel : outputs) {
    outMem.emplace_back();
    outFired.push_back(createThreadState(
        b, (channel.name.getValue() + "_fired").str(), i1, outMem.back()));
  }
  // Whether the target clock ticks at the end of this cycle: with a clock
  // channel, the edge its last token enabled; without, every cycle but the
  // first after reset.
  Value clkEnabledMem;
  Value clkEnabled =
      tickFromFirst ? Value(one)
                    : createThreadState(b,
                                        clockChannel.empty()
                                            ? "clk_enabled"
                                            : (clockChannel + "_enabled").str(),
                                        i1, clkEnabledMem);

  // The thread finishes its target cycle when every input is valid and every
  // output has fired or fires now.
  auto finishing = backedges.get(i1);
  SmallVector<Value> outValid, firedOrFiring, finishTerms;
  for (auto [i, channel] : llvm::enumerate(outputs)) {
    Value fired = outFired[i], ready = sourceReady[i];
    // An idle host cycle offers no token, so none fires on it.
    if (idle)
      ready = comb::AndOp::create(b, ready, notOf(idle));
    Value valid, fof;
    if (channel.eager) {
      valid = comb::OrOp::create(b, notOf(fired),
                                 comb::AndOp::create(b, finishing, fired));
      fof = comb::OrOp::create(b, fired,
                               comb::AndOp::create(b, ready, notOf(fired)));
    } else {
      SmallVector<Value> terms;
      for (auto dep : channel.deps.set_bits())
        terms.push_back(sinkValid[dep]);
      terms.push_back(notOf(fired));
      valid = comb::AndOp::create(b, ValueRange(terms), false);
      fof = comb::OrOp::create(b, fired, comb::AndOp::create(b, ready, valid));
    }
    outValid.push_back(valid);
    firedOrFiring.push_back(fof);
    finishTerms.push_back(fof);
  }
  finishTerms.append(sinkValid.begin(), sinkValid.end());
  // No thread finishes, takes or offers a token while they are initialized,
  // nor on an idle host cycle of its turn.
  Value running = notOf(initializing);
  if (idle)
    running = comb::AndOp::create(b, running, notOf(idle));
  finishTerms.push_back(running);
  finishing.setValue(
      finishTerms.size() == 1
          ? finishTerms[0]
          : comb::AndOp::create(b, ValueRange(finishTerms), false));
  for (auto &valid : outValid)
    valid = comb::AndOp::create(b, valid, running);

  if (threads > 1) {
    auto idxType = threadIdx.getType();
    auto constant = [&](Type type, uint64_t value) -> Value {
      return hw::ConstantOp::create(b, type, value);
    };
    // With registered state, the thread chosen now runs after the next one,
    // `aheadIdx`, which it must not be: that one's state is read before it
    // finishes.
    Value from = aheadIdx ? aheadIdx : threadIdx;
    auto isLast = comb::ICmpOp::create(b, comb::ICmpPredicate::eq, from,
                                       constant(idxType, threads - 1));
    Value inTurn = comb::MuxOp::create(
        b, isLast, constant(idxType, 0),
        comb::AddOp::create(b, from, constant(idxType, 1)));
    if (aheadIdx)
      idxNext.setValue(aheadIdx);
    auto &chosen = aheadIdx ? aheadNext : idxNext;
    if (scheduleNext) {
      // The position from outside: the thread in the high bits.
      unsigned phaseBits = phases > 1 ? llvm::Log2_32(phases) : 0;
      chosen.setValue(b.createOrFold<comb::ExtractOp>(idxType, scheduleNext,
                                                      phaseBits));
      if (aheadPhase)
        aheadPhaseNext.setValue(b.createOrFold<comb::ExtractOp>(
            aheadPhase.getType(), scheduleNext, 0));
      separatesFinishes = true;
    } else if (!threadReady && aheadPhase) {
      // The next in turn is another thread once the turn's idle host cycles
      // are over.
      auto phaseType = aheadPhase.getType();
      aheadPhaseNext.setValue(
          comb::AddOp::create(b, aheadPhase, constant(phaseType, 1)));
      chosen.setValue(comb::MuxOp::create(
          b,
          comb::ICmpOp::create(b, comb::ICmpPredicate::eq, aheadPhase,
                               constant(phaseType, phases - 1)),
          inTurn, aheadIdx));
      separatesFinishes = true;
    } else if (!threadReady) {
      // The next in turn is another thread.
      chosen.setValue(inTurn);
      separatesFinishes = true;
    } else {
      // A thread finishing now takes its inputs: it is not ready next.
      auto vecType = b.getIntegerType(threads);
      Value ones = hw::ConstantOp::create(b, APInt::getAllOnes(threads));
      auto oneHot = [&](Value index) -> Value {
        return comb::ShlOp::create(
            b, constant(vecType, 1),
            b.createOrFold<comb::ConcatOp>(
                constant(b.getIntegerType(threads -
                                          idxType.getIntOrFloatBitWidth()),
                         0),
                index));
      };
      Value current = oneHot(from);
      Value finishingMask = comb::AndOp::create(
          b, aheadIdx ? oneHot(threadIdx) : current,
          comb::ReplicateOp::create(b, vecType, finishing));
      Value ready = comb::AndOp::create(
          b, threadReady, comb::XorOp::create(b, finishingMask, ones));
      if (aheadIdx)
        ready = comb::AndOp::create(b, ready,
                                    comb::XorOp::create(b, current, ones));
      // The threads after the current one.
      Value upTo = comb::SubOp::create(
          b, comb::ShlOp::create(b, current, constant(vecType, 1)),
          constant(vecType, 1));
      Value after =
          comb::AndOp::create(b, ready, comb::XorOp::create(b, upTo, ones));
      // The lowest set bit of `v`, as a thread index.
      auto lowest = [&](Value v) -> Value {
        Value index = constant(idxType, 0);
        for (unsigned t = threads; t-- > 0;)
          index = comb::MuxOp::create(
              b, comb::ExtractOp::create(b, b.getI1Type(), v, t),
              constant(idxType, t), index);
        return index;
      };
      auto any = [&](Value v) -> Value {
        return comb::ICmpOp::create(b, comb::ICmpPredicate::ne, v,
                                    constant(vecType, 0));
      };
      // Without a ready thread, the next in turn; with the state written two
      // host cycles late, the next in turn that is neither the thread after
      // the current one nor the current one if it finishes now.
      Value fallback = inTurn;
      if (deepState()) {
        Value allowed = comb::AndOp::create(
            b, comb::XorOp::create(b, current, ones),
            comb::XorOp::create(b, finishingMask, ones));
        Value allowedAfter =
            comb::AndOp::create(b, allowed, comb::XorOp::create(b, upTo, ones));
        fallback = comb::MuxOp::create(b, any(allowedAfter),
                                       lowest(allowedAfter), lowest(allowed));
      }
      // A thread finishing now is not ready, and the fallbacks are other
      // threads: in turn while the threads are initialized, too.
      separatesFinishes = true;
      chosen.setValue(comb::MuxOp::create(
          b, sweeping, inTurn,
          comb::MuxOp::create(
              b, any(after), lowest(after),
              comb::MuxOp::create(b, any(ready), lowest(ready), fallback))));
    }
  }

  auto unlessReset = [&](Value v) {
    return comb::MuxOp::create(b, initializing, zero, v);
  };
  // With a clock channel, a cycle whose edge is not enabled has no tokens: its
  // outputs start out fired, as do its inputs.
  for (auto [i, channel] : llvm::enumerate(outputs)) {
    Value fired = outFired[i];
    Value onFinish = channel.eager
                         ? Value(comb::AndOp::create(b, fired, sourceReady[i]))
                         : Value(zero);
    if (clockBits)
      onFinish = comb::OrOp::create(b, notOf(clkEnabled), onFinish);
    writeThreadState(b, outMem[i], hostClock,
                     unlessReset(comb::MuxOp::create(b, finishing, onFinish,
                                                     firedOrFiring[i])));
  }
  SmallVector<Value> sinkReady;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    Value ready = comb::AndOp::create(b, finishing, notOf(inFired[i]));
    sinkReady.push_back(ready);
    Value accepted = comb::OrOp::create(
        b, inFired[i], comb::AndOp::create(b, ready, sinkValid[i]));
    if (clockBits) {
      // Fired until the first clock token says which inputs a cycle takes.
      writeThreadState(
          b, inMem[i], hostClock,
          comb::MuxOp::create(
              b, initializing, one,
              comb::MuxOp::create(b, finishing, notOf(clockBits), accepted)));
      continue;
    }
    writeThreadState(
        b, inMem[i], hostClock,
        unlessReset(comb::MuxOp::create(b, finishing, zero, accepted)));
  }

  // The target clock ticks for the thread when it finishes, from its second
  // finish after reset on.
  if (!tickFromFirst)
    writeThreadState(
        b, clkEnabledMem, hostClock,
        unlessReset(clockBits
                        ? Value(comb::MuxOp::create(b, finishing, clockBits,
                                                    clkEnabled))
                        : Value(comb::OrOp::create(b, finishing, clkEnabled))));
  Value enable = comb::AndOp::create(b, clkEnabled, finishing);
  targetEdge.setValue(enable);
  if (gate) {
    auto clockGate =
        hw::InstanceOp::create(b, gate, b.getStringAttr("clk_buffer"),
                               ArrayRef<Value>{hostClock, enable});
    targetClock.setValue(clockGate.getResult(0));
  } else {
    // A kept black box: on the host clock, enabled on the target's edges.
    for (auto *op : keptBoxes) {
      auto inst = cast<hw::InstanceOp>(op);
      auto *module = SymbolTable::lookupNearestSymbolFrom(
          inst, inst.getReferencedModuleNameAttr());
      auto contract = **foldsim::getPipelineContract(module);
      for (auto &port : cast<hw::HWModuleLike>(module).getPortList()) {
        if (port.dir != hw::ModulePort::Direction::Input)
          continue;
        auto operand = inst.getOperand(port.argNum);
        if (port.getName() == contract.clock) {
          auto conv = operand.getDefiningOp<seq::FromClockOp>();
          inst->setOperand(port.argNum,
                           conv ? Value(seq::FromClockOp::create(b, hostClock))
                                : Value(hostClock));
          if (conv && conv->use_empty())
            conv.erase();
        } else if (port.getName() == contract.enable) {
          inst->setOperand(port.argNum,
                           comb::AndOp::create(b, operand, enable));
        }
      }
    }
  }
  if (!gate && failed(enableOnHostClock(b, targetClock, enable))) {
    return failure();
  }

  if (foldEdges && failed(finishFolds(b, enable, hostReset)))
    return failure();

  SmallVector<Value> results;
  if (clockBits)
    results.push_back(finishing);
  results.append(sinkReady.begin(), sinkReady.end());
  for (auto [i, channel] : llvm::enumerate(outputs)) {
    results.push_back(outValid[i]);
    if (hw::getBitWidth(channel.type) == 0)
      continue;
    // An eager output that has fired offers the next cycle's value, which is
    // the current one if the target clock does not tick on this finish.
    Value bits = channel.eager
                     ? Value(comb::MuxOp::create(
                           b,
                           b.createOrFold<comb::AndOp>(outFired[i], clkEnabled),
                           channel.next, channel.current))
                     : channel.current;
    bits = b.createOrFold<hw::BitcastOp>(
        b.getIntegerType(hw::getBitWidth(channel.type)), bits);
    if (channel.fields.empty()) {
      results.push_back(bits);
      continue;
    }
    unsigned high = hw::getBitWidth(channel.type);
    for (auto &field : channel.fields) {
      auto width = hw::getBitWidth(field.type);
      high -= width;
      results.push_back(
          b.createOrFold<comb::ExtractOp>(b.getIntegerType(width), bits, high));
    }
  }
  if (threadIndexPort && threads > 1)
    results.push_back(threadIdx);
  if (threadIndexPort)
    results.push_back(enable);
  if (scheduleNext) {
    // The position register: the next host cycle's with registered state.
    SmallVector<Value> position;
    if (threads > 1)
      position.push_back(aheadIdx ? aheadIdx : threadIdx);
    if (phases > 1)
      position.push_back(aheadPhase ? aheadPhase : phase);
    results.push_back(b.createOrFold<comb::ConcatOp>(position));
  }
  auto output = hw::OutputOp::create(b, results);
  // Regions and registered state replace values the outputs may be, so they
  // come last.
  b.setInsertionPoint(output);
  if (regions)
    buildRegions(b, enable, initializing, hostReset);
  if (commitSelect && threads == 1 && !gate)
    commitBySelect(b, enable, initializing);
  b.setInsertionPointToEnd(body);

  // A single thread's state that is never written holds its value.
  for (auto &[memory, entry] : unwritten) {
    auto [next, regClock] = entry;
    next.setValue(memory);
    regClock.setValue(hostClock);
  }
  unwritten.clear();

  if (readPortDepth)
    pipelineReadPorts(b, threaded);

  // Which output channels fire eagerly is part of the model's timing, which
  // the simulator around it relies on.
  SmallVector<Attribute> eager;
  for (auto &channel : outputs)
    if (channel.eager)
      eager.push_back(channel.name);
  threaded->setAttr("foldsim.eager_channels", b.getArrayAttr(eager));
  if (!bypasses.empty())
    threaded->setAttr("foldsim.bypass", b.getArrayAttr(bypasses));
  model.erase();
  return success();
}

unsigned foldsim::turnPhases(unsigned threads, unsigned stateStages) {
  if (threads < 2)
    return 1;
  unsigned slots = stateStages > 1 ? 4 : 3, k = 1;
  while (threads * k < slots)
    k *= 2;
  return k;
}

FailureOr<hw::HWModuleOp>
foldsim::threadModel(hw::HWModuleOp model, unsigned threads, StringAttr name,
                     bool readySchedule, const ThreadOptions &options) {
  auto *parent = model->getParentOp();
  Threader::Slices slices;
  Threader::Fields fields;
  if (failed(Threader(model, threads, name, /*gate=*/{}, slices, fields,
                      /*clockChannel=*/"", /*threadIndexPort=*/true,
                      /*tickFromFirst=*/true, readySchedule, options)
                 .run()))
    return failure();
  return cast<hw::HWModuleOp>(SymbolTable::lookupSymbolIn(parent, name));
}

void FAME5Pass::runOnOperation() {
  auto top = getOperation();
  auto model = dyn_cast_or_null<hw::HWModuleOp>(
      SymbolTable::lookupSymbolIn(top, moduleName));
  if (!model) {
    top.emitError("no module '") << moduleName << "' to thread";
    return signalPassFailure();
  }
  if (threads < 1) {
    model.emitError("a model simulates at least one instance");
    return signalPassFailure();
  }
  if (!clockChannel.empty() && threads > 1) {
    model.emitError("a clock channel is for a model of one instance");
    return signalPassFailure();
  }

  // The clock gate the platform provides.
  auto gate = dyn_cast_or_null<hw::HWModuleExternOp>(
      SymbolTable::lookupSymbolIn(top, "ClockGate"));
  if (!gate && !clockEnable) {
    OpBuilder builder(top.getBody(), top.getBody()->begin());
    auto clock = seq::ClockType::get(&getContext());
    SmallVector<hw::PortInfo> ports(3);
    ports[0] = {
        {builder.getStringAttr("I"), clock, hw::ModulePort::Direction::Input},
        0};
    ports[1] = {{builder.getStringAttr("CE"), builder.getI1Type(),
                 hw::ModulePort::Direction::Input},
                1};
    ports[2] = {
        {builder.getStringAttr("O"), clock, hw::ModulePort::Direction::Output},
        0};
    gate = hw::HWModuleExternOp::create(
        builder, top.getLoc(), builder.getStringAttr("ClockGate"),
        hw::ModulePortInfo(ports));
  }

  auto name = threadedName.empty()
                  ? StringAttr::get(&getContext(),
                                    moduleName + "_threaded_" + moduleName)
                  : StringAttr::get(&getContext(), threadedName);
  Threader::Slices slices;
  for (auto &slice : portSlices) {
    SmallVector<StringRef, 3> fields;
    StringRef(slice).split(fields, ':');
    unsigned hi, lo;
    if (fields.size() != 3 || fields[1].getAsInteger(10, hi) ||
        fields[2].getAsInteger(10, lo)) {
      model.emitError("port slice '") << slice << "' is not <port>:<hi>:<lo>";
      return signalPassFailure();
    }
    slices[fields[0]].push_back({hi, lo});
  }
  Threader::Fields channelFields;
  for (auto [rank, entry] : llvm::enumerate(this->channelFields)) {
    auto [channelField, port] = StringRef(entry).split('=');
    auto [channel, field] = channelField.split('.');
    if (port.empty() || field.empty()) {
      model.emitError("channel field '")
          << entry << "' is not <channel>.<field>=<port>";
      return signalPassFailure();
    }
    channelFields[port] = {channel.str(), field.str(), unsigned(rank)};
  }
  if (clockEnable)
    gate = {};
  if (stateStages < 1 || stateStages > 2) {
    model.emitError("a thread's state is written one or two host cycles "
                    "after its edge");
    return signalPassFailure();
  }
  if ((regionLoads || registeredState || commitSelect || stagedRamWrites ||
       pipelineMultipliers || compactStaging || foldPipelines || idleSlots ||
       pipelineDepth || memoryLoads || readPortDepth || unreverseAddresses) &&
      !clockEnable) {
    model.emitError("regions are for state written with `clock-enable`");
    return signalPassFailure();
  }
  foldsim::ThreadOptions threadOptions{regionLoads,         registeredState,
                                       stateStages,         commitSelect,
                                       stagedRamWrites,     pipelineMultipliers,
                                       compactStaging,      foldPipelines,
                                       idleSlots,           pipelineDepth};
  threadOptions.memoryLoads = memoryLoads;
  threadOptions.readPortDepth = readPortDepth;
  threadOptions.unreverseAddresses = unreverseAddresses;
  if (failed(Threader(model, threads, name, gate, slices, channelFields,
                      clockChannel, threadIndex, /*tickFromFirst=*/false,
                      /*readySchedule=*/false,
                      threadOptions)
                 .run()))
    return signalPassFailure();
}
