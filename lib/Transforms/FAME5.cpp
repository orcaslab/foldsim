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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "foldsim/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <numeric>

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
           bool tickFromFirst = false, bool readySchedule = false)
      : model(model), threads(threads), name(name), gate(gate), slices(slices),
        fields(fields), clockChannel(clockChannel),
        threadIndexPort(threadIndexPort), tickFromFirst(tickFromFirst),
        readySchedule(readySchedule) {}

  LogicalResult run();

private:
  LogicalResult checkModel();
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
  struct Write {
    bool holds;
    Value enable, data;
  };
  /// The write of a register whose next value is `next`: the branches of its
  /// muxes that hold the register are the cycles it is not written.
  Write peelHolds(ImplicitLocOpBuilder &b, Value next, Value reg);
  /// Moves everything on the target clock to the host clock, written only
  /// when `enable`.
  LogicalResult enableOnHostClock(ImplicitLocOpBuilder &b, Value targetClock,
                                  Value enable);

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
    if (isa<hw::OutputOp, seq::FirMemReadOp, seq::FirMemWriteOp>(op))
      continue;
    if (isa<hw::HWInstanceLike>(op) || op.getDialect()->getNamespace() == "seq")
      return op.emitError("a threaded model must be flat and keep its state "
                          "in registers and memories");
  }
  for (auto &use : body->getArgument(clockArg).getUses())
    if (!isa<seq::FirRegOp, seq::FirMemReadOp, seq::FirMemWriteOp>(
            use.getOwner()))
      return use.getOwner()->emitError(
          "clock used other than by a register or memory");
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
             isa<seq::FirRegOp, hw::ConstantOp, hw::AggregateConstantOp>(op)) {
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

Threader::Write Threader::peelHolds(ImplicitLocOpBuilder &b, Value next,
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

LogicalResult Threader::run() {
  if (failed(checkModel()))
    return failure();
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
    threadIdx = seq::FirRegOp::create(
        b, idxNext, hostClock, b.getStringAttr("threadIdx"), hostReset,
        hw::ConstantOp::create(b, idxType, 0), hw::InnerSymAttr{},
        /*isAsync=*/false, b.getIntegerAttr(idxType, 0));
    auto sweepNext = backedges.get(i1);
    sweeping = seq::FirRegOp::create(
        b, sweepNext, hostClock, b.getStringAttr("init_sweep"), hostReset,
        hw::ConstantOp::create(b, i1, 1), hw::InnerSymAttr{},
        /*isAsync=*/false, b.getIntegerAttr(i1, 1));
    Value last = comb::ICmpOp::create(
        b, comb::ICmpPredicate::eq, threadIdx,
        hw::ConstantOp::create(b, idxType, threads - 1));
    sweepNext.setValue(comb::AndOp::create(
        b, sweeping,
        comb::XorOp::create(b, last, hw::ConstantOp::create(b, i1, 1))));
    initializing = comb::OrOp::create(b, hostReset, sweeping);
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
  for (auto reg : registers) {
    Value next = reg.getNext();
    if (reg.getReset())
      next = comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), next);
    auto write = peelHolds(b, next, reg.getResult());
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
    reg.erase();
  }

  // Each memory holds every thread's contents, the thread in the low address
  // bits, and is written on the target clock. A registered read reads on the
  // target clock too, and its data is kept per thread in a buffer filled on
  // the next host cycle, so that the memory stays a block RAM.
  Value edgeSeen;
  auto getEdgeSeen = [&]() {
    if (edgeSeen)
      return edgeSeen;
    auto zero = b.getIntegerAttr(i1, 0);
    auto toggleNext = backedges.get(i1);
    auto toggle = seq::FirRegOp::create(b, toggleNext, targetClock,
                                        b.getStringAttr("target_toggle"),
                                        hw::InnerSymAttr{}, zero);
    toggleNext.setValue(
        comb::XorOp::create(b, toggle, hw::ConstantOp::create(b, i1, 1)));
    auto seen = seq::FirRegOp::create(b, toggle, hostClock,
                                      b.getStringAttr("target_toggle_seen"),
                                      hw::InnerSymAttr{}, zero);
    edgeSeen = comb::XorOp::create(b, toggle, seen);
    return edgeSeen;
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
    unsigned addrWidth = llvm::Log2_64_Ceil(depth) - idxWidth;
    auto addressOf = [&](Value addr) -> Value {
      if (!idxWidth)
        return addr;
      if (!addrWidth)
        return threadIdx;
      Value low =
          b.createOrFold<comb::ExtractOp>(b.getIntegerType(addrWidth), addr, 0);
      return b.createOrFold<comb::ConcatOp>(low, threadIdx);
    };
    for (auto *user : llvm::make_early_inc_range(memory->getUsers())) {
      if (auto write = dyn_cast<seq::FirMemWriteOp>(user)) {
        seq::FirMemWriteOp::create(
            b, threadedMemory, addressOf(write.getAddress()), targetClock,
            write.getEnable(), write.getData(), write.getMask());
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
        // A registered read's data is only there on the host cycle after the
        // target edge that read it, even for one thread: once the target
        // clock runs on the host clock with an enable, so does the read. The
        // thread keeps it until its next read.
        Value dout = seq::FirMemReadOp::create(b, threadedMemory,
                                               addressOf(read.getAddress()),
                                               targetClock, read.getEnable());
        Value enable = read.getEnable()
                           ? read.getEnable()
                           : Value(hw::ConstantOp::create(b, i1, 1));
        auto readEdge = seq::FirRegOp::create(b, enable, targetClock,
                                              b.getStringAttr("read_edge"));
        Value buffer;
        data = createThreadState(
            b, (memory.getName().value_or("mem") + "_rdata").str(),
            dout.getType(), buffer);
        Value capture = comb::AndOp::create(b, getEdgeSeen(), readEdge);
        if (threads == 1) {
          // The thread runs again on the host cycle that fills the buffer, so
          // it reads the data as it comes. With more threads, that cycle is
          // another thread's: the next thread is never one that just finished.
          data = comb::MuxOp::create(b, capture, dout, data);
          writeThreadState(b, buffer, hostClock, data);
        } else {
          auto readThread = seq::FirRegOp::create(
              b, threadIdx, targetClock, b.getStringAttr("read_thread"));
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
  // No thread finishes, takes or offers a token while they are initialized.
  Value running = notOf(initializing);
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
    auto isLast = comb::ICmpOp::create(b, comb::ICmpPredicate::eq, threadIdx,
                                       constant(idxType, threads - 1));
    Value inTurn = comb::MuxOp::create(
        b, isLast, constant(idxType, 0),
        comb::AddOp::create(b, threadIdx, constant(idxType, 1)));
    if (!threadReady) {
      idxNext.setValue(inTurn);
    } else {
      // A thread finishing now takes its inputs: it is not ready next.
      auto vecType = b.getIntegerType(threads);
      Value ones = hw::ConstantOp::create(b, APInt::getAllOnes(threads));
      Value current = comb::ShlOp::create(
          b, constant(vecType, 1),
          b.createOrFold<comb::ConcatOp>(
              constant(
                  b.getIntegerType(threads - idxType.getIntOrFloatBitWidth()),
                  0),
              threadIdx));
      Value finishingMask = comb::AndOp::create(
          b, current, comb::ReplicateOp::create(b, vecType, finishing));
      Value ready = comb::AndOp::create(
          b, threadReady, comb::XorOp::create(b, finishingMask, ones));
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
      // In turn while the threads are initialized.
      idxNext.setValue(comb::MuxOp::create(
          b, sweeping, inTurn,
          comb::MuxOp::create(
              b, any(after), lowest(after),
              comb::MuxOp::create(b, any(ready), lowest(ready), inTurn))));
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
  } else if (failed(enableOnHostClock(b, targetClock, enable))) {
    return failure();
  }

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
  hw::OutputOp::create(b, results);

  // A single thread's state that is never written holds its value.
  for (auto &[memory, entry] : unwritten) {
    auto [next, regClock] = entry;
    next.setValue(memory);
    regClock.setValue(hostClock);
  }
  unwritten.clear();

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

FailureOr<hw::HWModuleOp> foldsim::threadModel(hw::HWModuleOp model,
                                                     unsigned threads,
                                                     StringAttr name,
                                                     bool readySchedule) {
  auto *parent = model->getParentOp();
  Threader::Slices slices;
  Threader::Fields fields;
  if (failed(Threader(model, threads, name, /*gate=*/{}, slices, fields,
                      /*clockChannel=*/"", /*threadIndexPort=*/true,
                      /*tickFromFirst=*/true, readySchedule)
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
  if (failed(Threader(model, threads, name, gate, slices, channelFields,
                      clockChannel, threadIndex)
                 .run()))
    return signalPassFailure();
}
