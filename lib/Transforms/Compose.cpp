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
#include "llvm/Support/Debug.h"
#include <map>

namespace foldsim {
#define GEN_PASS_DEF_COMPOSE
#include "foldsim/Passes.h.inc"
} // namespace foldsim

#define DEBUG_TYPE "foldsim-compose"

using namespace mlir;
using namespace circt;

namespace {

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
  /// The model, erased once threaded, and its name.
  hw::HWModuleOp model;
  std::string name;
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
  /// Whether every thread reads the same bits of the same producers.
  bool regular = false, direct = false;
  /// Built: for each thread, for each piece, the count of tokens read.
  SmallVector<SmallVector<Value>> deq;
};

class Composer {
public:
  Composer(ModuleOp top, hw::HWModuleOp netlist, StringRef simName,
           bool readySchedule, ArrayRef<std::string> relay)
      : top(top), netlist(netlist), simName(simName),
        readySchedule(readySchedule) {
    for (auto &name : relay)
      relayModels.insert(name);
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

  /// The read ports of a copy of a token memory.
  static constexpr unsigned kMaxReadPorts = 8;
  Value hostClock;

  ModuleOp top;
  hw::HWModuleOp netlist;
  StringRef simName;
  bool readySchedule;
  llvm::StringSet<> relayModels;
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
        if (auto inner = dyn_cast<hw::InstanceOp>(clone))
          inner.setInstanceNameAttr(builder.getStringAttr(
              instance.getInstanceName() + "_" + inner.getInstanceName()));
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
      for (auto name : instance.getArgNames())
        cls.inputNames.push_back(cast<StringAttr>(name).getValue().str());
      for (auto name : instance.getResultNames())
        cls.outputNames.push_back(cast<StringAttr>(name).getValue().str());
    }
    auto &cls = classes[it->second];
    threadOf[instance] = {it->second, cls.instances.size()};
    cls.instances.push_back(instance);
  }

  // Each class threaded, its channels found by name.
  for (auto &cls : classes) {
    auto name = StringAttr::get(top.getContext(), cls.name + "_threaded");
    auto threaded = foldsim::threadModel(cls.model, cls.instances.size(), name,
                                        readySchedule);
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

  // A producer read only regularly, for every thread, keeps its slots in a
  // memory read at the reader's thread.
  for (auto &producer : producers)
    producer.memory = producer.threads > 1 && producer.width > 0 &&
                      llvm::all_of(producer.reads, [&](auto &read) {
                        return consumers[read.consumer].regular;
                      });

  // A channel between two models, one of them relayed, gets a relay stage.
  for (auto &producer : producers) {
    if (producer.cls < 0 || producer.direct || producer.reads.empty() ||
        llvm::any_of(producer.reads, [&](auto &read) {
          return consumers[read.consumer].cls < 0;
        }))
      continue;
    bool mine = relayModels.contains(classes[producer.cls].name);
    producer.relay = llvm::any_of(producer.reads, [&](auto &read) {
      int c = consumers[read.consumer].cls;
      if (c == producer.cls)
        return false;
      auto &p = classes[producer.cls].name, &q = classes[c].name;
      return mine || relayModels.contains(q) ||
             relayModels.contains(p + ":" + q) ||
             relayModels.contains(q + ":" + p);
    });
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
    unsigned slots = producer.relay ? 4 : 2;
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
    Value data = producer.writeData;
    if (width != producer.width)
      data = b.createOrFold<comb::ExtractOp>(loc, b.getIntegerType(width), data,
                                             low);
    seq::FirMemWriteOp::create(b, loc, mem, producer.writeAddress, hostClock,
                               producer.writeEnable, data, Value{});
    memories.copies.push_back(mem);
  }
  return memories.copies.back();
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
  auto sim = hw::HWModuleOp::create(outer, netlist.getLoc(),
                                    StringAttr::get(context, simName),
                                    hw::ModulePortInfo(ports));
  auto *body = sim.getBodyBlock();
  body->getTerminator()->erase();
  auto b = ImplicitLocOpBuilder::atBlockEnd(netlist.getLoc(), body);
  BackedgeBuilder backedges(b, netlist.getLoc());
  hostClock = body->getArgument(0);
  Value hostReset = body->getArgument(1);
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
        bool relay = producers[piece.producer].relay;
        deq.push_back(counter(consumer.name + (relay ? "_rdeq" : "_deq") +
                                  Twine(t) + "_" + Twine(j),
                              relay ? 3 : 2));
      }
    }
  }
  for (auto [p, producer] : llvm::enumerate(producers)) {
    if (producer.direct)
      continue;
    auto source = sourceOf(p);
    unsigned countBits = producer.relay ? 3 : 2;
    unsigned numSlots = producer.relay ? 4 : 2;
    for (unsigned k = 0; k < producer.threads; ++k)
      producer.enq.push_back(
          counter(producer.name + (producer.relay ? "_renq" : "_enq") + Twine(k),
                  countBits));
    // Ready while every reader of the thread has read all but one token; with
    // a relay, while the reader's count of a host cycle ago leaves a slot.
    SmallVector<SmallVector<Value>> notFull(producer.threads);
    for (auto &read : producer.reads) {
      auto &consumer = consumers[read.consumer];
      auto &piece = consumer.pieces[read.thread][read.piece];
      Value deq = consumer.deq[read.thread][read.piece];
      if (producer.relay)
        deq = delayed(deq, consumer.name + "_deqd" + Twine(read.thread) + "_" +
                               Twine(read.piece));
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
    unsigned slotBits = producer.relay ? 2 : 1;
    auto slotOf = [&](Value count) {
      return comb::ExtractOp::create(b, b.getIntegerType(slotBits), count, 0);
    };
    // With a relay, the token, its thread and slot are staged a host cycle.
    Value writeFire = fire, writeThread = source.thread,
          writeBits = source.bits, writeSlot;
    if (producer.memory || producer.relay)
      writeSlot = slotOf(select(producer.enq, source.thread));
    if (producer.relay) {
      writeFire = delayed(fire, producer.name + "_stg_fire");
      if (source.thread)
        writeThread = delayed(source.thread, producer.name + "_stg_thread");
      writeSlot = delayed(writeSlot, producer.name + "_stg_slot");
      writeBits = seq::FirRegOp::create(b, source.bits, hostClock,
                                        b.getStringAttr(producer.name + "_stg_bits"));
    }
    if (producer.memory) {
      producer.writeAddress = comb::ConcatOp::create(b, writeThread, writeSlot);
      producer.writeEnable = writeFire;
      producer.writeData = writeBits;
      continue;
    }
    for (unsigned k = 0; k < producer.threads; ++k) {
      auto &slots = producer.slots.emplace_back();
      Value write = comb::AndOp::create(b, writeFire, isThread(writeThread, k));
      Value slot = producer.relay ? writeSlot : slotOf(producer.enq[k]);
      for (unsigned s = 0; s < numSlots; ++s) {
        auto next = backedges.get(writeBits.getType());
        auto reg = seq::FirRegOp::create(
            b, next, hostClock,
            b.getStringAttr(producer.name + (producer.relay ? "_rslot" : "_slot") +
                            Twine(k) + "_" + Twine(s)));
        Value at =
            producer.relay
                ? Value(comb::ICmpOp::create(b, comb::ICmpPredicate::eq, slot,
                                             constant(APInt(slotBits, s))))
            : s ? slot
                : Value(comb::XorOp::create(b, slot, bit(true)));
        Value here = comb::AndOp::create(b, write, at);
        next.setValue(comb::MuxOp::create(b, here, writeBits, reg));
        slots.push_back(reg);
      }
    }
  }

  // With a relay, each reader sees a producer thread's count a host cycle
  // late, from its own copy.
  std::map<std::tuple<unsigned, unsigned, unsigned>, Value> enqCopies;
  auto enqSeen = [&](unsigned c, unsigned p, unsigned k) -> Value {
    auto &producer = producers[p];
    if (!producer.relay)
      return producer.enq[k];
    auto &copy = enqCopies[{c, p, k}];
    if (!copy)
      copy = delayed(producer.enq[k], consumers[c].name + "_enqd_" +
                                          producer.name + "_" + Twine(k));
    return copy;
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
      auto &slots = producer.slots[piece.thread];
      Value data;
      if (producer.relay) {
        Value slot = comb::ExtractOp::create(b, b.getIntegerType(2),
                                             consumer.deq[t][j], 0);
        data = select(slots, slot);
      } else {
        Value slot = comb::ExtractOp::create(b, i1, consumer.deq[t][j], 0);
        data = comb::MuxOp::create(b, slot, slots[1], slots[0]);
      }
      return b.createOrFold<comb::ExtractOp>(b.getIntegerType(piece.width),
                                             data, piece.low);
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
    if (consumer.width > 0 && consumer.regular) {
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
                b, b.getIntegerType(producer.relay ? 2 : 1),
                consumer.deq[t][j], 0));
          }
          Value address = comb::ConcatOp::create(
              b, select(threads, sink.thread), select(slots, sink.thread));
          parts.push_back(seq::FirMemReadOp::create(
              b, tokenMemory(producer, piece.low, piece.width, consumer.cls),
              address,
              hostClock, bit(true)));
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
  return success();
}

LogicalResult Composer::run() {
  inlineNetlists();
  if (failed(collect()) || failed(build()))
    return failure();
  netlist.erase();
  for (auto *op : inlined)
    op->erase();
  return success();
}

void ComposePass::runOnOperation() {
  auto top = getOperation();
  auto netlist = dyn_cast_or_null<hw::HWModuleOp>(
      SymbolTable::lookupSymbolIn(top, moduleName));
  if (!netlist) {
    top.emitError("no netlist '") << moduleName << "'";
    return signalPassFailure();
  }
  if (failed(Composer(top, netlist, simName, readySchedule, relayModels).run()))
    signalPassFailure();
}
