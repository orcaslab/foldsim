//===- ExtractModels.cpp - Cut child models out of a parent ---------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simulator simulates each class of instances as a model of its own,
// connected to the others by channels. This pass cuts the instances of those
// classes out of a flat parent, leaving the parent's own logic as a model whose
// ports are what it exchanges with them.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "foldsim/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <functional>

namespace foldsim {
#define GEN_PASS_DEF_EXTRACTMODELS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

struct ExtractModelsPass
    : public foldsim::impl::ExtractModelsBase<ExtractModelsPass> {
  using ExtractModelsBase::ExtractModelsBase;
  void runOnOperation() override;
};

/// Where one bit of a value comes from: logic, a constant, or a bit of a port
/// it is wired from.
struct BitSource {
  enum Kind { Logic, Constant, Port } kind = Logic;
  Value port;
  unsigned bit = 0;
  bool value = false;
};

using BitSources = SmallVector<BitSource>;

/// Traces the bits of values through the ops that only move bits around and,
/// optionally, through register bits that only ever take one constant value.
class BitTracer {
public:
  BitTracer(const DenseSet<Value> &ports, bool throughRegisters)
      : ports(ports), throughRegisters(throughRegisters) {}

  /// Whether `op` only moves bits around, and so is traced through.
  static bool isWiring(Operation *op);
  const BitSources &get(Value value);

private:
  BitSources compute(Value value);
  BitSources computeRegister(seq::FirRegOp reg);
  /// The values `compute` traces `value` through, in the order it reads them.
  void readsOf(Value value, SmallVectorImpl<Value> &reads);

  const DenseSet<Value> &ports;
  bool throughRegisters;
  DenseMap<Value, std::unique_ptr<BitSources>> cache;
};

} // namespace

static unsigned getElementWidth(Type arrayType) {
  return hw::getBitWidth(
      hw::type_cast<hw::ArrayType>(arrayType).getElementType());
}

bool BitTracer::isWiring(Operation *op) {
  if (isa<hw::ConstantOp, hw::AggregateConstantOp, comb::ConcatOp,
          comb::ExtractOp, hw::BitcastOp, hw::ArrayCreateOp, hw::ArrayConcatOp,
          hw::StructCreateOp, hw::StructExtractOp>(op))
    return true;
  if (auto get = dyn_cast<hw::ArrayGetOp>(op))
    return matchPattern(get.getIndex(), m_Constant());
  if (auto slice = dyn_cast<hw::ArraySliceOp>(op))
    return matchPattern(slice.getLowIndex(), m_Constant());
  return false;
}

/// The bits of an aggregate constant: the first element of an array or field
/// of a struct is in the high bits.
static std::optional<APInt> getAggregateValue(Attribute attr, Type type) {
  auto width = hw::getBitWidth(type);
  if (width < 0)
    return std::nullopt;
  if (auto integer = dyn_cast<IntegerAttr>(attr))
    return integer.getValue().zextOrTrunc(width);
  auto elements = dyn_cast<ArrayAttr>(attr);
  if (!elements)
    return std::nullopt;
  SmallVector<Type> types;
  if (auto array = hw::type_dyn_cast<hw::ArrayType>(type))
    types.assign(array.getNumElements(), array.getElementType());
  else if (auto structType = hw::type_dyn_cast<hw::StructType>(type))
    for (auto &field : structType.getElements())
      types.push_back(field.type);
  if (types.size() != elements.size())
    return std::nullopt;
  std::optional<APInt> result;
  for (auto [element, elementType] : llvm::zip(elements, types)) {
    auto value = getAggregateValue(element, elementType);
    if (!value)
      return std::nullopt;
    result = result ? result->concat(*value) : *value;
  }
  return result;
}

/// Without recursion: wiring and registers chain far more values than a
/// thread's stack holds frames. Each value is computed once the values it
/// reads are, depth first in the order `compute` reads them, so that a value
/// reached again while it is traced, through a register, is logic.
const BitSources &BitTracer::get(Value root) {
  if (auto it = cache.find(root); it != cache.end())
    return *it->second;
  struct Frame {
    Value value;
    SmallVector<Value, 4> reads;
    unsigned next = 0;
  };
  SmallVector<Frame> frames;
  auto enter = [&](Value value) {
    auto width = std::max<int64_t>(hw::getBitWidth(value.getType()), 0);
    cache[value] = std::make_unique<BitSources>(width);
    frames.push_back({value, {}, 0});
    readsOf(value, frames.back().reads);
  };
  enter(root);
  while (!frames.empty()) {
    auto &frame = frames.back();
    if (frame.next < frame.reads.size()) {
      Value read = frame.reads[frame.next++];
      if (!cache.count(read))
        enter(read);
      continue;
    }
    Value value = frame.value;
    frames.pop_back();
    auto sources = compute(value);
    cache[value] = std::make_unique<BitSources>(std::move(sources));
  }
  return *cache[root];
}

void BitTracer::readsOf(Value value, SmallVectorImpl<Value> &reads) {
  if (hw::getBitWidth(value.getType()) <= 0 || ports.contains(value))
    return;
  auto *op = value.getDefiningOp();
  if (auto reg = dyn_cast_or_null<seq::FirRegOp>(op); reg && throughRegisters) {
    reads.push_back(reg.getNext());
    if (reg.hasReset())
      reads.push_back(reg.getResetValue());
    return;
  }
  if (!op || !isWiring(op) || isa<hw::ConstantOp, hw::AggregateConstantOp>(op))
    return;
  // Concatenations and creates read their last operand first.
  if (isa<comb::ConcatOp, hw::ArrayCreateOp, hw::ArrayConcatOp,
          hw::StructCreateOp>(op)) {
    reads.append(op->operand_begin(), op->operand_end());
    std::reverse(reads.begin(), reads.end());
    return;
  }
  // The others read the value they take bits of (an index is a constant).
  reads.push_back(op->getOperand(0));
}

BitSources BitTracer::compute(Value value) {
  auto width = hw::getBitWidth(value.getType());
  BitSources sources(std::max<int64_t>(width, 0));
  if (width <= 0)
    return sources;
  if (ports.contains(value)) {
    for (unsigned bit = 0; bit < sources.size(); ++bit)
      sources[bit] = {BitSource::Port, value, bit};
    return sources;
  }
  auto *op = value.getDefiningOp();
  if (auto reg = dyn_cast_or_null<seq::FirRegOp>(op); reg && throughRegisters)
    return computeRegister(reg);
  if (!op || !isWiring(op))
    return sources;

  // The bits of `operands`, the last one in the low bits.
  auto appendReversed = [&](ValueRange operands) {
    BitSources result;
    for (auto operand : llvm::reverse(operands)) {
      auto bits = get(operand);
      result.append(bits.begin(), bits.end());
    }
    return result;
  };
  auto slice = [&](Value input, unsigned low) {
    auto bits = get(input);
    return BitSources(bits.begin() + low, bits.begin() + low + width);
  };

  if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
    for (auto [bit, source] : llvm::enumerate(sources)) {
      source.kind = BitSource::Constant;
      source.value = constant.getValue()[bit];
    }
    return sources;
  }
  if (auto constant = dyn_cast<hw::AggregateConstantOp>(op)) {
    auto value = getAggregateValue(constant.getFields(), constant.getType());
    if (!value)
      return sources;
    for (auto [bit, source] : llvm::enumerate(sources)) {
      source.kind = BitSource::Constant;
      source.value = (*value)[bit];
    }
    return sources;
  }
  if (auto concat = dyn_cast<comb::ConcatOp>(op))
    return appendReversed(concat.getInputs());
  if (auto extract = dyn_cast<comb::ExtractOp>(op))
    return slice(extract.getInput(), extract.getLowBit());
  if (auto bitcast = dyn_cast<hw::BitcastOp>(op))
    return get(bitcast.getInput());
  // Array element 0 is in the low bits, as the first operand of a create or a
  // concatenation is in the high ones.
  if (auto create = dyn_cast<hw::ArrayCreateOp>(op))
    return appendReversed(create.getInputs());
  if (auto concat = dyn_cast<hw::ArrayConcatOp>(op))
    return appendReversed(concat.getInputs());
  // A struct's first field is in the high bits.
  if (auto create = dyn_cast<hw::StructCreateOp>(op))
    return appendReversed(create.getInput());
  if (auto extract = dyn_cast<hw::StructExtractOp>(op)) {
    auto elements = hw::type_cast<hw::StructType>(extract.getInput().getType())
                        .getElements();
    unsigned low = 0;
    for (auto &element : elements.drop_front(extract.getFieldIndex() + 1))
      low += hw::getBitWidth(element.type);
    return slice(extract.getInput(), low);
  }
  APInt index;
  if (auto get = dyn_cast<hw::ArrayGetOp>(op);
      get && matchPattern(get.getIndex(), m_ConstantInt(&index)))
    return slice(get.getInput(), index.getZExtValue() *
                                     getElementWidth(get.getInput().getType()));
  if (auto arraySlice = dyn_cast<hw::ArraySliceOp>(op);
      arraySlice &&
      matchPattern(arraySlice.getLowIndex(), m_ConstantInt(&index)))
    return slice(arraySlice.getInput(),
                 index.getZExtValue() *
                     getElementWidth(arraySlice.getInput().getType()));
  return sources;
}

/// A register bit is that constant if its next value and its reset value, if
/// any, are always the same constant: without a preset value, its value is
/// undefined only before the first clock edge. A preset value must be that
/// constant too.
BitSources BitTracer::computeRegister(seq::FirRegOp reg) {
  auto width = hw::getBitWidth(reg.getType());
  BitSources sources(width);
  auto next = get(reg.getNext());
  BitSources reset;
  if (reg.hasReset())
    reset = get(reg.getResetValue());
  std::optional<APInt> preset = reg.getPreset();
  if (preset)
    preset = preset->zextOrTrunc(width);
  for (auto [bit, source] : llvm::enumerate(sources)) {
    if (next[bit].kind != BitSource::Constant)
      continue;
    if (!reset.empty() && (reset[bit].kind != BitSource::Constant ||
                           reset[bit].value != next[bit].value))
      continue;
    if (preset && (*preset)[bit] != next[bit].value)
      continue;
    source = next[bit];
  }
  return sources;
}

/// The name of the port carrying bits `lo` to `hi` of a port `width` wide.
static std::string sliceName(const Twine &base, unsigned hi, unsigned lo,
                             unsigned width) {
  if (lo == 0 && hi + 1 == width)
    return base.str();
  if (hi == lo)
    return (base + "_" + Twine(hi) + "_").str();
  return (base + "_" + Twine(hi) + "_" + Twine(lo) + "_").str();
}

/// A name usable as a Verilog identifier.
static std::string sanitize(StringRef name) {
  std::string result = name.str();
  for (auto &c : result)
    if (!llvm::isAlnum(c) && c != '_')
      c = '_';
  return result;
}

/// `sources`, most significant bit first, as a Verilog concatenation of the
/// named ports' bits and constants.
static std::string describe(const BitSources &sources,
                            function_ref<std::string(Value)> nameOf) {
  SmallVector<std::string> parts;
  for (unsigned hi = sources.size(); hi > 0;) {
    auto &top = sources[hi - 1];
    unsigned lo = hi - 1;
    // A run of bits of one port, in order, or of constants.
    while (lo > 0) {
      auto &next = sources[lo - 1];
      bool same = top.kind == BitSource::Constant
                      ? next.kind == BitSource::Constant
                      : next.kind == BitSource::Port && next.port == top.port &&
                            next.bit + (hi - lo) == top.bit;
      if (!same)
        break;
      --lo;
    }
    if (top.kind == BitSource::Constant) {
      std::string bits;
      for (unsigned bit = hi; bit > lo; --bit)
        bits += sources[bit - 1].value ? '1' : '0';
      parts.push_back(std::to_string(hi - lo) + "'b" + bits);
    } else {
      auto width = hw::getBitWidth(top.port.getType());
      unsigned low = top.bit - (hi - 1 - lo);
      auto name = nameOf(top.port);
      if (low == 0 && top.bit + 1 == unsigned(width))
        parts.push_back(name);
      else if (low == top.bit)
        parts.push_back(name + "[" + std::to_string(low) + "]");
      else
        parts.push_back(name + "[" + std::to_string(top.bit) + ":" +
                        std::to_string(low) + "]");
    }
    hi = lo;
  }
  if (parts.size() == 1)
    return parts[0];
  std::string result = "{";
  llvm::interleave(
      parts, [&](auto &part) { result += part; }, [&] { result += ", "; });
  return result + "}";
}

/// Whether the parent's own logic reads `value`, rather than only the inputs
/// of the models being cut out.
///
/// Without recursion (logic chains far more values than a thread's stack
/// holds frames), in the order of a depth-first search over the users: a value
/// reached again while it is searched counts as not read.
static bool isReadByParent(Value root, const DenseSet<Operation *> &models,
                           DenseMap<Value, bool> &read) {
  if (auto it = read.find(root); it != read.end())
    return it->second;
  struct Frame {
    Value value;
    SmallVector<Operation *, 4> users;
    unsigned nextUser = 0, nextResult = 0;
  };
  SmallVector<Frame> frames;
  auto enter = [&](Value value) {
    read[value] = false;
    frames.push_back({value, {}, 0, 0});
    frames.back().users.append(value.getUsers().begin(),
                               value.getUsers().end());
  };
  enter(root);
  // The result of the frame just left, for its caller.
  bool returned = false;
  bool pending = false;
  while (!frames.empty()) {
    auto &frame = frames.back();
    bool result = false;
    if (pending) {
      pending = false;
      result = returned;
      if (!result)
        ++frame.nextResult;
    }
    while (!result && frame.nextUser < frame.users.size()) {
      Operation *user = frame.users[frame.nextUser];
      if (models.contains(user)) {
        ++frame.nextUser;
        continue;
      }
      if (!BitTracer::isWiring(user) &&
          !isa<comb::CombDialect>(user->getDialect())) {
        result = true;
        break;
      }
      if (frame.nextResult == user->getNumResults()) {
        ++frame.nextUser;
        frame.nextResult = 0;
        continue;
      }
      Value next = user->getResult(frame.nextResult);
      if (auto it = read.find(next); it != read.end()) {
        if (it->second)
          result = true;
        else
          ++frame.nextResult;
        continue;
      }
      break;
    }
    if (!result && frame.nextUser < frame.users.size()) {
      // A result of the user not searched yet.
      enter(frame.users[frame.nextUser]->getResult(frame.nextResult));
      continue;
    }
    read[frame.value] = result;
    frames.pop_back();
    returned = result;
    pending = true;
  }
  return read[root];
}

/// For each output of a model, the bits that are always the same constant.
static SmallVector<BitSources> getConstantOutputs(hw::HWModuleOp model) {
  auto *body = model.getBodyBlock();
  DenseSet<Value> ports(body->args_begin(), body->args_end());
  BitTracer tracer(ports, /*throughRegisters=*/true);
  SmallVector<BitSources> outputs;
  for (auto output : body->getTerminator()->getOperands())
    outputs.push_back(tracer.get(output));
  return outputs;
}

namespace {
/// Moves the logic in a parent that drives models' inputs into the models.
class LogicAbsorber {
public:
  LogicAbsorber(ModuleOp top, hw::HWModuleOp parent,
                const llvm::StringSet<> &modelNames, unsigned limit)
      : top(top), parent(parent), modelNames(modelNames), limit(limit) {}
  void run();

private:
  bool dependsOnPorts(Value value);
  bool isConeOp(Operation *op);
  bool collect(Value root, SmallVectorImpl<Operation *> &ops,
               SmallVectorImpl<Value> &leaves);
  bool match(Value a, Value b, DenseMap<Value, Value> &leafMap,
             DenseMap<Value, Value> &matched);

  ModuleOp top;
  hw::HWModuleOp parent;
  const llvm::StringSet<> &modelNames;
  unsigned limit;
  DenseMap<Value, bool> portDeps;
  /// Inputs added to a model, by the value each instance gives them.
  DenseMap<Operation *, SmallVector<std::pair<SmallVector<Value>, Value>>>
      absorbedInputs;
};
} // namespace

/// Whether `value` is computed, through logic, from an input of the parent or
/// an output of a model: what is not is the parent's state and constants.
/// Without recursion (logic chains far more values than a thread's stack holds
/// frames), in the order of a depth-first search over the operands: a value
/// reached again while it is searched counts as not depending on them.
bool LogicAbsorber::dependsOnPorts(Value root) {
  if (auto it = portDeps.find(root); it != portDeps.end())
    return it->second;
  // The result of a value that needs no search of its operands.
  auto leaf = [&](Value value) -> std::optional<bool> {
    auto *op = value.getDefiningOp();
    if (!op || isa<hw::InstanceOp>(op))
      return true;
    if (!isConeOp(op))
      return false;
    return std::nullopt;
  };
  if (auto result = leaf(root))
    return portDeps[root] = *result;
  struct Frame {
    Value value;
    Operation *op;
    unsigned next = 0;
  };
  SmallVector<Frame> frames;
  auto enter = [&](Value value) {
    portDeps[value] = false;
    frames.push_back({value, value.getDefiningOp(), 0});
  };
  enter(root);
  std::optional<bool> returned;
  while (!frames.empty()) {
    auto &frame = frames.back();
    bool result = returned.value_or(false);
    returned.reset();
    while (!result && frame.next < frame.op->getNumOperands()) {
      Value operand = frame.op->getOperand(frame.next);
      if (auto it = portDeps.find(operand); it != portDeps.end()) {
        result = it->second;
        ++frame.next;
        continue;
      }
      if (auto known = leaf(operand)) {
        result = portDeps[operand] = *known;
        ++frame.next;
        continue;
      }
      break;
    }
    if (!result && frame.next < frame.op->getNumOperands()) {
      Value operand = frame.op->getOperand(frame.next++);
      enter(operand);
      continue;
    }
    Value value = frame.value;
    portDeps[value] = result;
    frames.pop_back();
    returned = result;
  }
  return portDeps[root];
}

/// Whether `op` is logic that can be computed anywhere: no state, no regions,
/// no side effects.
bool LogicAbsorber::isConeOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         !isa<hw::InstanceOp, seq::FirRegOp, seq::FirMemReadOp,
              seq::FirMemReadWriteOp>(op) &&
         op->getNumResults() >= 1;
}

/// The logic computing `root` from the ports, in operand-first order, and the
/// values it reads: ports, the parent's state, or other logic's results that do
/// not depend on ports. Fails if there is none or it is over the limit.
bool LogicAbsorber::collect(Value root, SmallVectorImpl<Operation *> &ops,
                            SmallVectorImpl<Value> &leaves) {
  DenseSet<Operation *> seen;
  DenseSet<Value> leafSet;
  // Operand first, without recursion: a cone may chain far more ops than a
  // thread's stack holds frames before it is found over the limit.
  auto visit = [&](Value root) -> bool {
    SmallVector<std::pair<Operation *, unsigned>> frames;
    // Whether `value` is an op of the cone not seen yet: a leaf or an op seen
    // is done with.
    auto enter = [&](Value value) {
      auto *op = value.getDefiningOp();
      bool inside = op && isConeOp(op) &&
                    (dependsOnPorts(value) || op->getNumOperands() == 0);
      if (!inside) {
        if (leafSet.insert(value).second)
          leaves.push_back(value);
        return;
      }
      if (seen.insert(op).second)
        frames.push_back({op, 0});
    };
    enter(root);
    while (!frames.empty()) {
      auto &[op, next] = frames.back();
      if (next < op->getNumOperands()) {
        enter(op->getOperand(next++));
        continue;
      }
      ops.push_back(op);
      frames.pop_back();
      if (ops.size() > limit)
        return false;
    }
    return true;
  };
  auto *rootOp = root.getDefiningOp();
  if (!rootOp || !isConeOp(rootOp) || !dependsOnPorts(root))
    return false;
  if (!visit(root))
    return false;
  // Only wiring is connected directly already.
  return llvm::any_of(ops, [](Operation *op) {
    return !BitTracer::isWiring(op) && !isa<hw::ConstantOp>(op);
  });
}

/// Whether two ops have the same attributes, names aside.
static bool sameAttributes(Operation *a, Operation *b) {
  auto filtered = [](Operation *op) {
    NamedAttrList attrs(op->getAttrDictionary());
    attrs.erase("sv.namehint");
    return attrs.getDictionary(op->getContext());
  };
  return filtered(a) == filtered(b);
}

/// Whether `b` is the same logic as `a`, its leaves in `leafMap`.
bool LogicAbsorber::match(Value a, Value b, DenseMap<Value, Value> &leafMap,
                          DenseMap<Value, Value> &matched) {
  if (auto it = leafMap.find(a); it != leafMap.end()) {
    if (!it->second)
      return (it->second = b), true;
    return it->second == b;
  }
  if (auto it = matched.find(a); it != matched.end())
    return it->second == b;
  matched[a] = b;
  auto *opA = a.getDefiningOp(), *opB = b.getDefiningOp();
  if (!opA || !opB || opA->getName() != opB->getName() ||
      !sameAttributes(opA, opB) ||
      opA->getResultTypes() != opB->getResultTypes() ||
      opA->getNumOperands() != opB->getNumOperands() ||
      cast<OpResult>(a).getResultNumber() !=
          cast<OpResult>(b).getResultNumber())
    return false;
  for (auto [x, y] : llvm::zip(opA->getOperands(), opB->getOperands()))
    if (!match(x, y, leafMap, matched))
      return false;
  return true;
}

void LogicAbsorber::run() {
  llvm::MapVector<StringAttr, SmallVector<hw::InstanceOp>> byModel;
  for (auto instance : parent.getBodyBlock()->getOps<hw::InstanceOp>())
    if (modelNames.contains(instance.getModuleName()))
      byModel[instance.getModuleNameAttr().getAttr()].push_back(instance);

  for (auto &[name, instances] : byModel) {
    // A netlist holds only models and wiring.
    auto model = dyn_cast_or_null<hw::HWModuleOp>(
        SymbolTable::lookupSymbolIn(top, name));
    if (!model || model->hasAttr("foldsim.netlist"))
      continue;
    // The model gains ports, so every instance of it must be one of these.
    unsigned numInstances = 0;
    StringAttr modelName = name;
    top.walk([&](hw::InstanceOp instance) {
      numInstances += instance.getModuleNameAttr().getAttr() == modelName;
    });
    if (numInstances != instances.size())
      continue;
    for (unsigned index = 0; index < model.getNumInputPorts(); ++index) {
      auto arg = model.getBodyBlock()->getArgument(index);
      if (arg.use_empty() || isa<seq::ClockType>(arg.getType()))
        continue;
      // The logic of the first instance, and every other instance's leaves
      // for it.
      SmallVector<Operation *> ops;
      SmallVector<Value> leaves;
      portDeps.clear();
      if (!collect(instances[0].getOperand(index), ops, leaves))
        continue;
      SmallVector<SmallVector<Value>> leavesOf;
      bool same = true;
      for (auto instance : instances) {
        DenseMap<Value, Value> leafMap;
        for (auto leaf : leaves)
          leafMap[leaf] = Value();
        DenseMap<Value, Value> matched;
        if (!match(instances[0].getOperand(index), instance.getOperand(index),
                   leafMap, matched)) {
          same = false;
          break;
        }
        auto &mapped = leavesOf.emplace_back();
        for (auto leaf : leaves)
          mapped.push_back(leafMap[leaf]);
      }
      if (!same)
        continue;

      // A leaf that is every instance's own output is that output inside the
      // model; any other is a new input.
      auto *output = model.getBodyBlock()->getTerminator();
      SmallVector<std::optional<unsigned>> ownOutput(leaves.size());
      for (auto [k, leaf] : llvm::enumerate(leaves)) {
        auto result = dyn_cast<OpResult>(leaf);
        if (!result || result.getOwner() != instances[0].getOperation())
          continue;
        bool own = true;
        for (auto [i, instance] : llvm::enumerate(instances)) {
          auto other = dyn_cast<OpResult>(leavesOf[i][k]);
          own &= other && other.getOwner() == instance.getOperation() &&
                 other.getResultNumber() == result.getResultNumber();
        }
        if (own)
          ownOutput[k] = result.getResultNumber();
      }
      IRMapping mapping;
      auto baseName = model.getInputName(index);
      SmallVector<bool> newPort(leaves.size(), false);
      for (auto [k, leaf] : llvm::enumerate(leaves)) {
        if (ownOutput[k]) {
          mapping.map(leaf, output->getOperand(*ownOutput[k]));
          continue;
        }
        // A value every instance already gives an earlier moved input.
        SmallVector<Value> values;
        for (auto &mapped : leavesOf)
          values.push_back(mapped[k]);
        auto &added = absorbedInputs[model.getOperation()];
        auto it = llvm::find_if(
            added, [&](auto &entry) { return entry.first == values; });
        if (it != added.end()) {
          mapping.map(leaf, it->second);
          continue;
        }
        auto name = (baseName + "_" + Twine(k)).str();
        auto port = model.appendInput(name, leaf.getType()).second;
        added.push_back({values, port});
        mapping.map(leaf, port);
        newPort[k] = true;
      }
      OpBuilder b = OpBuilder::atBlockBegin(model.getBodyBlock());
      for (auto *op : ops)
        b.clone(*op, mapping);
      arg.replaceAllUsesWith(mapping.lookup(instances[0].getOperand(index)));

      // Each instance takes the leaves of its own logic. A leaf may be
      // another instance's output, so all are created before any goes.
      SmallVector<hw::InstanceOp> replacements;
      for (auto [i, instance] : llvm::enumerate(instances)) {
        SmallVector<Value> operands(instance.getInputs());
        for (auto [k, leaf] : llvm::enumerate(leavesOf[i]))
          if (newPort[k])
            operands.push_back(leaf);
        OpBuilder ib(instance);
        replacements.push_back(hw::InstanceOp::create(
            ib, instance.getLoc(), model, instance.getInstanceNameAttr(),
            operands, instance.getParametersAttr(),
            instance.getInnerSymAttr()));
      }
      for (auto [instance, replacement] : llvm::zip(instances, replacements))
        instance.replaceAllUsesWith(replacement.getResults());
      for (auto instance : instances)
        instance.erase();
      instances = replacements;
    }
  }
}

void ExtractModelsPass::runOnOperation() {
  auto top = getOperation();
  auto parent = dyn_cast_or_null<hw::HWModuleOp>(
      SymbolTable::lookupSymbolIn(top, moduleName));
  if (!parent) {
    top.emitError("no module '") << moduleName << "' to extract models from";
    return signalPassFailure();
  }
  llvm::StringSet<> modelNames;
  for (auto &name : models)
    modelNames.insert(name);
  llvm::StringMap<SmallVector<std::pair<unsigned, unsigned>>> givenInputs;
  for (auto &entry : inputBits) {
    SmallVector<StringRef, 3> fields;
    StringRef(entry).split(fields, ':');
    unsigned hi, lo;
    if (fields.size() != 3 || fields[1].getAsInteger(10, hi) ||
        fields[2].getAsInteger(10, lo) || hi < lo) {
      parent.emitError("input bits '") << entry << "' are not <port>:<hi>:<lo>";
      return signalPassFailure();
    }
    givenInputs[fields[0]].push_back({hi, lo});
  }

  if (absorbLogic)
    LogicAbsorber(top, parent, modelNames, absorbLogic).run();

  auto *body = parent.getBodyBlock();
  auto originalPorts = parent.getPortList();
  auto numOutputs = parent.getNumOutputPorts();
  SmallVector<hw::InstanceOp> instances;
  DenseSet<Value> ports(body->args_begin(), body->args_end());
  DenseMap<StringAttr, SmallVector<BitSources>> constantOutputs;
  for (auto instance : body->getOps<hw::InstanceOp>()) {
    if (!modelNames.contains(instance.getModuleName()))
      continue;
    for (auto type : instance->getOperandTypes())
      if (isa<hw::InOutType>(type)) {
        instance.emitError("inout port on a model");
        return signalPassFailure();
      }
    instances.push_back(instance);
    ports.insert(instance.getResults().begin(), instance.getResults().end());
    auto name = instance.getModuleNameAttr().getAttr();
    if (!constantOutputs.count(name))
      if (auto model = dyn_cast_or_null<hw::HWModuleOp>(
              SymbolTable::lookupSymbolIn(top, name)))
        constantOutputs[name] = getConstantOutputs(model);
  }

  // Each model input the model reads becomes an output of the parent, unless it
  // is wired from inputs of the parent and constants, or is an output of a
  // model: it is connected to those directly. One that only constants drive is
  // a channel of tokens, with no bits.
  BitTracer tracer(ports, /*throughRegisters=*/false);
  auto builder = ImplicitLocOpBuilder::atBlockTerminator(parent.getLoc(), body);
  SmallVector<std::pair<StringAttr, Value>> newOutputs;
  DenseMap<Value, std::string> portNames;
  for (auto [index, arg] : llvm::enumerate(body->getArguments()))
    portNames[arg] = parent.getInputName(index).str();
  for (auto instance : instances)
    for (auto [result, name] :
         llvm::zip(instance.getResults(), instance.getResultNames()))
      portNames[result] = sanitize(instance.getInstanceName()) + "_" +
                          cast<StringAttr>(name).getValue().str();
  SmallVector<Attribute> directInputs;
  // For the netlist: how each model input is connected, by instance and input.
  // A channel is a new output of the parent; a direct input, its bits.
  struct InputPlan {
    std::optional<unsigned> output;
    std::optional<BitSources> bits;
  };
  DenseMap<std::pair<unsigned, unsigned>, InputPlan> inputPlans;
  auto hasConstantBits = [&](Value output) {
    auto result = cast<OpResult>(output);
    auto instance = cast<hw::InstanceOp>(result.getOwner());
    auto constants =
        constantOutputs.lookup(instance.getModuleNameAttr().getAttr());
    return !constants.empty() &&
           llvm::any_of(constants[result.getResultNumber()], [](auto &bit) {
             return bit.kind == BitSource::Constant;
           });
  };
  for (auto [instanceIndex, instance] : llvm::enumerate(instances)) {
    auto base = sanitize(instance.getInstanceName());
    auto model = dyn_cast_or_null<hw::HWModuleOp>(SymbolTable::lookupSymbolIn(
        top, instance.getModuleNameAttr().getAttr()));
    for (auto [index, operand, name] :
         llvm::enumerate(instance.getInputs(), instance.getArgNames())) {
      auto width = hw::getBitWidth(operand.getType());
      if (isa<seq::ClockType>(operand.getType()) || width <= 0)
        continue;
      // An input the model does not read is not a channel of the model.
      if (model && model.getBodyBlock()->getArgument(index).use_empty())
        continue;
      auto sources = tracer.get(operand);
      auto isConstant = [](auto &source) {
        return source.kind == BitSource::Constant;
      };
      // In a netlist, a bit a model's output always holds is that constant: the
      // model does not export it.
      for (auto &source : sources) {
        if (!netlist || source.kind != BitSource::Port ||
            isa<BlockArgument>(source.port))
          continue;
        auto result = cast<OpResult>(source.port);
        auto constants =
            constantOutputs.lookup(cast<hw::InstanceOp>(result.getOwner())
                                       .getModuleNameAttr()
                                       .getAttr());
        if (constants.empty())
          continue;
        auto &bit = constants[result.getResultNumber()][source.bit];
        if (bit.kind == BitSource::Constant)
          source = bit;
      }
      // Wired from inputs of the parent and constants, or exactly one output
      // of a model with no constant bits, which is a channel as a whole.
      bool direct = llvm::all_of(sources, [&](auto &source) {
        return isConstant(source) || (source.kind == BitSource::Port &&
                                      isa<BlockArgument>(source.port));
      });
      if (!direct && !sources.empty()) {
        auto output = sources[0].port;
        direct = output && !isa<BlockArgument>(output) &&
                 hw::getBitWidth(output.getType()) == width &&
                 !hasConstantBits(output) &&
                 llvm::all_of(llvm::enumerate(sources), [&](auto bit) {
                   return bit.value().kind == BitSource::Port &&
                          bit.value().port == output &&
                          bit.value().bit == bit.index();
                 });
      }
      // In a netlist, any mix of model outputs' bits, inputs of the parent and
      // constants is connected directly: the consumer reads each producer, and
      // the parent's model does not pass the bits on a cycle later.
      if (!direct && netlist)
        direct = !sources.empty() && llvm::none_of(sources, [](auto &source) {
          return source.kind == BitSource::Logic;
        });
      auto portName = base + "_" + cast<StringAttr>(name).getValue().str();
      if (direct && netlist) {
        inputPlans[{instanceIndex, index}].bits = sources;
        continue;
      }
      if (direct && !llvm::all_of(sources, isConstant)) {
        directInputs.push_back(builder.getStringAttr(
            portName + " = " +
            describe(sources, [&](Value port) { return portNames[port]; })));
        continue;
      }
      inputPlans[{instanceIndex, index}].output = newOutputs.size();
      Value value = operand;
      if (direct)
        value = hw::ConstantOp::create(builder, builder.getIntegerType(0), 0);
      else if (!value.getType().isInteger(width))
        value = hw::BitcastOp::create(builder, builder.getIntegerType(width),
                                      value);
      newOutputs.push_back({builder.getStringAttr(portName), value});
    }
  }
  if (!directInputs.empty())
    parent->setAttr("foldsim.direct_inputs", builder.getArrayAttr(directInputs));
  if (!newOutputs.empty())
    parent.insertOutputs(parent.getNumOutputPorts(), newOutputs);

  // Each model output becomes inputs of its bits, but for those that are
  // always the same constant. One the parent's logic does not read, only the
  // models it is wired to, is a channel of tokens, with no bits.
  builder.setInsertionPoint(body, body->begin());
  DenseSet<Operation *> modelOps;
  for (auto instance : instances)
    modelOps.insert(instance);
  DenseMap<Value, bool> readByParent;
  // For the netlist: the model output bits each new input of the parent takes.
  struct InputSource {
    Value result;
    unsigned low, width;
  };
  SmallVector<InputSource> newInputs;
  auto appendInput = [&](Value result, StringRef portName, unsigned low,
                         unsigned width) {
    newInputs.push_back({result, low, width});
    return parent.appendInput(portName, builder.getIntegerType(width)).second;
  };
  for (auto instance : instances) {
    auto base = sanitize(instance.getInstanceName());
    auto constants =
        constantOutputs.lookup(instance.getModuleNameAttr().getAttr());
    for (auto [index, result, name] :
         llvm::enumerate(instance.getResults(), instance.getResultNames())) {
      auto width = hw::getBitWidth(result.getType());
      if (isa<seq::ClockType>(result.getType()) || width <= 0) {
        if (!result.use_empty()) {
          instance.emitError("model output '")
              << name << "' has no bits but is used";
          return signalPassFailure();
        }
        continue;
      }
      // Runs of bits from the low one: each a constant or an input.
      SmallVector<Value> parts;
      unsigned lo = 0;
      const BitSources *bits = constants.empty() ? nullptr : &constants[index];
      auto isConstant = [&](unsigned bit) {
        return bits && (*bits)[bit].kind == BitSource::Constant;
      };
      auto portBase = base + "_" + cast<StringAttr>(name).getValue().str();
      auto constantPart = [&](unsigned lo, unsigned hi) {
        APInt value(hi - lo + 1, 0);
        for (unsigned bit = lo; bit <= hi; ++bit)
          value.setBitVal(bit - lo, (*bits)[bit].value);
        parts.push_back(hw::ConstantOp::create(builder, value));
      };
      auto givenIt = givenInputs.find(portBase);
      bool allConstant = bits && llvm::all_of(*bits, [](auto &source) {
                           return source.kind == BitSource::Constant;
                         });
      if (givenIt == givenInputs.end() && !allConstant &&
          !isReadByParent(result, modelOps, readByParent)) {
        if (!netlist)
          parent.appendInput(portBase, builder.getIntegerType(0));
        if (!result.use_empty()) {
          Value zero = hw::ConstantOp::create(builder, APInt::getZero(width));
          if (zero.getType() != result.getType())
            zero = hw::BitcastOp::create(builder, result.getType(), zero);
          result.replaceAllUsesWith(zero);
        }
        continue;
      }
      if (auto it = givenIt; it != givenInputs.end()) {
        // The inputs are given; the bits they leave out must be constants.
        auto ranges = it->second;
        llvm::sort(ranges, [](auto a, auto b) { return a.second < b.second; });
        for (auto [hi, lo2] : ranges) {
          if (lo2 < lo || hi >= unsigned(width)) {
            instance.emitError("input bits of '")
                << portBase << "' overlap or exceed it";
            return signalPassFailure();
          }
          for (unsigned bit = lo; bit < lo2; ++bit)
            if (!isConstant(bit)) {
              instance.emitError("bit ")
                  << bit << " of '" << portBase
                  << "' is not a constant and is not in an input";
              return signalPassFailure();
            }
          if (lo2 > lo)
            constantPart(lo, lo2 - 1);
          parts.push_back(appendInput(
              result, sliceName(portBase, hi, lo2, width), lo2, hi - lo2 + 1));
          lo = hi + 1;
        }
        for (unsigned bit = lo; bit < unsigned(width); ++bit)
          if (!isConstant(bit)) {
            instance.emitError("bit ")
                << bit << " of '" << portBase
                << "' is not a constant and is not in an input";
            return signalPassFailure();
          }
        if (lo < unsigned(width))
          constantPart(lo, width - 1);
        lo = width;
      }
      while (lo < unsigned(width)) {
        unsigned hi = lo;
        while (hi + 1 < unsigned(width) && isConstant(hi + 1) == isConstant(lo))
          ++hi;
        if (isConstant(lo)) {
          constantPart(lo, hi);
        } else {
          parts.push_back(appendInput(
              result, sliceName(portBase, hi, lo, width), lo, hi - lo + 1));
        }
        lo = hi + 1;
      }
      std::reverse(parts.begin(), parts.end());
      Value value = parts.size() == 1
                        ? parts[0]
                        : comb::ConcatOp::create(builder, parts).getResult();
      if (value.getType() != result.getType())
        value = hw::BitcastOp::create(builder, result.getType(), value);
      result.replaceAllUsesWith(value);
    }
  }
  if (netlist) {
    // The parent's logic is a model of its own, and the parent its netlist.
    auto name = parent.getSymNameAttr();
    parent.setSymName((name.getValue() + "_model").str());
    OpBuilder outer(parent);
    outer.setInsertionPointAfter(parent);
    auto net = hw::HWModuleOp::create(outer, parent.getLoc(), name,
                                      hw::ModulePortInfo(originalPorts));
    net.setVisibility(parent.getVisibility());
    parent.setPrivate();
    net->setAttr("foldsim.netlist", outer.getUnitAttr());
    auto *netBody = net.getBodyBlock();
    netBody->getTerminator()->erase();
    auto b = ImplicitLocOpBuilder::atBlockEnd(parent.getLoc(), netBody);
    BackedgeBuilder backedges(b, parent.getLoc());

    // The instances, their inputs to be connected.
    DenseMap<Value, Value> netValues;
    for (auto [index, arg] : llvm::enumerate(netBody->getArguments()))
      netValues[body->getArgument(index)] = arg;
    SmallVector<Backedge> logicInputs;
    SmallVector<Value> operands(netBody->getArguments().begin(),
                                netBody->getArguments().end());
    for (auto source : newInputs) {
      logicInputs.push_back(backedges.get(b.getIntegerType(source.width)));
      operands.push_back(logicInputs.back());
    }
    auto logic = hw::InstanceOp::create(b, parent, "logic", operands);
    SmallVector<SmallVector<Backedge>> modelInputs;
    SmallVector<hw::InstanceOp> netInstances;
    for (auto instance : instances) {
      auto &inputs = modelInputs.emplace_back();
      SmallVector<Value> instanceOperands;
      for (auto operand : instance.getInputs()) {
        inputs.push_back(backedges.get(operand.getType()));
        instanceOperands.push_back(inputs.back());
      }
      auto netInstance = hw::InstanceOp::create(
          b,
          SymbolTable::lookupSymbolIn(top,
                                      instance.getModuleNameAttr().getAttr()),
          instance.getInstanceNameAttr(), instanceOperands,
          instance.getParametersAttr(), instance.getInnerSymAttr());
      netInstances.push_back(netInstance);
      for (auto [result, netResult] :
           llvm::zip(instance.getResults(), netInstance.getResults()))
        netValues[result] = netResult;
    }

    // `width` bits of `value` from `low`, as an integer.
    auto bitsOf = [&](Value value, unsigned low, unsigned width) -> Value {
      auto valueWidth = hw::getBitWidth(value.getType());
      if (!value.getType().isInteger(valueWidth))
        value =
            b.createOrFold<hw::BitcastOp>(b.getIntegerType(valueWidth), value);
      return b.createOrFold<comb::ExtractOp>(b.getIntegerType(width), value,
                                             low);
    };
    auto castTo = [&](Value value, Type type) -> Value {
      return value.getType() == type
                 ? value
                 : b.createOrFold<hw::BitcastOp>(type, value);
    };
    for (auto [backedge, source] : llvm::zip(logicInputs, newInputs))
      backedge.setValue(
          bitsOf(netValues[source.result], source.low, source.width));
    for (auto [instanceIndex, instance] : llvm::enumerate(instances)) {
      for (auto [index, operand] : llvm::enumerate(instance.getInputs())) {
        auto type = operand.getType();
        Value value;
        auto plan = inputPlans.find({instanceIndex, index});
        if (plan != inputPlans.end() && plan->second.output) {
          value =
              castTo(logic.getResult(numOutputs + *plan->second.output), type);
        } else if (plan != inputPlans.end() && plan->second.bits) {
          // The bits, from the most significant, in runs of one source.
          auto &bits = *plan->second.bits;
          SmallVector<Value> parts;
          for (unsigned hi = bits.size(); hi > 0;) {
            auto &first = bits[hi - 1];
            unsigned lo = hi - 1;
            while (lo > 0) {
              auto &next = bits[lo - 1];
              bool same = first.kind == BitSource::Constant
                              ? next.kind == BitSource::Constant
                              : next.kind == BitSource::Port &&
                                    next.port == first.port &&
                                    next.bit + (hi - lo) == first.bit;
              if (!same)
                break;
              --lo;
            }
            if (first.kind == BitSource::Constant) {
              APInt constant(hi - lo, 0);
              for (unsigned bit = lo; bit < hi; ++bit)
                constant.setBitVal(bit - lo, bits[bit].value);
              parts.push_back(hw::ConstantOp::create(b, constant));
            } else {
              parts.push_back(bitsOf(netValues[first.port],
                                     first.bit - (hi - 1 - lo), hi - lo));
            }
            hi = lo;
          }
          value =
              castTo(parts.size() == 1 ? parts[0]
                                       : b.createOrFold<comb::ConcatOp>(parts),
                     type);
        } else if (isa<seq::ClockType>(type)) {
          value = netValues.lookup(operand);
          if (!value) {
            instance.emitError("clock of a model not from a parent input");
            return signalPassFailure();
          }
        } else {
          // Not a channel: an input the model does not read.
          auto width = std::max<int64_t>(hw::getBitWidth(type), 0);
          value =
              castTo(hw::ConstantOp::create(b, APInt::getZero(width)), type);
        }
        modelInputs[instanceIndex][index].setValue(value);
      }
    }
    hw::OutputOp::create(b,
                         ValueRange(logic.getResults().take_front(numOutputs)));
  }
  for (auto instance : instances)
    instance.erase();
}
