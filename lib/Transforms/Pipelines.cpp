//===- Pipelines.cpp - fixed-latency pipelines of a model -----------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The analysis works on bits: a register of a shift register holds every
// stage of it, so its bits are at different layers. Each bit of each value is
// given the set of state bits (register bits, black-box output bits) it is
// computed from combinationally, as intervals of atoms; atom 0 stands for
// anything else (an input, a memory's contents).
//
//===----------------------------------------------------------------------===//

#include "Pipelines.h"
#include "FAME5.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <set>

#define DEBUG_TYPE "foldsim-pipelines"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Contracts
//===----------------------------------------------------------------------===//

static StringRef unquote(StringRef s) {
  s = s.trim();
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
    return s.drop_front().drop_back();
  return s;
}

bool foldsim::isBlackBox(Operation *module) {
  if (auto attrs = sv::getSVAttributes(module))
    for (auto attr : attrs)
      if (auto sva = dyn_cast<sv::SVAttributeAttr>(attr);
          sva && sva.getName().getValue() == "black_box")
        return true;
  return false;
}

FailureOr<std::optional<foldsim::PipelineContract>>
foldsim::getPipelineContract(Operation *module) {
  static const char *keys[] = {"fixed_latency", "clock",    "clock_enable",
                               "data_in",       "data_out", "tie",
                               "pure_pipeline"};
  llvm::StringMap<std::string> values;
  auto attrs = sv::getSVAttributes(module);
  if (attrs)
    for (auto attr : attrs) {
      auto sva = dyn_cast<sv::SVAttributeAttr>(attr);
      if (!sva || !llvm::is_contained(keys, sva.getName().getValue()))
        continue;
      values[sva.getName().getValue()] =
          sva.getExpression() ? unquote(sva.getExpression().getValue()).str()
                              : "";
    }
  if (values.empty())
    return std::optional<PipelineContract>{};
  auto name = SymbolTable::getSymbolName(module).getValue();
  auto fail = [&](const Twine &message) {
    module->emitError("pipeline contract of '") << name << "': " << message;
    return failure();
  };
  for (auto *key : keys)
    if (StringRef(key) != "tie" && !values.count(key))
      return fail(Twine("no `") + key + "`");
  PipelineContract contract;
  if (StringRef(values["fixed_latency"]).getAsInteger(10, contract.latency) ||
      contract.latency < 1)
    return fail("`fixed_latency` is not a latency of at least one cycle");
  contract.clock = values["clock"];
  contract.enable = values["clock_enable"];
  auto ports = cast<hw::HWModuleLike>(module).getPortList();
  auto port = [&](StringRef portName,
                  bool input) -> std::optional<hw::PortInfo> {
    for (auto &p : ports)
      if (p.getName() == portName &&
          (p.dir == hw::ModulePort::Direction::Input) == input)
        return p;
    return std::nullopt;
  };
  for (auto [key, input] :
       {std::make_pair("clock", true), std::make_pair("clock_enable", true)})
    if (!port(values[key], input))
      return fail(Twine("no input '") + values[key] + "'");
  SmallVector<StringRef> words;
  for (auto [key, input] :
       {std::make_pair("data_in", true), std::make_pair("data_out", false)}) {
    auto &list = input ? contract.dataIn : contract.dataOut;
    words.clear();
    StringRef(values[key]).split(words, ' ', -1, false);
    for (auto w : words) {
      if (!port(w, input))
        return fail(Twine("no ") + (input ? "input" : "output") + " '" + w +
                    "'");
      list.push_back(w.str());
    }
    if (list.empty())
      return fail(Twine("no ") + key + " ports");
  }
  words.clear();
  StringRef(values["tie"]).split(words, ' ', -1, false);
  for (auto w : words) {
    auto [portName, value] = w.split('=');
    int64_t v;
    if (value.getAsInteger(0, v) || !port(portName, true))
      return fail(Twine("tie '") + w + "' is not <input>=<value>");
    contract.tied.push_back({portName.str(), v});
  }
  // Every input is the clock, the enable, data or tied.
  for (auto &p : ports) {
    if (p.dir != hw::ModulePort::Direction::Input)
      continue;
    auto n = p.getName();
    if (n == contract.clock || n == contract.enable ||
        llvm::is_contained(contract.dataIn, n) ||
        llvm::any_of(contract.tied, [&](auto &t) { return t.first == n; }))
      continue;
    return fail(Twine("input '") + n + "' is neither data nor tied");
  }
  return std::optional<PipelineContract>{contract};
}

//===----------------------------------------------------------------------===//
// Sets of atoms
//===----------------------------------------------------------------------===//

namespace {

using Ranges = SmallVector<std::pair<uint32_t, uint32_t>, 2>;

/// Sets of atoms as sorted disjoint intervals, each stored once. Set 0 is the
/// empty set.
class AtomSets {
public:
  AtomSets() { intern({}); }
  unsigned single(uint32_t atom) { return intern({{atom, atom + 1}}); }
  unsigned range(uint32_t lo, uint32_t hi) { return intern({{lo, hi}}); }
  unsigned unite(unsigned a, unsigned b) {
    if (a == b || b == 0)
      return a;
    if (a == 0)
      return b;
    auto key = std::make_pair(std::min(a, b), std::max(a, b));
    if (auto it = unions.find(key); it != unions.end())
      return it->second;
    auto &x = sets[a];
    auto &y = sets[b];
    Ranges merged;
    size_t i = 0, j = 0;
    auto push = [&](std::pair<uint32_t, uint32_t> r) {
      if (!merged.empty() && r.first <= merged.back().second)
        merged.back().second = std::max(merged.back().second, r.second);
      else
        merged.push_back(r);
    };
    while (i < x.size() || j < y.size()) {
      if (j == y.size() || (i < x.size() && x[i].first < y[j].first))
        push(x[i++]);
      else
        push(y[j++]);
    }
    return unions[key] = intern(std::move(merged));
  }
  const Ranges &get(unsigned id) const { return sets[id]; }
  size_t size() const { return sets.size(); }

private:
  unsigned intern(Ranges set) {
    llvm::hash_code hash = llvm::hash_value(set.size());
    for (auto [lo, hi] : set)
      hash = llvm::hash_combine(hash, lo, hi);
    auto &ids = byHash[hash];
    for (auto id : ids)
      if (sets[id] == set)
        return id;
    ids.push_back(sets.size());
    sets.push_back(std::move(set));
    return sets.size() - 1;
  }
  std::vector<Ranges> sets;
  DenseMap<llvm::hash_code, SmallVector<unsigned, 1>> byHash;
  DenseMap<std::pair<unsigned, unsigned>, unsigned> unions;
};

using BitSets = SmallVector<unsigned>;

/// A bit of a value, or a constant bit (`value` null).
struct Leaf {
  Value value;
  unsigned bit = 0;
  bool operator==(const Leaf &o) const {
    return value == o.value && bit == o.bit;
  }
  bool operator<(const Leaf &o) const {
    if (value != o.value)
      return value.getAsOpaquePointer() < o.value.getAsOpaquePointer();
    return bit < o.bit;
  }
};
using Leaves = SmallVector<Leaf, 4>;

} // namespace

//===----------------------------------------------------------------------===//
// The analysis
//===----------------------------------------------------------------------===//

struct foldsim::PipelineAnalysis::Impl {
  explicit Impl(hw::HWModuleOp model) : model(model) {}

  hw::HWModuleOp model;
  AtomSets atoms;
  /// Atom 0 is "anything else".
  uint32_t numAtoms = 1;
  DenseMap<Value, uint32_t> stateBase;
  DenseMap<Value, std::unique_ptr<BitSets>> sets;
  DenseMap<Value, unsigned> allSets;
  /// The members of the combinational cycle being solved: their sets are
  /// partial until it is.
  DenseSet<Value> open;

  const BitSets &get(Value value);
  BitSets compute(Value value);
  /// The values whose sets `compute` reads.
  void operandsOf(Value value, SmallVectorImpl<Value> &out);
  /// Computes the sets of `value` and of all it reads: the strongly connected
  /// components of the values (Tarjan's, without recursion: a model's logic
  /// may chain far more operations than a thread's stack holds frames), each
  /// once its reads are, a combinational cycle's to a fixed point.
  void solve(Value value);
  unsigned all(Value value) {
    if (auto it = allSets.find(value); it != allSets.end())
      return it->second;
    unsigned any = 0;
    auto bits = get(value);
    for (auto bit : bits)
      any = atoms.unite(any, bit);
    if (open.contains(value))
      return any;
    return allSets[value] = any;
  }

  // Tracing a bit through wiring and muxes, for the valid chains.
  std::map<Leaf, int> assume;
  bool trace(Value value, unsigned bit, Leaves &out, unsigned depth = 0);
  std::optional<bool> evalBool(Value value, unsigned depth);
  /// The atoms of each pipeline found.
  std::vector<llvm::BitVector> memberAtom;

  std::optional<Leaf> single(Value value, unsigned bit = 0) {
    Leaves leaves;
    if (!trace(value, bit, leaves) || leaves.size() != 1 ||
        !leaves[0].value)
      return std::nullopt;
    return leaves[0];
  }
};

foldsim::PipelineAnalysis::PipelineAnalysis(hw::HWModuleOp model)
    : impl(std::make_unique<Impl>(model)) {}
foldsim::PipelineAnalysis::~PipelineAnalysis() = default;

const BitSets &foldsim::PipelineAnalysis::Impl::get(Value value) {
  auto it = sets.find(value);
  if (it == sets.end()) {
    solve(value);
    it = sets.find(value);
  }
  return *it->second;
}

void foldsim::PipelineAnalysis::Impl::operandsOf(Value value,
                                                SmallVectorImpl<Value> &out) {
  if (stateBase.count(value))
    return;
  auto *op = value.getDefiningOp();
  if (!op || isa<seq::FirRegOp, hw::InstanceOp, hw::ConstantOp,
                 hw::AggregateConstantOp>(op))
    return;
  if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
    auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
    if (!memory || memory.getReadLatency() == 0) {
      out.push_back(read.getAddress());
      if (read.getEnable())
        out.push_back(read.getEnable());
    }
    return;
  }
  out.append(op->operand_begin(), op->operand_end());
}

void foldsim::PipelineAnalysis::Impl::solve(Value root) {
  // Each value visited: its index and lowest reachable index on the stack.
  DenseMap<Value, std::pair<unsigned, unsigned>> link;
  SmallVector<Value> stack;
  DenseSet<Value> onStack;
  struct Frame {
    Value value;
    SmallVector<Value, 4> reads;
    unsigned next = 0;
  };
  SmallVector<Frame> frames;
  auto visit = [&](Value value) {
    unsigned index = link.size();
    link[value] = {index, index};
    stack.push_back(value);
    onStack.insert(value);
    Frame frame{value, {}, 0};
    operandsOf(value, frame.reads);
    frames.push_back(std::move(frame));
  };
  visit(root);
  while (!frames.empty()) {
    auto &frame = frames.back();
    if (frame.next < frame.reads.size()) {
      Value read = frame.reads[frame.next++];
      if (sets.count(read))
        continue;
      auto it = link.find(read);
      if (it == link.end()) {
        visit(read);
        continue;
      }
      if (onStack.contains(read)) {
        auto &low = link[frame.value].second;
        low = std::min(low, it->second.first);
      }
      continue;
    }
    Value value = frame.value;
    frames.pop_back();
    auto [index, low] = link[value];
    if (!frames.empty()) {
      auto &callerLow = link[frames.back().value].second;
      callerLow = std::min(callerLow, low);
    }
    if (low != index)
      continue;
    SmallVector<Value> component;
    do {
      component.push_back(stack.pop_back_val());
      onStack.erase(component.back());
    } while (component.back() != value);
    // Its reads are solved: a value alone once, a combinational cycle from
    // empty sets until none changes.
    bool cycle = component.size() > 1;
    if (!cycle) {
      SmallVector<Value, 4> reads;
      operandsOf(value, reads);
      cycle = llvm::is_contained(reads, value);
    }
    for (auto member : component) {
      auto width = std::max<int64_t>(hw::getBitWidth(member.getType()), 0);
      sets[member] = std::make_unique<BitSets>(width, 0);
      if (cycle)
        open.insert(member);
    }
    for (bool changed = true; changed;) {
      changed = false;
      for (auto member : component) {
        auto bits = compute(member);
        if (bits != *sets[member]) {
          *sets[member] = std::move(bits);
          changed = true;
        }
      }
      if (!cycle)
        break;
    }
    for (auto member : component)
      open.erase(member);
  }
}

BitSets foldsim::PipelineAnalysis::Impl::compute(Value value) {
  auto width = std::max<int64_t>(hw::getBitWidth(value.getType()), 0);
  BitSets bits(width, 0);
  auto copyBits = [&](Value operand, unsigned from, unsigned to,
                      unsigned count) {
    auto operandBits = get(operand);
    for (unsigned i = 0; i < count && from + i < operandBits.size(); ++i)
      bits[to + i] = atoms.unite(bits[to + i], operandBits[from + i]);
  };
  auto fill = [&](unsigned set) {
    for (auto &bit : bits)
      bit = atoms.unite(bit, set);
  };
  if (auto it = stateBase.find(value); it != stateBase.end()) {
    for (unsigned i = 0; i < bits.size(); ++i)
      bits[i] = atoms.single(it->second + i);
    return bits;
  }
  auto *op = value.getDefiningOp();
  APInt index;
  if (!op || isa<seq::FirRegOp, hw::InstanceOp>(op)) {
    // An input, or state that is not a register or black box of a pipeline.
    fill(atoms.single(0));
  } else if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
    fill(atoms.single(0));
    auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
    if (!memory || memory.getReadLatency() == 0) {
      fill(all(read.getAddress()));
      if (read.getEnable())
        fill(all(read.getEnable()));
    }
  } else if (isa<hw::ConstantOp, hw::AggregateConstantOp>(op)) {
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
    for (unsigned at = 0; w > 0 && at + w <= unsigned(width); at += w)
      copyBits(replicate.getInput(), 0, at, w);
  } else if (isa<hw::BitcastOp>(op)) {
    copyBits(op->getOperand(0), 0, 0, width);
  } else if (isa<hw::ArrayCreateOp, hw::ArrayConcatOp, hw::StructCreateOp>(
                 op)) {
    unsigned at = 0;
    for (auto operand : llvm::reverse(op->getOperands())) {
      auto w = hw::getBitWidth(operand.getType());
      copyBits(operand, 0, at, w);
      at += w;
    }
  } else if (auto get = dyn_cast<hw::ArrayGetOp>(op)) {
    auto size =
        hw::type_cast<hw::ArrayType>(get.getInput().getType()).getNumElements();
    if (matchPattern(get.getIndex(), m_ConstantInt(&index))) {
      if (index.getZExtValue() < size)
        copyBits(get.getInput(), index.getZExtValue() * width, 0, width);
    } else {
      for (unsigned e = 0; e < size; ++e)
        copyBits(get.getInput(), e * width, 0, width);
      fill(all(get.getIndex()));
    }
  } else if (auto slice = dyn_cast<hw::ArraySliceOp>(op);
             slice &&
             matchPattern(slice.getLowIndex(), m_ConstantInt(&index))) {
    auto elementWidth = hw::getBitWidth(
        hw::type_cast<hw::ArrayType>(slice.getType()).getElementType());
    copyBits(slice.getInput(), index.getZExtValue() * elementWidth, 0, width);
  } else if (auto extract = dyn_cast<hw::StructExtractOp>(op)) {
    auto elements =
        hw::type_cast<hw::StructType>(extract.getInput().getType())
            .getElements();
    unsigned at = 0;
    for (auto i = elements.size(); i-- > extract.getFieldIndex() + 1;)
      at += hw::getBitWidth(elements[i].type);
    copyBits(extract.getInput(), at, 0, width);
  } else if (isa<comb::AndOp, comb::OrOp, comb::XorOp>(op)) {
    for (auto operand : op->getOperands())
      copyBits(operand, 0, 0, width);
  } else if (auto mux = dyn_cast<comb::MuxOp>(op)) {
    copyBits(mux.getTrueValue(), 0, 0, width);
    copyBits(mux.getFalseValue(), 0, 0, width);
    fill(all(mux.getCond()));
  } else {
    unsigned any = 0;
    for (auto operand : op->getOperands())
      any = atoms.unite(any, all(operand));
    fill(any);
  }
  return bits;
}

std::optional<bool> foldsim::PipelineAnalysis::Impl::evalBool(Value value,
                                                              unsigned depth) {
  Leaves leaves;
  if (!trace(value, 0, leaves, depth + 1) || leaves.empty())
    return std::nullopt;
  for (auto &leaf : leaves)
    if (leaf.value || leaf.bit != leaves[0].bit)
      return std::nullopt;
  return leaves[0].bit != 0;
}

/// The bits `value`'s bit `bit` may be, through wiring, muxes whose condition
/// is not known (both sides) and logic with constants: a constant (null
/// value, `bit` its value) or a bit of a value it does not look through.
/// Fails if there are more than four, or the path is too long.
bool foldsim::PipelineAnalysis::Impl::trace(Value value, unsigned bit,
                                           Leaves &out, unsigned depth) {
  if (depth > 64)
    return false;
  auto add = [&](Leaf leaf) {
    if (!llvm::is_contained(out, leaf))
      out.push_back(leaf);
    return out.size() <= 4;
  };
  if (auto it = assume.find({value, bit}); it != assume.end())
    return add({Value(), unsigned(it->second)});
  auto *op = value.getDefiningOp();
  APInt index;
  if (!op || isa<seq::FirRegOp, hw::InstanceOp, seq::FirMemReadOp>(op))
    return add({value, bit});
  if (auto constant = dyn_cast<hw::ConstantOp>(op))
    return add({Value(), unsigned(constant.getValue()[bit])});
  if (auto extract = dyn_cast<comb::ExtractOp>(op))
    return trace(extract.getInput(), extract.getLowBit() + bit, out,
                 depth + 1);
  if (isa<hw::BitcastOp>(op))
    return trace(op->getOperand(0), bit, out, depth + 1);
  if (auto replicate = dyn_cast<comb::ReplicateOp>(op))
    return trace(replicate.getInput(),
                 bit % hw::getBitWidth(replicate.getInput().getType()), out,
                 depth + 1);
  if (isa<comb::ConcatOp, hw::ArrayCreateOp, hw::ArrayConcatOp,
          hw::StructCreateOp>(op)) {
    for (auto operand : llvm::reverse(op->getOperands())) {
      auto w = hw::getBitWidth(operand.getType());
      if (bit < w)
        return trace(operand, bit, out, depth + 1);
      bit -= w;
    }
    return false;
  }
  if (auto get = dyn_cast<hw::ArrayGetOp>(op);
      get && matchPattern(get.getIndex(), m_ConstantInt(&index)))
    return trace(get.getInput(),
                 index.getZExtValue() * hw::getBitWidth(get.getType()) + bit,
                 out, depth + 1);
  if (auto extract = dyn_cast<hw::StructExtractOp>(op)) {
    auto elements =
        hw::type_cast<hw::StructType>(extract.getInput().getType())
            .getElements();
    for (auto &element : elements.drop_front(extract.getFieldIndex() + 1))
      bit += hw::getBitWidth(element.type);
    return trace(extract.getInput(), bit, out, depth + 1);
  }
  if (auto mux = dyn_cast<comb::MuxOp>(op)) {
    auto cond = evalBool(mux.getCond(), depth);
    if (cond)
      return trace(*cond ? mux.getTrueValue() : mux.getFalseValue(), bit, out,
                   depth + 1);
    return trace(mux.getTrueValue(), bit, out, depth + 1) &&
           trace(mux.getFalseValue(), bit, out, depth + 1);
  }
  if (isa<comb::AndOp, comb::OrOp, comb::XorOp>(op)) {
    // The operands that are not the identity's constant; one absorbing
    // constant decides it.
    bool isAnd = isa<comb::AndOp>(op), isXor = isa<comb::XorOp>(op);
    unsigned identity = isAnd ? 1 : 0;
    SmallVector<Leaves> rest;
    unsigned flips = 0;
    for (auto operand : op->getOperands()) {
      Leaves leaves;
      if (!trace(operand, bit, leaves, depth + 1))
        return add({value, bit});
      if (leaves.size() == 1 && !leaves[0].value) {
        if (leaves[0].bit == identity)
          continue;
        if (!isXor)
          return add({Value(), 1 - identity});
        ++flips;
        continue;
      }
      rest.push_back(leaves);
    }
    if (rest.empty())
      return add({Value(), (isXor ? flips % 2 : identity)});
    if (rest.size() == 1 && flips % 2 == 0) {
      for (auto &leaf : rest[0])
        if (!add(leaf))
          return false;
      return true;
    }
    return add({value, bit});
  }
  return add({value, bit});
}

bool foldsim::PipelineAnalysis::dependsOnPipeline(Value value,
                                                  unsigned pipeline) {
  auto &d = *impl;
  auto &members = d.memberAtom[pipeline];
  for (auto set : d.get(value))
    for (auto [lo, hi] : d.atoms.get(set))
      if (std::max<uint32_t>(lo, 1) < hi &&
          members.find_first_in(std::max<uint32_t>(lo, 1), hi) >= 0)
        return true;
  return false;
}

/// The operands of an `and` of one bit, nested ands flattened; the value
/// itself if it is not one.
static void andFactors(Value value, SmallVectorImpl<Value> &factors) {
  if (!value)
    return;
  if (auto andOp = value.getDefiningOp<comb::AndOp>();
      andOp && andOp.getType().isInteger(1)) {
    for (auto operand : andOp.getInputs())
      andFactors(operand, factors);
    return;
  }
  if (!llvm::is_contained(factors, value))
    factors.push_back(value);
}

/// The common hierarchical prefix of `names`, up to its last separator.
static std::string commonPath(ArrayRef<StringRef> names) {
  if (names.empty())
    return "";
  StringRef prefix = names[0];
  for (auto name : names.drop_front()) {
    size_t n = 0;
    while (n < prefix.size() && n < name.size() && prefix[n] == name[n])
      ++n;
    prefix = prefix.take_front(n);
  }
  auto cut = prefix.find_last_of('/');
  if (cut == StringRef::npos)
    return names.size() == 1 ? names[0].str() : "";
  return prefix.take_front(cut).str();
}

LogicalResult foldsim::PipelineAnalysis::run(ImplicitLocOpBuilder &b) {
  auto &d = *impl;
  auto *body = d.model.getBodyBlock();
  std::map<std::string, uint64_t> reasons;

  // The black boxes with a contract, their contract checked.
  std::vector<PipelineBox> allBoxes;
  DenseSet<Operation *> boxOps;
  for (auto inst : body->getOps<hw::InstanceOp>()) {
    auto *module = SymbolTable::lookupNearestSymbolFrom(
        inst, inst.getReferencedModuleNameAttr());
    if (!module)
      continue;
    auto contract = getPipelineContract(module);
    if (failed(contract))
      return failure();
    if (!*contract) {
      if (isBlackBox(module))
        reasons["black box without a pipeline contract"] += 0;
      continue;
    }
    auto &c = **contract;
    auto portList = cast<hw::HWModuleLike>(module).getPortList();
    for (auto &port : portList) {
      if (port.dir == hw::ModulePort::Direction::Input) {
        auto tie = llvm::find_if(
            c.tied, [&](auto &t) { return t.first == port.getName(); });
        if (tie == c.tied.end())
          continue;
        APInt value;
        auto operand = inst.getOperand(port.argNum);
        if (!matchPattern(operand, m_ConstantInt(&value)) ||
            value != APInt(value.getBitWidth(), uint64_t(tie->second),
                           /*isSigned=*/false, /*implicitTrunc=*/true))
          return inst.emitError("input '")
                 << port.getName() << "' of black box '"
                 << inst.getInstanceName() << "' is not tied to "
                 << tie->second << " as its pipeline contract says";
      } else if (!llvm::is_contained(c.dataOut, port.getName()) &&
                 !inst.getResult(port.argNum).use_empty()) {
        return inst.emitError("output '")
               << port.getName() << "' of black box '"
               << inst.getInstanceName()
               << "' is read, but its pipeline contract does not say what it "
                  "is";
      }
    }
    PipelineBox box;
    box.inst = inst;
    box.contract = c;
    allBoxes.push_back(box);
    boxOps.insert(inst);
  }
  auto boxPort = [&](PipelineBox &box, StringRef portName,
                     bool input) -> Value {
    auto module = SymbolTable::lookupNearestSymbolFrom(
        box.inst, box.inst.getReferencedModuleNameAttr());
    for (auto &port : cast<hw::HWModuleLike>(module).getPortList())
      if (port.getName() == portName &&
          (port.dir == hw::ModulePort::Direction::Input) == input)
        return input ? box.inst.getOperand(port.argNum)
                     : box.inst.getResult(port.argNum);
    return {};
  };

  // The candidate registers: no reset or preset, written on some enable.
  std::vector<PipelineRegister> candidates;
  SmallVector<seq::FirRegOp> allRegisters(body->getOps<seq::FirRegOp>());
  DenseMap<Operation *, unsigned> candidateIndex;
  for (auto reg : allRegisters) {
    auto width = hw::getBitWidth(reg.getType());
    if (width <= 0)
      continue;
    if (reg.getReset() || reg.getPreset() || reg.getIsAsync()) {
      reasons["reset or preset"] += width;
      continue;
    }
    auto write = peelHolds(b, reg.getNext(), reg.getResult());
    if (write.holds)
      continue;
    PipelineRegister candidate;
    candidate.reg = reg;
    candidate.enable = write.enable;
    candidate.data = write.enable ? write.data : reg.getNext();
    candidate.layer.assign(width, 0);
    candidateIndex[reg] = candidates.size();
    candidates.push_back(candidate);
  }

  // Atoms and nodes: a node per candidate bit and per black box.
  struct Node {
    int reg = -1, box = -1;
    unsigned bit = 0;
    unsigned sources = 0;
    Value enable;
  };
  std::vector<Node> nodes;
  std::vector<int> atomNode(1, -1);
  SmallVector<unsigned> regNode;
  for (auto [i, candidate] : llvm::enumerate(candidates)) {
    d.stateBase[candidate.reg.getResult()] = d.numAtoms;
    regNode.push_back(nodes.size());
    for (unsigned bit = 0; bit < candidate.layer.size(); ++bit) {
      atomNode.push_back(nodes.size());
      ++d.numAtoms;
      Node node;
      node.reg = i;
      node.bit = bit;
      node.enable = candidate.enable;
      nodes.push_back(node);
    }
  }
  SmallVector<unsigned> boxNode;
  for (auto [i, box] : llvm::enumerate(allBoxes)) {
    boxNode.push_back(nodes.size());
    for (auto &out : box.contract.dataOut) {
      auto result = boxPort(box, out, false);
      d.stateBase[result] = d.numAtoms;
      for (int64_t k = 0; k < hw::getBitWidth(result.getType()); ++k) {
        atomNode.push_back(nodes.size());
        ++d.numAtoms;
      }
    }
    Node node;
    node.box = i;
    node.enable = boxPort(box, box.contract.enable, true);
    nodes.push_back(node);
  }
  for (auto &node : nodes) {
    if (node.reg >= 0) {
      node.sources = d.get(candidates[node.reg].data)[node.bit];
      continue;
    }
    for (auto &in : allBoxes[node.box].contract.dataIn)
      node.sources =
          d.atoms.unite(node.sources, d.all(boxPort(allBoxes[node.box], in,
                                                    true)));
  }
  auto nodeBits = [&](unsigned n) -> uint64_t {
    if (nodes[n].reg >= 0)
      return 1;
    uint64_t bits = 0;
    for (auto &out : allBoxes[nodes[n].box].contract.dataOut)
      bits += hw::getBitWidth(
          boxPort(allBoxes[nodes[n].box], out, false).getType());
    return bits;
  };

  // Nodes are connected only through compatible enables: both always
  // written, or with a term in common. Others read each other as inputs.
  std::vector<SmallVector<Value>> factorsOf(nodes.size());
  for (unsigned n = 0; n < nodes.size(); ++n)
    andFactors(nodes[n].enable, factorsOf[n]);
  auto compatible = [&](unsigned n, unsigned m) {
    if (factorsOf[n].empty() || factorsOf[m].empty())
      return factorsOf[n].empty() && factorsOf[m].empty();
    return llvm::any_of(factorsOf[n], [&](Value v) {
      return llvm::is_contained(factorsOf[m], v);
    });
  };
  std::vector<char> included(nodes.size(), 1);
  std::vector<unsigned> layer(nodes.size(), 0), inLayer(nodes.size(), 0);
  std::vector<int> component(nodes.size(), -1);
  struct Component {
    SmallVector<unsigned> nodes;
    unsigned layers = 0;
    Value base;
    bool gated = false, stored = false;
  };
  std::vector<Component> components;
  SmallVector<std::pair<unsigned, std::string>> violations;
  auto violate = [&](unsigned n, StringRef why) {
    violations.push_back({n, why.str()});
  };

  // Which included nodes a set has, and whether it has anything else.
  DenseMap<unsigned, std::pair<SmallVector<unsigned>, bool>> info;
  auto infoOf = [&](unsigned set) -> std::pair<SmallVector<unsigned>, bool> & {
    auto [it, inserted] = info.try_emplace(set);
    if (!inserted)
      return it->second;
    auto &[list, ext] = it->second;
    for (auto [lo, hi] : d.atoms.get(set)) {
      if (lo == 0)
        ext = true;
      for (uint32_t a = std::max<uint32_t>(lo, 1); a < hi; ++a) {
        auto n = atomNode[a];
        if (!included[n])
          ext = true;
        else if (list.empty() || list.back() != unsigned(n))
          list.push_back(n);
      }
    }
    llvm::sort(list);
    list.erase(std::unique(list.begin(), list.end()), list.end());
    return it->second;
  };
  auto outLayerOf = [&](unsigned n) { return layer[n]; };
  // The nodes `n` reads, and whether it reads anything else.
  std::vector<std::optional<std::pair<SmallVector<unsigned>, bool>>> predCache;
  auto predsOf = [&](unsigned n) -> std::pair<SmallVector<unsigned>, bool> & {
    if (predCache[n])
      return *predCache[n];
    auto &[list, ext] = infoOf(nodes[n].sources);
    auto &result = predCache[n].emplace();
    result.second = ext;
    for (auto m : list) {
      if (compatible(n, m))
        result.first.push_back(m);
      else
        result.second = true;
    }
    return result;
  };

  for (bool changed = true; changed;) {
    changed = false;
    info.clear();
    predCache.assign(nodes.size(), std::nullopt);
    violations.clear();
    components.clear();
    std::fill(component.begin(), component.end(), -1);

    // Strongly connected components, dependences first (Tarjan).
    std::vector<int> index(nodes.size(), -1), low(nodes.size(), 0);
    std::vector<char> onStack(nodes.size(), 0);
    SmallVector<unsigned> tstack;
    SmallVector<SmallVector<unsigned>> sccs;
    int counter = 0;
    for (unsigned root = 0; root < nodes.size(); ++root) {
      if (!included[root] || index[root] >= 0)
        continue;
      SmallVector<std::pair<unsigned, unsigned>> work = {{root, 0}};
      index[root] = low[root] = counter++;
      tstack.push_back(root);
      onStack[root] = 1;
      while (!work.empty()) {
        auto &[n, e] = work.back();
        auto &preds = predsOf(n).first;
        if (e < preds.size()) {
          unsigned m = preds[e++];
          if (index[m] < 0) {
            index[m] = low[m] = counter++;
            tstack.push_back(m);
            onStack[m] = 1;
            work.push_back({m, 0});
          } else if (onStack[m]) {
            low[n] = std::min(low[n], index[m]);
          }
          continue;
        }
        if (low[n] == index[n]) {
          SmallVector<unsigned> scc;
          unsigned m;
          do {
            m = tstack.pop_back_val();
            onStack[m] = 0;
            scc.push_back(m);
          } while (m != n);
          sccs.push_back(scc);
        }
        unsigned done = n;
        work.pop_back();
        if (!work.empty())
          low[work.back().first] =
              std::min(low[work.back().first], low[done]);
      }
    }

    // Layers, in dependence order.
    std::vector<char> bad(nodes.size(), 0);
    for (auto &scc : sccs) {
      auto n = scc[0];
      auto [preds, ext] = predsOf(n);
      if (scc.size() > 1 || llvm::is_contained(preds, n)) {
        for (auto m : scc) {
          violate(m, "feedback");
          bad[m] = 1;
        }
        continue;
      }
      unsigned from = 0;
      bool ok = true;
      for (auto m : preds) {
        if (bad[m]) {
          ok = false;
          break;
        }
        if (from && outLayerOf(m) != from) {
          violate(n, "unbalanced paths");
          ok = false;
          break;
        }
        from = outLayerOf(m);
      }
      if (ok && from && ext) {
        violate(n, "inputs at a later stage");
        ok = false;
      }
      if (!ok) {
        bad[n] = 1;
        continue;
      }
      if (nodes[n].reg >= 0) {
        layer[n] = from + 1;
      } else {
        inLayer[n] = from;
        layer[n] = from + allBoxes[nodes[n].box].contract.latency;
      }
    }

    if (violations.empty()) {
      // Connected sets.
      llvm::EquivalenceClasses<unsigned> classes;
      for (unsigned n = 0; n < nodes.size(); ++n) {
        if (!included[n])
          continue;
        classes.insert(n);
        for (auto m : predsOf(n).first)
          classes.unionSets(n, m);
      }
      DenseMap<unsigned, unsigned> leaderComponent;
      for (unsigned n = 0; n < nodes.size(); ++n) {
        if (!included[n])
          continue;
        auto leader = classes.getLeaderValue(n);
        auto [it, inserted] =
            leaderComponent.try_emplace(leader, components.size());
        if (inserted)
          components.emplace_back();
        component[n] = it->second;
        components[it->second].nodes.push_back(n);
      }
      for (auto &comp : components) {
        for (auto n : comp.nodes)
          comp.layers = std::max(comp.layers, layer[n]);
        for (auto n : comp.nodes)
          if (nodes[n].box >= 0 && layer[n] == comp.layers)
            comp.stored = true;
        // The enable common to all, and each one's gate.
        SmallVector<Value> common;
        bool first = true;
        DenseMap<unsigned, SmallVector<Value>> factors;
        for (auto n : comp.nodes) {
          auto &f = factors[n];
          andFactors(nodes[n].enable, f);
          if (first) {
            common = f;
            first = false;
            continue;
          }
          llvm::erase_if(common,
                         [&](Value v) { return !llvm::is_contained(f, v); });
        }
        LLVM_DEBUG({
          std::map<std::string, unsigned> regs;
          for (auto n : comp.nodes)
            regs[nodes[n].reg >= 0
                     ? candidates[nodes[n].reg].reg.getName().str()
                     : allBoxes[nodes[n].box].inst.getInstanceName().str()]++;
          llvm::dbgs() << "component L=" << comp.layers << " common "
                       << common.size() << ":";
          for (auto &[name, count] : regs)
            llvm::dbgs() << " " << name << "x" << count;
          llvm::dbgs() << "\n";
        });
        DenseMap<unsigned, Value> gate;
        for (auto n : comp.nodes) {
          SmallVector<Value> extra;
          for (auto v : factors[n])
            if (!llvm::is_contained(common, v))
              extra.push_back(v);
          if (extra.size() > 1 ||
              (extra.size() == 1 && !extra[0].getType().isInteger(1)))
            violate(n, "enable not the pipeline's and one gate");
          // A gated box holds its items through the gate's bubbles, so its
          // latency follows later valids; the copy runs it ungated.
          else if (extra.size() == 1 && nodes[n].box >= 0)
            violate(n, "black box with a gate on its enable");
          else if (extra.size() == 1)
            gate[n] = extra[0];
        }
        if (!gate.empty())
          comp.gated = true;
        if (common.empty()) {
          comp.base = {};
        } else if (common.size() == 1) {
          comp.base = common[0];
        } else if (comp.gated) {
          for (auto n : comp.nodes)
            violate(n, "gated, with an enable of several terms");
          continue;
        } else {
          comp.base = nodes[comp.nodes[0]].enable;
        }
        if (!comp.gated)
          continue;
        if (comp.stored) {
          for (auto [n, g] : gate)
            violate(n, "gated pipeline into a black box");
          continue;
        }
        // Each gate is the bit of the valid chain of its layer: the chain's
        // bit k back from it is the item's valid at issue.
        std::optional<Leaf> enableLeaf;
        if (comp.base) {
          enableLeaf = d.single(comp.base);
          if (!enableLeaf) {
            for (auto [n, g] : gate)
              violate(n, "gated, with an enable that is not one signal");
            continue;
          }
        }
        auto chainStep = [&](Leaf leaf) -> std::optional<Leaf> {
          auto reg = leaf.value.getDefiningOp<seq::FirRegOp>();
          if (!reg)
            return std::nullopt;
          if (reg.getReset()) {
            Leaves r;
            if (!d.trace(reg.getResetValue(), leaf.bit, r) || r.size() != 1 ||
                r[0].value || r[0].bit != 0)
              return std::nullopt;
          }
          d.assume.clear();
          if (enableLeaf)
            d.assume[*enableLeaf] = 1;
          Leaves on;
          bool ok = d.trace(reg.getNext(), leaf.bit, on);
          std::optional<Leaf> from;
          for (auto &l : on) {
            if (!l.value && l.bit == 0)
              continue;
            if (!l.value || from)
              ok = false;
            from = l;
          }
          if (enableLeaf) {
            d.assume[*enableLeaf] = 0;
            Leaves off;
            ok = ok && d.trace(reg.getNext(), leaf.bit, off);
            for (auto &l : off)
              if (!(l == leaf) && !(!l.value && l.bit == 0))
                ok = false;
          }
          d.assume.clear();
          if (!ok)
            return std::nullopt;
          return from;
        };
        std::optional<Leaf> issueValid;
        unsigned lastGated = 0;
        for (auto n : comp.nodes) {
          if (nodes[n].reg < 0)
            continue;
          auto it = gate.find(n);
          if (it == gate.end()) {
            if (layer[n] == comp.layers)
              violate(n, "ungated last layer of a gated pipeline");
            continue;
          }
          auto leaf = d.single(it->second);
          for (unsigned k = 1; leaf && k < layer[n]; ++k)
            leaf = chainStep(*leaf);
          if (!leaf || (issueValid && !(*leaf == *issueValid))) {
            violate(n, "gate not the valid of its stage");
            continue;
          }
          issueValid = leaf;
          lastGated = std::max(lastGated, layer[n]);
        }
        (void)lastGated;
      }
    }

    if (violations.empty()) {
      // A bit read outside the pipeline must be of its last layer.
      DenseSet<unsigned> seen;
      auto sink = [&](unsigned set) {
        if (!seen.insert(set).second)
          return;
        for (auto [lo, hi] : d.atoms.get(set))
          for (uint32_t a = std::max<uint32_t>(lo, 1); a < hi; ++a) {
            auto n = atomNode[a];
            if (included[n] &&
                layer[n] < components[component[n]].layers)
              violate(n, "read before the last stage");
          }
      };
      auto sinkAll = [&](Value v) {
        if (v)
          for (auto set : d.get(v))
            sink(set);
      };
      for (auto operand : body->getTerminator()->getOperands())
        sinkAll(operand);
      // A pipeline read by another, through an enable they do not share.
      for (unsigned n = 0; n < nodes.size(); ++n) {
        if (!included[n])
          continue;
        for (auto m : infoOf(nodes[n].sources).first)
          if (component[m] != component[n] &&
              layer[m] < components[component[m]].layers)
            violate(m, "read before the last stage");
      }
      for (auto reg : allRegisters) {
        auto it = candidateIndex.find(reg);
        if (it == candidateIndex.end()) {
          sinkAll(reg.getNext());
          sinkAll(reg.getReset());
          sinkAll(reg.getResetValue());
          continue;
        }
        bool any = false;
        auto &next = d.get(reg.getNext());
        for (unsigned bit = 0; bit < next.size(); ++bit) {
          if (included[regNode[it->second] + bit])
            any = true;
          else
            sink(next[bit]);
        }
        if (any)
          sinkAll(candidates[it->second].enable);
      }
      for (auto &op : *body) {
        if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
          auto box = llvm::find_if(allBoxes, [&](PipelineBox &x) {
            return x.inst == inst;
          });
          if (box != allBoxes.end() &&
              included[boxNode[box - allBoxes.begin()]]) {
            sinkAll(boxPort(*box, box->contract.enable, true));
            continue;
          }
          for (auto operand : inst.getOperands())
            sinkAll(operand);
          continue;
        }
        if (isa<seq::FirMemReadOp, seq::FirMemWriteOp>(op))
          for (auto operand : op.getOperands())
            if (!isa<seq::ClockType, seq::FirMemType>(operand.getType()))
              sinkAll(operand);
      }
    }

    for (auto &[n, why] : violations) {
      if (!included[n])
        continue;
      LLVM_DEBUG({
        auto &node = nodes[n];
        llvm::dbgs() << "exclude "
                     << (node.reg >= 0
                             ? candidates[node.reg].reg.getName()
                             : allBoxes[node.box].inst.getInstanceName())
                     << "[" << node.bit << "] layer " << layer[n] << ": "
                     << why << "\n";
      });
      included[n] = 0;
      reasons[why] += nodeBits(n);
      changed = true;
    }
  }

  // The pipelines: connected sets of at least three folded layers, merged by
  // enable and layers.
  std::map<std::tuple<void *, bool, unsigned>, unsigned> byKey;
  DenseMap<unsigned, unsigned> registerIndex, boxIndex;
  for (auto &comp : components) {
    unsigned depth = comp.stored ? comp.layers : comp.layers - 1;
    uint64_t folded = 0;
    for (auto n : comp.nodes)
      if (nodes[n].reg >= 0 && (comp.stored || layer[n] < comp.layers))
        ++folded;
    // The copy issues a host cycle late and writes a host cycle late: with
    // the thread's edges two host cycles apart, results are back in time
    // from three slots on.
    if (depth < 3) {
      uint64_t bits = 0;
      for (auto n : comp.nodes)
        bits += nodeBits(n);
      reasons["fewer than three stages to fold"] += bits;
      continue;
    }
    auto key = std::make_tuple(comp.base.getAsOpaquePointer(), comp.stored,
                               comp.layers);
    auto [it, inserted] = byKey.try_emplace(key, pipelines.size());
    if (inserted) {
      Pipeline p;
      p.enable = comp.base;
      p.layers = comp.layers;
      p.lastLayerStored = comp.stored;
      p.layerBits.assign(comp.layers, 0);
      pipelines.push_back(p);
      d.memberAtom.emplace_back(d.numAtoms);
    }
    auto &members = d.memberAtom[it->second];
    auto &p = pipelines[it->second];
    p.gated |= comp.gated;
    p.foldedBits += folded;
    SmallVector<StringRef> names;
    for (auto n : comp.nodes) {
      auto &node = nodes[n];
      if (node.reg >= 0) {
        auto [ri, newReg] =
            registerIndex.try_emplace(node.reg, registers.size());
        if (newReg) {
          registers.push_back(candidates[node.reg]);
          registers.back().pipeline.assign(
              registers.back().layer.size(), 0);
        }
        if (!llvm::is_contained(p.registers, ri->second))
          p.registers.push_back(ri->second);
        if (!llvm::is_contained(names, candidates[node.reg].reg.getName()))
          names.push_back(candidates[node.reg].reg.getName());
        registers[ri->second].layer[node.bit] = layer[n];
        registers[ri->second].pipeline[node.bit] = it->second;
        members.set(d.stateBase[candidates[node.reg].reg.getResult()] +
                    node.bit);
        p.layerBits[layer[n] - 1] += 1;
        if (layer[n] == 1)
          p.inputBits += 1;
        if (layer[n] == comp.layers)
          p.storeBits += 1;
        continue;
      }
      auto &box = allBoxes[node.box];
      box.inLayer = inLayer[n];
      box.outLayer = layer[n];
      boxIndex[node.box] = boxes.size();
      p.boxes.push_back(boxes.size());
      boxes.push_back(box);
      names.push_back(box.inst.getInstanceName());
      for (auto &out : box.contract.dataOut) {
        auto result = boxPort(box, out, false);
        auto base = d.stateBase[result];
        auto width = hw::getBitWidth(result.getType());
        for (int64_t k = 0; k < width; ++k)
          members.set(base + k);
        if (box.outLayer == comp.layers)
          p.storeBits += width;
      }
      if (box.inLayer == 0)
        for (auto &in : box.contract.dataIn)
          p.inputBits += hw::getBitWidth(boxPort(box, in, true).getType());
    }
    auto path = commonPath(names);
    if (!llvm::is_contained(p.paths, path))
      p.paths.push_back(path);
  }
  for (auto &[why, bits] : reasons)
    rejected.push_back({why, bits});
  return success();
}
