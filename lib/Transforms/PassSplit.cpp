//===- PassSplit.cpp - Lanes of a multicycle copy over passes -------------===//
//
// A static FPGA keeps the configuration of all its logic at once: P stages of
// different logic evaluated in turn cost the LUTs of all P. What P passes can
// save is the logic the passes share. Pass splitting therefore folds lanes: a
// copy's fanout-free cones (each value whose uses all lie inside the cone of
// its root, MFFC) of one structure -- operations, attributes, result and leaf
// types alike, leaves (captures, constants, logic outside the cone) taken by
// position -- evaluated on ceil(R/P) shared lanes, one replica a pass. A leaf
// position every replica of a lane shares is wired; one whose values differ
// is chosen by the pass. A replica's result is held from the end of its pass
// in a register that replaces it, so the logic downstream (and the end
// registers) read it unchanged once the passes are done. No lane may read
// what a held result reaches (it would read it before its pass): such groups
// are dropped. Within one evaluation the captures hold, so each pass computes
// exactly what the replica computed: the fold is cycle-exact by construction,
// provided the end registers are taken after the last pass (the caller's
// span), which the multicycle check verifies on every simulation.
//
//===----------------------------------------------------------------------===//

#include "PassSplit.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>

using namespace mlir;
using namespace circt;

namespace foldsim {
namespace {

constexpr unsigned kMaxCone = 4096;

int64_t bitsOf(Type type) { return std::max<int64_t>(1, hw::getBitWidth(type)); }

bool isConstant(Value value) {
  auto *op = value.getDefiningOp();
  return op && isa<hw::ConstantOp, hw::AggregateConstantOp>(op);
}

bool constantIndex(Value index) { return isConstant(index); }

/// LUT6s of one op, roughly (the pricer's foldsim_cost model is finer): what
/// decides whether a lane is worth its choices.
double lutsOf(Operation *op) {
  if (op->getNumResults() != 1)
    return 0;
  int64_t w = bitsOf(op->getResult(0).getType());
  unsigned live = 0;
  for (auto operand : op->getOperands())
    live += !isConstant(operand);
  if (isa<comb::AndOp, comb::OrOp, comb::XorOp>(op))
    return live < 2 ? 0 : double(w) * std::ceil((live - 1) / 5.0);
  if (auto mux = dyn_cast<comb::MuxOp>(op))
    return isConstant(mux.getCond()) ? 0 : 0.5 * double(w);
  if (auto cmp = dyn_cast<comb::ICmpOp>(op))
    return std::ceil(double(bitsOf(cmp.getLhs().getType())) / (live < 2 ? 6 : 3));
  if (isa<comb::AddOp, comb::SubOp>(op))
    return double(w);
  if (isa<comb::MulOp>(op))
    return live < 2 ? double(w) : (w <= 8 ? 0.5 * w * w : 0);
  if (isa<comb::DivUOp, comb::DivSOp, comb::ModUOp, comb::ModSOp>(op))
    return live < 2 ? 2.0 * w : double(w) * w;
  if (isa<comb::ShlOp, comb::ShrUOp, comb::ShrSOp>(op))
    return live < 2 ? 0 : double(w) * std::ceil(llvm::Log2_64_Ceil(w) / 2.0);
  if (isa<comb::ParityOp>(op))
    return std::ceil(w / 5.0);
  if (auto get = dyn_cast<hw::ArrayGetOp>(op)) {
    if (constantIndex(get.getIndex()))
      return 0;
    auto type = hw::type_cast<hw::ArrayType>(get.getInput().getType());
    return double(w) * (type.getNumElements() - 1) / 3.0;
  }
  if (auto inject = dyn_cast<hw::ArrayInjectOp>(op)) {
    if (constantIndex(inject.getIndex()))
      return 0;
    return 0.5 * double(w);
  }
  return 0;
}

bool isGather(Operation *op) {
  return isa<hw::ArrayCreateOp, comb::ConcatOp, hw::ArrayInjectOp,
             hw::StructCreateOp>(op);
}

/// One candidate replica: a root, its cone in post-order (operands first),
/// and its leaves by position.
struct Cone {
  Value root;
  SmallVector<Operation *> order;
  /// Per op of `order`, per operand: >= 0 the operand's op in `order`, < 0
  /// leaf position -1 - n.
  SmallVector<SmallVector<int>> refs;
  SmallVector<Value> leaves;
  double luts = 0;
};

std::string keyOf(Operation *op) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << op->getName().getStringRef();
  for (auto attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name == "sv.namehint" || name == "name" || name == "inner_sym" ||
        name == "sv.attributes")
      continue;
    os << ' ' << name << '=' << attr.getValue();
  }
  os << " :";
  for (auto type : op->getResultTypes())
    os << ' ' << type;
  return key;
}

} // namespace

PassSplitStats foldLanes(ImplicitLocOpBuilder &b, ArrayRef<Operation *> logicOps,
                         Value hostClock, Value capNow, unsigned passes,
                         unsigned span, StringRef prefix) {
  PassSplitStats stats;
  if (passes < 2 || span < 1)
    return stats;
  DenseSet<Operation *> logic;
  for (auto *op : logicOps)
    if (op->getNumResults() == 1 && op->getNumRegions() == 0 &&
        !isConstant(op->getResult(0)))
      logic.insert(op);
  DenseMap<Operation *, unsigned> uses;
  for (auto *op : logic) {
    unsigned n = 0;
    for (auto &use : op->getResult(0).getUses()) {
      (void)use;
      ++n;
    }
    uses[op] = n;
  }

  // Candidate roots: a fanout point, a value leaving the logic, or a part of
  // a gather (an array, concatenation or struct of the replicas).
  SmallVector<Operation *> roots;
  for (auto *op : logicOps) {
    if (!logic.contains(op))
      continue;
    Value v = op->getResult(0);
    bool candidate = uses[op] != 1;
    for (auto *user : v.getUsers())
      candidate |= !logic.contains(user) || isGather(user);
    if (candidate)
      roots.push_back(op);
  }

  auto coneOf = [&](Operation *rootOp, Cone &cone) -> bool {
    DenseSet<Operation *> in{rootOp};
    DenseMap<Operation *, unsigned> ref;
    SmallVector<Operation *> stack{rootOp};
    while (!stack.empty()) {
      auto *op = stack.pop_back_val();
      for (auto operand : op->getOperands()) {
        auto *def = operand.getDefiningOp();
        if (!def || !logic.contains(def) || in.contains(def))
          continue;
        auto it = ref.try_emplace(def, uses[def]).first;
        if (--it->second == 0) {
          in.insert(def);
          if (in.size() > kMaxCone)
            return false;
          stack.push_back(def);
        }
      }
    }
    // Post-order from the root, operands in order.
    DenseMap<Operation *, int> index;
    SmallVector<std::pair<Operation *, bool>> work{{rootOp, false}};
    while (!work.empty()) {
      auto [op, done] = work.pop_back_val();
      if (index.count(op))
        continue;
      if (!done) {
        work.push_back({op, true});
        for (auto operand : llvm::reverse(op->getOperands()))
          if (auto *def = operand.getDefiningOp();
              def && in.contains(def) && !index.count(def))
            work.push_back({def, false});
        continue;
      }
      SmallVector<int> r;
      for (auto operand : op->getOperands()) {
        auto *def = operand.getDefiningOp();
        if (def && in.contains(def)) {
          r.push_back(index[def]);
        } else {
          r.push_back(-1 - int(cone.leaves.size()));
          cone.leaves.push_back(operand);
        }
      }
      index[op] = int(cone.order.size());
      cone.order.push_back(op);
      cone.refs.push_back(std::move(r));
      cone.luts += lutsOf(op);
    }
    cone.root = rootOp->getResult(0);
    return true;
  };

  // Group the cones by structure (the key: each op's name, attributes and
  // result types, its operands' references, and the leaves' types).
  std::map<std::string, SmallVector<Cone>> byKey;
  for (auto *rootOp : roots) {
    Cone cone;
    if (!coneOf(rootOp, cone) || cone.order.size() < 2 || cone.luts <= 0)
      continue;
    std::string key;
    llvm::raw_string_ostream os(key);
    for (unsigned i = 0; i < cone.order.size(); ++i) {
      os << keyOf(cone.order[i]) << '(';
      for (int r : cone.refs[i])
        os << r << ',';
      os << ");";
    }
    for (auto leaf : cone.leaves)
      os << leaf.getType() << ';';
    byKey[key].push_back(std::move(cone));
  }

  // A group's saving at `passes`: the replicas' logic less the lanes' and
  // their choices (a P:1 choice of a bit ~ (P - 1) / 3 LUTs; constants, a
  // table of the pass).
  struct Group {
    SmallVector<Cone *> members;
    double saving = 0;
    int64_t selectBits = 0;
  };
  auto price = [&](Group &g) {
    unsigned r = g.members.size(), p = std::min(passes, r);
    unsigned lanes = (r + p - 1) / p;
    int64_t sel = 0, cst = 0;
    auto &leaves0 = g.members[0]->leaves;
    for (unsigned i = 0; i < leaves0.size(); ++i) {
      bool same = true, consts = true;
      for (auto *m : g.members) {
        same &= m->leaves[i] == leaves0[i];
        consts &= isConstant(m->leaves[i]);
      }
      if (same)
        continue;
      (consts ? cst : sel) += bitsOf(leaves0[i].getType());
    }
    g.selectBits = sel;
    g.saving = double(r - lanes) * g.members[0]->luts -
               double(lanes) * (double(sel) * (p - 1) / 3.0 + 0.25 * double(cst));
  };
  SmallVector<Group> candidates;
  for (auto &[key, cones] : byKey) {
    if (cones.size() < 2)
      continue;
    Group g;
    for (auto &c : cones)
      g.members.push_back(&c);
    double total = 0;
    for (auto *m : g.members)
      total += m->luts;
    g.saving = total;
    candidates.push_back(std::move(g));
  }
  llvm::stable_sort(candidates,
                    [](const Group &a, const Group &b) { return a.saving > b.saving; });
  DenseSet<Operation *> taken;
  SmallVector<Group> chosen;
  for (auto &g : candidates) {
    Group pick;
    DenseSet<Operation *> own;
    for (auto *m : g.members) {
      bool clash = false;
      for (auto *op : m->order)
        clash |= taken.contains(op) || own.contains(op);
      if (clash)
        continue;
      for (auto *op : m->order)
        own.insert(op);
      pick.members.push_back(m);
    }
    if (pick.members.size() < 2)
      continue;
    price(pick);
    if (pick.saving <= 0)
      continue;
    for (auto *op : own)
      taken.insert(op);
    chosen.push_back(std::move(pick));
  }
  // No lane may read (through its cone or its leaves) what a held result
  // reaches: drop such groups until none does.
  for (bool changed = true; changed;) {
    changed = false;
    DenseSet<Operation *> down;
    DenseSet<Value> rootSet;
    SmallVector<Operation *> stack;
    for (auto &g : chosen)
      for (auto *m : g.members) {
        rootSet.insert(m->root);
        for (auto *user : m->root.getUsers())
          if (logic.contains(user) && down.insert(user).second)
            stack.push_back(user);
      }
    while (!stack.empty()) {
      auto *op = stack.pop_back_val();
      for (auto *user : op->getResult(0).getUsers())
        if (logic.contains(user) && down.insert(user).second)
          stack.push_back(user);
    }
    SmallVector<Group> keep;
    for (auto &g : chosen) {
      bool bad = false;
      for (auto *m : g.members) {
        for (auto *op : m->order)
          bad |= down.contains(op);
        for (auto leaf : m->leaves)
          bad |= rootSet.contains(leaf) ||
                 (leaf.getDefiningOp() && down.contains(leaf.getDefiningOp()));
      }
      if (bad)
        changed = true;
      else
        keep.push_back(std::move(g));
    }
    chosen = std::move(keep);
  }
  if (chosen.empty())
    return stats;

  // The pass: `pass` counts the passes from the capture (0 in the host cycle
  // after capNow's), `sub` the host cycles of one; a pass ends (its holds
  // load) at its last host cycle. Both saturate after the last pass.
  auto i1 = b.getI1Type();
  Value one = hw::ConstantOp::create(b, i1, 1);
  BackedgeBuilder edges(b, b.getLoc());
  unsigned passWidth = llvm::Log2_32_Ceil(passes + 1);
  auto passType = b.getIntegerType(passWidth);
  auto passNext = edges.get(passType);
  Value pass = seq::CompRegClockEnabledOp::create(
      b, passType, passNext, hostClock, one, b.getStringAttr(prefix + "pass"),
      Value(), Value(), Value(), hw::InnerSymAttr{});
  Value passEnd = one;
  if (span > 1) {
    auto subType = b.getIntegerType(llvm::Log2_32_Ceil(span));
    auto subNext = edges.get(subType);
    Value sub = seq::CompRegClockEnabledOp::create(
        b, subType, subNext, hostClock, one, b.getStringAttr(prefix + "sub"),
        Value(), Value(), Value(), hw::InnerSymAttr{});
    passEnd = comb::ICmpOp::create(b, comb::ICmpPredicate::eq, sub,
                                   hw::ConstantOp::create(b, subType, span - 1));
    Value inc = comb::AddOp::create(b, sub, hw::ConstantOp::create(b, subType, 1));
    Value zero = hw::ConstantOp::create(b, subType, 0);
    subNext.setValue(comb::MuxOp::create(b, capNow, zero,
                                         comb::MuxOp::create(b, passEnd, zero, inc)));
  }
  {
    Value last = comb::ICmpOp::create(b, comb::ICmpPredicate::eq, pass,
                                      hw::ConstantOp::create(b, passType, passes));
    Value step = comb::AndOp::create(b, passEnd, comb::XorOp::create(b, last, one));
    Value inc = comb::AddOp::create(b, pass, hw::ConstantOp::create(b, passType, 1));
    passNext.setValue(comb::MuxOp::create(
        b, capNow, hw::ConstantOp::create(b, passType, 0),
        comb::MuxOp::create(b, step, inc, pass)));
  }
  // The choice index: the pass, over the next power of two of the passes.
  unsigned indexWidth = std::max(1u, llvm::Log2_32_Ceil(passes));
  unsigned slots = 1u << indexWidth;
  Value index = passWidth == indexWidth
                    ? pass
                    : comb::ExtractOp::create(b, b.getIntegerType(indexWidth), pass, 0).getResult();
  SmallVector<Value> loads;
  for (unsigned k = 0; k < passes; ++k)
    loads.push_back(comb::AndOp::create(
        b, passEnd,
        comb::ICmpOp::create(b, comb::ICmpPredicate::eq, pass,
                             hw::ConstantOp::create(b, passType, k))));

  unsigned holds = 0;
  for (auto &g : chosen) {
    unsigned r = g.members.size(), p = std::min(passes, r);
    stats.groups += 1;
    stats.replicas += r;
    stats.sizes.push_back(r);
    stats.before += g.members[0]->luts * r;
    stats.selectBits += g.selectBits;
    Cone &rep = *g.members[0];
    for (unsigned l = 0; l * p < r; ++l) {
      SmallVector<Cone *> lane;
      for (unsigned k = 0; k < p && l * p + k < r; ++k)
        lane.push_back(g.members[l * p + k]);
      stats.lanes += 1;
      stats.after += rep.luts;
      // The leaves: shared, or chosen by the pass.
      SmallVector<Value> leaf(rep.leaves.size());
      DenseMap<std::pair<unsigned, unsigned>, Value> chosenOf;
      for (unsigned i = 0; i < rep.leaves.size(); ++i) {
        bool same = true;
        for (auto *m : lane)
          same &= m->leaves[i] == lane[0]->leaves[i];
        if (same) {
          leaf[i] = lane[0]->leaves[i];
          continue;
        }
        // The same column of values (a leaf used at several positions) is
        // one choice.
        for (unsigned j = 0; j < i; ++j) {
          bool col = true;
          for (auto *m : lane)
            col &= m->leaves[j] == m->leaves[i];
          if (col) {
            leaf[i] = leaf[j];
            break;
          }
        }
        if (leaf[i])
          continue;
        // hw.array_create's first operand is the highest element.
        SmallVector<Value> elements;
        for (unsigned k = slots; k-- > 0;)
          elements.push_back(lane[std::min<unsigned>(k, lane.size() - 1)]->leaves[i]);
        leaf[i] = hw::ArrayGetOp::create(b, hw::ArrayCreateOp::create(b, elements),
                                         index);
        stats.after += double(bitsOf(rep.leaves[i].getType())) * (p - 1) / 3.0;
      }
      // The lane: the representative's cone, its leaves rewired.
      SmallVector<Value> cloned;
      for (unsigned j = 0; j < rep.order.size(); ++j) {
        Operation *copy = b.clone(*rep.order[j]);
        copy->removeAttr("sv.namehint");
        for (unsigned o = 0; o < copy->getNumOperands(); ++o) {
          int ref = rep.refs[j][o];
          copy->setOperand(o, ref >= 0 ? cloned[ref] : leaf[-1 - ref]);
        }
        cloned.push_back(copy->getResult(0));
      }
      Value out = cloned.back();
      Value bits = out;
      auto width = hw::getBitWidth(out.getType());
      if (!isa<IntegerType>(out.getType()))
        bits = b.createOrFold<hw::BitcastOp>(b.getIntegerType(width), out);
      for (unsigned k = 0; k < lane.size(); ++k) {
        Value hold = seq::CompRegClockEnabledOp::create(
            b, bits.getType(), bits, hostClock, loads[k],
            b.getStringAttr(prefix + "hold" + Twine(holds++)), Value(), Value(),
            Value(), hw::InnerSymAttr{});
        if (hold.getType() != out.getType())
          hold = b.createOrFold<hw::BitcastOp>(out.getType(), hold);
        lane[k]->root.replaceAllUsesWith(hold);
        stats.holdBits += width;
      }
    }
    // The replicas' cones are dead now.
    for (auto *m : g.members)
      for (auto *op : llvm::reverse(m->order))
        if (op->use_empty())
          op->erase();
  }
  return stats;
}

} // namespace foldsim
