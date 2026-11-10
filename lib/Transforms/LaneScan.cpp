//===- LaneScan.cpp - Fold an unfolded parent's per-child glue -----------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `foldsim-lane-scan`: the unfolded parent (the top, x1) of a class with many
// instances holds glue over them: per-child logic (decodes, masks, one-hot
// terms, per-child registers) and reductions across the children (arbiter
// selects, popcounts, AND/OR/XOR reduces, priority encoders, indexed selects).
// Spatially that logic, and the link registers that bring every child's
// outputs into the parent, grow with the number of children. This pass folds
// it: the children are *lanes*; the per-lane logic, found as packs of
// isomorphic per-lane values (SLP-style, including packed lane vectors: an
// N-bit vector whose bit j is lane j's), moves into a generated *lane cell*
// module with one instance per lane, a class like any other (one engine copy,
// folded); the reductions become zero-latency chains through the cell
// instances (scans), the parent reading only a chain's end. The parent keeps
// fixed-size logic and state.
//
// Exact by construction: every value is computed by the same operators on the
// same operands in the same target cycle; a chain evaluates an associative
// reduction (or a priority / indexed select) in lane order, which is the
// target's own result. Logic the pass does not recognize stays in the parent
// (reported as residual), reading the cell's per-lane values where needed.
//
// Recognized reductions (roots in the parent):
//   ac      comb.xor/or/and/add whose operands are per-lane values of one
//           pack (every lane once), plus parent values;
//   vec     icmp eq/ne of a packed lane vector against all-ones / zero, and
//           parity: AND / OR / XOR of the per-lane terms;
//   prio    mux chains `c_j ? v_j : rest` over lanes in lane order (first
//           wins) or reverse order (last wins), the chain's last `rest` the
//           parent's default;
//   select  hw.array_get of an array of per-lane values at a parent index;
//   bitsel  bit `s` of a packed lane vector of one bit a lane (shru +
//           extract 0);
//   carry   comb.add of a packed lane vector and a constant or another lane
//           vector of the same layout: the carry ripples through the lanes,
//           the per-lane sums staying lane values.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <climits>
#include <map>

namespace foldsim {
#define GEN_PASS_DEF_LANESCAN
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

enum class Kind : uint8_t { Scalar, Elem, Vec, Arr, Multi, Pfx };

/// What a value of the parent is, lane-wise.
struct Info {
  Kind kind = Kind::Scalar;
  /// Elem: the lane it depends on.
  int lane = -1;
  /// Vec: lane j's bits are [lo(j), lo(j) + w), lo(j) = base + j * stride,
  /// stride = +w or -w, the lanes covering the value's whole width.
  int64_t w = 0, base = 0, stride = 0;
  /// Arr: the lane of each element index (index 0 = the last operand).
  SmallVector<int, 0> arr;
  /// Vec: assembled from its lanes' bits (concat / OR of shifted pieces /
  /// extracts), each lane's slice gathered from its own values' bits.
  bool gather = false;

  static Info scalar() { return {}; }
  static Info multi() {
    Info i;
    i.kind = Kind::Multi;
    return i;
  }
  static Info elem(int lane) {
    Info i;
    i.kind = Kind::Elem;
    i.lane = lane;
    return i;
  }
  /// A stage of a prefix chain: a value over lanes 0..lane, computed at lane
  /// `lane` from that lane's values and the previous stage's.
  static Info pfx(int lane) {
    Info i;
    i.kind = Kind::Pfx;
    i.lane = lane;
    return i;
  }
  int64_t lo(int j) const { return base + j * stride; }
  bool operator==(const Info &o) const {
    return kind == o.kind && lane == o.lane && w == o.w && base == o.base &&
           stride == o.stride && arr == o.arr && gather == o.gather;
  }
  bool operator!=(const Info &o) const { return !(*this == o); }
};

static Info join(const Info &a, const Info &b) {
  if (a.kind == Kind::Scalar)
    return b;
  if (b.kind == Kind::Scalar)
    return a;
  if (a.kind == Kind::Vec && b.kind == Kind::Vec && a.w == b.w &&
      a.base == b.base && a.stride == b.stride) {
    Info r = a;
    r.gather = false; // an op over lane vectors: its slices are the op's
    return r;
  }
  if (a == b && a.kind != Kind::Arr)
    return a;
  return Info::multi();
}

enum class PatKind { AC, VecRed, Prio, Rot, Select, BitSel };

/// A reduction recognized at a root op of the parent.
struct Pattern {
  PatKind kind;
  Operation *root;
  // AC: the op's name; the packs whose members are chained; the operands
  // that stay in the parent.
  SmallVector<int> packs;
  SmallVector<Value> rest;
  // AC over the bits of lane vectors of one bit a lane (gathered whole).
  SmallVector<Value> vecs;
  // VecRed: the vector, and how to finish (and / or / xor, invert).
  Value vec;
  StringRef red;
  bool invert = false;
  // Prio: conditions and values per lane (null: lane not in the chain),
  // the default, and first-wins (levels in lane order) or last-wins.
  SmallVector<Value> conds, vals;
  Value terminal;
  bool firstWins = true;
  // Rot: a priority chain over the lanes in a rotated order: the condition
  // pack (array `elems` of the conditions' array), the free value p, the
  // start lane s(p) = (rotA + rotE * p) mod F, the direction (+1: s, s+1,
  // ...; -1: s, s-1, ...) and each lane's value (a constant: rotVals).
  int64_t rotA = 0, rotE = 1, rotDir = 1;
  SmallVector<APInt> rotVals;
  // Select / BitSel: per-lane elements (Select) or the vector (BitSel), the
  // index, and each lane's position in the index's terms.
  SmallVector<Value> elems;
  Value index;
  SmallVector<int64_t> pos;
};

struct LaneScanPass : public foldsim::impl::LaneScanBase<LaneScanPass> {
  using LaneScanBase::LaneScanBase;
  void runOnOperation() override;
};

/// One lane group of one parent module.
/// Where a bit of an assembled value comes from.
struct BitSrc {
  enum K : uint8_t { Zero, One, Scalar, Lane, Opaque } k = Zero;
  int lane = -1;
  Value src; // Scalar: the parent value; Lane: lane `lane`'s value (an Elem
             // value, or a lane vector whose lane slice it is); Opaque: a
             // value over several lanes that only wiring passes on
  unsigned bit = 0;
  bool operator==(const BitSrc &o) const {
    return k == o.k && lane == o.lane && src == o.src && bit == o.bit;
  }
};
/// A run of bits of one source, from the low bits up.
struct Piece {
  BitSrc::K k;
  Value src;
  unsigned bit, len;
};

struct Scan {
  hw::HWModuleOp parent;
  StringRef callee;
  SmallVector<hw::InstanceOp> lanes;
  unsigned F = 0;
  DenseMap<Operation *, int> laneOf;

  DenseMap<Value, Info> info;
  DenseSet<Value> red;       // roots whose value the parent computes as scalar
  DenseMap<Operation *, Pattern> patterns;
  // Packs: Elem values grouped by refined signature.
  DenseMap<Value, int> packOf;
  // Parent values made lane values because one lane alone reads them.
  DenseSet<Value> backward;
  std::vector<std::vector<Value>> packMembers; // pack -> lane -> member
  std::vector<bool> packComplete;
  // The first lane (stage) of a pack: 0 but for a prefix chain's packs,
  // whose first stages the parent computes (they read the start).
  std::vector<int> packFirst;
  Value tmpl(int p) { return packMembers[p][packFirst[p]]; }
  // Carry chains: comb.add ops whose result is a Vec.
  DenseSet<Operation *> carries;
  // Assembled values: the bits of values over several lanes built by
  // wiring ops (concat, extract, OR of disjoint pieces, constant shifts),
  // and the per-lane values made of their bits (Elem, as pieces).
  DenseMap<Value, std::vector<BitSrc>> bits, prevBits;
  bool bitsChanged = false;
  DenseMap<Value, SmallVector<Piece>> assembled;
  // A value that is another's bits passed through wiring (a tree node put in
  // a packed array and extracted back): that value.
  DenseMap<Value, Value> aliasOf;
  Value unalias(Value v) {
    for (int k = 0; k < 64; ++k) {
      auto it = aliasOf.find(v);
      if (it == aliasOf.end())
        break;
      v = it->second;
    }
    return v;
  }
  bool bitsOf(Value v, std::vector<BitSrc> &out);
  Info assemble(Operation *op, Value res);
  static SmallVector<Piece> pieces(ArrayRef<BitSrc> bs);

  const Info &get(Value v) {
    static const Info s;
    auto it = info.find(v);
    return it == info.end() ? s : it->second;
  }
  bool isElem(Value v) { return get(v).kind == Kind::Elem; }
  bool isPfx(Value v) { return get(v).kind == Kind::Pfx; }
  bool isLaneVal(Value v) { return isElem(v) || isPfx(v); }

  void classify();
  Info transfer(Operation *op, unsigned result);
  Info transferBase(Operation *op, unsigned result);
  void computePacks();
  bool matchPatterns();
  bool matchRotated(comb::MuxOp root);
  bool finishPrio(Pattern &p, ArrayRef<int> order);
  bool orLanes(Value v, SmallVectorImpl<std::pair<int, Value>> &out, int depth);
  bool treeOf(Value v, SmallVectorImpl<std::pair<int, Value>> &pinned,
              DenseMap<int, Value> &valid, Value &tail, int depth);
  bool matchTree(comb::MuxOp root, Pattern &p, SmallVectorImpl<int> &order);
  bool smallTree(Value v, SmallVectorImpl<std::pair<int, Value>> &pinned,
                 DenseMap<int, Value> &valid, Value &tail,
                 SmallVectorImpl<int> &lanes,
                 ArrayRef<std::pair<int, Value>> hint = {});
  /// A gather of one lane's bit or element: hw.array_get of a lane array,
  /// or bit `x` of a lane vector of one bit a lane (shru + extract 0).
  bool gather(Value v, Value &src, Value &index);
  /// The lane at index / bit `x` of a gather's source, -1 if none.
  int laneAt(Value src, uint64_t x);
  bool fullPack(ArrayRef<Value> values, int &pack);
};

static StringRef nameOf(Operation *op) {
  return op->getName().getStringRef();
}

/// The width of an integer value, 0 for others.
static int64_t widthOf(Value v) {
  if (auto t = dyn_cast<IntegerType>(v.getType()))
    return t.getWidth();
  return 0;
}

static bool isConstant(Value v, APInt &value) {
  if (auto c = v.getDefiningOp<hw::ConstantOp>()) {
    value = c.getValue();
    return true;
  }
  return false;
}

/// The value of `v` with `free` = `p`, constants elsewhere: comb logic only.
static bool evalAt(Value v, Value free, const APInt &p,
                   DenseMap<Value, APInt> &memo, APInt &out, int depth = 0) {
  if (v == free) {
    out = p;
    return true;
  }
  if (auto it = memo.find(v); it != memo.end()) {
    out = it->second;
    return true;
  }
  auto *op = v.getDefiningOp();
  if (!op || depth > 48)
    return false;
  if (auto c = dyn_cast<hw::ConstantOp>(op)) {
    out = c.getValue();
    return true;
  }
  if (!isa<comb::CombDialect>(op->getDialect()))
    return false;
  SmallVector<APInt> a;
  for (Value o : op->getOperands()) {
    APInt x;
    if (!evalAt(o, free, p, memo, x, depth + 1))
      return false;
    a.push_back(x);
  }
  unsigned w = widthOf(v);
  auto fold = [&](auto f) {
    APInt r = a[0];
    for (unsigned k = 1; k < a.size(); ++k)
      r = f(r, a[k]);
    return r;
  };
  if (isa<comb::AddOp>(op))
    out = fold([](APInt x, const APInt &y) { return x + y; });
  else if (isa<comb::MulOp>(op))
    out = fold([](APInt x, const APInt &y) { return x * y; });
  else if (isa<comb::AndOp>(op))
    out = fold([](APInt x, const APInt &y) { return x & y; });
  else if (isa<comb::OrOp>(op))
    out = fold([](APInt x, const APInt &y) { return x | y; });
  else if (isa<comb::XorOp>(op))
    out = fold([](APInt x, const APInt &y) { return x ^ y; });
  else if (isa<comb::SubOp>(op))
    out = a[0] - a[1];
  else if (isa<comb::DivUOp, comb::ModUOp, comb::DivSOp, comb::ModSOp>(op)) {
    if (a[1].isZero())
      return false;
    if (isa<comb::DivUOp>(op))
      out = a[0].udiv(a[1]);
    else if (isa<comb::ModUOp>(op))
      out = a[0].urem(a[1]);
    else if (isa<comb::DivSOp>(op))
      out = a[0].sdiv(a[1]);
    else
      out = a[0].srem(a[1]);
  } else if (isa<comb::ShlOp, comb::ShrUOp, comb::ShrSOp>(op)) {
    uint64_t sh = a[1].getLimitedValue(w);
    out = isa<comb::ShlOp>(op)    ? a[0].shl(sh)
          : isa<comb::ShrUOp>(op) ? a[0].lshr(sh)
                                  : a[0].ashr(sh);
  } else if (auto c = dyn_cast<comb::ConcatOp>(op)) {
    out = a[0];
    for (unsigned k = 1; k < a.size(); ++k)
      out = out.concat(a[k]);
  } else if (auto e = dyn_cast<comb::ExtractOp>(op)) {
    out = a[0].extractBits(w, e.getLowBit());
  } else if (isa<comb::ReplicateOp>(op)) {
    out = APInt(w, 0);
    unsigned iw = a[0].getBitWidth();
    for (unsigned k = 0; k < w / iw; ++k)
      out.insertBits(a[0], k * iw);
  } else if (isa<comb::MuxOp>(op)) {
    out = a[0].isOne() ? a[1] : a[2];
  } else if (auto i = dyn_cast<comb::ICmpOp>(op)) {
    bool r;
    switch (i.getPredicate()) {
    case comb::ICmpPredicate::eq: r = a[0] == a[1]; break;
    case comb::ICmpPredicate::ne: r = a[0] != a[1]; break;
    case comb::ICmpPredicate::ult: r = a[0].ult(a[1]); break;
    case comb::ICmpPredicate::ule: r = a[0].ule(a[1]); break;
    case comb::ICmpPredicate::ugt: r = a[0].ugt(a[1]); break;
    case comb::ICmpPredicate::uge: r = a[0].uge(a[1]); break;
    case comb::ICmpPredicate::slt: r = a[0].slt(a[1]); break;
    case comb::ICmpPredicate::sle: r = a[0].sle(a[1]); break;
    case comb::ICmpPredicate::sgt: r = a[0].sgt(a[1]); break;
    case comb::ICmpPredicate::sge: r = a[0].sge(a[1]); break;
    default: return false;
    }
    out = APInt(1, r);
  } else {
    return false;
  }
  memo[v] = out;
  return true;
}

/// The one value other than constants that `v`'s comb cone reads, if any
/// (null and true for constants alone).
static bool freeValue(Value v, Value &free, DenseSet<Value> &seen,
                      int depth = 0) {
  if (!seen.insert(v).second)
    return true;
  auto *op = v.getDefiningOp();
  if (op && isa<hw::ConstantOp>(op))
    return true;
  if (op && isa<comb::CombDialect>(op->getDialect()) && depth < 48) {
    for (Value o : op->getOperands())
      if (!freeValue(o, free, seen, depth + 1))
        return false;
    return true;
  }
  if (free && free != v)
    return false;
  free = v;
  return true;
}

bool Scan::bitsOf(Value v, std::vector<BitSrc> &out) {
  int64_t w = widthOf(v);
  if (w <= 0 || w > 16384)
    return false;
  out.clear();
  auto &i = get(v);
  APInt c;
  switch (i.kind) {
  case Kind::Scalar:
    if (isConstant(v, c)) {
      for (int64_t b = 0; b < w; ++b)
        out.push_back({c[b] ? BitSrc::One : BitSrc::Zero});
    } else {
      for (int64_t b = 0; b < w; ++b)
        out.push_back({BitSrc::Scalar, -1, v, unsigned(b)});
    }
    return true;
  case Kind::Elem: {
    // A lane's value built by wiring (a lane bit and constants placed by a
    // concat, say) gives its bits' own sources.
    if (auto it = bits.find(v); it != bits.end()) {
      out = it->second;
      return true;
    }
    auto *op = v.getDefiningOp();
    std::vector<BitSrc> tmp;
    if (op && isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp>(op)) {
      std::vector<std::vector<BitSrc>> in;
      bool ok = true;
      for (Value o : op->getOperands()) {
        in.emplace_back();
        ok &= bitsOf(o, in.back());
      }
      if (ok) {
        if (isa<comb::ConcatOp>(op))
          for (auto &x : llvm::reverse(in))
            tmp.insert(tmp.end(), x.begin(), x.end());
        else if (auto ex = dyn_cast<comb::ExtractOp>(op))
          tmp.assign(in[0].begin() + ex.getLowBit(),
                     in[0].begin() + ex.getLowBit() + w);
        else
          for (int64_t b = 0; b < w; ++b)
            tmp.push_back(in[0][b % in[0].size()]);
      }
    }
    if (int64_t(tmp.size()) != w) {
      tmp.clear();
      for (int64_t b = 0; b < w; ++b)
        tmp.push_back({BitSrc::Lane, i.lane, v, unsigned(b)});
    }
    bits[v] = tmp;
    out = std::move(tmp);
    return true;
  }
  case Kind::Pfx:
    for (int64_t b = 0; b < w; ++b)
      out.push_back({BitSrc::Lane, i.lane, v, unsigned(b)});
    return true;
  case Kind::Vec:
    out.resize(w);
    for (unsigned j = 0; j < F; ++j)
      for (int64_t b = 0; b < i.w; ++b)
        out[i.lo(j) + b] = {BitSrc::Lane, int(j), v, unsigned(b)};
    return true;
  default: {
    auto it = bits.find(v);
    if (it != bits.end()) {
      out = it->second;
      return true;
    }
    // Not yet this sweep (the netlist is not in order): last sweep's.
    if (auto pt = prevBits.find(v); pt != prevBits.end()) {
      out = pt->second;
      return true;
    }
    if (i.kind != Kind::Multi)
      return false;
    for (int64_t b = 0; b < w; ++b)
      out.push_back({BitSrc::Opaque, -1, v, unsigned(b)});
    return true;
  }
  }
}

SmallVector<Piece> Scan::pieces(ArrayRef<BitSrc> bs) {
  SmallVector<Piece> ps;
  for (auto &b : bs) {
    if (!ps.empty()) {
      auto &p = ps.back();
      if (p.k == b.k &&
          (b.k == BitSrc::Zero || b.k == BitSrc::One ||
           (p.src == b.src && p.bit + p.len == b.bit))) {
        ++p.len;
        continue;
      }
    }
    ps.push_back({b.k, b.src, b.bit, 1});
  }
  return ps;
}

Info Scan::assemble(Operation *op, Value res) {
  int64_t w = widthOf(res);
  if (w <= 0 || w > 16384)
    return Info::multi();
  std::vector<BitSrc> out;
  std::vector<std::vector<BitSrc>> in;
  for (Value o : op->getOperands()) {
    in.emplace_back();
    if (!bitsOf(o, in.back()))
      return Info::multi();
  }
  BitSrc zero{BitSrc::Zero}, one{BitSrc::One};
  if (isa<comb::ConcatOp>(op)) {
    for (auto &v : llvm::reverse(in))
      out.insert(out.end(), v.begin(), v.end());
  } else if (auto ex = dyn_cast<comb::ExtractOp>(op)) {
    out.assign(in[0].begin() + ex.getLowBit(),
               in[0].begin() + ex.getLowBit() + w);
  } else if (isa<comb::ReplicateOp>(op)) {
    for (int64_t k = 0; k < w; ++k)
      out.push_back(in[0][k % in[0].size()]);
  } else if (isa<comb::ShlOp, comb::ShrUOp>(op)) {
    APInt sh;
    if (!isConstant(op->getOperand(1), sh))
      return Info::multi();
    int64_t n = sh.getLimitedValue(w);
    out.assign(w, zero);
    for (int64_t b = 0; b < w; ++b) {
      int64_t from = isa<comb::ShlOp>(op) ? b - n : b + n;
      if (from >= 0 && from < w)
        out[b] = in[0][from];
    }
  } else if (auto mux = dyn_cast<comb::MuxOp>(op)) {
    if (get(mux.getCond()).kind != Kind::Scalar)
      return Info::multi();
    for (int64_t b = 0; b < w; ++b) {
      if (!(in[1][b] == in[2][b]))
        return Info::multi();
      out.push_back(in[1][b]);
    }
  } else {
    // and / or / xor of disjoint pieces: every bit one operand's, the others'
    // that bit the op's identity (or, for AND, all ones).
    bool isAnd = isa<comb::AndOp>(op), isOr = isa<comb::OrOp>(op);
    for (int64_t b = 0; b < w; ++b) {
      BitSrc got = isAnd ? one : zero;
      bool have = false, dead = false;
      for (auto &v : in) {
        auto &x = v[b];
        if (isAnd && x.k == BitSrc::Zero)
          dead = true;
        if (isOr && x.k == BitSrc::One)
          got = one, have = true, dead = true;
        if (dead)
          break;
        if ((isAnd && x.k == BitSrc::One) || (!isAnd && x.k == BitSrc::Zero))
          continue;
        if (have)
          return Info::multi();
        got = x;
        have = true;
      }
      if (isAnd && dead)
        got = zero;
      out.push_back(got);
    }
  }
  // Another value's bits, whole and in place: that value.
  if (!out.empty() && out[0].k == BitSrc::Opaque && out[0].bit == 0 &&
      widthOf(out[0].src) == w) {
    bool same = true;
    for (int64_t b = 0; b < w && same; ++b)
      same = out[b].k == BitSrc::Opaque && out[b].src == out[0].src &&
             out[b].bit == unsigned(b);
    if (same) {
      aliasOf[res] = out[0].src;
      bits[res] = std::move(out);
      return Info::multi();
    }
  }
  // One lane alone: a value of that lane, made of its bits.
  int lane = -2;
  bool scalarOnly = true, opaque = false;
  for (auto &b : out) {
    if (b.k == BitSrc::Opaque)
      opaque = true;
    if (b.k == BitSrc::Lane) {
      scalarOnly = false;
      lane = lane == -2 ? b.lane : (lane == b.lane ? lane : -1);
    }
  }
  if (opaque) {
    bits[res] = std::move(out);
    return Info::multi();
  }
  if (scalarOnly)
    return Info::multi(); // constants: the canonicalizer's, not ours
  if (lane >= 0) {
    assembled[res] = pieces(out);
    return Info::elem(lane);
  }
  // A full lane vector: slices of one width in lane order (up or down), each
  // bit its slice's lane's or a constant (a lane the RTL leaves out).
  int64_t sw = w / int64_t(F);
  Info v;
  v.kind = Kind::Vec;
  v.w = sw;
  v.gather = true;
  bool full = false;
  if (sw > 0 && sw * int64_t(F) == w) {
    for (int dir : {1, -1}) {
      v.stride = dir * sw;
      v.base = dir > 0 ? 0 : w - sw;
      bool ok = true, any = false;
      for (int64_t b = 0; b < w && ok; ++b) {
        auto &x = out[b];
        if (x.k == BitSrc::Scalar)
          ok = false;
        if (x.k != BitSrc::Lane)
          continue;
        any = true;
        int64_t owner = dir > 0 ? b / sw : (w - 1 - b) / sw;
        ok = x.lane == owner;
      }
      if (ok && any) {
        full = true;
        break;
      }
    }
  }
  bits[res] = std::move(out);
  return full ? v : Info::multi();
}

Info Scan::transfer(Operation *op, unsigned result) {
  Info r = transferBase(op, result);
  Value res = op->getResult(result);
  bool wiring = isa<comb::ConcatOp, comb::ExtractOp, comb::AndOp, comb::OrOp,
                    comb::XorOp, comb::ReplicateOp, comb::ShlOp, comb::ShrUOp,
                    comb::MuxOp>(op);
  if (r.kind == Kind::Multi && wiring)
    return assemble(op, res);
  // Two neighbors' pieces put side by side (an OR of disjoint shifted
  // pieces) are wiring, not a prefix stage.
  if (r.kind == Kind::Pfx && wiring) {
    Info a = assemble(op, res);
    if (a.kind != Kind::Multi)
      return a;
    auto it = bits.find(res);
    if (it != bits.end() &&
        llvm::none_of(it->second,
                      [](const BitSrc &b) { return b.k == BitSrc::Opaque; }))
      return a;
    bits.erase(res);
  }
  return r;
}

Info Scan::transferBase(Operation *op, unsigned result) {
  Value res = op->getResult(result);
  if (red.count(res))
    return Info::scalar();
  if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
    auto it = laneOf.find(op);
    return it == laneOf.end() ? Info::scalar() : Info::elem(it->second);
  }
  if (isa<hw::ConstantOp>(op))
    return Info::scalar();
  auto operands = op->getOperands();
  bool anyLane = false;
  for (Value v : operands)
    if (get(v).kind != Kind::Scalar)
      anyLane = true;
  if (!anyLane)
    return Info::scalar();

  if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
    // Segments from the low bits up.
    int64_t at = 0, w = -1;
    SmallVector<std::pair<int, int64_t>> lanesAt; // (lane, offset)
    int vecOperands = 0;
    Info vec;
    int64_t vecAt = 0;
    int single = -2;
    for (Value v : llvm::reverse(concat.getInputs())) {
      auto &i = get(v);
      int64_t width = widthOf(v);
      switch (i.kind) {
      case Kind::Scalar:
        break;
      case Kind::Elem:
        if (w >= 0 && w != width)
          w = -2;
        else if (w == -1)
          w = width;
        lanesAt.push_back({i.lane, at});
        single = single == -2 ? i.lane : (single == i.lane ? single : -1);
        break;
      case Kind::Vec:
        ++vecOperands;
        vec = i;
        vecAt = at;
        break;
      default:
        return Info::multi();
      }
      at += width;
    }
    if (vecOperands == 0 && lanesAt.size() == 1)
      return Info::elem(lanesAt[0].first);
    if (vecOperands == 0 && single >= 0)
      return Info::elem(single); // one lane's pieces and parent bits
    if (vecOperands)
      return Info::multi(); // a vector with padding: not handled
    if (w <= 0 || lanesAt.size() != F || at != int64_t(F) * w)
      return Info::multi();
    // Lanes at base + j * stride, every lane once.
    SmallVector<int64_t> offset(F, -1);
    for (auto [lane, off] : lanesAt) {
      if (offset[lane] != -1)
        return Info::multi();
      offset[lane] = off;
    }
    Info v;
    v.kind = Kind::Vec;
    v.w = w;
    v.stride = F > 1 ? offset[1] - offset[0] : w;
    v.base = offset[0];
    if (v.stride != w && v.stride != -w)
      return Info::multi();
    for (unsigned j = 0; j < F; ++j)
      if (offset[j] != v.lo(j))
        return Info::multi();
    return v;
  }

  if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
    auto &i = get(extract.getInput());
    if (i.kind == Kind::Elem)
      return i;
    if (i.kind != Kind::Vec)
      return Info::multi();
    int64_t lo = extract.getLowBit(), n = widthOf(extract.getResult());
    for (unsigned j = 0; j < F; ++j)
      if (lo >= i.lo(j) && lo + n <= i.lo(j) + i.w)
        return Info::elem(j);
    return Info::multi();
  }

  if (auto create = dyn_cast<hw::ArrayCreateOp>(op)) {
    auto inputs = create.getInputs();
    if (inputs.size() != F)
      return Info::multi();
    Info a;
    a.kind = Kind::Arr;
    a.arr.assign(F, -1);
    SmallVector<bool> seen(F, false);
    for (auto [k, v] : llvm::enumerate(inputs)) {
      auto &i = get(v);
      if (i.kind != Kind::Elem || seen[i.lane])
        return Info::multi();
      seen[i.lane] = true;
      a.arr[F - 1 - k] = i.lane;
    }
    return a;
  }
  if (auto aget = dyn_cast<hw::ArrayGetOp>(op)) {
    auto &a = get(aget.getInput());
    APInt idx;
    if (a.kind == Kind::Arr && isConstant(aget.getIndex(), idx) &&
        idx.getZExtValue() < a.arr.size())
      return Info::elem(a.arr[idx.getZExtValue()]);
    return Info::multi();
  }

  // A prefix chain's stage: comb logic over lane j's values and stage j - 1's
  // (or lane j - 1's), and the parent's.
  if (isa<comb::CombDialect>(op->getDialect()) &&
      !isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp>(op)) {
    int hi = -1, lo = INT_MAX;
    bool lanesOnly = true, anyPfx = false;
    for (Value v : operands) {
      auto &i = get(v);
      if (i.kind == Kind::Scalar)
        continue;
      if (i.kind != Kind::Elem && i.kind != Kind::Pfx) {
        lanesOnly = false;
        break;
      }
      anyPfx |= i.kind == Kind::Pfx;
      hi = std::max(hi, i.lane);
      lo = std::min(lo, i.lane);
    }
    if (lanesOnly && hi >= 0 && hi - lo == 1)
      return Info::pfx(hi);
    if (lanesOnly && hi >= 0 && hi == lo && anyPfx)
      return Info::pfx(hi);
  }
  // Ops whose packed lane vectors stay lane vectors.
  bool bitwise = isa<comb::AndOp, comb::OrOp, comb::XorOp, seq::FirRegOp,
                     seq::CompRegOp>(op);
  if (auto mux = dyn_cast<comb::MuxOp>(op))
    bitwise = get(mux.getCond()).kind == Kind::Scalar;
  if (carries.count(op))
    bitwise = true;
  Info r;
  for (Value v : operands) {
    auto &i = get(v);
    if ((i.kind == Kind::Vec && !bitwise) || i.kind == Kind::Arr)
      return Info::multi();
    r = join(r, i);
  }
  if (r.kind == Kind::Vec && widthOf(res) != r.w * F)
    r = Info::multi();
  // A register whose next value the parent computes from several lanes is
  // the parent's state: one register, fixed during the cycle.
  if (isa<seq::FirRegOp, seq::CompRegOp>(op) && r.kind == Kind::Multi)
    return Info::scalar();
  return r;
}

void Scan::classify() {
  auto *body = parent.getBodyBlock();
  bits.clear();
  prevBits.clear();
  assembled.clear();
  aliasOf.clear();
  // A carry chain: comb.add of exactly two operands, a lane vector and a
  // constant or a lane vector of the same layout.
  int iterations = 0;
  for (bool changed = true; changed && iterations < 256; ++iterations) {
    changed = false;
    // The bits of assembled values follow their operands': anew each sweep,
    // last sweep's for values not reached yet; a change sweeps again.
    prevBits = std::move(bits);
    bits.clear();
    assembled.clear();
    aliasOf.clear();
    bitsChanged = false;
    for (auto &op : *body) {
      if (isa<hw::OutputOp>(op))
        continue;
      if (auto add = dyn_cast<comb::AddOp>(op)) {
        bool carry = false;
        if (add.getInputs().size() == 2) {
          auto &a = get(add.getInputs()[0]), &b = get(add.getInputs()[1]);
          APInt c;
          if (a.kind == Kind::Vec &&
              ((b.kind == Kind::Vec && a == b) ||
               (b.kind == Kind::Scalar && isConstant(add.getInputs()[1], c))))
            carry = true;
          if (b.kind == Kind::Vec && a.kind == Kind::Scalar &&
              isConstant(add.getInputs()[0], c))
            carry = true;
        }
        if (carry != carries.count(&op)) {
          if (carry)
            carries.insert(&op);
          else
            carries.erase(&op);
          changed = true;
        }
      }
      for (unsigned r = 0, e = op.getNumResults(); r < e; ++r) {
        Info i = transfer(&op, r);
        Value v = op.getResult(r);
        auto it = info.find(v);
        if (it == info.end()) {
          if (i.kind != Kind::Scalar) {
            info[v] = i;
            changed = true;
          }
          continue;
        }
        if (it->second != i) {
          it->second = i;
          changed = true;
        }
      }
    }
    if (!changed && iterations < 255) {
      // Settled kinds: settled bits too?
      for (auto &[v, b] : bits) {
        auto pt = prevBits.find(v);
        if (pt == prevBits.end() || !(pt->second == b)) {
          changed = true;
          break;
        }
      }
    }
  }
  // Parent values read by one lane alone (a decode of a parent index for
  // lane j's input, a per-lane mask bit): lane j's, computed in its cell
  // from the parent's values.
  DenseMap<Value, int> only;
  auto sole = [&](Value v) -> int {
    int lane = -2;
    for (auto &use : v.getUses()) {
      Operation *u = use.getOwner();
      int l = -1;
      if (auto it = laneOf.find(u); it != laneOf.end())
        l = it->second;
      else if (u->getNumResults() == 1 && isElem(u->getResult(0)))
        l = get(u->getResult(0)).lane;
      else if (u->getNumResults() == 1)
        if (auto it = only.find(u->getResult(0)); it != only.end())
          l = it->second;
      if (l < 0 || (lane >= 0 && l != lane))
        return -1;
      lane = l;
    }
    return lane;
  };
  for (bool changed = true; changed;) {
    changed = false;
    for (auto &op : llvm::reverse(*body)) {
      if (op.getNumResults() != 1 || isa<hw::ConstantOp, hw::InstanceOp>(op) ||
          op.getNumRegions() || !isMemoryEffectFree(&op))
        continue;
      Value v = op.getResult(0);
      if (get(v).kind != Kind::Scalar || red.count(v))
        continue;
      int l = sole(v);
      auto it = only.find(v);
      if (l >= 0 && (it == only.end() || it->second != l)) {
        only[v] = l;
        changed = true;
      } else if (l < 0 && it != only.end()) {
        only.erase(it);
        changed = true;
      }
    }
  }
  backward.clear();
  for (auto [v, l] : only) {
    info[v] = Info::elem(l);
    backward.insert(v);
  }
}

/// A value's operand descriptor for the local signature.
static void describe(Scan &s, Value v, raw_ostream &os, int at = -1) {
  auto &i = s.get(v);
  switch (i.kind) {
  case Kind::Elem:
  case Kind::Pfx:
    // A lane value: by its lane relative to the reader's (a prefix stage
    // reads the previous lane's), whether a lane's own or a stage's.
    os << "L" << (at >= 0 ? i.lane - at : 0);
    return;
  case Kind::Vec:
    os << "V" << v.getAsOpaquePointer();
    return;
  case Kind::Scalar:
    break;
  default:
    os << "M" << v.getAsOpaquePointer();
    return;
  }
  if (v.getDefiningOp<hw::ConstantOp>()) {
    os << "C" << v.getType();
    return;
  }
  if (auto ex = v.getDefiningOp<comb::ExtractOp>())
    if (s.get(ex.getInput()).kind == Kind::Scalar) {
      os << "X" << ex.getInput().getAsOpaquePointer() << v.getType();
      return;
    }
  if (auto ac = v.getDefiningOp<hw::ArrayGetOp>())
    if (s.get(ac.getInput()).kind == Kind::Scalar &&
        ac.getIndex().getDefiningOp<hw::ConstantOp>()) {
      os << "A" << ac.getInput().getAsOpaquePointer() << v.getType();
      return;
    }
  os << "S" << v.getAsOpaquePointer();
}

static bool skipAttr(StringRef name) {
  return name == "name" || name == "sv.namehint" || name == "inner_sym" ||
         name == "sv.attributes";
}

void Scan::computePacks() {
  packOf.clear();
  packMembers.clear();
  packComplete.clear();
  packFirst.clear();
  // The Elem values and their local signatures.
  SmallVector<Value> elems;
  DenseMap<Value, std::string> local;
  for (auto &op : *parent.getBodyBlock())
    for (auto [r, v] : llvm::enumerate(op.getResults())) {
      if (!isLaneVal(v))
        continue;
      std::string sig;
      llvm::raw_string_ostream os(sig);
      if (isa<hw::InstanceOp>(op)) {
        os << "port:" << callee << ":" << r;
      } else if (auto it = assembled.find(v); it != assembled.end()) {
        os << "asm:" << v.getType() << ":";
        for (auto &pc : it->second) {
          os << int(pc.k) << "," << pc.bit << "," << pc.len;
          if (pc.k == BitSrc::Lane)
            os << (isElem(pc.src) ? std::string("E")
                                  : "V" + std::to_string(uintptr_t(
                                              pc.src.getAsOpaquePointer())));
          if (pc.k == BitSrc::Scalar)
            os << "S" << pc.src.getAsOpaquePointer();
          os << ";";
        }
      } else if (auto ex = dyn_cast<comb::ExtractOp>(op);
                 ex && get(ex.getInput()).kind == Kind::Vec) {
        auto &vi = get(ex.getInput());
        int j = get(v).lane;
        os << "slice:" << ex.getInput().getAsOpaquePointer() << ":"
           << (ex.getLowBit() - vi.lo(j)) << ":" << v.getType();
      } else {
        os << nameOf(&op) << ":" << r << ":";
        for (auto attr : op.getAttrs())
          if (!skipAttr(attr.getName()))
            os << attr.getName() << "=" << attr.getValue() << ";";
        for (auto t : op.getResultTypes())
          os << t << ",";
        os << "(";
        for (Value o : op.getOperands()) {
          describe(*this, o, os, get(v).lane);
          os << ",";
        }
        os << ")";
      }
      local[v] = sig;
      elems.push_back(v);
    }
  // Refine by the operands' packs until the partition is stable.
  DenseMap<Value, int> pid;
  unsigned count = 0;
  for (int iter = 0; iter < 64; ++iter) {
    std::map<std::string, int> ids;
    DenseMap<Value, int> next;
    for (Value v : elems) {
      std::string sig = local[v];
      if (iter > 0)
        if (auto *op = v.getDefiningOp(); op && !isa<hw::InstanceOp>(op)) {
          bool slice = false;
          if (auto ex = dyn_cast<comb::ExtractOp>(op))
            slice = get(ex.getInput()).kind == Kind::Vec;
          if (auto it = assembled.find(v); it != assembled.end()) {
            sig += "|";
            for (auto &pc : it->second)
              if (pc.k == BitSrc::Lane && isElem(pc.src))
                sig += std::to_string(pid.lookup(pc.src)) + ",";
          } else if (!slice) {
            sig += "|";
            for (Value o : op->getOperands()) {
              // A stage's read of the previous stage: by what it is, not by
              // its own pack (which a chain's start makes differ stage by
              // stage).
              if (isLaneVal(o) && get(o).lane == get(v).lane - 1)
                sig += "L" + local[o] + ",";
              else
                sig += isLaneVal(o) ? std::to_string(pid.lookup(o)) + ","
                                    : "-,";
            }
          }
        }
      auto [it, _] = ids.try_emplace(sig, ids.size());
      next[v] = it->second;
    }
    pid = std::move(next);
    if (ids.size() == count)
      break;
    count = ids.size();
  }
  packMembers.assign(count, std::vector<Value>(F));
  packComplete.assign(count, true);
  packFirst.assign(count, 0);
  std::vector<bool> isP(count, false);
  for (Value v : elems) {
    int p = pid[v];
    packOf[v] = p;
    isP[p] = isP[p] || isPfx(v);
    auto &slot = packMembers[p][get(v).lane];
    if (slot)
      packComplete[p] = false; // two members in one lane
    slot = v;
  }
  // A prefix chain's pack starts at its first stage (the parent computes
  // the ones before, the chain's start); from there every stage once.
  for (unsigned p = 0; p < count; ++p) {
    unsigned first = 0;
    if (isP[p])
      while (first < F && !packMembers[p][first])
        ++first;
    packFirst[p] = first;
    // A stage pack may also end early (the last stage's value unread, so
    // gone): its cells compute it anyway, unused.
    unsigned last = F;
    if (isP[p])
      while (last > first && !packMembers[p][last - 1])
        --last;
    if (first >= F || (isP[p] && (first > 2 || last - first < 2)))
      packComplete[p] = false;
    for (unsigned j = first; j < last; ++j)
      if (!packMembers[p][j])
        packComplete[p] = false;
  }
}

/// Whether `values` are the members of one complete pack, every lane once.
bool Scan::fullPack(ArrayRef<Value> values, int &pack) {
  if (values.size() != F)
    return false;
  pack = -1;
  SmallVector<bool> seen(F, false);
  for (Value v : values) {
    if (!isElem(v))
      return false;
    int p = packOf.lookup(v);
    if (pack >= 0 && p != pack)
      return false;
    pack = p;
    int j = get(v).lane;
    if (seen[j])
      return false;
    seen[j] = true;
  }
  return pack >= 0 && packComplete[pack];
}

bool Scan::matchPatterns() {
  bool any = false;
  auto *body = parent.getBodyBlock();
  auto add = [&](Pattern p) {
    Value v = p.root->getResult(0);
    patterns[p.root] = std::move(p);
    red.insert(v);
    any = true;
  };
  // Rotated chains first: the gathers they read are theirs.
  DenseSet<Value> rotGathers;
  for (auto &op : *body) {
    auto mux = dyn_cast<comb::MuxOp>(op);
    if (!mux || patterns.count(&op) || get(mux.getResult()).kind != Kind::Multi)
      continue;
    bool inner = llvm::any_of(mux.getResult().getUsers(), [&](Operation *u) {
      auto m = dyn_cast<comb::MuxOp>(u);
      return m && m.getFalseValue() == mux.getResult();
    });
    if (!inner && matchRotated(mux))
      any = true;
  }
  for (auto &[root, p] : patterns)
    if (p.kind == PatKind::Rot)
      for (Value cur = root->getResult(0); auto m = cur.getDefiningOp<comb::MuxOp>();
           cur = m.getFalseValue()) {
        if (cur == p.terminal)
          break;
        rotGathers.insert(m.getCond());
      }
  for (auto &op : *body) {
    if (op.getNumResults() != 1 || patterns.count(&op))
      continue;
    Value res = op.getResult(0);
    if (rotGathers.count(res))
      continue;
    if (get(res).kind != Kind::Multi)
      continue;
    // ac
    if (isa<comb::XorOp, comb::OrOp, comb::AndOp, comb::AddOp>(op) &&
        !carries.count(&op)) {
      llvm::MapVector<int, SmallVector<Value>> byPack;
      SmallVector<Value> rest;
      bool ok = true;
      // Gathers of one lane array at parent indices: when for every value
      // of the one free index variable they read each lane once, they are
      // the array's lanes (the op is commutative).
      llvm::MapVector<Value, SmallVector<Value>> gathers;
      SmallVector<Value> vecTerms;
      // A tree of the same bitwise op over several lanes: its leaves.
      SmallVector<Value> leaves;
      std::function<void(Value, int)> flatten = [&](Value v, int depth) {
        if (depth)
          v = unalias(v);
        auto *d = v.getDefiningOp();
        if (depth < 64 && d && d->getName() == op.getName() &&
            !isa<comb::AddOp>(op) &&
            (get(v).kind == Kind::Multi || get(v).kind == Kind::Pfx) &&
            !patterns.count(d)) {
          for (Value o : d->getOperands())
            flatten(o, depth + 1);
          return;
        }
        leaves.push_back(v);
      };
      for (Value v : op.getOperands())
        flatten(v, 0);
      for (Value v : leaves) {
        auto &i = get(v);
        if (i.kind == Kind::Elem)
          byPack[packOf.lookup(v)].push_back(v);
        else if (i.kind == Kind::Scalar)
          rest.push_back(v);
        else if (Value src, x; gather(v, src, x))
          gathers[src].push_back(x);
        else
          ok = false;
      }
      for (auto &[arr, idxs] : gathers) {
        if (!ok || idxs.size() != F) {
          ok = false;
          break;
        }
        Value free;
        DenseSet<Value> seen;
        for (Value x : idxs)
          ok &= freeValue(x, free, seen);
        if (!ok || !free || widthOf(free) == 0 || widthOf(free) > 12) {
          ok = false;
          break;
        }
        unsigned pw = widthOf(free);
        for (uint64_t pv = 0; ok && pv < (uint64_t(1) << pw); ++pv) {
          DenseMap<Value, APInt> memo;
          SmallVector<bool> hit(F, false);
          for (Value x : idxs) {
            APInt xv;
            int l = -1;
            if (evalAt(x, free, APInt(pw, pv), memo, xv) &&
                xv.getActiveBits() <= 32)
              l = laneAt(arr, xv.getZExtValue());
            if (l < 0 || hit[l]) {
              ok = false;
              break;
            }
            hit[l] = true;
          }
        }
        if (!ok)
          break;
        if (auto create = arr.getDefiningOp<hw::ArrayCreateOp>()) {
          for (Value m : create.getInputs())
            byPack[packOf.lookup(m)].push_back(m);
        } else {
          vecTerms.push_back(arr);
        }
      }
      Pattern p{PatKind::AC, &op};
      for (auto &[pk, vs] : byPack) {
        int pack;
        if (!fullPack(vs, pack))
          ok = false;
        p.packs.push_back(pk);
      }
      if (ok && (!p.packs.empty() || !vecTerms.empty())) {
        p.rest = rest;
        p.vecs = vecTerms;
        add(std::move(p));
      }
      continue;
    }
    // vec reductions
    if (auto icmp = dyn_cast<comb::ICmpOp>(op)) {
      Value a = icmp.getLhs(), b = icmp.getRhs();
      if (get(b).kind == Kind::Vec)
        std::swap(a, b);
      APInt c;
      if (get(a).kind == Kind::Vec && isConstant(b, c) &&
          (c.isZero() || c.isAllOnes())) {
        auto pred = icmp.getPredicate();
        Pattern p{PatKind::VecRed, &op};
        p.vec = a;
        if (pred == comb::ICmpPredicate::eq && c.isAllOnes())
          p.red = "and";
        else if (pred == comb::ICmpPredicate::ne && c.isAllOnes())
          p.red = "and", p.invert = true;
        else if (pred == comb::ICmpPredicate::ne && c.isZero())
          p.red = "or";
        else if (pred == comb::ICmpPredicate::eq && c.isZero())
          p.red = "or", p.invert = true;
        if (!p.red.empty())
          add(std::move(p));
      }
      continue;
    }
    if (auto parity = dyn_cast<comb::ParityOp>(op)) {
      if (get(parity.getInput()).kind == Kind::Vec) {
        Pattern p{PatKind::VecRed, &op};
        p.vec = parity.getInput();
        p.red = "xor";
        add(std::move(p));
      }
      continue;
    }
    // select
    if (auto aget = dyn_cast<hw::ArrayGetOp>(op)) {
      auto &a = get(aget.getInput());
      if (a.kind != Kind::Arr || get(aget.getIndex()).kind != Kind::Scalar)
        continue;

      auto create = aget.getInput().getDefiningOp<hw::ArrayCreateOp>();
      if (!create)
        continue;
      Pattern p{PatKind::Select, &op};
      p.elems.assign(F, Value());
      p.pos.assign(F, 0);
      SmallVector<Value> vs;
      for (auto [k, v] : llvm::enumerate(create.getInputs())) {
        int idx = F - 1 - k;
        p.elems[a.arr[idx]] = v;
        p.pos[a.arr[idx]] = idx;
        vs.push_back(v);
      }
      int pack;
      if (!fullPack(vs, pack))
        continue;
      p.index = aget.getIndex();
      add(std::move(p));
      continue;
    }
    // bitsel: extract(shru(V, s), 0, 1), V one bit a lane
    if (auto ex = dyn_cast<comb::ExtractOp>(op)) {
      auto shr = ex.getInput().getDefiningOp<comb::ShrUOp>();
      if (!shr || ex.getLowBit() != 0 || widthOf(ex.getResult()) != 1)
        continue;
      auto &v = get(shr.getLhs());
      if (v.kind != Kind::Vec || v.w != 1 ||
          get(shr.getRhs()).kind != Kind::Scalar)
        continue;
      Pattern p{PatKind::BitSel, &op};
      p.vec = shr.getLhs();
      p.index = shr.getRhs();
      for (unsigned j = 0; j < F; ++j)
        p.pos.push_back(v.lo(j));
      add(std::move(p));
      continue;
    }
    // prio: a mux chain, from its outermost mux
    if (auto mux = dyn_cast<comb::MuxOp>(op)) {
      bool inner = false;
      for (auto *user : res.getUsers())
        if (auto m = dyn_cast<comb::MuxOp>(user))
          if (m.getFalseValue() == res && isElem(m.getCond()))
            inner = true;
      if (inner)
        continue;
      Pattern p{PatKind::Prio, &op};
      p.conds.assign(F, Value());
      p.vals.assign(F, Value());
      SmallVector<int> order;
      Value cur = res;
      while (auto m = cur.getDefiningOp<comb::MuxOp>()) {
        auto &c = get(m.getCond());
        auto &t = get(m.getTrueValue());
        if (c.kind != Kind::Elem || p.conds[c.lane])
          break;
        if (t.kind == Kind::Elem ? t.lane != c.lane : t.kind != Kind::Scalar)
          break;
        if (m != mux && !m.getResult().hasOneUse())
          break;
        p.conds[c.lane] = m.getCond();
        p.vals[c.lane] = m.getTrueValue();
        order.push_back(c.lane);
        cur = m.getFalseValue();
      }
      if (order.size() < 2) {
        // A priority tree (find-first): a mux of the left range's OR'ed
        // valids picking the left range's first valid or the right range's.
        Pattern q{PatKind::Prio, &op};
        SmallVector<int> qorder;
        if (matchTree(mux, q, qorder) && finishPrio(q, qorder))
          add(std::move(q));
        continue;
      }
      p.terminal = cur;
      if (!finishPrio(p, order))
        continue;
      add(std::move(p));
      continue;
    }
  }
  return any;
}

bool Scan::finishPrio(Pattern &p, ArrayRef<int> order) {
  {
    {
      bool up = true, down = true;
      for (unsigned k = 1; k < order.size(); ++k) {
        up &= order[k] > order[k - 1];
        down &= order[k] < order[k - 1];
      }
      if (!up && !down)
        return false;
      p.firstWins = up;
      // The conditions: one pack, every lane present (lanes outside the
      // chain take theirs with a lane enable of 0). Values: one pack, or
      // parent values.
      int pc = -1, pv = -1;
      bool ok = true, scalarVals = false, elemVals = false;
      for (int j : order) {
        int c = packOf.lookup(p.conds[j]);
        ok &= pc < 0 || c == pc;
        pc = c;
        if (isElem(p.vals[j]) && !backward.count(p.vals[j])) {
          int v = packOf.lookup(p.vals[j]);
          ok &= pv < 0 || v == pv;
          pv = v;
          elemVals = true;
        } else {
          scalarVals = true;
        }
      }
      ok &= !(scalarVals && elemVals) && pc >= 0 && packComplete[pc] &&
            (pv < 0 || packComplete[pv]);
      return ok;
    }
  }
}

/// The lanes whose one-bit values an OR tree reads (each lane once).
bool Scan::orLanes(Value v, SmallVectorImpl<std::pair<int, Value>> &out,
                   int depth) {
  v = unalias(v);
  APInt c;
  if (isConstant(v, c) && c.isZero())
    return true;
  if (isElem(v) && widthOf(v) == 1) {
    int j = get(v).lane;
    for (auto &[l, x] : out)
      if (l == j)
        return false;
    out.push_back({j, v});
    return true;
  }
  auto o = v.getDefiningOp<comb::OrOp>();
  if (!o || depth > 64 ||
      (get(v).kind != Kind::Multi && get(v).kind != Kind::Pfx))
    return false;
  for (Value x : o.getInputs())
    if (!orLanes(x, out, depth + 1))
      return false;
  return true;
}

/// A find-first tree at `v`: the lanes it orders (pinned, each with its
/// valid and value) and its last leaf's value (`tail`, the default).
bool Scan::treeOf(Value v, SmallVectorImpl<std::pair<int, Value>> &pinned,
                  DenseMap<int, Value> &valid, Value &tail, int depth) {
  v = unalias(v);
  auto m = v.getDefiningOp<comb::MuxOp>();
  if (!m || depth > 64 || get(v).kind == Kind::Scalar)
    return false;
  SmallVector<std::pair<int, Value>> cl;
  if (!orLanes(m.getCond(), cl, 0) || cl.empty())
    return false;
  for (auto &[l, x] : cl) {
    auto it = valid.find(l);
    if (it != valid.end() && it->second != x)
      return false;
    valid[l] = x;
  }
  // The left range: a tree whose lanes are the condition's, or one lane's
  // value.
  SmallVector<std::pair<int, Value>> lp;
  Value lt;
  if (treeOf(m.getTrueValue(), lp, valid, lt, depth + 1)) {
    if (lt) {
      if (lp.size() + 1 != cl.size())
        return false;
      int missing = -1;
      for (auto &[l, x] : cl)
        if (llvm::none_of(lp, [&](auto &e) { return e.first == l; }))
          missing = l;
      if (missing < 0)
        return false;
      auto &ti = get(lt);
      if (ti.kind == Kind::Elem ? ti.lane != missing : ti.kind != Kind::Scalar)
        return false;
      lp.push_back({missing, lt});
    } else if (lp.size() != cl.size()) {
      return false;
    }
  } else if (cl.size() > 1) {
    // A collapsed small tree over the condition's lanes.
    SmallVector<int> sl;
    lp.clear();
    if (!smallTree(m.getTrueValue(), lp, valid, lt, sl, cl) ||
        sl.size() != cl.size())
      return false;
    for (auto &[l, x] : cl)
      if (!llvm::is_contained(sl, l))
        return false;
    if (lp.size() + 1 == cl.size())
      lp.push_back({sl.back(), lt});
  } else {
    auto &ti = get(m.getTrueValue());
    if (ti.kind == Kind::Elem ? ti.lane != cl[0].first
                              : ti.kind != Kind::Scalar)
      return false;
    lp.push_back({cl[0].first, m.getTrueValue()});
  }
  for (auto &[l, x] : cl) {
    auto it = valid.find(l);
    if (it != valid.end() && it->second != x)
      return false;
    valid[l] = x;
  }
  // The right range: a tree, or its last leaf.
  SmallVector<std::pair<int, Value>> rp;
  Value rt;
  if (!treeOf(m.getFalseValue(), rp, valid, rt, depth + 1)) {
    rp.clear();
    SmallVector<int> sl;
    if (!smallTree(m.getFalseValue(), rp, valid, rt, sl)) {
      rp.clear();
      rt = m.getFalseValue();
    }
  }
  pinned.append(lp.begin(), lp.end());
  pinned.append(rp.begin(), rp.end());
  tail = rt;
  return true;
}

/// A small find-first the canonicalizer collapsed (a pair of leaves with
/// constant values becomes a concat of a NOT, say): a value of a few lanes'
/// one-bit values and constants. Its truth table, over every choice of
/// those bits, must be a priority select: lane order by pairs, each lane's
/// value when it wins, and a default when none is set.
bool Scan::smallTree(Value v, SmallVectorImpl<std::pair<int, Value>> &pinned,
                     DenseMap<int, Value> &valid, Value &tail,
                     SmallVectorImpl<int> &lanes,
                     ArrayRef<std::pair<int, Value>> hint) {
  v = unalias(v);
  if (get(v).kind == Kind::Scalar)
    return false;
  // The cone's leaves: lanes' one-bit values and constants.
  SmallVector<std::pair<int, Value>> leaves;
  DenseSet<Value> seen;
  bool ok = true;
  std::function<void(Value, int)> walk = [&](Value x, int depth) {
    x = unalias(x);
    if (!ok || !seen.insert(x).second)
      return;
    if (x.getDefiningOp<hw::ConstantOp>())
      return;
    // A lane's one-bit value: a known valid (the conditions'), or one that
    // logic in the lane does not make (an instance's output, a register).
    bool known = llvm::any_of(hint, [&](auto &e) { return e.second == x; });
    for (auto &[l, y] : valid)
      known |= y == x;
    auto *xd = x.getDefiningOp();
    bool made = xd && isa<comb::CombDialect>(xd->getDialect());
    if (isElem(x) && widthOf(x) == 1 && (known || !made)) {
      int j = get(x).lane;
      for (auto &[l, y] : leaves)
        if (l == j && y != x)
          ok = false;
      leaves.push_back({j, x});
      return;
    }
    auto *d = x.getDefiningOp();
    if (!d || depth > 32 || !isa<comb::CombDialect>(d->getDialect()) ||
        get(x).kind == Kind::Scalar) {
      ok = false;
      return;
    }
    for (Value o : d->getOperands())
      walk(o, depth + 1);
  };
  walk(v, 0);
  // The condition's lanes the value does not read (a lane whose own value
  // equals the default) are in its range too.
  for (auto &[l, x] : hint)
    if (llvm::none_of(leaves, [&](auto &e) { return e.first == l; }))
      leaves.push_back({l, x});
  if (!ok || leaves.empty() || leaves.size() > 6 ||
      (!hint.empty() && leaves.size() != hint.size()))
    return false;
  unsigned k = leaves.size(), rw = widthOf(v);
  if (rw == 0 || rw > 64)
    return false;
  auto eval = [&](unsigned mask, APInt &out) {
    DenseMap<Value, APInt> memo;
    for (unsigned i = 0; i < k; ++i)
      memo[leaves[i].second] = APInt(1, (mask >> i) & 1);
    return evalAt(v, Value(), APInt(), memo, out);
  };
  SmallVector<APInt> val(k);
  APInt def;
  if (!eval(0, def))
    return false;
  for (unsigned i = 0; i < k; ++i)
    if (!eval(1u << i, val[i]))
      return false;
  // The order: i before j when both set picks i's.
  SmallVector<unsigned> ord;
  for (unsigned i = 0; i < k; ++i)
    ord.push_back(i);
  bool consistent = true;
  auto before = [&](unsigned i, unsigned j) {
    APInt w;
    if (!eval((1u << i) | (1u << j), w)) {
      consistent = false;
      return false;
    }
    if (val[i] == val[j]) // ambiguous: keep the lanes' order
      return leaves[i].first < leaves[j].first;
    if (w == val[i])
      return true;
    if (w != val[j])
      consistent = false;
    return false;
  };
  llvm::stable_sort(ord, before);
  if (!consistent)
    return false;
  // Check every choice against first-set-in-order, else the default.
  for (unsigned mask = 0; mask < (1u << k); ++mask) {
    APInt got, want = def;
    if (!eval(mask, got))
      return false;
    for (unsigned i : ord)
      if (mask >> i & 1) {
        want = val[i];
        break;
      }
    if (got != want)
      return false;
  }
  for (auto &[l, x] : leaves) {
    auto it = valid.find(l);
    if (it != valid.end() && it->second != x)
      return false;
    valid[l] = x;
  }
  OpBuilder cb(parent.getBodyBlock(), parent.getBodyBlock()->begin());
  auto cst = [&](const APInt &c) -> Value {
    return hw::ConstantOp::create(cb, parent.getLoc(), c);
  };
  // The default is the last lane's value (find-first's rightmost leaf): the
  // last lane is the tail; otherwise every lane is ordered and the default
  // is the tail.
  unsigned n = ord.size();
  if (n == 1 && def == val[0])
    return false; // a constant, not a select
  if (n > 1 && def == val[ord[n - 1]]) {
    for (unsigned i = 0; i + 1 < n; ++i)
      pinned.push_back({leaves[ord[i]].first, cst(val[ord[i]])});
    tail = cst(def);
  } else {
    for (unsigned i = 0; i < n; ++i)
      pinned.push_back({leaves[ord[i]].first, cst(val[ord[i]])});
    tail = cst(def);
  }
  for (unsigned i : ord)
    lanes.push_back(leaves[i].first);
  return true;
}

bool Scan::matchTree(comb::MuxOp root, Pattern &p, SmallVectorImpl<int> &order) {
  // Only from the root: not a subtree of another tree.
  for (auto *u : root.getResult().getUsers())
    if (auto m = dyn_cast<comb::MuxOp>(u))
      if (m.getCond() != root.getResult() &&
          get(m.getCond()).kind != Kind::Scalar)
        return false;
  SmallVector<std::pair<int, Value>> pinned;
  DenseMap<int, Value> valid;
  Value tail;
  if (!treeOf(root.getResult(), pinned, valid, tail, 0) || pinned.size() < 2)
    return false;
  p.conds.assign(F, Value());
  p.vals.assign(F, Value());
  for (auto &[l, x] : pinned) {
    if (p.conds[l])
      return false;
    p.conds[l] = valid.lookup(l);
    p.vals[l] = x;
    order.push_back(l);
    if (!p.conds[l])
      return false;
  }
  p.terminal = tail;
  return true;
}

bool Scan::gather(Value v, Value &src, Value &index) {
  if (auto g = v.getDefiningOp<hw::ArrayGetOp>()) {
    if (get(g.getInput()).kind != Kind::Arr ||
        get(g.getIndex()).kind != Kind::Scalar)
      return false;
    src = g.getInput();
    index = g.getIndex();
    return true;
  }
  if (auto ex = v.getDefiningOp<comb::ExtractOp>()) {
    auto shr = ex.getInput().getDefiningOp<comb::ShrUOp>();
    if (!shr || ex.getLowBit() != 0 || widthOf(ex.getResult()) != 1)
      return false;
    auto &vi = get(shr.getLhs());
    if (vi.kind != Kind::Vec || vi.w != 1 ||
        get(shr.getRhs()).kind != Kind::Scalar)
      return false;
    src = shr.getLhs();
    index = shr.getRhs();
    return true;
  }
  return false;
}

int Scan::laneAt(Value src, uint64_t x) {
  auto &i = get(src);
  if (i.kind == Kind::Arr)
    return x < i.arr.size() ? i.arr[x] : -1;
  for (unsigned j = 0; j < F; ++j)
    if (uint64_t(i.lo(j)) == x)
      return j;
  return -1;
}

/// A priority mux chain whose level i reads `A[x_i(p)]` and yields v_i(p),
/// A an array of one pack (a lane each), p one parent value: when the lanes
/// the levels visit are, for every p, a rotation of the lanes from s(p) =
/// (a + e p) mod F (up or down), and v_i(p) a function of the lane alone,
/// the chain is a two-segment scan of the lanes.
bool Scan::matchRotated(comb::MuxOp root) {
  SmallVector<Value> xs, vs;
  Value arr, cur = root.getResult();
  while (auto m = cur.getDefiningOp<comb::MuxOp>()) {
    Value src, x;
    if (!gather(m.getCond(), src, x) || (arr && src != arr))
      break;
    if (get(m.getTrueValue()).kind != Kind::Scalar)
      break;
    if (m != root && !m.getResult().hasOneUse())
      break;
    arr = src;
    xs.push_back(x);
    vs.push_back(m.getTrueValue());
    cur = m.getFalseValue();
  }
  if (xs.size() != F || F < 2)
    return false;
  SmallVector<Value> members;
  if (auto create = arr.getDefiningOp<hw::ArrayCreateOp>()) {
    members.assign(create.getInputs().begin(), create.getInputs().end());
    int pack;
    if (!fullPack(members, pack))
      return false;
  } else if (get(arr).kind != Kind::Vec) {
    return false;
  }
  // The one free value.
  Value free;
  DenseSet<Value> seen;
  for (unsigned i = 0; i < F; ++i)
    if (!freeValue(xs[i], free, seen) || !freeValue(vs[i], free, seen))
      return false;
  if (!free || get(free).kind != Kind::Scalar || widthOf(free) == 0 ||
      widthOf(free) > 12)
    return false;
  unsigned pw = widthOf(free), rw = widthOf(vs[0]);
  int64_t dir = 0, e = 0, a0 = 0;
  SmallVector<std::optional<APInt>> V(F);
  for (uint64_t pv = 0; pv < (uint64_t(1) << pw); ++pv) {
    DenseMap<Value, APInt> memo;
    APInt p(pw, pv);
    SmallVector<int64_t> lane(F);
    for (unsigned i = 0; i < F; ++i) {
      APInt x, v;
      if (!evalAt(xs[i], free, p, memo, x) || !evalAt(vs[i], free, p, memo, v))
        return false;
      if (x.getActiveBits() > 32)
        return false;
      lane[i] = laneAt(arr, x.getZExtValue());
      if (lane[i] < 0)
        return false;
      if (V[lane[i]] && *V[lane[i]] != v)
        return false;
      V[lane[i]] = v;
    }
    int64_t d = (lane[1] - lane[0] + F) % F == 1 ? 1 : -1;
    if (dir && d != dir)
      return false;
    dir = d;
    for (unsigned i = 0; i < F; ++i)
      if (lane[i] != ((lane[0] + dir * int64_t(i)) % int64_t(F) + F) % F)
        return false;
    // s(p) = (a + e p) mod F
    if (pv == 0)
      a0 = lane[0];
    else if (pv == 1)
      e = (lane[0] - a0 + F) % F == 1 ? 1 : -1;
    int64_t want = ((a0 + e * int64_t(pv)) % int64_t(F) + F) % F;
    if (pv > 0 && lane[0] != want)
      return false;
  }
  if (pw == 1 && F > 2)
    return false; // too few points to know the rotation's slope
  Pattern p{PatKind::Rot, root};
  p.index = free;
  p.terminal = cur;
  p.rotA = a0;
  p.rotE = e;
  p.rotDir = dir;
  p.elems = members;
  p.vec = members.empty() ? arr : Value();
  for (auto &v : V)
    p.rotVals.push_back(v ? *v : APInt(rw, 0));
  patterns[root] = std::move(p);
  red.insert(root.getResult());
  return true;
}

void LaneScanPass::runOnOperation() {
  auto top = getOperation();
  auto parent = top.lookupSymbol<hw::HWModuleOp>(module);
  if (!parent) {
    top.emitError("lane-scan: no module ") << module;
    return signalPassFailure();
  }
  // Lane groups: the instances of one module, at least min-lanes of them.
  llvm::MapVector<StringRef, SmallVector<hw::InstanceOp>> groups;
  for (auto inst : parent.getBodyBlock()->getOps<hw::InstanceOp>())
    groups[inst.getModuleName()].push_back(inst);
  DenseSet<StringRef> wanted;
  for (auto &l : lanes)
    wanted.insert(l);
  auto *ctx = &getContext();
  for (auto &[callee, insts] : groups) {
    if (insts.size() < minLanes || (!wanted.empty() && !wanted.count(callee)))
      continue;
    Scan s;
    s.parent = parent;
    s.callee = callee;
    s.lanes = insts;
    s.F = insts.size();
    for (auto [j, inst] : llvm::enumerate(insts))
      s.laneOf[inst] = j;
    // Rounds: classify, group into packs, recognize reductions (whose
    // results are parent values from then on), until nothing new.
    for (int round = 0; round < 32; ++round) {
      s.info.clear();
      s.classify();
      s.computePacks();
      if (!s.matchPatterns())
        break;
    }
    s.info.clear();
    s.classify();
    s.computePacks();
    if (getenv("FOLDSIM_LANE_DEBUG"))
      for (auto &op : *parent.getBodyBlock()) {
        if (op.getNumResults() != 1 || isa<hw::InstanceOp>(op) ||
            (s.get(op.getResult(0)).kind != Kind::Multi &&
             !(getenv("FOLDSIM_LANE_DEBUG")[0] == '2' &&
               s.isLaneVal(op.getResult(0)))))
          continue;
        if (s.isLaneVal(op.getResult(0))) {
          Value v = op.getResult(0);
          int pk = s.packOf.lookup(v);
          llvm::errs() << "lane-debug " << (s.isPfx(v) ? "P" : "E")
                       << s.get(v).lane << " pack " << pk << " first "
                       << s.packFirst[pk] << " complete "
                       << s.packComplete[pk] << " : " << op << "\n";
          continue;
        }
        llvm::errs() << "lane-debug " << callee << " " << nameOf(&op) << " i"
                     << widthOf(op.getResult(0)) << " (";
        for (Value o : op.getOperands()) {
          auto &i = s.get(o);
          llvm::errs() << "SEVAM"[int(i.kind)];
          if (i.kind == Kind::Elem)
            llvm::errs() << i.lane;
          if (auto *d = o.getDefiningOp())
            llvm::errs() << ":" << nameOf(d);
          llvm::errs() << ":i" << widthOf(o) << " ";
        }
        llvm::errs() << ")\n";
      }

    // ------------------------------------------------------------------
    // What moves into the cell: the complete packs and the lane vectors
    // whose sources move too.
    auto *body = parent.getBodyBlock();
    unsigned F = s.F;
    std::vector<bool> cellPack(s.packMembers.size());
    for (unsigned p = 0; p < cellPack.size(); ++p)
      cellPack[p] = s.packComplete[p];
    DenseSet<Value> cellVec;
    for (auto &op : *body)
      for (Value v : op.getResults())
        if (s.get(v).kind == Kind::Vec)
          cellVec.insert(v);
    auto valueIn = [&](Value v) {
      auto &i = s.get(v);
      if (i.kind == Kind::Elem || i.kind == Kind::Pfx)
        return bool(cellPack[s.packOf.lookup(v)]);
      if (i.kind == Kind::Vec)
        return cellVec.count(v) != 0;
      return i.kind == Kind::Scalar;
    };
    for (bool changed = true; changed;) {
      changed = false;
      for (auto &op : *body) {
        for (Value v : op.getResults()) {
          auto &i = s.get(v);
          bool lv = i.kind == Kind::Elem || i.kind == Kind::Pfx;
          bool in = lv ? bool(cellPack[s.packOf.lookup(v)])
                       : cellVec.count(v) != 0;
          if (!in || isa<hw::InstanceOp>(op))
            continue;
          bool ok;
          if (auto it = s.assembled.find(v); it != s.assembled.end()) {
            // A lane's value of its own bits: its sources move.
            ok = llvm::all_of(it->second, [&](const Piece &pc) {
              return pc.k != BitSrc::Lane || valueIn(pc.src);
            });
          } else if (i.kind == Kind::Vec && i.gather) {
            // Each lane's slice gathered from the same bits of the same pack
            // (or lane vector) as lane 0's.
            auto &bs = s.bits[v];
            ok = true;
            for (int64_t k = 0; k < i.w && ok; ++k) {
              // The reference: the first lane whose bit k is a lane bit.
              const BitSrc *a = nullptr;
              for (unsigned j = 0; j < F && !a; ++j)
                if (bs[i.lo(j) + k].k == BitSrc::Lane)
                  a = &bs[i.lo(j) + k];
              for (unsigned j = 0; j < F && ok && a; ++j) {
                auto &x = bs[i.lo(j) + k];
                if (x.k != BitSrc::Lane)
                  continue; // a constant in this lane
                ok = x.bit == a->bit;
                if (ok && s.isElem(x.src))
                  ok = s.isElem(a->src) &&
                       s.packOf.lookup(x.src) == s.packOf.lookup(a->src) &&
                       valueIn(x.src);
                else if (ok)
                  ok = x.src == a->src && valueIn(x.src);
              }
            }
          } else {
            // (The previous lane's values a stage reads: below.)
            ok = llvm::all_of(op.getOperands(), [&](Value o) {
              return (s.isLaneVal(o) && s.get(o).lane == i.lane - 1) ||
                     valueIn(o);
            });
            // A stage reads packs that start no later than its own; what it
            // reads of the previous stage, the cell makes one stage on.
            int pk = s.packOf.lookup(v);
            for (bool retry = true; ok && retry;) {
              retry = false;
              int first = s.packFirst[pk];
              if (i.lane < first)
                break; // not a member the cell makes
              for (auto [k, o] : llvm::enumerate(op.getOperands())) {
                if (!ok || !s.isLaneVal(o))
                  continue;
                bool late = false;
                if (s.get(o).lane == i.lane - 1) {
                  Value next = first + 1 < int(F) ? s.packMembers[pk][first + 1]
                                                  : Value();
                  if (!next) {
                    ok = false;
                    continue;
                  }
                  Value role = next.getDefiningOp()->getOperand(k);
                  if (!s.isLaneVal(role) || !valueIn(role))
                    late = true;
                  else if (s.packFirst[s.packOf.lookup(role)] > first)
                    late = true;
                } else if (s.packFirst[s.packOf.lookup(o)] > first) {
                  late = true;
                }
                // A stage that reads a pack starting later starts later
                // itself (the parent computes its first stages).
                if (late) {
                  int last = F;
                  while (last > first && !s.packMembers[pk][last - 1])
                    --last;
                  if (s.isPfx(s.tmpl(pk)) && first + 3 <= last) {
                    s.packFirst[pk] = first + 1;
                    changed = retry = true;
                  } else {
                    ok = false;
                  }
                  break;
                }
              }
            }
          }
          if (isa<comb::ConcatOp>(op) && i.kind == Kind::Vec && !i.gather) {
            // The lanes' pieces: one pack.
            SmallVector<Value> parts;
            for (Value o : op.getOperands())
              parts.push_back(o);
            int pack;
            ok &= s.fullPack(parts, pack);
          }
          if (auto reg = dyn_cast<seq::FirRegOp>(op))
            if (reg.getPresetAttr() && i.kind == Kind::Vec)
              ok = false;
          if (!ok) {
            if (lv)
              cellPack[s.packOf.lookup(v)] = false;
            else
              cellVec.erase(v);
            changed = true;
          }
        }
      }
    }
    // Patterns whose inputs move.
    SmallVector<Pattern *> pats;
    for (auto &op : *body) {
      auto it = s.patterns.find(&op);
      if (it == s.patterns.end())
        continue;
      auto &p = it->second;
      bool ok = true;
      switch (p.kind) {
      case PatKind::AC:
        for (int pk : p.packs)
          ok &= cellPack[pk];
        for (Value v : p.vecs)
          ok &= cellVec.count(v) != 0;
        break;
      case PatKind::VecRed:
      case PatKind::BitSel:
        ok = cellVec.count(p.vec) != 0;
        break;
      case PatKind::Select:
        ok = cellPack[s.packOf.lookup(p.elems[0])];
        break;
      case PatKind::Rot:
        ok = p.vec ? cellVec.count(p.vec) != 0
                   : bool(cellPack[s.packOf.lookup(p.elems[0])]);
        break;
      case PatKind::Prio:
        for (unsigned j = 0; j < F; ++j)
          if (p.conds[j]) {
            ok &= valueIn(p.conds[j]) &&
                  (s.backward.count(p.vals[j]) || valueIn(p.vals[j]));
          }
        break;
      }
      if (ok)
        pats.push_back(&p);
    }

    // Statistics before building.
    unsigned movedPacks = 0, movedOps = 0, movedVecs = 0;
    for (unsigned p = 0; p < cellPack.size(); ++p)
      if (cellPack[p]) {
        ++movedPacks;
        if (!isa<hw::InstanceOp>(s.tmpl(p).getDefiningOp()))
          movedOps += F;
      }
    movedVecs = cellVec.size();
    std::map<std::string, unsigned> kinds;
    for (auto *p : pats)
      kinds[p->kind == PatKind::AC       ? "ac"
            : p->kind == PatKind::VecRed ? "vec"
            : p->kind == PatKind::Prio   ? "prio"
            : p->kind == PatKind::Select ? "select"
            : p->kind == PatKind::Rot    ? "rot"
                                         : "bitsel"]++;
    for (auto &op : *body)
      if (s.carries.count(&op) && cellVec.count(op.getResult(0)))
        kinds["carry"]++;
    for (unsigned p = 0; p < cellPack.size(); ++p)
      if (cellPack[p] && s.isPfx(s.tmpl(p)))
        kinds["stage"]++;
    // Nothing to scan: the per-lane logic stays where it is (a cell would
    // only hand it back to the parent or the lanes).
    bool stages = false;
    for (unsigned p = 0; p < cellPack.size(); ++p)
      if (cellPack[p] && s.isPfx(s.tmpl(p)))
        stages = true;
    if (pats.empty() && !stages)
      continue;

    DenseSet<Operation *> before;
    for (auto &op : *body)
      before.insert(&op);

    // ------------------------------------------------------------------
    // The cell module.
    std::string cellName = (module + "_lane_" + callee).str();
    SmallVector<hw::PortInfo> ports;
    unsigned numIn = 0, numOut = 0;
    auto addPort = [&](const Twine &name, Type type, bool input) {
      hw::PortInfo port;
      port.name = StringAttr::get(ctx, name);
      port.type = type;
      port.dir = input ? hw::ModulePort::Direction::Input
                       : hw::ModulePort::Direction::Output;
      port.argNum = input ? numIn++ : numOut++;
      ports.push_back(port);
    };
    // Inputs, decided as the body is built: kind and per-lane parent values.
    struct In {
      std::string name;
      Type type;
      // Per lane (Lane / Param) or one (Broadcast) parent value; for Chain,
      // the chain index.
      SmallVector<Value> values;
      int chain = -1;
    };
    struct Out {
      std::string name;
      Value cell;
      // Per lane: the parent's value it replaces (Export), or chain index.
      SmallVector<Value> replaces;
      int chain = -1;
      // A value made later than the output is declared: its parent value,
      // looked up when the outputs are made.
      Value key = {};
    };
    struct Chain {
      Type type;
      Value init;      // the parent's value into the first lane
      bool down = false; // lane F-1 first
      int in = -1, out = -1;
      int start = 0;     // the lane that takes `init` (lanes before ignore)
    };
    SmallVector<In> ins;
    SmallVector<Out> outs;
    SmallVector<Chain> chains;

    // Build the body in a temporary module first, then fix the ports: the
    // body's block arguments are added as inputs are found.
    OpBuilder ob(parent);
    auto cell = hw::HWModuleOp::create(ob, parent.getLoc(),
                                       StringAttr::get(ctx, cellName),
                                       hw::ModulePortInfo({}));
    auto *cbody = cell.getBodyBlock();
    cbody->getTerminator()->erase();
    auto b = ImplicitLocOpBuilder::atBlockEnd(parent.getLoc(), cbody);
    BackedgeBuilder backedges(b, parent.getLoc());
    DenseMap<Value, Value> cellOf;           // lane-0 member / vec -> cell
    DenseMap<Value, Backedge> pending;
    DenseMap<Value, Value> broadcast;        // parent value -> cell input
    auto newInput = [&](const Twine &name, Type type,
                        SmallVector<Value> values) {
      ins.push_back({name.str(), type, std::move(values)});
      return cbody->addArgument(type, parent.getLoc());
    };
    auto bcast = [&](Value v) -> Value {
      if (auto c = v.getDefiningOp<hw::ConstantOp>())
        return hw::ConstantOp::create(b, c.getValue());
      auto it = broadcast.find(v);
      if (it != broadcast.end())
        return it->second;
      Value arg = newInput("bc" + Twine(ins.size()), v.getType(), {v});
      broadcast[v] = arg;
      return arg;
    };
    // The parent's builder (constants, slices, the scans' starts and ends).
    OpBuilder pb(body, body->getTerminator()->getIterator());
    pb.setInsertionPoint(body->getTerminator());
    // The lane's index: one input of constants, j for lane j (a per-instance
    // constant, the instance index).
    unsigned laneBits = std::max(1u, llvm::Log2_64_Ceil(F));
    Value laneArg;
    auto laneIndex = [&]() -> Value {
      if (laneArg)
        return laneArg;
      SmallVector<Value> vs;
      for (unsigned j = 0; j < F; ++j)
        vs.push_back(hw::ConstantOp::create(pb, parent.getLoc(),
                                            APInt(laneBits, j)));
      laneArg = newInput("lane", b.getIntegerType(laneBits), vs);
      return laneArg;
    };
    // The lane index as a w-bit value.
    auto laneAs = [&](unsigned w) -> Value {
      Value l = laneIndex();
      if (w == laneBits)
        return l;
      if (w < laneBits)
        return comb::ExtractOp::create(b, l, 0, w);
      return comb::ConcatOp::create(
          b, hw::ConstantOp::create(b, APInt(w - laneBits, 0)), l);
    };
    // Per-lane constants: one constant when all equal, a + b * lane when
    // affine in the lane, else a per-lane input of constants.
    auto laneConstAP = [&](ArrayRef<APInt> values) -> Value {
      unsigned w = values[0].getBitWidth();
      if (llvm::all_of(values, [&](const APInt &v) { return v == values[0]; }))
        return hw::ConstantOp::create(b, values[0]);
      if (w == 1) {
        // One run of ones: a range of lanes.
        int lo = -1, hi = -1;
        bool run = true;
        for (unsigned j = 0; j < F; ++j)
          if (values[j].isOne()) {
            if (lo < 0)
              lo = j;
            else if (hi != int(j) - 1)
              run = false;
            hi = j;
          }
        if (run && lo >= 0) {
          unsigned lw = laneBits + 1;
          Value l = laneAs(lw);
          Value ge = comb::ICmpOp::create(
              b, comb::ICmpPredicate::uge, l,
              hw::ConstantOp::create(b, APInt(lw, lo)));
          Value le = comb::ICmpOp::create(
              b, comb::ICmpPredicate::ule, l,
              hw::ConstantOp::create(b, APInt(lw, hi)));
          return comb::AndOp::create(b, ge, le, false);
        }
      }
      APInt a = values[0], step = values[1] - values[0];
      bool affine = true;
      for (unsigned j = 0; j < F && affine; ++j)
        affine = values[j] == a + step * APInt(w, j, false, true);
      if (affine) {
        Value l = laneAs(w);
        Value t = l;
        if (step.isAllOnes())
          t = comb::SubOp::create(b, hw::ConstantOp::create(b, APInt(w, 0)), l,
                                  false);
        else if (!step.isOne())
          t = comb::MulOp::create(b, l, hw::ConstantOp::create(b, step), false);
        if (a.isZero())
          return t;
        return comb::AddOp::create(b, t, hw::ConstantOp::create(b, a), false);
      }
      SmallVector<Value> vs;
      for (auto &v : values)
        vs.push_back(hw::ConstantOp::create(pb, parent.getLoc(), v));
      return newInput("pl" + Twine(ins.size()), vs[0].getType(), vs);
    };
    auto laneConst = [&](Type type, ArrayRef<int64_t> values) -> Value {
      unsigned w = type.getIntOrFloatBitWidth();
      SmallVector<APInt> vs;
      for (int64_t v : values)
        vs.push_back(APInt(w, v, false, true));
      return laneConstAP(vs);
    };
    // Lane j's bits [lo(j), lo(j) + w) of a parent value, lo affine in the
    // lane: the value broadcast, shifted by the lane's offset in the cell.
    auto dynSlice = [&](Value scalar, int64_t lo0, int64_t stride,
                        unsigned w) -> Value {
      unsigned sw = widthOf(scalar);
      Value s = bcast(scalar);
      SmallVector<APInt> offs;
      for (unsigned j = 0; j < F; ++j)
        offs.push_back(APInt(sw, lo0 + stride * int64_t(j), false, true));
      Value off = laneConstAP(offs);
      Value sh = comb::ShrUOp::create(b, s, off, false);
      return comb::ExtractOp::create(b, sh, 0, w);
    };
    // Lane j's values of one operand position: the same value everywhere
    // (broadcast or constant), per-lane constants, slices of one parent value
    // at offsets affine in the lane, or a per-lane input.
    auto perLane = [&](ArrayRef<Value> values) -> Value {
      if (llvm::all_equal(values))
        return bcast(values[0]);
      SmallVector<APInt> cs;
      for (Value v : values) {
        APInt c;
        if (!isConstant(v, c))
          break;
        cs.push_back(c);
      }
      if (cs.size() == values.size())
        return laneConstAP(cs);
      // Slices of one value.
      Value src;
      SmallVector<int64_t> los;
      unsigned w = widthOf(values[0]);
      for (Value v : values) {
        auto ex = v.getDefiningOp<comb::ExtractOp>();
        if (!ex || (src && ex.getInput() != src))
          break;
        src = ex.getInput();
        los.push_back(ex.getLowBit());
      }
      if (los.size() == values.size() && w > 0) {
        int64_t stride = F > 1 ? los[1] - los[0] : 0;
        bool affine = true;
        for (unsigned j = 0; j < F; ++j)
          affine &= los[j] == los[0] + stride * int64_t(j);
        if (affine)
          return dynSlice(src, los[0], stride, w);
      }
      return newInput("pl" + Twine(ins.size()), values[0].getType(),
                      SmallVector<Value>(values.begin(), values.end()));
    };
    // A slice of a parent scalar for each lane's bits of a vector.
    auto laneSlices = [&](Value scalar, const Info &layout) -> Value {
      APInt c;
      if (isConstant(scalar, c)) {
        SmallVector<APInt> vals;
        for (unsigned j = 0; j < F; ++j)
          vals.push_back(c.extractBits(layout.w, layout.lo(j)));
        return laneConstAP(vals);
      }
      return dynSlice(scalar, layout.lo(0), layout.stride, layout.w);
    };


    // The cell's value of a parent value (lane 0's member for packs).
    // Aliases: a vector of one pack's members is that pack's value, a whole
    // lane slice of a vector is the vector's.
    DenseMap<Value, Value> alias;
    auto canon = [&](Value v) {
      for (;;) {
        if (s.isLaneVal(v))
          v = s.tmpl(s.packOf.lookup(v));
        auto it = alias.find(v);
        if (it == alias.end())
          return v;
        v = it->second;
      }
    };
    std::function<Value(Value)> cellValue = [&](Value v) -> Value {
      auto &i = s.get(v);
      Value key = canon(v);
      auto it = cellOf.find(key);
      if (it != cellOf.end())
        return it->second;
      // A lane input: the pack of an instance output, read on first use.
      if (auto inst = key.getDefiningOp<hw::InstanceOp>()) {
        int p = s.packOf.lookup(key);
        unsigned r = cast<OpResult>(key).getResultNumber();
        SmallVector<Value> vs(s.packMembers[p].begin(),
                              s.packMembers[p].end());
        ins.push_back(
            {("in_" + cast<StringAttr>(inst.getResultNames()[r]).getValue())
                 .str(),
             key.getType(), vs});
        Value arg = cbody->addArgument(key.getType(), parent.getLoc());
        cellOf[key] = arg;
        return arg;
      }
      auto &ki = s.get(key);
      (void)i;
      Type t = ki.kind == Kind::Vec ? b.getIntegerType(ki.w) : key.getType();
      auto pit = pending.find(key);
      if (pit != pending.end())
        return pit->second;
      auto e = backedges.get(t);
      pending.insert({key, e});
      return e;
    };
    auto define = [&](Value key, Value cellV) {
      cellOf[key] = cellV;
      auto pit = pending.find(key);
      if (pit != pending.end()) {
        pit->second.setValue(cellV);
        pending.erase(pit);
      }
    };
    // A value of lane 0 made of bits: its pieces, from the low bits up.
    auto fromPieces = [&](ArrayRef<Piece> ps) -> Value {
      SmallVector<Value> parts; // low first
      for (auto &pc : ps) {
        Value part;
        switch (pc.k) {
        case BitSrc::Zero:
          part = hw::ConstantOp::create(b, APInt(pc.len, 0));
          break;
        case BitSrc::One:
          part = hw::ConstantOp::create(b, APInt::getAllOnes(pc.len));
          break;
        case BitSrc::Scalar: {
          Value x = bcast(pc.src);
          part = widthOf(x) == pc.len
                     ? x
                     : comb::ExtractOp::create(b, x, pc.bit, pc.len).getResult();
          break;
        }
        case BitSrc::Opaque: // never in a lane's value
          part = hw::ConstantOp::create(b, APInt(pc.len, 0));
          break;
        case BitSrc::Lane: {
          Value x = cellValue(pc.src);
          part = widthOf(x) == pc.len
                     ? x
                     : comb::ExtractOp::create(b, x, pc.bit, pc.len).getResult();
          break;
        }
        }
        parts.push_back(part);
      }
      if (parts.size() == 1)
        return parts[0];
      std::reverse(parts.begin(), parts.end());
      return comb::ConcatOp::create(b, parts);
    };
    // Operand k of a pack member op, as the cell sees it.
    // A prefix chain's link: the previous stage's value of a pack (lane j - 1
    // read at lane j), from the cell before; the parent's value of it into
    // the pack's first stage.
    DenseMap<std::pair<Value, int>, Value> stageIn;
    auto prevStage = [&](int pack, unsigned k, Value o0) -> Value {
      int start = s.packFirst[pack];
      // The value this operand reads, one stage on: what the cell makes.
      Value role = s.packMembers[pack][start + 1].getDefiningOp()->getOperand(k);
      Value key = canon(role);
      auto it = stageIn.find({key, start});
      if (it != stageIn.end())
        return it->second;
      o0 = role;
      Value init = s.packMembers[pack][start].getDefiningOp()->getOperand(k);
      int c = chains.size();
      chains.push_back({o0.getType(), init, false});
      chains[c].start = start;
      Value in = cbody->addArgument(o0.getType(), parent.getLoc());
      ins.push_back({"st" + std::to_string(c) + "_in", o0.getType(), {}});
      ins.back().chain = c;
      chains[c].in = ins.size() - 1;
      outs.push_back({"st" + std::to_string(c) + "_out", Value(), {}, c, o0});
      chains[c].out = outs.size() - 1;
      stageIn[{key, start}] = in;
      return in;
    };
    auto operandOf = [&](int pack, unsigned k) -> Value {
      Value t = s.tmpl(pack);
      Value o0 = t.getDefiningOp()->getOperand(k);
      if (s.isLaneVal(o0)) {
        if (s.get(o0).lane == s.get(t).lane - 1)
          return prevStage(pack, k, o0);
        return cellValue(o0);
      }
      if (s.get(o0).kind == Kind::Vec)
        return cellValue(o0);
      SmallVector<Value> vs;
      for (unsigned j = 0; j < F; ++j) {
        Value m = s.packMembers[pack][j] ? s.packMembers[pack][j] : t;
        vs.push_back(m.getDefiningOp()->getOperand(k));
      }
      return perLane(vs);
    };


    // The packs and vectors, in the parent's order.
    for (auto &op : *body) {
      if (isa<hw::InstanceOp, hw::OutputOp>(op))
        continue;
      for (auto [r, v] : llvm::enumerate(op.getResults())) {
        auto &i = s.get(v);
        if (i.kind == Kind::Elem || i.kind == Kind::Pfx) {
          int p = s.packOf.lookup(v);
          if (!cellPack[p] || s.tmpl(p) != v)
            continue;
          // A lane's value of its own bits.
          if (auto it = s.assembled.find(v); it != s.assembled.end()) {
            define(v, fromPieces(it->second));
            continue;
          }
          // A lane slice of a vector.
          if (auto ex = dyn_cast<comb::ExtractOp>(op);
              ex && s.get(ex.getInput()).kind == Kind::Vec) {
            auto &vi = s.get(ex.getInput());
            int64_t rel = ex.getLowBit() - vi.lo(0);
            int64_t n = widthOf(v);
            if (n == vi.w)
              alias[v] = ex.getInput();
            else
              define(v, comb::ExtractOp::create(b, cellValue(ex.getInput()),
                                                rel, n));
            continue;
          }
          IRMapping map;
          for (unsigned k = 0; k < op.getNumOperands(); ++k)
            map.map(op.getOperand(k), operandOf(p, k));
          auto *clone = b.clone(op, map);
          clone->removeAttr("inner_sym");
          define(v, clone->getResult(r));
          continue;
        }
        if (i.kind != Kind::Vec || !cellVec.count(v))
          continue;
        // A lane vector: its op on the lanes' slices.
        Value slice;
        if (i.gather) {
          auto &bs = s.bits[v];
          SmallVector<BitSrc> lane0(bs.begin() + i.lo(0),
                                    bs.begin() + i.lo(0) + i.w);
          bool regular = true;
          for (int64_t k = 0; k < i.w && regular; ++k)
            for (unsigned j = 0; j < F && regular; ++j)
              regular = (bs[i.lo(j) + k].k == BitSrc::Lane) ==
                        (lane0[k].k == BitSrc::Lane);
          if (regular) {
            slice = fromPieces(Scan::pieces(lane0));
          } else {
            // Bit by bit: the lanes' common source, or each lane's constant.
            SmallVector<Value> parts;
            for (int64_t k = i.w - 1; k >= 0; --k) {
              const BitSrc *a = nullptr;
              SmallVector<int64_t> isLane, cst;
              for (unsigned j = 0; j < F; ++j) {
                auto &x = bs[i.lo(j) + k];
                if (x.k == BitSrc::Lane && !a)
                  a = &x;
                isLane.push_back(x.k == BitSrc::Lane);
                cst.push_back(x.k == BitSrc::One);
              }
              Value c = laneConst(b.getI1Type(), cst);
              if (!a) {
                parts.push_back(c);
                continue;
              }
              Value src = cellValue(a->src);
              Value bit = comb::ExtractOp::create(b, src, a->bit, 1);
              Value sel = laneConst(b.getI1Type(), isLane);
              parts.push_back(comb::MuxOp::create(b, sel, bit, c, false));
            }
            slice = parts.size() == 1 ? parts[0]
                                      : comb::ConcatOp::create(b, parts)
                                            .getResult();
          }
        } else if (isa<comb::ConcatOp>(op)) {
          for (Value o : op.getOperands())
            if (s.get(o).lane == 0)
              alias[v] = o;
          continue;
        } else if (s.carries.count(&op)) {
          // The carry chain: {cout, sum} = a + b + cin, a lane at a time
          // from the lowest bits.
          auto add = cast<comb::AddOp>(op);
          SmallVector<Value> terms;
          for (Value o : add.getInputs())
            terms.push_back(s.get(o).kind == Kind::Vec ? cellValue(o)
                                                       : laneSlices(o, i));
          int c = chains.size();
          chains.push_back({b.getI1Type(),
                            hw::ConstantOp::create(pb, parent.getLoc(),
                                                   APInt(1, 0)),
                            i.stride < 0});
          Value cin = cbody->addArgument(b.getI1Type(), parent.getLoc());
          ins.push_back({"ch" + std::to_string(c) + "_in", b.getI1Type(), {}});
          ins.back().chain = c;
          chains[c].in = ins.size() - 1;
          auto wide = b.getIntegerType(i.w + 1);
          auto zext = [&](Value x) {
            return comb::ConcatOp::create(
                b, hw::ConstantOp::create(b, APInt(1, 0)), x);
          };
          Value sum = comb::AddOp::create(
              b, wide,
              ValueRange{zext(terms[0]), zext(terms[1]),
                         comb::ConcatOp::create(
                             b, hw::ConstantOp::create(b, APInt(i.w, 0)),
                             cin)},
              false);
          slice = comb::ExtractOp::create(b, sum, 0, i.w);
          Value cout = comb::ExtractOp::create(b, sum, i.w, 1);
          outs.push_back({"ch" + std::to_string(c) + "_out", cout, {}, c});
          chains[c].out = outs.size() - 1;
        } else if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
          Value next = cellValue(reg.getNext());
          Value clk = bcast(reg.getClk());
          if (reg.hasReset()) {
            Value rst = bcast(reg.getReset());
            Value rv = s.get(reg.getResetValue()).kind == Kind::Vec
                           ? cellValue(reg.getResetValue())
                           : laneSlices(reg.getResetValue(), i);
            slice = seq::FirRegOp::create(b, next, clk, reg.getNameAttr(),
                                          rst, rv, hw::InnerSymAttr(),
                                          reg.getIsAsync());
          } else {
            slice = seq::FirRegOp::create(b, next, clk, reg.getNameAttr());
          }
        } else if (auto mux = dyn_cast<comb::MuxOp>(op)) {
          auto side = [&](Value x) {
            return s.get(x).kind == Kind::Vec ? cellValue(x)
                                              : laneSlices(x, i);
          };
          slice = comb::MuxOp::create(b, bcast(mux.getCond()),
                                      side(mux.getTrueValue()),
                                      side(mux.getFalseValue()),
                                      mux.getTwoState());
        } else {
          // and / or / xor
          SmallVector<Value> operands;
          for (Value o : op.getOperands())
            operands.push_back(s.get(o).kind == Kind::Vec ? cellValue(o)
                                                          : laneSlices(o, i));
          OperationState st(op.getLoc(), op.getName());
          st.addOperands(operands);
          st.addTypes(b.getIntegerType(i.w));
          st.addAttributes(op.getAttrs());
          st.attributes.erase("sv.namehint");
          slice = b.create(st)->getResult(0);
        }
        define(v, slice);
      }
    }

    // The scans.
    struct Ending {
      Operation *root;
      int chain;
      std::function<Value(OpBuilder &, Value)> finish;
    };
    SmallVector<Ending> endings;
    auto newChain = [&](Type t, Value init, bool down) {
      int c = chains.size();
      chains.push_back({t, init, down});
      Value in = cbody->addArgument(t, parent.getLoc());
      ins.push_back({"ch" + std::to_string(c) + "_in", t, {}});
      ins.back().chain = c;
      chains[c].in = ins.size() - 1;
      return std::make_pair(c, in);
    };
    auto endChain = [&](int c, Value out) {
      outs.push_back({"ch" + std::to_string(c) + "_out", out, {}, c});
      chains[c].out = outs.size() - 1;
    };
    auto pconst = [&](Type t, const APInt &v) -> Value {
      return hw::ConstantOp::create(pb, parent.getLoc(), v);
    };
    for (auto *p : pats) {
      Operation *root = p->root;
      Type rt = root->getResult(0).getType();
      unsigned rw = widthOf(root->getResult(0));
      switch (p->kind) {
      case PatKind::AC: {
        StringRef name = nameOf(root);
        APInt id = name == "comb.and" ? APInt::getAllOnes(rw) : APInt(rw, 0);
        auto [c, in] = newChain(rt, pconst(rt, id), false);
        SmallVector<Value> operands{in};
        for (int pk : p->packs)
          operands.push_back(cellValue(s.tmpl(pk)));
        for (Value v : p->vecs) {
          Value bit = cellValue(v);
          if (rw > 1)
            bit = comb::ConcatOp::create(
                b, hw::ConstantOp::create(b, APInt(rw - 1, 0)), bit);
          operands.push_back(bit);
        }
        OperationState st(root->getLoc(), root->getName());
        st.addOperands(operands);
        st.addTypes(rt);
        if (root->hasAttr("twoState"))
          st.addAttribute("twoState", root->getAttr("twoState"));
        endChain(c, b.create(st)->getResult(0));
        SmallVector<Value> rest = p->rest;
        endings.push_back({root, c, [rest, root](OpBuilder &ob, Value end) -> Value {
                             if (rest.empty())
                               return end;
                             SmallVector<Value> ops(rest);
                             ops.push_back(end);
                             OperationState st(root->getLoc(),
                                               root->getName());
                             st.addOperands(ops);
                             st.addTypes(end.getType());
                             return ob.create(st)->getResult(0);
                           }});
        break;
      }
      case PatKind::VecRed: {
        auto i1 = b.getI1Type();
        APInt id = p->red == "and" ? APInt(1, 1) : APInt(1, 0);
        auto [c, in] = newChain(i1, pconst(i1, id), false);
        Value slice = cellValue(p->vec);
        unsigned w = widthOf(slice);
        Value term;
        if (p->red == "and")
          term = comb::ICmpOp::create(b, comb::ICmpPredicate::eq, slice,
                                      hw::ConstantOp::create(
                                          b, APInt::getAllOnes(w)));
        else if (p->red == "or")
          term = comb::ICmpOp::create(b, comb::ICmpPredicate::ne, slice,
                                      hw::ConstantOp::create(b, APInt(w, 0)));
        else
          term = comb::ParityOp::create(b, slice);
        Value out;
        if (p->red == "and")
          out = comb::AndOp::create(b, in, term, false);
        else if (p->red == "or")
          out = comb::OrOp::create(b, in, term, false);
        else
          out = comb::XorOp::create(b, in, term, false);
        endChain(c, out);
        bool invert = p->invert;
        endings.push_back({root, c, [invert](OpBuilder &ob, Value end) -> Value {
                             if (!invert)
                               return end;
                             return comb::XorOp::create(
                                        ob, end.getLoc(), end,
                                        hw::ConstantOp::create(
                                            ob, end.getLoc(), APInt(1, 1)),
                                        false)
                                 .getResult();
                           }});
        break;
      }
      case PatKind::Prio: {
        // Chain bits: {found, value}.
        auto t = b.getIntegerType(rw + 1);
        auto [c, in] = newChain(t, pconst(t, APInt(rw + 1, 0)), false);
        Value found = comb::ExtractOp::create(b, in, rw, 1);
        Value val = comb::ExtractOp::create(b, in, 0, rw);
        // Each lane's condition and value; lanes outside the chain are
        // disabled.
        int lane0 = -1;
        for (unsigned j = 0; j < F; ++j)
          if (p->conds[j]) {
            lane0 = j;
            break;
          }
        int pc = s.packOf.lookup(p->conds[lane0]);
        Value cond = cellValue(s.tmpl(pc));
        SmallVector<int64_t> en;
        for (unsigned j = 0; j < F; ++j)
          en.push_back(p->conds[j] ? 1 : 0);
        Value enable = laneConst(b.getI1Type(), en);
        Value v;
        if (s.isElem(p->vals[lane0]) && !s.backward.count(p->vals[lane0])) {
          v = cellValue(s.packMembers[s.packOf.lookup(p->vals[lane0])][0]);
        } else {
          // Constants: a disabled lane's value is free; it continues the
          // enabled lanes' line where they are affine in the lane.
          SmallVector<APInt> cs(F, APInt(rw, 0));
          SmallVector<int> on;
          bool consts = true;
          for (unsigned j = 0; j < F; ++j)
            if (p->conds[j]) {
              consts &= isConstant(p->vals[j], cs[j]);
              on.push_back(j);
            }
          if (consts && on.size() >= 2) {
            APInt d = cs[on[1]] - cs[on[0]];
            int64_t dj = on[1] - on[0];
            bool line = dj == 1;
            for (unsigned k = 0; k < on.size() && line; ++k)
              line = cs[on[k]] ==
                     cs[on[0]] + d * APInt(rw, on[k] - on[0], false, true);
            if (line)
              for (unsigned j = 0; j < F; ++j)
                if (!p->conds[j])
                  cs[j] = cs[on[0]] + d * APInt(rw, int64_t(j) - on[0], true,
                                                true);
          }
          if (consts) {
            v = laneConstAP(cs);
          } else {
            SmallVector<Value> vs;
            for (unsigned j = 0; j < F; ++j)
              vs.push_back(p->conds[j] ? p->vals[j]
                                       : pconst(rt, APInt(rw, 0)));
            v = perLane(vs);
          }
        }
        Value hit = comb::AndOp::create(b, cond, enable, false);
        Value take = hit;
        if (p->firstWins)
          take = comb::AndOp::create(
              b, hit,
              comb::XorOp::create(b, found,
                                  hw::ConstantOp::create(b, APInt(1, 1)),
                                  false),
              false);
        Value nval = comb::MuxOp::create(b, take, v, val, false);
        Value nfound = comb::OrOp::create(b, found, hit, false);
        endChain(c, comb::ConcatOp::create(b, nfound, nval));
        Value terminal = p->terminal;
        endings.push_back(
            {root, c, [terminal, rw](OpBuilder &ob, Value end) -> Value {
               auto loc = end.getLoc();
               Value f = comb::ExtractOp::create(ob, loc, end, rw, 1);
               Value v = comb::ExtractOp::create(ob, loc, end, 0, rw);
               return comb::MuxOp::create(ob, loc, f, v, terminal, false)
                   .getResult();
             }});
        break;
      }
      case PatKind::Rot: {
        // Chain bits {hf, hv, lf, lv}: the first (or last) hit in the
        // segment from the start lane on, and in the one before it.
        unsigned W = 2 * (rw + 1);
        auto t = b.getIntegerType(W);
        auto [c, in] = newChain(t, pconst(t, APInt(W, 0)), false);
        Value lv = comb::ExtractOp::create(b, in, 0, rw);
        Value lf = comb::ExtractOp::create(b, in, rw, 1);
        Value hv = comb::ExtractOp::create(b, in, rw + 1, rw);
        Value hf = comb::ExtractOp::create(b, in, 2 * rw + 1, 1);
        Value cond =
            p->vec ? cellValue(p->vec)
                   : cellValue(s.packMembers[s.packOf.lookup(p->elems[0])][0]);
        unsigned K = llvm::Log2_64_Ceil(F) + 1;
        SmallVector<int64_t> idx;
        for (unsigned j = 0; j < F; ++j)
          idx.push_back(j);
        Value lj = laneConst(b.getIntegerType(K), idx);
        // The start lane, computed in the parent: (a + e p) mod F.
        unsigned pw = widthOf(p->index), WW = pw + K + 2;
        auto loc = parent.getLoc();
        Value zp = comb::ConcatOp::create(
            pb, loc, hw::ConstantOp::create(pb, loc, APInt(WW - pw, 0)),
            p->index);
        Value sum;
        if (p->rotE == 1)
          sum = comb::AddOp::create(
              pb, loc, zp, hw::ConstantOp::create(pb, loc, APInt(WW, p->rotA)),
              false);
        else
          sum = comb::SubOp::create(
              pb, loc,
              hw::ConstantOp::create(
                  pb, loc, APInt(WW, p->rotA + (int64_t(F) << pw))),
              zp, false);
        Value smod = comb::ModUOp::create(
            pb, loc, sum, hw::ConstantOp::create(pb, loc, APInt(WW, F)),
            false);
        Value start = comb::ExtractOp::create(pb, loc, smod, 0, K);
        Value sv = bcast(start);
        Value inHi = comb::ICmpOp::create(
            b,
            p->rotDir == 1 ? comb::ICmpPredicate::uge
                           : comb::ICmpPredicate::ule,
            lj, sv);
        Value one = hw::ConstantOp::create(b, APInt(1, 1));
        Value notHi = comb::XorOp::create(b, inHi, one, false);
        SmallVector<Value> vals;
        bool same = true;
        for (unsigned j = 0; j < F; ++j)
          same &= p->rotVals[j] == p->rotVals[0];
        (void)same;
        Value v = laneConstAP(p->rotVals);
        Value hitH = comb::AndOp::create(b, cond, inHi, false);
        Value hitL = comb::AndOp::create(b, cond, notHi, false);
        Value takeH = hitH, takeL = hitL;
        if (p->rotDir == 1) {
          takeH = comb::AndOp::create(
              b, hitH, comb::XorOp::create(b, hf, one, false), false);
          takeL = comb::AndOp::create(
              b, hitL, comb::XorOp::create(b, lf, one, false), false);
        }
        Value nhv = comb::MuxOp::create(b, takeH, v, hv, false);
        Value nlv = comb::MuxOp::create(b, takeL, v, lv, false);
        Value nhf = comb::OrOp::create(b, hf, hitH, false);
        Value nlf = comb::OrOp::create(b, lf, hitL, false);
        endChain(c, comb::ConcatOp::create(b, ValueRange{nhf, nhv, nlf, nlv}));
        Value terminal = p->terminal;
        endings.push_back(
            {root, c, [terminal, rw](OpBuilder &ob, Value end) -> Value {
               auto loc = end.getLoc();
               Value lv = comb::ExtractOp::create(ob, loc, end, 0, rw);
               Value lf = comb::ExtractOp::create(ob, loc, end, rw, 1);
               Value hv = comb::ExtractOp::create(ob, loc, end, rw + 1, rw);
               Value hf = comb::ExtractOp::create(ob, loc, end, 2 * rw + 1, 1);
               Value low = comb::MuxOp::create(ob, loc, lf, lv, terminal, false);
               return comb::MuxOp::create(ob, loc, hf, hv, low, false)
                   .getResult();
             }});
        break;
      }
      case PatKind::Select:
      case PatKind::BitSel: {
        auto [c, in] = newChain(rt, pconst(rt, APInt(rw, 0)), false);
        Value idx = bcast(p->index);
        Value pos = laneConst(p->index.getType(), p->pos);
        Value hit = comb::ICmpOp::create(b, comb::ICmpPredicate::eq, idx, pos);
        Value elem;
        if (p->kind == PatKind::Select)
          elem = cellValue(s.packMembers[s.packOf.lookup(p->elems[0])][0]);
        else
          elem = cellValue(p->vec);
        endChain(c, comb::MuxOp::create(b, hit, elem, in, false));
        endings.push_back(
            {root, c, [](OpBuilder &, Value end) -> Value { return end; }});
        break;
      }
      }
    }

    // ------------------------------------------------------------------
    // Exports: what the parent still reads of the moved values. Liveness
    // from the parent's outputs, every instance's inputs, ops with effects
    // and the cell's own inputs, through the parent's ops; a reduction's
    // root reads only what its ending keeps (its other operands, the prio
    // default, the index); a moved value reached is an export.
    DenseSet<Operation *> moved;
    for (unsigned p = 0; p < cellPack.size(); ++p)
      if (cellPack[p])
        for (auto [j, m] : llvm::enumerate(s.packMembers[p]))
          if (m && int(j) >= s.packFirst[p] &&
              !isa<hw::InstanceOp>(m.getDefiningOp()))
            moved.insert(m.getDefiningOp());
    for (Value v : cellVec)
      moved.insert(v.getDefiningOp());
    DenseMap<Operation *, Pattern *> rootOf;
    for (auto *p : pats)
      rootOf[p->root] = p;
    DenseSet<Operation *> live;
    SetVector<Value> exported;
    SmallVector<Value> work;
    auto reach = [&](Value v) {
      auto *d = v.getDefiningOp();
      if (!d)
        return;
      if (moved.count(d)) {
        exported.insert(v);
        return;
      }
      if (live.insert(d).second)
        work.push_back(v);
    };
    auto keep = [&](Operation *op) {
      if (auto it = rootOf.find(op); it != rootOf.end()) {
        auto *p = it->second;
        for (Value r : p->rest)
          reach(r);
        if (p->terminal)
          reach(p->terminal);
        if (p->index)
          reach(p->index);
        return;
      }
      for (Value o : op->getOperands())
        reach(o);
    };
    for (auto &op : *body) {
      bool root = isa<hw::OutputOp, hw::InstanceOp>(op) ||
                  op.getNumResults() == 0 ||
                  (!isa<seq::FirRegOp, seq::CompRegOp>(op) &&
                   !isMemoryEffectFree(&op));
      if (root && !moved.count(&op)) {
        live.insert(&op);
        keep(&op);
      }
    }
    for (auto &in : ins)
      if (in.chain < 0)
        for (Value v : in.values)
          reach(v);
    for (auto &ch : chains)
      if (ch.init)
        reach(ch.init);
    while (!work.empty()) {
      Value v = work.pop_back_val();
      keep(v.getDefiningOp());
    }
    // A pack member exported exports its pack (an output of every lane).
    SetVector<int> exportPacks;
    SetVector<Value> exportVecs;
    for (Value v : exported) {
      if (s.isLaneVal(v))
        exportPacks.insert(s.packOf.lookup(v));
      else
        exportVecs.insert(v);
    }
    for (int p : exportPacks) {
      Value cv = cellValue(s.tmpl(p));
      SmallVector<Value> ms(s.packMembers[p].begin(), s.packMembers[p].end());
      for (int j = 0; j < s.packFirst[p]; ++j)
        ms[j] = Value(); // the parent's own
      outs.push_back({"out" + std::to_string(outs.size()), cv, ms});
    }
    for (Value v : exportVecs)
      outs.push_back({"vec" + std::to_string(outs.size()), cellValue(v), {v}});

    if (dryRun) {
      backedges.abandon();
      cell.erase();
      llvm::errs() << "lane-scan: " << module << " lanes " << callee << " x"
                   << F << " (dry run)\n";
      continue;
    }
    if (!pending.empty()) {
      parent.emitError("lane-scan: a cell value was never defined");
      return signalPassFailure();
    }

    // Report. The residual: live parent ops over several lanes that no scan
    // took (they read the cells' exports).
    std::map<std::string, unsigned> residual;
    unsigned residualOps = 0, residualBits = 0;
    for (auto &op : *body)
      if (before.count(&op) && live.count(&op) && !rootOf.count(&op) &&
          op.getNumResults() &&
          (s.get(op.getResult(0)).kind == Kind::Multi ||
           (s.isPfx(op.getResult(0)) && !moved.count(&op))) &&
          !isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp,
               hw::ArrayCreateOp, hw::BitcastOp, hw::StructCreateOp,
               hw::StructExtractOp, hw::ArrayConcatOp, hw::ArraySliceOp>(
              op)) {
        // Wiring (concat, extract, aggregates) costs no logic: not counted.
        ++residualOps;
        if (getenv("FOLDSIM_LANE_RESIDUAL")) {
          llvm::errs() << "lane-residual " << callee << " " << nameOf(&op)
                       << " i" << widthOf(op.getResult(0)) << " (";
          for (Value o : op.getOperands()) {
            auto &oi = s.get(o);
            llvm::errs() << "SEVAM"[int(oi.kind)];
            if (auto *d = o.getDefiningOp())
              llvm::errs() << ":" << nameOf(d);
            if (auto ex = o.getDefiningOp<comb::ExtractOp>()) {
              std::vector<BitSrc> bs;
              if (s.bitsOf(ex.getInput(), bs)) {
                auto &bb = bs[ex.getLowBit()];
                llvm::errs() << "[k" << int(bb.k);
                if (bb.src)
                  llvm::errs() << " "
                               << (bb.src.getDefiningOp()
                                       ? nameOf(bb.src.getDefiningOp())
                                       : StringRef("arg"))
                               << " i" << widthOf(bb.src) << "@" << bb.bit;
                llvm::errs() << "]";
              }
            }
            llvm::errs() << ":i" << widthOf(o) << " ";
          }
          llvm::errs() << ")\n";
        }
        residualBits += widthOf(op.getResult(0));
        residual[nameOf(&op).str()]++;
      }
    unsigned paramBits = 0, params = 0;
    for (auto &in : ins)
      if (in.chain < 0 && in.values.size() == F &&
          !StringRef(in.name).starts_with("in_")) {
        ++params;
        paramBits += in.type.getIntOrFloatBitWidth();
      }
    unsigned exportBits = 0, exports = 0;
    for (auto &o : outs)
      if (o.chain < 0) {
        ++exports;
        exportBits += widthOf(o.cell);
      }
    unsigned chainBits = 0;
    for (auto &c : chains)
      chainBits += c.type.getIntOrFloatBitWidth();
    // The cell's ports and terminator.
    for (auto &in : ins)
      addPort(in.name, in.type, true);
    for (auto &o : outs)
      if (o.key)
        o.cell = cellValue(o.key);
    SmallVector<Value> outVals;
    for (auto &o : outs) {
      addPort(o.name, o.cell.getType(), false);
      outVals.push_back(o.cell);
    }
    hw::OutputOp::create(b, outVals);
    if (failed(backedges.clearOrEmitError()))
      return signalPassFailure();
    // Rebuild the module with its ports, moving the body.
    {
      OpBuilder mb(cell);
      auto real = hw::HWModuleOp::create(mb, parent.getLoc(),
                                         StringAttr::get(ctx, cellName),
                                         hw::ModulePortInfo(ports));
      auto *rbody = real.getBodyBlock();
      rbody->getTerminator()->erase();
      for (auto [k, arg] : llvm::enumerate(cbody->getArguments()))
        arg.replaceAllUsesWith(rbody->getArgument(k));
      rbody->getOperations().splice(rbody->end(), cbody->getOperations());
      cell.erase();
      cell = real;
    }
    // Move the cell before the parent so it is defined first.
    cell->moveBefore(parent);

    // ------------------------------------------------------------------
    // The instances, one a lane, chained.
    BackedgeBuilder pedges(pb, parent.getLoc());
    SmallVector<SmallVector<Backedge>> instIn(F);
    SmallVector<hw::InstanceOp> cells(F);
    for (unsigned j = 0; j < F; ++j) {
      SmallVector<Value> operands;
      for (auto &in : ins) {
        auto e = pedges.get(in.type);
        instIn[j].push_back(e);
        operands.push_back(e);
      }
      cells[j] = hw::InstanceOp::create(
          pb, parent.getLoc(), cell,
          StringAttr::get(ctx, "lane_" + callee + "_" + std::to_string(j)),
          operands);
    }
    // Roots replaced by their chains' ends.
    DenseMap<Value, Value> replaced;
    for (auto &e : endings) {
      auto &ch = chains[e.chain];
      unsigned last = ch.down ? 0 : F - 1;
      Value end = cells[last].getResult(ch.out);
      Value v = e.finish(pb, end);
      Value old = e.root->getResult(0);
      replaced[old] = v;
    }
    auto resolve = [&](Value v) {
      auto it = replaced.find(v);
      return it == replaced.end() ? v : it->second;
    };
    for (unsigned j = 0; j < F; ++j)
      for (auto [k, in] : llvm::enumerate(ins)) {
        Value v;
        if (in.chain >= 0) {
          auto &ch = chains[in.chain];
          bool first = ch.down ? j == F - 1 : j <= unsigned(ch.start);
          if (first)
            v = ch.init;
          else
            v = cells[ch.down ? j + 1 : j - 1].getResult(ch.out);
        } else if (in.values.size() == 1) {
          v = resolve(in.values[0]);
        } else {
          v = resolve(in.values[j]);
        }
        instIn[j][k].setValue(v);
      }
    // Exports.
    for (auto [k, o] : llvm::enumerate(outs)) {
      if (o.chain >= 0)
        continue;
      if (o.name.rfind("vec", 0) == 0) {
        Value v = o.replaces[0];
        auto &i = s.get(v);
        SmallVector<Value> parts(F);
        for (unsigned j = 0; j < F; ++j)
          parts[j] = cells[j].getResult(k);
        // concat takes the high bits first.
        SmallVector<Value> byBits;
        SmallVector<unsigned> order(F);
        for (unsigned j = 0; j < F; ++j)
          order[j] = j;
        llvm::sort(order, [&](unsigned a, unsigned b2) {
          return i.lo(a) > i.lo(b2);
        });
        for (unsigned j : order)
          byBits.push_back(parts[j]);
        Value whole = comb::ConcatOp::create(pb, parent.getLoc(), byBits);
        v.replaceUsesWithIf(whole, [&](OpOperand &use) {
          return !moved.count(use.getOwner());
        });
        continue;
      }
      for (unsigned j = 0; j < F; ++j) {
        Value m = o.replaces[j];
        if (!m)
          continue;
        m.replaceUsesWithIf(cells[j].getResult(k), [&](OpOperand &use) {
          return !moved.count(use.getOwner()) &&
                 use.getOwner()->getBlock() == body;
        });
      }
    }
    for (auto &[old, v] : replaced)
      old.replaceAllUsesWith(v);
    // What is not live is dead now: the moved ops, the roots and the ops
    // only they read (cycles of registers too).
    SmallVector<Operation *> dead;
    for (auto &op : *body)
      if (before.count(&op) && !isa<hw::OutputOp>(op) &&
          (!live.count(&op) || rootOf.count(&op)))
        dead.push_back(&op);
    {
      DenseSet<Operation *> deadSet(dead.begin(), dead.end());
      for (auto *op : dead)
        for (Value r : op->getResults())
          for (auto *u : r.getUsers())
            if (!deadSet.count(u)) {
              op->emitError("lane-scan: a value still read is dropped: ")
                  << *u;
              return signalPassFailure();
            }
    }
    for (auto *op : dead)
      op->dropAllReferences();
    for (auto *op : dead)
      op->erase();

    llvm::errs() << "lane-scan: cell " << cellName << "\n";
    llvm::errs() << "lane-scan: " << module << " lanes " << callee << " x" << F
                 << ": " << movedPacks << " packs (" << movedOps
                 << " ops moved), " << movedVecs << " lane vectors; scans";
    for (auto &[k, n] : kinds)
      llvm::errs() << " " << k << " " << n;
    llvm::errs() << " (" << chains.size() << " chains, " << chainBits
                 << " bits); per lane: " << params << " params (" << paramBits
                 << " bits), " << exports << " exports (" << exportBits
                 << " bits); residual " << residualOps << " ops ("
                 << residualBits << " bits)";
    std::vector<std::pair<unsigned, std::string>> rk;
    for (auto &[k, n] : residual)
      rk.push_back({n, k});
    llvm::sort(rk, [](auto &a, auto &b2) { return a.first > b2.first; });
    for (unsigned k = 0; k < rk.size() && k < 4; ++k)
      llvm::errs() << (k ? ", " : ": ") << rk[k].second << " " << rk[k].first;
    llvm::errs() << "\n";
  }
}

} // namespace
