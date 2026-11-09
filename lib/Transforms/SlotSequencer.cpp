//===- SlotSequencer.cpp - Slot tables as fixed-size decoders -------------===//
//
// A slot table of the simulator reads a constant per frame position (or per
// instance). As a constant array it takes 2^(index width) rows, so its bits
// grow with the frame (II) and the classes' instances: O(links x II) over the
// simulator. Its nonzero rows, though, follow the schedule: a class's passes
// sweep its instances, one every `stride` positions, the instance one more at
// each. Each such run is an affine segment: index s + j*ds holds v + j*dv.
// A segment decodes with a range compare on the index, a compare of its low
// bits (ds a power of two), and v + ((index - s) >> log2 ds) << log2 |dv|;
// the segments are disjoint, so their values (zero where they miss) are ORed.
// The decoder's size is the segments', independent of the frame's length and
// of how many instances a segment runs over.
//
//===----------------------------------------------------------------------===//

#include "SlotSequencer.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;

namespace foldsim {

SlotSequencerStats &slotSequencerStats() {
  static SlotSequencerStats stats;
  return stats;
}


bool csdTerms(uint64_t x, unsigned width, unsigned maxTerms,
              SmallVectorImpl<std::pair<bool, unsigned>> &terms) {
  terms.clear();
  uint64_t mask = width >= 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
  x &= mask;
  for (unsigned i = 0; x && i < width; ++i) {
    if (x & 1) {
      // A run of ones ending here is better as -1 here, +1 above it.
      bool neg = (x & 3) == 3 && i + 1 < width;
      terms.push_back({neg, i});
      x = neg ? x + 1 : x - 1;
      x &= mask;
      if (terms.size() > maxTerms)
        return false;
    }
    x >>= 1;
    mask >>= 1;
  }
  return true;
}

namespace {
constexpr unsigned kMaxStepTerms = 3;
bool stepOk(uint64_t x, unsigned width) {
  SmallVector<std::pair<bool, unsigned>> t;
  return csdTerms(x, width, kMaxStepTerms, t);
}
} // namespace

SmallVector<SlotSegment> slotSegments(ArrayRef<uint64_t> rows, unsigned width,
                                      unsigned maxSegments) {
  uint64_t mask = width >= 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
  size_t n = rows.size();
  auto row = [&](uint64_t k) { return k < n ? rows[k] & mask : 0; };
  std::vector<bool> covered(n, false);
  SmallVector<SlotSegment> segments;
  // The blocks of `seg` (start, stride, len, value, step, inner, digit set)
  // from its start: rows not covered, each its pattern's value (a zero row
  // only where the pattern gives zero); count = the blocks up to the last
  // with a nonzero row; returns the nonzero rows covered.
  auto extend = [&](SlotSegment &seg) -> uint64_t {
    uint64_t k = 0, last = 0, live = 0, liveAtLast = 0;
    for (;; ++k) {
      uint64_t base = seg.start + k * seg.stride;
      if (base + (seg.len - 1) * seg.istride >= n || (k && !seg.stride))
        break;
      bool ok = true;
      uint64_t blockLive = 0;
      for (uint64_t j = 0; j < seg.len && ok; ++j) {
        uint64_t p = base + j * seg.istride;
        ok = !covered[p] && row(p) == (seg.at(k, j) & mask);
        blockLive += row(p) != 0;
      }
      if (!ok)
        break;
      live += blockLive;
      if (blockLive) {
        last = k + 1;
        liveAtLast = live;
      }
    }
    seg.count = last;
    return liveAtLast;
  };
  for (size_t k0 = 0; k0 < n; ++k0) {
    if (covered[k0] || !row(k0))
      continue;
    SlotSegment best;
    best.start = k0;
    best.value = row(k0);
    best.count = 1;
    uint64_t bestLive = 1;
    auto consider = [&](SlotSegment seg) {
      uint64_t live = extend(seg);
      if (seg.count && live > bestLive) {
        best = seg;
        bestLive = live;
      }
    };
    // A block may start a row early, on a zero row its pattern gives (an
    // instance numbered from 0 whose first row is 0).
    SmallVector<uint64_t, 2> starts{k0};
    if (k0 > 0 && !covered[k0 - 1] && row(k0 - 1) == 0)
      starts.push_back(k0 - 1);
    for (uint64_t s : starts)
      for (uint64_t stride = 1; stride < n - s; stride *= 2) {
        SlotSegment seg;
        seg.start = s;
        seg.stride = stride;
        seg.value = row(s);
        // A plain run: the step from the next row of the stride.
        uint64_t d = row(s + stride) - row(s);
        if (stepOk(d, width) && !(s + stride < n && covered[s + stride])) {
          seg.step = int64_t(d & mask);
          consider(seg);
        }
        // Blocks: `len` consecutive rows of a step `inner` each stride.
        // (Rows `e` apart within the block, e a power of two below the
        // stride: a stream's reads at a reader's passes, the reader's
        // instances in a member set.) A block with e > 1 does not cross the
        // stride's boundary.
        for (uint64_t e = 1; stride >= 2 && e * 2 <= stride; e *= 2) {
          int64_t inner;
          uint64_t c = (row(s + e) - row(s)) & mask;
          uint64_t negc = (~c + 1) & mask;
          if (c == 0)
            inner = 0;
          else if (llvm::isPowerOf2_64(c))
            inner = int64_t(c);
          else if (llvm::isPowerOf2_64(negc))
            inner = -int64_t(negc);
          else
            continue;
          uint64_t r = s & (stride - 1), len = 1;
          while ((len + 1) * e <= stride && s + len * e < n &&
                 (e == 1 || r + len * e < stride) && !covered[s + len * e] &&
                 row(s + len * e) == ((row(s) + uint64_t(inner) * len) & mask))
            ++len;
          if (len >= 2 && stepOk(d, width)) {
            SlotSegment blk = seg;
            blk.len = len;
            blk.istride = e;
            blk.inner = inner;
            blk.step = int64_t(d & mask);
            consider(blk);
          }
        }
        // A digit of k: bits [shift, shift + bits) of k, the step from the
        // first k whose digit is 1 (stride 1, where no run is long).
        if (stride == 1 && bestLive < 64)
          for (unsigned shift = 0; shift < 10; ++shift)
            for (unsigned bits = 1; bits <= 10; ++bits) {
              uint64_t k1 = s + (uint64_t(1) << shift);
              if (k1 >= n || covered[k1])
                break;
              uint64_t dd = (row(k1) - row(s)) & mask;
              if (!dd || !stepOk(dd, width))
                break;
              SlotSegment f = seg;
              f.shift = shift;
              f.bits = bits;
              f.step = int64_t(dd);
              consider(f);
            }
      }
    if (best.count == 1 && best.len == 1)
      best.stride = 0;
    for (uint64_t k = 0; k < best.count; ++k)
      for (uint64_t j = 0; j < best.len; ++j)
        covered[best.start + k * best.stride + j * best.istride] = true;
    segments.push_back(best);
    if (segments.size() > maxSegments)
      break;
  }
  return segments;
}

/// The value the decoder built from `segments` gives at each index of `w`
/// bits, computed as the hardware computes it, compared with the rows.
static bool decodesExactly(ArrayRef<SlotSegment> segments,
                           ArrayRef<uint64_t> rows, unsigned width, unsigned w) {
  uint64_t mask = width >= 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
  uint64_t imask = (uint64_t(1) << w) - 1;
  uint64_t top = imask;
  for (uint64_t idx = 0; idx <= top; ++idx) {
    uint64_t got = 0;
    for (auto &seg : segments) {
      unsigned a = seg.stride > 1 ? llvm::Log2_64(seg.stride) : 0;
      uint64_t r = a ? seg.start & (seg.stride - 1) : 0;
      bool wraps = seg.len > 1 && r + seg.len > seg.stride;
      unsigned le = llvm::Log2_64(seg.istride);
      uint64_t offset = (idx - seg.start) & imask;
      bool hit;
      if (seg.count == 1 && seg.len == 1) {
        hit = idx == seg.start;
      } else {
        hit = idx >= seg.start && idx <= seg.end();
        if (a && seg.count > 1) {
          uint64_t lowMask = seg.stride - 1;
          if (seg.len == 1)
            hit &= (idx & lowMask) == r;
          else if (seg.istride > 1) {
            uint64_t low = idx & lowMask;
            hit &= (low & (seg.istride - 1)) == (r & (seg.istride - 1)) &&
                   (low >> le) >= (r >> le) && (low >> le) < (r >> le) + seg.len;
          } else if (seg.len < seg.stride) {
            uint64_t low = wraps ? offset & lowMask : idx & lowMask;
            uint64_t lo = wraps ? 0 : r;
            hit &= low >= lo && low < lo + seg.len;
          }
        }
      }
      if (!hit)
        continue;
      bool varies = (seg.step != 0 && seg.count > 1) || (seg.inner != 0 && seg.len > 1);
      uint64_t value = seg.value;
      if (varies) {
        uint64_t sa = seg.start >> a;
        bool plainK = seg.bits >= 64 && seg.shift == 0;
        bool direct = !wraps && (plainK || (seg.shift + seg.bits < 64 &&
                                            (sa & ((uint64_t(1) << (seg.shift + seg.bits)) - 1)) == 0));
        uint64_t c0 = seg.value, kv, jv;
        if (direct) {
          if (plainK)
            c0 -= uint64_t(seg.step) * sa;
          c0 -= uint64_t(seg.inner) * (r >> le);
          kv = idx >> a;
          jv = (idx & ((uint64_t(1) << a) - 1)) >> le;
        } else {
          kv = offset >> a;
          jv = offset & ((uint64_t(1) << a) - 1);
        }
        if (seg.shift || seg.bits < 64) {
          kv >>= seg.shift;
          if (seg.bits < 64)
            kv &= (uint64_t(1) << seg.bits) - 1;
        }
        value = c0;
        if (seg.step && seg.count > 1)
          value += uint64_t(seg.step) * kv;
        if (seg.inner && seg.len > 1)
          value += uint64_t(seg.inner) * jv;
      }
      got |= value & mask;
    }
    uint64_t want = idx < rows.size() ? rows[idx] & mask : 0;
    if (got != want)
      return false;
  }
  return true;
}

Value decodeSlotTable(ImplicitLocOpBuilder &b, Value index,
                      ArrayRef<uint64_t> rows, unsigned width,
                      unsigned maxSegments) {
  unsigned w = hw::getBitWidth(index.getType());
  if (w == 0 || w > 63 || width == 0 || width > 64)
    return {};
  auto segments = slotSegments(rows, width, maxSegments);
  if (segments.size() > maxSegments)
    return {};
  // The decoder's own arithmetic, at every index, against the rows: a
  // segment model or lowering slip keeps the table a constant (counted)
  // instead of building a wrong decoder.
  if (!decodesExactly(segments, rows, width, w)) {
    if (getenv("FOLDSIM_AFFINE_LOG"))
      llvm::errs() << "slot-sequencer: a decoder of " << segments.size()
                   << " segments disagrees with its table; table kept\n";
    return {};
  }
  auto &stats = slotSequencerStats();
  ++stats.decoded;
  stats.decodedBits += (uint64_t(1) << w) * width;
  stats.segments += segments.size();
  stats.maxSegments = std::max<uint64_t>(stats.maxSegments, segments.size());

  auto type = b.getIntegerType(width);
  auto indexType = index.getType();
  uint64_t wmask = width >= 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
  auto constant = [&](Type t, uint64_t v) -> Value {
    return hw::ConstantOp::create(b, t, v);
  };
  // `x` (any width) as `width` bits, shifted left by `sh`.
  auto shifted = [&](Value x, unsigned sh) -> Value {
    unsigned xw = hw::getBitWidth(x.getType());
    if (sh >= width)
      return constant(type, 0);
    unsigned keep = std::min(xw, width - sh);
    SmallVector<Value> parts;
    if (keep + sh < width)
      parts.push_back(constant(b.getIntegerType(width - keep - sh), 0));
    parts.push_back(keep == xw ? x : comb::ExtractOp::create(b, x, 0, keep));
    if (sh)
      parts.push_back(constant(b.getIntegerType(sh), 0));
    return parts.size() == 1 ? parts[0] : comb::ConcatOp::create(b, parts);
  };
  if (segments.empty())
    return constant(type, 0);
  uint64_t top = (uint64_t(1) << w) - 1;
  SmallVector<Value> hits, values;
  bool uniform = true;
  for (auto &seg : segments) {
    uint64_t end = seg.end();
    unsigned a = seg.stride > 1 ? llvm::Log2_64(seg.stride) : 0;
    uint64_t r = a ? seg.start & (seg.stride - 1) : 0;
    // The block's rows within each stride: [r, r + len) of the index's low
    // bits, unless the block wraps past the stride's boundary.
    bool wraps = seg.len > 1 && r + seg.len > seg.stride;
    unsigned le = llvm::Log2_64(seg.istride);
    SmallVector<Value> conds;
    Value offset;   // index - start, where needed
    auto offsetOf = [&]() {
      if (!offset)
        offset = seg.start ? Value(comb::SubOp::create(
                                 b, index, constant(indexType, seg.start)))
                           : index;
      return offset;
    };
    if (seg.count == 1 && seg.len == 1) {
      conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::eq, index,
                                           constant(indexType, seg.start)));
    } else {
      if (seg.start > 0)
        conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::uge, index,
                                             constant(indexType, seg.start)));
      if (end < top)
        conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::ule, index,
                                             constant(indexType, end)));
      if (a && seg.count > 1) {
        auto lowType = b.getIntegerType(a);
        if (seg.len == 1) {
          conds.push_back(comb::ICmpOp::create(
              b, comb::ICmpPredicate::eq, comb::ExtractOp::create(b, index, 0, a),
              constant(lowType, r)));
        } else if (seg.istride > 1) {
          // The rows' own low bits, then the block's [r, r + len) of the
          // bits above them.
          conds.push_back(comb::ICmpOp::create(
              b, comb::ICmpPredicate::eq, comb::ExtractOp::create(b, index, 0, le),
              constant(b.getIntegerType(le), r & (seg.istride - 1))));
          auto midType = b.getIntegerType(a - le);
          Value mid = comb::ExtractOp::create(b, index, le, a - le);
          uint64_t lo = r >> le, hiExcl = lo + seg.len;
          if (lo)
            conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::uge, mid,
                                                 constant(midType, lo)));
          if (hiExcl < (uint64_t(1) << (a - le)))
            conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::ult, mid,
                                                 constant(midType, hiExcl)));
        } else if (seg.len < seg.stride) {
          Value low = wraps ? Value(comb::ExtractOp::create(b, offsetOf(), 0, a))
                            : Value(comb::ExtractOp::create(b, index, 0, a));
          uint64_t lo = wraps ? 0 : r;
          if (lo)
            conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::uge, low,
                                                 constant(lowType, lo)));
          // (lo + len reaching the stride needs no upper bound: it is not
          // representable in the low bits.)
          if (lo + seg.len < seg.stride)
            conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::ult, low,
                                                 constant(lowType, lo + seg.len)));
        }
      }
    }
    Value hit = conds.empty() ? constant(b.getI1Type(), 1)
                              : conds.size() == 1
                                    ? conds[0]
                                    : Value(comb::AndOp::create(b, conds, false));
    hits.push_back(hit);
    bool varies = (seg.step != 0 && seg.count > 1) || (seg.inner != 0 && seg.len > 1);
    Value value = constant(type, seg.value);
    if (varies) {
      uniform = false;
      // value = v + step * f(k) + inner * j, k = (index - start) >> a and
      // j its low a bits. Where the block does not wrap, the index's own
      // bits stand for them (k less a constant, j less r): no subtraction,
      // one carry chain less; a digit of k needs the start's low digits 0.
      uint64_t sa = seg.start >> a;
      bool plainK = seg.bits >= 64 && seg.shift == 0;
      bool direct = !wraps && (plainK || (seg.shift + seg.bits < 64 &&
                                          (sa & ((uint64_t(1) << (seg.shift + seg.bits)) - 1)) == 0));
      uint64_t c0 = seg.value;
      Value kv, jv;
      if (direct) {
        if (plainK)
          c0 -= uint64_t(seg.step) * sa;
        c0 -= uint64_t(seg.inner) * (r >> le);
        if (a < w)
          kv = comb::ExtractOp::create(b, index, a, w - a);
        if (a > le)
          jv = comb::ExtractOp::create(b, index, le, a - le);
      } else {
        Value x = offsetOf();
        if (a < w)
          kv = comb::ExtractOp::create(b, x, a, w - a);
        if (a)
          jv = comb::ExtractOp::create(b, x, 0, a);
      }
      // f(k): the digit [shift, shift + bits) of k.
      if (kv && (seg.shift || seg.bits < 64)) {
        unsigned kw = hw::getBitWidth(kv.getType());
        if (seg.shift >= kw)
          kv = Value();
        else
          kv = comb::ExtractOp::create(b, kv, seg.shift,
                                       std::min<unsigned>(kw - seg.shift, seg.bits));
      }
      SmallVector<Value> plus, minus;
      SmallVector<std::pair<bool, unsigned>> terms;
      if (kv && seg.step && seg.count > 1 &&
          csdTerms(uint64_t(seg.step), width, kMaxStepTerms, terms))
        for (auto [neg, sh] : terms)
          (neg ? minus : plus).push_back(shifted(kv, sh));
      if (jv && seg.inner && seg.len > 1) {
        uint64_t mag = seg.inner < 0 ? uint64_t(-seg.inner) : uint64_t(seg.inner);
        (seg.inner < 0 ? minus : plus).push_back(shifted(jv, llvm::Log2_64(mag)));
      }
      // One sum of every term (a subtracted term as its complement, its +1s
      // in the constant): a single adder tree, not a chain of carry chains.
      SmallVector<Value> terms2(plus.begin(), plus.end());
      for (auto v : minus)
        terms2.push_back(comb::XorOp::create(b, v, constant(type, wmask)));
      uint64_t c = (c0 + uint64_t(minus.size())) & wmask;
      if (c || terms2.empty())
        terms2.push_back(constant(type, c));
      value = terms2.size() == 1 ? terms2[0]
                                 : Value(comb::AddOp::create(b, terms2, false));
    } else if (seg.value != segments[0].value) {
      uniform = false;
    }
    values.push_back(value);
  }
  if (uniform) {
    Value any = hits.size() == 1 ? hits[0]
                                 : Value(comb::OrOp::create(b, hits, false));
    if (width == 1 && segments[0].value == 1)
      return any;
    return comb::MuxOp::create(b, any, constant(type, segments[0].value),
                               constant(type, 0));
  }
  SmallVector<Value> masked;
  for (auto [hit, value] : llvm::zip(hits, values))
    masked.push_back(comb::MuxOp::create(b, hit, value, constant(type, 0)));
  return masked.size() == 1 ? masked[0]
                            : Value(comb::OrOp::create(b, masked, false));
}

void countFallback(unsigned indexWidth, unsigned width, unsigned segments) {
  auto &stats = slotSequencerStats();
  ++stats.fallback;
  stats.fallbackBits += (uint64_t(1) << indexWidth) * width;
  stats.fallbackMaxSegments =
      std::max<uint64_t>(stats.fallbackMaxSegments, segments);
}

} // namespace foldsim
