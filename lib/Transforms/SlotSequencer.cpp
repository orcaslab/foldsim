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

namespace {
/// `x` (a `width`-bit value) as a power-of-two step: 0, +2^m or -2^m.
bool stepOf(uint64_t x, unsigned width, int64_t &step) {
  uint64_t mask = width >= 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
  x &= mask;
  if (x == 0) {
    step = 0;
    return true;
  }
  if (llvm::isPowerOf2_64(x)) {
    step = int64_t(x);
    return true;
  }
  uint64_t neg = (~x + 1) & mask;
  if (llvm::isPowerOf2_64(neg)) {
    step = -int64_t(neg);
    return true;
  }
  return false;
}
} // namespace

SmallVector<SlotSegment> slotSegments(ArrayRef<uint64_t> rows, unsigned width,
                                      unsigned maxSegments) {
  uint64_t mask = width >= 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
  size_t n = rows.size();
  auto row = [&](uint64_t k) { return k < n ? rows[k] & mask : 0; };
  std::vector<bool> covered(n, false);
  SmallVector<SlotSegment> segments;
  // The run of `seg` (start, stride, value, step, shift, bits set) from its
  // start: rows not covered, each its pattern's value; a zero row only
  // where the pattern gives zero, and never the run's last.
  auto extend = [&](SlotSegment &seg) {
    uint64_t j = 0, last = 0;
    for (uint64_t k = seg.start; k < n; k += seg.stride, ++j) {
      if (covered[k] || row(k) != (seg.at(j) & mask))
        break;
      if (row(k))
        last = j + 1;
      if (!seg.stride)
        break;
    }
    seg.count = last;
  };
  for (size_t k0 = 0; k0 < n; ++k0) {
    if (covered[k0] || !row(k0))
      continue;
    SlotSegment best;
    best.start = k0;
    best.value = row(k0);
    best.count = 1;
    for (uint64_t stride = 1; stride < n - k0; stride *= 2) {
      // Plain affine: the step from the next row of the stride.
      SlotSegment seg;
      seg.start = k0;
      seg.stride = stride;
      seg.value = row(k0);
      int64_t step;
      if (stepOf(row(k0 + stride) - row(k0), width, step) &&
          !(k0 + stride < n && covered[k0 + stride])) {
        seg.step = step;
        extend(seg);
        if (seg.count > best.count)
          best = seg;
      }
      // A digit of j: bits [shift, shift + bits) of j, the step from the
      // first j whose digit is 1 (stride 1, where no affine run is long).
      if (stride == 1 && best.count < 64)
      for (unsigned shift = 0; shift < 10; ++shift)
        for (unsigned bits = 1; bits <= 10; ++bits) {
          uint64_t j1 = uint64_t(1) << shift;
          uint64_t k1 = k0 + j1 * stride;
          if (k1 >= n || covered[k1])
            break;
          SlotSegment f = seg;
          f.shift = shift;
          f.bits = bits;
          if (!stepOf(row(k1) - row(k0), width, step) || !step)
            break;
          f.step = step;
          extend(f);
          if (f.count > best.count)
            best = f;
        }
    }
    if (best.count == 1)
      best.stride = 0;
    for (uint64_t j = 0; j < best.count; ++j)
      covered[best.start + j * best.stride] = true;
    segments.push_back(best);
    if (segments.size() > maxSegments)
      break;
  }
  return segments;
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
  auto &stats = slotSequencerStats();
  ++stats.decoded;
  stats.decodedBits += (uint64_t(1) << w) * width;
  stats.segments += segments.size();
  stats.maxSegments = std::max<uint64_t>(stats.maxSegments, segments.size());

  auto type = b.getIntegerType(width);
  auto indexType = index.getType();
  auto constant = [&](Type t, uint64_t v) -> Value {
    return hw::ConstantOp::create(b, t, v);
  };
  if (segments.empty())
    return constant(type, 0);
  uint64_t top = (uint64_t(1) << w) - 1;
  SmallVector<Value> hits, values;
  bool uniform = true;
  for (auto &seg : segments) {
    uint64_t end = seg.start + (seg.count - 1) * seg.stride;
    SmallVector<Value> conds;
    if (seg.count == 1) {
      conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::eq, index,
                                           constant(indexType, seg.start)));
    } else {
      if (seg.start > 0)
        conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::uge, index,
                                             constant(indexType, seg.start)));
      if (end < top)
        conds.push_back(comb::ICmpOp::create(b, comb::ICmpPredicate::ule, index,
                                             constant(indexType, end)));
      if (seg.stride > 1) {
        unsigned low = llvm::Log2_64(seg.stride);
        conds.push_back(comb::ICmpOp::create(
            b, comb::ICmpPredicate::eq,
            comb::ExtractOp::create(b, index, 0, low),
            constant(b.getIntegerType(low), seg.start & (seg.stride - 1))));
      }
    }
    Value hit = conds.empty() ? constant(b.getI1Type(), 1)
                              : conds.size() == 1
                                    ? conds[0]
                                    : Value(comb::AndOp::create(b, conds, false));
    hits.push_back(hit);
    Value value = constant(type, seg.value);
    if (seg.step != 0 && seg.count > 1) {
      uniform = false;
      // j's digit: bits [low + seg.shift, + seg.bits) of index - start.
      unsigned low = (seg.stride > 1 ? llvm::Log2_64(seg.stride) : 0) + seg.shift;
      uint64_t mag = seg.step < 0 ? uint64_t(-seg.step) : uint64_t(seg.step);
      unsigned shift = llvm::Log2_64(mag);
      // f = (index - start) >> low, of its bits; (f << shift) in `width`.
      // Where the start's own digits allow, the index's bits stand for j's
      // (no subtraction, one carry chain less): for an affine run, j's
      // multiple is the index's less a constant; for a digit, when the
      // start's low digits are zero, j's digit is the index's.
      uint64_t a = seg.stride > 1 ? llvm::Log2_64(seg.stride) : 0;
      uint64_t sa = seg.start >> a;
      Value offset = index;
      if (seg.bits >= 64 && seg.shift == 0) {
        value = constant(type, seg.value - uint64_t(seg.step) * (sa << seg.shift));
      } else if (seg.shift + seg.bits < 64 &&
                 (sa & ((uint64_t(1) << (seg.shift + seg.bits)) - 1)) == 0) {
        // the index's digit is j's
      } else if (seg.start) {
        offset = comb::SubOp::create(b, index, constant(indexType, seg.start));
      }
      if (shift < width && low < w) {
        unsigned jBits = std::min<unsigned>(w - low, seg.bits),
                 keep = std::min(jBits, width - shift);
        Value j = comb::ExtractOp::create(b, offset, low, keep);
        SmallVector<Value> parts;
        if (keep + shift < width)
          parts.push_back(constant(b.getIntegerType(width - keep - shift), 0));
        parts.push_back(j);
        if (shift)
          parts.push_back(constant(b.getIntegerType(shift), 0));
        Value delta = parts.size() == 1 ? parts[0]
                                        : Value(comb::ConcatOp::create(b, parts));
        value = seg.step > 0
                    ? Value(comb::AddOp::create(b, value, delta))
                    : Value(comb::SubOp::create(b, value, delta));
      }
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
