//===- SlotSequencer.h - Slot tables as fixed-size decoders ---------------===//
//
// foldsim-fold-static's `slot-segments`: a slot table (a constant per index
// of the frame's position, or of an instance) built as arithmetic on the
// index instead of a constant array: the table's nonzero rows cut into affine
// segments (index s + j*ds, value v + j*dv, j < n; ds and dv powers of two or
// zero), each a range compare, a residue compare and a shift-add, their
// values ORed. Its size depends on the segments, not on the frame's length or
// the instances: with an affine schedule (`affine-slots`) a class's rows are a
// segment per pass. See SlotSequencer.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_SLOTSEQUENCER_H
#define FOLDSIM_SLOTSEQUENCER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace foldsim {

/// A run of a table's rows, in blocks: rows[start + k * stride + j * istride] =
/// value + step * f(k) + inner * j (modulo 2^width) for k < count, j < len;
/// f(k) the `bits` bits of k from bit `shift` (k itself by default: an affine
/// run; else a digit of k, as an instance's place among its parent's
/// children, or its parent). `len` 1 is a plain run; a block of len > 1 is a
/// copy's consecutive passes (a PP evaluation's rows), or the instances of
/// one parent less the ones placed elsewhere (`rank with holes`: 3k + j).
/// The stride is a power of two (or 0 for a single row); `step` any constant
/// of at most three signed powers of two; `inner` 0 or a signed power of two.
struct SlotSegment {
  uint64_t start = 0, stride = 0, count = 0, len = 1;
  /// A block's rows `istride` apart (a power of two; len * istride <= stride).
  uint64_t istride = 1;
  uint64_t value = 0;
  int64_t step = 0, inner = 0;
  unsigned shift = 0, bits = 64;
  uint64_t at(uint64_t k, uint64_t j = 0) const {
    uint64_t f = k >> shift;
    if (bits < 64)
      f &= (uint64_t(1) << bits) - 1;
    return value + uint64_t(step) * f + uint64_t(inner) * j;
  }
  uint64_t end() const {
    return start + (count - 1) * stride + (len - 1) * istride;
  }
};

/// `x` (`width` bits) as at most `maxTerms` signed powers of two (canonical
/// signed digits), as (negative, shift) pairs; false if it takes more.
bool csdTerms(uint64_t x, unsigned width, unsigned maxTerms,
              llvm::SmallVectorImpl<std::pair<bool, unsigned>> &terms);

/// The table's nonzero rows (indices past `rows` are zero) as segments a
/// decoder takes: strides and steps zero or powers of two (a step may be
/// negative), runs of one stride interleaving with others; each run the
/// longest from the lowest row left (a run may take zero rows its pattern
/// gives). `width` is the rows' width. Stops past `maxSegments`.
llvm::SmallVector<SlotSegment> slotSegments(llvm::ArrayRef<uint64_t> rows,
                                            unsigned width,
                                            unsigned maxSegments = 64);

/// Counts over a pass's run: tables decoded and left as constants, and their
/// bits as constant arrays (2^index width x width).
struct SlotSequencerStats {
  uint64_t decoded = 0, decodedBits = 0, segments = 0, maxSegments = 0;
  uint64_t fallback = 0, fallbackBits = 0, fallbackMaxSegments = 0;
  void reset() { *this = SlotSequencerStats(); }
};
SlotSequencerStats &slotSequencerStats();

/// `rows[index]` (`width` bits, zero past the rows) as a decoder of at most
/// `maxSegments` segments, or null if the rows need more (the caller then
/// builds the constant table, and counts it with `countFallback`).
mlir::Value decodeSlotTable(mlir::ImplicitLocOpBuilder &b, mlir::Value index,
                            llvm::ArrayRef<uint64_t> rows, unsigned width,
                            unsigned maxSegments);
void countFallback(unsigned indexWidth, unsigned width, unsigned segments);

} // namespace foldsim

#endif // FOLDSIM_SLOTSEQUENCER_H
