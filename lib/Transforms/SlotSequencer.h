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
#include "llvm/ADT/STLFunctionalExtras.h"
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
                                            unsigned maxSegments = 64,
                                            bool anyStride = false,
                                            bool wrapAny = false,
                                            llvm::ArrayRef<uint8_t> care = {});

/// A class's pass sequence: instance i's pass at position `slots[i]` (mod
/// the frame), as `instAt` (the instance at each position, -1 for none).
struct PassSequence {
  llvm::SmallVector<int> instAt;
  unsigned instances = 0;
};

/// A position table (rows 0..ii, the last the idle position) through a pass
/// sequence: rows[p] = (a pass of `seq` at p + offset) ? perInstance[its
/// instance] : 0, for every p. Found by trying the offsets that put the
/// table's first live rows on the sequence's passes, then checking every row
/// (each instance one value).
struct SequenceFactor {
  unsigned seq = 0, offset = 0;
  llvm::SmallVector<uint64_t> perInstance;
};
bool factorThroughSequence(llvm::ArrayRef<uint64_t> rows, unsigned width,
                           unsigned ii, llvm::ArrayRef<PassSequence> seqs,
                           SequenceFactor &out, llvm::ArrayRef<uint8_t> care = {});

/// Counts over a pass's run: tables decoded and left as constants, and their
/// bits as constant arrays (2^index width x width).
struct SlotSequencerStats {
  uint64_t decoded = 0, decodedBits = 0, segments = 0, maxSegments = 0;
  uint64_t fallback = 0, fallbackBits = 0, fallbackMaxSegments = 0;
  /// Position tables read through a class's pass sequence.
  uint64_t factored = 0, factoredBits = 0;
  void reset() { *this = SlotSequencerStats(); }
};
SlotSequencerStats &slotSequencerStats();

/// `rows[index]` (`width` bits, zero past the rows) as a decoder of at most
/// `maxSegments` segments, or null if the rows need more (the caller then
/// builds the constant table, and counts it with `countFallback`).
/// `care` (empty: every row): the rows whose value is read; elsewhere (a
/// valid bit beside the table low there) any value may come.
/// `divmod`, where the index is the frame's position: its quotient and
/// remainder of a period (PosCounters), for runs whose stride is not a power
/// of two.
mlir::Value decodeSlotTable(
    mlir::ImplicitLocOpBuilder &b, mlir::Value index,
    llvm::ArrayRef<uint64_t> rows, unsigned width, unsigned maxSegments,
    llvm::function_ref<std::pair<mlir::Value, mlir::Value>(unsigned)> divmod =
        nullptr,
    llvm::ArrayRef<uint8_t> care = {});
void countFallback(unsigned indexWidth, unsigned width, unsigned segments);

/// A table separable in a period D of its index: rows[k] = F(q) op G(r) for
/// k in [start, end), q and r the quotient and remainder of k - start by D,
/// op XOR or a sum (mod 2^width), zero elsewhere. F is affine (a segment or
/// two: the parent's number), G a table of D rows (a child's constant): a
/// hierarchical identifier (a parent's id with a child's constant joined), as
/// large at any count of parents.
struct Separable {
  uint64_t start = 0, end = 0, period = 0;
  bool isXor = true;
  llvm::SmallVector<uint64_t> f, g;
};
/// The smallest such period (powers of two up to `maxPeriod`, any period with
/// `anyPeriod`), or false.
bool findSeparable(llvm::ArrayRef<uint64_t> rows, unsigned width,
                   bool anyPeriod, unsigned maxPeriod, Separable &out);

/// A table cut into column-affine pieces: on [start, end), with q and r the
/// quotient and remainder of (k - start) by `period`, rows[k] = a[r] +
/// s[r] * q (mod 2^width): a run of a parent's children whose constants
/// (a child's place, a link's source among them) step by a few slopes from
/// one parent to the next. Instance tables of a class nested `period`
/// children a parent (a switch network's stages, its ports) are a piece or
/// two whatever the parents' count. `residual`: the rows outside the pieces
/// (zero inside them), for the segments.
struct ColumnAffine {
  struct Piece {
    uint64_t start = 0, end = 0, period = 0;
    llvm::SmallVector<uint64_t> a, s;
  };
  llvm::SmallVector<Piece> pieces;
  llvm::SmallVector<uint64_t> residual;
};
/// The pieces, greedily the longest from the lowest row left: periods 2 to
/// `maxPeriod`, each piece at least 4 periods long, at most `maxSlopes`
/// distinct nonzero slopes each of at most three signed powers of two; at
/// most `maxPieces` pieces, covering at least half the nonzero rows (false
/// otherwise).
bool findColumnAffine(llvm::ArrayRef<uint64_t> rows, unsigned width,
                      unsigned maxPeriod, unsigned maxPieces,
                      unsigned maxSlopes, ColumnAffine &out);

} // namespace foldsim

#endif // FOLDSIM_SLOTSEQUENCER_H
