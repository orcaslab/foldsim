//===- VmemRings.h - Fixed read rings for virtual memories -----*- C++ -*-===//
//
// foldsim-fold-static's `vmem-rings`: a virtual memory's per-instance read
// buffer (`<memory>_rdata`, an entry per instance) becomes a fixed ring of R
// entries, R set by the schedule's window and not by the instance count.
//
// The read data an instance's evaluations of frame f+1 use is the answer to
// the read its state pass of frame f issued. That answer is written, as it
// comes, to the instance's row of an image in device memory (a second
// virtual memory, a row per instance). A lead before the instance's first
// use of frame f+1, a prefetch reads that row into entry `instance mod R` of
// the ring, and every use reads the ring there. A read held by the target
// (its enable low) issues nothing: the image keeps the instance's last read
// data, as the target's read register does.
//
// Exact by construction, given the controllers' order (a read observes every
// earlier write of its row):
// - the prefetch of an instance is decided only once its read is answered
//   (its pending bit clear) and so its image row written, at least a decision
//   lead after the state pass that issued it;
// - a use is decided only once its entry has landed for its instance (a
//   landed bit and the instance's high bits per entry, the bit set by the
//   answer, cleared by the instance's last use of the frame);
// - two instances share an entry only if, frame after frame, the later's
//   prefetch comes after the earlier's last use (checked over the schedule).
//
// This file plans the positions: the prefetch of each instance and the
// smallest power-of-two R. The hardware is StaticFold.cpp's.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_VMEMRINGS_H
#define FOLDSIM_VMEMRINGS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>

namespace foldsim {

/// One instance's positions in the frame of its state pass (frame f at 0):
/// `issue` its state pass (the read's issue), `uses` the read positions of
/// its evaluations of frame f+1 (each its pass's slot less the stages, plus
/// the frame's II).
struct RingInstance {
  int issue = 0;
  llvm::SmallVector<int> uses;
};

/// The ring of one class: its entries, each instance's prefetch position
/// (in frame f's coordinates, before its first use), and the tables a
/// position reads (ii + 1 rows, the last a bubble's).
struct RingPlan {
  bool ok = false;
  std::string why;
  unsigned entries = 0;
  llvm::SmallVector<int> prefetch;
  /// Per position: a prefetch there and its instance; a use there, its
  /// instance, and whether its prefetch is a frame before the use's (a use
  /// of the first frame run then reads zero: nothing was prefetched).
  llvm::SmallVector<uint64_t> prefValid, prefInst, useValid, useInst, useCross;
  /// Per position: the use there is its instance's last of the frame (its
  /// entry's landed bit clears with it).
  llvm::SmallVector<uint64_t> useLast;
  /// The most entries in use at once (prefetched, not yet used last).
  unsigned window = 0;
};

/// Plans the ring of a class of `instances.size()` instances in a frame of
/// `ii` positions: each prefetch `lead` positions before the first use if it
/// can, at least `minGap` positions after the issue (the pending bit set by
/// the issue seen by the prefetch's decision) and at least one before the
/// first use; one prefetch a position; entries the smallest power of two
/// that no two live instances share, at most `maxEntries`.
RingPlan planRing(llvm::ArrayRef<RingInstance> instances, unsigned ii,
                  unsigned lead, unsigned minGap, unsigned maxEntries);

} // namespace foldsim

#endif // FOLDSIM_VMEMRINGS_H
