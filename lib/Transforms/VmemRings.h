//===- VmemRings.h - Fixed read rings for virtual memories -----*- C++ -*-===//
//
// foldsim-fold-static's `vmem-rings`: a virtual memory's per-instance read
// buffer (`<memory>_rdata`, an entry per instance) becomes a fixed ring of R
// entries, R set by the schedule's window and not by the instance count.
//
// The read data an instance's evaluations of frame f+1 use is the answer to
// the read its state pass of frame f issued. That answer is written, as it
// comes, to the instance's row of an image in device memory (a second
// virtual memory, a row per instance). The uses of frame f+1 fall in groups
// (uses less than a lead apart); a lead before each group's first use, a
// prefetch reads the instance's row into an entry of the ring (the
// prefetch's rank in the frame's prefetches, mod R; the request's tag), and
// the group's uses read the ring there. A read held by the target (its enable
// low) issues nothing: the image keeps the instance's last read data, as the
// target's read register does.
//
// Exact by construction, given the controllers' order (a read observes every
// earlier write of its row):
// - a prefetch is decided only once its instance's read is answered (its
//   pending bit clear) and so its image row written, at least a decision lead
//   after the state pass that issued it;
// - a prefetch is queued at its position (the entry's landed bit cleared then)
//   and leaves the queue once its instance has no read in flight (a fixed table
//   of the reads in flight), so its decision waits for nothing;
// - a use is decided only once its entry has landed: the bit is cleared by the
//   entry's prefetch, queued at least `useGap` positions before the group's
//   first use (so before that use's decision), and set by the prefetch's answer
//   (the entry's previous group answered before its own uses);
// - two prefetches share an entry only if, frame after frame, the later one
//   comes after the earlier one's last use (checked over the schedule).
//
// This file plans the positions: the prefetches, their entries, and the fewest
// entries R. The hardware is StaticFold.cpp's.
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

/// The ring of one class: its entries, its prefetches (a use group each), and
/// the tables a position reads (ii + 1 rows, the last a bubble's).
struct RingPlan {
  bool ok = false;
  std::string why;
  unsigned entries = 0, prefetches = 0;
  /// Per position: a prefetch there, its instance (the image's row) and its
  /// entry; a use there, its entry, its prefetch's position, whether the
  /// prefetch is a frame before the use (a use of the first frame run then
  /// reads zero: nothing was prefetched), and whether it is its group's last.
  llvm::SmallVector<uint64_t> prefValid, prefInst, prefEntry;
  llvm::SmallVector<uint64_t> useValid, useEntry, useOwner, useCross, useLast;
  /// The most entries in use at once (prefetched, not yet used last).
  unsigned window = 0;
};

/// Plans the ring of a class of `instances.size()` instances in a frame of
/// `ii` positions: an instance's uses less than `lead` positions apart share a
/// prefetch, each prefetch `lead` positions before its group's first use if it
/// can, at least `minGap` positions after the issue (the read in flight by the
/// time the prefetch is queued) and at least `useGap` before the first use (the
/// entry's landed bit cleared before that use's decision); one prefetch a
/// position; a prefetch's entry its rank in the frame's
/// prefetches mod R, R the smallest power of two (at least the window) with
/// which no two live prefetches share an entry, at most `maxEntries`.
RingPlan planRing(llvm::ArrayRef<RingInstance> instances, unsigned ii,
                  unsigned lead, unsigned minGap, unsigned useGap,
                  unsigned maxEntries);

} // namespace foldsim

#endif // FOLDSIM_VMEMRINGS_H
