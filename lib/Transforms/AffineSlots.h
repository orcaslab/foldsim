//===- AffineSlots.h - Affine placement of a modulo schedule --------------===//
//
// foldsim-fold-static's `affine-slots`: each group of evaluations (a copy's or
// ready lane's passes of one kind, over its instances) is placed as one
// affine run, the k-th instance of the group at o + k*P, P a power of two per
// resource, so a class's slot tables are a segment per pass (slot-segments)
// whatever the instances and the frame: a fixed-size sequencer. The frame
// (II) grows with the instances instead. See AffineSlots.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_AFFINESLOTS_H
#define FOLDSIM_AFFINESLOTS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <utility>

namespace foldsim {

/// An evaluation to place: its group, its rank in the group (by instance),
/// and the earliest slot it may take.
struct AffineNode {
  unsigned group = 0, index = 0;
  int earliest = 0;
};

/// A group: the resource it runs on (its copy or ready lane), its
/// evaluations, and the least stride its passes allow (a page, a link page, a
/// serialized link's k).
struct AffineGroup {
  unsigned resource = 0, count = 0, minStride = 1;
};

/// The groups' strides: per resource, the power of two at least its groups'
/// count and their least strides (so a resource's groups interleave).
llvm::SmallVector<unsigned> affineStrides(llvm::ArrayRef<AffineGroup> groups);

/// Places every group at o_g (node at o_g + index * stride_g), each node at
/// or after its predecessors' slots plus the latency (`preds[n]`: (p,
/// latency)) and its earliest, within two frames of `period`; `fits(n, t)`
/// and `take(n, t, on)` test and take a node's resources (taken in order, so
/// one group's own nodes see each other). Fills `slotOf` and returns true, or
/// returns false with the reason in `why` (the frame is then too short).
bool placeAffine(
    unsigned period, llvm::ArrayRef<AffineNode> nodes,
    llvm::ArrayRef<AffineGroup> groups, llvm::ArrayRef<unsigned> strides,
    llvm::ArrayRef<llvm::SmallVector<std::pair<unsigned, int>>> preds,
    llvm::function_ref<bool(unsigned, int)> fits,
    llvm::function_ref<void(unsigned, int, bool)> take,
    llvm::SmallVectorImpl<int> &slotOf, std::string &why,
    llvm::function_ref<std::string(unsigned)> groupName = nullptr);

} // namespace foldsim

#endif // FOLDSIM_AFFINESLOTS_H
