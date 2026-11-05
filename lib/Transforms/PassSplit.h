//===- PassSplit.h - Lanes of a multicycle copy over passes ---------------===//
//
// Pass splitting (foldsim-fold-static's `pass-split`): a multicycle copy's
// logic, between its capture and end registers, evaluated over P passes, the
// isomorphic fanout-free cones of it (lanes) one replica a pass on shared
// logic. See PassSplit.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_PASSSPLIT_H
#define FOLDSIM_PASSSPLIT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace foldsim {

struct PassSplitStats {
  unsigned groups = 0, replicas = 0, lanes = 0;
  /// Estimated LUTs of the replicas' logic before, and of the lanes and
  /// their choices after.
  double before = 0, after = 0;
  uint64_t holdBits = 0, selectBits = 0;
  /// Each folded group's replicas.
  llvm::SmallVector<unsigned> sizes;
};

/// Folds the lanes of `logic` (the copy's ops between its capture and end
/// registers, every operand of theirs outside the set taken as fixed for the
/// evaluation) over `passes` passes of `span` host cycles each, counted from
/// `capNow` (high in the host cycle whose edge loads the captures). A
/// replica's result is held from its pass on in a register (`<prefix>hold<n>`)
/// that replaces it; the end registers must be taken at least
/// (passes + 1) x span host cycles after the captures.
PassSplitStats foldLanes(mlir::ImplicitLocOpBuilder &b,
                         llvm::ArrayRef<mlir::Operation *> logic,
                         mlir::Value hostClock, mlir::Value capNow,
                         unsigned passes, unsigned span, llvm::StringRef prefix);

} // namespace foldsim

#endif // FOLDSIM_PASSSPLIT_H
