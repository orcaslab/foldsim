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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace foldsim {

struct PassSplitStats {
  unsigned groups = 0, replicas = 0, lanes = 0;
  /// Estimated LUTs of the replicas' logic before, and of the lanes and
  /// their choices after.
  double before = 0, after = 0;
  uint64_t holdBits = 0, selectBits = 0, captureBits = 0;
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

/// As foldLanes; with `apply` null, nothing is built: the groups are only
/// found and priced, and their roots appended to `roots`.
PassSplitStats findLanes(mlir::ImplicitLocOpBuilder *apply,
                         llvm::ArrayRef<mlir::Operation *> logic,
                         mlir::Value hostClock, mlir::Value capNow,
                         unsigned passes, unsigned span, llvm::StringRef prefix,
                         llvm::SmallVectorImpl<mlir::Value> *roots);

/// Deferred lanes: the registers whose next state is written late, and the
/// logic only their next values use (their exclusive cone).
struct DeferPlan {
  llvm::SmallVector<mlir::Operation *> regs;
  llvm::SmallVector<mlir::Operation *> logic;
  PassSplitStats stats;
};

/// Plans deferred lanes in `body` (a model's, or a copy's before its stores
/// are built): the lanes of the logic only registers' next values use, and
/// the registers they reach. `pinned`: values taken as used outside that
/// logic too (a model's registers' next values an early output's next value
/// reads, before buildCopy makes that read explicit).
DeferPlan planDeferred(mlir::Block *body, llvm::ArrayRef<mlir::Value> pinned,
                       unsigned passes);

/// Builds a plan: the logic's leaves taken into capture registers
/// (`<prefix>cap<n>`) where `stateNow` is high (the state pass), its lanes
/// folded over `passes` passes of `span` host cycles from there, and each
/// planned register's next value (its reset applied) returned in `next`,
/// valid (passes + 1) x span host cycles after the state pass and held until
/// the next one. The caller writes the registers then.
PassSplitStats applyDeferred(mlir::ImplicitLocOpBuilder &b, DeferPlan &plan,
                             mlir::Value hostClock, mlir::Value stateNow,
                             unsigned passes, unsigned span,
                             llvm::StringRef prefix,
                             llvm::DenseMap<mlir::Operation *, mlir::Value> &next);

} // namespace foldsim

#endif // FOLDSIM_PASSSPLIT_H
