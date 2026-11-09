//===- CsfReadySplit.h - Ready-cone lanes of the constant-shell fold ------===//
//
// foldsim-fold-static's `csf-ready-cones`: a model's late outputs (those its
// inputs reach through logic) split into a stateless ready-cone model of their
// own, fed the registers they read through one output of the model, so that
// its copies (the ready lanes) evaluate them apart from the model's copy. See
// CsfReadySplit.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_CSFREADYSPLIT_H
#define FOLDSIM_CSFREADYSPLIT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>

namespace foldsim {

/// What the split of one model did.
struct ReadySplit {
  std::string model, cone;
  /// The late outputs moved to the cone model, and those kept (a memory read
  /// or a black box in their cone); the registers the cone reads, and their
  /// bits; the cone's ops.
  unsigned moved = 0, kept = 0, regs = 0, ops = 0;
  uint64_t regBits = 0;
};

/// The name of the output that carries a split model's ready registers to its
/// cone model, and of the cone model's input that takes it.
inline constexpr llvm::StringLiteral kReadyRegsPort = "csf_ready_regs";

/// Splits each model of `models` (a module of `top`, instantiated anywhere in
/// it): its late outputs whose cones are logic alone (combinational and
/// aggregate ops, registers and inputs) go to `<model>_rc`, a stateless
/// model of the inputs they read and of `csf_ready_regs`, the registers they
/// read concatenated (the first in the low bits). The model loses those
/// outputs and gains `csf_ready_regs`; next to each instance of it an
/// instance of the cone model (`<name>_rc`) takes its inputs and that output,
/// and drives what its late outputs drove. A model with no such output is
/// left as it is (`moved` 0).
mlir::LogicalResult splitReadyCones(mlir::ModuleOp top,
                                    llvm::ArrayRef<std::string> models,
                                    llvm::SmallVectorImpl<ReadySplit> &done);

} // namespace foldsim

#endif // FOLDSIM_CSFREADYSPLIT_H
