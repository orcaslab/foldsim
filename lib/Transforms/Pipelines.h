//===- Pipelines.h - fixed-latency pipelines of a model ---------*- C++ -*-===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Finds the parts of a flat model that are fixed-latency pipelines: registers
// (and black boxes with a pipeline contract) whose outputs depend only on
// their inputs of exactly L enabled cycles earlier. Such a pipeline needs no
// threaded state of its own: `foldsim-fame5` with `fold-pipelines` simulates
// it with one unthreaded copy and a per-thread store of its results.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_LIB_TRANSFORMS_PIPELINES_H
#define FOLDSIM_LIB_TRANSFORMS_PIPELINES_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/BitVector.h"
#include <memory>
#include <vector>
#include <optional>
#include <string>

namespace foldsim {

/// What a black box declares about itself to be simulated as a pipeline, in
/// the `sv.attributes` of its module (the Verilog `(* ... *)` attributes):
///
///   (* black_box, fixed_latency = 16, clock = "aclk", clock_enable = "aclken",
///      data_in = "a b", data_out = "r flags", tie = "a_valid=1",
///      pure_pipeline *)
///
/// Its `data_out` outputs are a function of its `data_in` inputs of exactly
/// `fixed_latency` cycles of `clock` with `clock_enable` high earlier, and of
/// nothing else (`pure_pipeline`: no other state); the `tie` inputs must be driven with
/// those constants, and its other outputs must be unused.
struct PipelineContract {
  unsigned latency = 0;
  std::string clock, enable;
  llvm::SmallVector<std::string> dataIn, dataOut;
  llvm::SmallVector<std::pair<std::string, int64_t>> tied;
};

/// The contract of the module `module` (an `hw.module` or `hw.module.extern`),
/// none if it declares none, or failure, with an error, if it declares one
/// that is incomplete.
mlir::FailureOr<std::optional<PipelineContract>>
getPipelineContract(mlir::Operation *module);

/// Whether the module `module` declares itself a black box.
bool isBlackBox(mlir::Operation *module);

/// A register some bits of which are in a pipeline.
struct PipelineRegister {
  circt::seq::FirRegOp reg;
  /// Its write: data when enabled (always, if null).
  mlir::Value enable, data;
  /// The layer of each bit, from 1; 0 for a bit not in a pipeline.
  llvm::SmallVector<unsigned> layer;
  /// The pipeline of each bit in one, as an index of the analysis'.
  llvm::SmallVector<unsigned> pipeline;
};

/// A black box with a pipeline contract in a pipeline.
struct PipelineBox {
  circt::hw::InstanceOp inst;
  PipelineContract contract;
  /// The layer of its inputs (0: from outside) and of its outputs.
  unsigned inLayer = 0, outLayer = 0;
};

/// A pipeline: one or more connected sets of registers and black boxes, all
/// written on the same `enable`, each bit at a layer from 1 to `layers`. Its
/// inputs are the data of layer 1 (and of black boxes at layer 0); a bit of
/// layer k > 1 is computed from bits of layer k - 1 only, and only bits of the
/// last layer are read outside.
///
/// Each register of a gated pipeline is written on `enable` and, if gated, the
/// bit of a valid chain of its layer: a chain of bits that, written on
/// `enable`, each takes the one before or is cleared, the first being the gate
/// of layer 1. Its last layer stays target state, and only bits of the last
/// layer but one feed it.
struct Pipeline {
  /// Written on this enable; on every target edge if null.
  mlir::Value enable;
  unsigned layers = 0;
  /// The values the per-thread store holds: the last layer's bits if
  /// `lastLayerStored` (a black box in the last layer, read from the store),
  /// else the next values of the last layer's registers, which stay.
  bool lastLayerStored = false;
  bool gated = false;
  /// The registers and black boxes, as indices of the analysis'.
  llvm::SmallVector<unsigned> registers, boxes;
  /// Store slots per thread: the layers simulated by the unthreaded copy.
  unsigned depth() const { return lastLayerStored ? layers : layers - 1; }
  /// Bits per layer (index 0 is layer 1), of inputs and of a stored entry.
  llvm::SmallVector<uint64_t> layerBits;
  uint64_t inputBits = 0, storeBits = 0;
  /// The bits of state the unthreaded copy takes out of the threaded model.
  uint64_t foldedBits = 0;
  /// Where it came from: the common hierarchical prefix of each connected set.
  llvm::SmallVector<std::string> paths;
};

/// Finds the pipelines of the flat module `model`. Peels the registers' holds
/// with `b` (at the end of the module's body).
class PipelineAnalysis {
public:
  explicit PipelineAnalysis(circt::hw::HWModuleOp model);
  ~PipelineAnalysis();
  /// Fails, with an error, on an instance of a black box with a malformed or
  /// unmet contract.
  mlir::LogicalResult run(mlir::ImplicitLocOpBuilder &b);

  std::vector<PipelineRegister> registers;
  std::vector<PipelineBox> boxes;
  std::vector<Pipeline> pipelines;
  /// Why candidates were left out, with how many bits: reason -> bits.
  llvm::SmallVector<std::pair<std::string, uint64_t>> rejected;

  /// Whether a bit of `value` depends on a bit of a register or black box of
  /// the pipeline `pipeline`.
  bool dependsOnPipeline(mlir::Value value, unsigned pipeline);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace foldsim

#endif // FOLDSIM_LIB_TRANSFORMS_PIPELINES_H
