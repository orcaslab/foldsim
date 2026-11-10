//===- Beacon.h - The devices Beacon maps to and times on -------*- C++ -*-===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Beacon estimates the logic levels of every register-to-register path of a
// design on a target, and its area: `foldsim-beacon-map{target=<name>}` maps
// the design to the target's cells, `foldsim-estimate-timing{target=<name>}`
// counts their levels, and on a mapping for area (`area`)
// `foldsim-estimate-area{target=<name>}` counts them. A target is named after
// its FPGA family as hw/syn/xilinx/platforms names it, or after hw/syn/yosys's
// PDK, so a flow passes the name it has.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_BEACON_H
#define FOLDSIM_BEACON_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include <string>

namespace mlir {
class Operation;
} // namespace mlir

namespace foldsim {

/// A device Beacon maps a design to and counts its logic levels on. On an
/// FPGA, levels are as Vivado counts them: LUTs, 8-bit carry blocks, DSPs
/// and 64-word distributed RAMs. On an ASIC library, levels are cell stages.
struct BeaconTarget {
  /// The FPGA family or ASIC PDK.
  llvm::StringRef name;
  /// The inputs of a cut the mapper covers with one cell: a LUT's on an
  /// FPGA, a library cell's on an ASIC.
  unsigned cutSize;
  /// The operand widths of a DSP's multiplier.
  unsigned dspA, dspB;
  /// Levels from the fabric to a product the DSP registers, from there out
  /// to the fabric, and through each further DSP of a cascade.
  int dspIn, dspOut, dspCascade;
  /// An ASIC library, and its delays in ps: a stage, a flip-flop's clock to
  /// output and setup (hw/syn/yosys/liberty_stage.py), and what a path adds
  /// beyond its stages, fitted on OpenSTA (buffering, the inverter after a
  /// flip-flop's QN, slew).
  bool asic = false;
  int stagePs = 0, clk2qPs = 0, setupPs = 0, pathPs = 0;
  /// Vivado's LUTs per LUT of foldsim-beacon-map's area mapping, fitted on
  /// routed builds: Vivado's area recovery, LUT combining and clock-enable
  /// and reset pins leave fewer LUTs than a cut mapper's cover (1 on an
  /// ASIC, which counts cells).
  double lutScale = 1.0;
};

/// The target of a name, or null.
const BeaconTarget *lookupBeaconTarget(llvm::StringRef name);

/// Reports on `op` that `name` is no target, listing the targets.
void emitUnknownBeaconTarget(mlir::Operation *op, llvm::StringRef name);

/// The pass pipeline (on a builtin.module) of `foldsim-beacon-map` for
/// `target`: for depth, or for area.
std::string beaconMapPipeline(const BeaconTarget &target, bool area,
                              unsigned carryBlockSize = 0,
                              llvm::StringRef arrivals = {});

/// The levels of the deepest end point of the design `top` elaborates in
/// `design`, a design mapped by `beaconMapPipeline(target, false)`, as
/// `foldsim-estimate-timing` counts them; failure if it cannot time it.
mlir::FailureOr<int> beaconDeepestLevels(mlir::ModuleOp design,
                                         llvm::StringRef top,
                                         const BeaconTarget &target);

} // namespace foldsim

#endif // FOLDSIM_BEACON_H
