//===- foldsim-opt.cpp - The foldsim modular optimizer --------------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs foldsim's passes, with the CIRCT HW, Comb, Seq and SV dialects and
// passes a simulator's pipeline uses between them, and the Synth passes that
// map a design to LUTs for `foldsim-estimate-timing`.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/Version.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

/// Maps a design to LUT6 for `foldsim-estimate-timing`, as Vivado maps a
/// flattened design: modules of fewer than 300 ops (buffers, arbiters,
/// switches) are inlined so LUTs pack across their boundaries; the mapping
/// is depth-oriented, as Vivado restructures chains of associative logic and
/// priority muxes into trees; equality comparisons become logic so they pack
/// into LUTs with their neighbours; adders, order comparisons and multipliers
/// stay word-level, as carry chains and DSPs.
static const char *const timingMapPipeline =
    "hw-flatten-modules{hw-inline-all=false hw-inline-small "
    "hw-inline-with-state hw-small-threshold=300},"
    "hw.module(hw-bypass-inner-symbols,canonicalize,cse,comb-balance-mux,"
    "foldsim-lower-equality,hw-aggregate-to-comb,cse,"
    "convert-comb-to-synth{additional-legal-ops=comb.add,comb.sub,comb.icmp,"
    "comb.mul},"
    "cse,canonicalize,synth-lower-word-to-bits,cse,"
    "synth-lower-variadic{timing-aware},synth-structural-hash,canonicalize,"
    "synth-maximum-and-cover,synth-lower-variadic{timing-aware},"
    "synth-structural-hash,synth-sop-balancing{strategy=timing},"
    "synth-structural-hash,"
    "synth-generic-lut-mapper{max-lut-size=6 strategy=timing},cse)";

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::seq::SeqDialect, circt::sv::SVDialect,
                  circt::synth::SynthDialect>();

  foldsim::registerFoldsimPasses();
  circt::comb::registerPasses();
  circt::hw::registerPasses();
  circt::seq::registerPasses();
  circt::sv::registerPasses();
  circt::synth::registerPasses();
  circt::registerConvertCombToSynth();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();
  mlir::registerTopologicalSortPass();
  mlir::PassPipelineRegistration<>(
      "foldsim-timing-map", "Map a design to LUT6 for foldsim-estimate-timing",
      [](mlir::OpPassManager &pm) {
        if (failed(mlir::parsePassPipeline(timingMapPipeline, pm)))
          llvm::report_fatal_error("invalid foldsim-timing-map pipeline");
      });
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "foldsim modular optimizer", registry));
}
