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
#include "foldsim/Version.inc"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/Version.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

// The netlist the logic screen counts: an and-inverter graph balanced for
// depth, with adders, compares, multipliers and dividers kept word-level.
#define SCREEN_MAP_FLATTEN \
  "hw-flatten-modules{hw-inline-all=false hw-inline-small " \
    "hw-inline-with-state hw-small-threshold=300},"
#define SCREEN_MAP_PREFIX SCREEN_MAP_FLATTEN SCREEN_MAP_BODY
#define SCREEN_MAP_BODY \
    "hw.module(hw-bypass-inner-symbols,canonicalize,cse,comb-balance-mux," \
    "foldsim-lower-equality,foldsim-lower-constant-division,hw-aggregate-to-comb," \
    "cse,foldsim-fold-zero-width," \
    "convert-comb-to-synth{additional-legal-ops=comb.add,comb.sub,comb.icmp," \
    "comb.mul,comb.divu,comb.divs,comb.modu,comb.mods}," \
    "cse,canonicalize,synth-lower-word-to-bits,cse," \
    "synth-lower-variadic{timing-aware},synth-structural-hash,canonicalize," \
    "synth-maximum-and-cover,synth-lower-variadic{timing-aware}," \
    "synth-structural-hash,synth-sop-balancing{strategy=timing}," \
    "synth-structural-hash,"

/// Maps a design to LUT6 for `foldsim-estimate-timing`, as Vivado maps a
/// flattened design: modules of fewer than 300 ops (buffers, arbiters,
/// switches) are inlined so LUTs pack across their boundaries; the mapping
/// is depth-oriented, as Vivado restructures chains of associative logic and
/// priority muxes into trees; equality comparisons become logic so they pack
/// into LUTs with their neighbours; adders, order comparisons and multipliers
/// stay word-level, as carry chains and DSPs. The mapper times the bits of
/// adders, comparators and distributed-RAM reads as the 8-bit carry blocks
/// and LUT RAMs deliver them (as foldsim-estimate-timing counts them), so a
/// late carry is not buried under a deep cone as Vivado would not bury it.
///
/// Each module left after inlining (a fold's models) is mapped apart, its
/// input ports and its instances' outputs arriving at 0 unless `arrivals`
/// gives them: foldsim-estimate-timing's `arrivals` over a first mapping, so
/// a second one times them as the whole hierarchy delivers them (as Vivado
/// maps the flattened design).
static const char *const screenMapPipeline =
    SCREEN_MAP_FLATTEN "%s" SCREEN_MAP_BODY
    "synth-generic-lut-mapper{max-lut-size=6 strategy=timing "
    "carry-block-size=%u},cse)";

namespace {
struct ScreenMapOptions : public mlir::PassPipelineOptions<ScreenMapOptions> {
  Option<unsigned> carryBlockSize{
      *this, "carry-block-size",
      llvm::cl::desc("Bits per carry block the mapper times adders and "
                     "comparators with (0: their bits arrive at 0, as before "
                     "arrival-aware mapping)"),
      llvm::cl::init(8)};
  Option<std::string> arrivals{
      *this, "arrivals",
      llvm::cl::desc("Arrival times at the modules' ports, from "
                     "foldsim-estimate-timing's `arrivals` over a first "
                     "mapping (JSON); none: they arrive at 0")};
};
} // namespace

/// The same netlist mapped to 3-input cuts, for an ASIC library: a cut is one
/// cell (NAND, AOI, MAJ, XOR). Against OpenSTA on five ASAP7 DUTs, 3-input
/// cuts convert at the library's own cell delay; raw and-inverter gates and
/// 4-input cuts do not.
static const char *const screenMapAsicPipeline =
    SCREEN_MAP_PREFIX
    "synth-generic-lut-mapper{max-lut-size=3 strategy=timing},cse)";

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
  mlir::PassPipelineRegistration<ScreenMapOptions>(
      "foldsim-screen-map", "Map a design to LUT6 for foldsim-estimate-timing",
      [](mlir::OpPassManager &pm, const ScreenMapOptions &options) {
        std::string annotate, pipeline;
        if (!options.arrivals.empty())
          annotate = "foldsim-boundary-arrivals{file=" + options.arrivals +
                     "},";
        llvm::raw_string_ostream(pipeline)
            << llvm::format(screenMapPipeline, annotate.c_str(),
                            (unsigned)options.carryBlockSize);
        if (failed(mlir::parsePassPipeline(pipeline, pm)))
          llvm::report_fatal_error("invalid foldsim-screen-map pipeline");
      });
  mlir::PassPipelineRegistration<>(
      "foldsim-screen-map-asic",
      "Map a design to 3-input cells for foldsim-estimate-timing on an ASIC "
      "library",
      [](mlir::OpPassManager &pm) {
        if (failed(mlir::parsePassPipeline(screenMapAsicPipeline, pm)))
          llvm::report_fatal_error("invalid foldsim-screen-map-asic pipeline");
      });
  llvm::cl::AddExtraVersionPrinter([](llvm::raw_ostream &os) {
    os << circt::getCirctVersion() << '\n';
    os << "foldsim " << FOLDSIM_REVISION << '\n';
  });

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "foldsim modular optimizer", registry));
}
