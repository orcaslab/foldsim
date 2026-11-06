//===- Beacon.cpp - The targets Beacon maps to and times on ---------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "foldsim/Beacon.h"
#include "foldsim/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>

namespace foldsim {
#define GEN_PASS_DEF_BEACONMAP
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using foldsim::BeaconTarget;

namespace {

/// From a registered product, Vivado counts M_DATA, ALUMUX, ALUADD, OUTPUT
/// and the cells before the multiplier it does not absorb on a Versal DSP58;
/// a cascaded DSP adds ALUMUX, ALUADD and OUTPUT. An UltraScale+ DSP48E2 has
/// one ALU cell and an 18-bit B port.
#define VERSAL_DSP 27, 24, 1, 6, 3
#define ULTRASCALE_PLUS_DSP 27, 18, 1, 5, 2
/// The LUT6 families share a fabric and the area mapping; the factor is the
/// geometric mean of Vivado's logic LUTs over the area mapping's on routed
/// U55C builds of Vortex units.
#define LUT6_AREA false, 0, 0, 0, 0, 0.337

const BeaconTarget kTargets[] = {
    // V80, VCK5000.
    {"versal", 6, VERSAL_DSP, LUT6_AREA},
    // U200, U250.
    {"virtexuplus", 6, ULTRASCALE_PLUS_DSP, LUT6_AREA},
    // U50, U55C, U280.
    {"virtexuplusHBM", 6, ULTRASCALE_PLUS_DSP, LUT6_AREA},
    {"zynquplus", 6, ULTRASCALE_PLUS_DSP, LUT6_AREA},
    // ASAP7 7.5-track RVT at the TT corner, as hw/syn/yosys builds with it:
    // 3-input cuts, each one cell (NAND, AOI, MAJ, XOR), which against
    // OpenSTA on five DUTs convert at the library's own cell delay where raw
    // and-inverter gates and 4-input cuts do not; a cell at fanout 3 and its
    // own slew, a DFFHQNx1's clock to output and setup, and 167 ps per path,
    // the median over five asic_gate DUTs. Products are multipliers of cells.
    {"asap7", 3, 0, 0, 0, 0, 0, true, 24, 58, 6, 167},
};

#undef VERSAL_DSP
#undef ULTRASCALE_PLUS_DSP
#undef LUT6_AREA

/// The netlist Beacon counts: an and-inverter graph with adders, compares,
/// multipliers and dividers kept word-level, mapped to the target's cells.
/// For timing, small modules inlined, balanced for depth and mapped for it, as
/// Vivado restructures a path that misses its period; for area, its common
/// subexpressions shared and mapped to the fewest cells, as Vivado maps the
/// rest of the design, module by module.
std::string mapPipeline(const BeaconTarget &target, bool area) {
  // Cells pack across small modules on a critical path; area keeps every
  // module, so each instance's resources stay its own.
  std::string pipeline =
      area ? "hw.module(hw-bypass-inner-symbols,canonicalize,cse,"
           : "hw-flatten-modules{hw-inline-all=false hw-inline-small "
             "hw-inline-with-state hw-small-threshold=300},"
             "hw.module(hw-bypass-inner-symbols,canonicalize,cse,"
             "comb-balance-mux,";
  pipeline += "foldsim-lower-equality,foldsim-lower-constant-division,"
              "hw-aggregate-to-comb,cse,foldsim-fold-zero-width,"
              "convert-comb-to-synth{additional-legal-ops=comb.add,comb.sub,"
              "comb.icmp,comb.mul,comb.divu,comb.divs,comb.modu,comb.mods},"
              "cse,canonicalize,synth-lower-word-to-bits,cse,";
  if (area)
    pipeline += "synth-lower-variadic{timing-aware=false reuse-subsets},"
                "synth-structural-hash,canonicalize,";
  else
    pipeline += "synth-lower-variadic{timing-aware},synth-structural-hash,"
                "canonicalize,synth-maximum-and-cover,"
                "synth-lower-variadic{timing-aware},synth-structural-hash,"
                "synth-sop-balancing{strategy=timing},synth-structural-hash,";
  return pipeline + "synth-generic-lut-mapper{max-lut-size=" +
         std::to_string(target.cutSize) + " strategy=" +
         (area ? "area" : "timing") + "},cse)";
}

struct BeaconMapPass : public foldsim::impl::BeaconMapBase<BeaconMapPass> {
  using BeaconMapBase::BeaconMapBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pm(ModuleOp::getOperationName());
    if (const BeaconTarget *t = foldsim::lookupBeaconTarget(target))
      if (succeeded(parsePassPipeline(mapPipeline(*t, area), pm)))
        pm.getDependentDialects(registry);
  }

  void runOnOperation() override {
    const BeaconTarget *t = foldsim::lookupBeaconTarget(target);
    if (!t) {
      foldsim::emitUnknownBeaconTarget(getOperation(), target);
      return signalPassFailure();
    }
    OpPassManager pm(ModuleOp::getOperationName());
    if (failed(parsePassPipeline(mapPipeline(*t, area), pm)) ||
        failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

} // namespace

namespace foldsim {

const BeaconTarget *lookupBeaconTarget(llvm::StringRef name) {
  for (const BeaconTarget &target : kTargets)
    if (target.name == name)
      return &target;
  return nullptr;
}

void emitUnknownBeaconTarget(Operation *op, llvm::StringRef name) {
  InFlightDiagnostic diag = op->emitError("no Beacon target '") << name
                                                               << "'; one of";
  for (const BeaconTarget &target : kTargets)
    diag << " " << target.name;
}

} // namespace foldsim
