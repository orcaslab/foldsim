//===- foldsim-opt.cpp - The foldsim modular optimizer --------------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs foldsim's passes, with the CIRCT HW, Comb, Seq and SV dialects and
// passes a simulator's pipeline uses between them, and Beacon's passes
// (third_party/beacon) with the Synth passes `beacon-map` runs to map a design
// to a Beacon target's cells.
//
//===----------------------------------------------------------------------===//

#include "beacon/Passes.h"
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
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::seq::SeqDialect, circt::sv::SVDialect,
                  circt::synth::SynthDialect>();

  foldsim::registerFoldsimPasses();
  beacon::registerBeaconPasses();
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
  llvm::cl::AddExtraVersionPrinter([](llvm::raw_ostream &os) {
    os << circt::getCirctVersion() << '\n';
    os << "foldsim " << FOLDSIM_REVISION << '\n';
  });

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "foldsim modular optimizer", registry));
}
