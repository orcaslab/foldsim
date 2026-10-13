//===- foldsim-opt.cpp - The foldsim modular optimizer --------------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs foldsim's passes, with the CIRCT HW, Comb, Seq and SV dialects and
// passes a simulator's pipeline uses between them.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/Version.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::seq::SeqDialect, circt::sv::SVDialect>();

  foldsim::registerFoldsimPasses();
  circt::comb::registerPasses();
  circt::hw::registerPasses();
  circt::seq::registerPasses();
  circt::sv::registerPasses();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();
  mlir::registerTopologicalSortPass();
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "foldsim modular optimizer", registry));
}
