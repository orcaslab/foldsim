//===- FindPipelines.cpp - report the pipelines of each model -------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pipelines.h"
#include "foldsim/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace foldsim {
#define GEN_PASS_DEF_FINDPIPELINES
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {
struct FindPipelinesPass
    : public foldsim::impl::FindPipelinesBase<FindPipelinesPass> {
  using FindPipelinesBase::FindPipelinesBase;
  void runOnOperation() override;
};
} // namespace

/// LUTs of distributed RAM for `bits` bits of `depth` words, one write and
/// one read port, as Vivado packs memories that share their address: 32-deep
/// RAM32M16 (8 LUTs, 14 bits), 64-deep RAM64M8 (8 LUTs, 7 bits), deeper ones
/// in 64-deep pieces.
double foldsimLutramLuts(uint64_t bits, uint64_t depth) {
  if (depth <= 32)
    return bits * 8.0 / 14;
  return llvm::divideCeil(depth, 64) * bits * 8.0 / 7;
}

void FindPipelinesPass::runOnOperation() {
  auto top = getOperation();
  SmallVector<std::pair<hw::HWModuleOp, unsigned>> todo;
  if (models.empty()) {
    for (auto module : top.getOps<hw::HWModuleOp>())
      if (!module.getBodyBlock()->getOps<seq::FirRegOp>().empty())
        todo.push_back({module, 1});
  }
  for (auto &entry : models) {
    auto [name, count] = StringRef(entry).split('=');
    unsigned threads = 1;
    if (!count.empty() && count.getAsInteger(10, threads)) {
      top.emitError("model '") << entry << "' is not <module>=<threads>";
      return signalPassFailure();
    }
    auto module = top.lookupSymbol<hw::HWModuleOp>(name);
    if (!module) {
      top.emitError("no module '") << name << "'";
      return signalPassFailure();
    }
    todo.push_back({module, threads});
  }
  auto &os = llvm::outs();
  for (auto [module, threads] : todo) {
    // On a copy: peeling a register's holds adds logic.
    OpBuilder builder(module);
    builder.setInsertionPointAfter(module);
    auto copy = cast<hw::HWModuleOp>(builder.clone(*module));
    copy.setName((module.getName() + "_find_pipelines").str());
    ImplicitLocOpBuilder b(copy.getLoc(), copy.getBodyBlock()->getTerminator());
    foldsim::PipelineAnalysis analysis(copy);
    if (failed(analysis.run(b))) {
      copy.erase();
      return signalPassFailure();
    }
    os << "model " << module.getName() << " threads " << threads << "\n";
    uint64_t sumFolded = 0;
    double sumBefore = 0, sumAfter = 0;
    int64_t sumFF = 0;
    for (auto [i, p] : llvm::enumerate(analysis.pipelines)) {
      unsigned depth = p.depth();
      unsigned slot = llvm::Log2_32_Ceil(depth);
      unsigned tag = 1 + llvm::Log2_32_Ceil(threads) + slot;
      // Now: every folded bit is a thread memory with its staging registers.
      double before = foldsimLutramLuts(p.foldedBits, threads);
      // Folded: the store and the slot counter's thread state.
      double after = foldsimLutramLuts(p.storeBits, uint64_t(depth) * threads) +
                     foldsimLutramLuts(slot, threads);
      // The copy's registers, its registered issue (at most its inputs),
      // the tags along it and the registered write.
      int64_t ff = int64_t(p.foldedBits) + int64_t(p.inputBits) +
                   int64_t(tag) * (depth + 2) + p.storeBits +
                   int64_t(slot) * stagingRegisters -
                   int64_t(p.foldedBits) * stagingRegisters;
      sumFolded += p.foldedBits;
      sumBefore += before;
      sumAfter += after;
      sumFF += ff;
      os << "  pipeline " << i << ": L=" << p.layers << " depth=" << depth
         << (p.lastLayerStored ? " stores-last-layer" : " keeps-last-layer")
         << (p.gated ? " gated" : "") << " enable="
         << (p.enable ? "signal" : "always") << " registers="
         << p.registers.size() << " boxes=" << p.boxes.size() << "\n";
      os << "    layer bits:";
      for (auto bits : p.layerBits)
        os << " " << bits;
      os << "\n    input bits " << p.inputBits << ", store bits "
         << p.storeBits << ", folded state bits " << p.foldedBits << " (x"
         << threads << " threads)\n";
      os << "    LUTRAM LUTs now " << llvm::format("%.0f", before)
         << ", folded " << llvm::format("%.0f", after) << "; registers "
         << (ff >= 0 ? "+" : "") << ff << "\n";
      for (auto &path : p.paths)
        os << "    from " << path << "\n";
    }
    os << "  total: folded state bits " << sumFolded << ", LUTRAM LUTs "
       << llvm::format("%.0f", sumBefore) << " -> "
       << llvm::format("%.0f", sumAfter) << ", registers "
       << (sumFF >= 0 ? "+" : "") << sumFF << "\n";
    for (auto &[why, bits] : analysis.rejected)
      os << "  left out: " << why << ": " << bits << " bits\n";
    copy.erase();
  }
  markAllAnalysesPreserved();
}
