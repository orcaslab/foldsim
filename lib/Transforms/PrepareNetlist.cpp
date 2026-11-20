//===- PrepareNetlist.cpp - a netlist's resets and memories for folding --===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// foldsim-sync-async-resets and foldsim-split-masked-memories, the driver's
// preparation of an elaborated netlist (they lived beside Beacon's passes in
// EstimateTiming.cpp before Beacon became its own project).
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace foldsim {
#define GEN_PASS_DEF_SPLITMASKEDMEMORIES
#define GEN_PASS_DEF_SYNCASYNCRESETS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

struct SyncAsyncResetsPass
    : public foldsim::impl::SyncAsyncResetsBase<SyncAsyncResetsPass> {
  void runOnOperation() override {
    SmallVector<seq::FirRegOp> regs;
    getOperation().walk([&](seq::FirRegOp reg) {
      if (reg.getIsAsync())
        regs.push_back(reg);
    });
    for (auto reg : regs) {
      OpBuilder b(reg);
      auto sync = seq::FirRegOp::create(
          b, reg.getLoc(), reg.getNext(), reg.getClk(), reg.getNameAttr(),
          reg.getReset(), reg.getResetValue(), reg.getInnerSymAttr(),
          /*isAsync=*/false, reg.getPresetAttr());
      b.setInsertionPointAfter(sync);
      auto value = comb::MuxOp::create(b, reg.getLoc(), reg.getReset(),
                                       reg.getResetValue(), sync);
      reg.getResult().replaceAllUsesWith(value.getResult());
      reg.erase();
    }
  }
};

struct SplitMaskedMemoriesPass
    : public foldsim::impl::SplitMaskedMemoriesBase<SplitMaskedMemoriesPass> {
  void runOnOperation() override {
    constexpr unsigned kColumn = 64, kByte = 8;
    SmallVector<seq::FirMemOp> memories;
    getOperation().walk([&](seq::FirMemOp memory) {
      auto type = memory.getType();
      auto maskWidth = type.getMaskWidth();
      unsigned width = type.getWidth();
      if (!maskWidth || *maskWidth * kByte != width || width <= kColumn ||
          width % kColumn || memory.getInnerSymAttr() || memory.getInit())
        return;
      bool split = llvm::all_of(memory->getUsers(), [](Operation *user) {
        if (auto write = dyn_cast<seq::FirMemWriteOp>(user))
          return bool(write.getMask());
        return isa<seq::FirMemReadOp>(user);
      });
      if (split)
        memories.push_back(memory);
    });
    for (auto memory : memories) {
      auto type = memory.getType();
      unsigned columns = type.getWidth() / kColumn;
      auto columnType = seq::FirMemType::get(&getContext(), type.getDepth(),
                                             kColumn, kColumn / kByte);
      OpBuilder b(memory);
      SmallVector<Value> parts;
      for (unsigned c = 0; c < columns; ++c) {
        StringAttr name;
        if (auto n = memory.getNameAttr())
          name = b.getStringAttr(n.getValue() + "_c" + Twine(c));
        auto part = seq::FirMemOp::create(
            b, memory.getLoc(), columnType, memory.getReadLatency(),
            memory.getWriteLatency(), memory.getRuw(), memory.getWuw(), name,
            {}, {}, memory.getPrefixAttr(), memory.getOutputFileAttr());
        // A column is the memory's: its storage (the ram_style a storage
        // rule or the RTL set) goes with it.
        part->setDialectAttrs(memory->getDialectAttrs());
        parts.push_back(part);
      }
      for (auto *user : llvm::make_early_inc_range(memory->getUsers())) {
        OpBuilder ub(user);
        if (auto read = dyn_cast<seq::FirMemReadOp>(user)) {
          SmallVector<Value> data;
          for (unsigned c = columns; c-- > 0;)
            data.push_back(seq::FirMemReadOp::create(
                ub, read.getLoc(), parts[c], read.getAddress(), read.getClk(),
                read.getEnable()));
          read.replaceAllUsesWith(
              comb::ConcatOp::create(ub, read.getLoc(), data).getResult());
        } else {
          auto write = cast<seq::FirMemWriteOp>(user);
          for (unsigned c = 0; c < columns; ++c) {
            Value data = comb::ExtractOp::create(ub, write.getLoc(),
                                                 write.getData(), c * kColumn,
                                                 kColumn);
            Value mask = comb::ExtractOp::create(
                ub, write.getLoc(), write.getMask(), c * (kColumn / kByte),
                kColumn / kByte);
            seq::FirMemWriteOp::create(ub, write.getLoc(), parts[c],
                                       write.getAddress(), write.getClk(),
                                       write.getEnable(), data, mask);
          }
        }
        user->erase();
      }
      memory.erase();
    }
  }
};

} // namespace
