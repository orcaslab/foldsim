//===- SplitArrayRegisters.cpp - array registers read by constant index ---===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A register of array type whose every reader takes one element at a constant
// index is a register per element: each element's register takes that
// element of the next value (and of the reset value). The same bits in every
// cycle. An RTL array written in one process (`always_ff for (o ...)
// s_hit[o][i] <= ...`) is then per-element state that the logic of one
// element alone reads, which extract-models can move into a model with that
// logic (absorb-exclusive, absorb-registers). See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace foldsim {
#define GEN_PASS_DEF_SPLITARRAYREGISTERS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// The constant index of an array read, or none.
std::optional<uint64_t> constIndex(hw::ArrayGetOp get) {
  if (auto c = get.getIndex().getDefiningOp<hw::ConstantOp>())
    return c.getValue().getZExtValue();
  return std::nullopt;
}

struct SplitArrayRegistersPass
    : public foldsim::impl::SplitArrayRegistersBase<SplitArrayRegistersPass> {
  using SplitArrayRegistersBase::SplitArrayRegistersBase;

  /// Splits `reg` if every reader takes a constant-index element.
  bool split(seq::FirRegOp reg) {
    auto type = dyn_cast<hw::ArrayType>(reg.getType());
    if (!type || reg.getInnerSymAttr() || reg.getPresetAttr())
      return false;
    uint64_t n = type.getNumElements();
    if (n < 2 || (maxElements && n > maxElements))
      return false;
    // (Logic left dead by an earlier split -- the next value of a register it
    // replaced -- goes first: it is no reader.)
    for (bool erased = true; erased;) {
      erased = false;
      for (Operation *user : llvm::make_early_inc_range(reg->getUsers()))
        if (user->use_empty() && user != reg && isMemoryEffectFree(user) &&
            user->getNumRegions() == 0) {
          user->erase();
          erased = true;
        }
    }
    SmallVector<std::pair<hw::ArrayGetOp, uint64_t>> reads;
    for (Operation *user : reg->getUsers()) {
      auto get = dyn_cast<hw::ArrayGetOp>(user);
      if (!get || get.getInput() != reg.getResult())
        return false;
      auto index = constIndex(get);
      if (!index || *index >= n)
        return false;
      reads.push_back({get, *index});
    }
    OpBuilder b(reg);
    auto loc = reg.getLoc();
    auto idxType = b.getIntegerType(std::max(1u, llvm::Log2_64_Ceil(n)));
    SmallVector<Value> parts;
    for (uint64_t i = 0; i < n; ++i) {
      Value idx = hw::ConstantOp::create(b, loc, idxType, i);
      Value next = b.createOrFold<hw::ArrayGetOp>(loc, reg.getNext(), idx);
      auto name = b.getStringAttr(reg.getName() + "_" + Twine(i));
      seq::FirRegOp part;
      if (reg.getReset())
        part = seq::FirRegOp::create(
            b, loc, next, reg.getClk(), name, reg.getReset(),
            b.createOrFold<hw::ArrayGetOp>(loc, reg.getResetValue(), idx),
            hw::InnerSymAttr{}, reg.getIsAsync());
      else
        part = seq::FirRegOp::create(b, loc, next, reg.getClk(), name);
      parts.push_back(part);
    }
    for (auto [get, i] : reads) {
      get.getResult().replaceAllUsesWith(parts[i]);
      get.erase();
    }
    reg.erase();
    ++numSplit;
    return true;
  }

  void runOnOperation() override {
    if (!modules.empty() &&
        !llvm::is_contained(modules, getOperation().getModuleName().str()))
      return;
    // Nested arrays split level by level, until none splits.
    for (bool changed = true; changed;) {
      changed = false;
      SmallVector<seq::FirRegOp> regs(getOperation().getBodyBlock()->getOps<seq::FirRegOp>());
      for (auto reg : regs)
        changed |= split(reg);
    }
  }
};

} // namespace
