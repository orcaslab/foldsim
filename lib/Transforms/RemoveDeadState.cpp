//===- RemoveDeadState.cpp - remove state no output observes --------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers and memories whose values reach no output of their module, and
// the logic that only feeds them, are removed. Canonicalization removes an
// unused value, but not a register that feeds itself (a counter whose only
// reader is its own increment): a performance counter left unconnected keeps
// its whole cone. See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace foldsim {
#define GEN_PASS_DEF_REMOVEDEADSTATE
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// An op the pass may remove when nothing live reads it: logic without side
/// effects, and state without a symbol (a symbol may be referenced from
/// outside the module).
bool removable(Operation *op) {
  if (op->getNumRegions())
    return false;
  if (auto reg = dyn_cast<seq::FirRegOp>(op))
    return !reg.getInnerSymAttr();
  if (auto reg = dyn_cast<seq::CompRegOp>(op))
    return !reg.getInnerSymAttr();
  if (auto reg = dyn_cast<seq::CompRegClockEnabledOp>(op))
    return !reg.getInnerSymAttr();
  if (auto memory = dyn_cast<seq::FirMemOp>(op))
    return !memory.getInnerSymAttr();
  if (isa<seq::FirMemReadOp, seq::FirMemWriteOp, seq::FirMemReadWriteOp>(op))
    return true;
  if (isa<hw::InstanceOp, hw::OutputOp>(op))
    return false;
  return isMemoryEffectFree(op);
}

struct RemoveDeadStatePass
    : public foldsim::impl::RemoveDeadStateBase<RemoveDeadStatePass> {
  void runOnOperation() override {
    auto module = getOperation();
    Block *body = module.getBodyBlock();
    llvm::DenseSet<Operation *> live;
    SmallVector<Operation *> work;
    auto mark = [&](Operation *op) {
      if (op && op->getBlock() == body && live.insert(op).second)
        work.push_back(op);
    };
    for (auto &op : *body)
      if (!removable(&op))
        mark(&op);
    while (!work.empty()) {
      Operation *op = work.pop_back_val();
      for (Value operand : op->getOperands())
        mark(operand.getDefiningOp());
      // A memory read is a memory whose writes matter.
      if (auto memory = dyn_cast<seq::FirMemOp>(op))
        for (Operation *user : memory->getUsers())
          if (isa<seq::FirMemWriteOp, seq::FirMemReadWriteOp>(user))
            mark(user);
    }
    SmallVector<Operation *> dead;
    for (auto &op : *body)
      if (!live.contains(&op))
        dead.push_back(&op);
    if (dead.empty())
      return markAllAnalysesPreserved();
    // The dead read only the dead and live ops never read them: cut the
    // references (cycles through registers) before erasing.
    for (Operation *op : dead)
      op->dropAllReferences();
    for (Operation *op : dead)
      op->erase();
    numRemoved += dead.size();
  }
};

} // namespace
