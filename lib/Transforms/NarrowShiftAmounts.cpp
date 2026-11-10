//===- NarrowShiftAmounts.cpp - shift amounts to log2(width) bits ---------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A shift `x << a`, `x >> a` or `x >>> a` of w bits takes its amount `a` in w
// bits too; only its low ceil(log2 w) bits choose the shift, the rest only say
// it is out of range. The shift becomes `hi != 0 ? fill : x op zext(lo)`: the
// out-of-range test an OR of the high bits (logarithmic depth, where a
// w-bit `a >= w` compare is a chain), and a pass split that carries the amount
// across passes carries its ceil(log2 w) low bits and that one bit, not w.
// See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace foldsim {
#define GEN_PASS_DEF_NARROWSHIFTAMOUNTS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

struct NarrowShiftAmountsPass
    : public foldsim::impl::NarrowShiftAmountsBase<NarrowShiftAmountsPass> {
  using NarrowShiftAmountsBase::NarrowShiftAmountsBase;

  void runOnOperation() override {
    SmallVector<Operation *> shifts;
    getOperation().walk([&](Operation *op) {
      if (!isa<comb::ShlOp, comb::ShrUOp, comb::ShrSOp>(op))
        return;
      auto type = dyn_cast<IntegerType>(op->getResult(0).getType());
      if (!type || type.getWidth() < std::max(4u, unsigned(minWidth)))
        return;
      if (matchPattern(op->getOperand(1), m_Constant()))
        return;
      // Already narrow: the amount's high bits a constant zero.
      if (auto concat = op->getOperand(1).getDefiningOp<comb::ConcatOp>())
        if (matchPattern(concat.getOperand(0), m_Zero()) &&
            hw::getBitWidth(concat.getOperand(0).getType()) >=
                int64_t(type.getWidth() - llvm::Log2_32_Ceil(type.getWidth())))
          return;
      shifts.push_back(op);
    });
    for (Operation *op : shifts) {
      OpBuilder b(op);
      Location loc = op->getLoc();
      Value x = op->getOperand(0), amount = op->getOperand(1);
      unsigned w = op->getResult(0).getType().getIntOrFloatBitWidth();
      unsigned k = llvm::Log2_32_Ceil(w);
      Value lo = b.createOrFold<comb::ExtractOp>(loc, amount, 0, k);
      Value hi = b.createOrFold<comb::ExtractOp>(loc, amount, k, w - k);
      Value out = comb::ICmpOp::create(b, loc, comb::ICmpPredicate::ne, hi,
                                       hw::ConstantOp::create(b, loc, APInt(w - k, 0)));
      Value narrow = b.createOrFold<comb::ConcatOp>(
          loc, hw::ConstantOp::create(b, loc, APInt(w - k, 0)), lo);
      Value shifted, fill;
      if (isa<comb::ShlOp>(op)) {
        shifted = comb::ShlOp::create(b, loc, x, narrow);
        fill = hw::ConstantOp::create(b, loc, APInt(w, 0));
      } else if (isa<comb::ShrUOp>(op)) {
        shifted = comb::ShrUOp::create(b, loc, x, narrow);
        fill = hw::ConstantOp::create(b, loc, APInt(w, 0));
      } else {
        shifted = comb::ShrSOp::create(b, loc, x, narrow);
        // An arithmetic shift past the width: every bit the sign.
        fill = comb::ReplicateOp::create(
            b, loc, b.createOrFold<comb::ExtractOp>(loc, x, w - 1, 1), w);
      }
      Value result = comb::MuxOp::create(b, loc, out, fill, shifted);
      op->getResult(0).replaceAllUsesWith(result);
      op->erase();
    }
  }
};
} // namespace
