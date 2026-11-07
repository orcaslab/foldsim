//===- LowerWideNegation.cpp - wide negations as prefix logic -------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A negation `0 - x` of at least `min-width` bits becomes `x ^ (P(x) << 1)`,
// P the inclusive prefix OR of x's bits (bit i of -x is x[i] XOR the OR of the
// bits below it), computed in log2(width) stages of `p | (p << d)` (a
// Kogge-Stone prefix: fanout 2, depth logarithmic), where a synthesis tool
// lays the subtraction on a carry chain whose depth grows with the width. The
// isolate-lowest-set-bit idiom `x & -x` of a priority select over many inputs
// is the case that matters: its depth then grows with the log of the inputs.
// See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace foldsim {
#define GEN_PASS_DEF_LOWERWIDENEGATION
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// `v << d` within its own width, the low d bits zero.
Value shiftLeft(OpBuilder &b, Location loc, Value v, unsigned width,
                unsigned d) {
  Value low = b.createOrFold<comb::ExtractOp>(loc, v, 0, width - d);
  Value zeros = hw::ConstantOp::create(b, loc, APInt(d, 0));
  return b.createOrFold<comb::ConcatOp>(loc, low, zeros);
}

struct LowerWideNegationPass
    : public foldsim::impl::LowerWideNegationBase<LowerWideNegationPass> {
  using LowerWideNegationBase::LowerWideNegationBase;

  void runOnOperation() override {
    SmallVector<comb::SubOp> negations;
    getOperation().walk([&](comb::SubOp sub) {
      auto type = dyn_cast<IntegerType>(sub.getType());
      if (type && type.getWidth() >= std::max(2u, unsigned(minWidth)) &&
          matchPattern(sub.getLhs(), m_Zero()))
        negations.push_back(sub);
    });
    for (comb::SubOp sub : negations) {
      OpBuilder b(sub);
      Location loc = sub.getLoc();
      Value x = sub.getRhs();
      unsigned width = sub.getType().getIntOrFloatBitWidth();
      Value prefix = x;
      for (unsigned d = 1; d < width; d <<= 1)
        prefix = b.createOrFold<comb::OrOp>(
            loc, prefix, shiftLeft(b, loc, prefix, width, d), true);
      Value below = shiftLeft(b, loc, prefix, width, 1);
      Value negation = b.createOrFold<comb::XorOp>(loc, x, below, true);
      sub.replaceAllUsesWith(negation);
      sub.erase();
    }
  }
};
} // namespace
