//===- CompressAddTrees.cpp - wide sums as compressor trees ---------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A tree of additions is gathered into the addends it sums, and a sum of many
// addends is rebuilt as a compressor tree (CIRCT's datapath CompressorTree:
// full and half adders, Dadda's reduction) down to two rows, then one
// addition. A pairwise adder tree over many one-bit inputs (a population
// count) costs a carry chain per level of the tree; the compressor costs a
// full adder per level of a tree of base 1.5, in logic that packs into LUTs.
// See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/KnownBits.h"

namespace foldsim {
#define GEN_PASS_DEF_COMPRESSADDTREES
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// The value a zero extension `concat(0, ..., 0, x)` extends, or null.
Value zeroExtended(Value v) {
  auto concat = v.getDefiningOp<comb::ConcatOp>();
  if (!concat || concat.getNumOperands() < 2)
    return {};
  auto inputs = concat.getInputs();
  for (Value high : inputs.drop_back())
    if (!matchPattern(high, m_Zero()))
      return {};
  return inputs.back();
}

/// The largest value `v` can take, as its known zero bits allow.
APInt maxValue(Value v) { return ~comb::computeKnownBits(v).Zero; }

/// True if the sum of `add`'s operands never exceeds its width.
bool cannotWrap(comb::AddOp add) {
  unsigned width = add.getType().getIntOrFloatBitWidth();
  unsigned wide = width + llvm::Log2_32_Ceil(add.getNumOperands() + 1) + 1;
  APInt sum(wide, 0);
  for (Value operand : add.getInputs())
    sum += maxValue(operand).zext(wide);
  return sum.getActiveBits() <= width;
}

/// The addition an operand of a `width`-bit addition stands for, with the
/// ops that only carry it there (a zero extension), or null: an addition of
/// the same width used there alone, or the zero extension of a narrower one
/// that cannot wrap, both used there alone.
comb::AddOp absorbable(Value operand, unsigned width,
                       SmallVectorImpl<Operation *> &carriers) {
  if (auto add = operand.getDefiningOp<comb::AddOp>())
    return add.getType().getIntOrFloatBitWidth() == width &&
                   add->hasOneUse()
               ? add
               : comb::AddOp();
  Value inner = zeroExtended(operand);
  if (!inner || !operand.hasOneUse())
    return {};
  auto add = inner.getDefiningOp<comb::AddOp>();
  if (!add || !add->hasOneUse() || !cannotWrap(add))
    return {};
  carriers.push_back(operand.getDefiningOp());
  return add;
}

/// The addends of the tree rooted at `root`, each to be zero-extended to the
/// root's width; `absorbed` collects the additions and extensions replaced.
void gather(comb::AddOp root, SmallVectorImpl<Value> &addends,
            SmallVectorImpl<Operation *> &absorbed) {
  unsigned width = root.getType().getIntOrFloatBitWidth();
  SmallVector<Value> work(llvm::reverse(root.getInputs()));
  while (!work.empty()) {
    Value v = work.pop_back_val();
    SmallVector<Operation *> carriers;
    if (comb::AddOp add = absorbable(v, width, carriers)) {
      absorbed.append(carriers.begin(), carriers.end());
      absorbed.push_back(add);
      for (Value operand : llvm::reverse(add.getInputs()))
        work.push_back(operand);
      continue;
    }
    addends.push_back(v);
  }
}

struct CompressAddTreesPass
    : public foldsim::impl::CompressAddTreesBase<CompressAddTreesPass> {
  using CompressAddTreesBase::CompressAddTreesBase;

  void runOnOperation() override {
    // Roots: additions no other addition of the tree absorbs.
    SmallVector<comb::AddOp> roots;
    getOperation().walk([&](comb::AddOp add) {
      if (!isa<IntegerType>(add.getType()))
        return;
      if (add->hasOneUse()) {
        Operation *user = *add->user_begin();
        Value use = add.getResult();
        if (isa<comb::ConcatOp>(user) && user->hasOneUse() &&
            zeroExtended(user->getResult(0)) == use) {
          use = user->getResult(0);
          user = *user->user_begin();
        }
        if (auto parent = dyn_cast<comb::AddOp>(user)) {
          SmallVector<Operation *> carriers;
          if (absorbable(use, parent.getType().getIntOrFloatBitWidth(),
                         carriers) == add)
            return;
        }
      }
      roots.push_back(add);
    });

    for (comb::AddOp root : roots) {
      SmallVector<Value> addends;
      SmallVector<Operation *> absorbed;
      gather(root, addends, absorbed);
      if (addends.size() < std::max(3u, unsigned(minOperands)))
        continue;
      OpBuilder b(root);
      Location loc = root.getLoc();
      unsigned width = root.getType().getIntOrFloatBitWidth();
      Value falseBit = hw::ConstantOp::create(b, loc, APInt(1, 0));
      SmallVector<SmallVector<Value>> rows;
      for (Value &addend : addends) {
        // A zero extension adds known zero bits: the bits of what it extends.
        Operation *ext = addend.getDefiningOp();
        while (Value inner = zeroExtended(addend))
          addend = inner;
        if (ext && ext != addend.getDefiningOp())
          absorbed.push_back(ext);
        unsigned w = addend.getType().getIntOrFloatBitWidth();
        SmallVector<Value> row;
        for (unsigned i = 0; i < width; ++i)
          row.push_back(i < w ? b.createOrFold<comb::ExtractOp>(loc, addend, i, 1)
                              : falseBit);
        rows.push_back(std::move(row));
      }
      datapath::CompressorTree tree(width, rows, loc, b);
      SmallVector<Value> two = tree.compressToHeight(b, 2);
      Value sum = b.createOrFold<comb::AddOp>(loc, two, true);
      root.replaceAllUsesWith(sum);
      root.erase();
      // The replaced tree, users first (gather visits a node after its user;
      // an addend's extension comes last), each op once.
      for (Operation *op : llvm::SetVector<Operation *>(absorbed.begin(),
                                                         absorbed.end()))
        if (op->use_empty())
          op->erase();
      if (falseBit.use_empty())
        falseBit.getDefiningOp()->erase();
    }
  }
};
} // namespace
