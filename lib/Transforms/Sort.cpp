//===- Sort.cpp - topological order of graph regions in linear time --------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The order of MLIR's sortTopologically: rounds that scan a block's unsorted
// ops in order, each taking those whose operands are all sorted, and the first
// unsorted op when a round takes none. It rescans the block every round, and a
// netlist has a round per register loop: quadratic, hours for a whole GPU in
// one module. Here each op's round and place follow from its last operand's:
// taken later in the same round if it comes after that operand, in the next
// round if before.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include <queue>

namespace foldsim {
#define GEN_PASS_DEF_SORT
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;

namespace {

void sortBlock(Block &block) {
  SmallVector<Operation *> ops;
  DenseMap<Operation *, unsigned> index;
  for (Operation &op : block.without_terminator()) {
    index[&op] = ops.size();
    ops.push_back(&op);
  }
  const unsigned n = ops.size();
  if (n < 2)
    return;

  // Each op's operands from other ops of the block, counted per use, an
  // operand of an op nested in another counting for the outer one.
  SmallVector<unsigned> pending(n, 0);
  SmallVector<SmallVector<unsigned, 2>> users(n);
  for (unsigned i = 0; i < n; ++i)
    ops[i]->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands()) {
        Operation *def = operand.getDefiningOp();
        while (def && def->getBlock() != &block)
          def = def->getParentOp();
        if (!def || def == ops[i])
          continue;
        auto it = index.find(def);
        if (it == index.end())
          continue;
        ++pending[i];
        users[it->second].push_back(i);
      }
    });

  // Ops ready to take, by round, then place in the block.
  using Turn = std::pair<unsigned, unsigned>;
  std::priority_queue<Turn, std::vector<Turn>, std::greater<Turn>> ready;
  for (unsigned i = 0; i < n; ++i)
    if (pending[i] == 0)
      ready.push({0, i});
  SmallVector<bool> taken(n, false);
  SmallVector<Operation *> order;
  order.reserve(n);
  // Takes op i in `round` at `place` (n: a round of its own).
  auto take = [&](unsigned i, unsigned round, unsigned place) {
    taken[i] = true;
    order.push_back(ops[i]);
    for (unsigned u : users[i])
      if (--pending[u] == 0 && !taken[u])
        ready.push({place < u ? round : round + 1, u});
  };
  unsigned round = 0, first = 0;
  while (order.size() < n) {
    if (!ready.empty()) {
      auto [r, i] = ready.top();
      ready.pop();
      if (taken[i])
        continue;
      round = r;
      take(i, r, i);
      continue;
    }
    // A round that takes nothing: the first op not taken.
    while (taken[first])
      ++first;
    take(first, ++round, n);
  }

  Operation *end = block.mightHaveTerminator() ? block.getTerminator() : nullptr;
  for (Operation *op : order) {
    if (end)
      op->moveBefore(end);
    else
      op->moveBefore(&block, block.end());
  }
}

struct SortPass : public foldsim::impl::SortBase<SortPass> {
  void runOnOperation() override {
    getOperation()->walk([](RegionKindInterface op) {
      for (auto [i, region] : llvm::enumerate(op->getRegions())) {
        if (op.hasSSADominance(i))
          continue;
        for (Block &block : region)
          sortBlock(block);
      }
    });
  }
};

} // namespace
