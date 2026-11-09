//===- MergeMemories.cpp - memories of equal ports merged side by side ----===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memories with one write port and one read port whose addresses, enables
// and clocks are the same values -- up to copies: registers and logic of the
// same kind and attributes (but their names) over the same operands -- take
// the same entries at the same host cycles. Laid side by side in one memory,
// a row of each, they keep every read and write as it was, in fewer and
// fuller RAM columns: the constant-shell fold's link mailboxes (a row per
// producer and reader class pair, the links' bits side by side).
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include <map>

namespace foldsim {
#define GEN_PASS_DEF_MERGEMEMORIES
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// Value numbering up to copies: a value's number is its defining op's kind,
/// attributes (but names and inner symbols), result index and type, and its
/// operands' numbers. A block argument, an op with regions or side effects
/// other than a register's, and a register on a loop through itself keep a
/// number of their own.
class Numbering {
public:
  unsigned of(Value root) {
    if (auto it = ids.find(root); it != ids.end())
      return it->second;
    // Depth first without recursion: an entry is a value and whether its
    // operands are numbered.
    SmallVector<std::pair<Value, bool>> stack{{root, false}};
    while (!stack.empty()) {
      auto [value, expanded] = stack.back();
      if (ids.count(value)) {
        stack.pop_back();
        continue;
      }
      Operation *op = value.getDefiningOp();
      if (!op || op->getNumRegions() || !numbered(op)) {
        ids[value] = fresh();
        stack.pop_back();
        continue;
      }
      if (!expanded) {
        stack.back().second = true;
        if (!open.insert(op).second) {
          // A loop: this op's values keep numbers of their own.
          for (Value r : op->getResults())
            if (!ids.count(r))
              ids[r] = fresh();
          stack.pop_back();
          continue;
        }
        for (Value operand : op->getOperands())
          if (!ids.count(operand))
            stack.push_back({operand, false});
        continue;
      }
      open.erase(op);
      SmallVector<uint64_t> key;
      key.push_back(reinterpret_cast<uintptr_t>(
          op->getName().getAsOpaquePointer()));
      key.push_back(reinterpret_cast<uintptr_t>(
          attributesOf(op).getAsOpaquePointer()));
      for (Value operand : op->getOperands())
        key.push_back(ids.lookup(operand));
      for (auto [index, result] : llvm::enumerate(op->getResults())) {
        SmallVector<uint64_t> k = key;
        k.push_back(index);
        k.push_back(reinterpret_cast<uintptr_t>(
            result.getType().getAsOpaquePointer()));
        auto [it, inserted] = interned.try_emplace(
            std::vector<uint64_t>(k.begin(), k.end()), 0);
        if (inserted)
          it->second = fresh();
        ids[result] = it->second;
      }
      stack.pop_back();
    }
    return ids.lookup(root);
  }

private:
  unsigned fresh() { return ++last; }
  static bool numbered(Operation *op) {
    return isa<seq::FirRegOp, seq::CompRegOp, seq::CompRegClockEnabledOp>(op) ||
           (op->getDialect() &&
            isa<comb::CombDialect, hw::HWDialect>(op->getDialect()) &&
            !isa<hw::InstanceOp, hw::OutputOp>(op) &&
            op->getNumResults() > 0);
  }
  /// An op's attributes without its name and inner symbol.
  DictionaryAttr attributesOf(Operation *op) {
    SmallVector<NamedAttribute> kept;
    for (auto attr : op->getAttrs())
      if (attr.getName() != "name" && attr.getName() != "inner_sym" &&
          attr.getName() != "sv.namehint")
        kept.push_back(attr);
    return DictionaryAttr::get(op->getContext(), kept);
  }
  DenseMap<Value, unsigned> ids;
  DenseSet<Operation *> open;
  std::map<std::vector<uint64_t>, unsigned> interned;
  unsigned last = 0;
};

struct MergeMemoriesPass
    : public foldsim::impl::MergeMemoriesBase<MergeMemoriesPass> {
  using MergeMemoriesBase::MergeMemoriesBase;
  void runOnOperation() override {
    Numbering numbering;
    // The memories that may merge, by their ports' numbers and kind.
    std::map<std::vector<uint64_t>, SmallVector<seq::FirMemOp>> groups;
    SmallVector<std::vector<uint64_t>> order;
    getOperation().walk([&](seq::FirMemOp memory) {
      auto type = memory.getType();
      // (A mask width of 1 and no masked write: unmasked.)
      if ((type.getMaskWidth() && *type.getMaskWidth() > 1) || memory.getInit() ||
          memory.getInnerSymAttr() ||
          type.getWidth() >= maxBits)
        return;
      seq::FirMemWriteOp write;
      seq::FirMemReadOp read;
      for (auto *user : memory->getUsers()) {
        if (auto w = dyn_cast<seq::FirMemWriteOp>(user)) {
          if (write || w.getMask())
            return;
          write = w;
        } else if (auto r = dyn_cast<seq::FirMemReadOp>(user)) {
          if (read)
            return;
          read = r;
        } else {
          return;
        }
      }
      if (!write || !read)
        return;
      auto id = [&](Value v) -> uint64_t { return v ? numbering.of(v) : 0; };
      std::vector<uint64_t> key{
          type.getDepth(), memory.getReadLatency(), memory.getWriteLatency(),
          uint64_t(memory.getRuw()), uint64_t(memory.getWuw()),
          reinterpret_cast<uintptr_t>(
              memory->getDiscardableAttrDictionary().getAsOpaquePointer()),
          id(write.getAddress()), id(write.getEnable()), id(write.getClk()),
          id(read.getAddress()), id(read.getEnable()), id(read.getClk()),
          reinterpret_cast<uintptr_t>(memory->getBlock())};
      auto &group = groups[key];
      if (group.empty())
        order.push_back(key);
      group.push_back(memory);
    });
    unsigned merged = 0, into = 0;
    for (auto &key : order) {
      auto &group = groups[key];
      // Side by side, up to `maxBits` a merged memory.
      for (size_t at = 0; at < group.size();) {
        SmallVector<seq::FirMemOp> parts;
        unsigned width = 0;
        while (at < group.size() &&
               (parts.empty() ||
                width + group[at].getType().getWidth() <= maxBits)) {
          width += group[at].getType().getWidth();
          parts.push_back(group[at++]);
        }
        if (parts.size() < 2)
          continue;
        merge(parts, width);
        merged += parts.size();
        ++into;
      }
    }
    if (merged)
      getOperation()->emitRemark() << "merge-memories: " << merged
                                   << " memories into " << into;
  }

  void merge(SmallVector<seq::FirMemOp> parts, unsigned width) {
    auto first = parts.front();
    // The merged memory and its ports go after the last part's ports, where
    // every operand they take is defined (a graph region: any order holds,
    // but keep it readable).
    OpBuilder b(first);
    auto type = seq::FirMemType::get(&getContext(), first.getType().getDepth(),
                                     width, first.getType().getMaskWidth());
    StringAttr name;
    if (auto n = first.getNameAttr())
      name = b.getStringAttr(n.getValue() + "_mbox");
    auto memory = seq::FirMemOp::create(
        b, first.getLoc(), type, first.getReadLatency(),
        first.getWriteLatency(), first.getRuw(), first.getWuw(), name, {}, {},
        first.getPrefixAttr(), first.getOutputFileAttr());
    memory->setDialectAttrs(first->getDialectAttrs());
    seq::FirMemWriteOp write0;
    seq::FirMemReadOp read0;
    SmallVector<Value> data;
    SmallVector<seq::FirMemReadOp> reads;
    SmallVector<seq::FirMemWriteOp> writes;
    for (auto part : parts)
      for (auto *user : part->getUsers()) {
        if (auto w = dyn_cast<seq::FirMemWriteOp>(user))
          writes.push_back(w);
        else
          reads.push_back(cast<seq::FirMemReadOp>(user));
      }
    write0 = writes.front();
    read0 = reads.front();
    // The first part the most significant.
    for (auto w : writes)
      data.push_back(w.getData());
    OpBuilder wb(write0);
    Value all = comb::ConcatOp::create(wb, write0.getLoc(), data);
    seq::FirMemWriteOp::create(wb, write0.getLoc(), memory, write0.getAddress(),
                               write0.getClk(), write0.getEnable(), all,
                               Value());
    OpBuilder rb(read0);
    auto read = seq::FirMemReadOp::create(rb, read0.getLoc(), memory,
                                          read0.getAddress(), read0.getClk(),
                                          read0.getEnable());
    unsigned low = width;
    for (auto [k, r] : llvm::enumerate(reads)) {
      unsigned w = seq::FirMemOp(parts[k]).getType().getWidth();
      low -= w;
      Value slice =
          comb::ExtractOp::create(rb, r.getLoc(), read.getData(), low, w);
      r.getData().replaceAllUsesWith(slice);
    }
    for (auto w : writes)
      w.erase();
    for (auto r : reads)
      r.erase();
    for (auto part : parts)
      part.erase();
  }
};

} // namespace
