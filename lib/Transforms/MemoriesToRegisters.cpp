//===- MemoriesToRegisters.cpp - keep memories as registers ---------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A memory read at as many addresses as it has entries (an associative table
// matched entry by entry) folds into a store copied once per read port; kept
// as registers it is one row of the copy's state per instance. See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

namespace foldsim {
#define GEN_PASS_DEF_MEMORIESTOREGISTERS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// Why `memory` cannot be registers, or an empty string if it can.
std::string ineligible(seq::FirMemOp memory) {
  if (memory.getReadLatency() != 0)
    return "its reads are registered (read latency " +
           std::to_string(memory.getReadLatency()) + ")";
  if (memory.getWriteLatency() != 1)
    return "its writes land " + std::to_string(memory.getWriteLatency()) +
           " cycles late";
  if (memory.getInit())
    return "it has an initial content";
  if (memory.getInnerSymAttr())
    return "it has a symbol";
  Value clock;
  bool writes = false;
  for (auto *user : memory->getUsers()) {
    if (user->getBlock() != memory->getBlock())
      return "a port is in another block";
    if (isa<seq::FirMemReadOp>(user))
      continue;
    auto write = dyn_cast<seq::FirMemWriteOp>(user);
    if (!write)
      return "it has a read-write port";
    if (clock && write.getClk() != clock)
      return "its write ports have several clocks";
    clock = write.getClk();
    writes = true;
  }
  if (!writes)
    return "it is never written";
  return {};
}

void lower(seq::FirMemOp memory) {
  auto type = memory.getType();
  uint64_t depth = type.getDepth();
  unsigned width = type.getWidth();
  unsigned lanes = type.getMaskWidth().value_or(1);
  unsigned grain = lanes ? width / lanes : width;
  auto loc = memory.getLoc();

  SmallVector<seq::FirMemReadOp> reads;
  SmallVector<seq::FirMemWriteOp> writes;
  for (auto *user : memory->getUsers()) {
    if (auto read = dyn_cast<seq::FirMemReadOp>(user))
      reads.push_back(read);
    else
      writes.push_back(cast<seq::FirMemWriteOp>(user));
  }
  // A later port's write lands over an earlier one's, in the module's order.
  llvm::sort(writes, [](seq::FirMemWriteOp a, seq::FirMemWriteOp b) {
    return a->isBeforeInBlock(b);
  });

  OpBuilder b(memory);
  Value zero = hw::ConstantOp::create(b, loc, APInt(width, 0));
  Value clock = writes.front().getClk();
  StringRef name = memory.getNameAttr() ? memory.getNameAttr().getValue() : "";
  SmallVector<seq::FirRegOp> entries;
  for (uint64_t i = 0; i < depth; ++i)
    entries.push_back(seq::FirRegOp::create(
        b, loc, zero, clock, b.getStringAttr(name + "_" + Twine(i))));

  // Each entry's next value: itself, unless a write port addresses it.
  DenseMap<std::pair<unsigned, uint64_t>, Value> indices;
  for (uint64_t i = 0; i < depth; ++i) {
    Value next = entries[i];
    for (auto write : writes) {
      auto addressType = cast<IntegerType>(write.getAddress().getType());
      Value &index = indices[{addressType.getWidth(), i}];
      if (!index)
        index = hw::ConstantOp::create(b, loc,
                                       APInt(addressType.getWidth(), i));
      Value hit = comb::ICmpOp::create(b, loc, comb::ICmpPredicate::eq,
                                       write.getAddress(), index);
      if (auto enable = write.getEnable())
        hit = comb::AndOp::create(b, loc, enable, hit);
      auto mask = write.getMask();
      if (!mask || lanes <= 1) {
        if (mask)
          hit = comb::AndOp::create(b, loc, mask, hit);
        next = comb::MuxOp::create(b, loc, hit, write.getData(), next);
        continue;
      }
      SmallVector<Value> parts;
      for (unsigned l = lanes; l-- > 0;) {
        Value bit = comb::ExtractOp::create(b, loc, mask, l, 1);
        Value laneHit = comb::AndOp::create(b, loc, bit, hit);
        Value data = comb::ExtractOp::create(b, loc, write.getData(),
                                             l * grain, grain);
        Value old = comb::ExtractOp::create(b, loc, next, l * grain, grain);
        parts.push_back(comb::MuxOp::create(b, loc, laneHit, data, old));
      }
      next = comb::ConcatOp::create(b, loc, parts);
    }
    entries[i].getNextMutable().assign(next);
  }

  // Reads: a constant address is its entry, another a select among them;
  // past the depth, zero.
  for (auto read : reads) {
    OpBuilder rb(read);
    Value address = read.getAddress();
    unsigned bits = cast<IntegerType>(address.getType()).getWidth();
    Value data;
    if (auto c = address.getDefiningOp<hw::ConstantOp>()) {
      uint64_t i = c.getValue().getZExtValue();
      data = i < depth ? Value(entries[i]) : zero;
    } else if (bits == 0) {
      data = entries.front();
    } else {
      uint64_t size = uint64_t(1) << bits;
      SmallVector<Value> elements;
      // hw.array_create lists the highest index first.
      for (uint64_t i = size; i-- > 0;)
        elements.push_back(i < depth ? Value(entries[i]) : zero);
      Value array = hw::ArrayCreateOp::create(rb, read.getLoc(), elements);
      data = hw::ArrayGetOp::create(rb, read.getLoc(), array, address);
    }
    read.getData().replaceAllUsesWith(data);
    read.erase();
  }
  for (auto write : writes)
    write.erase();
  memory.erase();
}

struct MemoriesToRegistersPass
    : public foldsim::impl::MemoriesToRegistersBase<MemoriesToRegistersPass> {
  using MemoriesToRegistersBase::MemoriesToRegistersBase;
  void runOnOperation() override {
    llvm::StringSet<> named;
    for (auto &m : memories)
      named.insert(m);
    for (auto module : getOperation().getOps<hw::HWModuleOp>()) {
      SmallVector<seq::FirMemOp> chosen;
      module.walk([&](seq::FirMemOp memory) {
        auto name = memory.getNameAttr();
        if (!name ||
            !named.contains((module.getModuleName() + ":" + name.getValue()).str()))
          return;
        auto why = ineligible(memory);
        if (!why.empty()) {
          memory.emitWarning() << "memory '" << name.getValue()
                               << "' stays a memory: " << why;
          return;
        }
        chosen.push_back(memory);
      });
      for (auto memory : chosen)
        lower(memory);
    }
  }
};

} // namespace
