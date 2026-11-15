//===- TableRoms.cpp - a register index's constant tables as one block ROM ===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The static fold reads many constant tables at one register index: a class's
// instance index selects, for each of its inputs, the row of its parent's
// array that instance reads (VX_tcu_fedp_tfr: 129 tables of 1,024 entries at
// one index), and foldsim-replicate-controls gives the index replicas, each
// reading its own copy of the tables. Built of LUTs, each table costs about a
// LUT a varying bit per 64 entries, again at every replica.
//
// Here the tables of one depth read at one index register X (or at any of
// its replicas: registers with the same next value, reset, reset value and
// clock) become one wide table read at X's next value (reset applied) and
// registered: R(t) = M[D_X(t-1)] = M[X(t)], so each old read is a slice of R
// in every host cycle (R's preset is M[X's preset]). The table is a wire with
// rom_style "block": a block RAM ROM with its output register, no LUTs. Exact
// by construction; one host cycle of nothing moves.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

namespace foldsim {
#define GEN_PASS_DEF_TABLEROMS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

struct TableRomsPass : public foldsim::impl::TableRomsBase<TableRomsPass> {
  using TableRomsBase::TableRomsBase;

  void runOnOperation() override {
    hw::HWModuleOp mod = getOperation();
    if (!mod.getModuleName().starts_with(prefix))
      return;
    // The index register's (next, reset, reset value, clock) and the depth.
    using Key = std::tuple<Value, Value, Value, Value, uint64_t>;
    struct Group {
      SmallVector<hw::ArrayGetOp> reads;
      seq::FirRegOp index;
    };
    llvm::MapVector<Key, Group> groups;
    mod.walk([&](hw::ArrayGetOp get) {
      auto type = hw::type_dyn_cast<hw::ArrayType>(get.getInput().getType());
      if (!type || !isa<IntegerType>(type.getElementType()) ||
          type.getNumElements() < minDepth)
        return;
      if (!get.getInput().getDefiningOp<hw::AggregateConstantOp>())
        return;
      auto reg = get.getIndex().getDefiningOp<seq::FirRegOp>();
      if (!reg || reg.getIsAsync() || reg->getBlock() != get->getBlock())
        return;
      Key key{reg.getNext(), reg.getReset(), reg.getResetValue(), reg.getClk(),
              type.getNumElements()};
      auto &g = groups[key];
      g.reads.push_back(get);
      if (!g.index)
        g.index = reg;
    });
    unsigned built = 0, merged = 0;
    uint64_t bits = 0;
    for (auto &[key, g] : groups) {
      uint64_t depth = std::get<4>(key);
      // The distinct tables, each at its offset.
      llvm::MapVector<Operation *, unsigned> offset;
      unsigned width = 0;
      for (auto get : g.reads) {
        auto *table = get.getInput().getDefiningOp();
        if (offset.count(table))
          continue;
        offset[table] = width;
        width += hw::getBitWidth(get.getType());
      }
      if (depth * width < minBits || width == 0)
        continue;
      // The wide table: field j of each table at its offset (fields are listed
      // from the highest index down, the same for every table of a depth).
      SmallVector<APInt> rows(depth, APInt(width, 0));
      for (auto [table, off] : offset) {
        auto fields = cast<hw::AggregateConstantOp>(table).getFields();
        for (uint64_t j = 0; j < depth; ++j) {
          auto v = cast<IntegerAttr>(fields[j]).getValue();
          rows[j].insertBits(v, off);
        }
      }
      OpBuilder b(g.reads.front());
      b.setInsertionPointAfter(g.index);
      Location loc = g.index.getLoc();
      auto elem = b.getIntegerType(width);
      SmallVector<Attribute> attrs;
      for (auto &r : rows)
        attrs.push_back(b.getIntegerAttr(elem, r));
      auto arrayType = hw::ArrayType::get(elem, depth);
      Value rom = hw::AggregateConstantOp::create(b, loc, arrayType, b.getArrayAttr(attrs));
      std::string name = (g.index.getName() + "_rom").str();
      auto wire = hw::WireOp::create(b, loc, rom, b.getStringAttr(name));
      sv::addSVAttributes(wire, {sv::SVAttributeAttr::get(b.getContext(), "rom_style",
                                                          R"("block")", false)});
      Value at = std::get<0>(key);
      if (Value reset = std::get<1>(key))
        at = comb::MuxOp::create(b, loc, reset, std::get<2>(key), at);
      // (A test hook: the mutant reads at the index itself, a host cycle late.)
      static const bool late = getenv("FOLDSIM_TABLE_ROMS_LATE");
      if (late)
        at = g.index;
      Value read = hw::ArrayGetOp::create(b, loc, wire, at);
      IntegerAttr preset;
      if (auto p = g.index.getPresetAttr()) {
        uint64_t i = p.getValue().getZExtValue();
        if (i < depth)
          preset = b.getIntegerAttr(elem, rows[depth - 1 - i]);
      }
      auto reg = seq::FirRegOp::create(b, loc, read, std::get<3>(key),
                                       b.getStringAttr(name + "_q"), hw::InnerSymAttr{}, preset);
      for (auto get : g.reads) {
        OpBuilder rb(get);
        unsigned off = offset.lookup(get.getInput().getDefiningOp());
        Value slice = comb::ExtractOp::create(rb, get.getLoc(), get.getType(), reg, off);
        get.replaceAllUsesWith(slice);
        get.erase();
        ++merged;
      }
      ++built;
      bits += depth * width;
    }
    if (built && verbose)
      llvm::errs() << "foldsim-table-roms: " << mod.getModuleName() << ": " << merged
                   << " table reads -> " << built << " block ROMs (" << bits << " bits)\n";
  }
};

} // namespace
