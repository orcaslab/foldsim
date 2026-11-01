//===- EstimateArea.cpp - count the cells a design maps to ----------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Beacon's area: the LUTs, flip-flops, distributed RAM, block RAM, UltraRAM
// and DSPs of a design mapped to a target's cells (foldsim/Beacon.h), placed
// by the rules Vivado infers them with, per instance of the elaborated design.
// On an ASIC library, cells, flip-flops and SRAM macros. An estimate before
// synthesis, not a utilization report.
//
//===----------------------------------------------------------------------===//

#include "BeaconOps.h"
#include "foldsim/Beacon.h"
#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <functional>

namespace foldsim {
#define GEN_PASS_DEF_ESTIMATEAREA
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;
using foldsim::BeaconTarget;
using foldsim::beacon::kMaxTableBits;
using foldsim::beacon::significantBits;
using foldsim::beacon::variableBits;
using foldsim::beacon::widthOf;

namespace {

/// The resources of a part of the design. On an ASIC, `lut` counts cells.
/// `lut` is the mapping's LUTs, before the target's lutScale; `memLut` the
/// read multiplexers of memories, and `ipLut` the measured LUTs of vendor IP,
/// neither of which a cut mapper covers.
struct Area {
  double lut = 0;
  double memLut = 0;
  double ipLut = 0;
  double lutram = 0;
  double bram = 0;
  double uram = 0;
  uint64_t ff = 0;
  uint64_t dsp = 0;
  uint64_t macros = 0;
  /// The flip-flops inside vendor IP, which a folded copy cannot store.
  uint64_t ipFf = 0;

  Area &operator+=(const Area &o) {
    lut += o.lut;
    memLut += o.memLut;
    ipLut += o.ipLut;
    lutram += o.lutram;
    bram += o.bram;
    uram += o.uram;
    ff += o.ff;
    dsp += o.dsp;
    macros += o.macros;
    ipFf += o.ipFf;
    return *this;
  }
  Area scaled(uint64_t n) const {
    Area a = *this;
    a.lut *= n;
    a.memLut *= n;
    a.ipLut *= n;
    a.lutram *= n;
    a.bram *= n;
    a.uram *= n;
    a.ff *= n;
    a.dsp *= n;
    a.macros *= n;
    a.ipFf *= n;
    return a;
  }
};

/// A memory and where synthesis places it.
struct Memory {
  StringRef name;
  uint64_t depth;
  unsigned width;
  unsigned reads;
  unsigned writes;
  unsigned readWrites;
  unsigned latency;
  std::string style;
  StringRef placement;
  Area area;
};

/// Vendor IP the netlist keeps as external modules, from the measured
/// synthesis of each core on its own: LUTs, flip-flops, DSPs.
struct Ip {
  StringRef name;
  unsigned lut, ff, dsp;
};
const Ip kIps[] = {
    {"xil_fma", 850, 1400, 2}, {"xil_fdiv", 800, 1400, 0},
    {"xil_fsqrt", 500, 900, 0}, {"xil_fmul", 90, 260, 2},
    {"xil_fadd", 260, 500, 0},
};

/// LUT6s of one output bit of a table over `bits` inputs: one for six, and a
/// LUT per 64 entries past that, the wide multiplexers between them free.
double tableLuts(unsigned bits) {
  return bits <= 6 ? 1.0 : double(uint64_t(1) << (bits - 6));
}

class AreaModel {
public:
  AreaModel(ModuleOp design, ArrayRef<std::string> blackBoxes,
            const BeaconTarget &device, double lutScale)
      : device(device), lutScale(lutScale) {
    for (auto module : design.getOps<hw::HWModuleLike>()) {
      if (llvm::is_contained(blackBoxes, module.getModuleName()))
        stubs.insert(module.getModuleNameAttr());
      modules[module.getModuleNameAttr()] = module;
    }
  }

  Operation *counting = nullptr;
  /// The resources of a module's own ops, its instances left out.
  const Area &self(hw::HWModuleOp module);
  /// The resources of a module and everything it instantiates.
  const Area &total(StringAttr name);
  /// Every instance below `top`, depth first: its path, module and areas.
  void writeInstances(raw_ostream &os, hw::HWModuleOp top);
  void writeMemories(raw_ostream &os, hw::HWModuleOp top);
  void writeModules(raw_ostream &os, ArrayRef<unsigned> stateThreads);
  void writeReport(raw_ostream &os, hw::HWModuleOp top);

private:
  void count(Operation *op, Area &area);
  Memory place(seq::FirMemOp memory, uint64_t depthScale = 1) const;
  Area memoryArea(uint64_t depth, unsigned width, unsigned reads,
                  unsigned ports, unsigned latency, StringRef style,
                  StringRef &placement) const;
  unsigned dsps(unsigned a, unsigned b) const;
  void visit(raw_ostream &os, StringAttr name, const std::string &path,
             unsigned depth, bool memoriesOnly);
  void columns(raw_ostream &os, const Area &a);
  double luts(const Area &a) const;

  const BeaconTarget &device;
  double lutScale;
  DenseMap<StringAttr, Operation *> modules;
  DenseSet<StringAttr> stubs;
  DenseMap<Operation *, Area> selves;
  DenseMap<StringAttr, Area> totals;
  DenseMap<Operation *, SmallVector<Memory>> memories;
  /// Per module: the LUTs (cells) each kind of op takes, and the ops
  /// counted as free.
  DenseMap<Operation *, llvm::StringMap<double>> kinds;
  DenseMap<Operation *, llvm::StringMap<uint64_t>> freeOps;
  /// Per module: every memory in it and below it, and its instances.
  DenseMap<StringAttr, llvm::MapVector<Operation *, uint64_t>> memoriesBelow;
  const llvm::MapVector<Operation *, uint64_t> &below(StringAttr name);
};

} // namespace

unsigned AreaModel::dsps(unsigned a, unsigned b) const {
  auto n = [](unsigned x, unsigned y, unsigned dx, unsigned dy) {
    return llvm::divideCeil(x, dx) * llvm::divideCeil(y, dy);
  };
  return std::min(n(a, b, device.dspA, device.dspB),
                  n(a, b, device.dspB, device.dspA));
}

/// Where a memory goes, as Vivado infers it, and what it takes there. A
/// `ram_style` the RTL asks for wins. Otherwise a read of latency 0 is
/// distributed RAM, and a registered read of at least 32 words and 2 Kb is
/// block RAM. Every FPGA family Beacon targets has 64-bit LUTs of distributed
/// RAM (RAM64M8: 64 x 7 bits in eight LUTs, RAM32M16: 32 x 14), 36 Kb block
/// RAMs that split in two (512 x 72 at their widest) and 288 Kb UltraRAMs
/// (4096 x 72).
Area AreaModel::memoryArea(uint64_t depth, unsigned width, unsigned reads,
                           unsigned ports, unsigned latency, StringRef style,
                           StringRef &placement) const {
  Area area;
  reads = std::max(reads, 1u);
  if (device.asic) {
    // Without a macro, a memory is flip-flops, a write decoder and a read
    // multiplexer tree of 2:1 cells per read port.
    placement = "flops";
    area.ff = depth * width;
    area.memLut = double(depth) * width * reads + depth;
    return area;
  }
  if (style == "registers") {
    placement = "flops";
    area.ff = depth * width;
    area.memLut = double(width) * reads * std::ceil((depth - 1) / 5.0);
    return area;
  }
  bool ultra = style == "ultra";
  bool block = ultra || style == "block" ||
               (style != "distributed" && latency >= 1 && depth >= 32 &&
                depth * width >= 2048);
  if (!block) {
    placement = "lutram";
    if (depth <= 32) {
      area.lutram = width * 0.57 * reads;
    } else {
      double words = std::ceil(depth / 64.0);
      area.lutram = width * words * 1.14 * reads;
      // A multiplexer tree over the 64-word LUTs of each bit.
      area.memLut = width * (words - 1) / 3 * reads;
    }
    return area;
  }
  // Two ports per block, a read-write port one of them; more need copies.
  double copies = std::max(1.0, std::ceil(ports / 2.0));
  if (ultra) {
    placement = "uram";
    area.uram = copies * llvm::divideCeil(depth, 4096) *
                llvm::divideCeil(width, 72);
    return area;
  }
  placement = "bram";
  struct Shape {
    uint64_t depth;
    unsigned width;
    double blocks;
  };
  static const Shape shapes[] = {
      {512, 72, 1},     {1024, 36, 1},   {2048, 18, 1},  {4096, 9, 1},
      {8192, 4, 1},     {16384, 2, 1},   {32768, 1, 1},  {512, 36, 0.5},
      {1024, 18, 0.5},  {2048, 9, 0.5},  {4096, 4, 0.5}, {8192, 2, 0.5},
      {16384, 1, 0.5},
  };
  double best = HUGE_VAL;
  for (const Shape &s : shapes)
    best = std::min(best, s.blocks * llvm::divideCeil(depth, s.depth) *
                              llvm::divideCeil(width, s.width));
  area.bram = copies * best;
  return area;
}

Memory AreaModel::place(seq::FirMemOp memory, uint64_t depthScale) const {
  auto type = memory.getType();
  Memory m;
  m.name = memory.getName() ? *memory.getName() : StringRef("memory");
  m.depth = type.getDepth() * depthScale;
  m.width = type.getWidth();
  m.reads = m.writes = m.readWrites = 0;
  m.latency = memory.getReadLatency();
  for (Operation *user : memory->getUsers()) {
    if (isa<seq::FirMemReadOp>(user)) {
      ++m.reads;
    } else if (isa<seq::FirMemWriteOp>(user)) {
      ++m.writes;
    } else if (isa<seq::FirMemReadWriteOp>(user)) {
      ++m.readWrites;
    }
  }
  if (auto attrs = sv::getSVAttributes(memory))
    for (auto attr : attrs.getAsRange<sv::SVAttributeAttr>())
      if (attr.getName() == "ram_style" && attr.getExpression())
        m.style = attr.getExpression().getValue().trim('"').str();
  m.area = memoryArea(m.depth, m.width, m.reads + m.readWrites,
                      m.reads + m.writes + m.readWrites, m.latency, m.style,
                      m.placement);
  return m;
}

void AreaModel::count(Operation *op, Area &area) {
  if (isa<comb::TruthTableOp>(op)) {
    area.lut += 1;
    return;
  }
  if (auto gate = dyn_cast<synth::aig::AndInverterOp>(op)) {
    unsigned inputs = gate.getInputs().size();
    area.lut += device.asic ? std::max(inputs, 2u) - 1
                            : std::ceil((std::max(inputs, 2u) - 1) / 5.0);
    return;
  }
  if (isa<seq::FirRegOp, seq::CompRegOp, seq::CompRegClockEnabledOp>(op)) {
    area.ff += widthOf(op->getResult(0));
    return;
  }
  if (isa<comb::AddOp, comb::SubOp>(op)) {
    // A LUT per bit before the carry chain; on an ASIC, a prefix adder.
    unsigned w = widthOf(op->getResult(0));
    area.lut += device.asic ? w * (2.0 + std::ceil(std::log2(std::max(w, 2u))))
                            : w;
    return;
  }
  if (auto cmp = dyn_cast<comb::ICmpOp>(op)) {
    unsigned w = widthOf(cmp.getLhs());
    if (cmp.getPredicate() == comb::ICmpPredicate::eq ||
        cmp.getPredicate() == comb::ICmpPredicate::ne ||
        cmp.getPredicate() == comb::ICmpPredicate::ceq ||
        cmp.getPredicate() == comb::ICmpPredicate::cne ||
        cmp.getPredicate() == comb::ICmpPredicate::weq ||
        cmp.getPredicate() == comb::ICmpPredicate::wne) {
      area.lut += std::ceil(w / 3.0) + std::ceil(w / 18.0);
      return;
    }
    // Two bits per LUT into the carry chain.
    area.lut += device.asic ? 2.0 * w : std::ceil(w / 2.0);
    return;
  }
  if (isa<comb::MulOp>(op)) {
    unsigned w = widthOf(op->getResult(0));
    unsigned a = 0, b = 0, bits = 0;
    for (auto operand : op->getOperands()) {
      unsigned v = variableBits(operand);
      bits += v;
      if (v > a) {
        b = a;
        a = v;
      } else if (v > b) {
        b = v;
      }
    }
    if (!b) {
      // Times a constant: shifts and adders, a LUT per bit.
      area.lut += device.asic ? 3.0 * w : w;
      return;
    }
    if (bits <= kMaxTableBits) {
      // A table per output bit; a product has at most as many as its inputs.
      area.lut += std::min(w, bits) * tableLuts(bits);
      return;
    }
    if (device.asic) {
      // Partial products, a Dadda tree of full adders (two cells each) and
      // the final adder.
      area.lut += 3.0 * a * b;
      return;
    }
    area.dsp += dsps(a, b);
    return;
  }
  if (isa<comb::DivUOp, comb::DivSOp, comb::ModUOp, comb::ModSOp>(op)) {
    unsigned w = widthOf(op->getResult(0));
    unsigned dividend = variableBits(op->getOperand(0));
    unsigned divisor = variableBits(op->getOperand(1));
    // A quotient or remainder has no more bits than the dividend.
    unsigned out = std::min(w, dividend);
    if (dividend + divisor <= kMaxTableBits)
      area.lut += out * tableLuts(dividend + divisor);
    else if (!divisor)
      area.lut += 2.0 * out;
    else
      // One subtracter per quotient bit.
      area.lut += device.asic ? 3.0 * out * w : double(out) * w;
    return;
  }
  if (isa<hw::ConstantOp, hw::AggregateConstantOp, comb::ExtractOp,
          comb::ConcatOp, comb::ReplicateOp, hw::BitcastOp, hw::WireOp,
          hw::OutputOp, hw::InstanceOp, hw::StructCreateOp,
          hw::StructExtractOp, hw::StructExplodeOp, hw::ArrayCreateOp,
          hw::ArrayConcatOp, hw::ArraySliceOp, seq::FirMemOp,
          seq::FirMemReadOp, seq::FirMemWriteOp, seq::FirMemReadWriteOp,
          seq::FromClockOp, seq::ToClockOp, seq::ConstClockOp>(op))
    return;
  if (op->getNumResults() == 0)
    return;
  ++freeOps[counting][op->getName().getStringRef()];
}

const Area &AreaModel::self(hw::HWModuleOp module) {
  auto it = selves.find(module);
  if (it != selves.end())
    return it->second;
  Area area;
  SmallVector<Memory> mems;
  counting = module;
  module.walk([&](Operation *op) {
    if (auto memory = dyn_cast<seq::FirMemOp>(op)) {
      mems.push_back(place(memory));
      area += mems.back().area;
      return;
    }
    double before = area.lut;
    count(op, area);
    if (area.lut != before)
      kinds[module][op->getName().getStringRef()] += area.lut - before;
  });
  memories[module] = std::move(mems);
  return selves[module] = area;
}

const Area &AreaModel::total(StringAttr name) {
  auto it = totals.find(name);
  if (it != totals.end())
    return it->second;
  Area area;
  Operation *op = modules.lookup(name);
  if (stubs.contains(name) || !op) {
    // A harness stub: its area is not the design's.
  } else if (auto module = dyn_cast<hw::HWModuleOp>(op)) {
    area = self(module);
    module.walk([&](hw::InstanceOp inst) {
      area += total(inst.getModuleNameAttr().getAttr());
    });
  } else {
    StringRef ext = name.getValue();
    const Ip *ip = nullptr;
    for (const Ip &candidate : kIps)
      if (ext == candidate.name ||
          (ext.starts_with(candidate.name) &&
           ext.drop_front(candidate.name.size()).starts_with("_")))
        ip = &candidate;
    if (device.asic) {
      // An SRAM macro or another black box of the library.
      area.macros = 1;
    } else if (ip) {
      area.ipLut = ip->lut;
      area.ff = area.ipFf = ip->ff;
      area.dsp = ip->dsp;
    }
  }
  return totals[name] = area;
}

double AreaModel::luts(const Area &a) const {
  return a.lut * lutScale + a.memLut + a.ipLut;
}

void AreaModel::columns(raw_ostream &os, const Area &a) {
  os << llvm::format("%.0f,%.0f,%llu,%.1f,%.0f,%llu,%llu", luts(a), a.lutram,
                     (unsigned long long)a.ff, a.bram, a.uram,
                     (unsigned long long)a.dsp, (unsigned long long)a.macros);
}

void AreaModel::visit(raw_ostream &os, StringAttr name, const std::string &path,
                      unsigned depth, bool memoriesOnly) {
  auto module = dyn_cast_or_null<hw::HWModuleOp>(modules.lookup(name));
  if (!module || stubs.contains(name))
    return;
  if (memoriesOnly) {
    self(module);
    for (const Memory &m : memories[module]) {
      os << path << "," << m.name << "," << m.depth << "," << m.width << ","
         << m.reads << "," << m.writes << "," << m.readWrites << ","
         << m.latency << ","
         << (m.style.empty() ? "-" : m.style) << "," << m.placement << ",";
      columns(os, m.area);
      os << "\n";
    }
  } else {
    os << depth << "," << path << "," << name.getValue() << ",";
    columns(os, self(module));
    os << ",";
    columns(os, total(name));
    os << "\n";
  }
  module.walk([&](hw::InstanceOp inst) {
    StringAttr child = inst.getModuleNameAttr().getAttr();
    std::string sub =
        (path.empty() ? "" : path + "/") + inst.getInstanceName().str();
    if (!isa_and_nonnull<hw::HWModuleOp>(modules.lookup(child))) {
      if (!memoriesOnly && !stubs.contains(child)) {
        os << depth + 1 << "," << sub << "," << child.getValue() << ",";
        columns(os, total(child));
        os << ",";
        columns(os, total(child));
        os << "\n";
      }
      return;
    }
    visit(os, child, sub, depth + 1, memoriesOnly);
  });
}

void AreaModel::writeInstances(raw_ostream &os, hw::HWModuleOp top) {
  os << "depth,path,module,lut,lutram,ff,bram,uram,dsp,macros,"
        "total_lut,total_lutram,total_ff,total_bram,total_uram,total_dsp,"
        "total_macros\n";
  visit(os, top.getNameAttr(), "", 0, false);
}

void AreaModel::writeMemories(raw_ostream &os, hw::HWModuleOp top) {
  os << "path,name,depth,width,reads,writes,read_writes,latency,style,"
        "placement,"
        "lut,lutram,ff,bram,uram,dsp,macros\n";
  visit(os, top.getNameAttr(), "", 0, true);
}

const llvm::MapVector<Operation *, uint64_t> &
AreaModel::below(StringAttr name) {
  auto it = memoriesBelow.find(name);
  if (it != memoriesBelow.end())
    return it->second;
  llvm::MapVector<Operation *, uint64_t> mems;
  auto module = dyn_cast_or_null<hw::HWModuleOp>(modules.lookup(name));
  if (module && !stubs.contains(name))
    module.walk([&](Operation *op) {
      if (isa<seq::FirMemOp>(op))
        mems[op] += 1;
      else if (auto inst = dyn_cast<hw::InstanceOp>(op))
        for (auto &[mem, n] : below(inst.getModuleNameAttr().getAttr()))
          mems[mem] += n;
    });
  return memoriesBelow[name] = std::move(mems);
}

void AreaModel::writeModules(raw_ostream &os,
                             ArrayRef<unsigned> stateThreads) {
  // Per module and everything below it: its area (`lut` scaled, and apart
  // the mapping's LUTs before the scale and the LUTs no mapping covers), its
  // register bits and memories at `stateThreads` times their depth, as a
  // folded copy stores the state of that many instances (the registers as
  // one memory, a word each).
  llvm::json::Object out;
  out["target"] = device.name;
  out["lut_scale"] = lutScale;
  llvm::json::Object mods;
  for (auto &[name, op] : modules) {
    auto module = dyn_cast<hw::HWModuleOp>(op);
    if (!module || stubs.contains(name))
      continue;
    const Area &a = total(name);
    llvm::json::Object m{{"lut", luts(a)},
                         {"lut_mapped", a.lut},
                         {"lut_fixed", a.memLut + a.ipLut},
                         {"lutram", a.lutram},
                         {"ff", int64_t(a.ff)},
                         {"ip_ff", int64_t(a.ipFf)},
                         {"bram", a.bram},
                         {"uram", a.uram},
                         {"dsp", int64_t(a.dsp)}};
    if (!stateThreads.empty()) {
      const auto &mems = below(name);
      llvm::json::Object state;
      for (unsigned threads : stateThreads) {
        StringRef placement;
        Area regs =
            memoryArea(threads, a.ff - a.ipFf, 1, 2, 0, "", placement);
        Area mem;
        for (auto &[memory, copies] : mems)
          mem += place(cast<seq::FirMemOp>(memory), threads)
                     .area.scaled(copies);
        state[std::to_string(threads)] = llvm::json::Object{
            {"reg_lutram", regs.lutram + regs.memLut},
            {"mem_lutram", mem.lutram + mem.memLut},
            {"mem_bram", mem.bram}};
      }
      m["state"] = std::move(state);
    }
    mods[name.getValue()] = std::move(m);
  }
  out["modules"] = std::move(mods);
  os << llvm::formatv("{0:1}", llvm::json::Value(std::move(out))) << "\n";
}

void AreaModel::writeReport(raw_ostream &os, hw::HWModuleOp top) {
  const Area &a = total(top.getNameAttr());
  if (device.asic) {
    os << llvm::format("%s on %s: %.0f cells, %llu flip-flops, %llu macros\n",
                       top.getModuleName().str().c_str(),
                       device.name.str().c_str(), luts(a),
                       (unsigned long long)a.ff,
                       (unsigned long long)a.macros);
  } else {
    os << llvm::format("%s on %s: %.0f LUTs (%.0f logic, %.0f distributed "
                       "RAM), %llu flip-flops, %.1f block RAMs, %.0f "
                       "UltraRAMs, %llu DSPs\n",
                       top.getModuleName().str().c_str(),
                       device.name.str().c_str(), luts(a) + a.lutram, luts(a),
                       a.lutram, (unsigned long long)a.ff, a.bram, a.uram,
                       (unsigned long long)a.dsp);
    os << llvm::format("logic: %.0f LUTs mapped x %.3f (lutScale) + %.0f of "
                       "memory multiplexers + %.0f of vendor IP\n",
                       a.lut, lutScale, a.memLut, a.ipLut);
  }
  // The ops, vendor IP and free ops of every instance below `top`.
  llvm::StringMap<double> byKind;
  llvm::StringMap<uint64_t> externs, free;
  std::function<void(StringAttr, uint64_t)> sum = [&](StringAttr n,
                                                      uint64_t copies) {
    auto mod = dyn_cast_or_null<hw::HWModuleOp>(modules.lookup(n));
    if (!mod || stubs.contains(n))
      return;
    self(mod);
    for (auto &k : kinds[mod])
      byKind[k.getKey()] += k.getValue() * copies;
    for (auto &k : freeOps[mod])
      free[k.getKey()] += k.getValue() * copies;
    mod.walk([&](hw::InstanceOp inst) {
      StringAttr child = inst.getModuleNameAttr().getAttr();
      if (isa_and_nonnull<hw::HWModuleOp>(modules.lookup(child)))
        sum(child, copies);
      else if (!stubs.contains(child))
        externs[child.getValue()] += copies;
    });
  };
  sum(top.getNameAttr(), 1);
  auto sorted = [](const auto &map) {
    SmallVector<StringRef> keys;
    for (auto &e : map)
      keys.push_back(e.getKey());
    llvm::sort(keys);
    return keys;
  };
  os << (device.asic ? "cells" : "mapped LUTs") << " by op:";
  for (StringRef k : sorted(byKind))
    os << llvm::format(" %s %.0f", k.str().c_str(), byKind.lookup(k));
  os << "\n";
  if (!externs.empty()) {
    os << "external modules:";
    for (StringRef k : sorted(externs))
      os << " " << k << " x" << externs.lookup(k);
    os << "\n";
  }
  if (!free.empty()) {
    os << "ops counted as free:";
    for (StringRef k : sorted(free))
      os << " " << k << " x" << free.lookup(k);
    os << "\n";
  }
}

namespace {
struct EstimateAreaPass
    : public foldsim::impl::EstimateAreaBase<EstimateAreaPass> {
  using EstimateAreaBase::EstimateAreaBase;
  void runOnOperation() override;
};
} // namespace

void EstimateAreaPass::runOnOperation() {
  auto design = getOperation();
  hw::HWModuleOp top;
  if (!topName.empty()) {
    top = design.lookupSymbol<hw::HWModuleOp>(topName);
    if (!top) {
      design.emitError("no module '") << topName << "'";
      return signalPassFailure();
    }
  } else {
    DenseSet<StringAttr> instantiated;
    design.walk([&](hw::InstanceOp inst) {
      instantiated.insert(inst.getModuleNameAttr().getAttr());
    });
    for (auto module : design.getOps<hw::HWModuleOp>())
      if (!instantiated.contains(module.getNameAttr())) {
        top = module;
        break;
      }
  }
  if (!top && (!output.empty() || !memoryTable.empty() || !report.empty())) {
    design.emitError("no top module");
    return signalPassFailure();
  }

  const BeaconTarget *device = foldsim::lookupBeaconTarget(target);
  if (!device) {
    foldsim::emitUnknownBeaconTarget(design, target);
    return signalPassFailure();
  }
  AreaModel model(design, blackBoxes, *device,
                  lutScale >= 0 ? double(lutScale) : device->lutScale);

  auto write = [&](StringRef file, auto fn) -> LogicalResult {
    if (file.empty())
      return success();
    std::string error;
    auto out = openOutputFile(file, &error);
    if (!out) {
      design.emitError(error);
      return failure();
    }
    fn(out->os());
    out->keep();
    return success();
  };
  if (failed(write(output,
                   [&](raw_ostream &os) { model.writeInstances(os, top); })) ||
      failed(write(memoryTable,
                   [&](raw_ostream &os) { model.writeMemories(os, top); })) ||
      failed(write(moduleTable,
                   [&](raw_ostream &os) {
                     model.writeModules(os, stateThreads);
                   })) ||
      failed(write(report,
                   [&](raw_ostream &os) { model.writeReport(os, top); })))
    return signalPassFailure();
  markAllAnalysesPreserved();
}
