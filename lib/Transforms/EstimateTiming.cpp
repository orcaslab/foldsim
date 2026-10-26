//===- EstimateTiming.cpp - rank register-to-register paths by depth ------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A timing screen: the logic levels of the longest path to every end point
// of an elaborated, LUT-mapped design, counted as Vivado counts them on the
// V80. It finds the depth outliers before a Vivado run; it predicts neither
// slack nor wire delay.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace foldsim {
#define GEN_PASS_DEF_ESTIMATETIMING
#define GEN_PASS_DEF_FOLDZEROWIDTH
#define GEN_PASS_DEF_KEEPMODULES
#define GEN_PASS_DEF_LOWEREQUALITY
#define GEN_PASS_DEF_LOWERCONSTANTDIVISION
#define GEN_PASS_DEF_SPLITMASKEDMEMORIES
#define GEN_PASS_DEF_SYNCASYNCRESETS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// What drives a bit: logic whose levels follow from its inputs, or a start
/// point of a path.
enum class Kind : uint8_t {
  Lut,
  Gate,
  Arith,
  Order,
  Equal,
  Mul,
  Div,
  Table,
  AsyncRead,
  Other,
  // Start points.
  Reg,
  SyncRead,
  Extern,
  Stub,
  Port,
  Clock,
};

bool isStart(Kind kind) { return kind >= Kind::Reg; }

/// Vivado does not retime registers through distributed RAMs and DSPs.
bool isPinned(Kind kind) {
  return kind == Kind::AsyncRead || kind == Kind::Mul;
}

StringRef kindName(Kind kind) {
  switch (kind) {
  case Kind::Lut:
    return "lut";
  case Kind::Gate:
    return "gate";
  case Kind::Arith:
    return "carry";
  case Kind::Order:
    return "cmp";
  case Kind::Equal:
    return "eq";
  case Kind::Mul:
    return "dsp";
  case Kind::Div:
    return "div";
  case Kind::Table:
    return "table";
  case Kind::AsyncRead:
    return "lutram";
  case Kind::Other:
    return "other";
  case Kind::Reg:
    return "reg";
  case Kind::SyncRead:
    return "bram";
  case Kind::Extern:
    return "ip";
  case Kind::Stub:
    return "stub";
  case Kind::Port:
    return "port";
  case Kind::Clock:
    return "clock";
  }
  llvm_unreachable("unknown kind");
}

/// A bit driven by `value` in the elaborated instance `ctx`.
struct Node {
  Value value;
  unsigned ctx;
  unsigned pred = ~0u;
  uint32_t fanout = 0;
  uint32_t bit;
  int16_t levels = 0;
  Kind kind;
  uint8_t state = 0; // 0: new, 1: waiting for its inputs, 2: done.
};

/// An elaborated instance; 0 is the top.
struct Context {
  unsigned parent;
  hw::InstanceOp inst;
  hw::HWModuleOp module;
  unsigned depth;
};

/// An end point bit; `reg` is the register it is the input of, if Vivado can
/// retime that register. A `boundary` end point is a top output or an input
/// of a stub, which the harness registers. A `registered` end point is a
/// register's data input, which a DSP absorbs when a product feeds it.
struct EndPoint {
  unsigned node;
  unsigned ctx;
  Operation *op;
  StringRef port;
  uint32_t bit;
  unsigned reg;
  bool boundary = false;
  bool registered = false;
  int16_t levels = 0;
  int16_t retimed = 0;
};

/// An input of a node: the levels from it to the node, and whether the node
/// loads it (an adder bit reaches the lower bits' inputs over the carry
/// chain, not as their load).
struct Edge {
  unsigned in;
  int levels;
  bool load;
};

constexpr unsigned kNone = ~0u;

unsigned widthOf(Value value) {
  int64_t width = hw::getBitWidth(value.getType());
  return width < 0 ? 0 : width;
}

/// ceil(log6(n)): the levels of a LUT6 tree over n signals.
/// Whether module `name` is `wanted` or one of its specializations
/// (`<wanted>_<n>`).
bool matchesModule(StringRef name, ArrayRef<std::string> wanted) {
  for (auto &w : wanted)
    if (name == w || (name.starts_with(w) && name.size() > w.size() &&
                      name[w.size()] == '_' &&
                      llvm::all_of(name.drop_front(w.size() + 1),
                                   llvm::isDigit)))
      return true;
  return false;
}

unsigned lut6Depth(unsigned n) {
  unsigned levels = 0;
  for (unsigned reach = 1; reach < n; reach *= 6)
    ++levels;
  return levels;
}

/// Levels from an input bit of an adder or comparator to an output bit: the
/// LUT that computes propagate and generate, and the 8-bit carry blocks
/// (CARRY8 or LOOKAHEAD8) between the input's block and the output's.
int carryLevels(unsigned in, unsigned out) { return 2 + out / 8 - in / 8; }

/// A device's family. On an FPGA, levels are as Vivado counts them, and the
/// two families differ only in their DSPs: both have LUT6s, 8-bit carry
/// blocks and 64-word distributed RAMs. On an ASIC library, levels are gate
/// stages of the and-inverter graph.
struct Device {
  /// The operand widths of a DSP's multiplier.
  unsigned dspA, dspB;
  /// Levels from the fabric to a product the DSP registers, from there out
  /// to the fabric, and through each further DSP of a cascade.
  int dspIn, dspOut, dspCascade;
  /// An ASIC library, and its delays in ps: a stage, a flip-flop's clock to
  /// output and setup (hw/syn/yosys/liberty_stage.py), and what a path adds
  /// beyond its stages, fitted on OpenSTA (buffering, the inverter after a
  /// flip-flop's QN, slew).
  bool asic = false;
  int stagePs = 0, clk2qPs = 0, setupPs = 0, pathPs = 0;
};

/// The V80 (Versal, DSP58): from a registered product, Vivado counts M_DATA,
/// ALUMUX, ALUADD, OUTPUT and the cells before the multiplier it does not
/// absorb; a cascaded DSP adds ALUMUX, ALUADD and OUTPUT. The U55C
/// (Virtex UltraScale+ HBM, DSP48E2) has one ALU cell and an 18-bit B port.
const Device kVersal = {27, 24, 1, 6, 3};
const Device kVirtexUltraScalePlusHBM = {27, 18, 1, 5, 2};
/// ASAP7 7.5-track RVT at the TT corner, as hw/syn/yosys builds with it: a
/// cell at fanout 3 and its own slew, a DFFHQNx1's clock to output and setup,
/// and 167 ps per path, the median over five asic_gate DUTs. Products are
/// multipliers of cells, not DSPs.
const Device kAsap7 = {0, 0, 0, 0, 0, true, 24, 58, 6, 167};

/// The device of a family, by the names of hw/syn/xilinx/platforms (the V80's
/// and the U55C's) and of hw/syn/yosys's PDK (asap7).
const Device *lookupDevice(StringRef family) {
  return llvm::StringSwitch<const Device *>(family)
      .Case("versal", &kVersal)
      .Case("virtexuplusHBM", &kVirtexUltraScalePlusHBM)
      .Case("asap7", &kAsap7)
      .Default(nullptr);
}

/// The most bits a divider can depend on for synthesis to write it out as a
/// table over them; also the dividend widths whose quotient and remainder by
/// a constant foldsim-lower-constant-division writes out.
constexpr unsigned kMaxTableBits = 10;

/// The most nodes searched for the bits a divider depends on.
constexpr unsigned kMaxSupportCone = 4096;

/// Levels of a table over `bits` inputs: a LUT6 for six, and a level of 4:1
/// multiplexers per two more.
int tableLevels(unsigned bits) { return bits <= 6 ? 1 : 1 + (bits - 5) / 2; }

/// The low bits of `value` that are not known zero: a zero-extension is a
/// concatenation under a zero constant.
unsigned significantBits(Value value) {
  unsigned width = widthOf(value);
  if (auto concat = value.getDefiningOp<comb::ConcatOp>())
    if (auto top = concat.getInputs().front().getDefiningOp<hw::ConstantOp>())
      if (top.getValue().isZero())
        return width - widthOf(top);
  return width;
}

/// The bits of `value` a product depends on: none for a constant.
unsigned variableBits(Value value) {
  return value.getDefiningOp<hw::ConstantOp>() ? 0 : significantBits(value);
}

class Timing {
public:
  /// Instances of `blackBoxes` are external modules, as a harness's stubs.
  Timing(ModuleOp top, ArrayRef<std::string> blackBoxes, const Device &device)
      : device(device) {
    for (auto module : top.getOps<hw::HWModuleLike>()) {
      if (llvm::is_contained(blackBoxes, module.getModuleName()))
        stubs.insert(module.getModuleNameAttr());
      else
        modules[module.getModuleNameAttr()] = module;
    }
  }

  LogicalResult run(hw::HWModuleOp top, unsigned retimeLevels);
  void writeEndPoints(raw_ostream &os, unsigned minLevels);
  void writeInstances(raw_ostream &os);
  void writePorts(raw_ostream &os, ArrayRef<std::string> modules);
  void writePaths(raw_ostream &os, unsigned minLevels);
  void writeReport(raw_ostream &os, unsigned topPaths,
                   ArrayRef<std::string> trace);

private:
  unsigned child(unsigned ctx, hw::InstanceOp inst, hw::HWModuleOp module);
  unsigned node(unsigned ctx, Value value, unsigned bit, Kind kind);
  unsigned resolve(unsigned ctx, Value value, unsigned bit);
  Kind classify(Operation *op);
  int arith(unsigned in, unsigned out) const;
  int table(unsigned bits) const;
  int multiplier(unsigned a, unsigned b) const;
  void operands(unsigned id, SmallVectorImpl<Edge> &edges);
  void inputs(unsigned id, SmallVectorImpl<Edge> &edges);
  ArrayRef<unsigned> support(unsigned ctx, Operation *op, bool &small);
  void evaluate(unsigned root);
  void collect(unsigned ctx);
  void addEndPoints(unsigned ctx, Operation *op, StringRef port, Value value,
                    bool boundary = false);
  void addRegister(unsigned ctx, Operation *op, Value reg, Value next,
                   bool retimable);
  void retime(unsigned levels);
  unsigned startOf(unsigned ctx, unsigned id, uint32_t &fanout,
                   unsigned &crossings);
  void printName(raw_ostream &os, unsigned id);
  void printOp(raw_ostream &os, Operation *op, StringRef port, uint32_t bit);
  void printPath(raw_ostream &os, const EndPoint &end, unsigned rank);
  const std::string &path(unsigned ctx);

  const Device &device;
  DenseMap<StringAttr, Operation *> modules;
  DenseSet<StringAttr> stubs;
  std::vector<Context> contexts;
  DenseMap<std::pair<unsigned, Operation *>, unsigned> children;
  std::vector<std::string> paths;
  std::vector<Node> nodes;
  DenseMap<std::tuple<unsigned, void *, unsigned>, unsigned> ids;
  std::vector<EndPoint> endPoints;
  /// Evaluated nodes, each after its inputs.
  std::vector<unsigned> order;
  /// The input node of each register bit Vivado can retime (kNone for a
  /// constant).
  DenseMap<unsigned, unsigned> regInputs;
  /// The LUTs that read the register they feed.
  DenseSet<unsigned> feedbacks;
  /// The bits each divider depends on, if there are at most kMaxTableBits.
  DenseMap<std::pair<unsigned, Operation *>,
           std::optional<SmallVector<unsigned>>>
      supports;
  unsigned retimeLevels = 0;
  llvm::StringMap<uint64_t> otherOps;
  uint64_t loops = 0;
  SmallVector<std::pair<unsigned, unsigned>> loopSamples;
};

} // namespace

unsigned Timing::child(unsigned ctx, hw::InstanceOp inst,
                       hw::HWModuleOp module) {
  auto [it, inserted] = children.try_emplace({ctx, inst}, contexts.size());
  if (inserted)
    contexts.push_back({ctx, inst, module, contexts[ctx].depth + 1});
  return it->second;
}

unsigned Timing::node(unsigned ctx, Value value, unsigned bit, Kind kind) {
  auto [it, inserted] = ids.try_emplace(
      {ctx, value.getAsOpaquePointer(), bit}, (unsigned)nodes.size());
  if (inserted) {
    Node n;
    n.value = value;
    n.ctx = ctx;
    n.bit = bit;
    n.kind = kind;
    nodes.push_back(n);
  }
  return it->second;
}

Kind Timing::classify(Operation *op) {
  if (isa<comb::TruthTableOp>(op))
    return Kind::Lut;
  if (isa<synth::aig::AndInverterOp>(op))
    return Kind::Gate;
  if (isa<comb::AddOp, comb::SubOp>(op))
    return Kind::Arith;
  if (auto cmp = dyn_cast<comb::ICmpOp>(op)) {
    switch (cmp.getPredicate()) {
    case comb::ICmpPredicate::eq:
    case comb::ICmpPredicate::ne:
    case comb::ICmpPredicate::ceq:
    case comb::ICmpPredicate::cne:
    case comb::ICmpPredicate::weq:
    case comb::ICmpPredicate::wne:
      return Kind::Equal;
    default:
      return Kind::Order;
    }
  }
  if (isa<comb::MulOp>(op)) {
    // A product of few variable bits, as an index times a constant, is a
    // table of constants in LUTs; synthesis narrows it before choosing DSPs.
    unsigned bits = 0;
    for (auto operand : op->getOperands())
      bits += variableBits(operand);
    return bits <= kMaxTableBits ? Kind::Table : Kind::Mul;
  }
  if (isa<comb::DivUOp, comb::DivSOp, comb::ModUOp, comb::ModSOp>(op))
    return Kind::Div;
  if (isa<seq::FirRegOp, seq::CompRegOp, seq::CompRegClockEnabledOp>(op))
    return Kind::Reg;
  if (auto read = dyn_cast<seq::FirMemReadOp>(op))
    return read.getMemory().getDefiningOp<seq::FirMemOp>().getReadLatency()
               ? Kind::SyncRead
               : Kind::AsyncRead;
  if (isa<seq::FirMemReadWriteOp>(op))
    return Kind::SyncRead;
  if (isa<seq::FromClockOp>(op))
    return Kind::Clock;
  ++otherOps[op->getName().getStringRef()];
  return Kind::Other;
}

/// The node of the bit `bit` of `value` in instance `ctx`, seen through
/// wiring, extracts, concatenations and instance boundaries; kNone for a
/// constant.
unsigned Timing::resolve(unsigned ctx, Value value, unsigned bit) {
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (ctx == 0)
        return node(ctx, value, bit, Kind::Port);
      auto &c = contexts[ctx];
      value = c.inst.getOperand(arg.getArgNumber());
      ctx = c.parent;
      continue;
    }
    Operation *op = value.getDefiningOp();
    if (op->hasTrait<OpTrait::ConstantLike>())
      return kNone;
    if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
      value = extract.getInput();
      bit += extract.getLowBit();
      continue;
    }
    if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
      for (auto input : llvm::reverse(concat.getInputs())) {
        unsigned width = widthOf(input);
        if (bit < width) {
          value = input;
          break;
        }
        bit -= width;
      }
      continue;
    }
    if (auto replicate = dyn_cast<comb::ReplicateOp>(op)) {
      value = replicate.getInput();
      bit %= widthOf(value);
      continue;
    }
    if (auto bitcast = dyn_cast<hw::BitcastOp>(op)) {
      value = bitcast.getInput();
      continue;
    }
    if (auto wire = dyn_cast<hw::WireOp>(op)) {
      value = wire.getInput();
      continue;
    }
    if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
      auto module = dyn_cast_or_null<hw::HWModuleOp>(
          modules.lookup(inst.getModuleNameAttr().getAttr()));
      if (!module)
        return node(ctx, value, bit,
                    stubs.contains(inst.getModuleNameAttr().getAttr())
                        ? Kind::Stub
                        : Kind::Extern);
      ctx = child(ctx, inst, module);
      value = module.getBodyBlock()->getTerminator()->getOperand(
          cast<OpResult>(value).getResultNumber());
      continue;
    }
    auto key = std::make_tuple(ctx, value.getAsOpaquePointer(), bit);
    if (auto it = ids.find(key); it != ids.end())
      return it->second;
    return node(ctx, value, bit, classify(op));
  }
}

/// An adder or comparator bit of at most six input bits is one LUT6, not a
/// carry chain: the bits of a popcount's narrow adders, or a compare of two
/// 3-bit counts.
static void fitLut(SmallVectorImpl<Edge> &edges, size_t first) {
  if (edges.size() - first > 6)
    return;
  for (auto &edge : llvm::drop_begin(edges, first))
    edge.levels = 1;
}

/// Levels from an input bit of an adder to an output bit: on an FPGA a LUT
/// and the carry blocks; on an ASIC a prefix adder, a propagate-generate
/// gate, a prefix tree over the bits between them and the sum gate.
int Timing::arith(unsigned in, unsigned out) const {
  if (device.asic)
    return 2 + (int)llvm::Log2_32_Ceil(out - in + 1);
  return carryLevels(in, out);
}

/// Levels of a table over `bits` inputs: LUTs on an FPGA, a tree of 2:1
/// multiplexers on an ASIC.
int Timing::table(unsigned bits) const {
  return device.asic ? std::max(1u, bits) : tableLevels(bits);
}

/// Levels of an ASIC multiplier of `a` x `b` bits: the partial products, a
/// Dadda tree of full adders (two gates each) over the narrower operand's
/// rows, and a prefix adder over the product.
int Timing::multiplier(unsigned a, unsigned b) const {
  unsigned rows = std::min(a, b), stages = 0;
  for (unsigned height = 2; height < rows; height = height * 3 / 2)
    ++stages;
  return 1 + 2 * stages + arith(0, a + b - 1);
}

/// The inputs of a node that is not a start point, each with the levels from
/// it to the node. A product reaches the fabric through its DSP's output
/// stage, unless the node is the enable or feedback logic of a register the
/// DSP absorbs with it.
void Timing::inputs(unsigned id, SmallVectorImpl<Edge> &edges) {
  size_t first = edges.size();
  operands(id, edges);
  bool absorbed = feedbacks.contains(id);
  for (auto &edge : llvm::drop_begin(edges, first))
    if (nodes[edge.in].kind == Kind::Mul)
      edge.levels = absorbed ? 0 : edge.levels + device.dspOut;
}

/// The inputs of a node as its operation sees them, before its DSP inputs'
/// output stages.
void Timing::operands(unsigned id, SmallVectorImpl<Edge> &edges) {
  Node n = nodes[id];
  Operation *op = n.value.getDefiningOp();
  auto add = [&](Value operand, unsigned bit, int levels, bool load = true) {
    unsigned in = resolve(n.ctx, operand, bit);
    if (in != kNone)
      edges.push_back({in, levels, load});
  };
  switch (n.kind) {
  case Kind::Lut:
    for (auto operand : op->getOperands())
      add(operand, 0, 1);
    return;
  case Kind::Gate:
    for (auto operand : op->getOperands())
      add(operand, n.bit, 1);
    return;
  case Kind::Arith: {
    size_t first = edges.size();
    for (auto operand : op->getOperands())
      for (unsigned bit = 0; bit <= n.bit; ++bit)
        add(operand, bit, arith(bit, n.bit), bit == n.bit);
    if (!device.asic)
      fitLut(edges, first);
    return;
  }
  case Kind::Order: {
    // The chain ends at the highest bit that is not constant in both
    // operands: constant bits above it fold away (a compare of a
    // zero-extended index against a bound is a narrow compare).
    SmallVector<std::pair<unsigned, unsigned>> bits;
    unsigned top = 0;
    for (auto operand : op->getOperands())
      for (unsigned bit = 0, e = widthOf(operand); bit < e; ++bit) {
        unsigned in = resolve(n.ctx, operand, bit);
        if (in == kNone)
          continue;
        bits.push_back({in, bit});
        top = std::max(top, bit);
      }
    size_t first = edges.size();
    for (auto [in, bit] : bits)
      edges.push_back({in, arith(bit, top) - device.asic, true});
    if (!device.asic)
      fitLut(edges, first);
    return;
  }
  case Kind::Equal: {
    // Each LUT6 compares three bit pairs, or six bits with a constant.
    size_t first = edges.size();
    for (auto operand : op->getOperands())
      for (unsigned bit = 0, e = widthOf(operand); bit < e; ++bit)
        add(operand, bit, 0);
    unsigned bits = edges.size() - first;
    int levels = 1 + lut6Depth(llvm::divideCeil(bits, 6));
    for (auto &edge : llvm::drop_begin(edges, first))
      edge.levels = levels;
    return;
  }
  case Kind::Mul: {
    // A product wider than one DSP's multiplier is a cascade of DSPs over
    // the significant bits of the two widest operands, in the orientation
    // that takes fewer. Constant bits above the others, and copies of the
    // bit below (a sign extension), are not significant.
    size_t first = edges.size();
    unsigned a = 1, b = 1;
    for (auto operand : op->getOperands()) {
      unsigned bits = 0, below = kNone;
      for (unsigned bit = 0, e = widthOf(operand); bit < e; ++bit) {
        unsigned in = resolve(n.ctx, operand, bit);
        if (in != kNone && in != below) {
          edges.push_back({in, 0, true});
          bits = bit + 1;
        }
        below = in;
      }
      if (bits > a) {
        b = a;
        a = bits;
      } else if (bits > b) {
        b = bits;
      }
    }
    auto dsps = [](unsigned x, unsigned y, unsigned dx, unsigned dy) {
      return llvm::divideCeil(x, dx) * llvm::divideCeil(y, dy);
    };
    int levels;
    if (device.asic) {
      levels = multiplier(a, b);
    } else {
      unsigned count = std::min(dsps(a, b, device.dspA, device.dspB),
                                dsps(a, b, device.dspB, device.dspA));
      levels = device.dspIn + (count - 1) * device.dspCascade;
    }
    for (auto &edge : llvm::drop_begin(edges, first))
      edge.levels = levels;
    return;
  }
  case Kind::Div: {
    // A divider whose operands depend on few bits, as a quotient of
    // functions of a format field, is a table over them after synthesis.
    bool small;
    ArrayRef<unsigned> bitsIn = support(n.ctx, op, small);
    if (small) {
      for (unsigned in : bitsIn)
        edges.push_back({in, table(bitsIn.size()), true});
      return;
    }
    // Otherwise only the dividend's bits that are not constant count.
    // Synthesis propagates a constant divisor, leaving a table over those
    // bits. A variable divisor is a ripple of one subtracter per quotient
    // bit, each a carry chain over them.
    size_t first = edges.size();
    unsigned bits = 0;
    bool constantDivisor = true;
    for (auto [index, operand] : llvm::enumerate(op->getOperands()))
      for (unsigned bit = 0, e = widthOf(operand); bit < e; ++bit) {
        unsigned in = resolve(n.ctx, operand, bit);
        if (in == kNone)
          continue;
        edges.push_back({in, 0, true});
        if (index == 0)
          bits = std::max(bits, bit + 1);
        else
          constantDivisor = false;
      }
    int levels = constantDivisor ? table(bits)
                                 : bits * arith(0, bits ? bits - 1 : 0);
    for (auto &edge : llvm::drop_begin(edges, first))
      edge.levels = levels;
    return;
  }
  case Kind::Table: {
    size_t first = edges.size();
    for (auto operand : op->getOperands())
      for (unsigned bit = 0, e = widthOf(operand); bit < e; ++bit)
        add(operand, bit, 0);
    int levels = table(edges.size() - first);
    for (auto &edge : llvm::drop_begin(edges, first))
      edge.levels = levels;
    return;
  }
  case Kind::AsyncRead: {
    // RAM64X1D is one level; deeper memories add a MUXF7/MUXF8 or LUT level
    // per doubling.
    auto read = cast<seq::FirMemReadOp>(op);
    uint64_t depth = read.getMemory().getType().getDepth();
    int levels = device.asic
                     ? 1 + (int)llvm::Log2_64_Ceil(depth)
                     : 1 + std::max(0, (int)llvm::Log2_64_Ceil(depth) - 6);
    for (unsigned bit = 0, e = widthOf(read.getAddress()); bit < e; ++bit)
      add(read.getAddress(), bit, levels);
    if (auto enable = read.getEnable())
      add(enable, 0, 1);
    return;
  }
  case Kind::Other:
    for (auto operand : op->getOperands())
      for (unsigned bit = 0, e = widthOf(operand); bit < e; ++bit)
        add(operand, bit, 1);
    return;
  default:
    return;
  }
}

/// The bits the divider `op` of instance `ctx` depends on: the start points
/// and distributed-RAM reads its operands' cones reach. `small` is false if
/// there are more than kMaxTableBits of them.
ArrayRef<unsigned> Timing::support(unsigned ctx, Operation *op, bool &small) {
  auto key = std::make_pair(ctx, op);
  if (auto it = supports.find(key); it != supports.end()) {
    small = it->second.has_value();
    return small ? ArrayRef<unsigned>(*it->second) : ArrayRef<unsigned>();
  }
  // A divider met again inside its own cone closes a loop: not small.
  supports[key] = std::nullopt;
  SmallVector<unsigned> found, stack;
  DenseSet<unsigned> seen;
  for (auto operand : op->getOperands())
    for (unsigned bit = 0, e = widthOf(operand); bit < e; ++bit) {
      unsigned in = resolve(ctx, operand, bit);
      if (in != kNone && seen.insert(in).second)
        stack.push_back(in);
    }
  SmallVector<Edge> edges;
  small = true;
  while (small && !stack.empty()) {
    unsigned id = stack.pop_back_val();
    Kind kind = nodes[id].kind;
    if (isStart(kind) || kind == Kind::AsyncRead) {
      found.push_back(id);
      small = found.size() <= kMaxTableBits;
      continue;
    }
    edges.clear();
    inputs(id, edges);
    for (auto &edge : edges)
      if (seen.insert(edge.in).second)
        stack.push_back(edge.in);
    small = seen.size() <= kMaxSupportCone;
  }
  if (!small)
    return {};
  auto &entry = supports[key];
  entry = std::move(found);
  return *entry;
}

/// Computes the levels of `root` and of everything it depends on, depth
/// first without recursion. An input still waiting for its own inputs closes
/// a combinational loop; that edge is left out and counted.
void Timing::evaluate(unsigned root) {
  if (nodes[root].state == 2)
    return;
  SmallVector<unsigned> stack{root};
  SmallVector<Edge> edges;
  while (!stack.empty()) {
    unsigned id = stack.back();
    if (nodes[id].state == 2) {
      stack.pop_back();
      continue;
    }
    if (isStart(nodes[id].kind)) {
      nodes[id].state = 2;
      if (retimeLevels)
        order.push_back(id);
      stack.pop_back();
      continue;
    }
    edges.clear();
    inputs(id, edges);
    bool ready = true;
    if (nodes[id].state == 0) {
      nodes[id].state = 1;
      for (auto [in, levels, load] : edges) {
        if (nodes[in].state == 0) {
          stack.push_back(in);
          ready = false;
        } else if (nodes[in].state == 1) {
          if (loopSamples.size() < 8)
            loopSamples.push_back({id, in});
          ++loops;
        }
      }
      if (!ready)
        continue;
    }
    int best = 0;
    unsigned pred = kNone;
    for (auto [in, levels, load] : edges) {
      nodes[in].fanout += load;
      if (nodes[in].state != 2)
        continue;
      int total = nodes[in].levels + levels;
      if (pred == kNone || total > best) {
        best = total;
        pred = in;
      }
    }
    nodes[id].levels = best;
    nodes[id].pred = pred;
    nodes[id].state = 2;
    if (retimeLevels)
      order.push_back(id);
    stack.pop_back();
  }
}

void Timing::addEndPoints(unsigned ctx, Operation *op, StringRef port,
                          Value value, bool boundary) {
  if (!value)
    return;
  for (unsigned bit = 0, e = widthOf(value); bit < e; ++bit) {
    unsigned id = resolve(ctx, value, bit);
    if (id != kNone)
      endPoints.push_back({id, ctx, op, port, bit, kNone, boundary});
  }
}

/// Adds the end points of the input `next` of register `reg`, and records
/// each bit's input node if the register can be retimed.
void Timing::addRegister(unsigned ctx, Operation *op, Value reg, Value next,
                         bool retimable) {
  for (unsigned bit = 0, e = widthOf(next); bit < e; ++bit) {
    unsigned id = resolve(ctx, next, bit);
    unsigned r =
        retimable && retimeLevels ? node(ctx, reg, bit, Kind::Reg) : kNone;
    if (r != kNone)
      regInputs[r] = id;
    if (id == kNone)
      continue;
    endPoints.push_back({id, ctx, op, "d", bit, r, false, true});
    // A LUT that reads the register it feeds is its clock-enable
    // multiplexer or feedback, which a DSP absorbs with the register.
    if (nodes[id].kind != Kind::Lut)
      continue;
    for (auto operand : nodes[id].value.getDefiningOp()->getOperands()) {
      unsigned in = resolve(nodes[id].ctx, operand, 0);
      if (in != kNone && nodes[in].value == reg && nodes[in].bit == bit &&
          nodes[in].ctx == ctx) {
        feedbacks.insert(id);
        break;
      }
    }
  }
}

/// Whether a register's next value is always itself: only multiplexers (and
/// wires) whose every data input is, in the end, the register. Whatever its
/// selects, it holds its value forever, and Vivado removes it; its input is
/// no end point.
static bool holdsItself(Value next, Value reg) {
  SmallVector<Value> work{next};
  DenseSet<Value> seen;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (value == reg || !seen.insert(value).second)
      continue;
    auto *op = value.getDefiningOp();
    if (auto mux = dyn_cast_or_null<comb::MuxOp>(op)) {
      work.push_back(mux.getTrueValue());
      work.push_back(mux.getFalseValue());
    } else if (auto wire = dyn_cast_or_null<hw::WireOp>(op)) {
      work.push_back(wire.getInput());
    } else {
      return false;
    }
  }
  return true;
}

/// Collects the end points of instance `ctx` and of everything below it.
void Timing::collect(unsigned ctx) {
  auto module = contexts[ctx].module;
  for (auto &op : *module.getBodyBlock()) {
    // Vivado retimes registers with a synchronous reset, not an
    // asynchronous one.
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      if (holdsItself(reg.getNext(), reg))
        continue;
      addRegister(ctx, reg, reg, reg.getNext(), !reg.getIsAsync());
      if (auto reset = reg.getReset())
        addEndPoints(ctx, reg, "rst", reset);
    } else if (auto reg = dyn_cast<seq::CompRegOp>(op)) {
      if (holdsItself(reg.getInput(), reg))
        continue;
      addRegister(ctx, reg, reg, reg.getInput(), true);
      if (auto reset = reg.getReset())
        addEndPoints(ctx, reg, "rst", reset);
    } else if (auto reg = dyn_cast<seq::CompRegClockEnabledOp>(op)) {
      if (holdsItself(reg.getInput(), reg))
        continue;
      addRegister(ctx, reg, reg, reg.getInput(), true);
      addEndPoints(ctx, reg, "ce", reg.getClockEnable());
      if (auto reset = reg.getReset())
        addEndPoints(ctx, reg, "rst", reset);
    } else if (auto write = dyn_cast<seq::FirMemWriteOp>(op)) {
      addEndPoints(ctx, write, "wa", write.getAddress());
      addEndPoints(ctx, write, "wd", write.getData());
      addEndPoints(ctx, write, "we", write.getEnable());
      addEndPoints(ctx, write, "wm", write.getMask());
    } else if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
      if (read.getMemory().getDefiningOp<seq::FirMemOp>().getReadLatency()) {
        addEndPoints(ctx, read, "ra", read.getAddress());
        addEndPoints(ctx, read, "re", read.getEnable());
      }
    } else if (auto rw = dyn_cast<seq::FirMemReadWriteOp>(op)) {
      addEndPoints(ctx, rw, "a", rw.getAddress());
      addEndPoints(ctx, rw, "en", rw.getEnable());
      addEndPoints(ctx, rw, "mode", rw.getMode());
      addEndPoints(ctx, rw, "wd", rw.getWriteData());
      addEndPoints(ctx, rw, "wm", rw.getMask());
    } else if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
      auto *callee = modules.lookup(inst.getModuleNameAttr().getAttr());
      if (auto module = dyn_cast_or_null<hw::HWModuleOp>(callee)) {
        collect(child(ctx, inst, module));
        continue;
      }
      bool stub = stubs.contains(inst.getModuleNameAttr().getAttr());
      for (auto [operand, name] :
           llvm::zip(inst.getOperands(), inst.getArgNames()))
        if (!isa<seq::ClockType>(operand.getType()))
          addEndPoints(ctx, inst, cast<StringAttr>(name).getValue(), operand,
                       stub);
    } else if (ctx == 0 && isa<hw::OutputOp>(op)) {
      for (auto [operand, name] :
           llvm::zip(op.getOperands(), module.getOutputNames()))
        addEndPoints(ctx, &op, cast<StringAttr>(name).getValue(), operand,
                     true);
    }
  }
}

LogicalResult Timing::run(hw::HWModuleOp top, unsigned retimeLevels) {
  this->retimeLevels = retimeLevels;
  contexts.push_back({0, {}, top, 0});
  collect(0);
  for (auto &end : endPoints)
    evaluate(end.node);
  for (auto &end : endPoints) {
    auto &n = nodes[end.node];
    end.levels = n.levels;
    if (n.kind == Kind::Mul && !end.registered)
      end.levels += device.dspOut;
  }
  if (retimeLevels)
    retime(retimeLevels);
  else
    for (auto &end : endPoints)
      end.retimed = end.levels;
  return success();
}

/// The depth of each end point's worst path after one hop of balancing with
/// its neighbour stages, as Vivado's global retiming does with a period of
/// `period` levels: the end register moves backward into the room its
/// output stage leaves, and the start register forward into the room its
/// input stage leaves. Neither crosses a distributed RAM or a DSP, and only
/// a register Vivado can retime moves. Top ports and the pins of stubs are
/// registers with nothing beyond them, as a gate harness or the fold's links
/// make them.
void Timing::retime(unsigned period) {
  int c = period;
  // The deepest path from each node to an end point, consumers first.
  std::vector<int16_t> depart(nodes.size(), -1);
  for (auto &end : endPoints)
    depart[end.node] = std::max<int16_t>(depart[end.node], 0);
  SmallVector<Edge> edges;
  for (unsigned id : llvm::reverse(order)) {
    if (depart[id] < 0 || isStart(nodes[id].kind))
      continue;
    edges.clear();
    inputs(id, edges);
    for (auto [in, levels, load] : edges)
      depart[in] = std::max<int>(depart[in], depart[id] + levels);
  }

  for (auto &end : endPoints) {
    int a = end.levels;
    end.retimed = a;
    if (a <= c)
      continue;
    // The levels after the last pinned node of the path and before the
    // first.
    int trail = a, lead = a;
    unsigned start = end.node;
    bool pinned = false;
    for (unsigned id = end.node; id != kNone; id = nodes[id].pred) {
      start = id;
      if (!isPinned(nodes[id].kind))
        continue;
      if (!pinned)
        trail = a - nodes[id].levels;
      pinned = true;
      unsigned pred = nodes[id].pred;
      lead = pred == kNone ? 0 : nodes[pred].levels;
    }
    int borrowEnd = 0, borrowStart = 0;
    if (end.reg != kNone)
      borrowEnd = std::min(trail, std::max(0, c - std::max<int>(
                                                  depart[end.reg], 0)));
    else if (end.boundary)
      borrowEnd = std::min(trail, c);
    if (auto it = regInputs.find(start); it != regInputs.end()) {
      int in = it->second == kNone ? 0 : nodes[it->second].levels;
      borrowStart = std::min(lead, std::max(0, c - in));
    } else if (nodes[start].kind == Kind::Port ||
               nodes[start].kind == Kind::Stub) {
      borrowStart = std::min(lead, c);
    }
    end.retimed = std::max(c, a - borrowEnd - borrowStart);
  }
}

const std::string &Timing::path(unsigned ctx) {
  if (paths.size() < contexts.size())
    paths.resize(contexts.size());
  auto &p = paths[ctx];
  if (p.empty() && ctx != 0) {
    auto &c = contexts[ctx];
    p = path(c.parent);
    if (!p.empty())
      p += '/';
    p += c.inst.getInstanceName();
  }
  return paths[ctx];
}

/// Walks the worst path back from `id`, the bit an end point in instance
/// `ctx` takes, to its start point, with the largest fanout of a bit on it
/// and the instance boundaries it crosses.
unsigned Timing::startOf(unsigned ctx, unsigned id, uint32_t &fanout,
                         unsigned &crossings) {
  fanout = 0;
  crossings = 0;
  while (true) {
    auto &n = nodes[id];
    fanout = std::max(fanout, n.fanout);
    // Up from one and down to the other, through their common parent.
    for (unsigned a = ctx, b = n.ctx; a != b; ++crossings) {
      if (contexts[a].depth >= contexts[b].depth)
        a = contexts[a].parent;
      else
        b = contexts[b].parent;
    }
    if (n.pred == kNone)
      return id;
    ctx = n.ctx;
    id = n.pred;
  }
}

void Timing::printOp(raw_ostream &os, Operation *op, StringRef port,
                     uint32_t bit) {
  if (auto name = op->getAttrOfType<StringAttr>("name"))
    os << name.getValue();
  else if (auto inst = dyn_cast<hw::InstanceOp>(op))
    os << inst.getInstanceName();
  else if (auto read = dyn_cast<seq::FirMemReadOp>(op))
    os << read.getMemory().getDefiningOp<seq::FirMemOp>().getName();
  else if (auto write = dyn_cast<seq::FirMemWriteOp>(op))
    os << write.getMemory().getDefiningOp<seq::FirMemOp>().getName();
  else if (auto rw = dyn_cast<seq::FirMemReadWriteOp>(op))
    os << rw.getMemory().getDefiningOp<seq::FirMemOp>().getName();
  else if (auto hint = op->getAttrOfType<StringAttr>("sv.namehint"))
    os << hint.getValue();
  else
    os << op->getName().getStringRef();
  if (!port.empty())
    os << '.' << port;
  os << '[' << bit << ']';
}

void Timing::printName(raw_ostream &os, unsigned id) {
  auto &n = nodes[id];
  if (auto arg = dyn_cast<BlockArgument>(n.value)) {
    os << contexts[n.ctx].module.getInputName(arg.getArgNumber()) << '['
       << n.bit << ']';
    return;
  }
  Operation *op = n.value.getDefiningOp();
  StringRef port;
  if (auto inst = dyn_cast<hw::InstanceOp>(op))
    port = inst.getOutputName(cast<OpResult>(n.value).getResultNumber());
  printOp(os, op, port, n.bit);
}

void Timing::writeEndPoints(raw_ostream &os, unsigned minLevels) {
  os << "levels,retimed,kind,start_kind,fanout,crossings,start_inst,start,"
        "end_inst,end\n";
  for (auto &end : endPoints) {
    if ((unsigned)end.levels < minLevels)
      continue;
    uint32_t fanout;
    unsigned crossings;
    unsigned start = startOf(end.ctx, end.node, fanout, crossings);
    os << end.levels << ',' << end.retimed << ',' << end.port << ','
       << kindName(nodes[start].kind)
       << ',' << fanout << ',' << crossings << ',' << nodes[start].ctx << ',';
    printName(os, start);
    os << ',' << end.ctx << ',';
    printOp(os, end.op, end.port, end.bit);
    os << '\n';
  }
}

void Timing::writeInstances(raw_ostream &os) {
  os << "id,parent,instance,module\n";
  for (unsigned i = 0, e = contexts.size(); i < e; ++i) {
    auto &c = contexts[i];
    os << i << ',' << c.parent << ','
       << (i ? c.inst.getInstanceName() : StringRef("")) << ','
       << c.module.getModuleName() << '\n';
  }
}

/// For every instance of `modules`, the deepest bit of each port: the
/// levels at which an input arrives, and the levels behind an output.
void Timing::writePorts(raw_ostream &os, ArrayRef<std::string> modules) {
  os << "instance,module,direction,port,width,levels\n";
  auto deepest = [&](unsigned ctx, Value value) {
    int levels = -1;
    for (unsigned bit = 0, e = widthOf(value); bit < e; ++bit) {
      unsigned id = resolve(ctx, value, bit);
      if (id == kNone)
        continue;
      evaluate(id);
      levels = std::max<int>(levels, nodes[id].levels);
    }
    return levels;
  };
  for (unsigned ctx = 1, e = contexts.size(); ctx < e; ++ctx) {
    auto c = contexts[ctx];
    if (!matchesModule(c.module.getModuleName(), modules))
      continue;
    auto write = [&](StringRef dir, StringRef name, Value value, int levels) {
      os << path(ctx) << ',' << c.module.getModuleName() << ',' << dir << ','
         << name << ',' << widthOf(value) << ',' << levels << '\n';
    };
    for (auto [i, operand] : llvm::enumerate(c.inst.getOperands()))
      if (!isa<seq::ClockType>(operand.getType()))
        write("in", c.module.getInputName(i), operand,
              deepest(c.parent, operand));
    auto *output = c.module.getBodyBlock()->getTerminator();
    for (auto [i, operand] : llvm::enumerate(output->getOperands()))
      write("out", c.module.getOutputName(i), operand, deepest(ctx, operand));
  }
}

/// One JSON line per pair of start and end registers whose deepest bit has
/// at least `minLevels` levels, deepest first: that bit's path op by op, from
/// the end point back to its start, and the pair's end point bits. The bits
/// of a register from one start register share their path's shape, so the
/// pair is what there is to read.
void Timing::writePaths(raw_ostream &os, unsigned minLevels) {
  SmallVector<unsigned> deep;
  for (unsigned i = 0, e = endPoints.size(); i < e; ++i)
    if ((unsigned)endPoints[i].levels >= minLevels)
      deep.push_back(i);
  std::stable_sort(deep.begin(), deep.end(), [&](unsigned a, unsigned b) {
    return endPoints[a].levels > endPoints[b].levels;
  });

  struct Pair {
    unsigned end, start, bits;
  };
  SmallVector<Pair> pairs;
  DenseMap<std::tuple<unsigned, Operation *, StringRef, unsigned, Value>,
           unsigned>
      index;
  for (unsigned i : deep) {
    auto &end = endPoints[i];
    uint32_t fanout;
    unsigned crossings;
    unsigned start = startOf(end.ctx, end.node, fanout, crossings);
    auto [it, inserted] = index.try_emplace(
        {end.ctx, end.op, end.port, nodes[start].ctx, nodes[start].value},
        pairs.size());
    if (inserted)
      pairs.push_back({i, start, 0});
    ++pairs[it->second].bits;
  }

  auto str = [](auto print) {
    std::string s;
    llvm::raw_string_ostream os(s);
    print(os);
    return s;
  };
  // Mapped logic has no name; the first source line it came from names it.
  auto source = [&](unsigned id) {
    std::string s;
    nodes[id].value.getLoc()->walk([&](Location loc) {
      auto file = dyn_cast<FileLineColLoc>(loc);
      if (!file)
        return WalkResult::advance();
      s = (llvm::sys::path::filename(file.getFilename().getValue()) + ":" +
           Twine(file.getLine()))
              .str();
      return WalkResult::interrupt();
    });
    return s;
  };
  for (auto [i, start, bits] : pairs) {
    auto &end = endPoints[i];
    llvm::json::OStream j(os);
    j.object([&] {
      j.attribute("levels", end.levels);
      j.attribute("retimed", end.retimed);
      j.attribute("bits", bits);
      j.attribute("kind", end.port);
      j.attribute("start_kind", kindName(nodes[start].kind));
      j.attribute("start_inst", nodes[start].ctx);
      j.attribute("start",
                  str([&](raw_ostream &s) { printName(s, start); }));
      j.attribute("end_inst", end.ctx);
      j.attribute("end", str([&](raw_ostream &s) {
                    printOp(s, end.op, end.port, end.bit);
                  }));
      j.attributeArray("ops", [&] {
        for (unsigned id = end.node; id != kNone; id = nodes[id].pred) {
          auto &n = nodes[id];
          j.array([&] {
            j.value(n.levels);
            j.value(kindName(n.kind));
            j.value(n.fanout);
            j.value(n.ctx);
            j.value(str([&](raw_ostream &s) { printName(s, id); }));
            j.value(source(id));
          });
        }
      });
    });
    os << '\n';
  }
}

void Timing::printPath(raw_ostream &os, const EndPoint &end, unsigned rank) {
  os << "#" << rank << " levels " << end.levels << ": "
     << path(end.ctx) << '/';
  printOp(os, end.op, end.port, end.bit);
  os << " (" << contexts[end.ctx].module.getModuleName() << ")\n";
  for (unsigned id = end.node; id != kNone; id = nodes[id].pred) {
    auto &n = nodes[id];
    os << llvm::format("  %4d %-6s fo %-5u ", n.levels, kindName(n.kind).data(),
                       n.fanout)
       << path(n.ctx) << '/';
    printName(os, id);
    os << '\n';
  }
}

void Timing::writeReport(raw_ostream &os, unsigned topPaths,
                         ArrayRef<std::string> trace) {
  uint64_t computed = 0;
  std::map<int, uint64_t> histogram;
  for (auto &n : nodes)
    computed += n.state == 2;
  for (auto &end : endPoints)
    ++histogram[end.levels];
  os << "instances " << contexts.size() << ", bits " << nodes.size()
     << " (evaluated " << computed << "), end points " << endPoints.size()
     << ", combinational loops " << loops << "\n";
  for (auto [id, in] : loopSamples) {
    os << "  loop: " << path(nodes[id].ctx) << '/';
    printName(os, id);
    os << " <- " << path(nodes[in].ctx) << '/';
    printName(os, in);
    os << '\n';
  }
  if (!otherOps.empty()) {
    os << "ops without a delay model (one level each):";
    for (auto &entry : otherOps)
      os << ' ' << entry.getKey() << '=' << entry.getValue();
    os << '\n';
  }
  os << "end points by levels:";
  for (auto [levels, count] : histogram)
    os << ' ' << levels << ':' << count;
  os << '\n';
  if (device.asic && !histogram.empty()) {
    int stages = histogram.rbegin()->first;
    int period = device.clk2qPs + device.pathPs + stages * device.stagePs +
                 device.setupPs;
    os << "deepest path " << stages << " stages: " << device.clk2qPs << " + "
       << device.pathPs << " + " << stages << " x " << device.stagePs << " + "
       << device.setupPs << " = " << period << " ps ("
       << llvm::format("%.0f", 1e6 / period) << " MHz)\n";
  }
  if (retimeLevels) {
    std::map<int, uint64_t> retimed;
    for (auto &end : endPoints)
      ++retimed[end.retimed];
    os << "end points by levels after retiming to " << retimeLevels << ":";
    for (auto [levels, count] : retimed)
      os << ' ' << levels << ':' << count;
    os << '\n';
  }
  os << '\n';

  SmallVector<unsigned> order(endPoints.size());
  for (unsigned i = 0, e = order.size(); i < e; ++i)
    order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
    return endPoints[a].levels > endPoints[b].levels;
  });
  for (unsigned i = 0, e = std::min<size_t>(topPaths, order.size()); i < e;
       ++i)
    printPath(os, endPoints[order[i]], i + 1);

  // The deepest end point bit matching each trace string.
  for (auto &pattern : trace) {
    os << "\ntrace '" << pattern << "':\n";
    unsigned found = 0;
    for (unsigned i = 0, e = order.size(); i < e; ++i) {
      auto &end = endPoints[order[i]];
      std::string name;
      llvm::raw_string_ostream s(name);
      s << path(end.ctx) << '/';
      printOp(s, end.op, end.port, end.bit);
      if (!StringRef(name).contains(pattern))
        continue;
      printPath(os, end, i + 1);
      ++found;
      break;
    }
    if (!found)
      os << "  no end point\n";
  }
}

namespace {
struct FoldZeroWidthPass
    : public foldsim::impl::FoldZeroWidthBase<FoldZeroWidthPass> {
  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      if (!isa_and_nonnull<comb::CombDialect>(op->getDialect()) ||
          op->getNumResults() == 0 || !isMemoryEffectFree(op))
        return;
      for (auto type : op->getResultTypes())
        if (!type.isInteger(0))
          return;
      OpBuilder b(op);
      for (auto result : op->getResults())
        result.replaceAllUsesWith(
            hw::ConstantOp::create(b, op->getLoc(), APInt::getZero(0)));
      op->erase();
    });
  }
};

struct SyncAsyncResetsPass
    : public foldsim::impl::SyncAsyncResetsBase<SyncAsyncResetsPass> {
  void runOnOperation() override {
    SmallVector<seq::FirRegOp> regs;
    getOperation().walk([&](seq::FirRegOp reg) {
      if (reg.getIsAsync())
        regs.push_back(reg);
    });
    for (auto reg : regs) {
      OpBuilder b(reg);
      auto sync = seq::FirRegOp::create(
          b, reg.getLoc(), reg.getNext(), reg.getClk(), reg.getNameAttr(),
          reg.getReset(), reg.getResetValue(), reg.getInnerSymAttr(),
          /*isAsync=*/false, reg.getPresetAttr());
      b.setInsertionPointAfter(sync);
      auto value = comb::MuxOp::create(b, reg.getLoc(), reg.getReset(),
                                       reg.getResetValue(), sync);
      reg.getResult().replaceAllUsesWith(value.getResult());
      reg.erase();
    }
  }
};

struct SplitMaskedMemoriesPass
    : public foldsim::impl::SplitMaskedMemoriesBase<SplitMaskedMemoriesPass> {
  void runOnOperation() override {
    constexpr unsigned kColumn = 64, kByte = 8;
    SmallVector<seq::FirMemOp> memories;
    getOperation().walk([&](seq::FirMemOp memory) {
      auto type = memory.getType();
      auto maskWidth = type.getMaskWidth();
      unsigned width = type.getWidth();
      if (!maskWidth || *maskWidth * kByte != width || width <= kColumn ||
          width % kColumn || memory.getInnerSymAttr() || memory.getInit())
        return;
      bool split = llvm::all_of(memory->getUsers(), [](Operation *user) {
        if (auto write = dyn_cast<seq::FirMemWriteOp>(user))
          return bool(write.getMask());
        return isa<seq::FirMemReadOp>(user);
      });
      if (split)
        memories.push_back(memory);
    });
    for (auto memory : memories) {
      auto type = memory.getType();
      unsigned columns = type.getWidth() / kColumn;
      auto columnType = seq::FirMemType::get(&getContext(), type.getDepth(),
                                             kColumn, kColumn / kByte);
      OpBuilder b(memory);
      SmallVector<Value> parts;
      for (unsigned c = 0; c < columns; ++c) {
        StringAttr name;
        if (auto n = memory.getNameAttr())
          name = b.getStringAttr(n.getValue() + "_c" + Twine(c));
        parts.push_back(seq::FirMemOp::create(
            b, memory.getLoc(), columnType, memory.getReadLatency(),
            memory.getWriteLatency(), memory.getRuw(), memory.getWuw(), name,
            {}, {}, memory.getPrefixAttr(), memory.getOutputFileAttr()));
      }
      for (auto *user : llvm::make_early_inc_range(memory->getUsers())) {
        OpBuilder ub(user);
        if (auto read = dyn_cast<seq::FirMemReadOp>(user)) {
          SmallVector<Value> data;
          for (unsigned c = columns; c-- > 0;)
            data.push_back(seq::FirMemReadOp::create(
                ub, read.getLoc(), parts[c], read.getAddress(), read.getClk(),
                read.getEnable()));
          read.replaceAllUsesWith(
              comb::ConcatOp::create(ub, read.getLoc(), data).getResult());
        } else {
          auto write = cast<seq::FirMemWriteOp>(user);
          for (unsigned c = 0; c < columns; ++c) {
            Value data = comb::ExtractOp::create(ub, write.getLoc(),
                                                 write.getData(), c * kColumn,
                                                 kColumn);
            Value mask = comb::ExtractOp::create(
                ub, write.getLoc(), write.getMask(), c * (kColumn / kByte),
                kColumn / kByte);
            seq::FirMemWriteOp::create(ub, write.getLoc(), parts[c],
                                       write.getAddress(), write.getClk(),
                                       write.getEnable(), data, mask);
          }
        }
        user->erase();
      }
      memory.erase();
    }
  }
};

struct KeepModulesPass
    : public foldsim::impl::KeepModulesBase<KeepModulesPass> {
  using KeepModulesBase::KeepModulesBase;
  void runOnOperation() override {
    for (auto module : getOperation().getOps<hw::HWModuleOp>())
      if (matchesModule(module.getModuleName(), modules))
        SymbolTable::setSymbolVisibility(module,
                                         SymbolTable::Visibility::Public);
  }
};

struct LowerConstantDivisionPass
    : public foldsim::impl::LowerConstantDivisionBase<
          LowerConstantDivisionPass> {
  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      if (!isa<comb::DivUOp, comb::ModUOp>(op))
        return;
      auto divisor = op->getOperand(1).getDefiningOp<hw::ConstantOp>();
      if (!divisor || divisor.getValue().isZero())
        return;
      bool quotient = isa<comb::DivUOp>(op);
      unsigned width = widthOf(op->getResult(0));
      const APInt &d = divisor.getValue();
      Value x = op->getOperand(0);
      OpBuilder b(op);
      auto loc = op->getLoc();
      auto zeros = [&](unsigned n) {
        return hw::ConstantOp::create(b, loc, APInt::getZero(n));
      };
      Value result;
      if (d.isPowerOf2()) {
        // The kept bits: above k for a quotient, below k for a remainder.
        unsigned k = d.logBase2();
        unsigned low = quotient ? k : 0;
        unsigned kept = quotient ? width - k : k;
        if (kept == 0) {
          result = zeros(width);
        } else {
          result = b.createOrFold<comb::ExtractOp>(loc, x, low, kept);
          if (kept < width)
            result = b.createOrFold<comb::ConcatOp>(loc, zeros(width - kept),
                                                    result);
        }
      } else {
        unsigned bits = significantBits(x);
        if (bits > kMaxTableBits)
          return;
        // Entry i is the result for dividend i; each level of muxes halves
        // the table on one dividend bit, the lowest first.
        SmallVector<Value> table;
        for (uint64_t i = 0, e = uint64_t(1) << bits; i < e; ++i) {
          APInt n(width, i);
          table.push_back(hw::ConstantOp::create(
              b, loc, quotient ? n.udiv(d) : n.urem(d)));
        }
        for (unsigned bit = 0; bit < bits; ++bit) {
          Value sel = b.createOrFold<comb::ExtractOp>(loc, x, bit, 1);
          SmallVector<Value> half;
          for (unsigned i = 0, e = table.size(); i < e; i += 2)
            half.push_back(b.createOrFold<comb::MuxOp>(loc, sel, table[i + 1],
                                                       table[i], true));
          table = std::move(half);
        }
        result = table.front();
      }
      op->getResult(0).replaceAllUsesWith(result);
      op->erase();
    });
  }
};

struct LowerEqualityPass
    : public foldsim::impl::LowerEqualityBase<LowerEqualityPass> {
  void runOnOperation() override {
    getOperation().walk([](comb::ICmpOp cmp) {
      bool equal;
      switch (cmp.getPredicate()) {
      case comb::ICmpPredicate::eq:
      case comb::ICmpPredicate::ceq:
      case comb::ICmpPredicate::weq:
        equal = true;
        break;
      case comb::ICmpPredicate::ne:
      case comb::ICmpPredicate::cne:
      case comb::ICmpPredicate::wne:
        equal = false;
        break;
      default:
        return;
      }
      OpBuilder b(cmp);
      auto loc = cmp.getLoc();
      Value diff = b.createOrFold<comb::XorOp>(loc, cmp.getLhs(), cmp.getRhs(),
                                               true);
      unsigned width = widthOf(diff);
      if (width == 0) {
        cmp.replaceAllUsesWith(
            hw::ConstantOp::create(b, loc, APInt(1, equal)).getResult());
        cmp.erase();
        return;
      }
      SmallVector<Value> bits;
      for (unsigned i = 0; i < width; ++i)
        bits.push_back(b.createOrFold<comb::ExtractOp>(loc, diff, i, 1));
      Value any = width == 1
                      ? bits[0]
                      : b.createOrFold<comb::OrOp>(loc, bits, true);
      if (equal)
        any = b.createOrFold<comb::XorOp>(
            loc, any, hw::ConstantOp::create(b, loc, APInt(1, 1)), true);
      cmp.replaceAllUsesWith(any);
      cmp.erase();
    });
  }
};

struct EstimateTimingPass
    : public foldsim::impl::EstimateTimingBase<EstimateTimingPass> {
  using EstimateTimingBase::EstimateTimingBase;
  void runOnOperation() override;
};
} // namespace

void EstimateTimingPass::runOnOperation() {
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
    if (!top) {
      design.emitError("no top module");
      return signalPassFailure();
    }
  }

  const Device *device = lookupDevice(family);
  if (!device) {
    design.emitError("unknown FPGA family '") << family << "'";
    return signalPassFailure();
  }
  Timing timing(design, blackBoxes, *device);
  if (failed(timing.run(top, retimeLevels)))
    return signalPassFailure();

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
  if (failed(write(output, [&](raw_ostream &os) {
        timing.writeEndPoints(os, minLevels);
      })) ||
      failed(write(instances,
                   [&](raw_ostream &os) { timing.writeInstances(os); })) ||
      failed(write(ports, [&](raw_ostream &os) {
        timing.writePorts(os, portModules);
      })) ||
      failed(write(paths, [&](raw_ostream &os) {
        timing.writePaths(os, pathLevels);
      })) ||
      failed(write(report, [&](raw_ostream &os) {
        timing.writeReport(os, topPaths, trace);
      })))
    return signalPassFailure();
  markAllAnalysesPreserved();
}
