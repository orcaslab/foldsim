//===- IndexClassInputs.cpp - array inputs read at an index ---------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A model's array input that the model reads only element by element, at an
// index it computes (an arbiter's grant into a crossbar's data: `hw.array_get`
// of the input or of constant slices of it), becomes, per read, an index
// output of the model and an element input that the netlist drives with
// `hw.array_get` of the array at that index. The netlist computes the same
// values (exact); a folded simulator then reads one element per read, where
// it would otherwise hold the whole array for each instance. See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace foldsim {
#define GEN_PASS_DEF_INDEXCLASSINPUTS
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// One read of an array input: the `hw.array_get` and the slices it reads
/// through (outermost first).
struct Read {
  hw::ArrayGetOp get;
  SmallVector<hw::ArraySliceOp> slices;
};

struct IndexClassInputsPass
    : public foldsim::impl::IndexClassInputsBase<IndexClassInputsPass> {
  using IndexClassInputsBase::IndexClassInputsBase;
  void runOnOperation() override;
};

/// Whether `value` is wiring of the netlist's inputs and constants.
static bool wiringOfInputs(Value value) {
  SmallVector<Value> stack{value};
  DenseSet<Value> seen;
  while (!stack.empty()) {
    Value v = stack.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (isa<BlockArgument>(v))
      continue;
    auto *op = v.getDefiningOp();
    if (isa<hw::ConstantOp, hw::AggregateConstantOp>(op))
      continue;
    if (!isa<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp, hw::BitcastOp>(op))
      return false;
    for (auto operand : op->getOperands())
      stack.push_back(operand);
  }
  return true;
}

/// The reads of `arg`, if every use is one (through constant in-range slices
/// to a get of a power-of-two array).
static std::optional<SmallVector<Read>> readsOf(Value arg) {
  SmallVector<Read> reads;
  SmallVector<std::pair<Value, SmallVector<hw::ArraySliceOp>>> stack{{arg, {}}};
  while (!stack.empty()) {
    auto [value, slices] = stack.pop_back_val();
    for (auto *user : value.getUsers()) {
      if (auto get = dyn_cast<hw::ArrayGetOp>(user)) {
        if (get.getInput() != value)
          return std::nullopt;
        auto size = cast<hw::ArrayType>(value.getType()).getNumElements();
        if (!llvm::isPowerOf2_64(size))
          return std::nullopt;
        reads.push_back({get, slices});
        continue;
      }
      if (auto slice = dyn_cast<hw::ArraySliceOp>(user)) {
        APInt low;
        if (slice.getInput() != value ||
            !matchPattern(slice.getLowIndex(), m_ConstantInt(&low)))
          return std::nullopt;
        auto size = cast<hw::ArrayType>(value.getType()).getNumElements();
        auto len = cast<hw::ArrayType>(slice.getType()).getNumElements();
        if (low.getZExtValue() + len > size)
          return std::nullopt;
        auto chain = slices;
        chain.push_back(slice);
        stack.push_back({slice.getResult(), chain});
        continue;
      }
      return std::nullopt;
    }
  }
  return reads;
}

} // namespace

void IndexClassInputsPass::runOnOperation() {
  ModuleOp top = getOperation();
  auto netlist = dyn_cast_or_null<hw::HWModuleOp>(
      SymbolTable::lookupSymbolIn(top, StringAttr::get(&getContext(), module)));
  if (!netlist) {
    top.emitError("foldsim-index-class-inputs: no module ") << module;
    return signalPassFailure();
  }
  // The models the netlist instantiates, each with its instances; a model
  // instantiated anywhere else is left alone.
  MapVector<hw::HWModuleOp, SmallVector<hw::InstanceOp>> instancesOf;
  for (auto instance : netlist.getBodyBlock()->getOps<hw::InstanceOp>())
    if (auto model = dyn_cast_or_null<hw::HWModuleOp>(
            SymbolTable::lookupSymbolIn(top, instance.getModuleNameAttr().getAttr())))
      instancesOf[model].push_back(instance);
  DenseSet<Operation *> elsewhere;
  top.walk([&](hw::InstanceOp instance) {
    if (instance->getParentOp() != netlist)
      elsewhere.insert(SymbolTable::lookupSymbolIn(
          top, instance.getModuleNameAttr().getAttr()));
  });

  unsigned inputs = 0, reads = 0;
  uint64_t bitsBefore = 0, bitsAfter = 0;
  for (auto &[model, instances] : instancesOf) {
    if (elsewhere.contains(model) || instances.size() < minInstances)
      continue;
    auto *body = model.getBodyBlock();
    // Its array inputs, last first (erasing one keeps the earlier indices).
    for (unsigned i = model.getNumInputPorts(); i-- > 0;) {
      auto arg = body->getArgument(i);
      auto type = dyn_cast<hw::ArrayType>(arg.getType());
      if (!type || type.getNumElements() < minElements ||
          !llvm::isPowerOf2_64(type.getNumElements()) ||
          !isa<IntegerType>(type.getElementType()) || arg.use_empty())
        continue;
      auto found = readsOf(arg);
      if (!found || found->size() >= type.getNumElements())
        continue;
      if (inputsOnly && !llvm::all_of(instances, [&](hw::InstanceOp instance) {
            return wiringOfInputs(instance.getOperand(i));
          }))
        continue;
      unsigned n = type.getNumElements(), w = llvm::Log2_64(n);
      Type elem = type.getElementType();
      std::string name = model.getInputName(i).str();
      // Per read: the full index, an output; the element, an input.
      SmallVector<std::pair<StringAttr, Value>> outs;
      unsigned firstOut = model.getNumOutputPorts();
      unsigned firstIn = model.getNumInputPorts();
      for (auto [r, read] : llvm::enumerate(*found)) {
        OpBuilder b(read.get);
        auto loc = read.get.getLoc();
        auto widen = [&](Value v) -> Value {
          unsigned vw = v.getType().getIntOrFloatBitWidth();
          if (vw == w)
            return v;
          return comb::ConcatOp::create(
              b, loc, hw::ConstantOp::create(b, loc, b.getIntegerType(w - vw), 0), v);
        };
        Value index = widen(read.get.getIndex());
        for (auto slice : read.slices)
          index = comb::AddOp::create(b, loc, index, widen(slice.getLowIndex()));
        outs.push_back({b.getStringAttr(name + "_index" + Twine(r)), index});
        auto value = model.appendInput(name + "_at" + Twine(r), elem).second;
        read.get.getResult().replaceAllUsesWith(value);
      }
      model.appendOutputs(outs);
      for (auto &read : *found) {
        read.get.erase();
        for (auto slice : llvm::reverse(read.slices))
          if (slice->use_empty())
            slice.erase();
      }
      // (Slices left unread: dead, users before their inputs.)
      llvm::SetVector<Operation *> dead;
      SmallVector<Operation *> walk(arg.getUsers().begin(), arg.getUsers().end());
      while (!walk.empty()) {
        auto *op = walk.pop_back_val();
        if (!dead.insert(op))
          continue;
        for (auto *user : op->getUsers())
          walk.push_back(user);
      }
      bool erased = true;
      while (erased) {
        erased = false;
        for (auto *op : dead)
          if (op && op->use_empty()) {
            op->erase();
            dead.remove(op);
            erased = true;
            break;
          }
      }
      model.modifyPorts({}, {}, {i}, {});
      body->eraseArgument(i);
      firstIn -= 1;
      // Every instance: the elements read by the netlist at its indices.
      for (auto &old : instances) {
        OpBuilder b(old);
        auto loc = old.getLoc();
        Value array = old.getOperand(i);
        SmallVector<Value> operands;
        for (auto [j, operand] : llvm::enumerate(old.getOperands()))
          if (j != i)
            operands.push_back(operand);
        SmallVector<Operation *> holders;
        for (unsigned r = 0; r < found->size(); ++r) {
          auto holder = UnrealizedConversionCastOp::create(b, loc, TypeRange{elem},
                                                           ValueRange{});
          holders.push_back(holder);
          operands.push_back(holder.getResult(0));
        }
        auto fresh = hw::InstanceOp::create(b, loc, model, old.getInstanceNameAttr(),
                                            operands, old.getParametersAttr(),
                                            old.getInnerSymAttr());
        fresh->setDiscardableAttrs(old->getDiscardableAttrDictionary());
        for (unsigned j = 0; j < old.getNumResults(); ++j)
          old.getResult(j).replaceAllUsesWith(fresh.getResult(j));
        for (unsigned r = 0; r < found->size(); ++r) {
          Value got = hw::ArrayGetOp::create(b, loc, array,
                                             fresh.getResult(firstOut + r));
          fresh.setOperand(firstIn + r, got);
          holders[r]->erase();
        }
        old.erase();
        old = fresh;
      }
      ++inputs;
      reads += found->size();
      bitsBefore += uint64_t(n) * elem.getIntOrFloatBitWidth() * instances.size();
      bitsAfter += uint64_t(found->size()) * (elem.getIntOrFloatBitWidth() + w) *
                   instances.size();
      if (verbose)
        llvm::errs() << "foldsim-index-class-inputs: " << model.getModuleName()
                     << "." << name << ": " << n << " x "
                     << elem.getIntOrFloatBitWidth() << " b read at "
                     << found->size() << " index(es), " << instances.size()
                     << " instance(s)\n";
    }
  }
  if (verbose)
    llvm::errs() << "foldsim-index-class-inputs: " << inputs << " input(s), "
                 << reads << " read(s); instance input bits " << bitsBefore
                 << " -> " << bitsAfter << "\n";
}
