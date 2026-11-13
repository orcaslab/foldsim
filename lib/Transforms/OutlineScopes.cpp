//===- OutlineScopes.cpp - generate scopes of a flat module as instances --===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A generate loop of an RTL module (`for (genvar i ...) begin : g_in`) leaves,
// once the module is flat, its registers named `g_in_<i>.<...>` and no module
// of its own: the loop's iterations cannot be a class. Here each iteration's
// registers, with the logic that feeds them alone, become an instance of one
// generated module, when the iterations are the same logic (constants that
// differ are inputs of the instance). The same ops on the same values: no
// value moves. See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "foldsim/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace foldsim {
#define GEN_PASS_DEF_OUTLINESCOPES
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {

/// Logic that may move into an instance: no state, no regions, no effects.
bool isLogic(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         op->getNumResults() >= 1 &&
         !isa<hw::InstanceOp, hw::ConstantOp, seq::FirRegOp,
              seq::FirMemReadOp, seq::FirMemReadWriteOp>(op);
}

StringRef regName(seq::FirRegOp reg) { return reg.getName(); }

/// One iteration's ops in canonical order and the values it reads.
struct Group {
  unsigned index;
  /// The ops (registers and logic), in operand-before-user order from the
  /// registers sorted by their name without the scope prefix.
  SmallVector<Operation *> ops;
  DenseSet<Operation *> set;
  /// Values read from outside the group (not constants), in first-use order.
  SmallVector<Value> inputs;
  /// Constant operands, in first-use order (one per use position).
  SmallVector<hw::ConstantOp> constants;
  /// The canonical text of the group (constant values left out).
  std::string text;
};

struct OutlineScopesPass
    : public foldsim::impl::OutlineScopesBase<OutlineScopesPass> {
  using OutlineScopesBase::OutlineScopesBase;

  /// The group's ops: its registers, then logic whose users are all in it.
  void collect(Group &g, ArrayRef<seq::FirRegOp> regs, Block *body,
               const DenseSet<Operation *> &excluded) {
    g.set.clear();
    for (auto reg : regs)
      g.set.insert(reg);
    for (bool grew = true; grew;) {
      grew = false;
      SmallVector<Operation *> cand;
      for (Operation *op : g.set)
        for (Value v : op->getOperands())
          if (auto *d = v.getDefiningOp())
            if (d->getBlock() == body && !g.set.contains(d) && isLogic(d) &&
                !excluded.contains(d))
              cand.push_back(d);
      for (Operation *d : cand) {
        if (g.set.contains(d))
          continue;
        bool alone = llvm::all_of(d->getUsers(), [&](Operation *u) {
          return g.set.contains(u);
        });
        if (alone) {
          g.set.insert(d);
          grew = true;
        }
      }
    }
  }

  /// Orders the group and writes its canonical text: op names, attributes
  /// (but names), types, and operands as positions (internal), input numbers
  /// (external) or constant types (their values apart).
  void canonicalize(Group &g, ArrayRef<seq::FirRegOp> regs) {
    g.ops.clear();
    g.inputs.clear();
    g.constants.clear();
    DenseSet<Operation *> done;
    DenseMap<Operation *, unsigned> pos;
    // Operand first, without recursion.
    for (auto reg : regs) {
      SmallVector<std::pair<Operation *, unsigned>> stack{{reg, 0}};
      if (!done.insert(reg).second)
        continue;
      while (!stack.empty()) {
        auto &[op, next] = stack.back();
        if (next < op->getNumOperands()) {
          auto *d = op->getOperand(next++).getDefiningOp();
          // (A register's own next value may read it: the register is placed
          // first, the logic after; a cycle closes only through a register.)
          if (d && g.set.contains(d) && !isa<seq::FirRegOp>(d) &&
              done.insert(d).second)
            stack.push_back({d, 0});
          continue;
        }
        pos[op] = g.ops.size();
        g.ops.push_back(op);
        stack.pop_back();
      }
    }
    // Registers read by logic before they are placed: place every register
    // first (stable: registers in name order, then the logic in its order).
    SmallVector<Operation *> ordered;
    for (auto reg : regs)
      ordered.push_back(reg);
    for (auto *op : g.ops)
      if (!isa<seq::FirRegOp>(op))
        ordered.push_back(op);
    g.ops = ordered;
    pos.clear();
    for (auto [i, op] : llvm::enumerate(g.ops))
      pos[op] = i;
    DenseMap<Value, unsigned> inputOf;
    llvm::raw_string_ostream os(g.text);
    g.text.clear();
    for (auto *op : g.ops) {
      os << op->getName().getStringRef() << "(";
      for (Value v : op->getOperands()) {
        auto *d = v.getDefiningOp();
        if (d && g.set.contains(d)) {
          os << "p" << pos[d] << "." << cast<OpResult>(v).getResultNumber();
        } else if (auto c = dyn_cast_or_null<hw::ConstantOp>(d)) {
          os << "c" << c.getType();
          g.constants.push_back(c);
        } else {
          auto [it, fresh] = inputOf.try_emplace(v, g.inputs.size());
          if (fresh)
            g.inputs.push_back(v);
          os << "i" << it->second << ":" << v.getType();
        }
        os << ",";
      }
      os << ")";
      for (auto attr : op->getAttrs())
        if (attr.getName() != "name" && attr.getName() != "sv.namehint" &&
            attr.getName() != "sv.attributes")
          os << attr.getName().getValue() << "=" << attr.getValue() << ";";
      for (auto t : op->getResultTypes())
        os << ":" << t;
      os << "\n";
    }
  }

  LogicalResult outline(hw::HWModuleOp module, StringRef prefix,
                        StringRef newName, mlir::ModuleOp top) {
    Block *body = module.getBodyBlock();
    llvm::Regex re("^" + llvm::Regex::escape(prefix) + "([0-9]+)\\.(.*)$");
    std::map<unsigned, SmallVector<seq::FirRegOp>> byIndex;
    for (auto reg : body->getOps<seq::FirRegOp>()) {
      SmallVector<StringRef> m;
      if (reg.getInnerSymAttr() || !re.match(regName(reg), &m))
        continue;
      unsigned k;
      if (m[1].getAsInteger(10, k))
        continue;
      byIndex[k].push_back(reg);
    }
    if (byIndex.size() < 2) {
      llvm::errs() << "outline-scopes: " << module.getModuleName() << " "
                   << prefix << ": fewer than two iterations, none outlined\n";
      return success();
    }
    auto stripped = [&](seq::FirRegOp reg) {
      SmallVector<StringRef> m;
      re.match(regName(reg), &m);
      return m[2].str();
    };
    std::vector<Group> groups;
    SmallVector<SmallVector<seq::FirRegOp>> regsOf;
    for (auto &[k, regs] : byIndex) {
      llvm::stable_sort(regs, [&](seq::FirRegOp a, seq::FirRegOp b) {
        return stripped(a) < stripped(b);
      });
      regsOf.push_back(regs);
      groups.emplace_back();
      groups.back().index = k;
    }
    // The iterations must be the same logic: ops whose attributes differ
    // between iterations (an extract of each iteration's own bit) stay in
    // the parent, their results inputs; repeated until they agree.
    DenseSet<Operation *> excluded;
    for (unsigned round = 0;; ++round) {
      for (auto [g, regs] : llvm::zip(groups, regsOf)) {
        collect(g, regs, body, excluded);
        canonicalize(g, regs);
      }
      bool same = llvm::all_of(groups, [&](Group &g) {
        return g.ops.size() == groups[0].ops.size();
      });
      if (!same) {
        llvm::errs() << "outline-scopes: " << module.getModuleName() << " "
                     << prefix << ": iterations differ in size ("
                     << groups[0].ops.size() << " vs others), none outlined\n";
        return success();
      }
      bool changed = false;
      for (unsigned i = 0; i < groups[0].ops.size(); ++i) {
        auto line = [&](Group &g) {
          // The op's text line.
          StringRef t(g.text);
          for (unsigned l = 0; l < i; ++l)
            t = t.split('\n').second;
          return t.split('\n').first.str();
        };
        std::string l0 = line(groups[0]);
        bool agree = llvm::all_of(groups, [&](Group &g) { return line(g) == l0; });
        if (agree || isa<seq::FirRegOp>(groups[0].ops[i]))
          continue;
        for (auto &g : groups)
          excluded.insert(g.ops[i]);
        changed = true;
      }
      if (!changed)
        break;
      if (round > 64)
        return module.emitError("outline-scopes: no agreement after 64 rounds");
    }
    for (auto &g : groups)
      if (g.text != groups[0].text) {
        llvm::errs() << "outline-scopes: " << module.getModuleName() << " "
                     << prefix << ": registers differ between iterations, none "
                        "outlined\n";
        return success();
      }
    // Constant positions: the same value in every iteration is a constant of
    // the module, else an input.
    unsigned nc = groups[0].constants.size();
    SmallVector<bool> constInput(nc, false);
    for (unsigned c = 0; c < nc; ++c)
      for (auto &g : groups)
        if (g.constants[c].getValue() != groups[0].constants[c].getValue())
          constInput[c] = true;
    // The outputs: a group value read outside it, by op position and result.
    SmallVector<std::pair<unsigned, unsigned>> outs;
    {
      DenseSet<std::pair<unsigned, unsigned>> seen;
      for (auto &g : groups)
        for (auto [i, op] : llvm::enumerate(g.ops))
          for (auto r : op->getResults())
            if (llvm::any_of(r.getUsers(), [&](Operation *u) { return !g.set.contains(u); }))
              if (seen.insert({unsigned(i), r.getResultNumber()}).second)
                outs.push_back({unsigned(i), r.getResultNumber()});
      llvm::sort(outs);
    }
    // The module, from iteration 0.
    auto &g0 = groups[0];
    SmallVector<hw::PortInfo> ports;
    auto *ctx = module.getContext();
    unsigned argNum = 0;
    for (auto [i, v] : llvm::enumerate(g0.inputs)) {
      hw::PortInfo p;
      p.name = StringAttr::get(ctx, "in" + Twine(i));
      p.type = v.getType();
      p.dir = hw::ModulePort::Direction::Input;
      p.argNum = argNum++;
      ports.push_back(p);
    }
    SmallVector<unsigned> constArg(nc, ~0u);
    for (unsigned c = 0; c < nc; ++c)
      if (constInput[c]) {
        hw::PortInfo p;
        p.name = StringAttr::get(ctx, "k" + Twine(c));
        p.type = g0.constants[c].getType();
        p.dir = hw::ModulePort::Direction::Input;
        p.argNum = argNum;
        constArg[c] = argNum++;
        ports.push_back(p);
      }
    for (auto [o, pr] : llvm::enumerate(outs)) {
      hw::PortInfo p;
      p.name = StringAttr::get(ctx, "out" + Twine(o));
      p.type = g0.ops[pr.first]->getResult(pr.second).getType();
      p.dir = hw::ModulePort::Direction::Output;
      p.argNum = o;
      ports.push_back(p);
    }
    OpBuilder mb(module);
    auto newMod = hw::HWModuleOp::create(mb, module.getLoc(),
                                         StringAttr::get(ctx, newName),
                                         hw::ModulePortInfo(ports));
    auto *nb = newMod.getBodyBlock();
    nb->getTerminator()->erase();
    OpBuilder b = OpBuilder::atBlockEnd(nb);
    IRMapping map;
    for (auto [i, v] : llvm::enumerate(g0.inputs))
      map.map(v, nb->getArgument(i));
    // Constants: each use position its own (a shared constant op used at two
    // positions is mapped per use below).
    SmallVector<Operation *> clones;
    unsigned cpos = 0;
    for (auto *op : g0.ops) {
      auto *clone = b.clone(*op, map);
      // Its constant operands, in order, from the position list.
      for (auto &operand : clone->getOpOperands()) {
        auto *d = op->getOperand(operand.getOperandNumber()).getDefiningOp();
        if (!isa_and_nonnull<hw::ConstantOp>(d))
          continue;
        unsigned c = cpos++;
        if (constInput[c]) {
          operand.set(nb->getArgument(constArg[c]));
        } else {
          OpBuilder cb(clone);
          operand.set(hw::ConstantOp::create(cb, clone->getLoc(),
                                             g0.constants[c].getValue()));
        }
      }
      if (auto reg = dyn_cast<seq::FirRegOp>(clone))
        reg.setName(stripped(cast<seq::FirRegOp>(op)));
      clones.push_back(clone);
    }
    // Registers placed first: their next values are mapped once all ops are.
    for (auto [op, clone] : llvm::zip(g0.ops, clones))
      for (auto &operand : clone->getOpOperands())
        if (auto mapped = map.lookupOrNull(op->getOperand(operand.getOperandNumber())))
          if (!isa_and_nonnull<hw::ConstantOp>(
                  op->getOperand(operand.getOperandNumber()).getDefiningOp()))
            operand.set(mapped);
    SmallVector<Value> results;
    for (auto [i, r] : outs)
      results.push_back(clones[i]->getResult(r));
    hw::OutputOp::create(b, module.getLoc(), results);
    // Each iteration an instance. (An input that is an earlier iteration's
    // value reads that iteration's instance.)
    DenseMap<Value, Value> replaced;
    for (auto &g : groups) {
      OpBuilder ib(body->getTerminator());
      SmallVector<Value> operands;
      for (Value v : g.inputs)
        operands.push_back(replaced.lookup(v) ? replaced.lookup(v) : v);
      for (unsigned c = 0; c < nc; ++c)
        if (constInput[c])
          operands.push_back(g.constants[c]);
      auto inst = hw::InstanceOp::create(
          ib, module.getLoc(), newMod,
          StringAttr::get(ctx, prefix + Twine(g.index) + "." + newName),
          operands);
      for (auto [o, pr] : llvm::enumerate(outs)) {
        Value old = g.ops[pr.first]->getResult(pr.second);
        replaced[old] = inst.getResult(o);
        old.replaceUsesWithIf(inst.getResult(o), [&](OpOperand &u) {
          return !g.set.contains(u.getOwner());
        });
      }
      for (auto *op : g.ops)
        op->dropAllUses();
      for (auto *op : llvm::reverse(g.ops))
        op->erase();
    }
    llvm::errs() << "outline-scopes: cell " << newName << " ("
                 << module.getModuleName() << " " << prefix << "*, "
                 << groups.size() << " iterations, " << g0.ops.size()
                 << " ops, " << g0.inputs.size() << " inputs, "
                 << llvm::count(constInput, true) << " constant inputs, "
                 << outs.size() << " outputs)\n";
    return success();
  }

  void runOnOperation() override {
    auto top = getOperation();
    for (auto &spec : scopes) {
      // MODULE:PREFIX=NAME
      auto [mod, rest] = StringRef(spec).split(':');
      auto [prefix, name] = rest.split('=');
      auto module = top.lookupSymbol<hw::HWModuleOp>(mod);
      if (!module || prefix.empty() || name.empty()) {
        top.emitError("outline-scopes: bad scope '") << spec << "'";
        return signalPassFailure();
      }
      if (failed(outline(module, prefix, name, top)))
        return signalPassFailure();
    }
  }
};

} // namespace
