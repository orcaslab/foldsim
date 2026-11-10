//===- PosCounters.cpp - The frame position's quotient and remainder ------===//
//
// A frame's position steps by one, holds (a bubble), or jumps to 0 (the next
// frame) or to `ii` (idle). Its quotient and remainder of a period D step the
// same way: the remainder by one, wrapping to 0 and carrying into the
// quotient at D; a jump loads the jump's own quotient and remainder. So a
// register of them beside each position register stays equal to the
// position's divmod at every host cycle, at the cost of a small counter, and
// a slot table can test a residue and scale a quotient instead of dividing.
//
//===----------------------------------------------------------------------===//

#include "PosCounters.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;

namespace foldsim {

PosCounters *&currentPosCounters() {
  static PosCounters *current = nullptr;
  return current;
}

void PosCounters::begin(unsigned ii_, unsigned posWidth_, Value clock_,
                        Value reset_, Value go_, Value advance_) {
  on = true;
  ii = ii_;
  posWidth = posWidth_;
  clock = clock_;
  reset = reset_;
  go = go_;
  advance = advance_;
  known.clear();
  cache.clear();
  rankCache.clear();
}

void PosCounters::addReg(Value reg, Value next) {
  if (on)
    known[reg] = {Reg, next, Value()};
}
void PosCounters::addNext(Value v, Value pos) {
  if (on)
    known[v] = {Next, pos, Value()};
}
void PosCounters::addNextFree(Value v, Value pos) {
  if (on)
    known[v] = {NextFree, pos, Value()};
}
void PosCounters::addShown(Value v, Value pos, Value bubble) {
  if (on)
    known[v] = {Shown, pos, bubble};
}

std::pair<Value, Value> PosCounters::divmod(ImplicitLocOpBuilder &b, Value v,
                                            unsigned d) {
  auto key = std::make_pair(v.getAsOpaquePointer(), d);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  unsigned rw = std::max(1u, llvm::Log2_32_Ceil(d));
  auto qType = b.getIntegerType(posWidth), rType = b.getIntegerType(rw);
  auto cq = [&](uint64_t x) -> Value { return hw::ConstantOp::create(b, qType, x); };
  auto cr = [&](uint64_t x) -> Value { return hw::ConstantOp::create(b, rType, x); };
  auto eqc = [&](Value x, uint64_t k) -> Value {
    return comb::ICmpOp::create(b, comb::ICmpPredicate::eq, x,
                                hw::ConstantOp::create(b, x.getType(), k));
  };
  auto mux = [&](Value c, std::pair<Value, Value> t, std::pair<Value, Value> f) {
    return std::make_pair(Value(comb::MuxOp::create(b, c, t.first, f.first)),
                          Value(comb::MuxOp::create(b, c, t.second, f.second)));
  };
  auto idle = std::make_pair(cq(ii / d), cr(ii % d));
  auto zero = std::make_pair(cq(0), cr(0));
  auto inc = [&](std::pair<Value, Value> qr) {
    Value wrap = eqc(qr.second, d - 1);
    return std::make_pair(
        Value(comb::MuxOp::create(b, wrap,
                                  comb::AddOp::create(b, qr.first, cq(1)),
                                  qr.first)),
        Value(comb::MuxOp::create(b, wrap, cr(0),
                                  comb::AddOp::create(b, qr.second, cr(1)))));
  };
  auto node = known.find(v);
  assert(node != known.end() && "a position the counters do not know");
  Node n = node->second;
  std::pair<Value, Value> out;
  switch (n.kind) {
  case Reg: {
    // Registers first (the position's next may read this one), then their
    // next values.
    auto rq = seq::FirRegOp::create(b, cq(ii / d), clock,
                                    b.getStringAttr("posq" + std::to_string(d)),
                                    reset, cq(ii / d), hw::InnerSymAttr{},
                                    /*isAsync=*/false,
                                    b.getIntegerAttr(qType, ii / d));
    auto rr = seq::FirRegOp::create(b, cr(ii % d), clock,
                                    b.getStringAttr("posr" + std::to_string(d)),
                                    reset, cr(ii % d), hw::InnerSymAttr{},
                                    /*isAsync=*/false,
                                    b.getIntegerAttr(rType, ii % d));
    out = {rq, rr};
    cache[key] = out;
    auto next = divmod(b, n.a, d);
    rq.getNextMutable().assign(next.first);
    rr.getNextMutable().assign(next.second);
    return out;
  }
  case Next: {
    auto p = divmod(b, n.a, d);
    Value end = comb::OrOp::create(b, eqc(n.a, ii - 1), eqc(n.a, ii));
    auto step = mux(end, mux(go, zero, idle), inc(p));
    out = advance ? mux(advance, step, p) : step;
    break;
  }
  case NextFree: {
    auto p = divmod(b, n.a, d);
    Value end = comb::OrOp::create(b, eqc(n.a, ii - 1), eqc(n.a, ii));
    out = mux(end, zero, inc(p));
    break;
  }
  case Shown:
    out = mux(n.bubble, idle, divmod(b, n.a, d));
    break;
  }
  cache[key] = out;
  return out;
}

Value PosCounters::rank(ImplicitLocOpBuilder &b, Value v, ArrayRef<uint64_t> set,
                        uint64_t key, function_ref<Value(Value)> inSet) {
  auto ck = std::make_pair(v.getAsOpaquePointer(), key);
  auto it = rankCache.find(ck);
  if (it != rankCache.end())
    return it->second;
  uint64_t total = 0;
  for (unsigned p = 0; p < ii && p < set.size(); ++p)
    total += set[p] ? 1 : 0;
  unsigned w = std::max(1u, unsigned(llvm::Log2_64_Ceil(total + 1)));
  auto type = b.getIntegerType(w);
  uint64_t mask = w >= 64 ? ~uint64_t(0) : (uint64_t(1) << w) - 1;
  auto c = [&](uint64_t x) -> Value { return hw::ConstantOp::create(b, type, x & mask); };
  auto eqc = [&](Value x, uint64_t k) -> Value {
    return comb::ICmpOp::create(b, comb::ICmpPredicate::eq, x,
                                hw::ConstantOp::create(b, x.getType(), k));
  };
  // The rank after `pos` stepped once: its own, and one more where `pos`
  // is in the set.
  auto stepped = [&](Value pos) -> Value {
    Value r = rank(b, pos, set, key, inSet);
    Value bit = inSet(pos);
    Value inc = w > 1 ? b.createOrFold<comb::ConcatOp>(
                            hw::ConstantOp::create(b, b.getIntegerType(w - 1), 0), bit)
                      : bit;
    return comb::AddOp::create(b, r, inc);
  };
  auto node = known.find(v);
  assert(node != known.end() && "a position the counters do not know");
  Node n = node->second;
  Value out;
  switch (n.kind) {
  case Reg: {
    auto reg = seq::FirRegOp::create(b, c(total), clock,
                                     b.getStringAttr("posrank" + std::to_string(key % 100000)),
                                     reset, c(total), hw::InnerSymAttr{},
                                     /*isAsync=*/false, b.getIntegerAttr(type, total));
    rankCache[ck] = reg;
    reg.getNextMutable().assign(rank(b, n.a, set, key, inSet));
    return reg;
  }
  case Next: {
    Value end = comb::OrOp::create(b, eqc(n.a, ii - 1), eqc(n.a, ii));
    Value step = comb::MuxOp::create(b, end, comb::MuxOp::create(b, go, c(0), c(total)),
                                     stepped(n.a));
    out = advance ? Value(comb::MuxOp::create(b, advance, step,
                                              rank(b, n.a, set, key, inSet)))
                  : step;
    break;
  }
  case NextFree: {
    Value end = comb::OrOp::create(b, eqc(n.a, ii - 1), eqc(n.a, ii));
    out = comb::MuxOp::create(b, end, c(0), stepped(n.a));
    break;
  }
  case Shown:
    out = comb::MuxOp::create(b, n.bubble, c(total), rank(b, n.a, set, key, inSet));
    break;
  }
  rankCache[ck] = out;
  return out;
}

} // namespace foldsim
