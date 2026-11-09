//===- VmemRings.cpp - Fixed read rings for virtual memories --------------===//
//
// The planning half of foldsim-fold-static's `vmem-rings` (VmemRings.h).
//
//===----------------------------------------------------------------------===//

#include "VmemRings.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <numeric>

using namespace foldsim;
using llvm::SmallVector;

static int floorDiv(int x, int d) { return x >= 0 ? x / d : -((-x + d - 1) / d); }

RingPlan foldsim::planRing(llvm::ArrayRef<RingInstance> instances, unsigned ii,
                           unsigned lead, unsigned minGap, unsigned maxEntries) {
  RingPlan plan;
  unsigned n = instances.size();
  int frame = int(ii);
  auto wrap = [&](int p) { return unsigned(((p % frame) + frame) % frame); };
  plan.prefValid.assign(ii + 1, 0);
  plan.prefInst.assign(ii + 1, 0);
  plan.useValid.assign(ii + 1, 0);
  plan.useInst.assign(ii + 1, 0);
  plan.useCross.assign(ii + 1, 0);
  plan.useLast.assign(ii + 1, 0);
  plan.prefetch.assign(n, 0);
  if (!n || !ii) {
    plan.why = "no instances";
    return plan;
  }
  SmallVector<int> first(n), last(n), lo(n), hi(n), want(n);
  for (auto [i, inst] : llvm::enumerate(instances)) {
    if (inst.uses.empty()) {
      plan.why = "an instance without a use";
      return plan;
    }
    first[i] = *llvm::min_element(inst.uses);
    last[i] = *llvm::max_element(inst.uses);
    lo[i] = inst.issue + int(minGap);
    hi[i] = first[i] - 1;
    if (lo[i] > hi[i]) {
      plan.why = "instance " + std::to_string(i) + ": " +
                 std::to_string(first[i] - inst.issue) +
                 " positions from its read's issue to its next first use, "
                 "fewer than " + std::to_string(minGap + 1);
      return plan;
    }
    want[i] = std::max(lo[i], first[i] - int(lead));
  }
  // One prefetch a position: in order of the wanted positions, each the
  // first free at or after its wanted one, else the last free before it.
  SmallVector<unsigned> order(n);
  std::iota(order.begin(), order.end(), 0u);
  llvm::stable_sort(order, [&](unsigned x, unsigned y) { return want[x] < want[y]; });
  SmallVector<bool> taken(ii, false);
  for (unsigned i : order) {
    int at = INT32_MIN;
    for (int p = want[i]; p <= hi[i] && at == INT32_MIN; ++p)
      if (!taken[wrap(p)])
        at = p;
    for (int p = want[i] - 1; p >= lo[i] && at == INT32_MIN; --p)
      if (!taken[wrap(p)])
        at = p;
    if (at == INT32_MIN) {
      plan.why = "instance " + std::to_string(i) + ": no free position for its prefetch";
      return plan;
    }
    taken[wrap(at)] = true;
    plan.prefetch[i] = at;
    plan.prefValid[wrap(at)] = 1;
    plan.prefInst[wrap(at)] = i;
  }
  for (auto [i, inst] : llvm::enumerate(instances)) {
    // The next frame's prefetch of the instance comes after its last use.
    if (last[i] - plan.prefetch[i] >= frame) {
      plan.why = "instance " + std::to_string(i) + ": its uses span a frame";
      return plan;
    }
    for (int u : inst.uses) {
      unsigned at = wrap(u);
      if (plan.useValid[at] && plan.useInst[at] != i) {
        plan.why = "two instances use position " + std::to_string(at);
        return plan;
      }
      int cross = floorDiv(u, frame) - floorDiv(plan.prefetch[i], frame);
      if (cross < 0 || cross > 1) {
        plan.why = "instance " + std::to_string(i) + ": a use " +
                   std::to_string(cross) + " frames after its prefetch";
        return plan;
      }
      plan.useValid[at] = 1;
      plan.useInst[at] = i;
      plan.useCross[at] = unsigned(cross);
      plan.useLast[at] = u == last[i];
    }
  }
  // Two live intervals [prefetch, last use], each shorter than a frame and
  // repeated every frame, meet.
  auto meet = [&](unsigned x, unsigned y) {
    for (int k = -2; k <= 2; ++k) {
      int a1 = plan.prefetch[x], a2 = last[x];
      int b1 = plan.prefetch[y] + k * frame, b2 = last[y] + k * frame;
      if (std::max(a1, b1) <= std::min(a2, b2))
        return true;
    }
    return false;
  };
  for (unsigned p = 0; p < ii; ++p) {
    unsigned live = 0;
    for (unsigned i = 0; i < n; ++i) {
      int span = last[i] - plan.prefetch[i];
      int off = int(p) - plan.prefetch[i];
      if (((off % frame) + frame) % frame <= span)
        ++live;
    }
    plan.window = std::max(plan.window, live);
  }
  // Entries by rank in the prefetches' order (a frame's, from its first
  // position): consecutive prefetches take consecutive entries, so R a little
  // over the window suffices but where a frame's wrap meets R.
  SmallVector<unsigned> byQ(n);
  std::iota(byQ.begin(), byQ.end(), 0u);
  llvm::stable_sort(byQ, [&](unsigned x, unsigned y) {
    return wrap(plan.prefetch[x]) < wrap(plan.prefetch[y]);
  });
  SmallVector<unsigned> rank(n);
  for (auto [k, i] : llvm::enumerate(byQ))
    rank[i] = k;
  for (unsigned r = std::max(1u, plan.window); r <= std::min(n, std::max(1u, maxEntries)); ++r) {
    bool clash = false;
    for (unsigned a = 0; a < n && !clash; ++a)
      for (unsigned c = a + r; c < n && !clash; c += r)
        clash = meet(byQ[a], byQ[c]);
    if (clash)
      continue;
    plan.entries = r;
    plan.ok = true;
    plan.entryOf.assign(n, 0);
    plan.prefEntry.assign(ii + 1, 0);
    plan.useEntry.assign(ii + 1, 0);
    for (unsigned i = 0; i < n; ++i)
      plan.entryOf[i] = rank[i] % r;
    for (unsigned p = 0; p < ii; ++p) {
      if (plan.prefValid[p])
        plan.prefEntry[p] = plan.entryOf[plan.prefInst[p]];
      if (plan.useValid[p])
        plan.useEntry[p] = plan.entryOf[plan.useInst[p]];
    }
    return plan;
  }
  plan.why = "more than " + std::to_string(maxEntries) +
             " entries (a window of " + std::to_string(plan.window) + ")";
  return plan;
}
