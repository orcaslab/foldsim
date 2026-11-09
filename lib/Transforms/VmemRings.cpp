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
  int frame = int(ii);
  auto wrap = [&](int p) { return unsigned(((p % frame) + frame) % frame); };
  for (auto *row : {&plan.prefValid, &plan.prefInst, &plan.prefEntry, &plan.useValid,
                    &plan.useEntry, &plan.useOwner, &plan.useCross, &plan.useLast})
    row->assign(ii + 1, 0);
  if (instances.empty() || !ii) {
    plan.why = "no instances";
    return plan;
  }
  // The use groups: an instance's uses, a new group where the next use is
  // more than a lead after the last.
  struct Group {
    unsigned inst;
    SmallVector<int> uses;
    int lo, hi, want, at = 0;
  };
  SmallVector<Group> groups;
  for (auto [i, inst] : llvm::enumerate(instances)) {
    if (inst.uses.empty()) {
      plan.why = "an instance without a use";
      return plan;
    }
    SmallVector<int> uses(inst.uses.begin(), inst.uses.end());
    llvm::sort(uses);
    for (auto [k, u] : llvm::enumerate(uses))
      if (!k || u - uses[k - 1] > int(lead))
        groups.push_back({unsigned(i), {u}, 0, 0, 0});
      else
        groups.back().uses.push_back(u);
  }
  for (auto &g : groups) {
    g.lo = instances[g.inst].issue + int(minGap);
    g.hi = g.uses.front() - 1;
    if (g.lo > g.hi) {
      plan.why = "instance " + std::to_string(g.inst) + ": " +
                 std::to_string(g.uses.front() - instances[g.inst].issue) +
                 " positions from its read's issue to its next first use, fewer than " +
                 std::to_string(minGap + 1);
      return plan;
    }
    g.want = std::max(g.lo, g.uses.front() - int(lead));
  }
  unsigned n = groups.size();
  plan.prefetches = n;
  if (n > ii) {
    plan.why = std::to_string(n) + " prefetches in " + std::to_string(ii) + " positions";
    return plan;
  }
  // One prefetch a position: in order of the wanted positions, each the
  // first free at or after its wanted one, else the last free before it.
  SmallVector<unsigned> order(n);
  std::iota(order.begin(), order.end(), 0u);
  llvm::stable_sort(order, [&](unsigned x, unsigned y) { return groups[x].want < groups[y].want; });
  SmallVector<bool> taken(ii, false);
  for (unsigned x : order) {
    auto &g = groups[x];
    int at = INT32_MIN;
    for (int p = g.want; p <= g.hi && at == INT32_MIN; ++p)
      if (!taken[wrap(p)])
        at = p;
    for (int p = g.want - 1; p >= g.lo && at == INT32_MIN; --p)
      if (!taken[wrap(p)])
        at = p;
    if (at == INT32_MIN) {
      plan.why = "instance " + std::to_string(g.inst) + ": no free position for a prefetch";
      return plan;
    }
    taken[wrap(at)] = true;
    g.at = at;
  }
  for (auto &g : groups) {
    // The next frame's prefetch comes after the group's last use.
    if (g.uses.back() - g.at >= frame) {
      plan.why = "instance " + std::to_string(g.inst) + ": a use group spans a frame";
      return plan;
    }
    for (int u : g.uses) {
      int cross = floorDiv(u, frame) - floorDiv(g.at, frame);
      if (plan.useValid[wrap(u)] || cross < 0 || cross > 1) {
        plan.why = plan.useValid[wrap(u)]
                       ? "two uses at position " + std::to_string(wrap(u))
                       : "a use " + std::to_string(cross) + " frames after its prefetch";
        return plan;
      }
    }
  }
  // Two live intervals [prefetch, last use], each shorter than a frame and
  // repeated every frame, meet.
  auto meet = [&](const Group &x, const Group &y) {
    for (int k = -2; k <= 2; ++k)
      if (std::max(x.at, y.at + k * frame) <= std::min(x.uses.back(), y.uses.back() + k * frame))
        return true;
    return false;
  };
  for (unsigned p = 0; p < ii; ++p) {
    unsigned live = 0;
    for (auto &g : groups)
      if (int(wrap(int(p) - g.at)) <= g.uses.back() - g.at)
        ++live;
    plan.window = std::max(plan.window, live);
  }
  // Entries by rank in the frame's prefetches: consecutive prefetches take
  // consecutive entries, so R a little over the window suffices, but where a
  // frame's wrap meets R.
  SmallVector<unsigned> byAt(n);
  std::iota(byAt.begin(), byAt.end(), 0u);
  llvm::stable_sort(byAt, [&](unsigned x, unsigned y) {
    return wrap(groups[x].at) < wrap(groups[y].at);
  });
  for (unsigned r = std::max(2u, plan.window); r <= std::max(2u, std::min(n, maxEntries)); ++r) {
    bool clash = false;
    for (unsigned a = 0; a < n && !clash; ++a)
      for (unsigned c = a + r; c < n && !clash; c += r)
        clash = meet(groups[byAt[a]], groups[byAt[c]]);
    if (clash)
      continue;
    plan.entries = r;
    plan.ok = true;
    for (auto [k, x] : llvm::enumerate(byAt)) {
      auto &g = groups[x];
      unsigned q = wrap(g.at), entry = unsigned(k) % r;
      plan.prefValid[q] = 1;
      plan.prefInst[q] = g.inst;
      plan.prefEntry[q] = entry;
      for (int u : g.uses) {
        unsigned at = wrap(u);
        plan.useValid[at] = 1;
        plan.useEntry[at] = entry;
        plan.useOwner[at] = q;
        plan.useCross[at] = unsigned(floorDiv(u, frame) - floorDiv(g.at, frame));
        plan.useLast[at] = u == g.uses.back();
      }
    }
    return plan;
  }
  plan.why = "more than " + std::to_string(maxEntries) + " entries (a window of " +
             std::to_string(plan.window) + ")";
  return plan;
}
