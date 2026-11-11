//===- AffineSlots.cpp - Affine placement of a modulo schedule ------------===//
//
// A modulo list schedule places each evaluation alone, at the first free slot
// its predecessors allow, so a class's instances land in an order the
// dependences and the other classes' slots happen to leave: its slot tables
// then need a row per position (O(II)), or a segment per few instances
// (O(N)). Here a group (a resource's passes of one kind) is placed whole: its
// k-th instance at o + k*P, P the resource's stride, a power of two at least
// its groups' count and least spacing. Only the offsets o are chosen:
//
// - Every dependence n <- p (latency l) between groups g(n), g(p) bounds
//   o_g(n) >= o_g(p) + k_p*P_g(p) + l - k_n*P_g(n): a difference constraint
//   between offsets, the most of it over the group pair's dependences. A
//   dependence within a group needs (k_n - k_p) * P >= l.
// - The offsets are the longest paths of those constraints (Bellman-Ford from
//   each group's earliest), each group then put at the first offset from
//   there that its resource leaves free for all its evaluations, in offset
//   order; where a group moved past a successor placed before it, its bounds
//   rise and the round repeats, until every constraint holds (or an offset
//   passes two frames: the frame is too short).
//
//===----------------------------------------------------------------------===//

#include "AffineSlots.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <numeric>

using namespace llvm;

namespace foldsim {

SmallVector<unsigned> affineStrides(ArrayRef<AffineGroup> groups, bool exact) {
  // Each group its own stride: the resource's group count (so its kinds of
  // passes interleave) or its own spacing, whichever is more. A paged
  // copy's state passes keep their page apart; its ready passes, which take
  // no page port, are not held to it (one sweep of the frame, not two).
  DenseMap<unsigned, unsigned> count;
  for (auto &g : groups)
    ++count[g.resource];
  SmallVector<unsigned> strides;
  for (auto &g : groups) {
    unsigned least = std::max({1u, g.minStride, count[g.resource]});
    strides.push_back(exact ? least : unsigned(PowerOf2Ceil(least)));
  }
  return strides;
}

bool placeAffine(unsigned period, ArrayRef<AffineNode> nodes,
                 ArrayRef<AffineGroup> groups, ArrayRef<unsigned> strides,
                 ArrayRef<SmallVector<std::pair<unsigned, int>>> preds,
                 function_ref<bool(unsigned, int)> fits,
                 function_ref<void(unsigned, int, bool)> take,
                 SmallVectorImpl<int> &slotOf, std::string &why,
                 function_ref<std::string(unsigned)> groupName) {
  unsigned numGroups = groups.size();
  const long limit = 2 * long(period);
  // The group's nodes by rank, and each group pair's bound.
  SmallVector<SmallVector<unsigned>> members(numGroups);
  for (auto [n, node] : llvm::enumerate(nodes)) {
    auto &m = members[node.group];
    if (m.size() <= node.index)
      m.resize(node.index + 1, ~0u);
    m[node.index] = unsigned(n);
  }
  SmallVector<long> lower(numGroups, 0);
  for (auto [n, node] : llvm::enumerate(nodes))
    lower[node.group] = std::max(
        lower[node.group], long(node.earliest) - long(node.index) * strides[node.group]);
  std::map<std::pair<unsigned, unsigned>, long> bound;
  // The ranks of the dependence that sets each pair's bound (the log's).
  std::map<std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned>> why_;
  for (auto [n, node] : llvm::enumerate(nodes))
    for (auto [p, latency] : preds[n]) {
      auto &pred = nodes[p];
      long need = long(pred.index) * strides[pred.group] + latency -
                  long(node.index) * strides[node.group];
      if (pred.group == node.group) {
        if (need > 0) {
          why = "affine: a dependence within a group runs against its "
                "instances' order (rank " +
                std::to_string(pred.index) + " -> " +
                std::to_string(node.index) + ")";
          return false;
        }
        continue;
      }
      auto key = std::make_pair(pred.group, node.group);
      auto it = bound.find(key);
      if (it == bound.end() || need > it->second) {
        bound[key] = need;
        why_[key] = {pred.index, node.index};
      }
    }
  SmallVector<std::tuple<unsigned, unsigned, long>> edges;
  for (auto &[key, w] : bound)
    edges.push_back({key.first, key.second, w});

  SmallVector<long> at(numGroups, 0);
  for (unsigned round = 0; round < 4 * numGroups + 16; ++round) {
    // Longest paths from the lower bounds.
    SmallVector<int> via(numGroups, -1);
    for (unsigned pass = 0;; ++pass) {
      bool changed = false;
      int last = -1;
      for (auto &[p, n, w] : edges)
        if (lower[p] + w > lower[n]) {
          lower[n] = lower[p] + w;
          via[n] = int(p);
          last = int(n);
          changed = true;
        }
      if (!changed)
        break;
      if (pass > numGroups) {
        why = "affine: a cycle of dependences between groups";
        if (groupName) {
          // The cycle: back along the last relaxations from a group that
          // moved (numGroups steps land inside it).
          int g = last;
          for (unsigned k = 0; k < numGroups && g >= 0; ++k)
            g = via[g];
          if (g >= 0) {
            why += " (";
            int h = g;
            for (unsigned k = 0; k <= numGroups; ++k) {
              int p = via[h];
              if (p < 0)
                break;
              auto key = std::make_pair(unsigned(p), unsigned(h));
              why += groupName(unsigned(h)) + " <-" + std::to_string(bound[key]) + " (ranks " +
                     std::to_string(why_[key].first) + "->" + std::to_string(why_[key].second) +
                     ")- ";
              h = p;
              if (h == g)
                break;
            }
            why += groupName(unsigned(g)) + ")";
          }
        }
        return false;
      }
    }
    for (unsigned g = 0; g < numGroups; ++g)
      if (lower[g] < 0)
        lower[g] = 0;
    // Placed in offset order, each at the first offset its resource leaves
    // free.
    SmallVector<unsigned> order(numGroups);
    std::iota(order.begin(), order.end(), 0u);
    llvm::stable_sort(order, [&](unsigned a, unsigned b) { return lower[a] < lower[b]; });
    SmallVector<std::pair<unsigned, int>> taken;
    bool placed = true;
    for (unsigned g : order) {
      auto &m = members[g];
      long o = lower[g];
      for (;;) {
        long last = o + long(m.size() - 1) * strides[g];
        if (last >= limit) {
          why = "affine: group of " + std::to_string(m.size()) +
                " (stride " + std::to_string(strides[g]) + ") past two frames";
          placed = false;
          break;
        }
        unsigned k = 0;
        for (; k < m.size(); ++k) {
          int t = int(o + long(k) * strides[g]);
          if (m[k] == ~0u)
            continue;
          if (!fits(m[k], t))
            break;
          take(m[k], t, true);
        }
        if (k == m.size())
          break;
        for (unsigned j = 0; j < k; ++j)
          if (m[j] != ~0u)
            take(m[j], int(o + long(j) * strides[g]), false);
        ++o;
      }
      if (!placed)
        break;
      at[g] = o;
      for (unsigned k = 0; k < m.size(); ++k)
        if (m[k] != ~0u)
          taken.push_back({m[k], int(o + long(k) * strides[g])});
    }
    bool ok = placed;
    if (ok)
      for (auto &[p, n, w] : edges)
        if (at[p] + w > at[n])
          ok = false;
    if (ok && groupName && getenv("FOLDSIM_AFFINE_LOG")) {
      // Each group's offset and the bound that set it (its tightest
      // predecessor group), for the frame placed.
      for (unsigned g = 0; g < numGroups; ++g) {
        long best = LONG_MIN;
        int from = -1;
        for (auto &[p, n, w] : edges)
          if (n == g && at[p] + w > best) {
            best = at[p] + w;
            from = int(p);
          }
        llvm::errs() << "affine: period " << period << " " << groupName(g)
                     << " x" << members[g].size() << " stride " << strides[g]
                     << " at " << at[g] << " (bound "
                     << (from < 0 ? std::string("none")
                                  : std::to_string(best) + " from " + groupName(from) +
                                        " (ranks " +
                                        std::to_string(why_[{unsigned(from), g}].first) +
                                        " -> " +
                                        std::to_string(why_[{unsigned(from), g}].second) + ")")
                     << ")\n";
      }
    }
    if (ok) {
      slotOf.assign(nodes.size(), -1);
      for (auto [n, t] : taken)
        slotOf[n] = t;
      return true;
    }
    // Free the round's slots; the placed offsets are the next bounds.
    for (auto [n, t] : taken)
      take(n, t, false);
    if (!placed)
      return false;
    bool raised = false;
    for (unsigned g = 0; g < numGroups; ++g)
      if (at[g] > lower[g]) {
        lower[g] = at[g];
        raised = true;
      }
    if (!raised) {
      why = "affine: no progress";
      return false;
    }
  }
  why = "affine: offsets did not settle";
  return false;
}

} // namespace foldsim
