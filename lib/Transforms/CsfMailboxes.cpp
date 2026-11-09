//===- CsfMailboxes.cpp - Reader-indexed link mailboxes (CSF) ------------===//
//
// The plan and row layout of `csf-mailbox-builder` (CsfMailboxes.h).
//
//===----------------------------------------------------------------------===//

#include "CsfMailboxes.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>

using namespace llvm;

namespace foldsim {

void planMailboxes(ArrayRef<MailboxLink> links, unsigned minDomains,
                   SmallVectorImpl<MailboxPlan> &plans,
                   SmallVectorImpl<int> &planOf) {
  plans.clear();
  planOf.assign(links.size(), -1);
  // The open plans of each class pair and write key.
  std::map<std::tuple<unsigned, unsigned, unsigned>, SmallVector<unsigned>>
      open;
  for (auto [i, link] : llvm::enumerate(links)) {
    auto &candidates = open[{link.pcls, link.rcls, link.writeKey}];
    int chosen = -1;
    for (unsigned k : candidates) {
      auto &plan = plans[k];
      if (plan.rowOf.size() != link.readerOf.size())
        continue;
      bool agrees = true;
      for (auto [row, reader] : llvm::zip(plan.rowOf, link.readerOf))
        if (reader >= 0 && row >= 0 && row != reader) {
          agrees = false;
          break;
        }
      if (agrees) {
        chosen = int(k);
        break;
      }
    }
    if (chosen < 0) {
      chosen = int(plans.size());
      MailboxPlan plan;
      plan.pcls = link.pcls;
      plan.rcls = link.rcls;
      plan.writeKey = link.writeKey;
      plan.rowOf.assign(link.readerOf.size(), -1);
      plans.push_back(std::move(plan));
      candidates.push_back(unsigned(chosen));
    }
    auto &plan = plans[chosen];
    plan.links.push_back(unsigned(i));
    for (auto [row, reader] : llvm::zip(plan.rowOf, link.readerOf))
      if (reader >= 0)
        row = reader;
    planOf[i] = chosen;
  }
  // Each plan's write domains: the distinct sets of producer instances its
  // links are written by.
  SmallVector<int> renumber(plans.size(), -1);
  SmallVector<MailboxPlan> kept;
  for (auto [k, plan] : llvm::enumerate(plans)) {
    std::set<SmallVector<bool>> domains;
    for (unsigned i : plan.links) {
      SmallVector<bool> domain;
      for (int64_t reader : links[i].readerOf)
        domain.push_back(reader >= 0);
      domains.insert(domain);
    }
    plan.domains = unsigned(domains.size());
    if (plan.domains < minDomains)
      continue;
    renumber[k] = int(kept.size());
    kept.push_back(std::move(plan));
  }
  for (int &p : planOf)
    if (p >= 0)
      p = renumber[p];
  plans.assign(kept.begin(), kept.end());
}

MailboxLayout layoutMailbox(ArrayRef<unsigned> widths,
                            ArrayRef<unsigned> domainOf, unsigned maxBits) {
  MailboxLayout layout;
  maxBits = std::max(64u, maxBits / 64 * 64);
  SmallVector<unsigned> order(widths.size());
  for (unsigned i = 0; i < order.size(); ++i)
    order[i] = i;
  llvm::stable_sort(order, [&](unsigned x, unsigned y) {
    return domainOf[x] < domainOf[y];
  });
  // Lanes in domain order; a domain starts on a byte of its own, a piece
  // takes at most maxBits.
  unsigned piece = 0, offset = 0;
  SmallVector<unsigned> used{0};
  int domain = -1;
  for (unsigned lane : order) {
    if (int(domainOf[lane]) != domain) {
      domain = int(domainOf[lane]);
      offset = (offset + 7) / 8 * 8;
    }
    for (unsigned low = 0; low < widths[lane];) {
      if (offset >= maxBits) {
        ++piece;
        offset = 0;
        used.push_back(0);
      }
      unsigned n = std::min(widths[lane] - low, maxBits - offset);
      layout.segments.push_back({lane, low, piece, offset, n});
      offset += n;
      used[piece] = offset;
      low += n;
    }
    layout.laneBits += widths[lane];
  }
  layout.pieces.resize(used.size());
  for (auto [k, p] : llvm::enumerate(layout.pieces)) {
    std::set<unsigned> domains;
    for (auto &s : layout.segments)
      if (s.piece == k)
        domains.insert(domainOf[s.lane]);
    p.masked = domains.size() > 1;
    if (!p.masked) {
      p.width = used[k];
      p.domain = domains.empty() ? 0 : *domains.begin();
    } else {
      p.width = (used[k] + 7) / 8 * 8;
      if (p.width > 64)
        p.width = (p.width + 63) / 64 * 64;
      p.byteDomain.assign(p.width / 8, *domains.rbegin());
      for (auto &s : layout.segments)
        if (s.piece == k)
          for (unsigned byte = s.offset / 8;
               byte <= (s.offset + s.width - 1) / 8; ++byte)
            p.byteDomain[byte] = domainOf[s.lane];
    }
    layout.rowBits += p.width;
  }
  return layout;
}

} // namespace foldsim
