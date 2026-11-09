//===- CsfMailboxes.h - Reader-indexed link mailboxes (CSF) --------------===//
//
// foldsim-fold-static's `csf-mailbox-builder`: the links from one producer
// class to one reader class, each a memory of the producer's instances today,
// become lanes of one reader-indexed mailbox: a row per reader instance, each
// link a lane of the row, written by the producer instance it reads at its
// pass (a masked write of that lane), read whole at the reader's pass. A
// reader input of fan-in k then reads one row, not k memories of every
// producer instance. This file holds the plan (which links share a mailbox)
// and the row layout; StaticFold.cpp builds the memories. See CsfMailboxes.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_CSFMAILBOXES_H
#define FOLDSIM_CSFMAILBOXES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace foldsim {

/// A link offered to the mailbox builder: a reader input's piece, from one
/// producer source, whose every producer instance is read by at most one
/// reader instance (no fan-out: the producer writes one row a pass).
struct MailboxLink {
  /// The producer and reader classes, and the write's timing key: links of
  /// one mailbox are written in the same host cycle of the producer's pass.
  unsigned pcls = 0, rcls = 0, writeKey = 0;
  unsigned width = 0;
  /// Per producer instance: the reader instance that takes it, or -1.
  llvm::SmallVector<int64_t> readerOf;
};

/// A chain link: every instance j of a class but the last read by its
/// instance j + 1 (or every one but the first by j - 1), a self-link of a
/// chain class. It stays an on-chip one-entry forward: constant in N, and a
/// mailbox's latency would multiply along the chain.
inline bool isChainLink(const MailboxLink &link) {
  if (link.pcls != link.rcls || link.readerOf.size() < 2)
    return false;
  size_t up = 0, down = 0;
  for (auto [j, r] : llvm::enumerate(link.readerOf)) {
    up += r == int64_t(j) + 1;
    down += r == int64_t(j) - 1;
  }
  size_t chain = link.readerOf.size() - 1;
  return up == chain || down == chain;
}

/// A mailbox: links of one class pair and write key whose producer instances
/// write one reader row each (`rowOf`, consistent over its links).
struct MailboxPlan {
  unsigned pcls = 0, rcls = 0, writeKey = 0;
  llvm::SmallVector<unsigned> links;
  /// Per producer instance: the reader instance whose row it writes, or -1.
  llvm::SmallVector<int64_t> rowOf;
  /// Distinct write domains (sets of producer instances) among its links:
  /// lanes of one domain are written together and share mask bytes.
  unsigned domains = 0;
};

/// Plans the mailboxes of `links`: each joins the first plan of its class
/// pair and write key whose rows it agrees with (a producer instance writes
/// one reader row), else opens one. Plans of fewer than `minDomains` write
/// domains are dropped (a mailbox of one domain is today's link memory, its
/// links merged by foldsim-merge-memories). `planOf` gets each link's plan
/// index, or -1.
void planMailboxes(llvm::ArrayRef<MailboxLink> links, unsigned minDomains,
                   llvm::SmallVectorImpl<MailboxPlan> &plans,
                   llvm::SmallVectorImpl<int> &planOf);

/// A built mailbox's row: its lanes (the links built into it, by domain) cut
/// into pieces of at most `maxBits`, each a memory with its own controls.
/// A piece of several domains is byte-masked (8-bit bytes; wider than 64
/// bits, a multiple of 64, so foldsim-split-masked-memories cuts it into
/// 64-bit columns), each domain's bytes its own; a piece of one domain is
/// unmasked and as wide as its bits.
struct MailboxLayout {
  struct Segment {
    /// The lane, its low bit, the piece and the bit there, and the width.
    unsigned lane, laneLow, piece, offset, width;
  };
  struct Piece {
    unsigned width = 0;
    bool masked = false;
    /// Per mask byte: its domain (masked pieces).
    llvm::SmallVector<unsigned> byteDomain;
    /// The domain of an unmasked piece.
    unsigned domain = 0;
  };
  llvm::SmallVector<Segment> segments;
  llvm::SmallVector<Piece> pieces;
  /// Bits of the row (the pieces' widths) and of the lanes alone.
  uint64_t rowBits = 0, laneBits = 0;
};

/// Lays out lanes of `widths`, lane i of write domain `domainOf[i]`.
MailboxLayout layoutMailbox(llvm::ArrayRef<unsigned> widths,
                            llvm::ArrayRef<unsigned> domainOf,
                            unsigned maxBits);

} // namespace foldsim

#endif // FOLDSIM_CSFMAILBOXES_H
