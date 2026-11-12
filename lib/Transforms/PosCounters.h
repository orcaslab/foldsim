//===- PosCounters.h - The frame position's quotient and remainder --------===//
//
// slot-segments' counters for periods that are not powers of two: beside each
// register of the frame's position (the decision's register, the
// sequencers, the delayed copies), a quotient and a remainder register of a
// period D, loaded as the position is (advance, a frame's end, the next
// frame's start or idle). A value the simulator derives from the position
// (its next, a bubble's `ii`) has its quotient and remainder derived the
// same way. A slot table read at a position can then be decoded with runs of
// any period: a residue compare and the quotient's shift-add. See
// PosCounters.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_POSCOUNTERS_H
#define FOLDSIM_POSCOUNTERS_H

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include <map>
#include <utility>

namespace foldsim {

class PosCounters {
public:
  /// The frame's length (`ii`: the idle position), the position's width,
  /// the host clock and reset, `go` (the next frame starts) and `advance`
  /// (null without bubbles).
  void begin(unsigned ii, unsigned posWidth, mlir::Value clock,
             mlir::Value reset, mlir::Value go, mlir::Value advance);
  void end() {
    known.clear();
    cache.clear();
    rankCache.clear();
    aheadCache.clear();
    on = false;
  }
  bool active() const { return on; }

  /// `reg` is a position register loaded with `next` (a position value), its
  /// reset `ii`.
  void addReg(mlir::Value reg, mlir::Value next);
  /// `v` is the position after `pos` (nextPosition: held without advance).
  void addNext(mlir::Value v, mlir::Value pos);
  /// `v` is `pos` + 1, 0 past the frame's end (no hold, no idle).
  void addNextFree(mlir::Value v, mlir::Value pos);
  /// `v` is `ii` in a bubble, else `pos`.
  void addShown(mlir::Value v, mlir::Value pos, mlir::Value bubble);
  bool knows(mlir::Value v) const { return on && known.count(v); }

  /// `v`'s quotient and remainder of the period `d` (built at first use).
  std::pair<mlir::Value, mlir::Value> divmod(mlir::ImplicitLocOpBuilder &b,
                                             mlir::Value v, unsigned d);

  /// `v`'s rank in the set `set` of the frame's positions (rows 0..ii, the
  /// last the idle position's): how many of the set's positions are before
  /// it (`ii`: all of them), as a counter beside each position register,
  /// stepped as the position is. `inSet(pos)` builds the set's one-bit table
  /// at a position value (a slot table). `key` names the set (the same key,
  /// the same counters).
  mlir::Value rank(mlir::ImplicitLocOpBuilder &b, mlir::Value v,
                   llvm::ArrayRef<uint64_t> set, uint64_t key,
                   llvm::function_ref<mlir::Value(mlir::Value)> inSet);

  /// A position table's value at `v` (`rows`, rows 0..ii, the last the idle
  /// position's; `width` bits) from a register beside each position
  /// register, loaded a host cycle ahead (`slot-ahead`): the register's
  /// next is the table at the position's next, which is the position itself
  /// (no advance: the register holds), the position + 1 (a table of the
  /// rows shifted by one, decoded at the position), or at a frame's end 0
  /// or `ii` (constants, by `go`). A reader then takes the table's value
  /// from a register instead of through the position's decoder in its own
  /// host cycle: the same values, no host cycle moves. `decode(pos, rows,
  /// care)` builds a decoder of `rows` at a position value, read only where
  /// `care` (the shifted rows at a frame's end are never read). `care`
  /// (empty: every row): the rows of `rows` a reader takes (a valid bit
  /// beside the table low elsewhere). `key` names the table (the same key,
  /// the same registers).
  mlir::Value ahead(
      mlir::ImplicitLocOpBuilder &b, mlir::Value v, llvm::ArrayRef<uint64_t> rows,
      unsigned width, uint64_t key,
      llvm::function_ref<mlir::Value(mlir::Value, llvm::ArrayRef<uint64_t>,
                                     llvm::ArrayRef<uint8_t>)>
          decode,
      llvm::ArrayRef<uint8_t> care = {});
  /// Registers loaded so far by ahead().
  unsigned aheadRegisters() const { return aheadCount; }

  /// Whether two position values are equal in every host cycle: the same
  /// value, or the same kind of node of equal positions (registers taken as
  /// equal while their next values are compared: all reset to `ii`). The
  /// sequencers of a nested fold are such copies of the decision's register.
  bool same(mlir::Value x, mlir::Value y);

private:
  enum Kind { Reg, Next, NextFree, Shown };
  struct Node {
    Kind kind;
    mlir::Value a, bubble;
  };
  bool on = false;
  unsigned ii = 0, posWidth = 0;
  mlir::Value clock, reset, go, advance;
  llvm::DenseMap<mlir::Value, Node> known;
  std::map<std::pair<void *, unsigned>, std::pair<mlir::Value, mlir::Value>> cache;
  std::map<std::pair<void *, uint64_t>, mlir::Value> rankCache;
  std::map<std::pair<void *, uint64_t>, mlir::Value> aheadCache;
  unsigned aheadCount = 0;
  bool sameAssuming(mlir::Value x, mlir::Value y,
                    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::Value>> &assumed);
};

/// The simulator's position counters while it is built (null otherwise):
/// StaticFold's slot tables read them.
PosCounters *&currentPosCounters();

} // namespace foldsim

#endif // FOLDSIM_POSCOUNTERS_H
