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
  void end() { known.clear(); cache.clear(); on = false; }
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
};

/// The simulator's position counters while it is built (null otherwise):
/// StaticFold's slot tables read them.
PosCounters *&currentPosCounters();

} // namespace foldsim

#endif // FOLDSIM_POSCOUNTERS_H
