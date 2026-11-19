//===- FAME5.h - FAME-5 multithreading of a flat model ----*- C++ -*-===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_LIB_TRANSFORMS_FAME5_H
#define FOLDSIM_LIB_TRANSFORMS_FAME5_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace foldsim {

/// Replaces `model` with its FAME-5 threaded module `name` of `threads`
/// instances, as `foldsim-fame5` with `clock-enable` and `thread-index` builds
/// it but for one thing: a thread's target clock ticks from its first finish
/// after reset on, rather than from its second. Each target port is a channel,
/// the current thread on `threadIdx`. With `readySchedule`, a `thread_ready`
/// input of one bit per thread says whose inputs are all there, and the thread
/// after the current one among those is next. `options` are `foldsim-fame5`'s
/// options of the same names.
struct ThreadOptions {
  unsigned regionLoads = 0;
  bool registeredState = false;
  unsigned stateStages = 1;
  bool commitSelect = false;
  bool stagedRamWrites = false;
  bool pipelineMultipliers = false;
  bool compactStaging = false;
  bool foldPipelines = false;
  bool idleSlots = false;
  unsigned pipelineDepth = 0;
  /// With registered state: thread state of at least this many threads asks
  /// for block RAM, its read-ahead register the RAM's output (0: never).
  unsigned blockStateDepth = 0;
  /// With registered state and commit by select: a model of one thread takes
  /// every other host cycle and commits its state by the registered edge as
  /// a clock enable, rather than by a select per bit.
  bool commitEnable = false;
  /// With regions: a memory of more than this many loads (bits, times its
  /// depth in 32-entry LUTs) is split by bits into memories of at most that
  /// many, each in a region of its own (0: never).
  unsigned memoryLoads = 0;
  /// The model's turns come from outside rather than from a counter of its
  /// own: its position (thread and idle phase, `thread * phases + phase`)
  /// for the host cycle after next in on `schedule_next`, the register
  /// holding the next host cycle's (or, without registered state, this
  /// one's) out on `schedule`; `scheduleStart` is the position on the host
  /// cycle after reset.
  bool externalSchedule = false;
  /// The model counts its turns itself, in turn, from `scheduleStart`: on a
  /// fixed schedule the same positions outside turns would bring, with
  /// nothing from outside feeding them.
  bool localSchedule = false;
  uint64_t scheduleStart = 0;
  /// With `clock-enable`: the estimated logic levels from which a register-
  /// only part of a registered read's address or enable is computed a host
  /// cycle ahead and registered (0: never).
  unsigned readPortDepth = 0;
  /// Memories addressed by `x` rather than `~x` where every port takes
  /// `~x` (or a constant): rows reversed, which no port can tell.
  bool unreverseAddresses = false;
};
/// How a register is written: never (it only holds its value), or with
/// `data` when `enable` (always, if null).
struct HeldWrite {
  bool holds;
  mlir::Value enable, data;
};
/// The write of a register whose next value is `next`: the branches of its
/// muxes that hold the register are the cycles it is not written.
HeldWrite peelHolds(mlir::ImplicitLocOpBuilder &b, mlir::Value next,
                    mlir::Value reg);

/// The host cycles of a thread's turn in a model picked in turn with idle
/// slots: the first its own, the others idle (`foldsim-fame5`'s
/// `idle-slots`).
unsigned turnPhases(unsigned threads, unsigned stateStages);

mlir::FailureOr<circt::hw::HWModuleOp>
threadModel(circt::hw::HWModuleOp model, unsigned threads, mlir::StringAttr name,
            bool readySchedule, const ThreadOptions &options = {});

} // namespace foldsim

#endif // FOLDSIM_LIB_TRANSFORMS_FAME5_H
