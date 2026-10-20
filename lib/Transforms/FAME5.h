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
