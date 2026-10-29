//===- Passes.h - foldsim passes ---------------------------------*- C++ -*-===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_PASSES_H
#define FOLDSIM_PASSES_H

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace foldsim {

/// The attribute `foldsim-fold-static` puts on a register it makes as a copy
/// of a slot's controls, which `foldsim-replicate-controls` may replicate and
/// removes.
inline constexpr llvm::StringLiteral kControlAttrName = "foldsim.control";

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "foldsim/Passes.h.inc"

} // namespace foldsim

#endif // FOLDSIM_PASSES_H
