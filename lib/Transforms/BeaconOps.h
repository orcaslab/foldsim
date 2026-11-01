//===- BeaconOps.h - What Beacon's passes read off an op -------*- C++ -*-===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The widths foldsim-estimate-timing and foldsim-estimate-area read off a
// value, so the two passes agree on which products and dividers are tables.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_LIB_TRANSFORMS_BEACONOPS_H
#define FOLDSIM_LIB_TRANSFORMS_BEACONOPS_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"

namespace foldsim {
namespace beacon {

/// The most variable bits a product or a divider has for synthesis to write
/// it out as a table over them; also the dividend widths whose quotient and
/// remainder by a constant foldsim-lower-constant-division writes out.
constexpr unsigned kMaxTableBits = 10;

inline unsigned widthOf(mlir::Value value) {
  int64_t width = circt::hw::getBitWidth(value.getType());
  return width < 0 ? 0 : width;
}

/// The low bits of `value` that carry information: without the zeros of a
/// zero-extension or the copies of the sign bit of a sign-extension on top.
inline unsigned significantBits(mlir::Value value) {
  unsigned width = widthOf(value);
  auto concat = value.getDefiningOp<circt::comb::ConcatOp>();
  if (!concat || concat.getInputs().size() < 2)
    return width;
  mlir::Value top = concat.getInputs().front();
  if (auto c = top.getDefiningOp<circt::hw::ConstantOp>())
    return c.getValue().isZero() ? width - widthOf(top) : width;
  if (auto rep = top.getDefiningOp<circt::comb::ReplicateOp>())
    if (auto bit = rep.getInput().getDefiningOp<circt::comb::ExtractOp>())
      if (concat.getInputs().size() == 2 &&
          bit.getInput() == concat.getInputs()[1] &&
          bit.getLowBit() == widthOf(bit.getInput()) - 1)
        return width - widthOf(top);
  return width;
}

/// The bits of `value` a product or a divider depends on: none for a
/// constant.
inline unsigned variableBits(mlir::Value value) {
  return value.getDefiningOp<circt::hw::ConstantOp>() ? 0
                                                      : significantBits(value);
}

} // namespace beacon
} // namespace foldsim

#endif // FOLDSIM_LIB_TRANSFORMS_BEACONOPS_H
