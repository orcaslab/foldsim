//===- MemoryLoads.h - The pins a memory port's control bits drive --------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One model of how many pins a bit of a memory port's address, enable or mask
// drives once synthesis maps the memory, shared by foldsim-replicate-controls
// (which bounds those nets by replicas) and foldsim-estimate-timing (which
// reports a path's largest fanout for Beacon's fanout term).
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_MEMORYLOADS_H
#define FOLDSIM_MEMORYLOADS_H

#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "llvm/Support/MathExtras.h"

namespace foldsim {

/// A memory this shallow is LUTs or flops whatever its read latency.
constexpr uint64_t kLutMemoryDepth = 64;

/// The RAM style a memory states to synthesis, if any (`ram_style`).
inline llvm::StringRef ramStyleOf(mlir::Operation *op) {
  if (auto attrs = circt::sv::getSVAttributes(op))
    for (auto a : attrs)
      if (auto sva = llvm::dyn_cast<circt::sv::SVAttributeAttr>(a);
          sva && sva.getName().getValue() == "ram_style" && sva.getExpression())
        return sva.getExpression().getValue().trim('"');
  return {};
}

/// A memory synthesis builds of LUTs (distributed RAM) or flops: one stated
/// so, or of at most 64 entries, or of any depth read asynchronously (read
/// latency 0: a block RAM's read is registered, so such a memory is LUTs
/// however deep; firtool states it `ram_style = "distributed"`). Any other is
/// block RAM.
inline bool isLutMemory(circt::seq::FirMemOp op) {
  llvm::StringRef style = ramStyleOf(op);
  if (style == "block" || style == "ultra")
    return false;
  return style == "distributed" || style == "registers" ||
         op.getType().getDepth() <= kLutMemoryDepth || op.getReadLatency() == 0;
}

/// The pins one bit of a port's address, enable or mask drives, for `width`
/// bits of the memory: a LUT memory's width times its depth in 32-entry LUTs,
/// a block RAM's a pin per 36 bits and 1024 entries.
inline uint64_t memoryLoads(circt::seq::FirMemType type, bool lut,
                            uint64_t width) {
  uint64_t depth = type.getDepth();
  if (lut)
    return width * llvm::divideCeil(depth, 32);
  return llvm::divideCeil(width, 36) * llvm::divideCeil(depth, 1024);
}

/// The bits of a memory one mask bit writes.
inline uint64_t maskGrain(circt::seq::FirMemType type) {
  auto maskWidth = type.getMaskWidth();
  return maskWidth && *maskWidth ? type.getWidth() / *maskWidth : 1;
}

} // namespace foldsim

#endif // FOLDSIM_MEMORYLOADS_H
