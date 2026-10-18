//===- BlackBoxes.cpp - make black-box modules external -------------------===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pipelines.h"
#include "foldsim/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"

namespace foldsim {
#define GEN_PASS_DEF_BLACKBOXES
#include "foldsim/Passes.h.inc"
} // namespace foldsim

using namespace mlir;
using namespace circt;

namespace {
struct BlackBoxesPass : public foldsim::impl::BlackBoxesBase<BlackBoxesPass> {
  void runOnOperation() override {
    for (auto module :
         llvm::make_early_inc_range(getOperation().getOps<hw::HWModuleOp>())) {
      if (!foldsim::isBlackBox(module))
        continue;
      if (failed(foldsim::getPipelineContract(module)))
        return signalPassFailure();
      OpBuilder builder(module);
      auto ext = hw::HWModuleExternOp::create(
          builder, module.getLoc(), module.getNameAttr(), module.getPortList());
      for (auto attr : module->getDiscardableAttrs())
        ext->setAttr(attr.getName(), attr.getValue());
      ext.setPrivate();
      module.erase();
    }
  }
};
} // namespace
