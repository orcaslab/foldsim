// Staged evaluation sized by the mapped depth (foldsim-fold-static
// `stage-levels-mapped`): a 32-deep priority mux chain (a register's next
// value) counts a half level a mux in the estimate, 16 levels, so
// stage-levels 6 cuts it into 2 stages and 2 into 8; mapped to the V80's
// cells (foldsim-beacon-map's balancing and LUT6 packing) it is 4 levels
// deep: no stage at 6 levels, one at 2. An unknown target is an error.
// RUN: foldsim-opt %S/Inputs/stage_mux_chain.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=6})" -o %t.est.mlir 2>&1 | FileCheck %s --check-prefix=EST
// RUN: FileCheck %s --check-prefix=EST-IR < %t.est.mlir
// RUN: foldsim-opt %S/Inputs/stage_mux_chain.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=6 stage-levels-mapped=versal})" -o %t.map.mlir 2>&1 | FileCheck %s --check-prefix=MAP
// RUN: FileCheck %s --check-prefix=MAP-IR < %t.map.mlir
// RUN: foldsim-opt %S/Inputs/stage_mux_chain.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=2 stage-levels-mapped=versal})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=MAP2
// RUN: not foldsim-opt %S/Inputs/stage_mux_chain.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=6 stage-levels-mapped=nope})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD

// EST: foldsim-fold-static: Leaf staged: 2 stages
// EST-IR: %stage1_
// MAP: foldsim-fold-static: Leaf: mapped depth 4 levels at 6 a stage: 0 stages, not the estimate's 2
// MAP-NOT: Leaf staged:
// MAP-IR-NOT: %stage1_
// MAP2: foldsim-fold-static: Leaf: mapped depth 4 levels at 2 a stage: 1 stages, not the estimate's 8
// MAP2: foldsim-fold-static: Leaf staged: 1 stages
// BAD: error: no Beacon target 'nope'
