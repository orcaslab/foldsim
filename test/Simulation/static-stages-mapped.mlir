// Staged evaluation sized by the mapped depth (foldsim-fold-static
// `stage-levels-mapped`): a 32-deep priority mux chain (a register's next
// value) counts a half level a mux in the estimate, 16 levels, so
// stage-levels 6 cuts it into 2 stages and 2 into 8; mapped to the V80's
// cells (beacon-map's balancing and LUT6 packing) it is 4 levels
// deep: no stage at 6 levels. At 2, each count's cut is mapped and timed
// with registers at its boundaries: 1 stage leaves 4 levels, 3 leave 3, 5
// leave 2: 5 stages, not 8. A memory a copy keeps a row of per instance
// times that many rows deep: one copy of 4 instances reads a 64-word
// memory as 256 words, 2 levels more than 4 copies of one (no count below
// the estimate's 2 to time: the estimate's). The model is timed as Beacon
// screens the copy: adders by carry block, and each register's next value
// through the select that holds it between its instance's passes (one level
// on a full cone): the copies' own Beacon depths, 6 and 4. An unknown
// target is an error.
// RUN: foldsim-opt %S/Inputs/stage_mux_chain.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=6})" -o %t.est.mlir 2>&1 | FileCheck %s --check-prefix=EST
// RUN: FileCheck %s --check-prefix=EST-IR < %t.est.mlir
// RUN: foldsim-opt %S/Inputs/stage_mux_chain.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=6 stage-levels-mapped=versal})" -o %t.map.mlir 2>&1 | FileCheck %s --check-prefix=MAP
// RUN: FileCheck %s --check-prefix=MAP-IR < %t.map.mlir
// RUN: foldsim-opt %S/Inputs/stage_mux_chain.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=2 stage-levels-mapped=versal})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=MAP2
// RUN: foldsim-opt %S/Inputs/stage_mem_read.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=1 stage-levels-mapped=versal})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=MEM1
// RUN: foldsim-opt %S/Inputs/stage_mem_read.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top copies=Leaf:4 stage-levels=1 stage-levels-mapped=versal})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=MEM4
// RUN: foldsim-opt %S/Inputs/stage_mem_read.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=0})" -o %t.mem1u.mlir
// RUN: foldsim-opt %t.mem1u.mlir --pass-pipeline='builtin.module(beacon-map{carry-block-size=8},beacon-estimate-timing{output= report=- top=Leaf_static})' -o /dev/null | FileCheck %s --check-prefix=MEM1COPY
// RUN: foldsim-opt %S/Inputs/stage_mem_read.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top copies=Leaf:4 stage-levels=0})" -o %t.mem4u.mlir
// RUN: foldsim-opt %t.mem4u.mlir --pass-pipeline='builtin.module(beacon-map{carry-block-size=8},beacon-estimate-timing{output= report=- top=Leaf_static})' -o /dev/null | FileCheck %s --check-prefix=MEM4COPY
// RUN: not foldsim-opt %S/Inputs/stage_mux_chain.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Leaf netlist},foldsim-fold-static{module=Top stage-levels=6 stage-levels-mapped=nope})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD

// EST: foldsim-fold-static: Leaf staged: 2 stages
// EST-IR: %stage1_
// MAP: foldsim-fold-static: Leaf: mapped depth 4 levels at 6 a stage: 0 stages (estimate 2)
// MAP-NOT: Leaf staged:
// MAP-IR-NOT: %stage1_
// MAP2: foldsim-fold-static: Leaf: mapped depth 4 levels at 2 a stage; 1 stages: 4; 3 stages: 3; 5 stages: 2: 5 stages (estimate 8)
// MAP2: foldsim-fold-static: Leaf staged: 5 stages, {{.*}} (soonest)
// MEM1: foldsim-fold-static: Leaf: mapped depth 6 levels at 1 a stage: 2 stages (no timed cut fits) (estimate 2)
// MEM1COPY: #1 levels 6: /{{.*}}.d[{{[0-9]+}}] (Leaf_static)
// MEM4: foldsim-fold-static: Leaf: mapped depth 4 levels at 1 a stage
// MEM4COPY: #1 levels 4: /{{.*}}.d[{{[0-9]+}}] (Leaf_static)
// BAD: error: no Beacon target 'nope'
