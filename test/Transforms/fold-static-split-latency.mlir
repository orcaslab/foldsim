// RUN: foldsim-opt %S/fold-static-links.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top zero-latency-links})" 2>&1 | FileCheck %s --check-prefix=BASE
// RUN: foldsim-opt %S/fold-static-links.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top zero-latency-links split-latency-file=%S/Inputs/slr-split/queue.lags})" 2>&1 | FileCheck %s

// split-latency-file: one output of a class (Queue's output 0) 3 host cycles
// late (a copy split by foldsim-slr-split crossing=txrx, its lags file): its
// links are never taken in the producer's host cycle, every reader takes
// them 3 host cycles later (the frame one longer here), and their writes'
// enable, entry and bank registers come 3 host cycles later, in a link
// group of their own (`_late3_`); the class's other outputs keep theirs.
// BASE: foldsim-fold-static: II = 6
// CHECK: foldsim-fold-static: II = 7
// CHECK: %Queue_late3_links0_state1 = seq.firreg %Queue_state2
// CHECK: %Queue_late3_links0_state0 = seq.firreg %Queue_late3_links0_state1
// CHECK: %Queue_late3_links0_state_split1 = seq.firreg %Queue_late3_links0_state0
// CHECK: %Queue_late3_links0_state_split2 = seq.firreg %Queue_late3_links0_state_split1
// CHECK: %Queue_late3_links0_state_split3 = seq.firreg %Queue_late3_links0_state_split2
