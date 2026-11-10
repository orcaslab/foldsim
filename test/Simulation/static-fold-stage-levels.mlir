// REQUIRES: verilator
// Folded pipelines in stages (foldsim-fold-static `fold-stage-levels`): each
// layer of a folded pipeline's copy cut by host-clock registers so that no
// stage spans more than N estimated levels; each layer takes its deepest
// register's stages (the layers a copy register holds bits of, alike) and the
// results land that much later in the store, which the copy checks against
// the schedule. On fold_model_deep.mlir (fold_model.mlir with chains of ops
// in three layers). The statically folded simulator against the target, its
// channels stalled at random: without the option (no stage registers), at
// one level (two of fold0's layers a stage register more, its write data
// one: results 3 host cycles later), with the store read ahead in
// block RAM (fold-store-ram: its lead from the later landing), and on two
// copies.
// RUN: rm -rf %t && mkdir -p %t && cat %S/Inputs/fold_model_deep.mlir %s > %t/in.mlir
// RUN: firtool --format=mlir %t/in.mlir -o %t/target.sv
// RUN: mkdir -p %t/off && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top})" -o %t/off/sim.mlir 2> %t/off/fold.log
// RUN: not grep -e fold-stage-levels -e "fold[0-9]*_stage " %t/off/fold.log %t/off/sim.mlir
// RUN: firtool --format=mlir %t/off/sim.mlir -o %t/off/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/off/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/off/sim.sv > %t/off/build.log 2>&1
// RUN: %t/off/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: mkdir -p %t/s1 && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top fold-stage-levels=1})" -o %t/s1/sim.mlir 2> %t/s1/fold.log
// RUN: FileCheck %s --check-prefix=S1 < %t/s1/sim.mlir
// RUN: FileCheck %s --check-prefix=S1LOG < %t/s1/fold.log
// RUN: firtool --format=mlir %t/s1/sim.mlir -o %t/s1/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/s1/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/s1/sim.sv > %t/s1/build.log 2>&1
// RUN: %t/s1/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/s1/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: %t/s1/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// RUN: mkdir -p %t/ra && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top fold-stage-levels=1 fold-store-ram=1})" -o %t/ra/sim.mlir 2> %t/ra/fold.log
// RUN: FileCheck %s --check-prefix=S1LOG < %t/ra/fold.log
// RUN: firtool --format=mlir %t/ra/sim.mlir -o %t/ra/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/ra/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/ra/sim.sv > %t/ra/build.log 2>&1
// RUN: %t/ra/obj/Vtb +verilator+rand+reset+2 +verilator+seed+4 | FileCheck %s
// RUN: %t/ra/obj/Vtb +verilator+rand+reset+2 +verilator+seed+5 | FileCheck %s
// RUN: mkdir -p %t/c2 && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top copies=Model:2 fold-stage-levels=1})" -o %t/c2/sim.mlir 2> %t/c2/fold.log
// RUN: firtool --format=mlir %t/c2/sim.mlir -o %t/c2/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c2/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/c2/sim.sv > %t/c2/build.log 2>&1
// RUN: %t/c2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+6 | FileCheck %s
// RUN: %t/c2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+7 | FileCheck %s

// Per class (fold-stage-levels-class): Model's at one level alone (as above), off over a global
// one (none), and a spec of no class.
// RUN: foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top fold-stage-levels-class=Model:1})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=S1LOG
// RUN: foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top fold-stage-levels=1 fold-stage-levels-class=Mod*:off})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=KOFF
// RUN: not foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top fold-stage-levels-class=Nope:1})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=KNONE
// Staged evaluation (stage-levels) of a class it cannot stage says why (its
// logic keeps its depth): every reason, not the last one found.
// RUN: foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top stage-levels=4})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOSTAGE
// KOFF-NOT: fold-stage-levels:
// KNONE: fold-stage-levels-class: 'Nope:1' matches no class
// NOSTAGE: foldsim-fold-static: Model not staged at 4 levels: folded pipelines, black boxes

// CHECK: {{^}}MATCH 4 outputs x 88 cycles

// Stage registers on the host clock in fold0's copy, the tags waiting 3 host
// cycles more (valid5 -> valid8: outAt 4 + 3, the write at outAt + 1).
// S1: %fold0_stage = seq.firreg
// S1: seq.firmem.write_port %fold0_store[{{%.+}}] = %fold0_wdata, clock %hostClock enable %fold0_valid8
// S1LOG: fold-stage-levels: Model fold0: 4 layers, 2 stage registers on their way, write data 1 more: results 3 host cycles later

hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %a : i8, in %en : i4, in %v : i4, out y0 : i32, out y1 : i32, out y2 : i32, out y3 : i32) {
  %c1 = hw.constant 1 : i8
  %c2 = hw.constant 2 : i8
  %c3 = hw.constant 3 : i8
  %e0 = comb.extract %en from 0 : (i4) -> i1
  %e1 = comb.extract %en from 1 : (i4) -> i1
  %e2 = comb.extract %en from 2 : (i4) -> i1
  %e3 = comb.extract %en from 3 : (i4) -> i1
  %v0 = comb.extract %v from 0 : (i4) -> i1
  %v1 = comb.extract %v from 1 : (i4) -> i1
  %v2 = comb.extract %v from 2 : (i4) -> i1
  %v3 = comb.extract %v from 3 : (i4) -> i1
  %a1 = comb.xor %a, %c1 : i8
  %a2 = comb.xor %a, %c2 : i8
  %a3 = comb.xor %a, %c3 : i8
  %y0 = hw.instance "m0" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a: i8, en: %e0: i1, v: %v0: i1) -> (y: i32)
  %y1 = hw.instance "m1" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a1: i8, en: %e1: i1, v: %v1: i1) -> (y: i32)
  %y2 = hw.instance "m2" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a2: i8, en: %e2: i1, v: %v2: i1) -> (y: i32)
  %y3 = hw.instance "m3" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a3: i8, en: %e3: i1, v: %v3: i1) -> (y: i32)
  hw.output %y0, %y1, %y2, %y3 : i32, i32, i32, i32
}
