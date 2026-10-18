// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t && cat %S/Inputs/fold_model.mlir %s > %t/in.mlir
// RUN: firtool --format=mlir %t/in.mlir -o %t/target.sv
// The pipelines folded: their state a slot counter and a store per thread,
// written by one unthreaded copy and the black box itself.
// RUN: foldsim-opt %t/in.mlir --foldsim-extract-models="module=Top models=Model netlist" --foldsim-compose="module=Top fold-pipelines" -o %t/sim.mlir
// RUN: FileCheck %s --check-prefix=FOLD < %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// The same with the image flow's state options: read ahead, written two
// host cycles late, regions, staged memory writes and tokens.
// RUN: mkdir -p %t/r && foldsim-opt %t/in.mlir --foldsim-extract-models="module=Top models=Model netlist" --foldsim-compose="module=Top region-loads=16 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens fold-pipelines" -o %t/r/sim.mlir
// RUN: firtool --format=mlir %t/r/sim.mlir -o %t/r/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/r/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/r/sim.sv > %t/r/build.log 2>&1
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+4 | FileCheck %s
// The same, each model scheduled in turn rather than by readiness.
// RUN: mkdir -p %t/i && foldsim-opt %t/in.mlir --foldsim-extract-models="module=Top models=Model netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=16 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens fold-pipelines" -o %t/i/sim.mlir
// RUN: firtool --format=mlir %t/i/sim.mlir -o %t/i/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/i/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/i/sim.sv > %t/i/build.log 2>&1
// RUN: %t/i/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/i/obj/Vtb +verilator+rand+reset+2 +verilator+seed+5 | FileCheck %s
// The whole target one model of one thread: the black box stays an instance,
// on the host clock and enabled on the target's edges.
// RUN: mkdir -p %t/u && foldsim-opt %t/in.mlir --hw-flatten-modules --foldsim-extract-models="module=Top netlist" --foldsim-compose="module=Top" -o %t/u/sim.mlir
// RUN: FileCheck %s --check-prefix=ONE < %t/u/sim.mlir
// RUN: firtool --format=mlir %t/u/sim.mlir -o %t/u/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/u/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/u/sim.sv > %t/u/build.log 2>&1
// RUN: %t/u/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// Without folding, threaded: an error.
// RUN: mkdir -p %t/n && not foldsim-opt %t/in.mlir --foldsim-extract-models="module=Top models=Model netlist" --foldsim-compose="module=Top region-loads=16 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens" -o %t/n/sim.mlir 2>&1 | FileCheck %s --check-prefix=NOFOLD

// CHECK: MATCH 4 outputs

// FOLD-DAG: %fold0_store = seq.firmem 0, 1, undefined, undefined : <12 x 16, mask 1>
// FOLD-DAG: %fold1_store = seq.firmem 0, 1, undefined, undefined : <16 x 8, mask 1>
// FOLD-DAG: seq.firmem.write_port %fold0_store[{{%.+}}] = %fold0_wdata, clock %hostClock enable %fold0_valid5
// FOLD-DAG: seq.firmem.write_port %fold1_store[{{%.+}}] = %fold1_wdata, clock %hostClock enable %fold1_valid6
// FOLD-DAG: hw.instance "box_fold" @XBox
// FOLD-DAG: %s1_fold = seq.firreg {{%.+}} clock %hostClock : i8
// FOLD-DAG: %fold0_issue = seq.firreg %a_sink_bits clock %hostClock : i8
// FOLD-DAG: sv.fatal.procedural

// ONE: hw.instance "m0/box" @XBox(clk: %hostClock: !seq.clock, ce: [[CE:%[0-9]+]]: i1
// ONE: [[CE]] = comb.and {{%.+}}, {{%.+}} : i1

// A black box is only simulated folded.
// NOFOLD: error: black box 'XBox' in a threaded model: it is simulated only as a folded pipeline (fold-pipelines)

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
