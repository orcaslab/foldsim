// REQUIRES: verilator
// The tightest schedule folding allows: two threads, and a pipeline of four
// layers, the last one kept, so three slots: a result is readable six host
// cycles after its issue (registered issue, three layers, registered write),
// and read at the thread's edge three enabled edges later, at the earliest six
// host cycles later too, one host cycle between a thread's edges.
// RUN: rm -rf %t && mkdir -p %t && cat %S/Inputs/fold_model.mlir %s > %t/in.mlir
// RUN: firtool --format=mlir %t/in.mlir -o %t/target.sv
// RUN: foldsim-opt %t/in.mlir --foldsim-extract-models="module=Top models=Model netlist" --foldsim-compose="module=Top fold-pipelines" -o %t/sim.mlir
// RUN: FileCheck %s --check-prefix=FOLD < %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique +define+N=2 +define+N2 --top-module tb -Mdir %t/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+4 | FileCheck %s

// CHECK: MATCH 2 outputs

// Three slots per thread; the write five host cycles after issue.
// FOLD-DAG: %fold0_store = seq.firmem 0, 1, undefined, undefined : <6 x 16, mask 1>
// FOLD-DAG: seq.firmem.write_port %fold0_store[{{%.+}}] = %fold0_wdata, clock %hostClock enable %fold0_valid5

hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %a : i8, in %en : i2, in %v : i2, out y0 : i32, out y1 : i32) {
  %c1 = hw.constant 1 : i8
  %e0 = comb.extract %en from 0 : (i2) -> i1
  %e1 = comb.extract %en from 1 : (i2) -> i1
  %v0 = comb.extract %v from 0 : (i2) -> i1
  %v1 = comb.extract %v from 1 : (i2) -> i1
  %a1 = comb.xor %a, %c1 : i8
  %y0 = hw.instance "m0" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a: i8, en: %e0: i1, v: %v0: i1) -> (y: i32)
  %y1 = hw.instance "m1" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a1: i8, en: %e1: i1, v: %v1: i1) -> (y: i32)
  hw.output %y0, %y1 : i32, i32
}
