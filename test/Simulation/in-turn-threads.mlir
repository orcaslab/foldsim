// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// Picked in turn, a thread runs once every as many host cycles as there are
// threads, and its state is read a host cycle ahead: the write of its last
// edge is in the memory by then unless there are fewer threads than host
// cycles to that write and one more. Three accumulators of one model and one
// of another: with one stage to the write nothing is forwarded, with two the
// write landing on the cycle the thread is read ahead is.
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Acc,Acc1 netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/obj %S/Inputs/tb_threads.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/s && foldsim-opt %s --foldsim-extract-models="module=Top models=Acc,Acc1 netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 staged-tokens" -o %t/s/sim.mlir
// RUN: firtool --format=mlir %t/s/sim.mlir -o %t/s/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/s/obj %S/Inputs/tb_threads.sv %t/s/sim.sv > %t/s/build.log 2>&1
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// Two models of two threads each: the other thread runs between, and the
// write landing on the cycle a thread is read ahead is forwarded.
// RUN: sed -e 's/"a2" @Acc(/"a2" @Acc1(/' %s > %t/two.mlir
// RUN: mkdir -p %t/t && foldsim-opt %t/two.mlir --foldsim-extract-models="module=Top models=Acc,Acc1 netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select" -o %t/t/sim.mlir
// RUN: firtool --format=mlir %t/t/sim.mlir -o %t/t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/t/obj %S/Inputs/tb_threads.sv %t/t/sim.sv > %t/t/build.log 2>&1
// RUN: %t/t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// With idle host cycles in each thread's turn, nothing is forwarded either
// way, three threads with two stages or two threads.
// RUN: mkdir -p %t/u && foldsim-opt %s --foldsim-extract-models="module=Top models=Acc,Acc1 netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 staged-tokens idle-slots" -o %t/u/sim.mlir
// RUN: firtool --format=mlir %t/u/sim.mlir -o %t/u/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/u/obj %S/Inputs/tb_threads.sv %t/u/sim.sv > %t/u/build.log 2>&1
// RUN: %t/u/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/u/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/v && foldsim-opt %t/two.mlir --foldsim-extract-models="module=Top models=Acc,Acc1 netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select idle-slots" -o %t/v/sim.mlir
// RUN: firtool --format=mlir %t/v/sim.mlir -o %t/v/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/v/obj %S/Inputs/tb_threads.sv %t/v/sim.sv > %t/v/build.log 2>&1
// RUN: %t/v/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/v/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/w && foldsim-opt %t/two.mlir --foldsim-extract-models="module=Top models=Acc,Acc1 netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 staged-tokens idle-slots" -o %t/w/sim.mlir
// RUN: firtool --format=mlir %t/w/sim.mlir -o %t/w/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/w/obj %S/Inputs/tb_threads.sv %t/w/sim.sv > %t/w/build.log 2>&1
// RUN: %t/w/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/w/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// CHECK: y0: 00 02 05 09 0e
// CHECK: y1: 00 02 05 09 0e
// CHECK: y2: 00 02 05 09 0e
// CHECK: y3: 00 02 05 09 0e
hw.module private @Acc(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %s = comb.add %r, %a : i8
  %r = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module private @Acc1(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %s = comb.add %r, %a : i8
  %r = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y0 : i8, out y1 : i8, out y2 : i8, out y3 : i8) {
  %y0 = hw.instance "a0" @Acc(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y1 = hw.instance "a1" @Acc(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y2 = hw.instance "a2" @Acc(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y3 = hw.instance "a3" @Acc1(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  hw.output %y0, %y1, %y2, %y3 : i8, i8, i8, i8
}
