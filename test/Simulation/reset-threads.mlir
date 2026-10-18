// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Acc netlist" --foldsim-compose="module=Top" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/obj %S/Inputs/tb_threads.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// Regions of one memory each: every write lands a host cycle late, from
// staged registers, and the tokens are the same.
// RUN: mkdir -p %t/r && foldsim-opt %s --foldsim-extract-models="module=Top models=Acc netlist" --foldsim-compose="module=Top region-loads=1" -o %t/r/sim.mlir
// RUN: firtool --format=mlir %t/r/sim.mlir -o %t/r/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/r/obj %S/Inputs/tb_threads.sv %t/r/sim.sv > %t/r/build.log 2>&1
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// With the state read a host cycle ahead and committed by select, the
// tokens are the same too.
// RUN: mkdir -p %t/p && foldsim-opt %s --foldsim-extract-models="module=Top models=Acc netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select" -o %t/p/sim.mlir
// RUN: firtool --format=mlir %t/p/sim.mlir -o %t/p/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/p/obj %S/Inputs/tb_threads.sv %t/p/sim.sv > %t/p/build.log 2>&1
// RUN: %t/p/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/p/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: %t/p/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// With the state written two host cycles after the edge, too.
// RUN: mkdir -p %t/s && foldsim-opt %s --foldsim-extract-models="module=Top models=Acc netlist" --foldsim-compose="module=Top region-loads=1 registered-state state-stages=2" -o %t/s/sim.mlir
// RUN: firtool --format=mlir %t/s/sim.mlir -o %t/s/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/s/obj %S/Inputs/tb_threads.sv %t/s/sim.sv > %t/s/build.log 2>&1
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// As a tree of domains, with regions, the tokens are the same too.
// RUN: mkdir -p %t/d && foldsim-opt %s --foldsim-extract-models="module=Top models=Acc netlist" --foldsim-compose="module=Top region-loads=1 domains" -o %t/d/sim.mlir
// RUN: firtool --format=mlir %t/d/sim.mlir -o %t/d/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/d/obj %S/Inputs/tb_threads.sv %t/d/sim.sv > %t/d/build.log 2>&1
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s

// And with the state written two host cycles late and the staged tokens.
// RUN: mkdir -p %t/f && foldsim-opt %s --foldsim-extract-models="module=Top models=Acc netlist" --foldsim-compose="module=Top region-loads=1 registered-state state-stages=2 staged-tokens" -o %t/f/sim.mlir
// RUN: firtool --format=mlir %t/f/sim.mlir -o %t/f/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/f/obj %S/Inputs/tb_threads.sv %t/f/sim.sv > %t/f/build.log 2>&1
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// Four threads, whose state starts out random, initialized by a reset of one
// host cycle: each takes every token of `reset` and `x`, and gives every one
// of its `y`.
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
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y0 : i8, out y1 : i8, out y2 : i8, out y3 : i8) {
  %y0 = hw.instance "a0" @Acc(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y1 = hw.instance "a1" @Acc(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y2 = hw.instance "a2" @Acc(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y3 = hw.instance "a3" @Acc(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  hw.output %y0, %y1, %y2, %y3 : i8, i8, i8, i8
}
