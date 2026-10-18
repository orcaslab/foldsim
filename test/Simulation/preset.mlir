// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=P netlist" --foldsim-compose="module=Top" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/obj %S/Inputs/tb_preset.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// Regions of one memory each: every write lands a host cycle late, from
// staged registers, and the tokens are the same.
// RUN: mkdir -p %t/r && foldsim-opt %s --foldsim-extract-models="module=Top models=P netlist" --foldsim-compose="module=Top region-loads=1" -o %t/r/sim.mlir
// RUN: firtool --format=mlir %t/r/sim.mlir -o %t/r/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/r/obj %S/Inputs/tb_preset.sv %t/r/sim.sv > %t/r/build.log 2>&1
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// With the state read a host cycle ahead and committed by select, the
// tokens are the same too.
// RUN: mkdir -p %t/p && foldsim-opt %s --foldsim-extract-models="module=Top models=P netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select" -o %t/p/sim.mlir
// RUN: firtool --format=mlir %t/p/sim.mlir -o %t/p/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/p/obj %S/Inputs/tb_preset.sv %t/p/sim.sv > %t/p/build.log 2>&1
// RUN: %t/p/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/p/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// As a tree of domains, with regions, the tokens are the same too.
// RUN: mkdir -p %t/d && foldsim-opt %s --foldsim-extract-models="module=Top models=P netlist" --foldsim-compose="module=Top region-loads=1 domains" -o %t/d/sim.mlir
// RUN: firtool --format=mlir %t/d/sim.mlir -o %t/d/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/d/obj %S/Inputs/tb_preset.sv %t/d/sim.sv > %t/d/build.log 2>&1
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s

// Registers start out at their preset values: in each of `P`'s two threads,
// `r` at 9 and `f` at 1, though it is 0 from the first edge on, and in the
// one thread of the rest of `Top`, `t` at 5.
// CHECK: 0: 9 1 2 3
// CHECK: 1: 9 9 1 2
// CHECK: 2: 1 0 0 0
// CHECK: 3: 1 0 0 0
// CHECK: 4: 5 1 2 3
hw.module private @P(in %clk : !seq.clock, in %d : i4, out q : i4, out first : i1) {
  %false = hw.constant false
  %r = seq.firreg %d clock %clk preset 9 : i4
  %f = seq.firreg %false clock %clk preset 1 : i1
  hw.output %r, %f : i4, i1
}
hw.module @Top(in %clk : !seq.clock, in %x : i4, out q0 : i4, out q1 : i4, out f0 : i1, out f1 : i1, out t : i4) {
  %a.q, %a.first = hw.instance "a" @P(clk: %clk: !seq.clock, d: %x: i4) -> (q: i4, first: i1)
  %b.q, %b.first = hw.instance "b" @P(clk: %clk: !seq.clock, d: %a.q: i4) -> (q: i4, first: i1)
  %t = seq.firreg %x clock %clk preset 5 : i4
  hw.output %a.q, %b.q, %a.first, %b.first, %t : i4, i4, i1, i1, i4
}
