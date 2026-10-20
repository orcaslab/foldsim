// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Deep netlist" --foldsim-compose="module=Top" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/obj %S/Inputs/tb_threads.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// The two states' cones of eight levels (with the reset's select) cut at
// level four, in the middle: no level leaves both host cycles three or fewer.
// Their shared logic is computed once on the second host cycle: the same
// tokens.
// RUN: mkdir -p %t/d && foldsim-opt %s --foldsim-extract-models="module=Top models=Deep netlist" --foldsim-compose="module=Top region-loads=1 registered-state state-stages=2 pipeline-depth=3" -o %t/d/sim.mlir
// RUN: FileCheck %s --check-prefix=CUT < %t/d/sim.mlir
// RUN: FileCheck %s --check-prefix=SHARE < %t/d/sim.mlir
// RUN: firtool --format=mlir %t/d/sim.mlir -o %t/d/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/d/obj %S/Inputs/tb_threads.sv %t/d/sim.sv > %t/d/build.log 2>&1
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// Allowed four levels a host cycle, the cut is at level four, the one level
// that leaves both host cycles four; in turn, with idle slots and the image
// flow's other state options.
// RUN: mkdir -p %t/f && foldsim-opt %s --foldsim-extract-models="module=Top models=Deep netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens idle-slots pipeline-depth=4" -o %t/f/sim.mlir
// RUN: firtool --format=mlir %t/f/sim.mlir -o %t/f/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/f/obj %S/Inputs/tb_threads.sv %t/f/sim.sv > %t/f/build.log 2>&1
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// A cone no deeper than the limit is registered whole, as without the option.
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Deep netlist" --foldsim-compose="module=Top region-loads=1 registered-state state-stages=2 pipeline-depth=8" | FileCheck %s --check-prefix=WHOLE

// Four threads of u = (((r + a) ^ s) + 5) ^ (r & a)) + s, r <- (u ^ a) + 3,
// s <- (u + r) ^ 1, reset in their first cycle, as the unthreaded target
// computes them.
// CHECK: y0: 00 08 1e 2c ce
// CHECK: y1: 00 08 1e 2c ce
// CHECK: y2: 00 08 1e 2c ce
// CHECK: y3: 00 08 1e 2c ce
// The first host cycle computes u's first four levels and registers them;
// the second computes the rest of u once, for both states.
// CUT-NOT: %r_wdata =
// CUT-NOT: %s_wdata =
// CUT-DAG: %cone_in{{.*}} = seq.firreg {{%.+}} clock %hostClock : i8
// CUT-DAG: %r_wdata2 = seq.firreg
// CUT-DAG: %s_wdata2 = seq.firreg
// SHARE: [[U:%.+]] = comb.add %cone_in, %s{{.*}}_in : i8
// SHARE-NOT: comb.add %cone_in
// SHARE: comb.add [[U]], %r_state_in : i8
// WHOLE-DAG: %r_wdata = seq.firreg
// WHOLE-DAG: %s_wdata = seq.firreg
// WHOLE-NOT: cone_in
hw.module private @Deep(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %c1 = hw.constant 1 : i8
  %c3 = hw.constant 3 : i8
  %c5 = hw.constant 5 : i8
  %t1 = comb.add %r, %a : i8
  %t2 = comb.xor %t1, %s : i8
  %t3 = comb.add %t2, %c5 : i8
  %t4 = comb.and %r, %a : i8
  %t5 = comb.xor %t3, %t4 : i8
  %u = comb.add %t5, %s : i8
  %v = comb.xor %u, %a : i8
  %rn = comb.add %v, %c3 : i8
  %w = comb.add %u, %r : i8
  %sn = comb.xor %w, %c1 : i8
  %r = seq.firreg %rn clock %clk reset sync %reset, %c0 : i8
  %s = seq.firreg %sn clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y0 : i8, out y1 : i8, out y2 : i8, out y3 : i8) {
  %y0 = hw.instance "a0" @Deep(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y1 = hw.instance "a1" @Deep(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y2 = hw.instance "a2" @Deep(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y3 = hw.instance "a3" @Deep(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  hw.output %y0, %y1, %y2, %y3 : i8, i8, i8, i8
}
