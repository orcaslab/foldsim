// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Mac netlist" --foldsim-compose="module=Top" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/obj %S/Inputs/tb_threads.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// The multiply-accumulate's state computed from its inputs registered a host
// cycle late, its write two host cycles after the edge: the same tokens.
// RUN: mkdir -p %t/m && foldsim-opt %s --foldsim-extract-models="module=Top models=Mac netlist" --foldsim-compose="module=Top region-loads=1 registered-state state-stages=2 pipeline-multipliers" -o %t/m/sim.mlir
// RUN: FileCheck %s --check-prefix=CONE < %t/m/sim.mlir
// RUN: firtool --format=mlir %t/m/sim.mlir -o %t/m/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/m/obj %S/Inputs/tb_threads.sv %t/m/sim.sv > %t/m/build.log 2>&1
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s

// Four threads of r <- ((r + a) * 3) ^ a, reset in their first cycle, as the
// unthreaded target computes them.
// CHECK: y0: 00 04 16 4a e8
// CHECK: y1: 00 04 16 4a e8
// CHECK: y2: 00 04 16 4a e8
// CHECK: y3: 00 04 16 4a e8
// The stage is where the multiplier starts: the sum it multiplies is
// computed on the host cycle before and registered, as are the values the
// logic after it reads; constants are not.
// CONE-NOT: %r_wdata =
// CONE-NOT: seq.firreg %c
// CONE-DAG: [[SUM:%.+]] = comb.add %r_state, %a_sink_bits : i8
// CONE-DAG: %cone_in = seq.firreg [[SUM]] clock %hostClock : i8
// CONE-DAG: [[P:%.+]] = comb.mul %cone_in, %c3_i8 : i8
// CONE-DAG: %a_sink_bits_in = seq.firreg %a_sink_bits clock %hostClock : i8
// CONE-DAG: comb.xor [[P]], %a_sink_bits_in : i8
// CONE-DAG: %r_wdata2 = seq.firreg
hw.module private @Mac(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %c3 = hw.constant 3 : i8
  %q = comb.add %r, %a : i8
  %p = comb.mul %q, %c3 : i8
  %s = comb.xor %p, %a : i8
  %r = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y0 : i8, out y1 : i8, out y2 : i8, out y3 : i8) {
  %y0 = hw.instance "a0" @Mac(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y1 = hw.instance "a1" @Mac(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y2 = hw.instance "a2" @Mac(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %y3 = hw.instance "a3" @Mac(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  hw.output %y0, %y1, %y2, %y3 : i8, i8, i8, i8
}
