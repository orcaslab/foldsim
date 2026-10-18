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

// Four threads of r <- r * 3 + a, reset in their first cycle.
// CHECK: y0: [[Y:.*]]
// CHECK: y1: [[Y]]
// CHECK: y2: [[Y]]
// CHECK: y3: [[Y]]
// The product's operands are registered, not the product.
// CONE-NOT: %r_wdata =
// CONE-DAG: %r_in = seq.firreg %r_state clock %hostClock : i8
// CONE-DAG: [[P:%.+]] = comb.mul %r_in, {{%.+}} : i8
// CONE-DAG: %r_wdata2 = seq.firreg
hw.module private @Mac(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %c3 = hw.constant 3 : i8
  %p = comb.mul %r, %c3 : i8
  %s = comb.add %p, %a : i8
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
