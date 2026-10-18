// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Cap netlist" --foldsim-compose="module=Top" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/obj %S/Inputs/tb_memory.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb | FileCheck %s
// Committed by select, a single thread's registers whose next value is
// another register or a memory's read data take the same values.
// RUN: mkdir -p %t/p && foldsim-opt %s --foldsim-extract-models="module=Top models=Cap netlist" --foldsim-compose="module=Top commit-select" -o %t/p/sim.mlir
// RUN: firtool --format=mlir %t/p/sim.mlir -o %t/p/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/p/obj %S/Inputs/tb_memory.sv %t/p/sim.sv > %t/p/build.log 2>&1
// RUN: %t/p/obj/Vtb | FileCheck %s

// `z` takes the read data, `a` takes `z` once `e` is set: the reads of
// `memory-one-thread.mlir`, two target cycles later.
// CHECK: r 4 = a1
// CHECK: r 5 = b2
hw.module private @Cap(in %clk : !seq.clock, in %we : i1, in %waddr : i3, in %wdata : i8, in %raddr : i3, out r : i8) {
  %true = hw.constant true
  %a = seq.firreg %a_next clock %clk : i8
  %a_next = comb.mux %e, %z, %a : i8
  %e = seq.firreg %true clock %clk preset 0 : i1
  %ram = seq.firmem 1, 1, old, undefined : <8 x 8>
  %rd = seq.firmem.read_port %ram[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %ram[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  %z = seq.firreg %rd clock %clk : i8
  hw.output %a : i8
}
hw.module @Top(in %clk : !seq.clock, in %we : i1, in %waddr : i3, in %wdata : i8, in %raddr : i3, out r : i8) {
  %r = hw.instance "c" @Cap(clk: %clk: !seq.clock, we: %we: i1, waddr: %waddr: i3, wdata: %wdata: i8, raddr: %raddr: i3) -> (r: i8)
  hw.output %r : i8
}
