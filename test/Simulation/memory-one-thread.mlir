// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Mem netlist" --foldsim-compose="module=Top" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/obj %S/Inputs/tb_memory.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb | FileCheck %s
// With the state read a host cycle ahead and committed by select, the
// tokens are the same too.
// RUN: mkdir -p %t/p && foldsim-opt %s --foldsim-extract-models="module=Top models=Mem netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select" -o %t/p/sim.mlir
// RUN: firtool --format=mlir %t/p/sim.mlir -o %t/p/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/p/obj %S/Inputs/tb_memory.sv %t/p/sim.sv > %t/p/build.log 2>&1
// RUN: %t/p/obj/Vtb | FileCheck %s

// A registered read's data is the next target cycle's, with one thread too:
// the read of address 1 in cycle 2 sees the A1 written in cycle 0, that of
// address 2 in cycle 3 the B2 written in cycle 1. Token 1 is the read of
// address 0, never written.
// CHECK: r 2 = a1
// CHECK: r 3 = b2
// CHECK: r 4 = a1
// CHECK: r 5 = b2
hw.module private @Mem(in %clk : !seq.clock, in %we : i1, in %waddr : i3, in %wdata : i8, in %raddr : i3, out r : i8) {
  %true = hw.constant true
  %ram = seq.firmem 1, 1, old, undefined : <8 x 8>
  %r = seq.firmem.read_port %ram[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %ram[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %we : i1, in %waddr : i3, in %wdata : i8, in %raddr : i3, out r : i8) {
  %r = hw.instance "m" @Mem(clk: %clk: !seq.clock, we: %we: i1, waddr: %waddr: i3, wdata: %wdata: i8, raddr: %raddr: i3) -> (r: i8)
  hw.output %r : i8
}
