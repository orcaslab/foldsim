// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Mem netlist" --foldsim-compose="module=Top" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -DTWO -Mdir %t/obj %S/Inputs/tb_memory.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb | FileCheck %s
// Regions of one memory each: every write lands a host cycle late, from
// staged registers, and the tokens are the same.
// RUN: mkdir -p %t/r && foldsim-opt %s --foldsim-extract-models="module=Top models=Mem netlist" --foldsim-compose="module=Top region-loads=1" -o %t/r/sim.mlir
// RUN: firtool --format=mlir %t/r/sim.mlir -o %t/r/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -DTWO --top-module tb -Mdir %t/r/obj %S/Inputs/tb_memory.sv %t/r/sim.sv > %t/r/build.log 2>&1
// RUN: %t/r/obj/Vtb | FileCheck %s

// As with one thread (memory-one-thread.mlir), each thread's own.
// CHECK: r 2 = a1
// CHECK: r 3 = b2
// CHECK: r 4 = a1
// CHECK: r 5 = b2
// CHECK: r1 2 = a1
// CHECK: r1 3 = b2
// CHECK: r1 4 = a1
// CHECK: r1 5 = b2
hw.module private @Mem(in %clk : !seq.clock, in %we : i1, in %waddr : i3, in %wdata : i8, in %raddr : i3, out r : i8) {
  %true = hw.constant true
  %ram = seq.firmem 1, 1, old, undefined : <8 x 8>
  %r = seq.firmem.read_port %ram[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %ram[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %we : i1, in %waddr : i3, in %wdata : i8, in %raddr : i3, out r : i8, out r1 : i8) {
  %r = hw.instance "m" @Mem(clk: %clk: !seq.clock, we: %we: i1, waddr: %waddr: i3, wdata: %wdata: i8, raddr: %raddr: i3) -> (r: i8)
  %r1 = hw.instance "m1" @Mem(clk: %clk: !seq.clock, we: %we: i1, waddr: %waddr: i3, wdata: %wdata: i8, raddr: %raddr: i3) -> (r: i8)
  hw.output %r, %r1 : i8, i8
}
