// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Mem netlist" --foldsim-compose="module=Top" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir --add-vivado-ram-address-conflict-synthesis-bug-workaround -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/obj %S/Inputs/tb_memory_new.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb | FileCheck %s
// Regions: registered-read memories are written on the edge. With
// `staged-ram-writes` every write lands a host cycle late, from staged
// registers, and a registered read of the address written on its edge reads
// the staged data.
// RUN: mkdir -p %t/r && foldsim-opt %s --foldsim-extract-models="module=Top models=Mem netlist" --foldsim-compose="module=Top region-loads=1" -o %t/r/sim.mlir
// RUN: firtool --format=mlir %t/r/sim.mlir --add-vivado-ram-address-conflict-synthesis-bug-workaround -o %t/r/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/r/obj %S/Inputs/tb_memory_new.sv %t/r/sim.sv > %t/r/build.log 2>&1
// RUN: %t/r/obj/Vtb | FileCheck %s
// RUN: mkdir -p %t/w && foldsim-opt %s --foldsim-extract-models="module=Top models=Mem netlist" --foldsim-compose="module=Top region-loads=1 staged-ram-writes" -o %t/w/sim.mlir
// RUN: firtool --format=mlir %t/w/sim.mlir --add-vivado-ram-address-conflict-synthesis-bug-workaround -o %t/w/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/w/obj %S/Inputs/tb_memory_new.sv %t/w/sim.sv > %t/w/build.log 2>&1
// RUN: %t/w/obj/Vtb | FileCheck %s
// RUN: mkdir -p %t/p && foldsim-opt %s --foldsim-extract-models="module=Top models=Mem netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select staged-ram-writes staged-tokens" -o %t/p/sim.mlir
// RUN: firtool --format=mlir %t/p/sim.mlir --add-vivado-ram-address-conflict-synthesis-bug-workaround -o %t/p/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/p/obj %S/Inputs/tb_memory_new.sv %t/p/sim.sv > %t/p/build.log 2>&1
// RUN: %t/p/obj/Vtb | FileCheck %s

// Memories that read new data under a write, each thread its own: token k is
// the read of target cycle k - 1, which sees that cycle's write; `s`'s second
// write keeps the high nibble.
// CHECK: r0: a1 c3 b2 c3 b2
// CHECK: s0: a1 a3 b2 a3 b2
// CHECK: r1: a1 c3 b2 c3 b2
// CHECK: s1: a1 a3 b2 a3 b2
hw.module private @Mem(in %clk : !seq.clock, in %we : i1, in %wmask : i2, in %waddr : i3, in %wdata : i8, in %raddr : i3, out r : i8, out s : i8) {
  %true = hw.constant true
  %ram = seq.firmem 1, 1, new, undefined : <8 x 8>
  %r = seq.firmem.read_port %ram[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %ram[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  %ramm = seq.firmem 1, 1, new, undefined : <8 x 8, mask 2>
  %s = seq.firmem.read_port %ramm[%raddr], clock %clk enable %true : <8 x 8, mask 2>
  seq.firmem.write_port %ramm[%waddr] = %wdata, clock %clk enable %we mask %wmask : <8 x 8, mask 2>, i2
  hw.output %r, %s : i8, i8
}
hw.module @Top(in %clk : !seq.clock, in %we : i1, in %wmask : i2, in %waddr : i3, in %wdata : i8, in %raddr : i3, out r0 : i8, out s0 : i8, out r1 : i8, out s1 : i8) {
  %r0, %s0 = hw.instance "m0" @Mem(clk: %clk: !seq.clock, we: %we: i1, wmask: %wmask: i2, waddr: %waddr: i3, wdata: %wdata: i8, raddr: %raddr: i3) -> (r: i8, s: i8)
  %r1, %s1 = hw.instance "m1" @Mem(clk: %clk: !seq.clock, we: %we: i1, wmask: %wmask: i2, waddr: %waddr: i3, wdata: %wdata: i8, raddr: %raddr: i3) -> (r: i8, s: i8)
  hw.output %r0, %s0, %r1, %s1 : i8, i8, i8, i8
}
