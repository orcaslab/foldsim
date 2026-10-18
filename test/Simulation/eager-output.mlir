// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-fame5="module=M threads=1 clock-enable" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/obj %S/Inputs/tb_eager.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb | FileCheck %s
// With the state read a host cycle ahead and committed by select, the
// tokens are the same too.
// RUN: mkdir -p %t/p && foldsim-opt %s --foldsim-fame5="module=M threads=1 clock-enable registered-state commit-select" -o %t/p/sim.mlir
// RUN: firtool --format=mlir %t/p/sim.mlir -o %t/p/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --top-module tb -Mdir %t/p/obj %S/Inputs/tb_eager.sv %t/p/sim.sv > %t/p/build.log 2>&1
// RUN: %t/p/obj/Vtb | FileCheck %s

// `y` fires eagerly, ahead of `z`, which waits for `m`; both are `acc`. The
// target clock does not tick on the first finish after reset, so `y`'s second
// token is its first again, as is `z`'s. `acc` is reset in cycle 1 and then
// adds `a`.
// CHECK: y 1 is z 1: 1
// CHECK: y 2 = 00, z 2 = 00
// CHECK: y 3 = 02, z 3 = 02
// CHECK: y 4 = 06, z 4 = 06
// CHECK: y 5 = 0e, z 5 = 0e
hw.module @M(in %clk : !seq.clock, in %reset : i1, in %a : i8, in %m : i8, out y : i8, out z : i8) {
  %c0_i8 = hw.constant 0 : i8
  %sum = comb.add %acc, %a : i8
  %acc = seq.firreg %sum clock %clk reset sync %reset, %c0_i8 : i8
  %z = comb.and %acc, %m : i8
  hw.output %acc, %z : i8, i8
}
