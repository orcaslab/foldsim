// REQUIRES: verilator
// A class input whose bits are a constant wider than 64 bits differing by
// instance, where another instance's same bits are a link (an omega
// switch's port, tied off in one instance): the copy reads each instance's
// constant from a table by instance, 64 bits a slice. The same
// tokens as the dynamic simulator's, threaded and with a copy each.
// RUN: rm -rf %t && mkdir -p %t
// RUN: mkdir -p %t/ref && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=W netlist},foldsim-compose{module=Top})" -o %t/ref/sim.mlir
// RUN: firtool --format=mlir %t/ref/sim.mlir -o %t/ref/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/ref/obj %S/Inputs/tb_staged.sv %t/ref/sim.sv > %t/ref/build.log 2>&1
// RUN: %t/ref/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: mkdir -p %t/s && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=W netlist},foldsim-fold-static{module=Top})" -o %t/s/sim.mlir
// RUN: firtool --format=mlir %t/s/sim.mlir -o %t/s/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/s/obj %S/Inputs/tb_staged.sv %t/s/sim.sv > %t/s/build.log 2>&1
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/c && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=W netlist},foldsim-fold-static{module=Top modulo zero-latency-links copies=W:2})" -o %t/c/sim.mlir
// RUN: firtool --format=mlir %t/c/sim.mlir -o %t/c/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c/obj %S/Inputs/tb_staged.sv %t/c/sim.sv > %t/c/build.log 2>&1
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// CHECK: z0: 00 5b b7 14 72 d1 31 92 f4 57 bb
// CHECK: z1: 00 03 b6 21 4c 2f d2 35 58 43 ee
hw.module @W(in %clk : !seq.clock, in %reset : i1, in %k : i80, in %x : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %hi = comb.extract %k from 72 : (i80) -> i8
  %mid = comb.extract %k from 60 : (i80) -> i8
  %lo = comb.extract %k from 0 : (i80) -> i8
  %t = comb.xor %hi, %mid, %lo : i8
  %s = comb.add %q, %x, %t : i8
  %q = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %q : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z0 : i8, out z1 : i8) {
  %k0 = hw.constant 0x5A000000000000000003 : i80
  %c72 = hw.constant 0x030000000000000001 : i72
  %k1 = comb.concat %c72, %w0.q : i72, i8
  %w0.q = hw.instance "w0" @W(clk: %clk: !seq.clock, reset: %reset: i1, k: %k0: i80, x: %x: i8) -> (q: i8)
  %w1.q = hw.instance "w1" @W(clk: %clk: !seq.clock, reset: %reset: i1, k: %k1: i80, x: %w0.q: i8) -> (q: i8)
  hw.output %w0.q, %w1.q : i8, i8
}
