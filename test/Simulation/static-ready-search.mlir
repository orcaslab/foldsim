// REQUIRES: verilator
// A ready pass computes every late output it can, which here makes it both
// feed a class and read it: Top's ready pass computes the arbiter-like A's
// valid from S, and would read A's ready back for S; A's ready pass would
// feed B and read B back. No choice of ready passes places, so the search
// moves the outputs that close each cycle to their state pass (Top's two
// outputs to S, A's `k`): the same tokens as the dynamic simulator's, on
// every schedule.
// RUN: rm -rf %t && mkdir -p %t
// RUN: mkdir -p %t/ref && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=S,A,B netlist},foldsim-compose{module=Top})" -o %t/ref/sim.mlir
// RUN: firtool --format=mlir %t/ref/sim.mlir -o %t/ref/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/ref/obj %S/Inputs/tb_staged.sv %t/ref/sim.sv > %t/ref/build.log 2>&1
// RUN: %t/ref/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: mkdir -p %t/s && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=S,A,B netlist},foldsim-fold-static{module=Top})" -o %t/s/sim.mlir
// RUN: firtool --format=mlir %t/s/sim.mlir -o %t/s/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/s/obj %S/Inputs/tb_staged.sv %t/s/sim.sv > %t/s/build.log 2>&1
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/z && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=S,A,B netlist},foldsim-fold-static{module=Top modulo zero-latency-links zero-latency-chains})" -o %t/z/sim.mlir
// RUN: firtool --format=mlir %t/z/sim.mlir -o %t/z/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/z/obj %S/Inputs/tb_staged.sv %t/z/sim.sv > %t/z/build.log 2>&1
// RUN: %t/z/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/z/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/c && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=S,A,B netlist},foldsim-fold-static{module=Top copies=A:2,B:2})" -o %t/c/sim.mlir
// RUN: firtool --format=mlir %t/c/sim.mlir -o %t/c/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c/obj %S/Inputs/tb_staged.sv %t/c/sim.sv > %t/c/build.log 2>&1
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// CHECK: z0: 00 00 01 07 0f 17 1e 28 23 2e 3a
// CHECK: z1: 00 02 04 06 08 0a 0c 0e 10 11 13
hw.module @S(in %clk : !seq.clock, in %reset : i1, in %r_in : i1, in %k_in : i1, in %x : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c0_i6 = hw.constant 0 : i6
  %rk = comb.concat %c0_i6, %r_in, %k_in : i6, i1, i1
  %s = comb.add %q, %rk : i8
  %n = comb.xor %s, %x : i8
  %q = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  hw.output %q : i8
}
hw.module @A(in %clk : !seq.clock, in %reset : i1, in %v_in : i8, in %b_in : i8, out r : i1, out o : i8, out k : i1) {
  %c0 = hw.constant 0 : i8
  %v0 = comb.extract %v_in from 0 : (i8) -> i1
  %a0 = comb.extract %areg from 0 : (i8) -> i1
  %r = comb.xor %v0, %a0 : i1
  %o = comb.add %v_in, %areg : i8
  %b1 = comb.extract %b_in from 1 : (i8) -> i1
  %a1 = comb.extract %areg from 1 : (i8) -> i1
  %k = comb.and %b1, %a1 : i1
  %vb = comb.xor %v_in, %b_in : i8
  %an = comb.add %areg, %vb : i8
  %areg = seq.firreg %an clock %clk reset sync %reset, %c0 : i8
  hw.output %r, %o, %k : i1, i8, i1
}
hw.module @B(in %clk : !seq.clock, in %reset : i1, in %o_in : i8, out b : i8) {
  %c0 = hw.constant 0 : i8
  %bn = comb.add %breg, %o_in : i8
  %breg = seq.firreg %bn clock %clk reset sync %reset, %c0 : i8
  %b = comb.xor %o_in, %breg : i8
  hw.output %b : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z0 : i8, out z1 : i8) {
  %c0 = hw.constant 0 : i8
  %c1 = hw.constant 1 : i8
  %c0_i7 = hw.constant 0 : i7
  %s.q = hw.instance "s" @S(clk: %clk: !seq.clock, reset: %reset: i1, r_in: %r_out: i1, k_in: %k_out: i1, x: %x: i8) -> (q: i8)
  %v = comb.xor %treg, %s.q : i8
  %v1 = comb.add %v, %c1 : i8
  %a0.r, %a0.o, %a0.k = hw.instance "a0" @A(clk: %clk: !seq.clock, reset: %reset: i1, v_in: %v: i8, b_in: %b0.b: i8) -> (r: i1, o: i8, k: i1)
  %a1.r, %a1.o, %a1.k = hw.instance "a1" @A(clk: %clk: !seq.clock, reset: %reset: i1, v_in: %v1: i8, b_in: %b1.b: i8) -> (r: i1, o: i8, k: i1)
  %b0.b = hw.instance "b0" @B(clk: %clk: !seq.clock, reset: %reset: i1, o_in: %a0.o: i8) -> (b: i8)
  %b1.b = hw.instance "b1" @B(clk: %clk: !seq.clock, reset: %reset: i1, o_in: %a1.o: i8) -> (b: i8)
  %r_out = comb.or %a0.r, %a1.r : i1
  %k_out = comb.xor %a0.k, %a1.k : i1
  %rz = comb.concat %c0_i7, %r_out : i7, i1
  %tn = comb.add %treg, %rz, %c1 : i8
  %treg = seq.firreg %tn clock %clk reset sync %reset, %c0 : i8
  hw.output %s.q, %treg : i8, i8
}
