// REQUIRES: verilator
// A class with an associative table (foldsim-memories-to-registers): every
// entry matched against the input's low bits each cycle (a read at each constant
// address), one more read at an address from the input, a write at a
// rotating pointer on an enable independent of the table. Four instances in
// a chain. The dynamic simulator of the table as a memory is the reference;
// the static fold with the table as registers, a copy per two instances and
// one copy of all four, gives the same tokens (and the static fold of the
// memory does too).
// RUN: rm -rf %t && mkdir -p %t
// RUN: mkdir -p %t/ref && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=W netlist},foldsim-compose{module=Top})" -o %t/ref/sim.mlir
// RUN: firtool --format=mlir %t/ref/sim.mlir -o %t/ref/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/ref/obj %S/Inputs/tb_staged.sv %t/ref/sim.sv > %t/ref/build.log 2>&1
// RUN: %t/ref/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: mkdir -p %t/m && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=W netlist},foldsim-fold-static{module=Top modulo zero-latency-links copies=W:2})" -o %t/m/sim.mlir
// RUN: firtool --format=mlir %t/m/sim.mlir -o %t/m/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/m/obj %S/Inputs/tb_staged.sv %t/m/sim.sv > %t/m/build.log 2>&1
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: mkdir -p %t/c && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-memories-to-registers{memories=W:table},foldsim-extract-models{module=Top models=W netlist},foldsim-fold-static{module=Top modulo zero-latency-links copies=W:2})" -o %t/c/sim.mlir
// RUN: FileCheck %s --check-prefix=REGS < %t/c/sim.mlir
// RUN: firtool --format=mlir %t/c/sim.mlir -o %t/c/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c/obj %S/Inputs/tb_staged.sv %t/c/sim.sv > %t/c/build.log 2>&1
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/s && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-memories-to-registers{memories=W:table},foldsim-extract-models{module=Top models=W netlist},foldsim-fold-static{module=Top})" -o %t/s/sim.mlir
// RUN: firtool --format=mlir %t/s/sim.mlir -o %t/s/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/s/obj %S/Inputs/tb_staged.sv %t/s/sim.sv > %t/s/build.log 2>&1
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// The table of 8 x 8 bits, two instances deep, is no longer a store of the
// copies: their state is 64 bits wider.
// REGS-NOT: <16 x 8, mask 1>
// CHECK: z0: 08 0c 16 12 1f 1b 25 2c 25 20 3e
// CHECK: z1: 20 30 58 48 7c 6c 94 b0 9c 82 fa
hw.module @W(in %clk : !seq.clock, in %reset : i1, in %x : i8, out q : i8) {
  %k0 = hw.constant 0 : i3
  %k1 = hw.constant 1 : i3
  %k2 = hw.constant 2 : i3
  %k3 = hw.constant 3 : i3
  %k4 = hw.constant -4 : i3
  %k5 = hw.constant -3 : i3
  %k6 = hw.constant -2 : i3
  %k7 = hw.constant -1 : i3
  %c0_i3 = hw.constant 0 : i3
  %c1_i3 = hw.constant 1 : i3
  %c0_i5 = hw.constant 0 : i5
  %c0_i8 = hw.constant 0 : i8
  %c0_i7 = hw.constant 0 : i7
  %c1_i8 = hw.constant 1 : i8
  %c7_i8 = hw.constant 7 : i8
  %key = comb.and %x, %c7_i8 : i8
  %table = seq.firmem name "table" 0, 1, undefined, undefined : <8 x 8, mask 1>
  %r0 = seq.firmem.read_port %table[%k0], clock %clk : <8 x 8, mask 1>
  %r1 = seq.firmem.read_port %table[%k1], clock %clk : <8 x 8, mask 1>
  %r2 = seq.firmem.read_port %table[%k2], clock %clk : <8 x 8, mask 1>
  %r3 = seq.firmem.read_port %table[%k3], clock %clk : <8 x 8, mask 1>
  %r4 = seq.firmem.read_port %table[%k4], clock %clk : <8 x 8, mask 1>
  %r5 = seq.firmem.read_port %table[%k5], clock %clk : <8 x 8, mask 1>
  %r6 = seq.firmem.read_port %table[%k6], clock %clk : <8 x 8, mask 1>
  %r7 = seq.firmem.read_port %table[%k7], clock %clk : <8 x 8, mask 1>
  %e0 = comb.icmp eq %r0, %key : i8
  %e1 = comb.icmp eq %r1, %key : i8
  %e2 = comb.icmp eq %r2, %key : i8
  %e3 = comb.icmp eq %r3, %key : i8
  %e4 = comb.icmp eq %r4, %key : i8
  %e5 = comb.icmp eq %r5, %key : i8
  %e6 = comb.icmp eq %r6, %key : i8
  %e7 = comb.icmp eq %r7, %key : i8
  %hits = comb.concat %e7, %e6, %e5, %e4, %e3, %e2, %e1, %e0 : i1, i1, i1, i1, i1, i1, i1, i1
  %h = comb.and %hits, %valid : i8
  %we = comb.extract %x from 0 : (i8) -> i1
  seq.firmem.write_port %table[%wp] = %key, clock %clk enable %we : <8 x 8, mask 1>
  %wp1 = comb.add %wp, %c1_i3 : i3
  %wpn = comb.mux %we, %wp1, %wp : i3
  %wp = seq.firreg %wpn clock %clk reset sync %reset, %c0_i3 : i3
  %wpw = comb.concat %c0_i5, %wp : i5, i3
  %bit = comb.shl %c1_i8, %wpw : i8
  %set = comb.mux %we, %bit, %c0_i8 : i8
  %validn = comb.or %valid, %set : i8
  %valid = seq.firreg %validn clock %clk reset sync %reset, %c0_i8 : i8
  %ra = comb.extract %x from 3 : (i8) -> i3
  %rd = seq.firmem.read_port %table[%ra], clock %clk : <8 x 8, mask 1>
  %raw = comb.concat %c0_i5, %ra : i5, i3
  %vs = comb.shru %valid, %raw : i8
  %vb = comb.extract %vs from 0 : (i8) -> i1
  %vm = comb.replicate %vb : (i1) -> i8
  %rv = comb.and %rd, %vm : i8
  %any = comb.icmp ne %h, %c0_i8 : i8
  %anyw = comb.concat %c0_i7, %any : i7, i1
  %cntn = comb.add %cnt, %anyw : i8
  %cnt = seq.firreg %cntn clock %clk reset sync %reset, %c0_i8 : i8
  %xl = comb.extract %x from 0 : (i8) -> i7
  %xh = comb.extract %x from 7 : (i8) -> i1
  %xr = comb.concat %xl, %xh : i7, i1
  %q = comb.xor %h, %rv, %cnt, %xr : i8
  hw.output %q : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z0 : i8, out z1 : i8) {
  %w0.q = hw.instance "w0" @W(clk: %clk: !seq.clock, reset: %reset: i1, x: %x: i8) -> (q: i8)
  %w1.q = hw.instance "w1" @W(clk: %clk: !seq.clock, reset: %reset: i1, x: %w0.q: i8) -> (q: i8)
  %w2.q = hw.instance "w2" @W(clk: %clk: !seq.clock, reset: %reset: i1, x: %w1.q: i8) -> (q: i8)
  %w3.q = hw.instance "w3" @W(clk: %clk: !seq.clock, reset: %reset: i1, x: %w2.q: i8) -> (q: i8)
  hw.output %w1.q, %w3.q : i8, i8
}
