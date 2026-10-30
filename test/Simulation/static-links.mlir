// REQUIRES: verilator
// A statically folded simulator's links, against the dynamic one's tokens:
// four queues whose register loads on a ready the parent decides in the same
// cycle from their valids (zero-latency links into them, their early outputs'
// next values out of them), read by the parent and by two sinks. Forwarding
// into a read register only where the schedule reads a write's entry the
// host cycle after it, past the read register where the reader stays
// shallow, and the links of bits several classes read written a host cycle
// late from one register of them: the same tokens, whatever the schedule.
// RUN: rm -rf %t && mkdir -p %t
// RUN: mkdir -p %t/ref && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-compose{module=Top})" -o %t/ref/sim.mlir
// RUN: firtool --format=mlir %t/ref/sim.mlir -o %t/ref/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/ref/obj %S/Inputs/tb_staged.sv %t/ref/sim.sv > %t/ref/build.log 2>&1
// RUN: %t/ref/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: mkdir -p %t/m && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top modulo copies=Sink:2})" -o %t/m/sim.mlir
// RUN: firtool --format=mlir %t/m/sim.mlir -o %t/m/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/m/obj %S/Inputs/tb_staged.sv %t/m/sim.sv > %t/m/build.log 2>&1
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/z && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top zero-latency-links})" -o %t/z/sim.mlir
// RUN: firtool --format=mlir %t/z/sim.mlir -o %t/z/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/z/obj %S/Inputs/tb_staged.sv %t/z/sim.sv > %t/z/build.log 2>&1
// RUN: %t/z/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/z/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/c && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top modulo zero-latency-links zero-latency-chains copies=Queue:2})" -o %t/c/sim.mlir
// RUN: firtool --format=mlir %t/c/sim.mlir -o %t/c/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c/obj %S/Inputs/tb_staged.sv %t/c/sim.sv > %t/c/build.log 2>&1
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/q && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top copies=Queue:2,Sink:2})" -o %t/q/sim.mlir
// RUN: firtool --format=mlir %t/q/sim.mlir -o %t/q/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/q/obj %S/Inputs/tb_staged.sv %t/q/sim.sv > %t/q/build.log 2>&1
// RUN: %t/q/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// Without forwarding behind the read register or shared writes, and with
// two-cycle links (registered writes forwarded the host cycle they land).
// RUN: mkdir -p %t/o && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top zero-latency-links forward-after-levels=0 share-link-writes=false})" -o %t/o/sim.mlir
// RUN: firtool --format=mlir %t/o/sim.mlir -o %t/o/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/o/obj %S/Inputs/tb_staged.sv %t/o/sim.sv > %t/o/build.log 2>&1
// RUN: %t/o/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: mkdir -p %t/t && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top one-cycle-links=false})" -o %t/t/sim.mlir
// RUN: firtool --format=mlir %t/t/sim.mlir -o %t/t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/t/obj %S/Inputs/tb_staged.sv %t/t/sim.sv > %t/t/build.log 2>&1
// RUN: %t/t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// CHECK: z0: 00 00 05 0b 12 1a 23 2d 38 44 51
// CHECK: z1: 00 00 0a 16 24 34 46 5a 70 88 a2
hw.module @Queue(in %clk : !seq.clock, in %reset : i1, in %d : i8, in %rdy : i1, out q : i8, out v : i1) {
  %c0 = hw.constant 0 : i8
  %false = hw.constant false
  %c3 = hw.constant 3 : i8
  %dn = comb.add %d, %c3 : i8
  %qn = comb.mux %rdy, %dn, %q : i8
  %q = seq.firreg %qn clock %clk reset sync %reset, %c0 : i8
  %vn = comb.xor %v, %rdy : i1
  %v = seq.firreg %vn clock %clk reset sync %reset, %false : i1
  hw.output %q, %v : i8, i1
}
hw.module @Sink(in %clk : !seq.clock, in %reset : i1, in %a : i8, in %b : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %s = comb.add %r, %a, %b : i8
  %r = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z0 : i8, out z1 : i8) {
  %true = hw.constant true
  %c0 = hw.constant 0 : i8
  %c1 = hw.constant 1 : i8
  %q0.q, %q0.v = hw.instance "q0" @Queue(clk: %clk: !seq.clock, reset: %reset: i1, d: %x: i8, rdy: %r0: i1) -> (q: i8, v: i1)
  %d1 = comb.add %q0.q, %c1 : i8
  %q1.q, %q1.v = hw.instance "q1" @Queue(clk: %clk: !seq.clock, reset: %reset: i1, d: %d1: i8, rdy: %r1: i1) -> (q: i8, v: i1)
  %d2 = comb.xor %q1.q, %x : i8
  %q2.q, %q2.v = hw.instance "q2" @Queue(clk: %clk: !seq.clock, reset: %reset: i1, d: %d2: i8, rdy: %r2: i1) -> (q: i8, v: i1)
  %q3.q, %q3.v = hw.instance "q3" @Queue(clk: %clk: !seq.clock, reset: %reset: i1, d: %q2.q: i8, rdy: %r3: i1) -> (q: i8, v: i1)
  %n0 = comb.and %q0.v, %q1.v : i1
  %r0 = comb.xor %n0, %true : i1
  %r1 = comb.xor %q0.v, %q2.v : i1
  %n3 = comb.xor %q3.v, %true : i1
  %r2 = comb.or %n3, %q1.v : i1
  %n0b = comb.xor %q0.v, %true : i1
  %r3 = comb.and %q2.v, %n0b : i1
  %s = comb.add %acc, %q3.q, %q0.q : i8
  %acc = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  %s0.y = hw.instance "s0" @Sink(clk: %clk: !seq.clock, reset: %reset: i1, a: %q1.q: i8, b: %q3.q: i8) -> (y: i8)
  %s1.y = hw.instance "s1" @Sink(clk: %clk: !seq.clock, reset: %reset: i1, a: %q2.q: i8, b: %q0.q: i8) -> (y: i8)
  %z1 = comb.xor %s0.y, %s1.y : i8
  hw.output %acc, %z1 : i8, i8
}
