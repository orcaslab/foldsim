// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child,Sink netlist" --foldsim-compose="module=Top" -o %t/sim.mlir
// RUN: firtool --format=mlir %t/sim.mlir -o %t/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/obj %S/Inputs/tb_staged.sv %t/sim.sv > %t/build.log 2>&1
// RUN: %t/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// With staged tokens every slot is written a host cycle late and read
// through the staging registers until then: the same tokens on the same host
// cycles.
// RUN: mkdir -p %t/s && foldsim-opt %s --foldsim-extract-models="module=Top models=Child,Sink netlist" --foldsim-compose="module=Top staged-tokens" -o %t/s/sim.mlir
// RUN: firtool --format=mlir %t/s/sim.mlir -o %t/s/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/s/obj %S/Inputs/tb_staged.sv %t/s/sim.sv > %t/s/build.log 2>&1
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/s/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// And with registered state.
// RUN: mkdir -p %t/p && foldsim-opt %s --foldsim-extract-models="module=Top models=Child,Sink netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select staged-tokens" -o %t/p/sim.mlir
// RUN: firtool --format=mlir %t/p/sim.mlir -o %t/p/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/p/obj %S/Inputs/tb_staged.sv %t/p/sim.sv > %t/p/build.log 2>&1
// RUN: %t/p/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s --check-prefix=TOKENS

// Four slots per channel, with and without staged tokens: the same tokens.
// RUN: mkdir -p %t/d && foldsim-opt %s --foldsim-extract-models="module=Top models=Child,Sink netlist" --foldsim-compose="module=Top channel-depth=4" -o %t/d/sim.mlir
// RUN: firtool --format=mlir %t/d/sim.mlir -o %t/d/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/d/obj %S/Inputs/tb_staged.sv %t/d/sim.sv > %t/d/build.log 2>&1
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s --check-prefix=TOKENS
// RUN: mkdir -p %t/ds && foldsim-opt %s --foldsim-extract-models="module=Top models=Child,Sink netlist" --foldsim-compose="module=Top channel-depth=4 staged-tokens" -o %t/ds/sim.mlir
// RUN: firtool --format=mlir %t/ds/sim.mlir -o %t/ds/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal --x-initial unique --top-module tb -Mdir %t/ds/obj %S/Inputs/tb_staged.sv %t/ds/sim.sv > %t/ds/build.log 2>&1
// RUN: %t/ds/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s --check-prefix=TOKENS
// Two accumulators of `x`, each read by a register of its own thread (its
// slots in a memory), on the netlist's outputs (slots in registers).
// CHECK: z0: 00 00 02 05 09 0e 14 1b 23 2c 36
// CHECK: z1: 00 00 02 05 09 0e 14 1b 23 2c 36
// CHECK: last token on host cycle 30
// TOKENS: z0: 00 00 02 05 09 0e 14 1b 23 2c 36
// TOKENS: z1: 00 00 02 05 09 0e 14 1b 23 2c 36
hw.module @Child(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %s = comb.add %a, %r : i8
  %r = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Sink(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %r = seq.firreg %a clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z0 : i8, out z1 : i8) {
  %a0.y = hw.instance "a0" @Child(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %a1.y = hw.instance "a1" @Child(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %b0.y = hw.instance "b0" @Sink(clk: %clk: !seq.clock, reset: %reset: i1, a: %a0.y: i8) -> (y: i8)
  %b1.y = hw.instance "b1" @Sink(clk: %clk: !seq.clock, reset: %reset: i1, a: %a1.y: i8) -> (y: i8)
  hw.output %b0.y, %b1.y : i8, i8
}
