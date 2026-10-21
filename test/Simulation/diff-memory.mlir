// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: firtool --format=mlir %s -o %t/target.sv
// Plain threading.
// RUN: mkdir -p %t/b && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top" -o %t/b/sim.mlir
// RUN: firtool --format=mlir %t/b/sim.mlir -o %t/b/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/b/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/b/sim.sv > %t/b/build.log 2>&1
// RUN: %t/b/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/b/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// Regions of one memory each.
// RUN: mkdir -p %t/r && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1" -o %t/r/sim.mlir
// RUN: firtool --format=mlir %t/r/sim.mlir -o %t/r/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/r/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/r/sim.sv > %t/r/build.log 2>&1
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// State read a host cycle ahead and committed by select.
// RUN: mkdir -p %t/p && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select" -o %t/p/sim.mlir
// RUN: firtool --format=mlir %t/p/sim.mlir -o %t/p/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/p/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/p/sim.sv > %t/p/build.log 2>&1
// RUN: %t/p/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/p/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// The image flow's state options: written two host cycles late,
// multipliers pipelined, staged memory writes and tokens.
// RUN: mkdir -p %t/f && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens" -o %t/f/sim.mlir
// RUN: firtool --format=mlir %t/f/sim.mlir -o %t/f/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/f/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/f/sim.sv > %t/f/build.log 2>&1
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// The same, with compact staging.
// RUN: mkdir -p %t/c && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging" -o %t/c/sim.mlir
// RUN: firtool --format=mlir %t/c/sim.mlir -o %t/c/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/c/sim.sv > %t/c/build.log 2>&1
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// The same, each model scheduled in turn rather than by readiness.
// RUN: mkdir -p %t/i && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens" -o %t/i/sim.mlir
// RUN: firtool --format=mlir %t/i/sim.mlir -o %t/i/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/i/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/i/sim.sv > %t/i/build.log 2>&1
// RUN: %t/i/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/i/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// The same with token memories, memories and thread state split into pieces
// of at most 4 loads, each written by registers of its own.
// RUN: mkdir -p %t/k && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens memory-loads=4" -o %t/k/sim.mlir
// RUN: grep -q '_tokens_3_0_we = seq.firreg' %t/k/sim.mlir
// RUN: grep -q 'ram_rdata_p1 = seq.firmem' %t/k/sim.mlir
// RUN: firtool --format=mlir %t/k/sim.mlir -o %t/k/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/k/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/k/sim.sv > %t/k/build.log 2>&1
// RUN: %t/k/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/k/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// The same, as a tree of domains.
// RUN: mkdir -p %t/d && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens domains" -o %t/d/sim.mlir
// RUN: firtool --format=mlir %t/d/sim.mlir -o %t/d/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/d/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/d/sim.sv > %t/d/build.log 2>&1
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// Deeper channels with staged tokens.
// RUN: mkdir -p %t/q && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select state-stages=2 staged-tokens channel-depth=4" -o %t/q/sim.mlir
// RUN: firtool --format=mlir %t/q/sim.mlir -o %t/q/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/q/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/q/sim.sv > %t/q/build.log 2>&1
// RUN: %t/q/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/q/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// A relay stage on the memory model's channels.
// RUN: mkdir -p %t/l && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top relay=MemModel" -o %t/l/sim.mlir
// RUN: firtool --format=mlir %t/l/sim.mlir -o %t/l/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/l/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/l/sim.sv > %t/l/build.log 2>&1
// RUN: %t/l/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/l/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// The relay with the image flow's state options.
// RUN: mkdir -p %t/m && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens relay=MemModel" -o %t/m/sim.mlir
// RUN: firtool --format=mlir %t/m/sim.mlir -o %t/m/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/m/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/m/sim.sv > %t/m/build.log 2>&1
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// The relay between the two classes, as a tree of domains.
// RUN: mkdir -p %t/n && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens domains relay=MemModel:Post" -o %t/n/sim.mlir
// RUN: firtool --format=mlir %t/n/sim.mlir -o %t/n/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/n/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/n/sim.sv > %t/n/build.log 2>&1
// RUN: %t/n/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/n/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s

// Four instances of a model with a memory read a cycle late (old data under a
// write to the same address) and an accumulator of its reads through a
// multiplier, each instance on inputs of its own and feeding an instance of a
// second class, which registers it with feedback; the target reset at the
// start and in mid-run; against the unthreaded target (tb_diffmem.sv).
// CHECK: MATCH 4 outputs
hw.module private @MemModel(in %clk : !seq.clock, in %reset : i1, in %we : i1, in %waddr : i3, in %wdata : i8, in %raddr : i3, in %en : i1, out y : i16) {
  %true = hw.constant true
  %c0_i8 = hw.constant 0 : i8
  %c0_i16 = hw.constant 0 : i16
  %c3_i16 = hw.constant 3 : i16
  %ram = seq.firmem 1, 1, old, undefined : <8 x 8>
  %r = seq.firmem.read_port %ram[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %ram[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  %r16 = comb.concat %c0_i8, %r : i8, i8
  %m = comb.mul %acc, %c3_i16 : i16
  %s = comb.add %m, %r16 : i16
  %n = comb.mux %en, %s, %acc : i16
  %nr = comb.mux %reset, %c0_i16, %n : i16
  %acc = seq.firreg %nr clock %clk : i16
  %y = comb.xor %acc, %r16 : i16
  hw.output %y : i16
}
hw.module private @Post(in %clk : !seq.clock, in %reset : i1, in %y : i16, out z : i16) {
  %c0_i16 = hw.constant 0 : i16
  %c1_i16 = hw.constant 1 : i16
  %sh = comb.shl %z, %c1_i16 : i16
  %x = comb.xor %y, %sh : i16
  %nr = comb.mux %reset, %c0_i16, %x : i16
  %z = seq.firreg %nr clock %clk : i16
  hw.output %z : i16
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %we : i4, in %waddr : i12, in %wdata : i32, in %raddr : i12, in %en : i4, out y0 : i16, out y1 : i16, out y2 : i16, out y3 : i16) {
  %we0 = comb.extract %we from 0 : (i4) -> i1
  %wa0 = comb.extract %waddr from 0 : (i12) -> i3
  %wd0 = comb.extract %wdata from 0 : (i32) -> i8
  %ra0 = comb.extract %raddr from 0 : (i12) -> i3
  %en0 = comb.extract %en from 0 : (i4) -> i1
  %m0 = hw.instance "m0" @MemModel(clk: %clk: !seq.clock, reset: %reset: i1, we: %we0: i1, waddr: %wa0: i3, wdata: %wd0: i8, raddr: %ra0: i3, en: %en0: i1) -> (y: i16)
  %y0 = hw.instance "p0" @Post(clk: %clk: !seq.clock, reset: %reset: i1, y: %m0: i16) -> (z: i16)
  %we1 = comb.extract %we from 1 : (i4) -> i1
  %wa1 = comb.extract %waddr from 3 : (i12) -> i3
  %wd1 = comb.extract %wdata from 8 : (i32) -> i8
  %ra1 = comb.extract %raddr from 3 : (i12) -> i3
  %en1 = comb.extract %en from 1 : (i4) -> i1
  %m1 = hw.instance "m1" @MemModel(clk: %clk: !seq.clock, reset: %reset: i1, we: %we1: i1, waddr: %wa1: i3, wdata: %wd1: i8, raddr: %ra1: i3, en: %en1: i1) -> (y: i16)
  %y1 = hw.instance "p1" @Post(clk: %clk: !seq.clock, reset: %reset: i1, y: %m1: i16) -> (z: i16)
  %we2 = comb.extract %we from 2 : (i4) -> i1
  %wa2 = comb.extract %waddr from 6 : (i12) -> i3
  %wd2 = comb.extract %wdata from 16 : (i32) -> i8
  %ra2 = comb.extract %raddr from 6 : (i12) -> i3
  %en2 = comb.extract %en from 2 : (i4) -> i1
  %m2 = hw.instance "m2" @MemModel(clk: %clk: !seq.clock, reset: %reset: i1, we: %we2: i1, waddr: %wa2: i3, wdata: %wd2: i8, raddr: %ra2: i3, en: %en2: i1) -> (y: i16)
  %y2 = hw.instance "p2" @Post(clk: %clk: !seq.clock, reset: %reset: i1, y: %m2: i16) -> (z: i16)
  %we3 = comb.extract %we from 3 : (i4) -> i1
  %wa3 = comb.extract %waddr from 9 : (i12) -> i3
  %wd3 = comb.extract %wdata from 24 : (i32) -> i8
  %ra3 = comb.extract %raddr from 9 : (i12) -> i3
  %en3 = comb.extract %en from 3 : (i4) -> i1
  %m3 = hw.instance "m3" @MemModel(clk: %clk: !seq.clock, reset: %reset: i1, we: %we3: i1, waddr: %wa3: i3, wdata: %wd3: i8, raddr: %ra3: i3, en: %en3: i1) -> (y: i16)
  %y3 = hw.instance "p3" @Post(clk: %clk: !seq.clock, reset: %reset: i1, y: %m3: i16) -> (z: i16)
  hw.output %y0, %y1, %y2, %y3 : i16, i16, i16, i16
}
