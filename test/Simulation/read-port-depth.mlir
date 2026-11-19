// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: firtool --format=mlir %s -o %t/target.sv
// Memories addressed by the index rather than its complement, and the deep
// part of the read address on registers alone computed a host cycle ahead,
// with registered state read ahead, in regions.
// RUN: mkdir -p %t/a && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state unreverse-addresses read-port-depth=4" -o %t/a/sim.mlir 2> %t/a/remarks.txt
// RUN: FileCheck %s --check-prefix=REMARK < %t/a/remarks.txt
// RUN: FileCheck %s --check-prefix=AHEAD < %t/a/sim.mlir
// RUN: firtool --format=mlir %t/a/sim.mlir -o %t/a/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/a/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/a/sim.sv > %t/a/build.log 2>&1
// RUN: %t/a/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/a/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// The same with the image flow's state options: state written two host
// cycles late and its cones cut, staged memory writes and tokens, compact
// staging.
// RUN: mkdir -p %t/f && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state commit-select state-stages=2 pipeline-depth=3 staged-ram-writes staged-tokens compact-staging unreverse-addresses read-port-depth=4" -o %t/f/sim.mlir 2> %t/f/remarks.txt
// RUN: FileCheck %s --check-prefix=REMARK < %t/f/remarks.txt
// RUN: firtool --format=mlir %t/f/sim.mlir -o %t/f/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/f/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/f/sim.sv > %t/f/build.log 2>&1
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// The same, each model scheduled in turn rather than by readiness.
// RUN: mkdir -p %t/i && foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-depth=3 staged-ram-writes staged-tokens compact-staging unreverse-addresses read-port-depth=4" -o %t/i/sim.mlir 2> /dev/null
// RUN: firtool --format=mlir %t/i/sim.mlir -o %t/i/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/i/obj %S/Inputs/tb_diffmem.sv %t/target.sv %t/i/sim.sv > %t/i/build.log 2>&1
// RUN: %t/i/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/i/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// Without the options, the memory keeps its reversed addresses and nothing
// is computed ahead.
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=MemModel,Post netlist" --foldsim-compose="module=Top region-loads=1 registered-state" 2>&1 | FileCheck %s --check-prefix=PLAIN

// Four instances of a model with a memory every port of which takes the
// index's complement (as an array declared [0:N-1] imports), read a cycle
// late at a row picked by a hash of two registers (eight levels of adds and
// xors on registers alone) and an input, the registers updated from the
// reads and the write data; each feeds an instance of a second class;
// against the unthreaded target (tb_diffmem.sv), reset at the start and in
// mid-run.
// CHECK: MATCH 4 outputs
// The hash's low bits, three, computed a host cycle ahead from the
// registers' next values: the port's own host cycle only xors the input in.
// REMARK: remark: read-port-depth: ram address of {{[0-9]+}} estimated levels: 1 values (3 bits) computed a host cycle ahead
// AHEAD: %ram_ra_ahead = seq.firreg {{%.+}} clock %hostClock : i3
// PLAIN-NOT: read-port-depth
// PLAIN-NOT: _ahead
// PLAIN: comb.sub %c-1_i3
hw.module private @MemModel(in %clk : !seq.clock, in %reset : i1, in %we : i1, in %waddr : i3, in %wdata : i8, in %raddr : i3, in %en : i1, out y : i16) {
  %true = hw.constant true
  %ones = hw.constant -1 : i3
  %c0_i8 = hw.constant 0 : i8
  %c5_i8 = hw.constant 5 : i8
  %c29_i8 = hw.constant 29 : i8
  %ram = seq.firmem 1, 1, old, undefined : <8 x 8>
  %h1 = comb.add %p, %q : i8
  %h2 = comb.xor %h1, %c29_i8 : i8
  %h3 = comb.add %h2, %p : i8
  %h4 = comb.xor %h3, %q : i8
  %h5 = comb.add %h4, %c5_i8 : i8
  %hl = comb.extract %h5 from 0 : (i8) -> i3
  %ri = comb.xor %hl, %raddr : i3
  %ra = comb.sub %ones, %ri : i3
  %r = seq.firmem.read_port %ram[%ra], clock %clk enable %true : <8 x 8>
  %wa = comb.sub %ones, %waddr : i3
  seq.firmem.write_port %ram[%wa] = %wdata, clock %clk enable %we : <8 x 8>
  %pn0 = comb.add %p, %r : i8
  %pn1 = comb.mux %en, %pn0, %p : i8
  %pn = comb.mux %reset, %c0_i8, %pn1 : i8
  %p = seq.firreg %pn clock %clk : i8
  %qn0 = comb.xor %q, %wdata : i8
  %qn = comb.mux %reset, %c0_i8, %qn0 : i8
  %q = seq.firreg %qn clock %clk : i8
  %r16 = comb.concat %c0_i8, %r : i8, i8
  %pq = comb.concat %p, %q : i8, i8
  %y = comb.xor %pq, %r16 : i16
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
