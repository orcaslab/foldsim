// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-slr-split{module=copy map=%S/Inputs/slr-split/stages.map parts=3})' 2>&1 | FileCheck %s

// A 3-way split of a staged datapath at its stage registers (the stage
// boundary is the SLR boundary): stage 1's register `st1` and stage 2's
// `st2` are the crossings, in a chain part 0 -> part 1 -> part 2, and the
// stage-position counter `pos` (a function of itself and the fold's strobe
// alone) is replicated into each part that reads it. No register is added:
// every part has its own stage's registers only, so the latency is the
// copy's.

// CHECK: foldsim-slr-split: copy: 3 parts, 2 values (32 bits) cross from a register, 0 ops of logic copied
// CHECK-LABEL: hw.module private @copy_p0(
// CHECK:         %st1 = seq.firreg
// CHECK-NOT:     seq.firreg
// CHECK:         hw.output %st1
// CHECK-LABEL: hw.module private @copy_p1(in %clk : !seq.clock, in %st1 : i16
// CHECK:         %pos = seq.firreg
// CHECK:         %st2 = seq.firreg
// CHECK-NOT:     seq.firreg
// CHECK:         hw.output %st2
// CHECK-LABEL: hw.module private @copy_p2(in %clk : !seq.clock, in %st2 : i16
// CHECK:         %pos = seq.firreg
// CHECK:         %q = seq.firreg
// CHECK-NOT:     seq.firreg
// CHECK-LABEL: hw.module @copy(
// CHECK:         hw.instance "fs_split_copy_p0"
// CHECK:         hw.instance "fs_split_copy_p1" @copy_p1({{.*}}st1: %fs_split_copy_p0.st1: i16
// CHECK:         hw.instance "fs_split_copy_p2" @copy_p2({{.*}}st2: %fs_split_copy_p1.st2: i16

hw.module @copy(in %clk : !seq.clock, in %state : i1, in %a : i16, in %b : i16, out q : i16) {
  %c1_i2 = hw.constant 1 : i2
  %pos = seq.firreg %0 clock %clk : i2
  %0 = comb.add %pos, %c1_i2 : i2
  %1 = comb.mul %a, %b : i16
  %st1 = seq.firreg %1 clock %clk : i16
  %2 = comb.concat %pos, %pos, %pos, %pos, %pos, %pos, %pos, %pos : i2, i2, i2, i2, i2, i2, i2, i2
  %3 = comb.add %st1, %2 : i16
  %st2 = seq.firreg %3 clock %clk : i16
  %4 = comb.xor %st2, %2 : i16
  %5 = comb.mux %state, %4, %q : i16
  %q = seq.firreg %5 clock %clk : i16
  hw.output %q : i16
}
