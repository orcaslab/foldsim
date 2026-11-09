// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-slr-split{module=copy map=%S/Inputs/slr-split/two.map parts=2})' 2>%t.err | FileCheck %s
// RUN: FileCheck %s --check-prefix=STATS < %t.err

// A copy split in two at its registers: the fold's instance counter (and its
// increment) replicated into both parts, part 0's register `a` read by part
// 1 across the cut, part 1's registered memory read by the copy's output.
// No register is added: each part's flops are the copy's.

// CHECK-LABEL: hw.module private @copy_p0(in %clk : !seq.clock, in %x : i4, out a : i4, out inst : i4)
// CHECK:         %c1_i4 = hw.constant 1 : i4
// CHECK:         %inst = seq.firreg %[[N0:.+]] clock %clk : i4
// CHECK:         %[[N0]] = comb.add %inst, %c1_i4 : i4
// CHECK:         %[[X0:.+]] = comb.xor %x, %inst : i4
// CHECK:         %a = seq.firreg %[[X0]] clock %clk : i4
// CHECK:         hw.output %a, %inst : i4, i4
// CHECK-LABEL: hw.module private @copy_p1(in %clk : !seq.clock, in %a : i4, in %y : i4, out b : i4, out out : i4)
// CHECK:         %inst = seq.firreg
// CHECK:         %[[S1:.+]] = comb.add %a, %y : i4
// CHECK:         %b = seq.firreg %[[S1]] clock %clk : i4
// CHECK:         %mem = seq.firmem 1, 1
// CHECK:         seq.firmem.write_port %mem[%inst] = %b, clock %clk
// CHECK:         seq.firmem.read_port %mem[%inst], clock %clk
// CHECK:         hw.output %b, %{{.+}} : i4, i4
// CHECK-LABEL: hw.module @copy(
// CHECK:         %fs_split_copy_p0.a, %fs_split_copy_p0.inst = hw.instance "fs_split_copy_p0" @copy_p0(clk: %clk: !seq.clock, x: %x: i4)
// CHECK:         %fs_split_copy_p1.b, %fs_split_copy_p1.out = hw.instance "fs_split_copy_p1" @copy_p1(clk: %clk: !seq.clock, a: %fs_split_copy_p0.a: i4, y: %y: i4)
// CHECK:         hw.output %fs_split_copy_p1.b, %fs_split_copy_p1.out, %fs_split_copy_p0.inst

// STATS: foldsim-slr-split: copy: 2 parts, 1 values (4 bits) cross from a register, 0 ops of logic copied
// STATS: part 0: 5 ops, 2 inputs, 2 outputs
// STATS: part 1: 8 ops, 3 inputs, 2 outputs

hw.module @copy(in %clk : !seq.clock, in %x : i4, in %y : i4, out b : i4, out rd : i4, out inst : i4) {
  %c1_i4 = hw.constant 1 : i4
  %inst = seq.firreg %0 clock %clk : i4
  %0 = comb.add %inst, %c1_i4 : i4
  %1 = comb.xor %x, %inst : i4
  %a = seq.firreg %1 clock %clk : i4
  %2 = comb.add %a, %y : i4
  %b = seq.firreg %2 clock %clk : i4
  %mem = seq.firmem 1, 1, undefined, undefined : <16 x 4>
  seq.firmem.write_port %mem[%inst] = %b, clock %clk : <16 x 4>
  %rd = seq.firmem.read_port %mem[%inst], clock %clk : <16 x 4>
  hw.output %b, %rd, %inst : i4, i4, i4
}
