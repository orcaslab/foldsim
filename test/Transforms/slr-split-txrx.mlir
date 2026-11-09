// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-slr-split{module=copy map=%S/Inputs/slr-split/two.map parts=2 crossing=txrx latency=2})' 2>&1 | FileCheck %s

// crossing=txrx: part 0's flop `a` read in part 1 crosses as a send register
// (a copy of `a`, its own) and a receive register in part 1, both
// USER_SLL_REG and DONT_TOUCH: part 1 runs a host cycle late (lag 1). Its
// other inputs (`y`, the instance counter's value) are aligned by a register
// each; its memory (a combinational read, rows by the fold's instance
// counter: elastic) has its write (lag 1) land a host cycle after its read is
// taken (lag 0), within the slack. A memory read through a register (a store
// read ahead of its pass) or addressed by data keeps them in step. Every output is 2 host cycles late (the
// class's split-latency): part 1's `b` by one more register, the read and the
// counter by two.

// CHECK: foldsim-slr-split: crossing=txrx: 1 memories elastic (addressed by the fold's control), 0 in step
// CHECK: foldsim-slr-split: crossing=txrx: 1 send and 1 receive registers (4 bits), 7 alignment registers; outputs 2 host cycles late (the most lag 1)
// CHECK-LABEL: hw.module private @copy_p0(
// CHECK:         %a = seq.firreg %[[X:.+]] clock %clk : i4
// CHECK:         %a_lag1_tx = seq.firreg %[[X]] clock %clk {sv.attributes = [#sv.attribute<"USER_SLL_REG" = "\22TRUE\22">, #sv.attribute<"DONT_TOUCH" = "\22TRUE\22">]} : i4
// CHECK:         %inst_lag2_a0 = seq.firreg %inst
// CHECK:         %inst_lag2_a1 = seq.firreg %inst_lag2_a0
// CHECK:         hw.output %a_lag1_tx, %inst_lag2_a1
// CHECK-LABEL: hw.module private @copy_p1(
// CHECK:         comb.add %a_lag1_rx, %y_lag1_a0 : i4
// CHECK:         seq.firmem.write_port %mem[%inst_lag1_a0] = %b, clock %clk
// CHECK:         seq.firmem.read_port %mem[%inst], clock %clk
// CHECK:         %a_lag1_rx = seq.firreg %a_lag1_tx clock %clk {sv.attributes = [#sv.attribute<"USER_SLL_REG" = "\22TRUE\22">, #sv.attribute<"DONT_TOUCH" = "\22TRUE\22">]} : i4
// CHECK:         %y_lag1_a0 = seq.firreg %y
// CHECK:         %b_lag1_a0 = seq.firreg %b
// CHECK:         hw.output %b_lag1_a0, %split_lag2_a1

hw.module @copy(in %clk : !seq.clock, in %x : i4, in %y : i4, out b : i4, out rd : i4, out inst : i4) {
  %c1_i4 = hw.constant 1 : i4
  %inst = seq.firreg %0 clock %clk : i4
  %0 = comb.add %inst, %c1_i4 : i4
  %1 = comb.xor %x, %inst : i4
  %a = seq.firreg %1 clock %clk : i4
  %2 = comb.add %a, %y : i4
  %b = seq.firreg %2 clock %clk : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <16 x 4>
  seq.firmem.write_port %mem[%inst] = %b, clock %clk : <16 x 4>
  %rd = seq.firmem.read_port %mem[%inst], clock %clk : <16 x 4>
  hw.output %b, %rd, %inst : i4, i4, i4
}
