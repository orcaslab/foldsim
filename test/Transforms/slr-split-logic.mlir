// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-slr-split{module=copy map=%S/Inputs/slr-split/logic.map parts=2})' | FileCheck %s
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-slr-split{module=copy map=%S/Inputs/slr-split/logic.map parts=2 copy-logic=false})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOCOPY

// A cut through logic: part 1 reads part 0's xor, not a register. The xor
// (with its operands' registers staying in part 0, read across the cut) is
// copied into part 1, so the crossing still leaves part 0 from registers:
// `r` and `s` cross, the xor's result never does. Without copy-logic it is
// an error.

// CHECK-LABEL: hw.module private @copy_p0(
// CHECK:         %[[X0:.+]] = comb.xor %r, %s : i8
// CHECK:         %t = seq.firreg %[[X0]]
// CHECK:         hw.output %r, %s, %t : i8, i8, i8
// CHECK-LABEL: hw.module private @copy_p1(in %r : i8, in %s : i8, in %clk : !seq.clock, out u : i8)
// CHECK:         %[[X1:.+]] = comb.xor %r, %s : i8
// CHECK:         %u = seq.firreg %[[X1]]
// NOCOPY: error: slr-split: a value of part 0 read in part 1 is not from a register (a combinational value)

hw.module @copy(in %clk : !seq.clock, in %a : i8, in %b : i8, out t : i8, out u : i8) {
  %r = seq.firreg %a clock %clk : i8
  %s = seq.firreg %b clock %clk : i8
  %0 = comb.xor %r, %s : i8
  %t = seq.firreg %0 clock %clk : i8
  %u = seq.firreg %0 clock %clk : i8
  hw.output %t, %u : i8, i8
}
