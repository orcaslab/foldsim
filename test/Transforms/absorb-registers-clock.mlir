// RUN: foldsim-opt %s --foldsim-extract-models="module=P models=Ip absorb-logic=8 absorb-exclusive=Ip" > %t.a
// RUN: foldsim-opt %s --foldsim-extract-models="module=P models=Ip absorb-logic=8 absorb-exclusive=Ip absorb-registers" > %t.b
// RUN: diff %t.a %t.b
// RUN: FileCheck %s < %t.b
// A model without a clock port of its own (a vendor IP's i1 clock) whose
// clock input is logic over the parent's clock: with absorb-registers, logic
// that moves no register takes the clock as before (a 64-SM fold's FMA lanes
// were left unabsorbed, then refused by the fold, when it did not).
// CHECK-LABEL: hw.module @Ip(
// CHECK:         seq.from_clock
hw.module @Ip(in %c : i1, in %d : i4, out q : i4) {
  %0 = comb.replicate %c : (i1) -> i4
  %1 = comb.and %0, %d : i4
  hw.output %1 : i4
}
hw.module @P(in %clk : !seq.clock, in %a : i4, in %b : i4, out z0 : i4, out z1 : i4) {
  %c = seq.from_clock %clk
  %u0.q = hw.instance "u0" @Ip(c: %c: i1, d: %a: i4) -> (q: i4)
  %u1.q = hw.instance "u1" @Ip(c: %c: i1, d: %b: i4) -> (q: i4)
  hw.output %u0.q, %u1.q : i4, i4
}
