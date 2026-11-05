// RUN: foldsim-opt %s --foldsim-extract-models="module=Seq models=Port absorb-logic=8" | FileCheck %s

// Both ports take one control the parent computes once from 16 bits (`go`):
// absorbed, each port would read the 16 bits and compute it again, so it
// stays in the parent. The other input is each port's own logic (an invert of
// its own bit), absorbed as before.
// CHECK-LABEL: hw.module @Port(in %clk : !seq.clock, in %go : i1, in %d : i1, in %d_0 : i1, out q : i1)
// CHECK:         [[N:%.+]] = comb.xor %d_0, %true : i1
// CHECK:         [[A:%.+]] = comb.and %go, [[N]] : i1
// CHECK-LABEL: hw.module @Seq(
// CHECK:         [[GO:%.+]] = comb.icmp eq %a, %b : i8
// CHECK:         hw.output {{.*}}[[GO]]
hw.module @Port(in %clk : !seq.clock, in %go : i1, in %d : i1, out q : i1) {
  %x = comb.and %go, %d : i1
  %q = seq.firreg %x clock %clk : i1
  hw.output %q : i1
}
hw.module @Seq(in %clk : !seq.clock, in %a : i8, in %b : i8, in %e0 : i1, in %e1 : i1, out z0 : i1, out z1 : i1) {
  %go = comb.icmp eq %a, %b : i8
  %true = hw.constant true
  %n0 = comb.xor %e0, %true : i1
  %n1 = comb.xor %e1, %true : i1
  %u0.q = hw.instance "u0" @Port(clk: %clk: !seq.clock, go: %go: i1, d: %n0: i1) -> (q: i1)
  %u1.q = hw.instance "u1" @Port(clk: %clk: !seq.clock, go: %go: i1, d: %n1: i1) -> (q: i1)
  hw.output %u0.q, %u1.q : i1, i1
}
