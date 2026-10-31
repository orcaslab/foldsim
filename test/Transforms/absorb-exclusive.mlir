// RUN: foldsim-opt %s --foldsim-extract-models="module=Shared models=Buf absorb-logic=3" | FileCheck %s --check-prefix=LIMIT
// RUN: foldsim-opt %s --foldsim-extract-models="module=Shared models=Buf absorb-logic=3 absorb-exclusive=B*" | FileCheck %s

// Each buffer's input is a select by `s`, which both buffers' selects read
// and which takes more ops than the limit: no logic is absorbed without
// `absorb-exclusive`; with it, each buffer's own logic is (the select and its
// add), `s` and the select's data being inputs of the model. The add's
// constant differs between the instances: each instance's is an input too,
// which the instance's constant drives (a channel of no bits).
// LIMIT-LABEL: hw.module @Buf(in %clk : !seq.clock, in %d : i4, out q : i4)
// CHECK-LABEL: hw.module @Buf(in %clk : !seq.clock, in %d : i4, in %d_0 : i1, in %d_1 : i4, in %d_2 : i4, in %d_3 : i4, out q : i4)
// CHECK:         [[A:%.+]] = comb.add %d_2, %d_3 : i4
// CHECK:         [[M:%.+]] = comb.mux %d_0, %d_1, [[A]] : i4
// CHECK:         seq.firreg [[M]] clock %clk : i4
// CHECK-LABEL: hw.module @Shared(
// CHECK-SAME:    out u0_d_0 : i1, out u0_d_3 : i0, out u1_d_0 : i1, out u1_d_3 : i0)
// CHECK-SAME:    foldsim.direct_inputs = ["u0_d_1 = a", "u0_d_2 = b", "u1_d_1 = b", "u1_d_2 = c"]
// CHECK:         [[S:%.+]] = comb.icmp eq
// CHECK:         hw.output %u0_q, %u1_q, [[S]], %{{.+}}, [[S]], %{{.+}} : i4, i4, i1, i0, i1, i0
hw.module @Buf(in %clk : !seq.clock, in %d : i4, out q : i4) {
  %q = seq.firreg %d clock %clk : i4
  hw.output %q : i4
}
hw.module @Shared(in %clk : !seq.clock, in %a : i4, in %b : i4, in %c : i4, out z0 : i4, out z1 : i4) {
  %x = comb.xor %a, %b : i4
  %y = comb.and %x, %c : i4
  %w = comb.or %y, %b : i4
  %s = comb.icmp eq %w, %a : i4
  %c1 = hw.constant 1 : i4
  %c2 = hw.constant 2 : i4
  %p0 = comb.add %b, %c1 : i4
  %d0 = comb.mux %s, %a, %p0 : i4
  %u0.q = hw.instance "u0" @Buf(clk: %clk: !seq.clock, d: %d0: i4) -> (q: i4)
  %p1 = comb.add %c, %c2 : i4
  %d1 = comb.mux %s, %b, %p1 : i4
  %u1.q = hw.instance "u1" @Buf(clk: %clk: !seq.clock, d: %d1: i4) -> (q: i4)
  hw.output %u0.q, %u1.q : i4, i4
}
