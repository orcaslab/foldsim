// RUN: foldsim-opt %s --foldsim-extract-models="module=Shared models=Buf absorb-logic=8 absorb-exclusive=B*" | FileCheck %s --check-prefix=OFF
// RUN: foldsim-opt %s --foldsim-extract-models="module=Shared models=Buf absorb-logic=8 absorb-exclusive=B* absorb-registers" | FileCheck %s

// Each buffer's input is a register of the parent that holds its value unless
// its enable bit is set (its next value reads it): the register reaches that
// input alone, so with `absorb-registers` it moves into the model with its
// select, the enable bit (another per instance) and the data being inputs,
// its clock the model's. `k` is read by an output of the parent too: it stays.
// OFF-LABEL: hw.module @Buf(in %clk : !seq.clock, in %d : i4, in %s : i4, out q : i4)
// CHECK-LABEL: hw.module @Buf(in %clk : !seq.clock, in %d : i4, in %s : i4, in %d_0 : i1, in %d_1 : i4, out q : i4)
// CHECK:         [[M:%.+]] = comb.mux %d_0, %d_1, [[H:%.+]] : i4
// CHECK:         [[H]] = seq.firreg [[M]] clock %clk : i4
// CHECK:         comb.xor [[H]], %s : i4
// CHECK-LABEL: hw.module @Shared(
// CHECK-SAME:    foldsim.direct_inputs = ["u0_d_0 = e[0]", "u0_d_1 = a", "u1_d_0 = e[1]", "u1_d_1 = b"
// CHECK:         %k = seq.firreg
// CHECK:         hw.output %u0_q, %u1_q, %k, %k, %k :
hw.module @Buf(in %clk : !seq.clock, in %d : i4, in %s : i4, out q : i4) {
  %x = comb.xor %d, %s : i4
  %q = seq.firreg %x clock %clk : i4
  hw.output %q : i4
}
hw.module @Shared(in %clk : !seq.clock, in %a : i4, in %b : i4, in %e : i2, out z0 : i4, out z1 : i4, out w : i4) {
  %e0 = comb.extract %e from 0 : (i2) -> i1
  %e1 = comb.extract %e from 1 : (i2) -> i1
  %p0 = comb.mux %e0, %a, %h0 : i4
  %h0 = seq.firreg %p0 clock %clk {name = "h"} : i4
  %k0 = comb.xor %a, %b : i4
  %k = seq.firreg %k0 clock %clk {name = "k"} : i4
  %u0.q = hw.instance "u0" @Buf(clk: %clk: !seq.clock, d: %h0: i4, s: %k: i4) -> (q: i4)
  %p1 = comb.mux %e1, %b, %h1 : i4
  %h1 = seq.firreg %p1 clock %clk {name = "h"} : i4
  %u1.q = hw.instance "u1" @Buf(clk: %clk: !seq.clock, d: %h1: i4, s: %k: i4) -> (q: i4)
  hw.output %u0.q, %u1.q, %k : i4, i4, i4
}
