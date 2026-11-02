// RUN: foldsim-opt %s --foldsim-extract-models="module=Bus models=Buf absorb-logic=8 absorb-exclusive=B*" | FileCheck %s

// Each buffer's input is logic over its own element of a bus (an array_get
// at a constant index, the instances' only difference): the model absorbs the
// logic and takes the element, each instance its own; not the whole bus and
// the index as inputs.
// CHECK-LABEL: hw.module @Buf(in %clk : !seq.clock, in %d : i4, in %d_0 : i4, in %d_1 : i4, out q : i4)
// CHECK:         [[X:%.+]] = comb.xor %d_0, %d_1 : i4
// CHECK:         [[A:%.+]] = comb.add [[X]], %d_1 : i4
// CHECK:         [[M:%.+]] = comb.mul [[A]], [[X]] : i4
// CHECK:         seq.firreg [[M]] clock %clk : i4
// CHECK-NOT:     hw.array_get
// CHECK-LABEL: hw.module @Bus(
// CHECK-SAME:    foldsim.direct_inputs = ["u0_d_0 = bus[7:4]", "u0_d_1 = k", "u1_d_0 = bus[11:8]", "u1_d_1 = k"]
hw.module @Buf(in %clk : !seq.clock, in %d : i4, out q : i4) {
  %q = seq.firreg %d clock %clk : i4
  hw.output %q : i4
}
hw.module @Bus(in %clk : !seq.clock, in %bus : !hw.array<4xi4>, in %k : i4, out z0 : i4, out z1 : i4) {
  %c1 = hw.constant 1 : i2
  %c2 = hw.constant 2 : i2
  %e0 = hw.array_get %bus[%c1] : !hw.array<4xi4>, i2
  %x0 = comb.xor %e0, %k : i4
  %a0 = comb.add %x0, %k : i4
  %d0 = comb.mul %a0, %x0 : i4
  %u0.q = hw.instance "u0" @Buf(clk: %clk: !seq.clock, d: %d0: i4) -> (q: i4)
  %e1 = hw.array_get %bus[%c2] : !hw.array<4xi4>, i2
  %x1 = comb.xor %e1, %k : i4
  %a1 = comb.add %x1, %k : i4
  %d1 = comb.mul %a1, %x1 : i4
  %u1.q = hw.instance "u1" @Buf(clk: %clk: !seq.clock, d: %d1: i4) -> (q: i4)
  hw.output %u0.q, %u1.q : i4, i4
}
