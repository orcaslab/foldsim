// RUN: foldsim-opt %s --foldsim-fame5="module=Cycle threads=2" | FileCheck %s

// A combinational cycle through some bits of a value, which is no cycle bit
// by bit: `o` is `a` plus one through it, so it waits for `a`; `p` is the
// register plus one the same way, so it fires eagerly.
// CHECK-LABEL: hw.module @Cycle_threaded_Cycle(
// CHECK-SAME:    attributes {foldsim.eager_channels = ["p"]}

// `p`'s next value is the register's next value plus one, through a copy of
// the cycle.
// CHECK: comb.concat {{%.+}}, [[R:%[0-9]+]] : i4, i4
// CHECK: [[R_NEXT:%.+]] = comb.add [[R]], %c1_i4 : i4
// CHECK: %threadIdx = seq.firreg
// CHECK: [[LOW:%.+]] = comb.extract [[NEXT_T:%.+]] from 0 : (i8) -> i4
// CHECK: [[HIGH:%.+]] = comb.add [[LOW]], %c1_i4 : i4
// CHECK: [[NEXT_T]] = comb.concat [[HIGH]], [[R_NEXT]] : i4, i4
// CHECK: [[P_NEXT:%.+]] = comb.extract [[NEXT_T]] from 4 : (i8) -> i4
// CHECK: [[R]] = seq.firmem.read_port %r[%threadIdx]
// CHECK: comb.mux {{%.+}}, [[P_NEXT]], {{%.+}} : i4
hw.module @Cycle(in %clk : !seq.clock, in %a : i4, out o : i4, out p : i4) {
  %c1 = hw.constant 1 : i4
  %s = comb.concat %hi, %a : i4, i4
  %lo = comb.extract %s from 0 : (i8) -> i4
  %hi = comb.add %lo, %c1 : i4
  %o = comb.extract %s from 4 : (i8) -> i4
  %t = comb.concat %thi, %r : i4, i4
  %tlo = comb.extract %t from 0 : (i8) -> i4
  %thi = comb.add %tlo, %c1 : i4
  %p = comb.extract %t from 4 : (i8) -> i4
  %rn = comb.add %r, %c1 : i4
  %r = seq.firreg %rn clock %clk : i4
  hw.output %o, %p : i4, i4
}
