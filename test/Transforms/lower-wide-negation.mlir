// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-lower-wide-negation{min-width=4}))' | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-lower-wide-negation))' | FileCheck %s --check-prefix=DEFAULT

// CHECK-LABEL: hw.module @LowestSetBit
// DEFAULT-LABEL: hw.module @LowestSetBit
hw.module @LowestSetBit(in %x : i5, in %y : i3, out onehot : i5, out neg3 : i3) {
  %c0_i5 = hw.constant 0 : i5
  %c0_i3 = hw.constant 0 : i3
  // The prefix OR in three stages (shifts 1, 2, 4), then x ^ (prefix << 1).
  // CHECK:      %[[L1:.+]] = comb.extract %x from 0 : (i5) -> i4
  // CHECK:      %[[S1:.+]] = comb.concat %[[L1]], %{{false.*}} : i4, i1
  // CHECK:      %[[P1:.+]] = comb.or bin %x, %[[S1]] : i5
  // CHECK:      %[[L2:.+]] = comb.extract %[[P1]] from 0 : (i5) -> i3
  // CHECK:      %[[S2:.+]] = comb.concat %[[L2]], %c0_i2 : i3, i2
  // CHECK:      %[[P2:.+]] = comb.or bin %[[P1]], %[[S2]] : i5
  // CHECK:      %[[L4:.+]] = comb.extract %[[P2]] from 0 : (i5) -> i1
  // CHECK:      %[[S4:.+]] = comb.concat %[[L4]], %c0_i4 : i1, i4
  // CHECK:      %[[P4:.+]] = comb.or bin %[[P2]], %[[S4]] : i5
  // CHECK:      %[[LB:.+]] = comb.extract %[[P4]] from 0 : (i5) -> i4
  // CHECK:      %[[B:.+]] = comb.concat %[[LB]], %{{false.*}} : i4, i1
  // CHECK:      %[[NEG:.+]] = comb.xor bin %x, %[[B]] : i5
  // CHECK:      %[[OH:.+]] = comb.and %x, %[[NEG]] : i5
  // CHECK-NOT:  comb.sub %c0_i5
  // Narrower than min-width: stays a subtraction.
  // CHECK:      comb.sub %c0_i3, %y : i3
  // Below the default width (65) nothing changes.
  // DEFAULT:    comb.sub %c0_i5, %x : i5
  // DEFAULT:    comb.sub %c0_i3, %y : i3
  %neg = comb.sub %c0_i5, %x : i5
  %oh = comb.and %x, %neg : i5
  %neg3 = comb.sub %c0_i3, %y : i3
  hw.output %oh, %neg3 : i5, i3
}
