// RUN: foldsim-opt %s --foldsim-lower-equality | FileCheck %s

// CHECK-LABEL: hw.module @Cmp
hw.module @Cmp(in %a : i3, in %b : i3, out eq : i1, out ne : i1, out lt : i1) {
  // CHECK: %[[D:.+]] = comb.xor bin %a, %b : i3
  // CHECK: %[[B0:.+]] = comb.extract %[[D]] from 0 : (i3) -> i1
  // CHECK: %[[B1:.+]] = comb.extract %[[D]] from 1 : (i3) -> i1
  // CHECK: %[[B2:.+]] = comb.extract %[[D]] from 2 : (i3) -> i1
  // CHECK: %[[ANY:.+]] = comb.or bin %[[B0]], %[[B1]], %[[B2]] : i1
  // CHECK: %[[EQ:.+]] = comb.xor bin %[[ANY]], %true : i1
  %eq = comb.icmp eq %a, %b : i3
  // CHECK: %[[ANY2:.+]] = comb.or bin
  %ne = comb.icmp ne %a, %b : i3
  // Order comparisons are carry chains and stay.
  // CHECK: %[[LT:.+]] = comb.icmp ult %a, %b : i3
  %lt = comb.icmp ult %a, %b : i3
  // CHECK: hw.output %[[EQ]], %[[ANY2]], %[[LT]]
  hw.output %eq, %ne, %lt : i1, i1, i1
}
