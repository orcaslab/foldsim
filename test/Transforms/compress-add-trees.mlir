// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-compress-add-trees{min-operands=3}))' | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-compress-add-trees))' | FileCheck %s --check-prefix=DEFAULT

// A pairwise adder tree over four bits (a population count): the two-input
// additions of zero-extended halves cannot wrap, so the tree is the sum of the
// four bits, compressed by a full adder to two rows and one addition.
// CHECK-LABEL: hw.module @Count4
// CHECK-DAG:   %[[B0:.+]] = comb.extract %x from 0 : (i4) -> i1
// CHECK-DAG:   %[[B1:.+]] = comb.extract %x from 1 : (i4) -> i1
// CHECK-DAG:   %[[B2:.+]] = comb.extract %x from 2 : (i4) -> i1
// CHECK-DAG:   %[[B3:.+]] = comb.extract %x from 3 : (i4) -> i1
// CHECK:       %[[X:.+]] = comb.xor bin %[[B3]], %[[B2]] : i1
// CHECK:       %[[S:.+]] = comb.xor bin %[[X]], %[[B0]] : i1
// CHECK:       %[[C:.+]] = comb.or bin
// CHECK:       %[[R0:.+]] = comb.concat %{{false.*}}, %[[C]], %[[S]] : i1, i1, i1
// CHECK:       %[[R1:.+]] = comb.concat %{{false.*}}, %{{false.*}}, %[[B1]] : i1, i1, i1
// CHECK:       %[[N:.+]] = comb.add bin %[[R0]], %[[R1]] : i3
// CHECK-NOT:   comb.add
// CHECK:       hw.output %[[N]] : i3
// Below the default (64 addends) nothing changes.
// DEFAULT-LABEL: hw.module @Count4
// DEFAULT-COUNT-3: comb.add %
// DEFAULT-NOT: comb.xor
hw.module @Count4(in %x : i4, out n : i3) {
  %false = hw.constant false
  %b0 = comb.extract %x from 0 : (i4) -> i1
  %b1 = comb.extract %x from 1 : (i4) -> i1
  %b2 = comb.extract %x from 2 : (i4) -> i1
  %b3 = comb.extract %x from 3 : (i4) -> i1
  %e0 = comb.concat %false, %b0 : i1, i1
  %e1 = comb.concat %false, %b1 : i1, i1
  %e2 = comb.concat %false, %b2 : i1, i1
  %e3 = comb.concat %false, %b3 : i1, i1
  %s01 = comb.add %e0, %e1 : i2
  %s23 = comb.add %e2, %e3 : i2
  %z01 = comb.concat %false, %s01 : i1, i2
  %z23 = comb.concat %false, %s23 : i1, i2
  %n = comb.add %z01, %z23 : i3
  hw.output %n : i3
}

// An inner addition that can wrap (two full 2-bit values in 2 bits) is an
// addend as it is, never split into its operands.
// CHECK-LABEL: hw.module @Wraps
// CHECK:       %[[W:.+]] = comb.add %a, %b : i2
// CHECK-DAG:   comb.extract %[[W]] from 0 : (i2) -> i1
// CHECK-DAG:   comb.extract %[[W]] from 1 : (i2) -> i1
// CHECK:       comb.xor bin %d, %c : i1
// CHECK:       comb.add bin %{{.+}}, %{{.+}} : i3
hw.module @Wraps(in %a : i2, in %b : i2, in %c : i1, in %d : i1, out n : i3) {
  %false = hw.constant false
  %c0_i2 = hw.constant 0 : i2
  %s = comb.add %a, %b : i2
  %zs = comb.concat %false, %s : i1, i2
  %zc = comb.concat %c0_i2, %c : i2, i1
  %zd = comb.concat %c0_i2, %d : i2, i1
  %n = comb.add %zs, %zc, %zd : i3
  hw.output %n : i3
}

// An inner addition with another use stays, an addend of the tree.
// CHECK-LABEL: hw.module @Shared
// CHECK:       %[[S01:.+]] = comb.add %{{.+}}, %{{.+}} : i2
// CHECK:       comb.add bin %{{.+}}, %{{.+}} : i3
// CHECK:       hw.output %{{.+}}, %[[S01]] : i3, i2
hw.module @Shared(in %x : i4, out n : i3, out m : i2) {
  %false = hw.constant false
  %b0 = comb.extract %x from 0 : (i4) -> i1
  %b1 = comb.extract %x from 1 : (i4) -> i1
  %b2 = comb.extract %x from 2 : (i4) -> i1
  %b3 = comb.extract %x from 3 : (i4) -> i1
  %e0 = comb.concat %false, %b0 : i1, i1
  %e1 = comb.concat %false, %b1 : i1, i1
  %e2 = comb.concat %false, %b2 : i1, i1
  %e3 = comb.concat %false, %b3 : i1, i1
  %s01 = comb.add %e0, %e1 : i2
  %s23 = comb.add %e2, %e3 : i2
  %z01 = comb.concat %false, %s01 : i1, i2
  %z23 = comb.concat %false, %s23 : i1, i2
  %n = comb.add %z01, %z23 : i3
  hw.output %n, %s01 : i3, i2
}
