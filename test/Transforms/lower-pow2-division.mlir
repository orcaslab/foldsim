// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-lower-pow2-division))' | FileCheck %s

// CHECK-LABEL: hw.module @Top
// x / 16: the bits above 4, zero-extended.
// CHECK: %[[Q:.+]] = comb.extract %x from 4 : (i32) -> i28
// CHECK: %[[QX:.+]] = comb.concat %{{.+}}, %[[Q]] : i4, i28
// x % 16: the low 4 bits, zero-extended.
// CHECK: %[[R:.+]] = comb.extract %x from 0 : (i32) -> i4
// CHECK: %[[RX:.+]] = comb.concat %{{.+}}, %[[R]] : i28, i4
// x % 10 stays a divider.
// CHECK: %[[D:.+]] = comb.modu %x, %{{.+}} : i32
// CHECK: hw.output %[[QX]], %[[RX]], %[[D]]
hw.module @Top(in %x : i32, out q : i32, out r : i32, out d : i32) {
  %c16 = hw.constant 16 : i32
  %c10 = hw.constant 10 : i32
  %q = comb.divu %x, %c16 : i32
  %r = comb.modu %x, %c16 : i32
  %d = comb.modu %x, %c10 : i32
  hw.output %q, %r, %d : i32, i32, i32
}
