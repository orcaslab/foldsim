// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-lower-constant-division))' | FileCheck %s

// CHECK-LABEL: hw.module @Top
// x / 16: the bits above 4, zero-extended.
// CHECK: %[[Q:.+]] = comb.extract %x from 4 : (i32) -> i28
// CHECK: %[[QX:.+]] = comb.concat %{{.+}}, %[[Q]] : i4, i28
// x % 16: the low 4 bits, zero-extended.
// CHECK: %[[R:.+]] = comb.extract %x from 0 : (i32) -> i4
// CHECK: %[[RX:.+]] = comb.concat %{{.+}}, %[[R]] : i28, i4
// x % 10 stays a divider: x has 32 significant bits.
// CHECK: %[[D:.+]] = comb.modu %x, %{{.+}} : i32
// y / 3, y a zero-extended i2: a table over y's two bits, 0 0 0 1.
// CHECK: %[[ONE:.+]] = hw.constant 1 : i32
// CHECK: %[[Y0:.+]] = comb.extract %{{.+}} from 0 : (i32) -> i1
// CHECK: %[[T:.+]] = comb.mux bin %[[Y0]], %[[ONE]], %{{.+}} : i32
// CHECK: %[[Y1:.+]] = comb.extract %{{.+}} from 1 : (i32) -> i1
// CHECK: %[[S:.+]] = comb.mux bin %[[Y1]], %[[T]], %{{.+}} : i32
// CHECK: hw.output %[[QX]], %[[RX]], %[[D]], %[[S]]
hw.module @Top(in %x : i32, in %y : i2, out q : i32, out r : i32, out d : i32,
               out t : i32) {
  %c16 = hw.constant 16 : i32
  %c10 = hw.constant 10 : i32
  %c3 = hw.constant 3 : i32
  %c0_i30 = hw.constant 0 : i30
  %q = comb.divu %x, %c16 : i32
  %r = comb.modu %x, %c16 : i32
  %d = comb.modu %x, %c10 : i32
  %yz = comb.concat %c0_i30, %y : i30, i2
  %t = comb.divu %yz, %c3 : i32
  hw.output %q, %r, %d, %t : i32, i32, i32, i32
}
