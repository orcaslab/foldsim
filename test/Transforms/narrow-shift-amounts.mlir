// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-narrow-shift-amounts))' | FileCheck %s

// A shift of 512 bits by a 512-bit amount (as a pass split captures it): the
// amount's low 9 bits shift, an OR of the other 503 says out of range.
// CHECK-LABEL: hw.module @wide
// CHECK-DAG: %[[LO:.+]] = comb.extract %a from 0 : (i512) -> i9
// CHECK-DAG: %[[HI:.+]] = comb.extract %a from 9 : (i512) -> i503
// CHECK-DAG: %[[OUT:.+]] = comb.icmp ne %[[HI]], %{{.+}} : i503
// CHECK-DAG: %[[NARROW:.+]] = comb.concat %{{.+}}, %[[LO]] : i503, i9
// CHECK-DAG: %[[SH:.+]] = comb.shru %x, %[[NARROW]] : i512
// CHECK: comb.mux %[[OUT]], %{{.+}}, %[[SH]] : i512
hw.module @wide(in %x : i512, in %a : i512, out y : i512) {
  %0 = comb.shru %x, %a : i512
  hw.output %0 : i512
}

// An arithmetic shift past the width fills with the sign.
// CHECK-LABEL: hw.module @signed
// CHECK: %[[SH:.+]] = comb.shrs
// CHECK: %[[SIGN:.+]] = comb.extract %x from 63 : (i64) -> i1
// CHECK: %[[FILL:.+]] = comb.replicate %[[SIGN]] : (i1) -> i64
// CHECK: comb.mux %{{.+}}, %[[FILL]], %[[SH]] : i64
hw.module @signed(in %x : i64, in %a : i64, out y : i64) {
  %0 = comb.shrs %x, %a : i64
  hw.output %0 : i64
}

// Narrow shifts, constant amounts and zero-extended amounts stay.
// CHECK-LABEL: hw.module @kept
// CHECK-NOT: comb.mux
// CHECK: hw.output
hw.module @kept(in %x : i16, in %a : i16, in %b : i64, in %c : i6, out y : i16, out z : i64, out w : i64) {
  %0 = comb.shl %x, %a : i16
  %k = hw.constant 3 : i64
  %1 = comb.shl %b, %k : i64
  %z58 = hw.constant 0 : i58
  %e = comb.concat %z58, %c : i58, i6
  %2 = comb.shl %b, %e : i64
  hw.output %0, %1, %2 : i16, i64, i64
}
