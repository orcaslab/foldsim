// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-fold-zero-width))' | FileCheck %s

// A zero-width mux carries nothing: it becomes the constant, and the i1 one
// beside it stays.
// CHECK-LABEL: hw.module @Top
// CHECK-NOT: comb.mux {{.*}} : i0
// CHECK: %[[Z:.+]] = hw.constant 0 : i0
// CHECK: comb.mux %c, %a, %b : i1
// CHECK: hw.output %[[Z]], %{{.+}} : i0, i1
hw.module @Top(in %c : i1, in %x : i0, in %y : i0, in %a : i1, in %b : i1,
               out z : i0, out o : i1) {
  %z = comb.mux %c, %x, %y : i0
  %o = comb.mux %c, %a, %b : i1
  hw.output %z, %o : i0, i1
}
