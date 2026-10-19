// RUN: foldsim-opt %s --topological-sort -o %t.ref.mlir
// RUN: foldsim-opt %s --foldsim-sort -o %t.mlir
// RUN: diff %t.ref.mlir %t.mlir
// RUN: FileCheck %s < %t.mlir

// The order of `topological-sort`: uses after their operands where they can
// be, and where loops through registers leave none ready, the first op left
// taken as it is.

// CHECK-LABEL: hw.module @Chain
// CHECK-NEXT: %c1_i8 = hw.constant 1
// CHECK-NEXT: %0 = comb.add %a, %c1_i8
// CHECK-NEXT: %1 = comb.mul %0, %0
// CHECK-NEXT: %2 = comb.xor %1, %a
// CHECK-NEXT: hw.output %2
hw.module @Chain(in %a : i8, out y : i8) {
  %x = comb.xor %m, %a : i8
  %m = comb.mul %s, %s : i8
  %s = comb.add %a, %c1 : i8
  %c1 = hw.constant 1 : i8
  hw.output %x : i8
}

// CHECK-LABEL: hw.module @Loops
// CHECK-NEXT: %c1_i8 = hw.constant 1
// CHECK-NEXT: %0 = comb.xor %q, %r
// CHECK-NEXT: %1 = comb.mux %en, %r, %q
// CHECK-NEXT: %q = seq.firreg %1 clock %clk
// CHECK-NEXT: %r = seq.firreg %2 clock %clk
// CHECK-NEXT: %2 = comb.add %r, %c1_i8
// CHECK-NEXT: hw.output %0
hw.module @Loops(in %clk : !seq.clock, in %en : i1, out y : i8) {
  %y = comb.xor %q, %r : i8
  %qn = comb.mux %en, %r, %q : i8
  %r = seq.firreg %rn clock %clk : i8
  %q = seq.firreg %qn clock %clk : i8
  %rn = comb.add %r, %c1 : i8
  %c1 = hw.constant 1 : i8
  hw.output %y : i8
}

// An op whose region uses a value defined later waits for it; one using
// only its own region's values does not.
// CHECK-LABEL: hw.module @Nested
// CHECK-NEXT: %c2_i8 = hw.constant 2
// CHECK-NEXT: %0 = comb.add %a, %c2_i8
// CHECK-NEXT: sv.ifdef @A {
// CHECK-NEXT:   %w = sv.wire
// CHECK-NEXT:   sv.assign %w, %0
// CHECK-NEXT: }
// CHECK-NEXT: hw.output %0
sv.macro.decl @A
hw.module @Nested(in %a : i8, out y : i8) {
  sv.ifdef @A {
    %w = sv.wire : !hw.inout<i8>
    sv.assign %w, %s : i8
  }
  %s = comb.add %a, %c2 : i8
  %c2 = hw.constant 2 : i8
  hw.output %s : i8
}
