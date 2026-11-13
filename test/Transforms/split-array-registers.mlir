// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-split-array-registers{modules=Sw}),canonicalize)' | FileCheck %s

// A 2 x 2 array register written whole each cycle (an RTL array assigned in a
// loop) and read only at constant indices becomes a register per element, its
// next and reset values the elements of the array's; the nested arrays split
// level by level. The register read at a variable index (`v`) and the module
// not named (`Other`) keep theirs.
// CHECK-LABEL: hw.module @Sw(
// CHECK-DAG:     %s_0_0 = seq.firreg %a clock %clk reset sync %rst, %c0_i8 : i8
// CHECK-DAG:     %s_0_1 = seq.firreg %b clock %clk reset sync %rst, %c0_i8 : i8
// CHECK-DAG:     %s_1_0 = seq.firreg %s_0_0 clock %clk reset sync %rst, %c0_i8 : i8
// CHECK-DAG:     %s_1_1 = seq.firreg %s_0_1 clock %clk reset sync %rst, %c0_i8 : i8
// CHECK-DAG:     %v = seq.firreg {{%.+}} clock %clk : !hw.array<2xi8>
// CHECK-NOT:     !hw.array<2xarray<2xi8>>
// CHECK:         hw.output
// CHECK-LABEL: hw.module @Other(
// CHECK:         seq.firreg {{%.+}} clock %clk : !hw.array<2xi8>
hw.module @Sw(in %clk : !seq.clock, in %rst : i1, in %a : i8, in %b : i8, in %i : i1, out o0 : i8, out o1 : i8, out o2 : i8) {
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %z = hw.aggregate_constant [[0 : i8, 0 : i8], [0 : i8, 0 : i8]] : !hw.array<2xarray<2xi8>>
  %row0 = hw.array_create %b, %a : i8
  %old0 = hw.array_get %s[%c0] : !hw.array<2xarray<2xi8>>, i1
  %next = hw.array_create %old0, %row0 : !hw.array<2xi8>
  %s = seq.firreg %next clock %clk reset sync %rst, %z : !hw.array<2xarray<2xi8>>
  %r1 = hw.array_get %s[%c1] : !hw.array<2xarray<2xi8>>, i1
  %e10 = hw.array_get %r1[%c0] : !hw.array<2xi8>, i1
  %e11 = hw.array_get %r1[%c1] : !hw.array<2xi8>, i1
  %v = seq.firreg %row0 clock %clk : !hw.array<2xi8>
  %vi = hw.array_get %v[%i] : !hw.array<2xi8>, i1
  hw.output %e10, %e11, %vi : i8, i8, i8
}
hw.module @Other(in %clk : !seq.clock, in %a : i8, out o : i8) {
  %c0 = hw.constant 0 : i1
  %w = hw.array_create %a, %a : i8
  %r = seq.firreg %w clock %clk : !hw.array<2xi8>
  %e = hw.array_get %r[%c0] : !hw.array<2xi8>, i1
  hw.output %e : i8
}
