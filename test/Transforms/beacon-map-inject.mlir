// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=versal})' | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=versal},foldsim-estimate-timing{output=- report= top=Top})' -o /dev/null | FileCheck %s --check-prefix=LEVELS

// An array written at a dynamic index becomes a select per element, each
// under an equality of the index with the element's own: those compares are
// LUT logic like every other equality (Vivado decodes the index in LUTs), not
// comparators left word-level for carry chains.
// CHECK-LABEL: hw.module @Top
// CHECK-NOT: comb.icmp
// CHECK: hw.output
// LEVELS: 1,{{.*}},r.d[0]
hw.module @Top(in %clk : !seq.clock, in %i_in : i2, in %v_in : i16, out o : !hw.array<4xi16>) {
  %i = seq.firreg %i_in clock %clk : i2
  %v = seq.firreg %v_in clock %clk : i16
  %0 = hw.array_inject %r[%i], %v : !hw.array<4xi16>, i2
  %r = seq.firreg %0 clock %clk : !hw.array<4xi16>
  hw.output %r : !hw.array<4xi16>
}
