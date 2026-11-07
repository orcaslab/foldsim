// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-replicate-controls{loads=4}))' | FileCheck %s

// The loads of a control are grouped in module order whatever the order of
// its uses (a use list runs newest first): four selects of 4 loads and one
// op that takes the control twice (1 load each, the same position in the
// module) make four groups, the two uses of that op in one, so the register
// becomes four copies, each loading the next group down the module.

// CHECK-LABEL: hw.module @Order
// CHECK: %c = seq.firreg
// CHECK: %c_rep1 = seq.firreg
// CHECK: %c_rep2 = seq.firreg
// CHECK: %c_rep3 = seq.firreg
// CHECK-NOT: %c_rep4
// CHECK: comb.mux %c, %a, %b : i4
// CHECK-NEXT: comb.mux %c_rep1, %a, %b : i4
// CHECK-NEXT: comb.and %c_rep2, %c_rep2 : i1
// CHECK-NEXT: comb.mux %c_rep3, %a, %b : i4

hw.module @Order(in %clk: !seq.clock, in %n: i1, in %a: i4, in %b: i4,
                 out o: i4, out p: i4, out q: i1, out r: i4) {
  %c = seq.firreg %n clock %clk {foldsim.control} : i1
  %ma = comb.mux %c, %a, %b : i4
  %mb = comb.mux %c, %a, %b : i4
  %both = comb.and %c, %c : i1
  %md = comb.mux %c, %a, %b : i4
  hw.output %ma, %mb, %both, %md : i4, i4, i1, i4
}
