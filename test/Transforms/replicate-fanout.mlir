// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-replicate-controls{loads=96 fanout=8 fanout-logic}))' | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-replicate-controls{loads=96 fanout=8}))' | FileCheck %s --check-prefix=NOLOGIC
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-replicate-controls{loads=96}))' | FileCheck %s --check-prefix=OFF

// `fanout`: unmarked registers too. %sel, a read select of a 16-bit read
// register, drives 16 loads: the select becomes two of 8 bits and %sel two
// replicas. %flush drives, through one-LUT logic (%clr = flush & ~hold), the
// synchronous resets of three 8-bit registers (24 loads): with
// `fanout-logic` it becomes three replicas, each with its own copy of %clr
// (8 loads each); without, the loads past %clr are not its own (one load:
// left alone). %small (2 loads) stays one. Replicas and their originals get
// DONT_TOUCH; the logic copies compute the same function of equal values.

// CHECK-LABEL: hw.module @Copy
// CHECK: %sel = seq.firreg %next clock %clk {sv.attributes = [#sv.attribute<"DONT_TOUCH" = "\22true\22">]} : i1
// CHECK: %sel_rep1 = seq.firreg %next clock %clk
// CHECK-NOT: %sel_rep2
// CHECK-DAG: comb.mux %sel, {{.+}} : i8
// CHECK-DAG: comb.mux %sel_rep1, {{.+}} : i8
// CHECK: %flush = seq.firreg %fnext clock %clk {sv.attributes = [#sv.attribute<"DONT_TOUCH"
// CHECK: %flush_rep1 = seq.firreg %fnext
// CHECK: %flush_rep2 = seq.firreg %fnext
// CHECK-NOT: %flush_rep3
// CHECK-DAG: comb.and %flush_rep1,
// CHECK-DAG: comb.and %flush_rep2,
// CHECK-DAG: %small = seq.firreg %next clock %clk : i1

// NOLOGIC: %sel_rep1 = seq.firreg
// NOLOGIC-NOT: %flush_rep

// OFF-NOT: _rep
// OFF-NOT: DONT_TOUCH

hw.module @Copy(in %clk: !seq.clock, in %next: i1, in %fnext: i1, in %hold: i1,
                in %a: i16, in %b: i16, in %d0: i8, in %d1: i8, in %d2: i8,
                out o: i16, out r0: i8, out r1: i8, out r2: i8, out s: i1) {
  %true = hw.constant true
  %z = hw.constant 0 : i8
  %sel = seq.firreg %next clock %clk : i1
  %choice = comb.mux %sel, %a, %b : i16
  %read = seq.firreg %choice clock %clk : i16
  %flush = seq.firreg %fnext clock %clk : i1
  %nh = comb.xor %hold, %true : i1
  %clr = comb.and %flush, %nh : i1
  %r0 = seq.firreg %d0 clock %clk reset sync %clr, %z : i8
  %r1 = seq.firreg %d1 clock %clk reset sync %clr, %z : i8
  %r2 = seq.firreg %d2 clock %clk reset sync %clr, %z : i8
  %small = seq.firreg %next clock %clk : i1
  %sm = comb.and %small, %hold : i1
  %sm2 = comb.or %small, %hold : i1
  %s = comb.xor %sm, %sm2 : i1
  hw.output %read, %r0, %r1, %r2, %s : i16, i8, i8, i8, i1
}
