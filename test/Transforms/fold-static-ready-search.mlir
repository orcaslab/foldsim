// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=S,A,B netlist},foldsim-fold-static{module=Top schedule-only ready-levels=off})" -o /dev/null 2>&1 | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=S,A,B netlist},foldsim-fold-static{module=Top schedule-only})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=AUTO

// Every choice of ready passes leaves a cycle when each pass writes every
// late output it can (Top's ready pass feeding A and reading A's ready back
// for S; A's feeding B and reading B back). The search, from a ready pass for
// every model, moves the outputs that close a cycle to their state pass,
// a parent's first; the report marks them.
// With ready passes by level (auto), splitting Top's ready outputs gives the
// same frame; on a tie the split passes are kept, nothing moved.
// AUTO: foldsim-fold-static: II = 7
// AUTO: Top_model x1 (2 ready passes)
// AUTO-NOT: (moved)
// CHECK: foldsim-fold-static: II = 7
// CHECK: Top_model x1 (ready pass)
// CHECK: late s_r_in @state (moved) <- a0_r a1_r
// CHECK: late s_k_in @state (moved) <- a0_k a1_k
// CHECK: late a0_v_in @ready <- s_q
// CHECK: A x2 (ready pass)
// CHECK: late r @ready <- v_in
// CHECK: late o @ready <- v_in
// CHECK: late k @state (moved) <- b_in
// CHECK: B x2 state gap
// CHECK: late b @state <- o_in
hw.module @S(in %clk : !seq.clock, in %reset : i1, in %r_in : i1, in %k_in : i1, in %x : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c0_i6 = hw.constant 0 : i6
  %rk = comb.concat %c0_i6, %r_in, %k_in : i6, i1, i1
  %s = comb.add %q, %rk : i8
  %n = comb.xor %s, %x : i8
  %q = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  hw.output %q : i8
}
hw.module @A(in %clk : !seq.clock, in %reset : i1, in %v_in : i8, in %b_in : i8, out r : i1, out o : i8, out k : i1) {
  %c0 = hw.constant 0 : i8
  %v0 = comb.extract %v_in from 0 : (i8) -> i1
  %a0 = comb.extract %areg from 0 : (i8) -> i1
  %r = comb.xor %v0, %a0 : i1
  %o = comb.add %v_in, %areg : i8
  %b1 = comb.extract %b_in from 1 : (i8) -> i1
  %a1 = comb.extract %areg from 1 : (i8) -> i1
  %k = comb.and %b1, %a1 : i1
  %vb = comb.xor %v_in, %b_in : i8
  %an = comb.add %areg, %vb : i8
  %areg = seq.firreg %an clock %clk reset sync %reset, %c0 : i8
  hw.output %r, %o, %k : i1, i8, i1
}
hw.module @B(in %clk : !seq.clock, in %reset : i1, in %o_in : i8, out b : i8) {
  %c0 = hw.constant 0 : i8
  %bn = comb.add %breg, %o_in : i8
  %breg = seq.firreg %bn clock %clk reset sync %reset, %c0 : i8
  %b = comb.xor %o_in, %breg : i8
  hw.output %b : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z0 : i8, out z1 : i8) {
  %c0 = hw.constant 0 : i8
  %c1 = hw.constant 1 : i8
  %c0_i7 = hw.constant 0 : i7
  %s.q = hw.instance "s" @S(clk: %clk: !seq.clock, reset: %reset: i1, r_in: %r_out: i1, k_in: %k_out: i1, x: %x: i8) -> (q: i8)
  %v = comb.xor %treg, %s.q : i8
  %v1 = comb.add %v, %c1 : i8
  %a0.r, %a0.o, %a0.k = hw.instance "a0" @A(clk: %clk: !seq.clock, reset: %reset: i1, v_in: %v: i8, b_in: %b0.b: i8) -> (r: i1, o: i8, k: i1)
  %a1.r, %a1.o, %a1.k = hw.instance "a1" @A(clk: %clk: !seq.clock, reset: %reset: i1, v_in: %v1: i8, b_in: %b1.b: i8) -> (r: i1, o: i8, k: i1)
  %b0.b = hw.instance "b0" @B(clk: %clk: !seq.clock, reset: %reset: i1, o_in: %a0.o: i8) -> (b: i8)
  %b1.b = hw.instance "b1" @B(clk: %clk: !seq.clock, reset: %reset: i1, o_in: %a1.o: i8) -> (b: i8)
  %r_out = comb.or %a0.r, %a1.r : i1
  %k_out = comb.xor %a0.k, %a1.k : i1
  %rz = comb.concat %c0_i7, %r_out : i7, i1
  %tn = comb.add %treg, %rz, %c1 : i8
  %treg = seq.firreg %tn clock %clk reset sync %reset, %c0 : i8
  hw.output %s.q, %treg : i8, i8
}
