// fold_model.mlir with deeper layers (static-fold-stage-levels.mlir): a chain
// of ops in three layers, so that a stage of a level or two cuts them.
// The model of the folding simulation tests: a pipeline of four stages on
// an enable, one of four stages gated by the valid of each stage, a black
// box of four cycles with a pipeline contract, and an accumulator of the
// first pipeline's output, which is not a pipeline.
hw.module.extern @XBox(in %clk : !seq.clock, in %ce : i1, in %d : i8, in %d_valid : i1, out q : i8, out q_valid : i1) attributes {sv.attributes = [#sv.attribute<"black_box">, #sv.attribute<"fixed_latency" = "4">, #sv.attribute<"clock" = "\22clk\22">, #sv.attribute<"clock_enable" = "\22ce\22">, #sv.attribute<"data_in" = "\22d\22">, #sv.attribute<"data_out" = "\22q\22">, #sv.attribute<"tie" = "\22d_valid=1\22">, #sv.attribute<"pure_pipeline">]}
hw.module private @Model(in %clk : !seq.clock, in %reset : i1, in %a : i8, in %en : i1, in %v : i1, out y : i32) {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i3 = hw.constant 0 : i3
  %c0 = hw.constant 0 : i8
  %c1 = hw.constant 1 : i8
  %c3 = hw.constant 3 : i8
  %c5 = hw.constant 5 : i8
  %c7 = hw.constant 7 : i8
  %c90 = hw.constant 90 : i8
  %m = comb.mul %a, %c3 : i8
  %n1 = comb.mux %en, %m, %s1 : i8
  %s1 = seq.firreg %n1 clock %clk : i8
  %sh = comb.extract %s1 from 1 : (i8) -> i7
  %shz = comb.concat %false, %sh : i1, i7
  %x2a = comb.xor %s1, %shz : i8
  %x2 = comb.add %x2a, %c5 : i8
  %n2 = comb.mux %en, %x2, %s2 : i8
  %s2 = seq.firreg %n2 clock %clk : i8
  %x3a = comb.add %s2, %c7 : i8
  %x3 = comb.xor %x3a, %s2 : i8
  %n3 = comb.mux %en, %x3, %s3 : i8
  %s3 = seq.firreg %n3 clock %clk : i8
  %x4 = comb.mul %s3, %c5 : i8
  %n4 = comb.mux %en, %x4, %s4 : i8
  %s4 = seq.firreg %n4 clock %clk : i8
  %lo = comb.extract %vr from 0 : (i3) -> i2
  %vsh = comb.concat %lo, %v : i2, i1
  %vm = comb.mux %en, %vsh, %vr : i3
  %vn = comb.mux %reset, %c0_i3, %vm : i3
  %vr = seq.firreg %vn clock %clk : i3
  %v0 = comb.extract %vr from 0 : (i3) -> i1
  %v1 = comb.extract %vr from 1 : (i3) -> i1
  %v2 = comb.extract %vr from 2 : (i3) -> i1
  %g1 = comb.and %en, %v : i1
  %g2 = comb.and %en, %v0 : i1
  %g3 = comb.and %en, %v1 : i1
  %g4 = comb.and %en, %v2 : i1
  %u1 = comb.add %a, %c1 : i8
  %t1n = comb.mux %g1, %u1, %t1 : i8
  %t1 = seq.firreg %t1n clock %clk : i8
  %u2a = comb.mul %t1, %c5 : i8
  %u2 = comb.add %u2a, %t1 : i8
  %t2n = comb.mux %g2, %u2, %t2 : i8
  %t2 = seq.firreg %t2n clock %clk : i8
  %u3 = comb.sub %t2, %c3 : i8
  %t3n = comb.mux %g3, %u3, %t3 : i8
  %t3 = seq.firreg %t3n clock %clk : i8
  %u4 = comb.xor %t3, %c90 : i8
  %t4n = comb.mux %g4, %u4, %t4 : i8
  %t4 = seq.firreg %t4n clock %clk : i8
  %d = comb.xor %a, %c90 : i8
  %q, %qv = hw.instance "box" @XBox(clk: %clk: !seq.clock, ce: %en: i1, d: %d: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  %acc1 = comb.add %acc, %s4 : i8
  %accn = comb.mux %reset, %c0, %acc1 : i8
  %acc = seq.firreg %accn clock %clk : i8
  %y = comb.concat %s4, %t4, %q, %acc : i8, i8, i8, i8
  hw.output %y : i32
}
