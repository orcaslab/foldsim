// RUN: foldsim-opt %s --foldsim-find-pipelines="models=Ok=4,Short=4,Gated=4,Box=4,BoxOut=4,Feedback=4,MixedEnables=4,Unbalanced=4,ReadEarly=4,BadGate=4" | FileCheck %s

// Four stages on one enable: folded, the last layer staying.
// CHECK-LABEL: model Ok threads 4
// CHECK-NEXT: pipeline 0: L=4 depth=3 keeps-last-layer enable=signal registers=4 boxes=0
// CHECK-NEXT: layer bits: 8 8 8 8
// CHECK-NEXT: input bits 8, store bits 8, folded state bits 24 (x4 threads)
hw.module @Ok(in %clk : !seq.clock, in %en : i1, in %a : i8, out y : i8) {
  %c3 = hw.constant 3 : i8
  %c7 = hw.constant 7 : i8
  %m = comb.mul %a, %c3 : i8
  %n1 = comb.mux %en, %m, %s1 : i8
  %s1 = seq.firreg %n1 clock %clk : i8
  %x = comb.xor %s1, %c7 : i8
  %n2 = comb.mux %en, %x, %s2 : i8
  %s2 = seq.firreg %n2 clock %clk : i8
  %p = comb.add %s2, %c7 : i8
  %n3 = comb.mux %en, %p, %s3 : i8
  %s3 = seq.firreg %n3 clock %clk : i8
  %n4 = comb.mux %en, %s3, %s4 : i8
  %s4 = seq.firreg %n4 clock %clk : i8
  hw.output %s4 : i8
}

// Three stages: too short to fold.
// CHECK-LABEL: model Short threads 4
// CHECK-NOT: pipeline
// CHECK: left out: fewer than three stages to fold: 24 bits
hw.module @Short(in %clk : !seq.clock, in %en : i1, in %a : i8, out y : i8) {
  %n1 = comb.mux %en, %a, %s1 : i8
  %s1 = seq.firreg %n1 clock %clk : i8
  %n2 = comb.mux %en, %s1, %s2 : i8
  %s2 = seq.firreg %n2 clock %clk : i8
  %n3 = comb.mux %en, %s2, %s3 : i8
  %s3 = seq.firreg %n3 clock %clk : i8
  hw.output %s3 : i8
}

// Each stage written on the enable and the valid of its stage, a valid
// chain cleared on reset: folded, gated.
// CHECK-LABEL: model Gated threads 4
// CHECK-NEXT: pipeline 0: L=4 depth=3 keeps-last-layer gated enable=signal registers=4 boxes=0
hw.module @Gated(in %clk : !seq.clock, in %reset : i1, in %en : i1, in %v : i1, in %a : i8, out y : i8, out yv : i1) {
  %c0_i3 = hw.constant 0 : i3
  %c1 = hw.constant 1 : i8
  %c5 = hw.constant 5 : i8
  %lo = comb.extract %vr from 0 : (i3) -> i2
  %sh = comb.concat %lo, %v : i2, i1
  %vm = comb.mux %en, %sh, %vr : i3
  %vn = comb.mux %reset, %c0_i3, %vm : i3
  %vr = seq.firreg %vn clock %clk : i3
  %v0 = comb.extract %vr from 0 : (i3) -> i1
  %v1 = comb.extract %vr from 1 : (i3) -> i1
  %v2 = comb.extract %vr from 2 : (i3) -> i1
  %g1 = comb.and %en, %v : i1
  %g2 = comb.and %en, %v0 : i1
  %g3 = comb.and %en, %v1 : i1
  %u1 = comb.add %a, %c1 : i8
  %n1 = comb.mux %g1, %u1, %t1 : i8
  %t1 = seq.firreg %n1 clock %clk : i8
  %u2 = comb.mul %t1, %c5 : i8
  %n2 = comb.mux %g2, %u2, %t2 : i8
  %t2 = seq.firreg %n2 clock %clk : i8
  %u3 = comb.sub %t2, %c1 : i8
  %n3 = comb.mux %g3, %u3, %t3 : i8
  %t3 = seq.firreg %n3 clock %clk : i8
  %g4 = comb.and %en, %v2 : i1
  %n4 = comb.mux %g4, %t3, %t4 : i8
  %t4 = seq.firreg %n4 clock %clk : i8
  hw.output %t4, %v2 : i8, i1
}

// A black box with a contract, and a register of its enable after it: one
// pipeline of 5 layers, the box's 4 then the register, which stays.
// CHECK-LABEL: model Box threads 4
// CHECK-NEXT: pipeline 0: L=5 depth=4 keeps-last-layer enable=signal registers=1 boxes=1
// CHECK-NEXT: layer bits: 0 0 0 0 8
// CHECK-NEXT: input bits 8, store bits 8, folded state bits 0 (x4 threads)
hw.module.extern @XBox(in %clk : !seq.clock, in %ce : i1, in %d : i8, in %d_valid : i1, out q : i8, out q_valid : i1) attributes {sv.attributes = [#sv.attribute<"black_box">, #sv.attribute<"fixed_latency" = "4">, #sv.attribute<"clock" = "\22clk\22">, #sv.attribute<"clock_enable" = "\22ce\22">, #sv.attribute<"data_in" = "\22d\22">, #sv.attribute<"data_out" = "\22q\22">, #sv.attribute<"tie" = "\22d_valid=1\22">, #sv.attribute<"pure_pipeline">]}
hw.module @Box(in %clk : !seq.clock, in %en : i1, in %a : i8, out y : i8) {
  %true = hw.constant true
  %q, %qv = hw.instance "box" @XBox(clk: %clk: !seq.clock, ce: %en: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  %n = comb.mux %en, %q, %r : i8
  %r = seq.firreg %n clock %clk : i8
  hw.output %r : i8
}

// A black box read outside: its last layer is stored.
// CHECK-LABEL: model BoxOut threads 4
// CHECK-NEXT: pipeline 0: L=4 depth=4 stores-last-layer enable=signal registers=0 boxes=1
// CHECK-NEXT: layer bits: 0 0 0 0
// CHECK-NEXT: input bits 8, store bits 8, folded state bits 0 (x4 threads)
hw.module @BoxOut(in %clk : !seq.clock, in %en : i1, in %a : i8, out y : i8) {
  %true = hw.constant true
  %q, %qv = hw.instance "box" @XBox(clk: %clk: !seq.clock, ce: %en: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  hw.output %q : i8
}

// An accumulator: its register is on a loop.
// CHECK-LABEL: model Feedback threads 4
// CHECK-NOT: pipeline
// CHECK: left out: feedback: 8 bits
hw.module @Feedback(in %clk : !seq.clock, in %en : i1, in %a : i8, out y : i8) {
  %s = comb.add %acc, %a : i8
  %n1 = comb.mux %en, %s, %acc : i8
  %acc = seq.firreg %n1 clock %clk : i8
  %n2 = comb.mux %en, %acc, %r : i8
  %r = seq.firreg %n2 clock %clk : i8
  hw.output %r : i8
}

// The second stage on another enable: two pipelines of one layer, too short.
// CHECK-LABEL: model MixedEnables threads 4
// CHECK-NOT: pipeline
// CHECK: left out: fewer than three stages to fold: 24 bits
hw.module @MixedEnables(in %clk : !seq.clock, in %en : i1, in %en2 : i1, in %a : i8, out y : i8) {
  %n1 = comb.mux %en, %a, %s1 : i8
  %s1 = seq.firreg %n1 clock %clk : i8
  %n2 = comb.mux %en2, %s1, %s2 : i8
  %s2 = seq.firreg %n2 clock %clk : i8
  %n3 = comb.mux %en2, %s2, %s3 : i8
  %s3 = seq.firreg %n3 clock %clk : i8
  hw.output %s3 : i8
}

// The third stage adds the input: paths of 1 and 3 registers.
// CHECK-LABEL: model Unbalanced threads 4
// CHECK-NOT: pipeline
// CHECK: left out: inputs at a later stage: 8 bits
hw.module @Unbalanced(in %clk : !seq.clock, in %en : i1, in %a : i8, out y : i8) {
  %n1 = comb.mux %en, %a, %s1 : i8
  %s1 = seq.firreg %n1 clock %clk : i8
  %n2 = comb.mux %en, %s1, %s2 : i8
  %s2 = seq.firreg %n2 clock %clk : i8
  %p = comb.add %s2, %a : i8
  %n3 = comb.mux %en, %p, %s3 : i8
  %s3 = seq.firreg %n3 clock %clk : i8
  hw.output %s3 : i8
}

// The first stage is an output too: it stays, the rest is too short.
// CHECK-LABEL: model ReadEarly threads 4
// CHECK-NOT: pipeline
// CHECK: left out: read before the last stage: 8 bits
hw.module @ReadEarly(in %clk : !seq.clock, in %en : i1, in %a : i8, out y : i8, out z : i8) {
  %n1 = comb.mux %en, %a, %s1 : i8
  %s1 = seq.firreg %n1 clock %clk : i8
  %n2 = comb.mux %en, %s1, %s2 : i8
  %s2 = seq.firreg %n2 clock %clk : i8
  %n3 = comb.mux %en, %s2, %s3 : i8
  %s3 = seq.firreg %n3 clock %clk : i8
  hw.output %s3, %s1 : i8, i8
}

// The second stage gated by a bit that is not its stage's valid.
// CHECK-LABEL: model BadGate threads 4
// CHECK-NOT: pipeline
// CHECK: left out: gate not the valid of its stage: 16 bits
hw.module @BadGate(in %clk : !seq.clock, in %en : i1, in %v : i1, in %w : i1, in %a : i8, out y : i8) {
  %g1 = comb.and %en, %v : i1
  %g2 = comb.and %en, %w : i1
  %n1 = comb.mux %g1, %a, %s1 : i8
  %s1 = seq.firreg %n1 clock %clk : i8
  %n2 = comb.mux %g2, %s1, %s2 : i8
  %s2 = seq.firreg %n2 clock %clk : i8
  %n3 = comb.mux %g2, %s2, %s3 : i8
  %s3 = seq.firreg %n3 clock %clk : i8
  hw.output %s3 : i8
}
