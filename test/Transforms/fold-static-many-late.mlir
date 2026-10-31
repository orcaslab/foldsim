// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Chain models=C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12 netlist},foldsim-fold-static{module=Chain schedule-only})" -o /dev/null 2>&1 | FileCheck %s

// Thirteen models with late outputs (a chain of them, each output following
// its input in the same cycle) are too many to place every choice of ready
// passes: the search starts from none and from one for each, flips one or
// two, then flips single models while the frame gets shorter. The chain
// needs no ready pass but the parent's.
// CHECK: foldsim-fold-static: II = 20
// CHECK-NEXT: Chain_model x1 (ready pass)
// CHECK: C0 x1 state gap
// CHECK-NOT: (ready pass)
// CHECK: C12 x1 state gap
hw.module @C0(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c1 = hw.constant 1 : i8
  %n = comb.add %r, %d, %c1 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C1(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c2 = hw.constant 2 : i8
  %n = comb.add %r, %d, %c2 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C2(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c3 = hw.constant 3 : i8
  %n = comb.add %r, %d, %c3 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C3(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c4 = hw.constant 4 : i8
  %n = comb.add %r, %d, %c4 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C4(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c5 = hw.constant 5 : i8
  %n = comb.add %r, %d, %c5 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C5(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c6 = hw.constant 6 : i8
  %n = comb.add %r, %d, %c6 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C6(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c7 = hw.constant 7 : i8
  %n = comb.add %r, %d, %c7 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C7(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c8 = hw.constant 8 : i8
  %n = comb.add %r, %d, %c8 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C8(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c9 = hw.constant 9 : i8
  %n = comb.add %r, %d, %c9 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C9(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c10 = hw.constant 10 : i8
  %n = comb.add %r, %d, %c10 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C10(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c11 = hw.constant 11 : i8
  %n = comb.add %r, %d, %c11 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C11(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c12 = hw.constant 12 : i8
  %n = comb.add %r, %d, %c12 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @C12(in %clk : !seq.clock, in %reset : i1, in %d : i8, out q : i8) {
  %c0 = hw.constant 0 : i8
  %c13 = hw.constant 13 : i8
  %n = comb.add %r, %d, %c13 : i8
  %r = seq.firreg %n clock %clk reset sync %reset, %c0 : i8
  %q = comb.xor %d, %r : i8
  hw.output %q : i8
}
hw.module @Chain(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z0 : i8, out z1 : i8) {
  %c0 = hw.constant 0 : i8
  %t = seq.firreg %c12.q clock %clk reset sync %reset, %c0 : i8
  %d0 = comb.xor %t, %x : i8
  %c0.q = hw.instance "c0" @C0(clk: %clk: !seq.clock, reset: %reset: i1, d: %d0: i8) -> (q: i8)
  %c1.q = hw.instance "c1" @C1(clk: %clk: !seq.clock, reset: %reset: i1, d: %c0.q: i8) -> (q: i8)
  %c2.q = hw.instance "c2" @C2(clk: %clk: !seq.clock, reset: %reset: i1, d: %c1.q: i8) -> (q: i8)
  %c3.q = hw.instance "c3" @C3(clk: %clk: !seq.clock, reset: %reset: i1, d: %c2.q: i8) -> (q: i8)
  %c4.q = hw.instance "c4" @C4(clk: %clk: !seq.clock, reset: %reset: i1, d: %c3.q: i8) -> (q: i8)
  %c5.q = hw.instance "c5" @C5(clk: %clk: !seq.clock, reset: %reset: i1, d: %c4.q: i8) -> (q: i8)
  %c6.q = hw.instance "c6" @C6(clk: %clk: !seq.clock, reset: %reset: i1, d: %c5.q: i8) -> (q: i8)
  %c7.q = hw.instance "c7" @C7(clk: %clk: !seq.clock, reset: %reset: i1, d: %c6.q: i8) -> (q: i8)
  %c8.q = hw.instance "c8" @C8(clk: %clk: !seq.clock, reset: %reset: i1, d: %c7.q: i8) -> (q: i8)
  %c9.q = hw.instance "c9" @C9(clk: %clk: !seq.clock, reset: %reset: i1, d: %c8.q: i8) -> (q: i8)
  %c10.q = hw.instance "c10" @C10(clk: %clk: !seq.clock, reset: %reset: i1, d: %c9.q: i8) -> (q: i8)
  %c11.q = hw.instance "c11" @C11(clk: %clk: !seq.clock, reset: %reset: i1, d: %c10.q: i8) -> (q: i8)
  %c12.q = hw.instance "c12" @C12(clk: %clk: !seq.clock, reset: %reset: i1, d: %c11.q: i8) -> (q: i8)
  hw.output %t, %c6.q : i8, i8
}
