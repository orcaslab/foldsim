// Early outputs that do not start at 0 (zeroStartOutputs): an early link's
// first frame reads its storage unwritten (0), while the target's output has
// its initial state's value from the first cycle.
// Three netlists `Top`, one a split: the first `Dst`s read one source, the
// second's two sources start at different values, the third's at 0.
// RUN: foldsim-opt %s --split-input-file --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Src,Dst netlist},foldsim-fold-static{module=Top})" -o - 2>&1 | FileCheck %s

// `Src`'s output is its register XOR 0x5a: 0x5a at the initial state (no
// reset, as the target starts). Every `Dst` reads the same one: the output
// is written XOR 0x5a and `Dst` XORs it back at its input.
// CHECK: zero-start: 1 early outputs written XOR their initial value, 0 written late
// CHECK: hw.constant 90 : i8
// CHECK: comb.xor
// CHECK: hw.constant 90 : i8
// CHECK: comb.xor
// CHECK-LABEL: hw.module @foldsim_sim

// Two sources of different initial values, one read by each `Dst`: one mask
// would not do for both, so the outputs are written late, in their own frame.
// CHECK: zero-start: 0 early outputs written XOR their initial value, 1 written late
// CHECK-LABEL: hw.module @foldsim_sim

// An output that starts at 0 is left as it is.
// CHECK-NOT: zero-start:
// CHECK-LABEL: hw.module @foldsim_sim

hw.module @Src(in %clk : !seq.clock, in %d : i8, in %k : i8, out q : i8) {
  %r = seq.firreg %d clock %clk : i8
  %q = comb.xor %r, %k : i8
  hw.output %q : i8
}

hw.module @Dst(in %clk : !seq.clock, in %x : i8, out y : i8) {
  %r = seq.firreg %x clock %clk : i8
  hw.output %r : i8
}

hw.module @Top(in %clk : !seq.clock, in %d : i8, out y0 : i8, out y1 : i8) {
  %c90 = hw.constant 90 : i8
  %q = hw.instance "s" @Src(clk: %clk : !seq.clock, d: %d : i8, k: %c90 : i8) -> (q : i8)
  %y0 = hw.instance "d0" @Dst(clk: %clk : !seq.clock, x: %q : i8) -> (y : i8)
  %y1 = hw.instance "d1" @Dst(clk: %clk : !seq.clock, x: %q : i8) -> (y : i8)
  hw.output %y0, %y1 : i8, i8
}


// -----

hw.module @Src(in %clk : !seq.clock, in %d : i8, in %k : i8, out q : i8) {
  %r = seq.firreg %d clock %clk : i8
  %q = comb.xor %r, %k : i8
  hw.output %q : i8
}

hw.module @Dst(in %clk : !seq.clock, in %x : i8, out y : i8) {
  %r = seq.firreg %x clock %clk : i8
  hw.output %r : i8
}

hw.module @Top(in %clk : !seq.clock, in %d : i8, out y0 : i8, out y1 : i8) {
  %c90 = hw.constant 90 : i8
  %c165 = hw.constant 165 : i8
  %q0 = hw.instance "s0" @Src(clk: %clk : !seq.clock, d: %d : i8, k: %c90 : i8) -> (q : i8)
  %q1 = hw.instance "s1" @Src(clk: %clk : !seq.clock, d: %d : i8, k: %c165 : i8) -> (q : i8)
  %y0 = hw.instance "d0" @Dst(clk: %clk : !seq.clock, x: %q0 : i8) -> (y : i8)
  %y1 = hw.instance "d1" @Dst(clk: %clk : !seq.clock, x: %q1 : i8) -> (y : i8)
  hw.output %y0, %y1 : i8, i8
}


// -----

hw.module @Src(in %clk : !seq.clock, in %d : i8, in %k : i8, out q : i8) {
  %r = seq.firreg %d clock %clk : i8
  %q = comb.xor %r, %k : i8
  hw.output %q : i8
}

hw.module @Dst(in %clk : !seq.clock, in %x : i8, out y : i8) {
  %r = seq.firreg %x clock %clk : i8
  hw.output %r : i8
}

hw.module @Top(in %clk : !seq.clock, in %d : i8, out y0 : i8) {
  %c0 = hw.constant 0 : i8
  %q = hw.instance "s" @Src(clk: %clk : !seq.clock, d: %d : i8, k: %c0 : i8) -> (q : i8)
  %y0 = hw.instance "d0" @Dst(clk: %clk : !seq.clock, x: %q : i8) -> (y : i8)
  hw.output %y0 : i8
}
