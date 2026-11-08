// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Prod,Cons netlist},foldsim-fold-static{module=Top})" 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Prod,Cons netlist},foldsim-fold-static{module=Top drop-dead-inputs})" 2>&1 | FileCheck %s --check-prefix=DROP

// Model inputs never read (`drop-dead-inputs`): Cons takes two of Prod's
// outputs and reads only `a`; Prod reads its `i`. Here the unread `b` is
// already given a constant without the option (a flat netlist's cut port);
// the option makes it explicit for every model input whose block argument
// has no use -- in a nest, a parent's ports left over from its children's
// class cuts, which otherwise keep links (128 SMs: 466 target-model inputs,
// 364k bits; the gate netlists: 42, 4,580 bits). The schedule is the same.
// (Exactness: the small/mid and b4e gates with the option.)

// KEEP: foldsim-fold-static: II = [[II:[0-9]+]]
// KEEP-NOT: dead inputs
// KEEP: hw.instance "Cons_copy" @Cons_static({{.*}}a: %Cons_a_read: i16, b: %c0_i16{{(_[0-9]+)?}}: i16)
// DROP: foldsim-fold-static: II = [[II:[0-9]+]]
// DROP: dead inputs 2 (32 bits, no link)
// DROP-NOT: Prod_y_to_Cons_b
// DROP: hw.instance "Cons_copy" @Cons_static({{.*}}a: %Cons_a_read: i16, b: %c0_i16{{(_[0-9]+)?}}: i16)

hw.module private @Prod(in %clk : !seq.clock, in %i : i16, out x : i16, out y : i16) {
  %c1 = hw.constant 1 : i16
  %r = seq.firreg %n clock %clk : i16
  %n = comb.add %r, %i : i16
  %s = comb.xor %r, %c1 : i16
  hw.output %r, %s : i16, i16
}
hw.module private @Cons(in %clk : !seq.clock, in %a : i16, in %b : i16, out z : i16) {
  %r = seq.firreg %n clock %clk : i16
  %n = comb.add %r, %a : i16
  hw.output %r : i16
}
hw.module @Top(in %clk : !seq.clock, in %i : i16, out z0 : i16, out z1 : i16) {
  %x0, %y0 = hw.instance "p0" @Prod(clk: %clk: !seq.clock, i: %i: i16) -> (x: i16, y: i16)
  %x1, %y1 = hw.instance "p1" @Prod(clk: %clk: !seq.clock, i: %x0: i16) -> (x: i16, y: i16)
  %z0 = hw.instance "c0" @Cons(clk: %clk: !seq.clock, a: %x0: i16, b: %y1: i16) -> (z: i16)
  %z1 = hw.instance "c1" @Cons(clk: %clk: !seq.clock, a: %x1: i16, b: %y0: i16) -> (z: i16)
  hw.output %z0, %z1 : i16, i16
}
