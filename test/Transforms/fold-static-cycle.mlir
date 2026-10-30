// A frame whose evaluations wait on each other in a cycle for every choice of
// ready passes: the error names a shortest cycle, edge by edge.
// RUN: not foldsim-opt %s --mlir-print-op-on-diagnostic=false --pass-pipeline="builtin.module(foldsim-extract-models{module=Loop models=Pass netlist},foldsim-fold-static{module=Loop schedule-only})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=COMB
// RUN: not foldsim-opt %s --mlir-print-op-on-diagnostic=false --pass-pipeline="builtin.module(foldsim-extract-models{module=Shared models=Two netlist},foldsim-fold-static{module=Shared schedule-only ready-levels=off})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=SHARED

// `y` depends on `x` in the same cycle, and two instances feed each other:
// a combinational loop through both.
// COMB: error: the evaluations of a frame depend on each other in a cycle for every choice of ready passes
// COMB-NEXT: a shortest cycle, 2 edges, with 2 of 2 models given a ready pass
// COMB: note: cycle edge 1 of 2: Pass #{{[01]}} ({{[ab]}}) ready pass -> Pass #{{[01]}} ({{[ab]}}) ready pass: output 'y' -> input 'x' (1 host cycle), 8 of the input's 8 bits. 'y' is late: it depends combinationally on Pass's input 'x'. The ready pass reads it for its outputs 'y'. Through Pass: its output 'y' depends combinationally on 'x' (6 of 8 bits: [7:2] of the output): a combinational path through the model.
// COMB: note: cycle edge 2 of 2:
// COMB-SAME: a combinational path through the model.
// COMB: note: even with a ready pass per late output the frame has a cycle: a combinational loop through the ports, 2 outputs: Pass #{{[01]}} output 'y' -> Pass #{{[01]}} input 'x', on which its output 'y' depends; Pass #{{[01]}} output 'y' -> Pass #{{[01]}} input 'x', on which its output 'y' depends;
hw.module @Pass(in %clk : !seq.clock, in %x : i8, out y : i8) {
  %r = seq.firreg %x clock %clk : i8
  %hi = comb.extract %x from 2 : (i8) -> i6
  %lo = comb.extract %r from 0 : (i8) -> i2
  %y = comb.concat %hi, %lo : i6, i2
  hw.output %y : i8
}
hw.module @Loop(in %clk : !seq.clock, out z : i8) {
  %a.y = hw.instance "a" @Pass(clk: %clk: !seq.clock, x: %b.y: i8) -> (y: i8)
  %b.y = hw.instance "b" @Pass(clk: %clk: !seq.clock, x: %a.y: i8) -> (y: i8)
  hw.output %a.y : i8
}

// `Two`'s outputs each depend on one input, so no path through the netlist
// is combinational; but one ready pass writes both outputs, so `a`'s waits
// for `x2`, which `b`'s ready pass writes, which waits for its own `x2`.
// Without a ready pass, the state passes wait on each other. (Two ready
// passes by level schedule it: fold-static-ready-levels.mlir; here they are
// off.)
// SHARED: error: the evaluations of a frame depend on each other in a cycle for every choice of ready passes
// SHARED-NEXT: a shortest cycle, 2 edges, with 2 of 2 models given a ready pass
// SHARED: note: cycle edge 1 of 2: Two #{{[01]}} ({{[ab]}}) ready pass -> Two #{{[01]}} ({{[ab]}}) ready pass: output 'y1' -> input 'x2' (1 host cycle), 8 of the input's 8 bits. 'y1' is late: it depends combinationally on Two's input 'x1'. The ready pass reads it for its outputs 'y2'. Through Two: its output 'y1' does not depend on 'x2'; the cycle closes only because one ready pass writes all of Two's ready outputs (a second ready pass would break it).
// SHARED: note: cycle edge 2 of 2:
// SHARED-SAME: a second ready pass would break it
// SHARED: note: with a ready pass per late output instead {{.*}} the frame has no cycle: no path through the ports is combinational, so the netlist is legal and the schedule needs the models on the cycle to write their ready outputs in more than one pass
hw.module @Two(in %clk : !seq.clock, in %x1 : i8, in %x2 : i8, out y1 : i8, out y2 : i8) {
  %r = seq.firreg %x1 clock %clk : i8
  %y1 = comb.xor %x1, %r : i8
  %y2 = comb.xor %x2, %r : i8
  hw.output %y1, %y2 : i8, i8
}
hw.module @Shared(in %clk : !seq.clock, in %in : i8, out z : i8) {
  %a.y1, %a.y2 = hw.instance "a" @Two(clk: %clk: !seq.clock, x1: %in: i8, x2: %b.y1: i8) -> (y1: i8, y2: i8)
  %b.y1, %b.y2 = hw.instance "b" @Two(clk: %clk: !seq.clock, x1: %in: i8, x2: %a.y1: i8) -> (y1: i8, y2: i8)
  %z = comb.xor %a.y2, %b.y2 : i8
  hw.output %z : i8
}
