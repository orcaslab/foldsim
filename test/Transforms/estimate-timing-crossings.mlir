// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-keep-modules{modules=G0,G1},hw-flatten-modules{hw-inline-all=false hw-inline-small hw-inline-with-state hw-small-threshold=300},foldsim-estimate-timing{top=Top report= crossings=- port-modules=G0,G1})' -o /dev/null | not grep '^g1,level,0,0'
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-keep-modules{modules=G0,G1},hw-flatten-modules{hw-inline-all=false hw-inline-small hw-inline-with-state hw-small-threshold=300},foldsim-estimate-timing{top=Top report= crossings=- port-modules=G0,G1})' -o /dev/null | FileCheck %s

// Partitions: each kept G0/G1 instance and the rest (`level`). A crossing is a
// cell bit read in another partition; a port that only passes wires (G1's
// `thru`) is none.
// CHECK: from,to,driver,load,cell,start,end_inst,end
// G0's output is two LUTs behind its register: driver 2, read by G1's
// register straight (load 0, an end point of the other partition).
// CHECK-DAG: g0,g1,2,0,comb.truth_table[0],r0[0],g1,r1.d[0]
// G1's register is read by the level through two LUTs: driver 0, load 2.
// CHECK-DAG: g1,level,0,2,r1[0],r1[0],,q.d[0]
// G0's second output passes G1 as wires and reaches the level's register
// straight: one crossing g0 -> level, none through G1.
// CHECK-DAG: g0,level,0,0,s0[0],s0[0],,t.d[0]
// A top input read in G0 counts its load on through G1's logic: the port's
// path to G1's register.
// CHECK-DAG: level,g0,0,2,d[0],d[0],g1,r1.d[0]
hw.module private @G0(in %clk : !seq.clock, in %d : i1, out o : i1, out s : i1) {
  %r0 = seq.firreg %d clock %clk : i1
  %s0 = seq.firreg %d clock %clk : i1
  %n1 = comb.truth_table %r0, %d -> [false, true, true, false]
  %n2 = comb.truth_table %n1, %d -> [true, false, false, true]
  hw.output %n2, %s0 : i1, i1
}
hw.module private @G1(in %clk : !seq.clock, in %in : i1, in %thru_in : i1, out o : i1, out thru : i1) {
  %r1 = seq.firreg %in clock %clk : i1
  hw.output %r1, %thru_in : i1, i1
}
hw.module @Top(in %clk : !seq.clock, in %d : i1, in %e : i1, out x : i1, out y : i1) {
  %o0, %s = hw.instance "g0" @G0(clk: %clk: !seq.clock, d: %d: i1) -> (o: i1, s: i1)
  %o1, %th = hw.instance "g1" @G1(clk: %clk: !seq.clock, in: %o0: i1, thru_in: %s: i1) -> (o: i1, thru: i1)
  %m1 = comb.truth_table %o1, %e -> [false, true, true, true]
  %m2 = comb.truth_table %m1, %e -> [true, false, true, false]
  %q = seq.firreg %m2 clock %clk : i1
  %t = seq.firreg %th clock %clk : i1
  hw.output %q, %t : i1, i1
}
