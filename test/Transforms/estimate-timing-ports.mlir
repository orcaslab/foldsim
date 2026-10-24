// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-keep-modules{modules=Buf},hw-flatten-modules{hw-inline-all=false hw-inline-small hw-inline-with-state hw-small-threshold=300},foldsim-estimate-timing{top=Top report= ports=- port-modules=Buf})' -o /dev/null | FileCheck %s

// Buf_1 is a specialization of Buf, kept through inlining; Leaf is inlined.
// Its input arrives 2 levels deep, and its output is registered.
// CHECK: instance,module,direction,port,width,levels
// CHECK-NEXT: buf,Buf_1,in,in,1,2
// CHECK-NEXT: buf,Buf_1,out,out,1,0
// CHECK-NOT: Leaf
hw.module private @Leaf(in %a : i1, out o : i1) {
  %o = comb.truth_table %a -> [true, false]
  hw.output %o : i1
}
hw.module private @Buf_1(in %clk : !seq.clock, in %in : i1, out out : i1) {
  %r = seq.firreg %in clock %clk : i1
  hw.output %r : i1
}
hw.module @Top(in %clk : !seq.clock, in %d : i1, out q : i1) {
  %a = seq.firreg %d clock %clk : i1
  %l1 = hw.instance "leaf1" @Leaf(a: %a: i1) -> (o: i1)
  %l2 = hw.instance "leaf2" @Leaf(a: %l1: i1) -> (o: i1)
  %b = hw.instance "buf" @Buf_1(clk: %clk: !seq.clock, in: %l2: i1) -> (out: i1)
  hw.output %b : i1
}
