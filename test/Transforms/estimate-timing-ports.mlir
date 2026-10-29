// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-keep-modules{modules=Buf,Pass},hw-flatten-modules{hw-inline-all=false hw-inline-small hw-inline-with-state hw-small-threshold=300},foldsim-estimate-timing{top=Top report= ports=- port-modules=Buf,Pass})' -o /dev/null | FileCheck %s

// Buf_1 is a specialization of Buf, kept through inlining; Leaf is inlined.
// Its input arrives 2 levels deep, and its output is registered: no bit of it
// is reached from outside without a register.
// CHECK: instance,module,direction,port,width,levels,through
// CHECK-NEXT: buf,Buf_1,in,in,1,2,0
// CHECK-NEXT: buf,Buf_1,out,out,1,0,0
// Pass computes its output from its input combinationally in one bit and
// registers the other: one bit passes through.
// CHECK-NEXT: pass,Pass,in,in,2,0,0
// CHECK-NEXT: pass,Pass,out,out,2,{{[0-9]+}},1
// CHECK-NOT: Leaf
hw.module private @Leaf(in %a : i1, out o : i1) {
  %o = comb.truth_table %a -> [true, false]
  hw.output %o : i1
}
hw.module private @Buf_1(in %clk : !seq.clock, in %in : i1, out out : i1) {
  %r = seq.firreg %in clock %clk : i1
  hw.output %r : i1
}
hw.module private @Pass(in %clk : !seq.clock, in %in : i2, out out : i2) {
  %i0 = comb.extract %in from 0 : (i2) -> i1
  %i1 = comb.extract %in from 1 : (i2) -> i1
  %r = seq.firreg %i0 clock %clk : i1
  %n = comb.truth_table %i0, %i1 -> [false, true, true, false]
  %o = comb.concat %n, %r : i1, i1
  hw.output %o : i2
}
hw.module @Top(in %clk : !seq.clock, in %d : i1, in %e : i2, out q : i1, out p : i2) {
  %a = seq.firreg %d clock %clk : i1
  %l1 = hw.instance "leaf1" @Leaf(a: %a: i1) -> (o: i1)
  %l2 = hw.instance "leaf2" @Leaf(a: %l1: i1) -> (o: i1)
  %b = hw.instance "buf" @Buf_1(clk: %clk: !seq.clock, in: %l2: i1) -> (out: i1)
  %pe = hw.instance "pass" @Pass(clk: %clk: !seq.clock, in: %e: i2) -> (out: i2)
  hw.output %b, %pe : i1, i2
}
