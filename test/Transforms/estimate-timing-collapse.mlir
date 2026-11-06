// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=virtexuplusHBM},foldsim-estimate-timing{output=- report= top=Top target=virtexuplusHBM})' -o /dev/null | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=asap7},foldsim-estimate-timing{output=- report= top=Top target=asap7})' -o /dev/null | FileCheck %s --check-prefix=ASAP7

// A superpage mask, all ones shifted left by a 2-bit level times 9, depends on
// the two level bits alone: one LUT per bit, not a product, compares and a
// shifter.
// CHECK-DAG: 1,1,d,reg,{{[0-9]+}},0,0,l[{{[01]}}],0,m.d[20]
// On ASAP7, a 2:1 multiplexer tree over the two bits: two stages.
// ASAP7-DAG: 2,2,d,reg,{{[0-9]+}},0,0,l[{{[01]}}],0,m.d[20]
// Two 27-bit registers compared under that mask: too wide for one table,
// the compare is a LUT6 tree after the one-level mask, three levels.
// CHECK-DAG: 3,3,d,reg,{{[0-9]+}},0,0,{{.*}},0,h.d[0]
// ASAP7-DAG: 8,8,d,reg,{{[0-9]+}},0,0,{{.*}},0,h.d[0]
hw.module @Top(in %clk : !seq.clock, in %lvl : i2, in %v : i27, in %w : i27, out m : i27, out h : i1) {
  %l = seq.firreg %lvl clock %clk : i2
  %a = seq.firreg %v clock %clk : i27
  %b = seq.firreg %w clock %clk : i27
  %ones = hw.constant -1 : i27
  %nine = hw.constant 9 : i27
  %lz = comb.concat %zero, %l : i25, i2
  %zero = hw.constant 0 : i25
  %amt = comb.mul %lz, %nine : i27
  %mask = comb.shl %ones, %amt : i27
  %m = seq.firreg %mask clock %clk : i27
  %am = comb.and %a, %mask : i27
  %bm = comb.and %b, %mask : i27
  %eq = comb.icmp eq %am, %bm : i27
  %h = seq.firreg %eq clock %clk : i1
  hw.output %m, %h : i27, i1
}
