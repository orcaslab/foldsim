// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map,foldsim-estimate-timing{output=- report= top=Top})' -o /dev/null | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=virtexuplusHBM},foldsim-estimate-timing{output=- report= top=Top target=virtexuplusHBM})' -o /dev/null | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=asap7},foldsim-estimate-timing{output=- report= top=Top target=asap7})' -o /dev/null | FileCheck %s --check-prefix=ASAP7
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=virtexuplusHBM area},foldsim-estimate-area{output=- report= top=Top target=virtexuplusHBM lut-scale=1})' -o /dev/null | FileCheck %s --check-prefix=AREA

// An equality compare of two bits and the AND after it pack into one LUT6,
// across the boundary of the inlined `Eq`.
// CHECK: 1,1,d,reg,{{[0-9]+}},0,0,{{[ab]\[[01]\]}},0,e.d[0]
// On ASAP7 they map to 3-input cells, two stages.
// ASAP7: 2,2,d,reg,{{[0-9]+}},0,0,{{[ab]\[[01]\]}},0,e.d[0]
// A 16-bit adder stays a carry chain: bit 15 is a LUT and two CARRY8s away.
// CHECK: 3,3,d,reg,{{[0-9]+}},0,0,a[0],0,s.d[15]
// On ASAP7, a prefix adder: 2 + log2(16) stages.
// ASAP7: 6,6,d,reg,{{[0-9]+}},0,0,a[0],0,s.d[15]
// Mapped for area, `Eq` keeps its instance and its own LUT; Top has the AND's
// LUT and the adder's 16, and 49 register bits.
// AREA: 0,,Top,17,0,49,0.0,0,0,0,18,0,49,0.0,0,0,0
// AREA-NEXT: 1,eq,Eq,1,0,0,0.0,0,0,0,1,0,0,0.0,0,0,0
hw.module private @Eq(in %x : i2, in %y : i2, out o : i1) {
  %eq = comb.icmp eq %x, %y : i2
  hw.output %eq : i1
}
hw.module @Top(in %clk : !seq.clock, in %d : i16, out e : i1, out s : i16) {
  %a = seq.firreg %d clock %clk : i16
  %b = seq.firreg %d clock %clk : i16
  %a2 = comb.extract %a from 0 : (i16) -> i2
  %b2 = comb.extract %b from 0 : (i16) -> i2
  %eq = hw.instance "eq" @Eq(x: %a2: i2, y: %b2: i2) -> (o: i1)
  %b15 = comb.extract %b from 15 : (i16) -> i1
  %en = comb.and %eq, %b15 : i1
  %e = seq.firreg %en clock %clk : i1
  %sum = comb.add %a, %b : i16
  %s = seq.firreg %sum clock %clk : i16
  hw.output %e, %s : i1, i16
}
