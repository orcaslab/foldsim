// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{carry-block-size=8},foldsim-estimate-timing{output=- report= top=Top})' -o /dev/null | FileCheck %s --check-prefix=ARR
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map,foldsim-estimate-timing{output=- report= top=Top})' -o /dev/null | FileCheck %s --check-prefix=BLIND

// A 33-bit adder's carry out selects between two 216-input parities, three
// LUT levels each. The carry arrives at level 6 (a LUT and five carry
// blocks); timed as such, the mapper leaves it to the last LUT: 7 levels.
// Arrival-blind (every adder bit at 0), it buries the carry under a parity
// cone and the depth counted after it is 8.
// ARR: 7,{{.*}},o.d[0]
// BLIND: 8,{{.*}},o.d[0]
hw.module @Top(in %clk : !seq.clock, in %a_in : i32, in %b_in : i32,
               in %x_in : i216, in %y_in : i216, out o : i1) {
  %false = hw.constant false
  %a = seq.firreg %a_in clock %clk : i32
  %b = seq.firreg %b_in clock %clk : i32
  %x = seq.firreg %x_in clock %clk : i216
  %y = seq.firreg %y_in clock %clk : i216
  %0 = comb.concat %false, %a : i1, i32
  %1 = comb.concat %false, %b : i1, i32
  %2 = comb.add %0, %1 : i33
  %3 = comb.extract %2 from 32 : (i33) -> i1
  %4 = comb.parity %x : i216
  %5 = comb.parity %y : i216
  %6 = comb.mux %3, %4, %5 : i1
  %o = seq.firreg %6 clock %clk : i1
  hw.output %o : i1
}
