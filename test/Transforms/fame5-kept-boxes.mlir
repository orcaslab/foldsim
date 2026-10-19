// RUN: foldsim-opt %s --foldsim-fame5="module=Boxes threads=1 clock-enable thread-index" | FileCheck %s

// A model of one thread keeps its black boxes as instances on the host clock,
// each enabled on the target's edges. The enables are made in the boxes'
// order, so the output is the same on every run.
// CHECK: hw.instance "u0" @XBox(clk: %hostClock: !seq.clock, ce: [[C0:%[0-9]+]]: i1
// CHECK: hw.instance "u1" @XBox(clk: %hostClock: !seq.clock, ce: [[C1:%[0-9]+]]: i1
// CHECK: hw.instance "u2" @XBox(clk: %hostClock: !seq.clock, ce: [[C2:%[0-9]+]]: i1
// CHECK: hw.instance "u3" @XBox(clk: %hostClock: !seq.clock, ce: [[C3:%[0-9]+]]: i1
// CHECK: hw.instance "u4" @XBox(clk: %hostClock: !seq.clock, ce: [[C4:%[0-9]+]]: i1
// CHECK: hw.instance "u5" @XBox(clk: %hostClock: !seq.clock, ce: [[C5:%[0-9]+]]: i1
// CHECK: hw.instance "u6" @XBox(clk: %hostClock: !seq.clock, ce: [[C6:%[0-9]+]]: i1
// CHECK: hw.instance "u7" @XBox(clk: %hostClock: !seq.clock, ce: [[C7:%[0-9]+]]: i1
// CHECK: [[EDGE:%.+]] = comb.and %clk_enabled
// CHECK-NEXT: [[C0]] = comb.and %e0_sink_bits, [[EDGE]] : i1
// CHECK-NEXT: [[C1]] = comb.and %e1_sink_bits, [[EDGE]] : i1
// CHECK-NEXT: [[C2]] = comb.and %e2_sink_bits, [[EDGE]] : i1
// CHECK-NEXT: [[C3]] = comb.and %e3_sink_bits, [[EDGE]] : i1
// CHECK-NEXT: [[C4]] = comb.and %e4_sink_bits, [[EDGE]] : i1
// CHECK-NEXT: [[C5]] = comb.and %e5_sink_bits, [[EDGE]] : i1
// CHECK-NEXT: [[C6]] = comb.and %e6_sink_bits, [[EDGE]] : i1
// CHECK-NEXT: [[C7]] = comb.and %e7_sink_bits, [[EDGE]] : i1

hw.module.extern @XBox(in %clk : !seq.clock, in %ce : i1, in %d : i8, in %d_valid : i1, out q : i8, out q_valid : i1) attributes {sv.attributes = [#sv.attribute<"black_box">, #sv.attribute<"fixed_latency" = "4">, #sv.attribute<"clock" = "\22clk\22">, #sv.attribute<"clock_enable" = "\22ce\22">, #sv.attribute<"data_in" = "\22d\22">, #sv.attribute<"data_out" = "\22q\22">, #sv.attribute<"tie" = "\22d_valid=1\22">, #sv.attribute<"pure_pipeline">]}
hw.module @Boxes(in %clk : !seq.clock, in %a : i8, in %e0 : i1, in %e1 : i1, in %e2 : i1, in %e3 : i1, in %e4 : i1, in %e5 : i1, in %e6 : i1, in %e7 : i1, out y0 : i8, out y1 : i8, out y2 : i8, out y3 : i8, out y4 : i8, out y5 : i8, out y6 : i8, out y7 : i8) {
  %true = hw.constant true
  %q0, %qv0 = hw.instance "u0" @XBox(clk: %clk: !seq.clock, ce: %e0: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  %q1, %qv1 = hw.instance "u1" @XBox(clk: %clk: !seq.clock, ce: %e1: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  %q2, %qv2 = hw.instance "u2" @XBox(clk: %clk: !seq.clock, ce: %e2: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  %q3, %qv3 = hw.instance "u3" @XBox(clk: %clk: !seq.clock, ce: %e3: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  %q4, %qv4 = hw.instance "u4" @XBox(clk: %clk: !seq.clock, ce: %e4: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  %q5, %qv5 = hw.instance "u5" @XBox(clk: %clk: !seq.clock, ce: %e5: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  %q6, %qv6 = hw.instance "u6" @XBox(clk: %clk: !seq.clock, ce: %e6: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  %q7, %qv7 = hw.instance "u7" @XBox(clk: %clk: !seq.clock, ce: %e7: i1, d: %a: i8, d_valid: %true: i1) -> (q: i8, q_valid: i1)
  hw.output %q0, %q1, %q2, %q3, %q4, %q5, %q6, %q7 : i8, i8, i8, i8, i8, i8, i8, i8
}
