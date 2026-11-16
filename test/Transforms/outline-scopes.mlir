// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-outline-scopes{scopes=Sw:g_in_=Sw_in})' 2>&1 | FileCheck %s

// A generate loop's three iterations (registers g_in_<i>.r and g_in_<i>.c):
// each with the logic that feeds it alone (its counter's increment, the
// compare of the shared `sel` against its own index -- a constant that
// differs, an input of the cell) becomes an instance of one module Sw_in. The
// value the parent reads (each r) is an output; the shared `x` and `sel`
// are inputs. The other register (q) stays.
// CHECK: outline-scopes: cell Sw_in (Sw g_in_*, 3 iterations, 5 ops, 3 inputs, 1 constant inputs, 1 outputs)
// CHECK-LABEL: hw.module @Sw_in(in %in0 : !seq.clock, in %in1 : i2, in %in2 : i8, in %k{{[0-9]+}} : i2, out out0 : i8)
// CHECK-DAG:     %c = seq.firreg {{%.+}} clock %in0 : i8
// CHECK-DAG:     %r = seq.firreg {{%.+}} clock %in0 : i8
// CHECK-LABEL: hw.module @Sw(
// CHECK:         %q = seq.firreg
// CHECK-DAG:     hw.instance "g_in_0.Sw_in" @Sw_in(in0: %clk: !seq.clock, in1: %sel: i2, in2: %x: i8, k{{[0-9]+}}: %c0_i2: i2)
// CHECK-DAG:     hw.instance "g_in_1.Sw_in" @Sw_in(in0: %clk: !seq.clock, in1: %sel: i2, in2: %x: i8, k{{[0-9]+}}: %c1_i2: i2)
// CHECK-DAG:     hw.instance "g_in_2.Sw_in" @Sw_in(in0: %clk: !seq.clock, in1: %sel: i2, in2: %x: i8, k{{[0-9]+}}: %c-2_i2: i2)
hw.module @Sw(in %clk : !seq.clock, in %sel : i2, in %x : i8, out o0 : i8, out o1 : i8, out o2 : i8, out o3 : i8) {
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 1 : i2
  %c2 = hw.constant 2 : i2
  %one = hw.constant 1 : i8
  %s0 = comb.icmp eq %sel, %c0 : i2
  %n0 = comb.add %c_0, %one : i8
  %c_0 = seq.firreg %n0 clock %clk {name = "g_in_0.c"} : i8
  %m0 = comb.mux %s0, %x, %c_0 : i8
  %r_0 = seq.firreg %m0 clock %clk {name = "g_in_0.r"} : i8
  %s1 = comb.icmp eq %sel, %c1 : i2
  %n1 = comb.add %c_1, %one : i8
  %c_1 = seq.firreg %n1 clock %clk {name = "g_in_1.c"} : i8
  %m1 = comb.mux %s1, %x, %c_1 : i8
  %r_1 = seq.firreg %m1 clock %clk {name = "g_in_1.r"} : i8
  %s2 = comb.icmp eq %sel, %c2 : i2
  %n2 = comb.add %c_2, %one : i8
  %c_2 = seq.firreg %n2 clock %clk {name = "g_in_2.c"} : i8
  %m2 = comb.mux %s2, %x, %c_2 : i8
  %r_2 = seq.firreg %m2 clock %clk {name = "g_in_2.r"} : i8
  %q = seq.firreg %x clock %clk : i8
  hw.output %r_0, %r_1, %r_2, %q : i8, i8, i8, i8
}
