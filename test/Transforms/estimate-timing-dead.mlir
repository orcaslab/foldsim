// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-timing{output=- report= top=Top})' -o /dev/null | FileCheck %s

// A register that only holds its value, whatever its enables, is no end
// point: Vivado removes it. A deep enable logic before it does not count.
// CHECK: levels,retimed,kind,start_kind,fanout,crossings,start_inst,start,end_inst,end
// CHECK-NOT: ,dead.d[
// CHECK-NOT: ,dead2.d[
// A register that loads a value under the same enable is live.
// CHECK: {{[0-9]+}},{{[0-9]+}},d,reg,{{[0-9]+}},0,0,{{[a-z]+}}[{{[0-9]+}}],0,live.d[0]
// CHECK-NOT: ,dead.d[
hw.module @Top(in %clk : !seq.clock, in %d : i8, out q : i1) {
  %a = seq.firreg %d clock %clk : i8
  %b = seq.firreg %d clock %clk : i8
  %prod = comb.mul %a, %b : i8
  %en = comb.icmp ugt %prod, %a : i8
  %en2 = comb.icmp ult %prod, %b : i8
  // Held through two enable multiplexers: dead.
  %dead_in = comb.mux %en, %dead, %dead : i1
  %dead_hold = comb.mux %en2, %dead_in, %dead : i1
  %dead = seq.firreg %dead_hold clock %clk : i1
  // Held directly: dead.
  %dead2 = seq.firreg %dead2 clock %clk : i1
  // Loads a new value under the enable: live.
  %a0 = comb.extract %a from 0 : (i8) -> i1
  %live_in = comb.mux %en, %a0, %live : i1
  %live = seq.firreg %live_in clock %clk : i1
  %o = comb.xor %dead, %dead2, %live : i1
  %out = seq.firreg %o clock %clk : i1
  hw.output %out : i1
}
