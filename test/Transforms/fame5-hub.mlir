// RUN: foldsim-opt %s --foldsim-fame5="module=Hub threads=1 clock-channel=clocks_0" --canonicalize | FileCheck %s

// One instance: its state stays in registers. Its target clock comes in on
// `clocks_0`, a channel always valid whose token enables the next edge.
// CHECK-LABEL: hw.module @Hub_threaded_Hub(
// CHECK-SAME:    out clocks_0_sink_ready : i1, in %clocks_0_sink_bits : i1,
// No output fires eagerly: whether a finish makes an edge is in the clock token.
// CHECK-SAME:    attributes {foldsim.eager_channels = []}
// CHECK-NOT:     seq.firmem
// CHECK-DAG:     %r = seq.firreg {{%.+}} clock %clk_buffer.O : i8
// The enabled edge is the last clock token's.
// CHECK-DAG:     %clocks_0_enabled = seq.firreg {{%.+}} clock %hostClock : i1
// CHECK-DAG:     [[EN:%.+]] = comb.mux [[FINISHING:%.+]], %clocks_0_sink_bits, %clocks_0_enabled : i1
// An input is fired from reset until the first clock token.
// CHECK-DAG:     [[A:%.+]] = comb.or %hostReset, {{%.+}} : i1
// CHECK-DAG:     %a_fired = seq.firreg [[A]] clock %hostClock : i1
// CHECK:         hw.instance "clk_buffer" @FireSimFabricClockGate(I: %hostClock: !seq.clock, CE: {{%.+}}: i1)
// CHECK:         hw.output [[FINISHING]],
hw.module @Hub(in %clk : !seq.clock, in %a : i8, in %en : i1, out q : i8, out s : i8) {
  %r = seq.firreg %n clock %clk : i8
  %n = comb.mux %en, %a, %r : i8
  %s = comb.add %a, %r : i8
  hw.output %r, %s : i8, i8
}
