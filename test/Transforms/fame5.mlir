// RUN: foldsim-opt %s --foldsim-fame5="module=Model threads=4" | FileCheck %s

// The clock gate the platform provides.
// CHECK: hw.module.extern @ClockGate(in %I : !seq.clock, in %CE : i1, out O : !seq.clock)

// The threaded model: the host clock and reset, then a decoupled channel per
// target port, inputs first.
// CHECK-LABEL: hw.module @Model_threaded_Model(
// CHECK-SAME:    in %hostClock : !seq.clock, in %hostReset : i1,
// CHECK-SAME:    out reset_sink_ready : i1, in %reset_sink_valid : i1, in %reset_sink_bits : i1,
// CHECK-SAME:    out a_sink_ready : i1, in %a_sink_valid : i1, in %a_sink_bits : i8,
// CHECK-SAME:    in %y_source_ready : i1, out y_source_valid : i1, out y_source_bits : i8,
// CHECK-SAME:    in %z_source_ready : i1, out z_source_valid : i1, out z_source_bits : i8)

// The current thread advances every host cycle.
// CHECK: %threadIdx = seq.firreg {{%.+}} clock %hostClock : i2

// The register keeps one value per thread, written on the gated clock with
// its reset folded in.
// CHECK: %acc = seq.firmem 0, 1, undefined, undefined : <4 x 8, mask 1>
// CHECK: [[ACC:%.+]] = seq.firmem.read_port %acc[%threadIdx], clock %hostClock
// CHECK: seq.firmem.write_port %acc[%threadIdx] = {{%.+}}, clock %clk_buffer.O

// So do the channel flags, on the host clock.
// CHECK-COUNT-4: seq.firmem 0, 1, undefined, undefined : <4 x 1, mask 1>
// CHECK: %clk_enabled = seq.firmem 0, 1, undefined, undefined : <4 x 1, mask 1>

// The target clock is the host clock, enabled when the thread finishes.
// CHECK: %clk_buffer.O = hw.instance "clk_buffer" @ClockGate(I: %hostClock: !seq.clock, CE: {{%.+}}: i1) -> (O: !seq.clock)

// `y` depends on no input, so it fires eagerly, with the next value once the
// thread has finished; `z` depends on `a` and waits for it.
// CHECK: hw.output {{.*}}, [[Y_BITS:%.+]], {{.*}}, [[Z_BITS:%.+]] : i1, i1, i1, i8, i1, i8
hw.module @Model(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8, out z : i8) {
  %c0_i8 = hw.constant 0 : i8
  %sum = comb.add %acc, %a : i8
  %acc = seq.firreg %sum clock %clk reset sync %reset, %c0_i8 : i8
  %z = comb.and %acc, %a : i8
  hw.output %acc, %z : i8, i8
}

