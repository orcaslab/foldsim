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
// CHECK-SAME:    attributes {foldsim.eager_channels = ["y"]}

// `z` is the register's value and `a`; the register's next value is its value
// plus `a`, or 0 on reset.
// CHECK: [[SUM:%.+]] = comb.add [[ACC:%.+]], %a_sink_bits : i8
// CHECK: [[Z_BITS:%.+]] = comb.and [[ACC]], %a_sink_bits : i8

// The current thread advances every host cycle, from thread 0 after reset.
// Every thread's state is initialized on reset and on the host cycles after
// it, one thread each, up to the last: no thread runs until then.
// CHECK: %threadIdx = seq.firreg [[IDX_NEXT:%.+]] clock %hostClock reset sync %hostReset, %c0_i2 preset 0 : i2
// CHECK: %init_sweep = seq.firreg [[SWEEP_NEXT:%.+]] clock %hostClock reset sync %hostReset, %true preset 1 : i1
// CHECK: [[IS_LAST:%.+]] = comb.icmp eq %threadIdx, %c-1_i2 : i2
// CHECK: [[NOT_LAST:%.+]] = comb.xor [[IS_LAST]], %true{{.*}} : i1
// CHECK: [[SWEEP_NEXT]] = comb.and %init_sweep, [[NOT_LAST]] : i1
// CHECK: [[INIT:%.+]] = comb.or %hostReset, %init_sweep : i1
// CHECK: [[ACC_NEXT:%.+]] = comb.mux %reset_sink_bits, %c0_i8, [[SUM]] : i8

// The register keeps one value per thread, written on the gated clock with
// its reset folded in.
// CHECK: %acc = seq.firmem 0, 1, undefined, undefined : <4 x 8, mask 1>
// CHECK: [[ACC]] = seq.firmem.read_port %acc[%threadIdx], clock %hostClock
// CHECK: seq.firmem.write_port %acc[%threadIdx] = {{%.+}}, clock %clk_buffer.O

// So do the channel flags, on the host clock.
// CHECK-COUNT-2: seq.firmem 0, 1, undefined, undefined : <4 x 1, mask 1>
// CHECK: %y_fired = seq.firmem 0, 1, undefined, undefined : <4 x 1, mask 1>
// CHECK: [[Y_FIRED:%.+]] = seq.firmem.read_port %y_fired[%threadIdx]
// CHECK: %z_fired = seq.firmem 0, 1, undefined, undefined : <4 x 1, mask 1>
// CHECK: %clk_enabled = seq.firmem 0, 1, undefined, undefined : <4 x 1, mask 1>
// CHECK: [[CLK_ENABLED:%.+]] = seq.firmem.read_port %clk_enabled[%threadIdx]

// No thread finishes, nor offers a token, while they are initialized.
// CHECK: [[RUNNING:%.+]] = comb.xor [[INIT]], %true{{.*}} : i1
// CHECK: [[FINISHING:%.+]] = comb.and {{.*}}, %reset_sink_valid, %a_sink_valid, [[RUNNING]] : i1
// CHECK: [[Y_VALID:%.+]] = comb.and {{%.+}}, [[RUNNING]] : i1
// CHECK: [[Z_VALID:%.+]] = comb.and {{%.+}}, [[RUNNING]] : i1
// CHECK: [[IDX_NEXT]] = comb.mux

// While initialized, each thread's channel flags are cleared as it is current.
// CHECK: [[Y_FIRED_NEXT:%.+]] = comb.mux [[INIT]], %false, {{%.+}} : i1
// CHECK: seq.firmem.write_port %y_fired[%threadIdx] = [[Y_FIRED_NEXT]], clock %hostClock
// CHECK: seq.firmem.write_port %a_fired[%threadIdx]
// CHECK: [[CLK_ENABLED_NEXT:%.+]] = comb.mux [[INIT]], %false, {{%.+}} : i1
// CHECK: seq.firmem.write_port %clk_enabled[%threadIdx] = [[CLK_ENABLED_NEXT]], clock %hostClock

// The target clock is the host clock, enabled when the thread finishes.
// CHECK: [[EDGE:%.+]] = comb.and [[CLK_ENABLED]], [[FINISHING]] : i1
// CHECK: %clk_buffer.O = hw.instance "clk_buffer" @ClockGate(I: %hostClock: !seq.clock, CE: [[EDGE]]: i1) -> (O: !seq.clock)

// `y` depends on no input, so it fires eagerly, with the next value once the
// thread has finished, if its target clock ticks then; `z` depends on `a` and
// waits for it.
// CHECK: [[Y_AHEAD:%.+]] = comb.and [[Y_FIRED]], [[CLK_ENABLED]] : i1
// CHECK: [[Y_BITS:%.+]] = comb.mux [[Y_AHEAD]], [[ACC_NEXT]], [[ACC]] : i8
// CHECK: hw.output {{%.+}}, {{%.+}}, [[Y_VALID]], [[Y_BITS]], [[Z_VALID]], [[Z_BITS]] : i1, i1, i1, i8, i1, i8
hw.module @Model(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8, out z : i8) {
  %c0_i8 = hw.constant 0 : i8
  %sum = comb.add %acc, %a : i8
  %acc = seq.firreg %sum clock %clk reset sync %reset, %c0_i8 : i8
  %z = comb.and %acc, %a : i8
  hw.output %acc, %z : i8, i8
}

