// RUN: foldsim-opt %s --foldsim-fame5="module=P threads=1" | FileCheck %s --check-prefix=ONE
// RUN: foldsim-opt %s --foldsim-fame5="module=P threads=4" | FileCheck %s

// A register's preset value is its value before its first edge. It takes it
// on reset too, as a thread's state does: the target starts anew on each.

// One thread keeps it a register with the preset value, which also takes it
// on reset; it is written on the host clock, on the target clock's edges.
// ONE-LABEL: hw.module @P_threaded_P(
// ONE:       %r = seq.firreg [[NEXT:%.+]] clock %hostClock preset 1 : i1
// ONE:       [[DATA:%.+]] = comb.mux %hostReset, %true, %d_sink_bits : i1
// ONE:       [[WRITE:%.+]] = comb.or %hostReset, [[EDGE:%.+]] : i1
// ONE:       [[NEXT]] = comb.mux [[WRITE]], [[DATA]], %r : i1
// ONE:       [[EDGE]] = comb.and {{%.+}}, {{%.+}} : i1
// ONE:       hw.instance "clk_buffer" @ClockGate(I: %hostClock: !seq.clock, CE: [[EDGE]]: i1)

// With more threads, each thread's entry takes the preset value as the
// threads are initialized, from reset on, one per host cycle.
// CHECK-LABEL: hw.module @P_threaded_P(
// CHECK:       %init_sweep = seq.firreg
// CHECK:       [[INIT:%.+]] = comb.or %hostReset, %init_sweep : i1
// CHECK:       %r = seq.firmem 0, 1, undefined, undefined : <4 x 1, mask 1>
// CHECK:       [[DATA:%.+]] = comb.mux [[INIT]], %true{{.*}}, %d_sink_bits : i1
// CHECK:       [[WRITE:%.+]] = comb.or [[INIT]], [[EDGE:%.+]] : i1
// CHECK:       seq.firmem.write_port %r[%threadIdx] = [[DATA]], clock %hostClock enable [[WRITE]]
// CHECK:       [[EDGE]] = comb.and {{%.+}}, {{%.+}} : i1
// CHECK:       hw.instance "clk_buffer" @ClockGate(I: %hostClock: !seq.clock, CE: [[EDGE]]: i1)
hw.module @P(in %clk : !seq.clock, in %d : i1, out q : i1) {
  %r = seq.firreg %d clock %clk preset 1 : i1
  hw.output %r : i1
}

// RUN: foldsim-opt %s --foldsim-fame5="module=Held threads=4" | FileCheck %s --check-prefix=HELD

// A register that only holds its preset value is written only then.
// HELD-LABEL: hw.module @Held_threaded_Held(
// HELD:       [[INIT:%.+]] = comb.or %hostReset, %init_sweep : i1
// HELD:       seq.firmem.write_port %r[%threadIdx] = %c-6_i4, clock %hostClock enable [[INIT]]
hw.module @Held(in %clk : !seq.clock, out q : i4) {
  %r = seq.firreg %r clock %clk preset 10 : i4
  hw.output %r : i4
}
