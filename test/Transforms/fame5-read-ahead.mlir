// RUN: foldsim-opt %s --foldsim-fame5="module=Enable threads=4 clock-enable thread-index region-loads=1 registered-state" | FileCheck %s

// `b` is written when `a` is set. With registered state, `a`'s region, first
// by name, moves the uses of `a`'s read (at the thread after the current
// one) to its read-ahead register, the current thread's `a`, before `b`'s
// region stages `b`'s write enable from it: the staged enable takes the
// read-ahead register, not the read of the next thread.
// CHECK-LABEL: hw.module @Enable_threaded_Enable
// CHECK-DAG: %a_state = seq.firreg {{%.+}} clock %hostClock : i1
// CHECK-DAG: %b_wen = seq.firreg %a_state clock %hostClock
hw.module @Enable(in %clk : !seq.clock, in %x : i1, in %d : i8, out y : i8) {
  %a = seq.firreg %x clock %clk : i1
  %n = comb.mux %a, %d, %b : i8
  %b = seq.firreg %n clock %clk : i8
  hw.output %b : i8
}
