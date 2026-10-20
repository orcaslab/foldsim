// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=4 clock-enable thread-index region-loads=16 registered-state" | FileCheck %s
// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=1 clock-enable commit-select" | FileCheck %s --check-prefix=ONE
// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=4 clock-enable region-loads=16 registered-state state-stages=2" | FileCheck %s --check-prefix=DEEP
// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=2 clock-enable region-loads=16 registered-state" | FileCheck %s --check-prefix=TWO
// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=3 clock-enable region-loads=16 registered-state state-stages=2" | FileCheck %s --check-prefix=DEEP3
// RUN: not foldsim-opt %s --foldsim-fame5="module=Hold threads=4 clock-enable registered-state state-stages=3" 2>&1 | FileCheck %s --check-prefix=BAD

// With two stages to the state's write, the write data, enable and index go
// through a second register each. With four threads in turn the write is in
// the memory before the thread's state is read ahead again.
// DEEP-LABEL: hw.module @Hold_threaded_Hold(
// DEEP-DAG: seq.firmem.write_port %r[%threadIdx_v0] = %r_wdata2, clock %hostClock enable %r_wen2 :
// DEEP-DAG: %threadIdx_v0 = seq.firreg %threadIdx_w0 clock %hostClock preset 0 {sv.attributes = [#sv.attribute<"keep" = "\22true\22">]} : i2
// DEEP-DAG: %r_wdata2 = seq.firreg %r_wdata clock %hostClock : i8
// DEEP-DAG: [[DREAD:%.+]] = seq.firmem.read_port %r[%threadIdx_a0]
// DEEP-DAG: %r_state = seq.firreg [[DREAD]] clock %hostClock : i8
// With three, the thread after next is the one whose write lands now: the
// read ahead forwards it.
// DEEP3-LABEL: hw.module @Hold_threaded_Hold(
// DEEP3-DAG: [[DSAME:%.+]] = comb.icmp eq %threadIdx_v0, %threadIdx_a0 : i2
// DEEP3-DAG: [[DHIT:%.+]] = comb.and %r_wen2, [[DSAME]] : i1
// DEEP3-DAG: [[DNEXT:%.+]] = comb.mux [[DHIT]], %r_wdata2, {{%.+}} : i8
// DEEP3-DAG: %r_state = seq.firreg [[DNEXT]] clock %hostClock : i8
// BAD: a thread's state is written one or two host cycles after its edge

// With registered state the next thread is chosen a host cycle earlier: the
// current thread is the register `threadIdx_ahead` was, and the state is read
// for that next thread into a register, `<name>_state`, which the target
// logic reads. With four threads in turn, a thread's write has landed by then.
// CHECK-LABEL: hw.module @Hold_threaded_Hold(
// CHECK-DAG: %threadIdx = seq.firreg %threadIdx_ahead clock %hostClock reset sync %hostReset, %c0_i2 preset 0 : i2
// CHECK-DAG: %threadIdx_ahead = seq.firreg {{%.+}} clock %hostClock reset sync %hostReset, %c1_i2 preset 1 : i2
// CHECK-DAG: [[READ:%.+]] = seq.firmem.read_port %r[%threadIdx_a0]
// CHECK-DAG: seq.firmem.read_port %reset_fired[%threadIdx]
// CHECK-DAG: %threadIdx_a0 = seq.firreg {{%.+}} clock %hostClock reset sync %hostReset, %{{.+}} preset 1 {sv.attributes = [#sv.attribute<"keep" = "\22true\22">]} : i2
// CHECK-DAG: %r_state = seq.firreg [[READ]] clock %hostClock : i8
// CHECK-DAG: comb.mux %en_sink_bits, %d_sink_bits, %r_state : i8

// With two threads the other thread runs between, and the write landing on
// this cycle's edge is the next thread's: it is forwarded to the read ahead,
// the select built from the region's copies of the edge and of the indexes.
// TWO-LABEL: hw.module @Hold_threaded_Hold(
// TWO-DAG: [[TREAD:%.+]] = seq.firmem.read_port %r[%threadIdx_a0]
// TWO-DAG: [[SAME:%.+]] = comb.icmp eq %threadIdx_w0, %threadIdx_a0 : i1
// TWO-DAG: [[HIT:%.+]] = comb.and {{%.+}}, [[SAME]] : i1
// TWO-DAG: [[NEXT:%.+]] = comb.mux [[HIT]], %r_wdata, [[TREAD]] : i8
// TWO-DAG: %r_state = seq.firreg [[NEXT]] clock %hostClock : i8

// One thread commits by select: the candidate and its enable are registers,
// and the state is the candidate when the registered enable says so, the
// committed value otherwise.
// ONE-LABEL: hw.module @Hold_threaded_Hold(
// ONE-DAG: %edge_r0 = seq.firreg {{%.+}} clock %hostClock preset 0 {sv.attributes = [#sv.attribute<"keep" = "\22true\22">]} : i1
// ONE-DAG: %r_next = seq.firreg %d_sink_bits clock %hostClock : i8
// ONE-DAG: %r_wen = seq.firreg %en_sink_bits clock %hostClock preset 0 : i1
// ONE-DAG: [[EN:%.+]] = comb.and %r_wen, %edge_r0 : i1
// ONE-DAG: %r = seq.firreg [[STATE:%.+]] clock %hostClock : i8
// ONE-DAG: [[STATE]] = comb.mux [[EN]], %r_next, %r : i8
// ONE-DAG: %s = seq.firreg {{%.+}} clock %hostClock preset 3 : i8
// ONE-DAG: comb.or %init_r0,
hw.module @Hold(in %clk : !seq.clock, in %reset : i1, in %en : i1, in %d : i8, out q : i8, out p : i8) {
  %c0_i8 = hw.constant 0 : i8
  %m = comb.mux %en, %d, %r : i8
  %r = seq.firreg %m clock %clk : i8
  %n = comb.mux %en, %d, %s : i8
  %s = seq.firreg %n clock %clk reset sync %reset, %c0_i8 preset 3 : i8
  hw.output %r, %s : i8, i8
}
