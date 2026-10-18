// RUN: foldsim-opt %s --foldsim-fame5="module=Pipe threads=4 clock-enable thread-index fold-pipelines" | FileCheck %s
// RUN: foldsim-opt %s --foldsim-fame5="module=Pipe threads=4 clock-enable thread-index" | FileCheck %s --check-prefix=PLAIN
// RUN: foldsim-opt %s --foldsim-fame5="module=Pipe threads=1 clock-enable thread-index fold-pipelines" 2>&1 | FileCheck %s --check-prefix=ONE
// RUN: not foldsim-opt %s --foldsim-fame5="module=NoContract threads=4 clock-enable thread-index fold-pipelines" 2>&1 | FileCheck %s --check-prefix=NOCONTRACT

// Four stages on an enable. Folded: the first three stages are registers of
// the host clock, fed by the current thread's issue, registered; the fourth
// stays target state and takes its next value from a store of three slots
// per thread, at the thread's slot, its enabled edges modulo three. The store
// is written five host cycles after the issue (the thread's edge with the
// enable), at the issue's slot and thread.
// CHECK-LABEL: hw.module @Pipe_threaded_Pipe
// CHECK-DAG: %fold0_issue = seq.firreg %a_sink_bits clock %hostClock : i8
// CHECK-DAG: %s1_fold = seq.firreg %fold0_issue clock %hostClock : i8
// CHECK-DAG: %s2_fold = seq.firreg [[X:%.+]] clock %hostClock : i8
// CHECK-DAG: [[X]] = comb.xor %s1_fold, %c7_i8 : i8
// CHECK-DAG: %s3_fold = seq.firreg [[P:%.+]] clock %hostClock : i8
// CHECK-DAG: [[P]] = comb.add %s2_fold, %c7_i8 : i8
// CHECK-DAG: [[W:%.+]] = comb.mul %s3_fold, %c7_i8 : i8
// CHECK-DAG: %fold0_wdata = seq.firreg [[W]] clock %hostClock : i8
// CHECK-DAG: [[ISSUE:%.+]] = comb.and [[EDGE:%.+]], %en_sink_bits : i1
// CHECK-DAG: %fold0_valid1 = seq.firreg [[ISSUE]] clock %hostClock reset sync %hostReset, %false{{.*}} preset 0 : i1
// CHECK-DAG: %fold0_thread1 = seq.firreg %threadIdx clock %hostClock : i2
// CHECK-DAG: %fold0_valid5 = seq.firreg %fold0_valid4
// CHECK-DAG: %fold0_slot = seq.firmem 0, 1, undefined, undefined : <4 x 2, mask 1>
// CHECK-DAG: %fold0_store = seq.firmem 0, 1, undefined, undefined : <12 x 8, mask 1>
// CHECK-DAG: [[RADDR:%.+]] = comb.concat {{%.+}}, %threadIdx : i2, i2
// CHECK-DAG: [[STORED:%.+]] = seq.firmem.read_port %fold0_store[[[RADDR]]], clock %hostClock
// CHECK-DAG: [[WADDR:%.+]] = comb.concat %fold0_slot5, %fold0_thread5 : i2, i2
// CHECK-DAG: seq.firmem.write_port %fold0_store[[[WADDR]]] = %fold0_wdata, clock %hostClock enable %fold0_valid5
// CHECK-DAG: %s4 = seq.firmem 0, 1, undefined, undefined : <4 x 8, mask 1>
// The exactness condition, checked in simulation.
// CHECK-DAG: sv.ifdef @SYNTHESIS {
// CHECK-DAG: sv.fatal.procedural 1, "foldsim: a thread is current on the host cycle after its edge
// CHECK-NOT: %s1 = seq.firmem
// CHECK-NOT: %s2 = seq.firmem
// CHECK-NOT: %s3 = seq.firmem

// Without folding, every stage is a thread memory.
// PLAIN-DAG: %s1 = seq.firmem
// PLAIN-DAG: %s2 = seq.firmem
// PLAIN-DAG: %s3 = seq.firmem
// PLAIN-DAG: %s4 = seq.firmem
// PLAIN-NOT: fold

// One thread: nothing is folded, and why is said.
// ONE: remark: pipelines of 'Pipe' not folded: its one thread is current on every host cycle
// ONE-NOT: fold0

// NOCONTRACT: error: black box 'Opaque' in a threaded model has no pipeline contract

hw.module @Pipe(in %clk : !seq.clock, in %en : i1, in %a : i8, out y : i8) {
  %c7 = hw.constant 7 : i8
  %n1 = comb.mux %en, %a, %s1 : i8
  %s1 = seq.firreg %n1 clock %clk : i8
  %x = comb.xor %s1, %c7 : i8
  %n2 = comb.mux %en, %x, %s2 : i8
  %s2 = seq.firreg %n2 clock %clk : i8
  %p = comb.add %s2, %c7 : i8
  %n3 = comb.mux %en, %p, %s3 : i8
  %s3 = seq.firreg %n3 clock %clk : i8
  %m = comb.mul %s3, %c7 : i8
  %n4 = comb.mux %en, %m, %s4 : i8
  %s4 = seq.firreg %n4 clock %clk : i8
  hw.output %s4 : i8
}

hw.module.extern @Opaque(in %clk : !seq.clock, in %d : i8, out q : i8) attributes {sv.attributes = [#sv.attribute<"black_box">]}
hw.module @NoContract(in %clk : !seq.clock, in %a : i8, out y : i8) {
  %q = hw.instance "u" @Opaque(clk: %clk: !seq.clock, d: %a: i8) -> (q: i8)
  hw.output %q : i8
}
