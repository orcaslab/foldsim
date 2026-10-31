// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top zero-latency-links})" 2>&1 | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top zero-latency-links forward-after-levels=0 share-link-writes=false})" 2>&1 | FileCheck %s --check-prefix=PLAIN
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Queue,Sink netlist},foldsim-fold-static{module=Top})" 2>&1 | FileCheck %s --check-prefix=LATE

// Four queues load their register on a ready the parent (Top_model, one
// thread, its state pass at the frame's first slot) decides in the same
// cycle from their valids: each queue's link of its ready is taken around
// the read register in the parent's slot, and its early outputs (next
// values) depend on it.

// The schedule's links: forwarding only where a read register is loaded
// in the host cycle a write of its entry lands, past the read register where
// the reader's logic behind it is shallow; the bits several classes read
// written to their links from one register of them, a host cycle late.
// CHECK: link reads: 18, forwarded 2 (past the read register 1), taken now 1; written from a shared register 13 (forwarded 0)
// PLAIN: link reads: 18, forwarded 2 (past the read register 0), taken now 1; written from a shared register 0 (forwarded 0)

// Every link of a queue's output, read by the parent, a sink and a queue,
// written from one register of it, a host cycle after the state pass, at
// that pass's bank and instance a host cycle on.
// CHECK-LABEL: hw.module @foldsim_sim
// CHECK: %Queue_q_written_0 = seq.firreg %Queue_copy.q clock %hostClock : i8
// CHECK: %Queue_links0_state_late = seq.firreg %Queue_links0_state0
// CHECK: %[[AT:.+]] = comb.concat %Queue_links0_bank_late, %Queue_links0_inst_late
// CHECK: seq.firmem.write_port %Queue_q_to_Top_model_q0_q_0[%[[AT]]] = %Queue_q_written_0, clock %hostClock enable %Queue_links0_state_late
// The parent never reads one in the host cycle it lands: no forwarding into
// its read register.
// CHECK: %[[Q0:.+]] = seq.firmem.read_port %Queue_q_to_Top_model_q0_q_0
// CHECK-NEXT: %Top_model_q0_q_read = seq.firreg %[[Q0]] clock %hostClock : i8
// PLAIN-LABEL: hw.module @foldsim_sim
// PLAIN-NOT: _written_
// PLAIN: seq.firmem.write_port %Queue_q_to_Top_model_q0_q_0[%{{.+}}] = %{{.+}}, clock %hostClock enable %Queue_links0_state0
// PLAIN: %[[Q0:.+]] = seq.firmem.read_port %Queue_q_to_Top_model_q0_q_0
// PLAIN-NEXT: %Top_model_q0_q_read = seq.firreg %[[Q0]] clock %hostClock : i8

// The second queue's data, the parent's, read by the queues the host cycle
// after its write: forwarded past the read register, its hit (where the
// instance's source is the parent's) registered beside the read and the
// written bits from their register.
// CHECK: %Top_model_q1_d_written_0 = seq.firreg %Top_model_copy.q1_d
// CHECK: %Queue_d_select_after0_1 = seq.firreg
// CHECK: %[[MINE:.+]] = comb.icmp eq %Queue_d_select_after0_1, %c1_i2
// CHECK: %[[HIT:.+]] = comb.and %Top_model_links0_state0, %[[MINE]] : i1
// CHECK: %Queue_d_hit_0_1 = seq.firreg %[[HIT]]
// CHECK: %Queue_d_read = seq.firreg
// CHECK: comb.mux %Queue_d_hit_0_1, %Top_model_q1_d_written_0, %Queue_d_read : i8
// Into the read register without forwarding past it.
// PLAIN: %[[W:.+]] = comb.mux %Top_model_links0_state0, %Top_model_copy.q1_d, %Top_model_q1_d_to_Queue_d_0 : i8
// PLAIN: hw.array_create %{{.+}}, %{{.+}}, %[[W]], %{{.+}} : i8
// PLAIN-NOT: _hit_

// The ready links: only the first queue's is ever taken in the parent's
// slot (the first queue's, in that slot), so no select of the source; only
// the second queue's is forwarded.
// CHECK-NOT: select_now
// CHECK: %Top_model_q0_rdy_to_Queue_rdy_0 = seq.firreg %[[R0:.+]] clock
// CHECK-NEXT: %[[R0]] = comb.mux %Top_model_links0_state0, %Top_model_copy.q0_rdy, %Top_model_q0_rdy_to_Queue_rdy_0 : i1
// CHECK: %[[R1:.+]] = comb.mux %Top_model_links0_state0, %Top_model_copy.q1_rdy, %Top_model_q1_rdy_to_Queue_rdy_0 : i1
// CHECK: hw.array_create %Top_model_q3_rdy_to_Queue_rdy_0, %Top_model_q2_rdy_to_Queue_rdy_0, %[[R1]], %Top_model_q0_rdy_to_Queue_rdy_0 : i1
// CHECK: %Queue_rdy_read = seq.firreg
// CHECK-NEXT: comb.mux %Top_model_links0_state0, %Top_model_copy.q0_rdy, %Queue_rdy_read : i1

// Without zero-latency links the parent reads the last queue's output the
// host cycle its registered write lands: forwarded from the register.
// LATE: link reads: 18, forwarded 1 (past the read register 1), taken now 0; written from a shared register 13 (forwarded 3)
// LATE: %Queue_q_to_Top_model_q3_q_0 = seq.firmem
// LATE: %[[LAT:.+]] = comb.concat %Queue_links0_bank_late, %Queue_links0_inst_late
// LATE-NEXT: seq.firmem.write_port %Queue_q_to_Top_model_q3_q_0[%[[LAT]]] = %Queue_q_written_0
// LATE: %[[RD:.+]] = seq.firmem.read_port %Queue_q_to_Top_model_q3_q_0[%[[RAT:.+]]],
// LATE: %[[EQ:.+]] = comb.icmp eq %[[LAT]], %[[RAT]]
// LATE: %[[LHIT:.+]] = comb.and %Queue_links0_state_late, %[[EQ]]
// LATE: %[[LV:.+]] = comb.mux %[[LHIT]], %Queue_q_written_0, %[[RD]] : i8
// LATE: %Top_model_q3_q_read = seq.firreg %[[LV]]

hw.module @Queue(in %clk : !seq.clock, in %reset : i1, in %d : i8, in %rdy : i1, out q : i8, out v : i1) {
  %c0 = hw.constant 0 : i8
  %false = hw.constant false
  %c3 = hw.constant 3 : i8
  %dn = comb.add %d, %c3 : i8
  %qn = comb.mux %rdy, %dn, %q : i8
  %q = seq.firreg %qn clock %clk reset sync %reset, %c0 : i8
  %vn = comb.xor %v, %rdy : i1
  %v = seq.firreg %vn clock %clk reset sync %reset, %false : i1
  hw.output %q, %v : i8, i1
}
hw.module @Sink(in %clk : !seq.clock, in %reset : i1, in %a : i8, in %b : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %s = comb.add %r, %a, %b : i8
  %r = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z0 : i8, out z1 : i8) {
  %true = hw.constant true
  %c0 = hw.constant 0 : i8
  %c1 = hw.constant 1 : i8
  %q0.q, %q0.v = hw.instance "q0" @Queue(clk: %clk: !seq.clock, reset: %reset: i1, d: %x: i8, rdy: %r0: i1) -> (q: i8, v: i1)
  %d1 = comb.add %q0.q, %c1 : i8
  %q1.q, %q1.v = hw.instance "q1" @Queue(clk: %clk: !seq.clock, reset: %reset: i1, d: %d1: i8, rdy: %r1: i1) -> (q: i8, v: i1)
  %d2 = comb.xor %q1.q, %x : i8
  %q2.q, %q2.v = hw.instance "q2" @Queue(clk: %clk: !seq.clock, reset: %reset: i1, d: %d2: i8, rdy: %r2: i1) -> (q: i8, v: i1)
  %q3.q, %q3.v = hw.instance "q3" @Queue(clk: %clk: !seq.clock, reset: %reset: i1, d: %q2.q: i8, rdy: %r3: i1) -> (q: i8, v: i1)
  %n0 = comb.and %q0.v, %q1.v : i1
  %r0 = comb.xor %n0, %true : i1
  %r1 = comb.xor %q0.v, %q2.v : i1
  %n3 = comb.xor %q3.v, %true : i1
  %r2 = comb.or %n3, %q1.v : i1
  %n0b = comb.xor %q0.v, %true : i1
  %r3 = comb.and %q2.v, %n0b : i1
  %s = comb.add %acc, %q3.q, %q0.q : i8
  %acc = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  %s0.y = hw.instance "s0" @Sink(clk: %clk: !seq.clock, reset: %reset: i1, a: %q1.q: i8, b: %q3.q: i8) -> (y: i8)
  %s1.y = hw.instance "s1" @Sink(clk: %clk: !seq.clock, reset: %reset: i1, a: %q2.q: i8, b: %q0.q: i8) -> (y: i8)
  %z1 = comb.xor %s0.y, %s1.y : i8
  hw.output %acc, %z1 : i8, i8
}
