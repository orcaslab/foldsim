// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-lane-scan{module=T},canonicalize,cse)' 2>&1 | FileCheck %s

// Four instances of L in the unfolded T are lanes. The per-lane AND moves
// into the lane cell T_lane_L; each reduction becomes a chain through the
// four cells (ch<k>_in of cell j is ch<k>_out of cell j - 1), T keeping the
// chains' ends:
// - ac: the XOR of the lanes' terms, T's own operand applied at the end;
// - prio: lanes 0-2 in priority order (lane 3 disabled: its enable `pl` is
//   false), each lane's value a per-lane constant, T's default 3 when none;
// - select: the lane whose array index equals T's index (per-lane position);
// - carry: `vec - 1` on a packed lane vector rippling a carry through the
//   lanes, its per-lane result one-hot masked, then AND-reduced (vec).

// CHECK: lane-scan: cell T_lane_L
// CHECK: lane-scan: T lanes L x4: {{.*}}scans ac 1 carry 1 prio 1 select 1 vec 1 (5 chains, 21 bits)

// The lane's only per-lane input is its index `lane`; the per-lane constants
// (the prio enable and values, the select positions) are functions of it.
// CHECK-LABEL: hw.module @T_lane_L(in %in_o : i8, in %in_v : i1, in %ch0_in : i1, in %ch1_in : i8, in %ch2_in : i3, in %lane : i2, in %ch3_in : i8, in %bc7 : i2, in %ch4_in : i1, out ch0_out : i1, out ch1_out : i8, out ch2_out : i3, out ch3_out : i8, out ch4_out : i1)
// The carry: {cout, bit} = v + cin + 1.
// CHECK: %[[CIN:.+]] = comb.concat %false, %ch0_in
// CHECK: %[[SUM:.+]] = comb.add %{{.+}}, %[[CIN]], %c1_i2 : i2
// CHECK: %[[BIT:.+]] = comb.extract %[[SUM]] from 0
// CHECK: %[[COUT:.+]] = comb.extract %[[SUM]] from 1
// CHECK: %[[NBIT:.+]] = comb.xor %[[BIT]], %true
// The XOR scan.
// CHECK: %[[ACC:.+]] = comb.xor %ch1_in, %{{.+}} : i8
// The priority scan: {found, value}, first wins, lanes 0-2 enabled.
// CHECK: %[[FOUND:.+]] = comb.extract %ch2_in from 2
// CHECK: %[[VAL:.+]] = comb.extract %ch2_in from 0
// CHECK: %[[L3:.+]] = comb.concat %false, %lane
// CHECK: %[[EN:.+]] = comb.icmp ult %[[L3]], %c3_i3
// CHECK: %[[HIT:.+]] = comb.and %in_v, %[[EN]]
// CHECK: %[[NOTF:.+]] = comb.xor %[[FOUND]], %true
// CHECK: %[[TAKE:.+]] = comb.and %[[HIT]], %[[NOTF]]
// CHECK: %[[NVAL:.+]] = comb.mux %[[TAKE]], %lane, %[[VAL]]
// CHECK: %[[NF:.+]] = comb.or %[[FOUND]], %[[HIT]]
// CHECK: %[[PRIO:.+]] = comb.concat %[[NF]], %[[NVAL]]
// The select scan.
// CHECK: %[[EQ:.+]] = comb.icmp eq %bc7, %lane
// CHECK: %[[SEL:.+]] = comb.mux %[[EQ]], %in_o, %ch3_in
// The AND of the one-hot lane bits.
// CHECK: %[[ALL:.+]] = comb.and %ch4_in, %in_v, %[[NBIT]]
// CHECK: hw.output %[[COUT]], %[[ACC]], %[[PRIO]], %[[SEL]], %[[ALL]]

// CHECK-LABEL: hw.module @T(
// CHECK-COUNT-4: hw.instance "u{{[0-3]}}" @L(
// CHECK: hw.instance "lane_L_0" @T_lane_L(in_o: %u0.o: i8, in_v: %u0.v: i1, ch0_in: %false: i1, ch1_in: %c0_i8: i8, ch2_in: %c0_i3: i3, lane: %c0_i2: i2, ch3_in: %c0_i8: i8, bc7: %idx: i2, ch4_in: %true: i1)
// CHECK: hw.instance "lane_L_1" @T_lane_L(in_o: %u1.o: i8, in_v: %u1.v: i1, ch0_in: %lane_L_0.ch0_out: i1, ch1_in: %lane_L_0.ch1_out: i8, ch2_in: %lane_L_0.ch2_out: i3, lane: %c1_i2: i2, ch3_in: %lane_L_0.ch3_out: i8, bc7: %idx: i2, ch4_in: %lane_L_0.ch4_out: i1)
// CHECK: hw.instance "lane_L_2" @T_lane_L(
// CHECK: hw.instance "lane_L_3" @T_lane_L(in_o: %u3.o: i8, in_v: %u3.v: i1, ch0_in: %lane_L_2.ch0_out: i1, ch1_in: %lane_L_2.ch1_out: i8, ch2_in: %lane_L_2.ch2_out: i3, lane: %c-1_i2: i2, ch3_in: %lane_L_2.ch3_out: i8, bc7: %idx: i2, ch4_in: %lane_L_2.ch4_out: i1)
// CHECK: %[[Y:.+]] = comb.xor %x, %lane_L_3.ch1_out : i8
// CHECK: %[[F:.+]] = comb.extract %lane_L_3.ch2_out from 2
// CHECK: %[[V:.+]] = comb.extract %lane_L_3.ch2_out from 0
// CHECK: %[[E:.+]] = comb.mux %[[F]], %[[V]], %c-1_i2
// CHECK: hw.output %[[Y]], %[[E]], %lane_L_3.ch3_out, %lane_L_3.ch4_out, %{{.+}} : i8, i2, i8, i1, i1

hw.module private @L(in %a : i8, out o : i8, out v : i1) {
  %0 = comb.extract %a from 0 : (i8) -> i1
  hw.output %a, %0 : i8, i1
}
hw.module @T(in %x : i8, in %idx : i2, out y : i8, out e : i2, out s : i8, out z : i1, out d0 : i1) {
  %u0.o, %u0.v = hw.instance "u0" @L(a: %x: i8) -> (o: i8, v: i1)
  %u1.o, %u1.v = hw.instance "u1" @L(a: %x: i8) -> (o: i8, v: i1)
  %u2.o, %u2.v = hw.instance "u2" @L(a: %x: i8) -> (o: i8, v: i1)
  %u3.o, %u3.v = hw.instance "u3" @L(a: %x: i8) -> (o: i8, v: i1)
  %c3 = hw.constant 3 : i8
  // ac: XOR of a per-lane AND with a constant, and the parent's input
  %t0 = comb.and %u0.o, %c3 : i8
  %t1 = comb.and %u1.o, %c3 : i8
  %t2 = comb.and %u2.o, %c3 : i8
  %t3 = comb.and %u3.o, %c3 : i8
  %y = comb.xor %t0, %t1, %t2, %t3, %x : i8
  // prio: the lowest lane whose v is set, 3 by default
  %k0 = hw.constant 0 : i2
  %k1 = hw.constant 1 : i2
  %k2 = hw.constant 2 : i2
  %k3 = hw.constant 3 : i2
  %m2 = comb.mux %u2.v, %k2, %k3 : i2
  %m1 = comb.mux %u1.v, %k1, %m2 : i2
  %e = comb.mux %u0.v, %k0, %m1 : i2
  // select: the lane at the parent's index
  %arr = hw.array_create %u3.o, %u2.o, %u1.o, %u0.o : i8
  %s = hw.array_get %arr[%idx] : !hw.array<4xi8>, i2
  // vec + carry: a packed lane vector, its lowest set bit, AND-reduced
  %vec = comb.concat %u3.v, %u2.v, %u1.v, %u0.v : i1, i1, i1, i1
  %cm1 = hw.constant -1 : i4
  %dec = comb.add %vec, %cm1 : i4
  %low = comb.xor %dec, %cm1 : i4
  %oh = comb.and %vec, %low : i4
  %z = comb.icmp eq %oh, %cm1 : i4
  // a decode of the parent's index read by lane 0's input alone
  %d0 = comb.icmp eq %idx, %k0 : i2
  hw.output %y, %e, %s, %z, %d0 : i8, i2, i8, i1, i1
}
