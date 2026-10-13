// RUN: foldsim-opt %s --foldsim-fame5="module=Bus threads=1 channel-fields=r_fwd.bits_valid=rvalid,r_fwd.bits_bits_data=rdata,aw_fwd.bits_bits_addr=awaddr,aw_fwd.bits_valid=awvalid" --canonicalize | FileCheck %s

// Ports carried together as the fields of one channel: one handshake, a bits
// port per field, in the order given.
// CHECK-LABEL: hw.module @Bus_threaded_Bus(
// CHECK-SAME:    out r_fwd_sink_ready : i1, in %r_fwd_sink_valid : i1, in %r_fwd_sink_bits_valid : i1, in %r_fwd_sink_bits_bits_data : i8,
// CHECK-SAME:    in %aw_fwd_source_ready : i1, out aw_fwd_source_valid : i1, out aw_fwd_source_bits_bits_addr : i4, out aw_fwd_source_bits_valid : i1,
// CHECK-SAME:    in %q_source_ready : i1, out q_source_valid : i1, out q_source_bits : i8)
// CHECK:         comb.mux %r_fwd_sink_bits_valid, %r_fwd_sink_bits_bits_data, %r : i8
hw.module @Bus(in %clk : !seq.clock, in %rvalid : i1, in %rdata : i8, out awaddr : i4, out awvalid : i1, out q : i8) {
  %c5_i4 = hw.constant 5 : i4
  %r = seq.firreg %n clock %clk : i8
  %n = comb.mux %rvalid, %rdata, %r : i8
  %v = comb.extract %r from 0 : (i8) -> i1
  hw.output %c5_i4, %v, %r : i4, i1, i8
}
