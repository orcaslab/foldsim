// RUN: foldsim-opt %s --foldsim-fame5="module=Pass threads=2" | FileCheck %s

// `o` is the input `i` wired through: not a channel of the model, its tokens
// go around it. `t`, of no bits, is a channel of tokens only, kept although
// nothing reads it; `u`, of bits nothing reads, is dropped.
// CHECK-LABEL: hw.module @Pass_threaded_Pass(
// CHECK-SAME:    out i_sink_ready : i1, in %i_sink_valid : i1, in %i_sink_bits : i4,
// CHECK-SAME:    out t_sink_ready : i1, in %t_sink_valid : i1,
// CHECK-SAME:    in %q_source_ready : i1, out q_source_valid : i1, out q_source_bits : i4)
// CHECK-SAME:    attributes {foldsim.bypass = ["o = i"]
// CHECK-NOT:     u_sink
hw.module @Pass(in %clk : !seq.clock, in %i : i4, in %t : i0, in %u : i8, out o : i4, out q : i4) {
  %r = seq.firreg %i clock %clk : i4
  %a = comb.extract %i from 2 : (i4) -> i2
  %b = comb.extract %i from 0 : (i4) -> i2
  %o = comb.concat %a, %b : i2, i2
  hw.output %o, %r : i4, i4
}
