// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child netlist" | FileCheck %s --check-prefix=OFF
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child netlist direct-outputs" | FileCheck %s

// direct-outputs: a top output that only wires out a child's output (a port
// bundle passed straight out of the top: `q0`, `q1`, each child's own), a mix
// of children's bits, the top's inputs and constants (`mix`), or a model bit
// that is always a constant (`k`'s high bits), is wired so in the netlist,
// not passed through the top's model: the model gives a constant there and
// takes no input for those child outputs. An output with logic (`sum`) stays
// the model's, and so does a child output the model's logic reads.

hw.module private @Child(in %clk : !seq.clock, in %a : i8, out q : i8, out k : i4) {
  %c5_i2 = hw.constant 2 : i2
  %r = seq.firreg %a clock %clk : i8
  %lo = comb.extract %r from 0 : (i8) -> i2
  %k = comb.concat %c5_i2, %lo : i2, i2
  hw.output %r, %k : i8, i4
}

// Without the option the model passes the bundles on.
// OFF-LABEL: hw.module private @Top_model(
// OFF-SAME:    in %u0_q : i8
// OFF-SAME:    in %u1_q : i8
// OFF:         hw.output %u0_q, %u1_q,
// OFF-LABEL: hw.module @Top(
// OFF:         %logic.q0, %logic.q1, %logic.mix, %logic.k, %logic.sum = hw.instance "logic" @Top_model(
// OFF:         hw.output %logic.q0, %logic.q1, %logic.mix, %logic.k, %logic.sum

// With it, only u1.q is still read by the model (for `sum`).
// CHECK-LABEL: hw.module private @Top_model(in %clk : !seq.clock, in %x : i8, in %u1_q : i8, out q0 : i8, out q1 : i8, out mix : i8, out k : i4, out sum : i8)
// CHECK:       [[SUM:%.+]] = comb.add %x, %u1_q : i8
// CHECK:       hw.output {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, [[SUM]] : i8, i8, i8, i4, i8
// CHECK-LABEL: hw.module @Top(
// CHECK:         %logic.q0, %logic.q1, %logic.mix, %logic.k, %logic.sum = hw.instance "logic" @Top_model(
// CHECK:         %u0.q, %u0.k = hw.instance "u0" @Child(
// CHECK:         %u1.q, %u1.k = hw.instance "u1" @Child(
// CHECK-DAG:     [[XLO:%.+]] = comb.extract %x from 0 : (i8) -> i4
// CHECK-DAG:     [[QHI:%.+]] = comb.extract %u0.q from 4 : (i8) -> i4
// CHECK-DAG:     [[MIX:%.+]] = comb.concat [[QHI]], [[XLO]] : i4, i4
// CHECK-DAG:     [[KLO:%.+]] = comb.extract %u1.k from 0 : (i4) -> i2
// CHECK-DAG:     [[C2:%.+]] = hw.constant -2 : i2
// CHECK-DAG:     [[K:%.+]] = comb.concat [[C2]], [[KLO]] : i2, i2
// CHECK:         hw.output %u0.q, %u1.q, [[MIX]], [[K]], %logic.sum : i8, i8, i8, i4, i8
hw.module @Top(in %clk : !seq.clock, in %x : i8, out q0 : i8, out q1 : i8, out mix : i8, out k : i4, out sum : i8) {
  %u0.q, %u0.k = hw.instance "u0" @Child(clk: %clk: !seq.clock, a: %x: i8) -> (q: i8, k: i4)
  %u1.q, %u1.k = hw.instance "u1" @Child(clk: %clk: !seq.clock, a: %u0.q: i8) -> (q: i8, k: i4)
  %qhi = comb.extract %u0.q from 4 : (i8) -> i4
  %xlo = comb.extract %x from 0 : (i8) -> i4
  %mix = comb.concat %qhi, %xlo : i4, i4
  %sum = comb.add %x, %u1.q : i8
  hw.output %u0.q, %u1.q, %mix, %u1.k, %sum : i8, i8, i8, i4, i8
}
