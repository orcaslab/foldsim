// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=virtexuplusHBM},foldsim-estimate-timing{output=- report= top=Top target=virtexuplusHBM})' -o /dev/null | FileCheck %s

// An enable computed from eight register bits is a LUT of its own, and it
// loads the last cell of each of the 16 bits it enables.
// CHECK: end_inst,end,in_fanout
// CHECK-DAG: ,q.d[0],16
// CHECK-DAG: ,q.d[15],16
// A register bit fed by another register's bit: no logic net into it.
// CHECK-DAG: ,p.d[0],0
hw.module @Top(in %clk : !seq.clock, in %d : i16, in %c : i8, out q : i16, out p : i1) {
  %x = seq.firreg %d clock %clk : i16
  %s = seq.firreg %c clock %clk : i8
  %en = comb.icmp eq %s, %c_all : i8
  %c_all = hw.constant -1 : i8
  %next = comb.mux %en, %x, %q : i16
  %q = seq.firreg %next clock %clk : i16
  %x0 = comb.extract %x from 0 : (i16) -> i1
  %p = seq.firreg %x0 clock %clk : i1
  hw.output %q, %p : i16, i1
}
