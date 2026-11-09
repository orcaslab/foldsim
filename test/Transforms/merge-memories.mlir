// RUN: foldsim-opt %s --pass-pipeline="builtin.module(hw.module(foldsim-merge-memories{max-bits=40}))" | FileCheck %s

// `a` and `b` read and write at copies of the same registers: one memory,
// `a` in the high bits. `c` reads at another address and stays. `d` would
// make the merged memory wider than 40 bits and stays. `e` is masked.
// CHECK-LABEL: hw.module @Links
// CHECK: %[[M:.+]] = seq.firmem 0, 1, undefined, undefined : <4 x 24, mask 1>
// CHECK: seq.firmem.write_port %[[M]][{{.*}}] = %{{.+}}, clock %clk enable %{{.+}} : <4 x 24, mask 1>
// CHECK: %[[R:.+]] = seq.firmem.read_port %[[M]][
// CHECK-DAG: comb.extract %[[R]] from 16 : (i24) -> i8
// CHECK-DAG: comb.extract %[[R]] from 0 : (i24) -> i16
// CHECK-DAG: seq.firmem 0, 1, undefined, undefined : <4 x 8, mask 1>
// CHECK-DAG: seq.firmem 0, 1, undefined, undefined : <4 x 20, mask 1>
// CHECK-DAG: seq.firmem 0, 1, undefined, undefined : <4 x 16, mask 2>
hw.module @Links(in %clk : !seq.clock, in %wa : i2, in %ra : i2, in %rb : i2, in %we : i1,
                 in %x : i8, in %y : i16, in %z : i8, in %u : i20, in %v : i16, in %m : i2,
                 out a : i8, out b : i16, out c : i8, out d : i20, out e : i16) {
  %true = hw.constant true
  %wa1 = seq.firreg %wa clock %clk {name = "a_wa"} : i2
  %wa2 = seq.firreg %wa clock %clk {name = "b_wa"} : i2
  %we1 = seq.firreg %we clock %clk {name = "a_we"} : i1
  %we2 = seq.firreg %we clock %clk {name = "b_we"} : i1
  %ra1 = seq.firreg %ra clock %clk {name = "a_ra"} : i2
  %ra2 = seq.firreg %ra clock %clk {name = "b_ra"} : i2
  %ma = seq.firmem 0, 1, undefined, undefined : <4 x 8, mask 1>
  seq.firmem.write_port %ma[%wa1] = %x, clock %clk enable %we1 : <4 x 8, mask 1>
  %a = seq.firmem.read_port %ma[%ra1], clock %clk enable %true : <4 x 8, mask 1>
  %mb = seq.firmem 0, 1, undefined, undefined : <4 x 16, mask 1>
  seq.firmem.write_port %mb[%wa2] = %y, clock %clk enable %we2 : <4 x 16, mask 1>
  %b = seq.firmem.read_port %mb[%ra2], clock %clk enable %true : <4 x 16, mask 1>
  %mc = seq.firmem 0, 1, undefined, undefined : <4 x 8, mask 1>
  seq.firmem.write_port %mc[%wa1] = %z, clock %clk enable %we1 : <4 x 8, mask 1>
  %c = seq.firmem.read_port %mc[%rb], clock %clk enable %true : <4 x 8, mask 1>
  %md = seq.firmem 0, 1, undefined, undefined : <4 x 20, mask 1>
  seq.firmem.write_port %md[%wa1] = %u, clock %clk enable %we1 : <4 x 20, mask 1>
  %d = seq.firmem.read_port %md[%ra1], clock %clk enable %true : <4 x 20, mask 1>
  %me = seq.firmem 0, 1, undefined, undefined : <4 x 16, mask 2>
  seq.firmem.write_port %me[%wa1] = %v, clock %clk enable %we1 mask %m : <4 x 16, mask 2>, i2
  %e = seq.firmem.read_port %me[%ra1], clock %clk enable %true : <4 x 16, mask 2>
  hw.output %a, %b, %c, %d, %e : i8, i16, i8, i20, i16
}
