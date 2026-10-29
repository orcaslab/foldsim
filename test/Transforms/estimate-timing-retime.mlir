// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-timing{output=- report= top=Top retime-levels=4})' -o /dev/null | FileCheck %s

// Three stages of six LUTs each, a -> f -> g -> h, retimed to four levels.
// f's start register a moves forward: its input stage is empty.
// CHECK-DAG: 6,4,d,reg,{{[0-9]+}},2,0,a[0],0,f.d[0]
// g has no room on either side: f's input stage and g's output stage are
// six levels deep.
// CHECK-DAG: 6,6,d,reg,{{[0-9]+}},2,0,f[0],0,g.d[0]
// h moves backward: nothing follows it.
// CHECK-DAG: 6,4,d,reg,{{[0-9]+}},2,0,g[0],0,h.d[0]
// n can't move backward through the distributed RAM, and f can't move
// forward.
// CHECK-DAG: 6,6,d,reg,{{[0-9]+}},2,0,f[0],0,n.d[0]
hw.module private @Chain(in %x : i1, out o : i1) {
  %1 = comb.truth_table %x -> [true, false]
  %2 = comb.truth_table %1 -> [true, false]
  %3 = comb.truth_table %2 -> [true, false]
  %4 = comb.truth_table %3 -> [true, false]
  %5 = comb.truth_table %4 -> [true, false]
  %6 = comb.truth_table %5 -> [true, false]
  hw.output %6 : i1
}
hw.module private @Chain5(in %x : i1, out o : i1) {
  %1 = comb.truth_table %x -> [true, false]
  %2 = comb.truth_table %1 -> [true, false]
  %3 = comb.truth_table %2 -> [true, false]
  %4 = comb.truth_table %3 -> [true, false]
  %5 = comb.truth_table %4 -> [true, false]
  hw.output %5 : i1
}
hw.module @Top(in %clk : !seq.clock, in %d : i1, out h : i1, out n : i1) {
  %a = seq.firreg %d clock %clk : i1
  %fa = hw.instance "fa" @Chain(x: %a: i1) -> (o: i1)
  %f = seq.firreg %fa clock %clk : i1
  %gf = hw.instance "gf" @Chain(x: %f: i1) -> (o: i1)
  %g = seq.firreg %gf clock %clk : i1
  %hg = hw.instance "hg" @Chain(x: %g: i1) -> (o: i1)
  %h = seq.firreg %hg clock %clk : i1
  %af = hw.instance "af" @Chain5(x: %f: i1) -> (o: i1)
  %addr = comb.replicate %af : (i1) -> i6
  %ram = seq.firmem 0, 1, undefined, undefined : <64 x 1>
  %rd = seq.firmem.read_port %ram[%addr], clock %clk : <64 x 1>
  %n = seq.firreg %rd clock %clk : i1
  hw.output %h, %n : i1, i1
}
