// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-timing{top=Top report= paths=- path-levels=1})' -o /dev/null | FileCheck %s

// Both bits of b come from a through two LUTs: one path, of 2 bits, from
// the end point back to its start, each op with its source line. s is one
// level from a, and t none: s is listed after b, and t, below path-levels,
// is not.
// CHECK: {"levels":2,"retimed":2,"bits":2,"kind":"d","start_kind":"reg","start_inst":0,"start":"a[{{[01]}}]","end_inst":0,"end":"b.d[{{[01]}}]","ops":[
// CHECK-SAME: [2,"lut",{{[0-9]+}},0,"comb.truth_table[0]","estimate-timing-paths.mlir:{{1[67]}}"],[1,"lut",{{[0-9]+}},0,"comb.truth_table[0]","estimate-timing-paths.mlir:{{1[45]}}"],[0,"reg",{{[0-9]+}},0,"a[{{[01]}}]","estimate-timing-paths.mlir:12"]]}
// CHECK-NEXT: {"levels":1,{{.*}}"bits":1,{{.*}}"start":"a[0]",{{.*}}"end":"s.d[0]"
// CHECK-NOT: "t.d
hw.module @Top(in %clk : !seq.clock, in %d : i2, out q : i2, out p : i1, out r : i1) {
  %a = seq.firreg %d clock %clk : i2
  %a0 = comb.extract %a from 0 : (i2) -> i1
  %a1 = comb.extract %a from 1 : (i2) -> i1
  %x0 = comb.truth_table %a0 -> [true, false]
  %x1 = comb.truth_table %a1 -> [true, false]
  %y0 = comb.truth_table %x0 -> [true, false]
  %y1 = comb.truth_table %x1 -> [true, false]
  %c = comb.concat %y1, %y0 : i1, i1
  %b = seq.firreg %c clock %clk : i2
  %s = seq.firreg %x0 clock %clk : i1
  %t = seq.firreg %a0 clock %clk : i1
  hw.output %b, %s, %t : i2, i1, i1
}
