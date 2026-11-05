// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-split-masked-memories))' | FileCheck %s

// A byte-masked 128-bit memory in 64-bit columns: each column keeps the
// memory's ram_style (fold-static sets it from a storage rule before this
// pass runs; the 8-SM fold's GPR, priced and ruled ultra, came out of the
// split with no ram_style at all).

// CHECK-LABEL: hw.module @M
// CHECK-DAG: %mem_c0 = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <16 x 64, mask 8>
// CHECK-DAG: %mem_c1 = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <16 x 64, mask 8>
// CHECK-NOT: <16 x 128

hw.module @M(in %clk: !seq.clock, in %a: i4, in %d: i128, in %mask: i16, in %en: i1, out r: i128) {
  %mem = seq.firmem name "mem" 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <16 x 128, mask 16>
  seq.firmem.write_port %mem[%a] = %d, clock %clk enable %en mask %mask : <16 x 128, mask 16>, i16
  %rd = seq.firmem.read_port %mem[%a], clock %clk : <16 x 128, mask 16>
  hw.output %rd : i128
}
