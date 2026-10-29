// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-timing{output=- report=- top=Top target=asap7})' -o /dev/null | FileCheck %s

// Two and-inverter gates in a row: two stages.
// CHECK: 2,2,d,reg,{{[0-9]+}},0,0,{{[ab]}}[0],0,g.d[0]
// A 32-bit adder's top bit from bit 0: a propagate-generate gate, a prefix
// tree over 32 bits and the sum gate.
// CHECK: 7,7,d,reg,{{[0-9]+}},0,0,{{[ab]}}[0],0,s.d[31]
// A 16 x 16 multiplier: the partial products, a 6-stage Dadda tree of full
// adders and a 32-bit prefix adder.
// CHECK: 20,20,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,p.d[{{[0-9]+}}]
// A 128-word memory of flops: the decoder and a 7-level multiplexer tree.
// CHECK: 8,8,d,reg,{{[0-9]+}},0,0,a[{{[0-9]+}}],0,z.d[0]
// The report converts the deepest path with the library's delays.
// CHECK: deepest path 20 stages: 58 + 167 + 20 x 24 + 6 = 711 ps
hw.module @Top(in %clk : !seq.clock, in %d : i32) {
  %a = seq.firreg %d clock %clk : i32
  %b = seq.firreg %d clock %clk : i32
  %a0 = comb.extract %a from 0 : (i32) -> i1
  %b0 = comb.extract %b from 0 : (i32) -> i1
  %b1 = comb.extract %b from 1 : (i32) -> i1
  %g1 = synth.aig.and_inv %a0, not %b0 : i1
  %g2 = synth.aig.and_inv %g1, %b1 : i1
  %g = seq.firreg %g2 clock %clk : i1
  %sum = comb.add %a, %b : i32
  %s = seq.firreg %sum clock %clk : i32
  %a16 = comb.extract %a from 0 : (i32) -> i16
  %b16 = comb.extract %b from 0 : (i32) -> i16
  %c0_i16 = hw.constant 0 : i16
  %a32 = comb.concat %c0_i16, %a16 : i16, i16
  %b32 = comb.concat %c0_i16, %b16 : i16, i16
  %prod = comb.mul %a32, %b32 : i32
  %p = seq.firreg %prod clock %clk : i32
  %ram = seq.firmem 0, 1, undefined, undefined : <128 x 4>
  %addr = comb.extract %a from 0 : (i32) -> i7
  %rd = seq.firmem.read_port %ram[%addr], clock %clk : <128 x 4>
  %z = seq.firreg %rd clock %clk : i4
  hw.output
}
