// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-slr-split{module=copy map=%S/Inputs/slr-split/loop.map parts=2 crossing=txrx latency=2})' -o /dev/null 2>&1 | FileCheck %s

// A loop of registers alone (`a` in part 0, `b` in part 1, each reading the
// other) cut by the map: a send and receive register on each crossing would
// change the loop's cycles, and no lag absorbs them (no memory on the loop).
// CHECK: error: slr-split: crossing=txrx: no lags within 2 host cycles: a loop of registers alone crosses parts

hw.module @copy(in %clk : !seq.clock, in %x : i4, out q : i4) {
  %0 = comb.xor %b, %x : i4
  %a = seq.firreg %0 clock %clk : i4
  %b = seq.firreg %a clock %clk : i4
  hw.output %b : i4
}
