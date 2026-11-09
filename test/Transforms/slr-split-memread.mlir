// RUN: foldsim-opt %s --verify-diagnostics --pass-pipeline='builtin.module(foldsim-slr-split{module=copy map=%S/Inputs/slr-split/memread.map parts=2})' -o /dev/null

// A combinational memory read (a LUTRAM's asynchronous read) read across the
// cut: its memory would have to move with it, an error.

hw.module @copy(in %clk : !seq.clock, in %i : i4, in %d : i8, out q : i8) {
  %mem = seq.firmem 0, 1, undefined, undefined : <16 x 8>
  seq.firmem.write_port %mem[%i] = %d, clock %clk : <16 x 8>
  // expected-error @+1 {{slr-split: a value of part 0 read in part 1 is not from a register (a combinational memory read)}}
  %rd = seq.firmem.read_port %mem[%i], clock %clk : <16 x 8>
  %q = seq.firreg %rd clock %clk : i8
  hw.output %q : i8
}
