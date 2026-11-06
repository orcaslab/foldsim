// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-timing{output=- report=- top=Top})' -o /dev/null | FileCheck %s

// A counter that only counts itself reaches no output: Vivado removes it, so
// its deep increment is no end point. Nor is a memory nothing reads, or the
// register only that memory's write port reads.
// CHECK-NOT: ,cnt.d[
// CHECK-NOT: ,lost.d[
// CHECK-NOT: ,0,unread.
// A register that reaches an output through a memory and its read is live.
// CHECK-DAG: ,0,w.d[0]
// CHECK-DAG: ,0,o.d[0]
// CHECK: (47 unobservable left out)
hw.module @Top(in %clk : !seq.clock, in %d : i32, in %a : i7, out q : i4) {
  %one = hw.constant 1 : i32
  %inc = comb.add %cnt, %one : i32
  %cnt = seq.firreg %inc clock %clk : i32
  %true = hw.constant true
  %lost = seq.firreg %d4 clock %clk : i4
  %d4 = comb.extract %d from 0 : (i32) -> i4
  %unread = seq.firmem 0, 1, undefined, undefined : <128 x 4>
  seq.firmem.write_port %unread[%a] = %lost, clock %clk enable %true : <128 x 4>
  %w = seq.firreg %d4 clock %clk : i4
  %ram = seq.firmem 0, 1, undefined, undefined : <128 x 4>
  seq.firmem.write_port %ram[%a] = %w, clock %clk enable %true : <128 x 4>
  %rd = seq.firmem.read_port %ram[%a], clock %clk : <128 x 4>
  %o = seq.firreg %rd clock %clk : i4
  hw.output %o : i4
}
