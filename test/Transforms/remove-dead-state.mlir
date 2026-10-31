// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-remove-dead-state))' | FileCheck %s

// A counter read only by its own increment, and the logic feeding it, go; the
// register an output reads, a memory read by an output and its write, and a
// register with a symbol stay.
// CHECK-LABEL: hw.module @Top
// CHECK-NOT: name "dead
// CHECK-NOT: comb.icmp
// CHECK-DAG: %live = seq.firreg
// CHECK-DAG: %kept = seq.firreg {{.*}} sym @kept
// CHECK-DAG: %mem = seq.firmem
// CHECK-DAG: seq.firmem.write_port %mem
// CHECK-DAG: seq.firmem.read_port %mem
// CHECK-NOT: name "deadmem"
// CHECK: hw.output
hw.module @Top(in %clk : !seq.clock, in %a : i4, in %b : i4, in %v : i1, out o : i4, out r : i8) {
  %ai = comb.extract %a from 0 : (i4) -> i2
  %bi = comb.extract %b from 0 : (i4) -> i2
  %c1 = hw.constant 1 : i8
  %eq = comb.icmp eq %a, %b : i4
  %inc = comb.and %eq, %v : i1
  %ext = comb.concat %c0, %inc : i7, i1
  %c0 = hw.constant 0 : i7
  %next = comb.add %dead, %ext : i8
  %dead = seq.firreg %next clock %clk {name = "dead"} : i8
  %buf = seq.firreg %inc clock %clk {name = "deadbuf"} : i1
  %live = seq.firreg %a clock %clk : i4
  %kept = seq.firreg %kept clock %clk sym @kept : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <4 x 8>
  seq.firmem.write_port %mem[%ai] = %c1, clock %clk enable %v : <4 x 8>
  %rd = seq.firmem.read_port %mem[%bi], clock %clk : <4 x 8>
  %deadmem = seq.firmem 0, 1, undefined, undefined {name = "deadmem"} : <4 x 8>
  seq.firmem.write_port %deadmem[%ai] = %next, clock %clk : <4 x 8>
  %drd = seq.firmem.read_port %deadmem[%bi], clock %clk : <4 x 8>
  %x = comb.xor %drd, %next : i8
  hw.output %live, %rd : i4, i8
}
