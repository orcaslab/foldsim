// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-replicate-controls{loads=8}))' | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-replicate-controls{loads=0}))' | FileCheck %s --check-prefix=OFF

// A control of 12 + 20 + 20 + 1 loads at 8 a net: the 12-bit select becomes
// two of 6 bits, the 20-bit store three parts of 7, 7 and 6 bits (a port's
// address drives a LUT pin per bit), and the register eight copies of the
// same next value, each driving at most 8 loads in module order: a select, or
// a part's port, each. The wiring (the address's concatenation) is copied
// with each replica. The unmarked register stays one, and the marks go.

// CHECK-LABEL: hw.module @Copy
// CHECK-NOT: foldsim.control
// CHECK: %idx = seq.firreg %inst2 clock %clk {sv.attributes = [#sv.attribute<"DONT_TOUCH" = "\22true\22">]} : i1
// CHECK-NEXT: %idx_rep1 = seq.firreg %inst2 clock %clk {sv.attributes = [#sv.attribute<"DONT_TOUCH" = "\22true\22">]} : i1
// CHECK: %idx_rep7 = seq.firreg %inst2
// CHECK-NOT: %idx_rep8
// CHECK: comb.mux %idx, {{.+}} : i6
// CHECK: comb.mux %idx_rep1, {{.+}} : i6
// CHECK: %mem_p0 = seq.firmem 0, 1, undefined, undefined : <4 x 7, mask 1>
// CHECK: %mem_p1 = seq.firmem 0, 1, undefined, undefined : <4 x 7, mask 1>
// CHECK: %mem_p2 = seq.firmem 0, 1, undefined, undefined : <4 x 6, mask 1>
// CHECK-DAG: [[A2:%.+]] = comb.concat %low, %idx_rep2 : i1, i1
// CHECK-DAG: [[A3:%.+]] = comb.concat %low, %idx_rep3 : i1, i1
// CHECK-DAG: [[A4:%.+]] = comb.concat %low, %idx_rep4 : i1, i1
// CHECK-DAG: [[A5:%.+]] = comb.concat %low, %idx_rep5 : i1, i1
// CHECK-DAG: [[A6:%.+]] = comb.concat %low, %idx_rep6 : i1, i1
// CHECK-DAG: [[A7:%.+]] = comb.concat %low, %idx_rep7 : i1, i1
// CHECK-DAG: seq.firmem.write_port %mem_p0[[[A2]]]
// CHECK-DAG: seq.firmem.write_port %mem_p1[[[A3]]]
// CHECK-DAG: seq.firmem.write_port %mem_p2[[[A4]]]
// CHECK-DAG: seq.firmem.read_port %mem_p2[[[A5]]]
// CHECK-DAG: seq.firmem.read_port %mem_p1[[[A6]]]
// CHECK-DAG: seq.firmem.read_port %mem_p0[[[A7]]]
// CHECK-DAG: %plain = seq.firreg %inst2 clock %clk : i1
// CHECK-DAG: hw.output {{.+}}, {{.+}}, %idx_rep7,
// CHECK-NOT: %plain_rep

// OFF-NOT: foldsim.control
// OFF-NOT: _rep
// OFF-NOT: DONT_TOUCH
// OFF: seq.firmem 0, 1, undefined, undefined : <4 x 20, mask 1>

hw.module @Copy(in %clk: !seq.clock, in %inst2: i1, in %a: i12, in %b: i12,
                in %d: i20, in %low: i1, out o: i12, out r: i20, out s: i1,
                out p: i12) {
  %true = hw.constant true
  %idx = seq.firreg %inst2 clock %clk {foldsim.control} : i1
  %m = comb.mux %idx, %a, %b : i12
  %mem = seq.firmem 0, 1, undefined, undefined : <4 x 20, mask 1>
  %addr = comb.concat %low, %idx : i1, i1
  seq.firmem.write_port %mem[%addr] = %d, clock %clk enable %true : <4 x 20, mask 1>
  %rd = seq.firmem.read_port %mem[%addr], clock %clk : <4 x 20, mask 1>
  %plain = seq.firreg %inst2 clock %clk : i1
  %pm = comb.mux %plain, %a, %b : i12
  hw.output %m, %rd, %idx, %pm : i12, i20, i1, i12
}
