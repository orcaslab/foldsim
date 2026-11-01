// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-replicate-controls{loads=27}))' | FileCheck %s

// A masked memory of seven 9-bit mask bits split at 27 loads a pin: parts of
// three, three and one mask bit. The one-bit part's memory has mask width 1,
// which is no mask (lowering ignores it), so its mask bit enables its write
// (a paged copy's page RAM whose last part was one byte wrote that byte under
// an enable not taken: virt_1101, the vt target at p 4).

// CHECK-LABEL: hw.module @Copy
// CHECK-DAG: %mem_p0 = seq.firmem 0, 1, undefined, undefined : <4 x 27, mask 3>
// CHECK-DAG: %mem_p1 = seq.firmem 0, 1, undefined, undefined : <4 x 27, mask 3>
// CHECK-DAG: %mem_p2 = seq.firmem 0, 1, undefined, undefined : <4 x 9, mask 1>
// CHECK-DAG: [[M0:%.+]] = comb.extract %mask from 0 : (i7) -> i3
// CHECK-DAG: seq.firmem.write_port %mem_p0[{{.+}}] = {{.+}}, clock %clk enable %en mask [[M0]] :
// CHECK-DAG: [[M2:%.+]] = comb.extract %mask from 6 : (i7) -> i1
// CHECK-DAG: [[E2:%.+]] = comb.and %en, [[M2]] : i1
// CHECK-DAG: seq.firmem.write_port %mem_p2[{{.+}}] = {{.+}}, clock %clk enable [[E2]] : <4 x 9, mask 1>

hw.module @Copy(in %clk: !seq.clock, in %inst2: i2, in %d: i63, in %mask: i7, in %en: i1, out r: i63) {
  %idx = seq.firreg %inst2 clock %clk {foldsim.control} : i2
  %mem = seq.firmem 0, 1, undefined, undefined : <4 x 63, mask 7>
  seq.firmem.write_port %mem[%idx] = %d, clock %clk enable %en mask %mask : <4 x 63, mask 7>, i7
  %rd = seq.firmem.read_port %mem[%idx], clock %clk : <4 x 63, mask 7>
  hw.output %rd : i63
}
