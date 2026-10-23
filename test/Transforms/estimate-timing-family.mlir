// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-timing{output=- report= top=Top})' -o /dev/null | FileCheck %s --check-prefix=V80
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-timing{output=- report= top=Top family=virtexuplusHBM})' -o /dev/null | FileCheck %s --check-prefix=U55C
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-timing{output=- report= top=Top family=zynquplus})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN

// A 16 x 16 product into a register: one DSP, which absorbs the register.
// V80-DAG: 1,1,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,p16.d[0]
// U55C-DAG: 1,1,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,p16.d[0]
// The same product into a register with a clock enable: its enable
// multiplexer reads the register back, and the DSP absorbs both.
// V80-DAG: 1,1,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,q16.d[0]
// U55C-DAG: 1,1,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,q16.d[0]
// 33 x 33: a cascade of four DSP58s or four DSP48E2s.
// V80-DAG: 10,10,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,p33.d[0]
// U55C-DAG: 7,7,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,p33.d[0]
// The same product of the operands sign-extended to 66 bits: the copies of
// the sign bit are not significant.
// V80-DAG: 10,10,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,p66.d[0]
// U55C-DAG: 7,7,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,p66.d[0]
// The same product through a LUT: out of the DSP's output stage first.
// V80-DAG: 17,17,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,l33.d[0]
// U55C-DAG: 13,13,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,l33.d[0]
// A zero-extended bit times 10 is a table, not a DSP: through a LUT, two
// levels.
// V80-DAG: 2,2,d,reg,{{[0-9]+}},0,0,a[0],0,l10.d[0]
// U55C-DAG: 2,2,d,reg,{{[0-9]+}},0,0,a[0],0,l10.d[0]
// UNKNOWN: unknown FPGA family 'zynquplus'
hw.module @Top(in %clk : !seq.clock, in %d : i33) {
  %a = seq.firreg %d clock %clk : i33
  %b = seq.firreg %d clock %clk : i33
  %a16 = comb.extract %a from 0 : (i33) -> i16
  %b16 = comb.extract %b from 0 : (i33) -> i16
  %m16 = comb.mul %a16, %b16 : i16
  %p16 = seq.firreg %m16 clock %clk : i16
  %en = comb.extract %a from 20 : (i33) -> i1
  %m16b = comb.extract %m16 from 0 : (i16) -> i1
  %q16n = comb.truth_table %en, %m16b, %q16 -> [false, true, false, true, false, false, true, true]
  %q16 = seq.firreg %q16n clock %clk : i1
  %m33 = comb.mul %a, %b : i33
  %p33 = seq.firreg %m33 clock %clk : i33
  %sa = comb.extract %a from 32 : (i33) -> i1
  %sb = comb.extract %b from 32 : (i33) -> i1
  %ra = comb.replicate %sa : (i1) -> i33
  %rb = comb.replicate %sb : (i1) -> i33
  %a66 = comb.concat %ra, %a : i33, i33
  %b66 = comb.concat %rb, %b : i33, i33
  %m66 = comb.mul %a66, %b66 : i66
  %p66 = seq.firreg %m66 clock %clk : i66
  %m0 = comb.extract %m33 from 0 : (i33) -> i1
  %a0 = comb.extract %a from 0 : (i33) -> i1
  %t = comb.truth_table %m0, %a0 -> [false, true, true, false]
  %l33 = seq.firreg %t clock %clk : i1
  %c0_i31 = hw.constant 0 : i31
  %c10_i32 = hw.constant 10 : i32
  %x1 = comb.concat %c0_i31, %a0 : i31, i1
  %m10 = comb.mul %x1, %c10_i32 : i32
  %m10b = comb.extract %m10 from 1 : (i32) -> i1
  %b0 = comb.extract %b from 0 : (i33) -> i1
  %t10 = comb.truth_table %m10b, %b0 -> [false, true, true, false]
  %l10 = seq.firreg %t10 clock %clk : i1
  hw.output
}
