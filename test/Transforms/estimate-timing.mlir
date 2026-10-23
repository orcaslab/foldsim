// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-timing{output=- report= top=Top})' -o /dev/null | FileCheck %s

// CHECK: levels,retimed,kind,start_kind,fanout,crossings,start_inst,start,end_inst,end
// The adder's bit 15: a LUT and the CARRY8s of blocks 0 and 1 from bit 0.
// CHECK: 3,3,d,reg,{{[0-9]+}},0,0,a[0],0,s.d[15]
// The comparator after it: the carry from its input bit to bit 15.
// CHECK: 5,5,d,reg,{{[0-9]+}},0,0,a[0],0,c.d[0]
// A LUT in an instance of `Leaf`, two boundaries away from `a`.
// CHECK: 6,6,d,reg,{{[0-9]+}},2,0,a[0],0,q.d[0]
// A 128-deep distributed-RAM read: the RAM64 and a MUXF7.
// CHECK: 2,2,d,reg,{{[0-9]+}},0,0,a[0],0,z.d[0]
// An external module's input ends a path.
// CHECK: 5,5,x,reg,{{[0-9]+}},0,0,a[0],0,ip.x[0]
// An 8-bit divider: eight subtracters of a LUT and one CARRY8 each.
// CHECK: 16,16,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,q8.d[0]
// A 3-bit sum of two 2-bit counts: each bit reads at most six input bits,
// so it is one LUT, not a LUT and a CARRY8.
// CHECK: 1,1,d,reg,{{[0-9]+}},0,0,{{[ab]}}[{{[0-9]+}}],0,n3.d[2]
// A 32-bit divider whose operands are functions of a 3-bit field is a table
// over the field: one LUT, not a ripple of 32 subtracters.
// CHECK: 1,1,d,reg,{{[0-9]+}},0,0,f[{{[0-2]}}],0,qf.d[0]
hw.module.extern private @Ip(in %clk : !seq.clock, in %x : i1)
hw.module private @Leaf(in %a : i1, in %b : i1, out o : i1) {
  %o = comb.truth_table %a, %b -> [false, true, true, false]
  hw.output %o : i1
}
hw.module @Top(in %clk : !seq.clock, in %d : i16) {
  %a = seq.firreg %d clock %clk : i16
  %b = seq.firreg %d clock %clk : i16
  %sum = comb.add %a, %b : i16
  %s = seq.firreg %sum clock %clk : i16
  %lt = comb.icmp ult %sum, %b : i16
  %c = seq.firreg %lt clock %clk : i1
  %a0 = comb.extract %a from 0 : (i16) -> i1
  %o = hw.instance "leaf" @Leaf(a: %lt: i1, b: %a0: i1) -> (o: i1)
  %q = seq.firreg %o clock %clk : i1
  %ram = seq.firmem 0, 1, undefined, undefined : <128 x 4>
  %addr = comb.extract %a from 0 : (i16) -> i7
  %rd = seq.firmem.read_port %ram[%addr], clock %clk : <128 x 4>
  %z = seq.firreg %rd clock %clk : i4
  hw.instance "ip" @Ip(clk: %clk: !seq.clock, x: %lt: i1) -> ()
  %a8 = comb.extract %a from 0 : (i16) -> i8
  %b8 = comb.extract %b from 0 : (i16) -> i8
  %div = comb.divu %a8, %b8 : i8
  %q8 = seq.firreg %div clock %clk : i8
  %a2 = comb.extract %a from 0 : (i16) -> i2
  %b2 = comb.extract %b from 0 : (i16) -> i2
  %z1 = hw.constant 0 : i1
  %a3 = comb.concat %z1, %a2 : i1, i2
  %b3 = comb.concat %z1, %b2 : i1, i2
  %s3 = comb.add %a3, %b3 : i3
  %n3 = seq.firreg %s3 clock %clk : i3
  %f = seq.firreg %a8 clock %clk : i8
  %f3 = comb.extract %f from 0 : (i8) -> i3
  %c0_i29 = hw.constant 0 : i29
  %c1_i32 = hw.constant 1 : i32
  %w = comb.concat %c0_i29, %f3 : i29, i3
  %wp = comb.add %w, %c1_i32 : i32
  %n = comb.shl %wp, %wp : i32
  %fd = comb.divu %n, %wp : i32
  %qf = seq.firreg %fd clock %clk : i32
  hw.output
}
