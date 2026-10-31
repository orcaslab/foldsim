// Page write masks (foldsim-fold-static page-masks, on by default): a paged
// register written under an enable keeps it, registered at the state pass,
// as the write mask of its bytes; its next value goes straight into its
// write-back register. The registers written every state pass come first (e,
// qr: 13 bits, two 9-bit bytes), then each enable's (a and b under d[0], one
// run of two bytes; c under d[1], one byte): 44 bits, a page of 27 bits
// (three bytes) at p = 2. Page 0's bytes: always, always, d[0]; page 1's:
// d[0], d[1], padding.
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page=Leaf:2})' 2>&1 | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page=Leaf:2 page-masks=false})' 2>&1 | FileCheck %s --check-prefix=DENSE
// CHECK: pages Leaf: p 2, 5 registers, 33 bits (44 in 9-bit bytes, 2 enables); page RAMs 8 rows x 27 bits in 1 columns, 1 ports, block
// CHECK: %a_wen = seq.firreg [[AW:%.+]] clock %hostClock : i1
// CHECK: [[AW]] = comb.mux %region0_state, {{%.+}}, %a_wen : i1
// CHECK: %a_writeback = seq.firreg [[AN:%.+]] clock %hostClock : i8
// CHECK: [[AN]] = comb.mux %region0_state, {{%.+}}, %a_writeback : i8
// CHECK: %c_wen = seq.firreg
// CHECK: %region1_pages = seq.firmem 1, 1, old, undefined {{.*}} : <8 x 27, mask 3>
// CHECK: comb.concat %a_wen, %true{{.*}}, %true{{.*}} : i1, i1, i1
// CHECK: comb.concat %true{{.*}}, %c_wen, %a_wen : i1, i1, i1
// CHECK: seq.firmem.write_port %region1_pages[{{%.+}}] = {{%.+}}, clock %hostClock enable {{%.+}} mask {{%.+}} : <8 x 27, mask 3>, i3
// The dense page: 33 bits, the held registers' old values written back.
// DENSE: pages Leaf: p 2, 5 registers, 33 bits; page RAMs 8 rows x 17 bits in 1 columns, 1 ports, block
// DENSE-NOT: _wen =
// DENSE: %a_writeback = seq.firreg [[AN:%.+]] clock %hostClock : i8
// DENSE: [[AN]] = comb.mux %region0_state, [[AH:%.+]], %a_writeback : i8
// DENSE: %region1_pages = seq.firmem 1, 1, old, undefined {{.*}} : <8 x 17, mask 1>
// At p = 5 a page is one byte: a memory of mask width 1 has no mask, so the
// byte's enable is anded into the write's enable.
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page=Leaf:5})' 2>&1 | FileCheck %s --check-prefix=ONE
// ONE: pages Leaf: p 5, 5 registers, 33 bits (44 in 9-bit bytes, 2 enables); page RAMs 32 rows x 9 bits in 1 columns
// ONE: seq.firmem.write_port %region1_pages[{{%.+}}] = {{%.+}}, clock %hostClock enable {{%[^ ]+}} : <32 x 9, mask 1>
module {
  hw.module @Leaf(in %clk : !seq.clock, in %d : i8, out q : i8) {
    %c1 = hw.constant 1 : i8
    %e0 = comb.extract %d from 0 : (i8) -> i1
    %e1 = comb.extract %d from 1 : (i8) -> i1
    %an = comb.add %a, %d : i8
    %anext = comb.mux %e0, %an, %a : i8
    %a = seq.firreg %anext clock %clk : i8
    %bd = comb.extract %d from 2 : (i8) -> i4
    %bnext = comb.mux %e0, %bd, %b : i4
    %b = seq.firreg %bnext clock %clk : i4
    %cn = comb.xor %c, %d : i8
    %cnext = comb.mux %e1, %cn, %c : i8
    %c = seq.firreg %cnext clock %clk : i8
    %en = comb.extract %d from 3 : (i8) -> i5
    %e = seq.firreg %en clock %clk : i5
    %b8 = comb.concat %b, %e, %e1, %e0, %e1 : i4, i5, i1, i1, i1
    %b8x = comb.extract %b8 from 0 : (i12) -> i8
    %q = comb.xor %a, %c, %b8x : i8
    %qr = seq.firreg %q clock %clk : i8
    hw.output %qr : i8
  }
  hw.module private @Top_model(in %clk : !seq.clock, in %x : i8, in %l3_q : i8, out y : i8) {
    hw.output %l3_q : i8
  }
  hw.module @Top(in %clk : !seq.clock, in %x : i8, out y : i8) attributes {foldsim.netlist} {
    %l0.q = hw.instance "l0" @Leaf(clk: %clk: !seq.clock, d: %x: i8) -> (q: i8)
    %l1.q = hw.instance "l1" @Leaf(clk: %clk: !seq.clock, d: %l0.q: i8) -> (q: i8)
    %l2.q = hw.instance "l2" @Leaf(clk: %clk: !seq.clock, d: %l1.q: i8) -> (q: i8)
    %l3.q = hw.instance "l3" @Leaf(clk: %clk: !seq.clock, d: %l2.q: i8) -> (q: i8)
    %logic.y = hw.instance "logic" @Top_model(clk: %clk: !seq.clock, x: %x: i8, l3_q: %l3.q: i8) -> (y: i8)
    hw.output %logic.y : i8
  }
}
