// An ultra RAM deeper than one URAM288 (4,096 rows) states its cascade
// height, a URAM288 per 4,096 rows: Vivado's default (cascade height 1)
// builds such a memory of ~1.5x the primitives (a32-UR, 11-12). Four leaves
// on one copy: the memory of 2,048 words threaded over them is 8,192 deep
// (cascade 2), of 1,024 words 4,096 deep (one URAM288: no cascade), and a
// block RAM states none.
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Leaf:mem=ultra,Leaf:small=ultra})' | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Leaf:mem=block,Leaf:small=ultra})' | FileCheck %s --check-prefix=BLOCK
// CHECK: %mem = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">, #sv.attribute<"cascade_height" = "2">]} : <8192 x 8, mask 1>
// CHECK: %small = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <4096 x 8, mask 1>
// BLOCK-NOT: cascade_height
// BLOCK: %mem = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <8192 x 8, mask 1>
// BLOCK-NOT: cascade_height
module {
  hw.module @Leaf(in %clk : !seq.clock, in %d : i8, out q : i8) {
    %c1_i11 = hw.constant 1 : i11
    %c1_i10 = hw.constant 1 : i10
    %mem = seq.firmem name "mem" 1, 1, old, undefined : <2048 x 8, mask 1>
    %small = seq.firmem name "small" 1, 1, old, undefined : <1024 x 8, mask 1>
    seq.firmem.write_port %mem[%waddr] = %d, clock %clk : <2048 x 8, mask 1>
    seq.firmem.write_port %small[%saddr] = %d, clock %clk : <1024 x 8, mask 1>
    %0 = comb.add %waddr, %c1_i11 : i11
    %waddr = seq.firreg %0 clock %clk : i11
    %1 = comb.add %saddr, %c1_i10 : i10
    %saddr = seq.firreg %1 clock %clk : i10
    %2 = seq.firmem.read_port %mem[%0], clock %clk : <2048 x 8, mask 1>
    %3 = seq.firmem.read_port %small[%1], clock %clk : <1024 x 8, mask 1>
    %4 = comb.xor %2, %3 : i8
    %held = seq.firreg %4 clock %clk : i8
    hw.output %held : i8
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
