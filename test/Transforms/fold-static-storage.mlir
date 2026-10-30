// The storage of folded state stores (foldsim-fold-static `storage`): a
// model's register stores by width and its memories by name, each in a
// Vivado ram_style. Four leaves on one copy (a state gap of 6).
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top})' | FileCheck %s --check-prefix=OFF
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Leaf:regs>=64=ultra,Leaf:regs=distributed,Leaf:mem=block})' | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Leaf:regs=block})' | FileCheck %s --check-prefix=BLOCK
// Errors: a model, a memory or a ram_style that is not there, a rule that is
// not one, and a register store in block RAM at a state gap below 4 (the
// leaves' output is the memory's read data: a ready pass, a state gap of 1).
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Nope:regs=block})' 2>&1 | FileCheck %s --check-prefix=MODEL
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Leaf:nomem=block})' 2>&1 | FileCheck %s --check-prefix=MEM
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Leaf:regs=bram})' 2>&1 | FileCheck %s --check-prefix=KIND
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Leaf:regs})' 2>&1 | FileCheck %s --check-prefix=RULE
// RUN: not foldsim-opt %S/Inputs/fold-static-storage-late.mlir --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Leaf:regs>=64=ultra})' 2>&1 | FileCheck %s --check-prefix=GAP
// Without rules no store has a ram_style but the target's.
// OFF-NOT: ram_style
// OFF: %mem = seq.firmem 1, 1, old, undefined : <64 x 8, mask 1>
// OFF-NOT: ram_style
// OFF: %wide = seq.firmem 0, 1, undefined, undefined : <4 x 80, mask 1>
// OFF: seq.firmem.read_port %wide[%region0_inst1]
// The wide register in ultra RAM, read a host cycle early into its output
// register; the narrow ones in distributed RAM; the memory in block RAM.
// CHECK: %held = seq.firmem 0, 1, undefined, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22distributed\22">]} : <4 x 8, mask 1>
// CHECK: %mem = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <64 x 8, mask 1>
// CHECK: %narrow = seq.firmem 0, 1, undefined, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22distributed\22">]} : <4 x 8, mask 1>
// CHECK: %wide = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <4 x 80, mask 1>
// CHECK: seq.firmem.read_port %wide[%ahead0_inst2]
// BLOCK: %held = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <4 x 8, mask 1>
// BLOCK: seq.firmem.read_port %held[%ahead0_inst2]
// BLOCK: %mem = seq.firmem 1, 1, old, undefined : <64 x 8, mask 1>
// MODEL: storage: no model 'Nope'
// MEM: storage: no memory 'nomem' in model 'Leaf'
// KIND: storage: 'bram' is not a ram_style: registers, distributed, block, ultra
// RULE: storage: 'Leaf:regs' is not <model>:<store>=<kind>
// GAP: storage: 'ultra' RAM for the registers of 'Leaf' needs a state gap of 4, not 1
module {
  hw.module @Leaf(in %clk : !seq.clock, in %d : i8, out q : i8) {
    %c-3_i4 = hw.constant -3 : i4
    %c1_i4 = hw.constant 1 : i4
    %c-1_i4 = hw.constant -1 : i4
    %mem = seq.firmem name "mem" 1, 1, old, undefined : <16 x 8, mask 1>
    %0 = comb.extract %wide from 0 : (i80) -> i72
    %1 = comb.extract %wide from 72 : (i80) -> i8
    %2 = comb.xor %d, %1 : i8
    %3 = comb.concat %0, %2 : i72, i8
    %wide = seq.firreg %3 clock %clk : i80
    %4 = comb.extract %wide from 40 : (i80) -> i8
    %5 = comb.extract %wide from 0 : (i80) -> i8
    %6 = comb.xor %d, %5 : i8
    %7 = comb.add %narrow, %d : i8
    %narrow = seq.firreg %7 clock %clk : i8
    %8 = comb.sub %c-1_i4, %waddr : i4
    seq.firmem.write_port %mem[%8] = %6, clock %clk : <16 x 8, mask 1>
    %9 = comb.add %waddr, %c1_i4 : i4
    %waddr = seq.firreg %9 clock %clk : i4
    %10 = comb.add %waddr, %c-3_i4 : i4
    %11 = comb.sub %c-1_i4, %10 : i4
    %12 = seq.firmem.read_port %mem[%11], clock %clk : <16 x 8, mask 1>
    %held = seq.firreg %12 clock %clk : i8
    %13 = comb.xor %held, %narrow, %4 : i8
    hw.output %13 : i8
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

