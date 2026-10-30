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
    %13 = comb.xor %12, %narrow, %4 : i8
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

