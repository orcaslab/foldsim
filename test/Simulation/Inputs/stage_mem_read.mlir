hw.module @Leaf(in %clk : !seq.clock, in %reset : i1, in %a : i6, in %w : i8, out y : i8) {
  %z = hw.constant 0 : i8
  %t = hw.constant true
  %m = seq.firmem 0, 1, undefined, undefined : <64 x 8>
  seq.firmem.write_port %m[%a] = %w, clock %clk enable %t : <64 x 8>
  %d = seq.firmem.read_port %m[%r6], clock %clk : <64 x 8>
  %r6 = comb.extract %r from 0 : (i8) -> i6
  %x = comb.add %d, %w : i8
  %r = seq.firreg %x clock %clk reset sync %reset, %z : i8
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %a : i6, in %w : i8, out y0 : i8, out y1 : i8, out y2 : i8, out y3 : i8) {
  %l0.y = hw.instance "l0" @Leaf(clk: %clk : !seq.clock, reset: %reset : i1, a: %a : i6, w: %w : i8) -> (y : i8)
  %l1.y = hw.instance "l1" @Leaf(clk: %clk : !seq.clock, reset: %reset : i1, a: %a : i6, w: %w : i8) -> (y : i8)
  %l2.y = hw.instance "l2" @Leaf(clk: %clk : !seq.clock, reset: %reset : i1, a: %a : i6, w: %w : i8) -> (y : i8)
  %l3.y = hw.instance "l3" @Leaf(clk: %clk : !seq.clock, reset: %reset : i1, a: %a : i6, w: %w : i8) -> (y : i8)
  hw.output %l0.y, %l1.y, %l2.y, %l3.y : i8, i8, i8, i8
}
