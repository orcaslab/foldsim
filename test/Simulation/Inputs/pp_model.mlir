// The model of the CSF PP test (static-state-in-ram-pp.mlir): two accumulators, a counter, a
// comb-read memory (its data defined by a valid mask), a register pair that reads each other (a
// strongly connected group), a 48-bit product accumulated (a deep next value) and a chain of sums.
hw.module private @Model(in %clk : !seq.clock, in %reset : i1, in %a : i8, in %en : i1, in %v : i1, out y : i32) {
  %c0 = hw.constant 0 : i8
  %c1 = hw.constant 1 : i8
  %c5 = hw.constant 5 : i8
  %c0_5 = hw.constant 0 : i5
  %r1a = comb.add %r1, %a : i8
  %r1n = comb.mux %en, %r1a, %r1 : i8
  %r1z = comb.mux %reset, %c0, %r1n : i8
  %r1 = seq.firreg %r1z clock %clk : i8
  %r2a = comb.add %r2, %c1 : i8
  %r2n = comb.mux %v, %r2a, %r2 : i8
  %r2z = comb.mux %reset, %c0, %r2n : i8
  %r2 = seq.firreg %r2z clock %clk : i8
  %ram = seq.firmem 0, 1, undefined, undefined : <8 x 16>
  %wa = comb.extract %r1 from 0 : (i8) -> i3
  %ra = comb.extract %r2 from 0 : (i8) -> i3
  %wd = comb.concat %a, %r1 : i8, i8
  %true = hw.constant true
  seq.firmem.write_port %ram[%wa] = %wd, clock %clk enable %en : <8 x 16>
  %rd = seq.firmem.read_port %ram[%ra], clock %clk : <8 x 16>
  %c0_8 = hw.constant 0 : i8
  %one8 = hw.constant 1 : i8
  %wa8 = comb.concat %c0_5, %wa : i5, i3
  %bit = comb.shl %one8, %wa8 : i8
  %vmo = comb.or %vm, %bit : i8
  %vmn = comb.mux %en, %vmo, %vm : i8
  %vmz = comb.mux %reset, %c0_8, %vmn : i8
  %vm = seq.firreg %vmz clock %clk : i8
  %ra8 = comb.concat %c0_5, %ra : i5, i3
  %vs = comb.shru %vm, %ra8 : i8
  %ok = comb.extract %vs from 0 : (i8) -> i1
  %c0_16 = hw.constant 0 : i16
  %rdv = comb.mux %ok, %rd, %c0_16 : i16
  %rlo = comb.extract %rdv from 0 : (i16) -> i8
  %rhi = comb.extract %rdv from 8 : (i16) -> i8
  %r5z = comb.mux %reset, %c0, %rhi : i8
  %r5 = seq.firreg %r5z clock %clk : i8
  %r3a = comb.xor %r3, %rlo : i8
  %r3b = comb.add %r3a, %r2 : i8
  %r3z = comb.mux %reset, %c0, %r3b : i8
  %r3 = seq.firreg %r3z clock %clk : i8
  %r4m = comb.mul %r4, %c5 : i8
  %r4a = comb.add %r4m, %r3 : i8
  %r4z = comb.mux %reset, %c0, %r4a : i8
  %r4 = seq.firreg %r4z clock %clk : i8
  %p1x = comb.add %p2, %a : i8
  %p1z = comb.mux %reset, %c0, %p1x : i8
  %p1 = seq.firreg %p1z clock %clk : i8
  %p2x = comb.xor %p1, %r4 : i8
  %p2n = comb.mux %en, %p2x, %p2 : i8
  %p2z = comb.mux %reset, %c0, %p2n : i8
  %p2 = seq.firreg %p2z clock %clk : i8
  %w1 = comb.concat %r3, %r4, %p1, %r1, %r2, %a : i8, i8, i8, i8, i8, i8
  %c3_48 = hw.constant 3 : i48
  %w1m = comb.mul %w1, %c3_48 : i48
  %wbn = comb.add %wb, %w1m : i48
  %c0_48 = hw.constant 0 : i48
  %wbz = comb.mux %reset, %c0_48, %wbn : i48
  %wb = seq.firreg %wbz clock %clk : i48
  %wlo = comb.extract %wb from 8 : (i48) -> i8
  %r4y = comb.xor %r4, %p2 : i8
  %r4w = comb.add %r4y, %wlo : i8
  %y = comb.concat %r1, %r5, %r3, %r4w : i8, i8, i8, i8
  hw.output %y : i32
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %a : i8, in %en : i4, in %v : i4, out y0 : i32, out y1 : i32, out y2 : i32, out y3 : i32) {
  %c1 = hw.constant 1 : i8
  %c2 = hw.constant 2 : i8
  %c3 = hw.constant 3 : i8
  %e0 = comb.extract %en from 0 : (i4) -> i1
  %e1 = comb.extract %en from 1 : (i4) -> i1
  %e2 = comb.extract %en from 2 : (i4) -> i1
  %e3 = comb.extract %en from 3 : (i4) -> i1
  %v0 = comb.extract %v from 0 : (i4) -> i1
  %v1 = comb.extract %v from 1 : (i4) -> i1
  %v2 = comb.extract %v from 2 : (i4) -> i1
  %v3 = comb.extract %v from 3 : (i4) -> i1
  %a1 = comb.xor %a, %c1 : i8
  %a2 = comb.xor %a, %c2 : i8
  %a3 = comb.xor %a, %c3 : i8
  %y0 = hw.instance "m0" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a: i8, en: %e0: i1, v: %v0: i1) -> (y: i32)
  %y1 = hw.instance "m1" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a1: i8, en: %e1: i1, v: %v1: i1) -> (y: i32)
  %y2 = hw.instance "m2" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a2: i8, en: %e2: i1, v: %v2: i1) -> (y: i32)
  %y3 = hw.instance "m3" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a3: i8, en: %e3: i1, v: %v3: i1) -> (y: i32)
  hw.output %y0, %y1, %y2, %y3 : i32, i32, i32, i32
}
