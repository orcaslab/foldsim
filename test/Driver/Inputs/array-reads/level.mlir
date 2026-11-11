module {
  hw.module @foldsim_sim(in %hostClock : !seq.clock, in %i2 : i2, in %i3 : i3, in %a : i8, in %b : i8, in %c : i8, in %arr : !hw.array<4xi8>, out o0 : i8, out o1 : i8, out o2 : i8, out o3 : i8, out o4 : i8) {
    %c0_i8 = hw.constant 0 : i8
    %c1_i2 = hw.constant 1 : i2
    %t = hw.aggregate_constant [3 : i8, 1 : i8, 3 : i8, 1 : i8] : !hw.array<4xi8>
    %r0 = hw.array_get %t[%i2] : !hw.array<4xi8>, i2
    %w = hw.array_create %a, %b, %c0_i8, %c : i8
    %r1 = hw.array_get %w[%i2] : !hw.array<4xi8>, i2
    %r2 = hw.array_get %w[%c1_i2] : !hw.array<4xi8>, i2
    %r3 = hw.array_get %arr[%i2] : !hw.array<4xi8>, i2
    %big = hw.aggregate_constant [1 : i1, 0 : i1, 1 : i1, 1 : i1, 0 : i1, 1 : i1, 0 : i1, 0 : i1] : !hw.array<8xi1>
    %r4b = hw.array_get %big[%i3] : !hw.array<8xi1>, i3
    %r4 = comb.concat %c0_i8, %r4b : i8, i1
    %r4t = comb.extract %r4 from 0 : (i9) -> i8
    hw.output %r0, %r1, %r2, %r3, %r4t : i8, i8, i8, i8, i8
  }
}
