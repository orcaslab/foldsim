hw.module @M(in %a : !hw.array<32xi32>, in %v : i32, in %pid : i1, in %i : i5, in %j : i3, out o : !hw.array<32xi32>, out g : i32, out h : i32, out k : i32) {
  %c6_i4 = hw.constant 6 : i4
  %c3_i2 = hw.constant 3 : i2
  %idx = comb.concat %pid, %c6_i4 : i1, i4
  %0 = hw.array_inject %a[%idx], %v : !hw.array<32xi32>, i5
  %1 = hw.array_get %a[%idx] : !hw.array<32xi32>, i5
  %2 = hw.array_get %a[%i] : !hw.array<32xi32>, i5
  %jdx = comb.concat %c3_i2, %j : i2, i3
  %3 = hw.array_get %a[%jdx] : !hw.array<32xi32>, i5
  hw.output %0, %1, %2, %3 : !hw.array<32xi32>, i32, i32, i32
}
