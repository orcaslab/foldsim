hw.module @M(in %clk : !seq.clock, in %c0 : i1, in %c1 : i1, in %x : i8, out y : i8, out z : i8) {
  %r_lo = comb.extract %r from 0 : (i8) -> i4
  %r_hi = comb.extract %r from 4 : (i8) -> i4
  %x_lo = comb.extract %x from 0 : (i8) -> i4
  %x_hi = comb.extract %x from 4 : (i8) -> i4
  %m_lo = comb.mux %c0, %x_lo, %r_lo : i4
  %m_hi = comb.mux %c1, %x_hi, %r_hi : i4
  %n = comb.concat %m_hi, %m_lo : i4, i4
  %r = seq.firreg %n clock %clk : i8
  %s_lo = comb.extract %s from 0 : (i8) -> i4
  %s_hi = comb.extract %s from 4 : (i8) -> i4
  %k_lo = comb.mux %c0, %x_lo, %s_hi : i4
  %k_hi = comb.mux %c1, %x_hi, %s_lo : i4
  %t = comb.concat %k_hi, %k_lo : i4, i4
  %s = seq.firreg %t clock %clk : i8
  hw.output %r, %s : i8, i8
}
