module {
  hw.module @M(in %a : i8, in %b : i8, in %s : i8, in %t : i8, in %r : i64, in %k : i6, out o : i8, out p : i1, out q : i8) {
    %0 = comb.and %a, %s : i8
    %1 = comb.and %b, %t : i8
    %2 = comb.or %0, %1 : i8
    %3 = comb.shru %r, %k : i64
    %4 = comb.extract %3 from 0 : (i64) -> i1
    %5 = comb.and %a, %t : i8
    %6 = comb.or %5, %b : i8
    %7 = comb.xor %5, %s : i8
    hw.output %2, %4, %6 : i8, i1, i8
  }
}
