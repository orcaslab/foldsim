"""table_roms_gen.py: a level-like module for foldsim-table-roms: an index register with a reset and a preset, a
replica of it (the same next value), two tables of 512 entries read at it and at the replica, one of 256 entries (not
merged: below min-depth 512), and a table read at a register of another next value (a ROM of its own)."""
import random
r = random.Random(7)
def table(n, w):
    return '[' + ', '.join(f'{r.randrange(1 << w)} : i{w}' for _ in range(n)) + ']'
print(f'''hw.module @foldsim_sim(in %hostClock: !seq.clock, in %hostReset: i1, in %nx: i9, in %ny: i9,
                        out a: i8, out b: i5, out c: i3, out d: i8, out e: i6) {{
  %c5 = hw.constant 5 : i9
  %idx = seq.firreg %nx clock %hostClock reset sync %hostReset, %c5 preset 3 : i9
  %idx_rep1 = seq.firreg %nx clock %hostClock reset sync %hostReset, %c5 preset 3 : i9
  %idy = seq.firreg %ny clock %hostClock : i9
  %t1 = hw.aggregate_constant {table(512, 8)} : !hw.array<512xi8>
  %t2 = hw.aggregate_constant {table(512, 5)} : !hw.array<512xi5>
  %t3 = hw.aggregate_constant {table(256, 3)} : !hw.array<256xi3>
  %t4 = hw.aggregate_constant {table(512, 6)} : !hw.array<512xi6>
  %a = hw.array_get %t1[%idx] : !hw.array<512xi8>, i9
  %b = hw.array_get %t2[%idx_rep1] : !hw.array<512xi5>, i9
  %i8 = comb.extract %idx from 0 : (i9) -> i8
  %c = hw.array_get %t3[%i8] : !hw.array<256xi3>, i8
  %d = hw.array_get %t1[%idx_rep1] : !hw.array<512xi8>, i9
  %e = hw.array_get %t4[%idy] : !hw.array<512xi6>, i9
  hw.output %a, %b, %c, %d, %e : i8, i5, i3, i8, i6
}}''')
