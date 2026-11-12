hw.module @M(in %clk : !seq.clock, in %i7 : i7, in %i6 : i6, in %x : i4, in %y : i4, in %u : i4, in %v : i4, in %a12 : i12, in %d1 : i1, in %d72 : i72, in %m9 : i9, in %a13 : i13, in %we : i1, out t : i8, out r : i16, out p : i26, out q1 : i1, out q72 : i72, out q72c : i72) {
  %c0_i13 = hw.constant 0 : i13
  %c0_i5 = hw.constant 0 : i5
  %true = hw.constant true
  %tab = hw.aggregate_constant [68 : i8, 32 : i8, 130 : i8, 60 : i8, 253 : i8, 230 : i8, 241 : i8, 194 : i8, 107 : i8, 48 : i8, 249 : i8, 14 : i8, 199 : i8, 221 : i8, 1 : i8, 228 : i8, 136 : i8, 117 : i8, 52 : i8, 162 : i8, 15 : i8, 11 : i8, 13 : i8, 4 : i8, 195 : i8, 110 : i8, 216 : i8, 14 : i8, 113 : i8, 224 : i8, 253 : i8, 119 : i8, 176 : i8, 118 : i8, 112 : i8, 235 : i8, 148 : i8, 11 : i8, 213 : i8, 51 : i8, 95 : i8, 151 : i8, 61 : i8, 170 : i8, 216 : i8, 97 : i8, 155 : i8, 145 : i8, 255 : i8, 201 : i8, 17 : i8, 245 : i8, 124 : i8, 206 : i8, 212 : i8, 88 : i8, 187 : i8, 191 : i8, 44 : i8, 224 : i8, 55 : i8, 83 : i8, 201 : i8, 189 : i8, 250 : i8, 15 : i8, 240 : i8, 22 : i8, 157 : i8, 201 : i8, 87 : i8, 86 : i8, 116 : i8, 6 : i8, 102 : i8, 118 : i8, 207 : i8, 176 : i8, 180 : i8, 235 : i8, 137 : i8, 2 : i8, 196 : i8, 66 : i8, 105 : i8, 218 : i8, 28 : i8, 246 : i8, 186 : i8, 102 : i8, 211 : i8, 248 : i8, 182 : i8, 212 : i8, 177 : i8, 0 : i8, 169 : i8, 234 : i8, 14 : i8, 117 : i8, 90 : i8, 92 : i8, 46 : i8, 130 : i8, 16 : i8, 36 : i8, 42 : i8, 8 : i8, 231 : i8, 7 : i8, 143 : i8, 127 : i8, 137 : i8, 56 : i8, 94 : i8, 176 : i8, 148 : i8, 35 : i8, 85 : i8, 81 : i8, 130 : i8, 86 : i8, 139 : i8, 150 : i8, 232 : i8, 164 : i8, 254 : i8, 242 : i8] : !hw.array<128xi8>
  %0 = hw.array_get %tab[%i7] : !hw.array<128xi8>, i7
  %rom0 = hw.aggregate_constant [14967 : i16, 3097 : i16, 40895 : i16, 50666 : i16, 45002 : i16, 55170 : i16, 24646 : i16, 33871 : i16, 14255 : i16, 33221 : i16, 27405 : i16, 56577 : i16, 2728 : i16, 29540 : i16, 2341 : i16, 52076 : i16, 19197 : i16, 4630 : i16, 21001 : i16, 58414 : i16, 55923 : i16, 28914 : i16, 59093 : i16, 29254 : i16, 4023 : i16, 51760 : i16, 42106 : i16, 55875 : i16, 7705 : i16, 39138 : i16, 16473 : i16, 27804 : i16, 6218 : i16, 40158 : i16, 9270 : i16, 10019 : i16, 40679 : i16, 39043 : i16, 20736 : i16, 54548 : i16, 33077 : i16, 17090 : i16, 1111 : i16, 4969 : i16, 28520 : i16, 60404 : i16, 22481 : i16, 4905 : i16, 49541 : i16, 26267 : i16, 45472 : i16, 12979 : i16, 26969 : i16, 56747 : i16, 25443 : i16, 64533 : i16, 13687 : i16, 51126 : i16, 38806 : i16, 65509 : i16, 2254 : i16, 42643 : i16, 52733 : i16, 36877 : i16] : !hw.array<64xi16>
  %rom = hw.wire %rom0 {sv.attributes = [#sv.attribute<"rom_style" = "\22block\22">]} : !hw.array<64xi16>
  %1 = hw.array_get %rom[%i6] : !hw.array<64xi16>, i6
  %rd = seq.firreg %1 clock %clk : i16
  %2 = comb.concat %c0_i13, %x, %c0_i5, %y : i13, i4, i5, i4
  %3 = comb.concat %c0_i13, %u, %c0_i5, %v : i13, i4, i5, i4
  %4 = comb.mul %2, %3 : i26
  %tag = seq.firmem 0, 1, undefined, undefined : <4096 x 1, mask 1>
  seq.firmem.write_port %tag[%a12] = %d1, clock %clk enable %we : <4096 x 1, mask 1>
  %5 = seq.firmem.read_port %tag[%a12], clock %clk : <4096 x 1, mask 1>
  %res = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <8192 x 72, mask 9>
  seq.firmem.write_port %res[%a13] = %d72 mask %m9, clock %clk enable %we : <8192 x 72, mask 9>, i9
  %6 = seq.firmem.read_port %res[%a13], clock %clk enable %true : <8192 x 72, mask 9>
  %resc = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">, #sv.attribute<"cascade_height" = "2">]} : <8192 x 72, mask 9>
  seq.firmem.write_port %resc[%a13] = %d72 mask %m9, clock %clk enable %we : <8192 x 72, mask 9>, i9
  %7 = seq.firmem.read_port %resc[%a13], clock %clk enable %true : <8192 x 72, mask 9>
  hw.output %0, %rd, %4, %5, %6, %7 : i8, i16, i26, i1, i72, i72
}
