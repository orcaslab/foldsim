// A memory under its SSA name (the printer elides the `name` attribute when the name is a legal identifier), one with an
// explicit name, and one without a name: foldsim_cost.parse marks the first two named (a --storage rule can address them).
hw.module @Mems(in %clk : !seq.clock, in %a : i4, in %d : i8, out q : i8, out r : i8, out s : i8) {
  %true = hw.constant true
  %ram = seq.firmem 1, 1, old, undefined : <16 x 8>
  %0 = seq.firmem.read_port %ram[%a], clock %clk enable %true : <16 x 8>
  seq.firmem.write_port %ram[%a] = %d, clock %clk enable %true : <16 x 8>
  %1 = seq.firmem name "inner/x.ram" 1, 1, old, undefined : <16 x 8>
  %2 = seq.firmem.read_port %1[%a], clock %clk enable %true : <16 x 8>
  %3 = seq.firmem 1, 1, old, undefined : <16 x 8>
  %4 = seq.firmem.read_port %3[%a], clock %clk enable %true : <16 x 8>
  hw.output %0, %2, %4 : i8, i8, i8
}
