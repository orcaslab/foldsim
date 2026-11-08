module {
  hw.module @A_static(in %hostClock : !seq.clock, in %i : i8, in %ra : i12, in %wa : i12, in %wd : i80, in %we : i1, out o : i80) {
    %u = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <4096 x 80, mask 1>
    %0 = seq.firmem.read_port %u[%ra], clock %hostClock : <4096 x 80, mask 1>
    seq.firmem.write_port %u[%wa] = %wd, clock %hostClock enable %we : <4096 x 80, mask 1>
    %s = seq.firmem name "fold0_store" 0, 1, undefined, undefined : <1024 x 7, mask 1>
    %1 = seq.firmem.read_port %s[%ra], clock %hostClock : <1024 x 7, mask 1>
    %r = seq.firreg %i clock %hostClock : i8
    hw.output %0 : i80
  }
  hw.module @B_static(in %hostClock : !seq.clock, in %i : i8, out o : i8) {
    %r = seq.firreg %i clock %hostClock : i8
    %q = seq.firreg %r clock %hostClock : i8
    hw.output %q : i8
  }
  hw.module @foldsim_sim(in %hostClock : !seq.clock, in %x : i8, in %ra : i12, in %wa : i12, in %wd : i80, in %we : i1, out y : i8, out z : i80) {
    %A_copy.o = hw.instance "A_copy" @A_static(hostClock: %hostClock: !seq.clock, i: %x: i8, ra: %ra: i12, wa: %wa: i12, wd: %wd: i80, we: %we: i1) -> (o: i80)
    %A_o_to_B_i = seq.firreg %x clock %hostClock : i8
    %B_copy.o = hw.instance "B_copy" @B_static(hostClock: %hostClock: !seq.clock, i: %A_o_to_B_i: i8) -> (o: i8)
    hw.output %B_copy.o, %A_copy.o : i8, i80
  }
}
