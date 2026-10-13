// RUN: foldsim-opt %s --foldsim-fame5="module=Mem threads=4" | FileCheck %s
// RUN: foldsim-opt %s --foldsim-fame5="module=Mem threads=1" | FileCheck %s --check-prefix=ONE

// One thread keeps a registered read's data the same way: the read port's data
// is there only on the host cycle after the target edge, so a register holds it.
// That cycle is the thread's next one, which reads the read port's data
// itself, not the register it is only now written to.
// ONE-LABEL: hw.module @Mem_threaded_Mem(
// ONE: [[DOUT:%.+]] = seq.firmem.read_port %ram[%raddr_sink_bits], clock %clk_buffer.O
// ONE: %read_edge = seq.firreg %true clock %clk_buffer.O : i1
// ONE: %ram_rdata = seq.firreg [[R:%.+]] clock %hostClock : i8
// ONE: [[EDGE:%.+]] = comb.xor %target_toggle, %target_toggle_seen : i1
// ONE: [[FRESH:%.+]] = comb.and [[EDGE]], %read_edge : i1
// ONE: [[R]] = comb.mux [[FRESH]], [[DOUT]], %ram_rdata : i8
// ONE-NOT: read_thread
// ONE: hw.output {{.*}}, [[R]], {{%.+}} : i1, i1, i1, i1, i1, i8, i1, i8, i1

// `unused` is not read, so it has no channel; `t` has no bits, so its channel
// carries tokens only.
// CHECK-LABEL: hw.module @Mem_threaded_Mem(
// CHECK-SAME:    in %raddr_sink_bits : i3,
// CHECK-NOT:     unused
// CHECK-SAME:    in %t_source_ready : i1, out t_source_valid : i1)
// CHECK-SAME:    attributes {foldsim.eager_channels = ["t"]}

// The threads share each memory, a copy of it per thread side by side.
// CHECK: %threadIdx = seq.firreg {{%.+}} clock %hostClock reset sync %hostReset, %c0_i2 preset 0 : i2

// A combinational read is one of the threaded memory.
// CHECK: %lut = seq.firmem 0, 1, undefined, undefined : <32 x 8>
// CHECK: [[LUT_WADDR:%.+]] = comb.concat %waddr_sink_bits, %threadIdx : i3, i2
// CHECK: seq.firmem.write_port %lut[[[LUT_WADDR]]] = %wdata_sink_bits, clock %clk_buffer.O enable %we_sink_bits
// CHECK: [[LUT_RADDR:%.+]] = comb.concat %raddr_sink_bits, %threadIdx : i3, i2
// CHECK: [[Q:%.+]] = seq.firmem.read_port %lut[[[LUT_RADDR]]], clock %hostClock

// A registered read happens on the target clock, and its data is kept for
// the thread that read it until the thread is current again.
// CHECK: %ram = seq.firmem 1, 1, old, undefined : <32 x 8>
// CHECK: [[DOUT:%.+]] = seq.firmem.read_port %ram[{{%.+}}], clock %clk_buffer.O
// CHECK: %read_edge = seq.firreg %true clock %clk_buffer.O : i1
// CHECK: %ram_rdata = seq.firmem 0, 1, undefined, undefined : <4 x 8, mask 1>
// CHECK: [[R:%.+]] = seq.firmem.read_port %ram_rdata[%threadIdx], clock %hostClock
// CHECK: %target_toggle = seq.firreg {{%.+}} clock %clk_buffer.O preset 0 : i1
// CHECK: %target_toggle_seen = seq.firreg %target_toggle clock %hostClock preset 0 : i1
// CHECK: [[EDGE:%.+]] = comb.xor %target_toggle, %target_toggle_seen : i1
// CHECK: [[WRITE:%.+]] = comb.and [[EDGE]], %read_edge : i1
// CHECK: %read_thread = seq.firreg %threadIdx clock %clk_buffer.O : i2
// CHECK: seq.firmem.write_port %ram_rdata[%read_thread] = [[DOUT]], clock %hostClock enable [[WRITE]]

// CHECK: hw.output {{.*}}, [[Q]], {{.*}}, [[R]], {{%.+}} : i1, i1, i1, i1, i1, i8, i1, i8, i1
hw.module @Mem(in %clk : !seq.clock, in %reset : i1, in %waddr : i3, in %wdata : i8, in %we : i1, in %raddr : i3, in %unused : i8, out q : i8, out r : i8, out t : i0) {
  %true = hw.constant true
  %c0_i0 = hw.constant 0 : i0
  %lut = seq.firmem 0, 1, undefined, undefined : <8 x 8>
  %q = seq.firmem.read_port %lut[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %lut[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  %ram = seq.firmem 1, 1, old, undefined : <8 x 8>
  %r = seq.firmem.read_port %ram[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %ram[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  hw.output %q, %r, %c0_i0 : i8, i8, i0
}
