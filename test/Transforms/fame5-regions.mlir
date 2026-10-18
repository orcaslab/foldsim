// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=4 clock-enable thread-index region-loads=8" | FileCheck %s
// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=4 clock-enable thread-index region-loads=16" | FileCheck %s --check-prefix=ONE
// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=1 clock-enable region-loads=8" | FileCheck %s --check-prefix=SINGLE
// RUN: not foldsim-opt %s --foldsim-fame5="module=Hold threads=4 region-loads=8" 2>&1 | FileCheck %s --check-prefix=GATED

// With regions of 8 loads, each 8-bit register is a region of its own. Its
// memory is read at the region's copy of the thread index, a register with the
// same next value as `threadIdx`, and written a host cycle after the target
// edge: at the copy's value a cycle late, with the data and the write's own
// enable staged, on the region's copy of the edge. The channel flags, only
// read here, are read and written at `threadIdx`. Copies are kept so that
// synthesis does not merge them.
// CHECK-LABEL: hw.module @Hold_threaded_Hold(
// CHECK: %threadIdx = seq.firreg [[NEXT:%.+]] clock %hostClock reset sync %hostReset, %c0_i2 preset 0 : i2
// CHECK: seq.firmem.read_port %r[%threadIdx_r0]
// CHECK: seq.firmem.write_port %r[%threadIdx_w0] = %r_wdata, clock %hostClock enable [[REN:%.+]] :
// CHECK: seq.firmem.read_port %s[%threadIdx_r1]
// CHECK: seq.firmem.write_port %s[%threadIdx_w1] = %s_wdata, clock %hostClock enable [[SEN:%.+]] :
// CHECK: seq.firmem.read_port %reset_fired[%threadIdx]
// CHECK: seq.firmem.write_port %reset_fired[%threadIdx]
// CHECK: %threadIdx_r0 = seq.firreg [[NEXT]] clock %hostClock reset sync %hostReset, %{{.+}} preset 0 {sv.attributes = [#sv.attribute<"keep" = "\22true\22">]} : i2
// CHECK: %threadIdx_w0 = seq.firreg %threadIdx_r0 clock %hostClock preset 0 {sv.attributes = [#sv.attribute<"keep" = "\22true\22">]} : i2
// CHECK: %edge_r0 = seq.firreg [[EDGE:%.+]] clock %hostClock preset 0 {sv.attributes = [#sv.attribute<"keep" = "\22true\22">]} : i1
// CHECK: %r_wdata = seq.firreg %d_sink_bits clock %hostClock : i8
// CHECK: %r_wen = seq.firreg %en_sink_bits clock %hostClock preset 0 : i1
// CHECK: [[REN]] = comb.and %r_wen, %edge_r0 : i1
// CHECK: %threadIdx_r1 = seq.firreg [[NEXT]]
// CHECK: %edge_r1 = seq.firreg [[EDGE]]
// A register with a preset is also written while the threads are initialized,
// from the region's copy of that a cycle late.
// CHECK: [[SEDGE:%.+]] = comb.and %s_wen, %edge_r1 : i1
// CHECK: %init_r1 = seq.firreg {{%.+}} clock %hostClock preset 0 {sv.attributes = [#sv.attribute<"keep" = "\22true\22">]} : i1
// CHECK: [[SEN]] = comb.or %init_r1, [[SEDGE]] : i1
// CHECK-NOT: %threadIdx_r2
// CHECK: hw.output {{.*}}, %threadIdx, [[EDGE]] :

// With 16 loads both registers share one region.
// ONE-LABEL: hw.module @Hold_threaded_Hold(
// ONE: seq.firmem.read_port %r[%threadIdx_r0]
// ONE: seq.firmem.read_port %s[%threadIdx_r0]
// ONE: seq.firmem.read_port %reset_fired[%threadIdx]
// ONE-NOT: threadIdx_r1

// One thread keeps its state in registers: no regions.
// SINGLE-LABEL: hw.module @Hold_threaded_Hold(
// SINGLE-NOT: edge_r

// Regions are for state written with the clock enable.
// GATED: regions are for state written with `clock-enable`
hw.module @Hold(in %clk : !seq.clock, in %reset : i1, in %en : i1, in %d : i8, out q : i8, out p : i8) {
  %c0_i8 = hw.constant 0 : i8
  %m = comb.mux %en, %d, %r : i8
  %r = seq.firreg %m clock %clk : i8
  %n = comb.mux %en, %d, %s : i8
  %s = seq.firreg %n clock %clk reset sync %reset, %c0_i8 preset 3 : i8
  hw.output %r, %s : i8, i8
}

// RUN: foldsim-opt %s --foldsim-fame5="module=Mem threads=4 clock-enable region-loads=1000 staged-ram-writes" | FileCheck %s --check-prefix=MEM
// RUN: foldsim-opt %s --foldsim-fame5="module=Mem threads=4 clock-enable region-loads=1000" | FileCheck %s --check-prefix=ONEDGE

// Without `staged-ram-writes`, a registered read reads on the edge of a write
// what the memory's read-under-write says, so its memory is still written on
// the edge, at the region's index; the read's enable keeps the edge.
// ONEDGE-LABEL: hw.module @Mem_threaded_Mem(
// ONEDGE: seq.firmem.write_port %ram[[[RADDR:%.+]]] = %wdata_sink_bits, clock %hostClock enable
// ONEDGE: seq.firmem.read_port %ram[{{%.+}}], clock %hostClock enable {{%[0-9]+}} :
// ONEDGE: %threadIdx_r0 = seq.firreg
// ONEDGE-DAG: [[RADDR]] = comb.concat %waddr_sink_bits, %threadIdx_r0 : i3, i2
// ONEDGE-NOT: _fwd

// A memory's ports take the region's thread index, and with
// `staged-ram-writes` every memory is written a host cycle late like the
// state. A registered read reads on every
// host cycle (its data is used only after an edge); one of a memory that
// reads new data under a write takes the staged data when the edge wrote its
// address.
// MEM-LABEL: hw.module @Mem_threaded_Mem(
// MEM: seq.firmem.write_port %lut[[[LADDR:%.+]]] = %lut_wdata, clock %hostClock enable [[LEN:%.+]] :
// MEM: seq.firmem.read_port %lut[[[LREAD:%.+]]], clock %hostClock
// MEM: seq.firmem.write_port %ram[[[RADDR:%.+]]] = %ram_wdata, clock %hostClock enable [[REN:%.+]] :
// MEM: [[RDATA:%.+]] = seq.firmem.read_port %ram[{{%.+}}], clock %hostClock enable %true :
// MEM: seq.firmem.write_port %ram_rdata[%read_thread] = [[RDATA]],
// MEM: seq.firmem.write_port %ram2[{{%.+}}] = %ram2_wdata, clock %hostClock enable {{%.+}} :
// MEM: [[NDATA:%.+]] = seq.firmem.read_port %ram2[{{%.+}}], clock %hostClock enable %true :
// MEM: seq.firmem.write_port %ram2_rdata[%read_thread{{.*}}] = [[FWD:%[0-9]+]],
// MEM: %threadIdx_r0 = seq.firreg
// MEM-DAG: [[LREAD]] = comb.concat %raddr_sink_bits, %threadIdx_r0 : i3, i2
// MEM-DAG: %lut_waddr = seq.firreg %waddr_sink_bits clock %hostClock : i3
// MEM-DAG: [[LADDR]] = comb.concat %lut_waddr, %threadIdx_w0 : i3, i2
// MEM-DAG: %lut_wen = seq.firreg %we_sink_bits clock %hostClock preset 0 : i1
// MEM-DAG: [[LEN]] = comb.and %lut_wen, %edge_r0 : i1
// MEM-DAG: %ram_waddr = seq.firreg %waddr_sink_bits clock %hostClock : i3
// MEM-DAG: [[RADDR]] = comb.concat %ram_waddr, %threadIdx_w0 : i3, i2
// MEM-DAG: [[REN]] = comb.and %ram_wen, %edge_r0 : i1
// MEM-DAG: %ram2_fwd = seq.firreg [[HIT:%.+]] clock %hostClock preset 0 : i1
// MEM-DAG: [[SAME:%.+]] = comb.icmp eq %waddr_sink_bits, %raddr_sink_bits : i3
// MEM-DAG: [[HIT]] = comb.and %we_sink_bits, [[SAME]] : i1
// MEM-DAG: [[FWD]] = comb.mux %ram2_fwd, %ram2_wdata, [[NDATA]] : i8
// MEM-NOT: ram_fwd
hw.module @Mem(in %clk : !seq.clock, in %waddr : i3, in %wdata : i8, in %we : i1, in %raddr : i3, out q : i8, out r : i8, out n : i8) {
  %true = hw.constant true
  %lut = seq.firmem 0, 1, undefined, undefined : <8 x 8>
  %q = seq.firmem.read_port %lut[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %lut[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  %ram = seq.firmem 1, 1, old, undefined : <8 x 8>
  %r = seq.firmem.read_port %ram[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %ram[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  %ram2 = seq.firmem 1, 1, new, undefined : <8 x 8>
  %n = seq.firmem.read_port %ram2[%raddr], clock %clk enable %true : <8 x 8>
  seq.firmem.write_port %ram2[%waddr] = %wdata, clock %clk enable %we : <8 x 8>
  hw.output %q, %r, %n : i8, i8, i8
}
