// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child,Sink netlist" --foldsim-compose="module=Top staged-tokens" | FileCheck %s

// With staged tokens, a token, its thread and slot are registered when it is
// taken and written into its slot on the next host cycle, under the staged
// fire; the counts are those of the unstaged channel. A reader of the slot
// being written reads the staged token.
// CHECK-LABEL: hw.module @foldsim_sim(
// CHECK-DAG:     %x_enq0 = seq.firreg {{.*}} : i2
// CHECK-DAG:     %x_stg_fire = seq.firreg [[FIRE:%.+]] clock %hostClock reset sync %hostReset, {{.*}} : i1
// CHECK-DAG:     [[FIRE]] = comb.and %x_valid, {{%.+}} : i1
// CHECK-DAG:     %x_stg_slot = seq.firreg {{.*}} : i1
// CHECK-DAG:     %x_stg_bits = seq.firreg %x_bits clock %hostClock : i8
// CHECK-DAG:     %x_slot0_1 = seq.firreg [[NEXT:%.+]] clock %hostClock : i8
// CHECK-DAG:     [[NEXT]] = comb.mux [[WRITE:%.+]], %x_stg_bits, %x_slot0_1 : i8
// CHECK-DAG:     [[WRITE]] = comb.and {{%.+}}, %x_stg_slot : i1
// CHECK-DAG:     [[XEQ:%.+]] = comb.icmp eq %x_stg_slot, {{%.+}} : i1
// CHECK-DAG:     [[XHIT:%.+]] = comb.and %x_stg_fire, [[XEQ]] : i1
// CHECK-DAG:     comb.mux [[XHIT]], %x_stg_bits, {{%.+}} : i8
// A threaded producer read the same way by every thread of its reader keeps
// its slots in a memory, written from the staged token, thread and slot; the
// reader compares them with its own thread and slot.
// CHECK-DAG:     %Child_y_stg_fire = seq.firreg {{.*}} : i1
// CHECK-DAG:     %Child_y_stg_thread = seq.firreg %Child_i.threadIdx clock %hostClock reset sync %hostReset, {{.*}} : i1
// CHECK-DAG:     %Child_y_stg_bits = seq.firreg %Child_i.y_source_bits clock %hostClock : i8
// CHECK-DAG:     seq.firmem.write_port %Child_y_tokens[{{%.+}}] = %Child_y_stg_bits, clock %hostClock enable %Child_y_stg_fire : <4 x 8>
// CHECK-DAG:     [[READ:%.+]] = seq.firmem.read_port %Child_y_tokens[{{%.+}}], clock %hostClock enable {{%.+}} : <4 x 8>
// CHECK-DAG:     [[YEQ:%.+]] = comb.icmp eq %Child_y_stg_slot, {{%.+}} : i1
// CHECK-DAG:     [[YSLOT:%.+]] = comb.and %Child_y_stg_fire, [[YEQ]] : i1
// CHECK-DAG:     [[YTHREAD:%.+]] = comb.icmp eq %Child_y_stg_thread, {{%.+}} : i1
// CHECK-DAG:     [[YHIT:%.+]] = comb.and [[YSLOT]], [[YTHREAD]] : i1
// CHECK-DAG:     comb.mux [[YHIT]], %Child_y_stg_bits, [[READ]] : i8
// Without the option, nothing is staged.
// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child,Sink netlist" --foldsim-compose="module=Top" | FileCheck %s --check-prefix=PLAIN
// PLAIN-NOT: _stg_
hw.module @Child(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %s = comb.add %a, %r : i8
  %r = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Sink(in %clk : !seq.clock, in %a : i8, out y : i8) {
  %r = seq.firreg %a clock %clk : i8
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z0 : i8, out z1 : i8) {
  %a0.y = hw.instance "a0" @Child(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %a1.y = hw.instance "a1" @Child(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %b0.y = hw.instance "b0" @Sink(clk: %clk: !seq.clock, a: %a0.y: i8) -> (y: i8)
  %b1.y = hw.instance "b1" @Sink(clk: %clk: !seq.clock, a: %a1.y: i8) -> (y: i8)
  hw.output %b0.y, %b1.y : i8, i8
}
