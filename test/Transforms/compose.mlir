// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child netlist" --foldsim-compose="module=Top" | FileCheck %s

// Two instances of `Child`, threaded into one module; the rest of `Top` is a
// model of one thread. `x` is exactly an input of that model, and `z` one of
// its outputs, so both are connected directly; `reset` reaches both threads of
// `Child`, so it is a producer of tokens with a reader per thread.
// CHECK-LABEL: hw.module @foldsim_sim(in %hostClock : !seq.clock, in %hostReset : i1,
// CHECK-SAME:    in %reset_valid : i1, in %reset_bits : i1, out reset_ready : i1,
// CHECK-SAME:    in %x_valid : i1, in %x_bits : i8, out x_ready : i1,
// CHECK-SAME:    in %z_ready : i1, out z_valid : i1, out z_bits : i8, out profile_finish : i2)
// CHECK-SAME:    attributes {foldsim.threads = ["Top_model=1", "Child=2"]}
// CHECK:         %Top_model_i.x_sink_ready, {{.*}} = hw.instance "Top_model_i" @Top_model_threaded(hostClock: %hostClock: !seq.clock, hostReset: %hostReset: i1, x_sink_valid: %x_valid: i1, x_sink_bits: %x_bits: i8,
// CHECK:         {{.*}}, %Child_i.threadIdx, %Child_i.finish = hw.instance "Child_i" @Child_threaded(
// CHECK-DAG:     %Child_reset_deq0_0 = seq.firreg {{.*}} reset sync %hostReset
// CHECK-DAG:     %Child_reset_deq1_0 = seq.firreg {{.*}} reset sync %hostReset
// CHECK-DAG:     %reset_enq0 = seq.firreg {{.*}} reset sync %hostReset
// The output of thread 0 feeds thread 1, and thread 1's the rest of `Top`.
// CHECK-DAG:     %Child_y_enq0 = seq.firreg
// CHECK-DAG:     %Child_y_enq1 = seq.firreg
// CHECK:         hw.output %{{.+}}, %{{.+}}, %Top_model_i.z_source_valid, %Top_model_i.z_source_bits, %{{.+}} : i1, i1, i1, i8, i2
// CHECK-NOT:   hw.module @Top(
// CHECK-NOT:   hw.module private @Child(

// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child netlist" --foldsim-compose="module=Top" | FileCheck %s --check-prefix=READY

// Each thread of `Child` is ready when both its inputs have a token; the
// threaded module goes to a ready thread rather than the next in turn.
// READY-LABEL: hw.module @Child_threaded(
// READY-SAME:    in %thread_ready : i2
// READY:         comb.and %thread_ready,

// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child netlist" --foldsim-compose="module=Top ready-schedule=false" | FileCheck %s --check-prefix=TURN
// TURN-LABEL: hw.module @foldsim_sim(
// TURN-NOT:   thread_ready
hw.module @Child(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %s = comb.add %a, %r : i8
  %r = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z : i8) {
  %sum = comb.add %x, %x : i8
  %a.y = hw.instance "a" @Child(clk: %clk: !seq.clock, reset: %reset: i1, a: %sum: i8) -> (y: i8)
  %b.y = hw.instance "b" @Child(clk: %clk: !seq.clock, reset: %reset: i1, a: %a.y: i8) -> (y: i8)
  %z = comb.xor %b.y, %x : i8
  hw.output %z : i8
}

// RUN: foldsim-opt %s --foldsim-extract-models="module=Socket models=Leaf netlist" --foldsim-extract-models="module=Nested models=Socket netlist" --foldsim-compose="module=Nested sim=nested_sim" | FileCheck %s --check-prefix=NEST

// A netlist inside a netlist is inlined: the leaves of both sockets are four
// threads of one model. A leaf reads the same bits of the same producer in
// every thread, so that producer's slots are a memory.
// NEST-LABEL: hw.module @nested_sim(
// NEST-SAME:    attributes {foldsim.threads = ["Nested_model=1", "Socket_model=2", "Leaf=4"]}
// NEST:         seq.firmem 0, 1, undefined, undefined : <8 x 16>
// NEST-NOT:   foldsim.netlist
hw.module @Leaf(in %clk : !seq.clock, in %a : i16, out y : i16) {
  %s = comb.add %a, %r : i16
  %r = seq.firreg %s clock %clk : i16
  hw.output %r : i16
}
hw.module @Socket(in %clk : !seq.clock, in %in : i16, out out : i16) {
  %l0.y = hw.instance "l0" @Leaf(clk: %clk: !seq.clock, a: %t: i16) -> (y: i16)
  %l1.y = hw.instance "l1" @Leaf(clk: %clk: !seq.clock, a: %u: i16) -> (y: i16)
  %t = comb.add %in, %l1.y : i16
  %u = comb.xor %in, %l0.y : i16
  %o = comb.add %l0.y, %l1.y : i16
  hw.output %o : i16
}
hw.module @Nested(in %clk : !seq.clock, in %x : i16, out z : i16) {
  %s0.out = hw.instance "s0" @Socket(clk: %clk: !seq.clock, in: %x: i16) -> (out: i16)
  %s1.out = hw.instance "s1" @Socket(clk: %clk: !seq.clock, in: %s0.out: i16) -> (out: i16)
  hw.output %s1.out : i16
}
