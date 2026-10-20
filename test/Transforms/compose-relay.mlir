// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child netlist" --foldsim-compose="module=Top relay=Child" | FileCheck %s

// Every channel between `Child` and `Top`'s model is relayed: four slots and
// 3-bit counts; the producer sees each reader's count a host cycle late, the
// token is written a host cycle after it is taken, and the reader sees the
// producer's count a host cycle late, from a copy of its own.
// CHECK-LABEL: hw.module @foldsim_sim(
// CHECK-DAG:     %Child_a_rdeq0_0 = seq.firreg {{.*}} : i3
// CHECK-DAG:     %Top_model_a_a_renq0 = seq.firreg {{.*}} : i3
// CHECK-DAG:     %Child_a_deqd0_0 = seq.firreg %Child_a_rdeq0_0 clock %hostClock reset sync %hostReset, {{.*}} : i3
// CHECK-DAG:     [[COUNT:%.+]] = comb.sub %Top_model_a_a_renq0, %Child_a_deqd0_0 : i3
// CHECK-DAG:     comb.icmp ne [[COUNT]], %c-4_i3{{.*}} : i3
// CHECK-DAG:     %Top_model_a_a_stg_fire = seq.firreg
// CHECK-DAG:     %Top_model_a_a_stg_bits = seq.firreg %Top_model_i.a_a_source_bits clock %hostClock : i8
// The reading class has its own slots.
// CHECK-DAG:     %Top_model_a_a_rslot0_Child_3 = seq.firreg {{.*}} : i8
// CHECK-DAG:     %Child_a_enqd_Top_model_a_a_0 = seq.firreg %Top_model_a_a_renq0 clock %hostClock reset sync %hostReset, {{.*}} : i3
// CHECK-DAG:     comb.icmp ne %Child_a_enqd_Top_model_a_a_0, %Child_a_rdeq0_0 : i3
// So is a channel from the netlist's inputs a relayed model reads: it would
// otherwise reach the model from the top unregistered.
// CHECK-DAG:     %reset_renq0 = seq.firreg {{.*}} : i3
// CHECK-DAG:     %reset_stg_bits = seq.firreg %reset_bits clock %hostClock : i1
// CHECK-DAG:     %reset_rslot0_Child_3 = seq.firreg {{.*}} : i1
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

// RUN: foldsim-opt %s --foldsim-extract-models="module=Top models=Child netlist" --foldsim-compose="module=Top relay=Top_model:Child" | FileCheck %s --check-prefix=PAIR
// A relay names models: a netlist's own is `<netlist>_model`, and a name
// that is none is an error rather than a channel left without its relay.
// RUN: not foldsim-opt %s --foldsim-extract-models="module=Top models=Child netlist" --foldsim-compose="module=Top relay=Top:Child" 2>&1 | FileCheck %s --check-prefix=BAD
// BAD: relay: 'Top' is not a model of the netlist (models:
// A pair relays the channels between the two models.
// PAIR-DAG: %reset_renq0 = seq.firreg
// PAIR-DAG: %Top_model_a_a_renq0 = seq.firreg
// PAIR-DAG: %Child_y_renq0 = seq.firreg
