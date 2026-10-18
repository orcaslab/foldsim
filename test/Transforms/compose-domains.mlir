// RUN: foldsim-opt %s --foldsim-extract-models="module=Mid models=Leaf netlist" --foldsim-extract-models="module=Top models=Mid netlist" --foldsim-compose="module=Top domains" | FileCheck %s
// RUN: foldsim-opt %s --foldsim-extract-models="module=Mid models=Leaf netlist" --foldsim-extract-models="module=Top models=Mid netlist" --foldsim-compose="module=Top domains relay=Leaf" | FileCheck %s --check-prefix=RELAY

// The class tree is Top > Mid > Leaf: the simulator is a domain per class,
// nested as they are. Mid's domain holds Mid's model, the channels between it
// and the Leaf threads, and Leaf's domain, which holds Leaf's model; the top
// domain, the simulator, holds Top's model, its channels to Mid, and Mid's
// domain. Domains connect through their ports only.
// CHECK-LABEL: hw.module private @foldsim_sim_Mid(
// CHECK:         hw.instance "Mid_model_i" @Mid_model_threaded(
// CHECK:         hw.instance "Leaf_dom" @foldsim_sim_Leaf(
// CHECK-NOT:     sv.attributes
// CHECK-LABEL: hw.module private @foldsim_sim_Leaf(
// CHECK:         hw.instance "Leaf_i" @Leaf_threaded(
// CHECK-NOT:     hw.instance
// CHECK-LABEL: hw.module @foldsim_sim(
// CHECK:         hw.instance "Top_model_i" @Top_model_threaded(
// CHECK:         hw.instance "Mid_dom" @foldsim_sim_Mid(

// With Leaf's channels relayed, the relay's sides are in the two domains
// and the boundary is kept in synthesis.
// RELAY-LABEL: hw.module private @foldsim_sim_Mid(
// RELAY:         hw.instance "Leaf_dom" @foldsim_sim_Leaf({{.*}}) {sv.attributes = [#sv.attribute<"keep_hierarchy" = "\22yes\22">]}
// RELAY-LABEL: hw.module private @foldsim_sim_Leaf(
// RELAY-DAG:     %Leaf_y_renq{{[0-9]+}} = seq.firreg
// RELAY-DAG:     %Leaf_a_rdeq{{[0-9_]+}} = seq.firreg
hw.module @Leaf(in %clk : !seq.clock, in %reset : i1, in %a : i8, out y : i8) {
  %c0 = hw.constant 0 : i8
  %s = comb.add %a, %r : i8
  %r = seq.firreg %s clock %clk reset sync %reset, %c0 : i8
  hw.output %r : i8
}
hw.module @Mid(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z : i8) {
  %c0 = hw.constant 0 : i8
  %a.y = hw.instance "a" @Leaf(clk: %clk: !seq.clock, reset: %reset: i1, a: %x: i8) -> (y: i8)
  %b.y = hw.instance "b" @Leaf(clk: %clk: !seq.clock, reset: %reset: i1, a: %t: i8) -> (y: i8)
  %t = seq.firreg %a.y clock %clk reset sync %reset, %c0 : i8
  %z = comb.xor %b.y, %t : i8
  hw.output %z : i8
}
hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z : i8) {
  %m.z = hw.instance "m" @Mid(clk: %clk: !seq.clock, reset: %reset: i1, x: %x: i8) -> (z: i8)
  %n.z = hw.instance "n" @Mid(clk: %clk: !seq.clock, reset: %reset: i1, x: %m.z: i8) -> (z: i8)
  hw.output %n.z : i8
}
