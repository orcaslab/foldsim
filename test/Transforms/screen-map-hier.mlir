// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-screen-map,foldsim-estimate-timing{output=- report= top=Top arrivals=%t.json})' -o /dev/null | FileCheck %s --check-prefix=ONE
// RUN: FileCheck %s --check-prefix=JSON < %t.json
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-screen-map{arrivals=%t.json},foldsim-estimate-timing{output=- report= top=Top})' -o /dev/null | FileCheck %s --check-prefix=TWO

// The probe of screen-map-arrival.mlir across module boundaries: a 33-bit
// carry (level 6) selects between two 216-input parities (three levels
// each). In `Pick` the carry is an input port, computed in `Top`; in `Top`
// the selected carry is the output of `Carry`. Mapped apart, each module
// takes its boundary at 0 and buries the carry under a parity cone (8);
// with the first mapping's arrival times at the ports, the second keeps it
// for the last LUT (7), as the flattened design maps. The registers are outputs: foldsim-estimate-timing leaves
// out end points nothing observable reads (upstream c0363d0).
// ONE-DAG: 8,{{.*}},1,o.d[0]
// ONE-DAG: 8,{{.*}},0,p.d[0]
// JSON: {"Pick":{"in":{"sel":[6]},"out":{}},"Carry":{"in":{},"out":{"c":[6]}}}
// TWO-DAG: 7,{{.*}},1,o.d[0]
// TWO-DAG: 7,{{.*}},0,p.d[0]
hw.module @Pick(in %clk : !seq.clock, in %sel : i1, in %x : i216,
                in %y : i216, out o : i1) {
  %px = comb.parity %x : i216
  %py = comb.parity %y : i216
  %0 = comb.mux %sel, %px, %py : i1
  %o = seq.firreg %0 clock %clk : i1
  hw.output %o : i1
}
hw.module @Carry(in %a : i32, in %b : i32, out c : i1) {
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i32
  %1 = comb.concat %false, %b : i1, i32
  %2 = comb.add %0, %1 : i33
  %3 = comb.extract %2 from 32 : (i33) -> i1
  hw.output %3 : i1
}
hw.module @Top(in %clk : !seq.clock, in %a_in : i32, in %b_in : i32,
               in %x_in : i216, in %y_in : i216, out o : i1, out p : i1) {
  %a = seq.firreg %a_in clock %clk : i32
  %b = seq.firreg %b_in clock %clk : i32
  %x = seq.firreg %x_in clock %clk : i216
  %y = seq.firreg %y_in clock %clk : i216
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i32
  %1 = comb.concat %false, %b : i1, i32
  %2 = comb.add %0, %1 : i33
  %sel = comb.extract %2 from 32 : (i33) -> i1
  %po = hw.instance "pick" @Pick(clk: %clk: !seq.clock, sel: %sel: i1, x: %x: i216,
                                 y: %y: i216) -> (o: i1)
  %c = hw.instance "carry" @Carry(a: %a: i32, b: %b: i32) -> (c: i1)
  %px = comb.parity %x : i216
  %py = comb.parity %y : i216
  %3 = comb.mux %c, %px, %py : i1
  %p = seq.firreg %3 clock %clk : i1
  hw.output %po, %p : i1, i1
}
