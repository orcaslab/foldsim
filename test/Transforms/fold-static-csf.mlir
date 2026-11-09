// Constant-shell fold, schedule only: ready-cone lanes (csf-ready-copies) and
// RAM mailbox links (csf-mailboxes).
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Shared models=Two netlist},foldsim-fold-static{module=Shared schedule-only report=%t.base})" -o /dev/null 2>&1
// RUN: FileCheck %s --check-prefix=BASE < %t.base
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Shared models=Two netlist},foldsim-fold-static{module=Shared schedule-only csf-ready-copies=Two:2 report=%t.lanes})" -o /dev/null 2>&1
// RUN: FileCheck %s --check-prefix=LANES < %t.lanes
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Pairs models=P,R netlist},foldsim-fold-static{module=Pairs schedule-only csf-mailboxes report=%t.spaced})" -o /dev/null 2>&1
// RUN: FileCheck %s --check-prefix=SPACED < %t.spaced
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Halves models=P,R netlist},foldsim-fold-static{module=Halves schedule-only csf-mailboxes report=%t.banked})" -o /dev/null 2>&1
// RUN: FileCheck %s --check-prefix=BANKED < %t.banked
// RUN: not foldsim-opt %s --mlir-print-op-on-diagnostic=false --pass-pipeline="builtin.module(foldsim-extract-models{module=Shared models=Two netlist},foldsim-fold-static{module=Shared schedule-only csf-ready-copies=Nope:2})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD

// `Two` needs two ready passes a frame (fold-static-ready-levels.mlir): its
// copy takes 2 x 3 evaluations. With a ready-cone lane per instance the copy
// keeps only the state passes and each lane its instance's ready passes.
// BASE: II = 10
// BASE-NOT: csf:
// BASE: Two x2 (2 ready passes)
// BASE-NEXT: ready: 0 1
// BASE-NEXT: ready2: 2 3
// LANES: II = 8
// LANES: csf: 2 ready-cone lanes; mailboxes off
// LANES: csf: resource bounds: {{.*}}Two (2 x 3 passes, spacing 0, mailbox 1)
// LANES-SAME: Two ready lanes x2;
// LANES: csf: windows (first..last slot): Two
// Both instances' ready passes in one slot, each on its lane.
// LANES: Two x2 (2 ready passes)
// LANES-NEXT: ready: 0 0
// LANES-NEXT: ready2: 1 1
hw.module @Two(in %clk : !seq.clock, in %x1 : i8, in %x2 : i8, out y1 : i8, out y2 : i8) {
  %r = seq.firreg %x1 clock %clk : i8
  %y1 = comb.xor %x1, %r : i8
  %y2 = comb.xor %x2, %r : i8
  hw.output %y1, %y2 : i8, i8
}
hw.module @Shared(in %clk : !seq.clock, in %in : i8, out z : i8) {
  %a.y1, %a.y2 = hw.instance "a" @Two(clk: %clk: !seq.clock, x1: %in: i8, x2: %b.y1: i8) -> (y1: i8, y2: i8)
  %b.y1, %b.y2 = hw.instance "b" @Two(clk: %clk: !seq.clock, x1: %in: i8, x2: %a.y1: i8) -> (y1: i8, y2: i8)
  %z = comb.xor %a.y2, %b.y2 : i8
  hw.output %z : i8
}

// Ten readers each read a distinct pair of five producers: the pairs form a
// 5-clique, which no 4 lanes (2 x the 2 rows a reader) keep apart, so each
// reader pass holds its mailbox port 2 host cycles (10 passes x 2 = 20).
// SPACED: II = 20
// SPACED: csf: 0 ready-cone lanes; mailboxes on, link latency 2
// SPACED: R (10 x 1 passes, spacing 0, mailbox 2)
// SPACED: csf: mailbox R <- P: 2 rows a reader, spaced
// Two readers of disjoint pairs: lane j mod 2 keeps every pair apart.
// BANKED: csf: mailbox R <- P: 2 rows a reader, banked
// BANKED-NOT: spaced
// BAD: csf-ready-copies: no model 'Nope'
hw.module @P(in %clk : !seq.clock, in %x : i8, out y : i8) {
  %r = seq.firreg %x clock %clk : i8
  hw.output %r : i8
}
hw.module @R(in %clk : !seq.clock, in %a : i8, in %b : i8, out z : i8) {
  %s = comb.xor %a, %b : i8
  %r = seq.firreg %s clock %clk : i8
  hw.output %r : i8
}
hw.module @Pairs(in %clk : !seq.clock, in %in : i8, out z : i8) {
  %p0.y = hw.instance "p0" @P(clk: %clk: !seq.clock, x: %in: i8) -> (y: i8)
  %c1 = comb.add %in, %in : i8
  %p1.y = hw.instance "p1" @P(clk: %clk: !seq.clock, x: %c1: i8) -> (y: i8)
  %c2 = comb.mul %in, %in : i8
  %p2.y = hw.instance "p2" @P(clk: %clk: !seq.clock, x: %c2: i8) -> (y: i8)
  %c3 = comb.sub %in, %c1 : i8
  %p3.y = hw.instance "p3" @P(clk: %clk: !seq.clock, x: %c3: i8) -> (y: i8)
  %c4 = comb.or %in, %c2 : i8
  %p4.y = hw.instance "p4" @P(clk: %clk: !seq.clock, x: %c4: i8) -> (y: i8)
  %r0.z = hw.instance "r0" @R(clk: %clk: !seq.clock, a: %p0.y: i8, b: %p1.y: i8) -> (z: i8)
  %r1.z = hw.instance "r1" @R(clk: %clk: !seq.clock, a: %p0.y: i8, b: %p2.y: i8) -> (z: i8)
  %r2.z = hw.instance "r2" @R(clk: %clk: !seq.clock, a: %p0.y: i8, b: %p3.y: i8) -> (z: i8)
  %r3.z = hw.instance "r3" @R(clk: %clk: !seq.clock, a: %p0.y: i8, b: %p4.y: i8) -> (z: i8)
  %r4.z = hw.instance "r4" @R(clk: %clk: !seq.clock, a: %p1.y: i8, b: %p2.y: i8) -> (z: i8)
  %r5.z = hw.instance "r5" @R(clk: %clk: !seq.clock, a: %p1.y: i8, b: %p3.y: i8) -> (z: i8)
  %r6.z = hw.instance "r6" @R(clk: %clk: !seq.clock, a: %p1.y: i8, b: %p4.y: i8) -> (z: i8)
  %r7.z = hw.instance "r7" @R(clk: %clk: !seq.clock, a: %p2.y: i8, b: %p3.y: i8) -> (z: i8)
  %r8.z = hw.instance "r8" @R(clk: %clk: !seq.clock, a: %p2.y: i8, b: %p4.y: i8) -> (z: i8)
  %r9.z = hw.instance "r9" @R(clk: %clk: !seq.clock, a: %p3.y: i8, b: %p4.y: i8) -> (z: i8)
  %z = comb.xor %r0.z, %r1.z, %r2.z, %r3.z, %r4.z, %r5.z, %r6.z, %r7.z, %r8.z, %r9.z : i8
  hw.output %z : i8
}
hw.module @Halves(in %clk : !seq.clock, in %in : i8, out z : i8) {
  %p0.y = hw.instance "p0" @P(clk: %clk: !seq.clock, x: %in: i8) -> (y: i8)
  %c1 = comb.add %in, %in : i8
  %p1.y = hw.instance "p1" @P(clk: %clk: !seq.clock, x: %c1: i8) -> (y: i8)
  %c2 = comb.mul %in, %in : i8
  %p2.y = hw.instance "p2" @P(clk: %clk: !seq.clock, x: %c2: i8) -> (y: i8)
  %c3 = comb.sub %in, %c1 : i8
  %p3.y = hw.instance "p3" @P(clk: %clk: !seq.clock, x: %c3: i8) -> (y: i8)
  %r0.z = hw.instance "r0" @R(clk: %clk: !seq.clock, a: %p0.y: i8, b: %p1.y: i8) -> (z: i8)
  %r1.z = hw.instance "r1" @R(clk: %clk: !seq.clock, a: %p2.y: i8, b: %p3.y: i8) -> (z: i8)
  %z = comb.xor %r0.z, %r1.z : i8
  hw.output %z : i8
}
