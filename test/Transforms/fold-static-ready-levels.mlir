// Ready passes by level: where one ready pass per model leaves every choice
// of ready passes with a cycle, a model's ready outputs are written in as few
// ready passes as keep the frame acyclic, each once the inputs its outputs
// depend on are written.
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Shared models=Two netlist},foldsim-fold-static{module=Shared schedule-only})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=SHARED
// RUN: not foldsim-opt %s --mlir-print-op-on-diagnostic=false --pass-pipeline="builtin.module(foldsim-extract-models{module=Shared models=Two netlist},foldsim-fold-static{module=Shared schedule-only ready-levels=off})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Shared models=Two netlist},foldsim-fold-static{module=Shared})" 2>/dev/null | FileCheck %s --check-prefix=SIM
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Chain models=Csr,Slice netlist},foldsim-fold-static{module=Chain schedule-only})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHAIN
// RUN: not foldsim-opt %s --mlir-print-op-on-diagnostic=false --pass-pipeline="builtin.module(foldsim-extract-models{module=Chain models=Csr,Slice netlist},foldsim-fold-static{module=Chain schedule-only ready-levels=off})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Chain models=Csr,Slice netlist},foldsim-fold-static{module=Chain schedule-only ready-levels=rank})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHAIN
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Even models=Pair netlist},foldsim-fold-static{module=Even schedule-only})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=EVEN
// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=Even models=Pair netlist},foldsim-fold-static{module=Even schedule-only ready-levels=always})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=EVEN
// RUN: not foldsim-opt %s --mlir-print-op-on-diagnostic=false --pass-pipeline="builtin.module(foldsim-extract-models{module=Even models=Pair netlist},foldsim-fold-static{module=Even schedule-only ready-levels=some})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD

// `Two`'s outputs each depend on one input, and `a`'s `y1` feeds `b`'s `x2`
// and back: one ready pass writing both waits on itself through the other
// instance's. Two ready passes, `y1`'s then `y2`'s, schedule.
// SHARED: II =
// SHARED: Two x2 (2 ready passes)
// SHARED-NEXT: ready: [[R0:[0-9]+]] [[R1:[0-9]+]]
// SHARED-NEXT: ready2:
// SHARED-NEXT: state:
// SHARED-NEXT: late y1 @ready <- x1
// SHARED-NEXT: late y2 @ready2 <- x2
// OFF: error: the evaluations of a frame depend on each other in a cycle for every choice of ready passes
// OFF-NOT: split by level
// The second pass writes `y2`'s links at its own enable, a host cycle-delayed
// copy of the slot table's row as the first pass's.
// SIM: %Two_ready_l1_2 = seq.firreg
// SIM: %Two_links0_ready_l1_1 = seq.firreg %Two_ready_l1_2
// SIM-NEXT: %Two_links0_ready_l1_0 = seq.firreg %Two_links0_ready_l1_1
// SIM: seq.firmem.write_port %Two_y2_to_{{.*}} = %Two_copy.y2, clock %hostClock enable %Two_links0_ready_l1_0
// SIM: seq.firmem.write_port %Two_y1_to_{{.*}} = %Two_copy.y1, clock %hostClock enable %Two_links0_ready0
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

// A register written by any of three slices in one cycle, each slice reading
// the value the slices before it leave (Vortex's per-core SATP with VM): by
// model, slice `w` -> `s1` -> slice `w` is a loop, but by instance a chain,
// `a.w` -> `s1` -> `b.w` -> `s2` -> `c.w`. `Csr` writes `s1` and `s2` in two
// ready passes, each slice its `w` in one; by rank the same (the slices' `w`
// ranks 0, 2 and 4, one rank per instance).
// CHAIN: II =
// CHAIN: Csr x1 (2 ready passes)
// CHAIN: late s1 @ready <- w0
// CHAIN-NEXT: late s2 @ready2 <- w0 w1
// CHAIN: Slice x3 (ready pass)
// CHAIN: late w @ready <- s
hw.module @Slice(in %clk : !seq.clock, in %s : i8, out w : i8) {
  %r = seq.firreg %s clock %clk : i8
  %w = comb.add %s, %r : i8
  hw.output %w : i8
}
hw.module @Csr(in %clk : !seq.clock, in %w0 : i8, in %w1 : i8, in %w2 : i8, out s0 : i8, out s1 : i8, out s2 : i8) {
  %r = seq.firreg %w2 clock %clk : i8
  %s1 = comb.xor %r, %w0 : i8
  %s2 = comb.xor %s1, %w1 : i8
  hw.output %r, %s1, %s2 : i8, i8, i8
}
hw.module @Chain(in %clk : !seq.clock, out z : i8) {
  %csr.s0, %csr.s1, %csr.s2 = hw.instance "csr" @Csr(clk: %clk: !seq.clock, w0: %a.w: i8, w1: %b.w: i8, w2: %c.w: i8) -> (s0: i8, s1: i8, s2: i8)
  %a.w = hw.instance "a" @Slice(clk: %clk: !seq.clock, s: %csr.s0: i8) -> (w: i8)
  %b.w = hw.instance "b" @Slice(clk: %clk: !seq.clock, s: %csr.s1: i8) -> (w: i8)
  %c.w = hw.instance "c" @Slice(clk: %clk: !seq.clock, s: %csr.s2: i8) -> (w: i8)
  hw.output %c.w : i8
}

// No loop through the instances: the same schedule with `auto` (which never
// splits a schedule found without) and with `always` (whose levels merge).
// EVEN: II = 8
// EVEN: Pair x2 state gap 8
// EVEN-NEXT: state:
// EVEN-NEXT: late y1 @state <- x1
// EVEN-NEXT: late y2 @state <- x2
// BAD: error: ready-levels 'some' is none of off, auto, always and rank
hw.module @Pair(in %clk : !seq.clock, in %x1 : i8, in %x2 : i8, out y1 : i8, out y2 : i8) {
  %r = seq.firreg %x1 clock %clk : i8
  %y1 = comb.xor %x1, %r : i8
  %y2 = comb.xor %x2, %r : i8
  hw.output %y1, %y2 : i8, i8
}
hw.module @Even(in %clk : !seq.clock, in %in : i8, out z : i8) {
  %a.y1, %a.y2 = hw.instance "a" @Pair(clk: %clk: !seq.clock, x1: %in: i8, x2: %in: i8) -> (y1: i8, y2: i8)
  %b.y1, %b.y2 = hw.instance "b" @Pair(clk: %clk: !seq.clock, x1: %in: i8, x2: %a.y1: i8) -> (y1: i8, y2: i8)
  %z = comb.xor %a.y2, %b.y2 : i8
  hw.output %z : i8
}
