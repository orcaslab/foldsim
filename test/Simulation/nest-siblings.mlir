// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: firtool --format=mlir %s -o %t/target.sv
// Flat, in turn, the image flow's state options: the reference.
// RUN: mkdir -p %t/f && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots})" -o %t/f/sim.mlir
// RUN: firtool --format=mlir %t/f/sim.mlir -o %t/f/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/f/obj %S/Inputs/tb_nest.sv %t/target.sv %t/f/sim.sv > %t/f/build.log 2>&1
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// Nested, banked tokens, commit by enable.
// RUN: mkdir -p %t/n && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots nest banked-tokens commit-enable})" -o %t/n/sim.mlir
// RUN: firtool --format=mlir %t/n/sim.mlir -o %t/n/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/n/obj %S/Inputs/tb_nest.sv %t/target.sv %t/n/sim.sv > %t/n/build.log 2>&1
// RUN: %t/n/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/n/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// The same with the memories and thread state split into parts of at most 4
// loads.
// RUN: mkdir -p %t/m && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots nest banked-tokens commit-enable memory-loads=4})" -o %t/m/sim.mlir
// RUN: grep -q '_p1 = seq.firmem' %t/m/sim.mlir
// RUN: firtool --format=mlir %t/m/sim.mlir -o %t/m/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/m/obj %S/Inputs/tb_nest.sv %t/target.sv %t/m/sim.sv > %t/m/build.log 2>&1
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// The same as a tree of domains, with a relay between L2 and L3 and the leaves in two copies.
// RUN: mkdir -p %t/d && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots nest banked-tokens commit-enable domains relay=L2_model:L3 copies=L3:2})" -o %t/d/sim.mlir
// RUN: firtool --format=mlir %t/d/sim.mlir -o %t/d/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/d/obj %S/Inputs/tb_nest.sv %t/target.sv %t/d/sim.sv > %t/d/build.log 2>&1
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// Flat with the same domains, relay and copies.
// RUN: mkdir -p %t/r && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots domains relay=L2_model:L3 copies=L3:2})" -o %t/r/sim.mlir
// RUN: firtool --format=mlir %t/r/sim.mlir -o %t/r/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/r/obj %S/Inputs/tb_nest.sv %t/target.sv %t/r/sim.sv > %t/r/build.log 2>&1
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// The first child's input has constant high bits, the others none: the
// reader takes, by thread, a constant or a producer's bits at one position,
// and every threaded producer's slots are in memories.
// RUN: FileCheck %s --check-prefix=BANK < %t/n/sim.mlir
// BANK-NOT: L{{[12]}}_model_u{{[0-9]}}_x_slot
// The relay is on the copies of L3 too. With banked tokens as without, each
// reading class has its own memories and slots on its side, a copy apart from
// its original; the netlist's reset, read by relayed models, is relayed too.
// RUN: FileCheck %s --check-prefix=RELAY < %t/d/sim.mlir
// RELAY-DAG: L3_copy1_threaded
// RELAY-DAG: %L2_model_u0_x_rtokens_L3 = seq.firmem
// RELAY-DAG: %L2_model_u1_x_rtokens_L3_copy1 = seq.firmem
// RELAY-DAG: %L3_copy1_y_rtokens_L2_model = seq.firmem
// RELAY-DAG: %L3_y_rtokens_L2_model = seq.firmem
// RELAY-DAG: %reset_renq0 = seq.firreg
// RELAY-DAG: %reset_rslot0_L3_0 = seq.firreg
// RELAY-DAG: %reset_rslot0_L3_copy1_0 = seq.firreg
// RELAY-DAG: %reset_rslot0_L2_model_0 = seq.firreg
// RUN: FileCheck %s --check-prefix=SHARED < %t/d/sim.mlir
// SHARED-NOT: {{%(reset|L3_y|L3_copy1_y)_slot}}

// Three levels of two instances each; a level grants each child from all its
// children's outputs of the same cycle, so a child's cycle depends on its
// siblings'; the first child's input has constant high bits; against the
// unthreaded target (tb_nest.sv).
// CHECK: MATCH
module {
  hw.module @L3(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y : i8, in %g : i4) {
    %c49_i16 = hw.constant 49 : i16
    %c18567_i16 = hw.constant 18567 : i16
    %c1_i8 = hw.constant 1 : i8
    %c0_i4 = hw.constant 0 : i4
    %c0_i8 = hw.constant 0 : i8
    %0 = comb.replicate %x : (i8) -> i16
    %1 = comb.extract %c02Fr from 0 : (i16) -> i15
    %2 = comb.extract %c02Fr from 15 : (i16) -> i1
    %3 = comb.concat %1, %2 : i15, i1
    %4 = comb.xor %3, %0, %c18567_i16 : i16
    %5 = comb.mux %reset, %c49_i16, %4 : i16
    %c02Fr = seq.firreg %5 clock %clk {name = "c0/r"} : i16
    %6 = comb.add %t, %c1_i8 : i8
    %7 = comb.concat %c0_i4, %g : i4, i4
    %8 = comb.xor %6, %7 : i8
    %9 = comb.mux %reset, %c0_i8, %8 : i8
    %t = seq.firreg %9 clock %clk : i8
    %10 = comb.extract %c02Fr from 0 : (i16) -> i8
    %11 = comb.xor %10, %t, %x : i8
    hw.output %11 : i8
  }
  hw.module @L2(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y : i8, in %g : i4) {
    %c23_i8 = hw.constant 23 : i8
    %c101_i8 = hw.constant 101 : i8
    %c33_i16 = hw.constant 33 : i16
    %c1_i4 = hw.constant 1 : i4
    %true = hw.constant true
    %c0_i4 = hw.constant 0 : i4
    %c0_i8 = hw.constant 0 : i8
    %c2_i8 = hw.constant 2 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i4 = hw.constant 5 : i4
    %0 = comb.replicate %x : (i8) -> i16
    %1 = comb.extract %c02Fr from 0 : (i16) -> i15
    %2 = comb.extract %c02Fr from 15 : (i16) -> i1
    %3 = comb.concat %1, %2 : i15, i1
    %4 = comb.xor %18, %c23_i8 : i8
    %5 = comb.concat %c101_i8, %4 : i8, i8
    %6 = comb.xor %3, %0, %5 : i16
    %7 = comb.mux %reset, %c33_i16, %6 : i16
    %c02Fr = seq.firreg %7 clock %clk {name = "c0/r"} : i16
    %8 = comb.extract %c02Fr from 0 : (i16) -> i4
    %9 = comb.extract %t from 0 : (i8) -> i4
    %10 = comb.extract %x from 0 : (i8) -> i4
    %11 = comb.xor %8, %9, %10, %c1_i4 : i4
    %12 = comb.concat %c5_i4, %11 : i4, i4
    %u0.y = hw.instance "u0" @L3(clk: %clk: !seq.clock, reset: %reset: i1, x: %12: i8, g: %29: i4) -> (y: i8) {sv.namehint = "cy0"}
    %13 = comb.extract %c02Fr from 3 : (i16) -> i8
    %14 = comb.xor %13, %t, %x, %c2_i8 : i8
    %u1.y = hw.instance "u1" @L3(clk: %clk: !seq.clock, reset: %reset: i1, x: %14: i8, g: %35: i4) -> (y: i8) {sv.namehint = "cy1"}
    %15 = comb.extract %u1.y from 0 : (i8) -> i7
    %16 = comb.extract %u1.y from 7 : (i8) -> i1
    %17 = comb.concat %15, %16 : i7, i1
    %18 = comb.xor %u0.y, %17 : i8
    %19 = comb.extract %18 from 0 : (i8) -> i4
    %20 = comb.concat %c0_i4, %19 : i4, i4
    %21 = comb.add %t, %20, %c1_i8 : i8
    %22 = comb.concat %c0_i4, %g : i4, i4
    %23 = comb.xor %21, %22 : i8
    %24 = comb.mux %reset, %c0_i8, %23 : i8
    %t = seq.firreg %24 clock %clk : i8
    %25 = comb.extract %u0.y from 0 : (i8) -> i1
    %26 = comb.extract %u0.y from 1 : (i8) -> i3
    %27 = comb.extract %18 from 1 : (i8) -> i3
    %28 = comb.xor %26, %27 : i3
    %29 = comb.concat %25, %28 {sv.namehint = "g0"} : i1, i3
    %30 = comb.extract %u1.y from 0 : (i8) -> i1
    %31 = comb.xor %25, %true : i1
    %32 = comb.and %30, %31 : i1
    %33 = comb.extract %u1.y from 1 : (i8) -> i3
    %34 = comb.xor %33, %27 : i3
    %35 = comb.concat %32, %34 {sv.namehint = "g1"} : i1, i3
    %36 = comb.extract %c02Fr from 0 : (i16) -> i8
    %37 = comb.xor %36, %t, %18, %x : i8
    hw.output %37 : i8
  }
  hw.module @L1(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y : i8, in %g : i4) {
    %c-89_i8 = hw.constant -89 : i8
    %c-127_i8 = hw.constant -127 : i8
    %c17_i16 = hw.constant 17 : i16
    %c1_i4 = hw.constant 1 : i4
    %true = hw.constant true
    %c0_i4 = hw.constant 0 : i4
    %c0_i8 = hw.constant 0 : i8
    %c2_i8 = hw.constant 2 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i4 = hw.constant 5 : i4
    %0 = comb.replicate %x : (i8) -> i16
    %1 = comb.extract %c02Fr from 0 : (i16) -> i15
    %2 = comb.extract %c02Fr from 15 : (i16) -> i1
    %3 = comb.concat %1, %2 : i15, i1
    %4 = comb.xor %18, %c-89_i8 : i8
    %5 = comb.concat %c-127_i8, %4 : i8, i8
    %6 = comb.xor %3, %0, %5 : i16
    %7 = comb.mux %reset, %c17_i16, %6 : i16
    %c02Fr = seq.firreg %7 clock %clk {name = "c0/r"} : i16
    %8 = comb.extract %c02Fr from 0 : (i16) -> i4
    %9 = comb.extract %t from 0 : (i8) -> i4
    %10 = comb.extract %x from 0 : (i8) -> i4
    %11 = comb.xor %8, %9, %10, %c1_i4 : i4
    %12 = comb.concat %c5_i4, %11 : i4, i4
    %u0.y = hw.instance "u0" @L2(clk: %clk: !seq.clock, reset: %reset: i1, x: %12: i8, g: %29: i4) -> (y: i8) {sv.namehint = "cy0"}
    %13 = comb.extract %c02Fr from 3 : (i16) -> i8
    %14 = comb.xor %13, %t, %x, %c2_i8 : i8
    %u1.y = hw.instance "u1" @L2(clk: %clk: !seq.clock, reset: %reset: i1, x: %14: i8, g: %35: i4) -> (y: i8) {sv.namehint = "cy1"}
    %15 = comb.extract %u1.y from 0 : (i8) -> i7
    %16 = comb.extract %u1.y from 7 : (i8) -> i1
    %17 = comb.concat %15, %16 : i7, i1
    %18 = comb.xor %u0.y, %17 : i8
    %19 = comb.extract %18 from 0 : (i8) -> i4
    %20 = comb.concat %c0_i4, %19 : i4, i4
    %21 = comb.add %t, %20, %c1_i8 : i8
    %22 = comb.concat %c0_i4, %g : i4, i4
    %23 = comb.xor %21, %22 : i8
    %24 = comb.mux %reset, %c0_i8, %23 : i8
    %t = seq.firreg %24 clock %clk : i8
    %25 = comb.extract %u0.y from 0 : (i8) -> i1
    %26 = comb.extract %u0.y from 1 : (i8) -> i3
    %27 = comb.extract %18 from 1 : (i8) -> i3
    %28 = comb.xor %26, %27 : i3
    %29 = comb.concat %25, %28 {sv.namehint = "g0"} : i1, i3
    %30 = comb.extract %u1.y from 0 : (i8) -> i1
    %31 = comb.xor %25, %true : i1
    %32 = comb.and %30, %31 : i1
    %33 = comb.extract %u1.y from 1 : (i8) -> i3
    %34 = comb.xor %33, %27 : i3
    %35 = comb.concat %32, %34 {sv.namehint = "g1"} : i1, i3
    %36 = comb.extract %c02Fr from 0 : (i16) -> i8
    %37 = comb.xor %36, %t, %18, %x : i8
    hw.output %37 : i8
  }
  hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y : i8) {
    %c55_i8 = hw.constant 55 : i8
    %c-98_i8 = hw.constant -98 : i8
    %c1_i16 = hw.constant 1 : i16
    %c1_i4 = hw.constant 1 : i4
    %c0_i4 = hw.constant 0 : i4
    %true = hw.constant true
    %c0_i8 = hw.constant 0 : i8
    %c2_i8 = hw.constant 2 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i4 = hw.constant 5 : i4
    %0 = comb.replicate %x : (i8) -> i16
    %1 = comb.extract %c02Fr from 0 : (i16) -> i15
    %2 = comb.extract %c02Fr from 15 : (i16) -> i1
    %3 = comb.concat %1, %2 : i15, i1
    %4 = comb.xor %18, %c55_i8 : i8
    %5 = comb.concat %c-98_i8, %4 : i8, i8
    %6 = comb.xor %3, %0, %5 : i16
    %7 = comb.mux %reset, %c1_i16, %6 : i16
    %c02Fr = seq.firreg %7 clock %clk {name = "c0/r"} : i16
    %8 = comb.extract %c02Fr from 0 : (i16) -> i4
    %9 = comb.extract %t from 0 : (i8) -> i4
    %10 = comb.extract %x from 0 : (i8) -> i4
    %11 = comb.xor %8, %9, %10, %c1_i4 : i4
    %12 = comb.concat %c5_i4, %11 : i4, i4
    %u0.y = hw.instance "u0" @L1(clk: %clk: !seq.clock, reset: %reset: i1, x: %12: i8, g: %27: i4) -> (y: i8) {sv.namehint = "cy0"}
    %13 = comb.extract %c02Fr from 3 : (i16) -> i8
    %14 = comb.xor %13, %t, %x, %c2_i8 : i8
    %u1.y = hw.instance "u1" @L1(clk: %clk: !seq.clock, reset: %reset: i1, x: %14: i8, g: %33: i4) -> (y: i8) {sv.namehint = "cy1"}
    %15 = comb.extract %u1.y from 0 : (i8) -> i7
    %16 = comb.extract %u1.y from 7 : (i8) -> i1
    %17 = comb.concat %15, %16 : i7, i1
    %18 = comb.xor %u0.y, %17 : i8
    %19 = comb.extract %18 from 0 : (i8) -> i4
    %20 = comb.concat %c0_i4, %19 : i4, i4
    %21 = comb.add %t, %20, %c1_i8 : i8
    %22 = comb.mux %reset, %c0_i8, %21 : i8
    %t = seq.firreg %22 clock %clk : i8
    %23 = comb.extract %u0.y from 0 : (i8) -> i1
    %24 = comb.extract %u0.y from 1 : (i8) -> i3
    %25 = comb.extract %18 from 1 : (i8) -> i3
    %26 = comb.xor %24, %25 : i3
    %27 = comb.concat %23, %26 {sv.namehint = "g0"} : i1, i3
    %28 = comb.extract %u1.y from 0 : (i8) -> i1
    %29 = comb.xor %23, %true : i1
    %30 = comb.and %28, %29 : i1
    %31 = comb.extract %u1.y from 1 : (i8) -> i3
    %32 = comb.xor %31, %25 : i3
    %33 = comb.concat %30, %32 {sv.namehint = "g1"} : i1, i3
    %34 = comb.extract %c02Fr from 0 : (i16) -> i8
    %35 = comb.xor %34, %t, %18, %x : i8
    hw.output %35 : i8
  }
}

