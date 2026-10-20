// REQUIRES: verilator
// RUN: rm -rf %t && mkdir -p %t
// RUN: firtool --format=mlir %s -o %t/target.sv
// Flat, in turn, the image flow's state options: the reference for the rest.
// RUN: mkdir -p %t/f && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots})" -o %t/f/sim.mlir
// RUN: firtool --format=mlir %t/f/sim.mlir -o %t/f/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/f/obj %S/Inputs/tb_nest.sv %t/target.sv %t/f/sim.sv > %t/f/build.log 2>&1
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/f/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// Nested: each class's threads numbered after its parent's turns.
// RUN: mkdir -p %t/n && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots nest})" -o %t/n/sim.mlir
// RUN: firtool --format=mlir %t/n/sim.mlir -o %t/n/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/n/obj %S/Inputs/tb_nest.sv %t/target.sv %t/n/sim.sv > %t/n/build.log 2>&1
// RUN: %t/n/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/n/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// Nested, children current two host cycles after their parent.
// RUN: mkdir -p %t/k && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots nest nest-skew=2})" -o %t/k/sim.mlir
// RUN: firtool --format=mlir %t/k/sim.mlir -o %t/k/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/k/obj %S/Inputs/tb_nest.sv %t/target.sv %t/k/sim.sv > %t/k/build.log 2>&1
// RUN: %t/k/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/k/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// Banked tokens: a parent's port per child kept in memories.
// RUN: mkdir -p %t/b && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots banked-tokens})" -o %t/b/sim.mlir
// RUN: firtool --format=mlir %t/b/sim.mlir -o %t/b/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/b/obj %S/Inputs/tb_nest.sv %t/target.sv %t/b/sim.sv > %t/b/build.log 2>&1
// RUN: %t/b/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/b/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// Nested with banked tokens, plain threading.
// RUN: mkdir -p %t/m && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false nest banked-tokens})" -o %t/m/sim.mlir
// RUN: firtool --format=mlir %t/m/sim.mlir -o %t/m/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/m/obj %S/Inputs/tb_nest.sv %t/target.sv %t/m/sim.sv > %t/m/build.log 2>&1
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/m/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// Nested with banked tokens, the leaves and the middle level in two copies each.
// RUN: mkdir -p %t/c && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots nest nest-skew=2 banked-tokens copies=L3:2,L1_model:2})" -o %t/c/sim.mlir
// RUN: firtool --format=mlir %t/c/sim.mlir -o %t/c/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c/obj %S/Inputs/tb_nest.sv %t/target.sv %t/c/sim.sv > %t/c/build.log 2>&1
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/c/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// Nested by readiness: the numbering only, schedules free.
// RUN: mkdir -p %t/r && foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots nest banked-tokens})" -o %t/r/sim.mlir
// RUN: firtool --format=mlir %t/r/sim.mlir -o %t/r/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/r/obj %S/Inputs/tb_nest.sv %t/target.sv %t/r/sim.sv > %t/r/build.log 2>&1
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 +stall=50 | FileCheck %s
// RUN: %t/r/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 +stall=50 | FileCheck %s
// A parent's thread index is the low digits of its children's: L1's thread is
// current on even host cycles for u0, odd for u1, and so on down.
// RUN: FileCheck %s --check-prefix=ORDER < %t/m/sim.mlir
// ORDER: "L2_model=u0_u0_logic u1_u0_logic u0_u1_logic u1_u1_logic", "L3=u0_u0_u0 u1_u0_u0 u0_u1_u0 u1_u1_u0 u0_u0_u1 u1_u0_u1 u0_u1_u1 u1_u1_u1"
// The banked memories replace the per-child slot registers.
// RUN: FileCheck %s --check-prefix=BANK < %t/b/sim.mlir
// BANK-NOT: L1_model_u0_x_slot
// BANK: seq.firmem
// RUN: FileCheck %s --check-prefix=COPY < %t/c/sim.mlir
// COPY-DAG: hw.module @L3_threaded(
// COPY-DAG: hw.module @L3_copy1_threaded(
// COPY-DAG: hw.module @L1_model_copy1_threaded(

// Three levels of two instances each, every level with state of its own;
// a level's output depends combinationally on its input and its children's
// outputs, so every target cycle is a loop through the whole hierarchy;
// against the unthreaded target, reset at the start and in mid-run
// (tb_nest.sv).
// CHECK: MATCH
module {
  hw.module @L3(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y : i8) {
    %c49_i16 = hw.constant 49 : i16
    %c18567_i16 = hw.constant 18567 : i16
    %c1_i8 = hw.constant 1 : i8
    %c0_i8 = hw.constant 0 : i8
    %0 = comb.replicate %x : (i8) -> i16
    %1 = comb.extract %c02Fr from 0 : (i16) -> i15
    %2 = comb.extract %c02Fr from 15 : (i16) -> i1
    %3 = comb.concat %1, %2 : i15, i1
    %4 = comb.xor %3, %0, %c18567_i16 : i16
    %5 = comb.mux %reset, %c49_i16, %4 : i16
    %c02Fr = seq.firreg %5 clock %clk {name = "c0/r"} : i16
    %6 = comb.add %t, %c1_i8 : i8
    %7 = comb.mux %reset, %c0_i8, %6 : i8
    %t = seq.firreg %7 clock %clk : i8
    %8 = comb.extract %c02Fr from 0 : (i16) -> i8
    %9 = comb.xor %8, %t, %x : i8
    hw.output %9 : i8
  }
  hw.module @L2(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y : i8) {
    %c23_i8 = hw.constant 23 : i8
    %c101_i8 = hw.constant 101 : i8
    %c33_i16 = hw.constant 33 : i16
    %c0_i4 = hw.constant 0 : i4
    %c0_i8 = hw.constant 0 : i8
    %c2_i8 = hw.constant 2 : i8
    %c1_i8 = hw.constant 1 : i8
    %0 = comb.replicate %x : (i8) -> i16
    %1 = comb.extract %c02Fr from 0 : (i16) -> i15
    %2 = comb.extract %c02Fr from 15 : (i16) -> i1
    %3 = comb.concat %1, %2 : i15, i1
    %4 = comb.xor %15, %c23_i8 : i8
    %5 = comb.concat %c101_i8, %4 : i8, i8
    %6 = comb.xor %3, %0, %5 : i16
    %7 = comb.mux %reset, %c33_i16, %6 : i16
    %c02Fr = seq.firreg %7 clock %clk {name = "c0/r"} : i16
    %8 = comb.extract %c02Fr from 0 : (i16) -> i8
    %9 = comb.xor %8, %t, %x, %c1_i8 : i8
    %u0.y = hw.instance "u0" @L3(clk: %clk: !seq.clock, reset: %reset: i1, x: %9: i8) -> (y: i8) {sv.namehint = "cy0"}
    %10 = comb.extract %c02Fr from 3 : (i16) -> i8
    %11 = comb.xor %10, %t, %x, %c2_i8 : i8
    %u1.y = hw.instance "u1" @L3(clk: %clk: !seq.clock, reset: %reset: i1, x: %11: i8) -> (y: i8) {sv.namehint = "cy1"}
    %12 = comb.extract %u1.y from 0 : (i8) -> i7
    %13 = comb.extract %u1.y from 7 : (i8) -> i1
    %14 = comb.concat %12, %13 : i7, i1
    %15 = comb.xor %u0.y, %14 : i8
    %16 = comb.extract %15 from 0 : (i8) -> i4
    %17 = comb.concat %c0_i4, %16 : i4, i4
    %18 = comb.add %t, %17, %c1_i8 : i8
    %19 = comb.mux %reset, %c0_i8, %18 : i8
    %t = seq.firreg %19 clock %clk : i8
    %20 = comb.xor %8, %t, %15, %x : i8
    hw.output %20 : i8
  }
  hw.module @L1(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y : i8) {
    %c-89_i8 = hw.constant -89 : i8
    %c-127_i8 = hw.constant -127 : i8
    %c17_i16 = hw.constant 17 : i16
    %c0_i4 = hw.constant 0 : i4
    %c0_i8 = hw.constant 0 : i8
    %c2_i8 = hw.constant 2 : i8
    %c1_i8 = hw.constant 1 : i8
    %0 = comb.replicate %x : (i8) -> i16
    %1 = comb.extract %c02Fr from 0 : (i16) -> i15
    %2 = comb.extract %c02Fr from 15 : (i16) -> i1
    %3 = comb.concat %1, %2 : i15, i1
    %4 = comb.xor %15, %c-89_i8 : i8
    %5 = comb.concat %c-127_i8, %4 : i8, i8
    %6 = comb.xor %3, %0, %5 : i16
    %7 = comb.mux %reset, %c17_i16, %6 : i16
    %c02Fr = seq.firreg %7 clock %clk {name = "c0/r"} : i16
    %8 = comb.extract %c02Fr from 0 : (i16) -> i8
    %9 = comb.xor %8, %t, %x, %c1_i8 : i8
    %u0.y = hw.instance "u0" @L2(clk: %clk: !seq.clock, reset: %reset: i1, x: %9: i8) -> (y: i8) {sv.namehint = "cy0"}
    %10 = comb.extract %c02Fr from 3 : (i16) -> i8
    %11 = comb.xor %10, %t, %x, %c2_i8 : i8
    %u1.y = hw.instance "u1" @L2(clk: %clk: !seq.clock, reset: %reset: i1, x: %11: i8) -> (y: i8) {sv.namehint = "cy1"}
    %12 = comb.extract %u1.y from 0 : (i8) -> i7
    %13 = comb.extract %u1.y from 7 : (i8) -> i1
    %14 = comb.concat %12, %13 : i7, i1
    %15 = comb.xor %u0.y, %14 : i8
    %16 = comb.extract %15 from 0 : (i8) -> i4
    %17 = comb.concat %c0_i4, %16 : i4, i4
    %18 = comb.add %t, %17, %c1_i8 : i8
    %19 = comb.mux %reset, %c0_i8, %18 : i8
    %t = seq.firreg %19 clock %clk : i8
    %20 = comb.xor %8, %t, %15, %x : i8
    hw.output %20 : i8
  }
  hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %x : i8, out y : i8) {
    %c55_i8 = hw.constant 55 : i8
    %c-98_i8 = hw.constant -98 : i8
    %c1_i16 = hw.constant 1 : i16
    %c0_i4 = hw.constant 0 : i4
    %c0_i8 = hw.constant 0 : i8
    %c2_i8 = hw.constant 2 : i8
    %c1_i8 = hw.constant 1 : i8
    %0 = comb.replicate %x : (i8) -> i16
    %1 = comb.extract %c02Fr from 0 : (i16) -> i15
    %2 = comb.extract %c02Fr from 15 : (i16) -> i1
    %3 = comb.concat %1, %2 : i15, i1
    %4 = comb.xor %15, %c55_i8 : i8
    %5 = comb.concat %c-98_i8, %4 : i8, i8
    %6 = comb.xor %3, %0, %5 : i16
    %7 = comb.mux %reset, %c1_i16, %6 : i16
    %c02Fr = seq.firreg %7 clock %clk {name = "c0/r"} : i16
    %8 = comb.extract %c02Fr from 0 : (i16) -> i8
    %9 = comb.xor %8, %t, %x, %c1_i8 : i8
    %u0.y = hw.instance "u0" @L1(clk: %clk: !seq.clock, reset: %reset: i1, x: %9: i8) -> (y: i8) {sv.namehint = "cy0"}
    %10 = comb.extract %c02Fr from 3 : (i16) -> i8
    %11 = comb.xor %10, %t, %x, %c2_i8 : i8
    %u1.y = hw.instance "u1" @L1(clk: %clk: !seq.clock, reset: %reset: i1, x: %11: i8) -> (y: i8) {sv.namehint = "cy1"}
    %12 = comb.extract %u1.y from 0 : (i8) -> i7
    %13 = comb.extract %u1.y from 7 : (i8) -> i1
    %14 = comb.concat %12, %13 : i7, i1
    %15 = comb.xor %u0.y, %14 : i8
    %16 = comb.extract %15 from 0 : (i8) -> i4
    %17 = comb.concat %c0_i4, %16 : i4, i4
    %18 = comb.add %t, %17, %c1_i8 : i8
    %19 = comb.mux %reset, %c0_i8, %18 : i8
    %t = seq.firreg %19 clock %clk : i8
    %20 = comb.xor %8, %t, %15, %x : i8
    hw.output %20 : i8
  }
}

