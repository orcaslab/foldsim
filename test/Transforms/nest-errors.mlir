// The nested order's errors.
// RUN: not foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top nest})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=READY
// READY: nest: the nested order is a static order; it needs every model picked in turn
// RUN: not foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots nest nest-offsets=L2_model:2})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=HOP
// HOP: nest: the offset of 'L2_model' from 'L1_model' (0) is shorter than the control's hop (1)
// RUN: not foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false nest nest-control=free})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONTROL
// CONTROL: nest-control: 'free' is neither tree nor counters
// RUN: not foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false region-loads=1 registered-state commit-select state-stages=2 pipeline-multipliers staged-ram-writes staged-tokens compact-staging idle-slots nest nest-offsets=L1_model:1})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=PHASE
// PHASE: nest: the offset of 'L1_model' (1) is not a multiple of its turn's 2 host cycles
// RUN: not foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false nest copies=L3:4})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=SPLIT
// SPLIT: nest: the threads of 'L3' do not follow the turns of 'L2_model'
// RUN: not foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false copies=L3})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=SYNTAX
// SYNTAX: copies: 'L3' is not <model>:<count>
// RUN: not foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false nest nest-offsets=L9:2})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODEL
// MODEL: nest-offsets: no model 'L9'
// RUN: not foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false copies=L9:2})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=COPYMODEL
// COPYMODEL: copies: no model 'L9'
// RUN: not foldsim-opt %s --pass-pipeline="builtin.module(foldsim-extract-models{module=L2 models=L3 netlist},foldsim-extract-models{module=L1 models=L2 netlist},foldsim-extract-models{module=Top models=L1 netlist},foldsim-compose{module=Top ready-schedule=false copies=L3:3})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=DIVIDE
// DIVIDE: copies: 8 instances of 'L3' do not split into 3 copies

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

