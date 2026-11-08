// REQUIRES: verilator
// Folded pipelines' stores in RAM (foldsim-fold-static `fold-store-ram`): each
// store read through a register ahead of the read it serves, at the instance
// then due and its slot, the slot counter the copy's own (an entry per
// instance, stepped a host cycle after each issue), so that the store may be
// block or ultra RAM instead of LUTs. The statically folded simulator against
// the target, its channels stalled at random: without the option, with it
// (two host cycles ahead: the RAM's register and an output register), one
// host cycle ahead on two copies (copy 0's register pipeline store, of three
// slots: its issue's write lands outAt + 1 = 5 host cycles after the issue,
// its next read ahead is a state gap of 6 less one after it, so a write and a
// read ahead of one instance meet in a host cycle, at different slots), and
// only the stores of at least 150 bits.
// RUN: rm -rf %t && mkdir -p %t && cat %S/Inputs/fold_model.mlir %s > %t/in.mlir
// RUN: firtool --format=mlir %t/in.mlir -o %t/target.sv
// RUN: mkdir -p %t/off && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top})" -o %t/off/sim.mlir 2> %t/off/fold.log
// RUN: FileCheck %s --check-prefix=OFF < %t/off/sim.mlir
// RUN: not grep fold-store-ram %t/off/fold.log
// RUN: firtool --format=mlir %t/off/sim.mlir -o %t/off/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/off/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/off/sim.sv > %t/off/build.log 2>&1
// RUN: %t/off/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: mkdir -p %t/on && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top fold-store-ram=1})" -o %t/on/sim.mlir 2> %t/on/fold.log
// RUN: FileCheck %s --check-prefix=ON < %t/on/sim.mlir
// RUN: FileCheck %s --check-prefix=ONLOG < %t/on/fold.log
// RUN: firtool --format=mlir %t/on/sim.mlir -o %t/on/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/on/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/on/sim.sv > %t/on/build.log 2>&1
// RUN: %t/on/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/on/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: %t/on/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// RUN: mkdir -p %t/l1 && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top copies=Model:2 fold-store-ram=1 fold-store-lead=1 fold-store-style=ultra})" -o %t/l1/sim.mlir 2> %t/l1/fold.log
// RUN: FileCheck %s --check-prefix=L1 < %t/l1/sim.mlir
// RUN: FileCheck %s --check-prefix=L1LOG < %t/l1/fold.log
// RUN: firtool --format=mlir %t/l1/sim.mlir -o %t/l1/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/l1/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/l1/sim.sv > %t/l1/build.log 2>&1
// RUN: %t/l1/obj/Vtb +verilator+rand+reset+2 +verilator+seed+4 | FileCheck %s
// RUN: %t/l1/obj/Vtb +verilator+rand+reset+2 +verilator+seed+5 | FileCheck %s
// RUN: mkdir -p %t/min && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top fold-store-ram=150 storage=Model:fold0_store=ultra})" -o %t/min/sim.mlir 2> %t/min/fold.log
// RUN: FileCheck %s --check-prefix=MIN < %t/min/sim.mlir
// RUN: FileCheck %s --check-prefix=MINLOG < %t/min/fold.log
// RUN: firtool --format=mlir %t/min/sim.mlir -o %t/min/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/min/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/min/sim.sv > %t/min/build.log 2>&1
// RUN: %t/min/obj/Vtb +verilator+rand+reset+2 +verilator+seed+6 | FileCheck %s
// Errors: a style or a lead that is not one.
// RUN: not foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top fold-store-ram=1 fold-store-style=bram})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=STYLE
// RUN: not foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top fold-store-ram=1 fold-store-lead=3})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=LEAD

// CHECK: {{^}}MATCH 4 outputs x 88 cycles

// Without the option: read combinationally at the target slot counters.
// OFF: %fold0_store = seq.firmem 0, 1, undefined, undefined : <12 x 16, mask 1>
// OFF: seq.firmem.read_port %fold0_store[{{%.+}}], clock %hostClock
// OFF: %fold1_store = seq.firmem 0, 1, undefined, undefined : <16 x 8, mask 1>
// OFF-NOT: _slots

// Two ahead: the RAM (block: 12 and 16 entries) read at the instance two host
// cycles ahead and its slot, into the RAM's register and the output register;
// the slot counter written at the issue's instance a host cycle on, read at
// the issuing instance and at the instance ahead. Nothing to forward.
// ON: %fold0_store = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <12 x 16, mask 1>
// ON: %fold0_store_slots = seq.firmem 0, 1, undefined, undefined : <4 x 2, mask 1>
// ON: seq.firmem.write_port %fold0_store_slots[%fold0_thread1] = {{%.+}}, clock %hostClock enable %fold0_valid1
// ON: seq.firmem.read_port %fold0_store_slots[%fold_inst]
// ON: [[S:%.+]] = seq.firmem.read_port %fold0_store_slots[%ahead0_inst2]
// ON: [[A:%.+]] = comb.concat {{%.+}}, %ahead0_inst2
// ON: [[D:%.+]] = seq.firmem.read_port %fold0_store[[[A]]], clock %hostClock
// ON: %fold0_store_read = seq.firreg [[D]] clock %hostClock
// ON: %fold1_store = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <16 x 8, mask 1>
// ON-NOT: _ahead_
// ON: seq.firmem.write_port %fold0_store[{{%.+}}] = %fold0_wdata, clock %hostClock enable %fold0_valid5
// ONLOG: fold-store-ram: Model: 2 stores, 320 bits; lead 2/1/0: 2/0/0 (state gap 10, II 10)

// One ahead (two copies of two instances, ultra by option): the RAM's
// register alone, read at the instance a host cycle ahead.
// L1: %fold0_store = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <6 x 16, mask 1>
// L1: [[A:%.+]] = comb.concat {{%.+}}, %region0_inst1
// L1: seq.firmem.read_port %fold0_store[[[A]]], clock %hostClock
// L1-NOT: %fold0_store_read =
// L1LOG: fold-store-ram: Model: 2 stores, 160 bits; lead 2/1/0: 0/2/0 (state gap 6, II 8)
// L1LOG: fold-store-ram: Model_copy1: 2 stores, 160 bits; lead 2/1/0: 0/2/0 (state gap 5, II 8)

// At least 150 bits: the register pipeline's store (3 slots x 4 instances x
// 16 bits = 192), ultra by its storage rule; the black box's (128) stays.
// MIN: %fold0_store = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <12 x 16, mask 1>
// MIN: %fold1_store = seq.firmem 0, 1, undefined, undefined : <16 x 8, mask 1>
// MINLOG: fold-store-ram: Model: 1 stores, 192 bits; lead 2/1/0: 1/0/0

// STYLE: fold-store-style: 'bram' is not ultra, block or auto
// LEAD: fold-store-lead: 3 is not 1 or 2

hw.module @Top(in %clk : !seq.clock, in %reset : i1, in %a : i8, in %en : i4, in %v : i4, out y0 : i32, out y1 : i32, out y2 : i32, out y3 : i32) {
  %c1 = hw.constant 1 : i8
  %c2 = hw.constant 2 : i8
  %c3 = hw.constant 3 : i8
  %e0 = comb.extract %en from 0 : (i4) -> i1
  %e1 = comb.extract %en from 1 : (i4) -> i1
  %e2 = comb.extract %en from 2 : (i4) -> i1
  %e3 = comb.extract %en from 3 : (i4) -> i1
  %v0 = comb.extract %v from 0 : (i4) -> i1
  %v1 = comb.extract %v from 1 : (i4) -> i1
  %v2 = comb.extract %v from 2 : (i4) -> i1
  %v3 = comb.extract %v from 3 : (i4) -> i1
  %a1 = comb.xor %a, %c1 : i8
  %a2 = comb.xor %a, %c2 : i8
  %a3 = comb.xor %a, %c3 : i8
  %y0 = hw.instance "m0" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a: i8, en: %e0: i1, v: %v0: i1) -> (y: i32)
  %y1 = hw.instance "m1" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a1: i8, en: %e1: i1, v: %v1: i1) -> (y: i32)
  %y2 = hw.instance "m2" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a2: i8, en: %e2: i1, v: %v2: i1) -> (y: i32)
  %y3 = hw.instance "m3" @Model(clk: %clk: !seq.clock, reset: %reset: i1, a: %a3: i8, en: %e3: i1, v: %v3: i1) -> (y: i32)
  hw.output %y0, %y1, %y2, %y3 : i32, i32, i32, i32
}
