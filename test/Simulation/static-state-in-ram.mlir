// REQUIRES: verilator
// XFAIL: *
// State in RAM (SiR, foldsim-fold-static `state-in-ram`; DESIGN.md of fold-sir-1108): a class's
// register state kept in RAM (URAM / BRAM), a row an instance times P plus the pass, read in place
// two host cycles ahead and written back from the next-state outputs, no fetch buffer, no per-bit
// select. Written before the implementation (XFAIL until it lands: then drop the XFAIL and tune the
// CHECK names to the emitted ones). The statically folded simulator against the target, channels
// stalled at random (exactness): without the option (netlist unchanged), P = 1 (the whole state in
// one row: auto style, ultra, block), two copies, P > 1 (`state-in-ram-read-bits=16`: the model's
// four 8-bit pipeline registers s1..s4 in two passes of two; s3's next value (s2 + 7) is complete
// in pass 0, before its row is read in pass 1: the hazard register, kept in a capture until its
// write), and a paged class left paged (SiR skipped, logged).
// RUN: rm -rf %t && mkdir -p %t && cat %S/Inputs/fold_model.mlir %s > %t/in.mlir
// RUN: firtool --format=mlir %t/in.mlir -o %t/target.sv
// RUN: mkdir -p %t/off && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top})" -o %t/off/sim.mlir 2> %t/off/fold.log
// RUN: not grep -e state-in-ram -e _sir %t/off/sim.mlir %t/off/fold.log
// RUN: mkdir -p %t/p1 && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model})" -o %t/p1/sim.mlir 2> %t/p1/fold.log
// RUN: FileCheck %s --check-prefix=P1 < %t/p1/sim.mlir
// RUN: FileCheck %s --check-prefix=P1LOG < %t/p1/fold.log
// RUN: firtool --format=mlir %t/p1/sim.mlir -o %t/p1/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/p1/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/p1/sim.sv > %t/p1/build.log 2>&1
// RUN: %t/p1/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/p1/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/ul && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model storage=Model:sir=ultra})" -o %t/ul/sim.mlir 2> %t/ul/fold.log
// RUN: FileCheck %s --check-prefix=UL < %t/ul/sim.mlir
// RUN: firtool --format=mlir %t/ul/sim.mlir -o %t/ul/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/ul/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/ul/sim.sv > %t/ul/build.log 2>&1
// RUN: %t/ul/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// RUN: mkdir -p %t/bl && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model storage=Model:sir=block})" -o %t/bl/sim.mlir 2> %t/bl/fold.log
// RUN: firtool --format=mlir %t/bl/sim.mlir -o %t/bl/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/bl/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/bl/sim.sv > %t/bl/build.log 2>&1
// RUN: %t/bl/obj/Vtb +verilator+rand+reset+2 +verilator+seed+4 | FileCheck %s
// RUN: mkdir -p %t/c2 && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top copies=Model:2 state-in-ram=Model})" -o %t/c2/sim.mlir 2> %t/c2/fold.log
// RUN: firtool --format=mlir %t/c2/sim.mlir -o %t/c2/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c2/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/c2/sim.sv > %t/c2/build.log 2>&1
// RUN: %t/c2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+5 | FileCheck %s
// RUN: mkdir -p %t/p2 && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model state-in-ram-read-bits=16})" -o %t/p2/sim.mlir 2> %t/p2/fold.log
// RUN: FileCheck %s --check-prefix=P2LOG < %t/p2/fold.log
// RUN: firtool --format=mlir %t/p2/sim.mlir -o %t/p2/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/p2/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/p2/sim.sv > %t/p2/build.log 2>&1
// RUN: %t/p2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+6 | FileCheck %s
// RUN: %t/p2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+7 | FileCheck %s
// RUN: mkdir -p %t/pg && foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model page=Model:2})" -o %t/pg/sim.mlir 2> %t/pg/fold.log
// RUN: FileCheck %s --check-prefix=PGLOG < %t/pg/fold.log
// RUN: firtool --format=mlir %t/pg/sim.mlir -o %t/pg/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/pg/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/pg/sim.sv > %t/pg/build.log 2>&1
// RUN: %t/pg/obj/Vtb +verilator+rand+reset+2 +verilator+seed+8 | FileCheck %s
// Errors: a SiR class with its pages off chip; a style that is not block or ultra; a class that is not there.
// RUN: not foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model storage=Model:sir=distributed})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=STYLE
// RUN: not foldsim-opt %t/in.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Nope})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=CLASS

// CHECK: {{^}}MATCH 4 outputs x 88 cycles

// P = 1: the registers packed into one RAM per column (names of DESIGN.md section 3), read at the
// instance two host cycles ahead into the RAM's output register and a read flop, the write staged
// a host cycle after the state pass at the staged instance, with byte masks by enable.
// P1: seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "{{.*}}">]} : <4 x {{[0-9]+}}, mask {{[0-9]+}}>
// P1: seq.firmem.read_port %region0_sir0[%ahead0_inst2]
// P1: seq.firmem.write_port %region0_sir0[%region0_inst_staged]
// P1LOG: state-in-ram: Model: {{[0-9]+}} bits/instance, 4 instances, {{[0-9]+}} x 72 columns, P 1, {{ultra|block}}
// UL: ram_style" = "\22ultra\22
// P2LOG: state-in-ram: Model: {{[0-9]+}} bits/instance, 4 instances, {{[0-9]+}} x 72 columns, P 2
// P2LOG: state-in-ram: Model: {{[0-9]+}} hazard registers, {{[0-9]+}} capture bits
// PGLOG: state-in-ram: Model: paged, skipped
// STYLE: storage: 'distributed' is not block or ultra for state-in-ram
// CLASS: state-in-ram: no model 'Nope'

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
