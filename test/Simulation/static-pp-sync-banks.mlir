// REQUIRES: verilator
// pp-sync-banks=R (state-in-ram-pp): a memory of more than R threaded rows read combinationally and banked
// past state-in-ram-pp-fanout (each bank's address, enable and instance clock-enabled registers loaded
// at the stage before the read) is threaded as a registered read of those registers' next values
// (the register's input where it loads, else its own value): its data a host cycle later is what
// the combinational read of the registers gave, so the threaded store's multiplexers (entries x
// instances rows of LUT RAM, Beacon 1 + log2(rows) - 6 levels) end in the RAM's register, or a
// block RAM, instead of in the logic after the read. Exact against the target with channels
// stalled at random (Inputs/pp_model.mlir at fanout 1: 16 banks), one and two copies (two
// instances each, back within the pipeline's depth), two seeds each. Reading the registers
// themselves a host cycle late (a mutation) mismatches at cycle 9. A read is registered only where at
// least 16 estimated levels of logic follow it (the sector tables of the real caches: 51-113; a read
// whose data is registered at once would only move LUT RAM into block RAM); this model's are shallow:
// FOLDSIM_SYNC_AFTER_LEVELS=0 (a test hook) registers them, and without it none is (SNONE).
// RUN: mkdir -p %t/n && env FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model::4 state-in-ram-pp state-in-ram-pp-fanout=1 pp-sync-banks=16})" -o %t/n/sim.mlir 2> %t/n/fold.log
// RUN: FileCheck %s --check-prefix=SNONE < %t/n/fold.log
// RUN: rm -rf %t && mkdir -p %t
// RUN: firtool --format=mlir %S/Inputs/pp_model.mlir -o %t/target.sv
// RUN: mkdir -p %t/a && env FOLDSIM_SYNC_AFTER_LEVELS=0 FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model::4 state-in-ram-pp state-in-ram-pp-fanout=1 pp-sync-banks=16})" -o %t/a/sim.mlir 2> %t/a/fold.log
// RUN: FileCheck %s --check-prefix=LOG < %t/a/fold.log
// RUN: FileCheck %s --check-prefix=SIM < %t/a/sim.mlir
// RUN: firtool --format=mlir %t/a/sim.mlir -o %t/a/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/a/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/a/sim.sv > %t/a/build.log 2>&1
// RUN: %t/a/obj/Vtb +verilator+rand+reset+2 +verilator+seed+4 | FileCheck %s
// RUN: %t/a/obj/Vtb +verilator+rand+reset+2 +verilator+seed+5 | FileCheck %s
// RUN: mkdir -p %t/c2 && env FOLDSIM_SYNC_AFTER_LEVELS=0 FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top copies=Model:2 state-in-ram=Model::4 state-in-ram-pp state-in-ram-pp-fanout=1 pp-sync-banks=8})" -o %t/c2/sim.mlir 2> %t/c2/fold.log
// RUN: FileCheck %s --check-prefix=LOG < %t/c2/fold.log
// RUN: firtool --format=mlir %t/c2/sim.mlir -o %t/c2/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c2/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/c2/sim.sv > %t/c2/build.log 2>&1
// RUN: %t/c2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+7 | FileCheck %s
// RUN: %t/c2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+8 | FileCheck %s
// Only a read past the bound is registered (pp-sync-banks=R: 16 and 8 above); at 1,024 these shallow banks
// stay combinational (registered, a shallow bank would only move its LUT RAM into block RAM).
// RUN: mkdir -p %t/s && env FOLDSIM_SYNC_AFTER_LEVELS=0 FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model::4 state-in-ram-pp state-in-ram-pp-fanout=1 pp-sync-banks=1024})" -o %t/s/sim.mlir 2> %t/s/fold.log
// RUN: FileCheck %s --check-prefix=SLOG < %t/s/fold.log
// A memory banked for its threaded rows alone (past the bound, 8 here), at the default fanout: one bank,
// registered; exact.
// RUN: mkdir -p %t/d && env FOLDSIM_SYNC_AFTER_LEVELS=0 FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top copies=Model:2 state-in-ram=Model::4 state-in-ram-pp pp-sync-banks=8})" -o %t/d/sim.mlir 2> %t/d/fold.log
// RUN: FileCheck %s --check-prefix=DLOG < %t/d/fold.log
// RUN: firtool --format=mlir %t/d/sim.mlir -o %t/d/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/d/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/d/sim.sv > %t/d/build.log 2>&1
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+9 | FileCheck %s
// RUN: %t/d/obj/Vtb +verilator+rand+reset+2 +verilator+seed+10 | FileCheck %s

// CHECK: {{^}}MATCH 4 outputs x 88 cycles
// LOG: state-in-ram-pp: Model: P 4, {{.*}} 1 banked reads, 16 bank address copies (16 registered reads, pp-sync-banks)
// SNONE: state-in-ram-pp: Model: P 4, {{.*}} (0 registered reads, pp-sync-banks)
// SLOG: state-in-ram-pp: Model: P 4, {{.*}} 1 banked reads, 16 bank address copies (0 registered reads, pp-sync-banks)
// DLOG: state-in-ram-pp: Model: P 4, {{.*}} 1 banked reads, 1 bank address copies (1 registered reads, pp-sync-banks)
// SIM: %ram = seq.firmem 1, 1, old, undefined : <32 x 1>
// SIM: seq.firmem.read_port %ram{{(_[0-9]+)?}}[{{.*}}], clock %hostClock
