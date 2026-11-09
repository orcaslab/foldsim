// REQUIRES: verilator
// CSF pass-partitioned state in RAM (foldsim-fold-static `state-in-ram-pp`; scalable_state_arch_design_1109
// section 3.1, DESIGN_PP.md): a class of P > 1 staged with P stages, row j of each 72-bit column read in
// place at pass j (the RAM's output register feeds the logic, no fetch buffer) and written in the same pass
// from the logic through a P:1 row select (no write-back register); values that cross passes captured once
// at their pass; a group of registers that read one another kept in one row, its narrow registers moved to
// stores when it is too large for a row; a register whose next value needs a later pass than its row, or
// is a product (deep), stays in a store; a combinationally read memory banked by width and read a stage
// after its address, from an address, enable and instance register of its own per bank (the early
// address). The statically folded simulator against the target, channels stalled at random (exactness):
// P = 2, 3 (not a power of two) and 4 with balanced rows (FOLDSIM_PP_ROWS=balanced: several rows on a small
// model), the early address at fanout 1, ultra RAM, two copies (two instances each: the same instance comes
// back within the pipeline's depth of positions, covered by the state gap, no bypass needed), and the
// folded pipelines / black box / ready pass of fold_model.mlir.
// RUN: rm -rf %t && mkdir -p %t
// RUN: firtool --format=mlir %S/Inputs/pp_model.mlir -o %t/target.sv
// RUN: mkdir -p %t/p2 && env FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model::2 state-in-ram-pp})" -o %t/p2/sim.mlir 2> %t/p2/fold.log
// RUN: FileCheck %s --check-prefix=P2LOG < %t/p2/fold.log
// RUN: FileCheck %s --check-prefix=ROWS < %t/p2/sim.mlir
// RUN: firtool --format=mlir %t/p2/sim.mlir -o %t/p2/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/p2/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/p2/sim.sv > %t/p2/build.log 2>&1
// RUN: %t/p2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+1 | FileCheck %s
// RUN: %t/p2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+2 | FileCheck %s
// RUN: mkdir -p %t/p3 && env FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model::3 state-in-ram-pp})" -o %t/p3/sim.mlir 2> %t/p3/fold.log
// RUN: FileCheck %s --check-prefix=P3LOG < %t/p3/fold.log
// RUN: firtool --format=mlir %t/p3/sim.mlir -o %t/p3/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/p3/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/p3/sim.sv > %t/p3/build.log 2>&1
// RUN: %t/p3/obj/Vtb +verilator+rand+reset+2 +verilator+seed+3 | FileCheck %s
// RUN: mkdir -p %t/ea && env FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model::4 state-in-ram-pp state-in-ram-pp-fanout=1})" -o %t/ea/sim.mlir 2> %t/ea/fold.log
// RUN: FileCheck %s --check-prefix=EALOG < %t/ea/fold.log
// RUN: FileCheck %s --check-prefix=EA < %t/ea/sim.mlir
// RUN: firtool --format=mlir %t/ea/sim.mlir -o %t/ea/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/ea/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/ea/sim.sv > %t/ea/build.log 2>&1
// RUN: %t/ea/obj/Vtb +verilator+rand+reset+2 +verilator+seed+4 | FileCheck %s
// RUN: %t/ea/obj/Vtb +verilator+rand+reset+2 +verilator+seed+5 | FileCheck %s
// RUN: mkdir -p %t/ul && env FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model::4 state-in-ram-pp storage=Model:sir=ultra})" -o %t/ul/sim.mlir 2> %t/ul/fold.log
// RUN: FileCheck %s --check-prefix=UL < %t/ul/sim.mlir
// RUN: firtool --format=mlir %t/ul/sim.mlir -o %t/ul/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/ul/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/ul/sim.sv > %t/ul/build.log 2>&1
// RUN: %t/ul/obj/Vtb +verilator+rand+reset+2 +verilator+seed+6 | FileCheck %s
// RUN: mkdir -p %t/c2 && env FOLDSIM_PP_ROWS=balanced foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top copies=Model:2 state-in-ram=Model::4 state-in-ram-pp})" -o %t/c2/sim.mlir 2> %t/c2/fold.log
// RUN: FileCheck %s --check-prefix=C2LOG < %t/c2/fold.log
// RUN: firtool --format=mlir %t/c2/sim.mlir -o %t/c2/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/c2/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/target.sv %t/c2/sim.sv > %t/c2/build.log 2>&1
// RUN: %t/c2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+7 | FileCheck %s
// RUN: %t/c2/obj/Vtb +verilator+rand+reset+2 +verilator+seed+8 | FileCheck %s
// Folded pipelines, a black box and a ready pass (fold_model.mlir with static-state-in-ram.mlir's Top).
// RUN: cat %S/Inputs/fold_model.mlir > %t/fm.mlir && sed -n '/^hw.module @Top/,$p' %S/static-state-in-ram.mlir >> %t/fm.mlir
// RUN: firtool --format=mlir %t/fm.mlir -o %t/fm_target.sv
// RUN: mkdir -p %t/fm && env FOLDSIM_PP_ROWS=balanced foldsim-opt %t/fm.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model::3 state-in-ram-pp})" -o %t/fm/sim.mlir 2> %t/fm/fold.log
// RUN: FileCheck %s --check-prefix=FMLOG < %t/fm/fold.log
// RUN: firtool --format=mlir %t/fm/sim.mlir -o %t/fm/sim.sv
// RUN: %verilator --binary --timing -Wno-fatal -Wno-lint --x-initial unique --top-module tb -Mdir %t/fm/obj %S/Inputs/tb_fold.sv %S/Inputs/xbox.sv %t/fm_target.sv %t/fm/sim.sv > %t/fm/build.log 2>&1
// RUN: %t/fm/obj/Vtb +verilator+rand+reset+2 +verilator+seed+9 | FileCheck %s
// RUN: %t/fm/obj/Vtb +verilator+rand+reset+2 +verilator+seed+10 | FileCheck %s
// Errors: not with pages.
// RUN: not foldsim-opt %S/Inputs/pp_model.mlir --pass-pipeline="builtin.module(foldsim-extract-models{module=Top models=Model netlist},foldsim-fold-static{module=Top state-in-ram=Model::2 state-in-ram-pp storage=Model:sir-rows=hbm})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=ROWSERR

// CHECK: {{^}}MATCH 4 outputs x 88 cycles

// Read and write stage placement: the class staged with P stages; each column a RAM primitive of a row
// per {instance, pass}, read at the pass's instance and row two host cycles ahead (from the ports three
// ahead) into its output register, which the logic reads; written now, at the pass's instance and row, from
// the row select (a column's own select register), enabled by the pass; the captures enabled by their pass.
// P2LOG: state-in-ram: Model: PP: 2 passes, rows read in place and written in pass, staged with 2 stages
// P2LOG: Model x4 {{.*}}stages 2
// P2LOG: state-in-ram-pp: Model: P 2, {{[0-9]+}} registers, {{[0-9]+}} bits in 2 rows x {{[0-9]+}} columns of 72 (block, 9-bit lanes); rows {{[1-9][0-9]*}} {{[1-9][0-9]*}};
// P2LOG-SAME: deep registers ({{[0-9]+}} bits) stored
// P2LOG-SAME: capture bits
// P2LOG: state-in-ram-pp: Model: built {{[0-9]+}} registers in {{[0-9]+}} columns x 8 rows (block)
// ROWS: hw.module @Model_static(in %hostClock : !seq.clock, in %state3 : i1, in %inst3 : i2, in %rinst3 : i2, in %live3 : i1, in %ppvalid3 : i1, in %ppinst3 : i2, in %pprow3 : i1
// ROWS: %pp_pass0_en0 = seq.compreg.ce
// ROWS: %pp_cap0 = seq.compreg.ce {{.*}}, %hostClock, %pp_pass0_en
// ROWS: %pp_g0_rrow2 = seq.firreg %pprow3
// ROWS: %pp_g0_rinst2 = seq.firreg %ppinst3
// ROWS: %[[RA:.+]] = comb.concat %pp_g0_rinst2, %pp_g0_rrow2
// ROWS: %pp_g0_wvalid0 = seq.firreg %pp_g0_wvalid1
// ROWS: %pp_g0_wrow0 = seq.firreg %pp_g0_wrow1
// ROWS: %pp_g0_winst0 = seq.firreg %pp_g0_winst1
// ROWS: %[[WA:.+]] = comb.concat %pp_g0_winst0, %pp_g0_wrow0
// ROWS: %pp_col0 = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <8 x 72, mask 8>
// ROWS: seq.firmem.read_port %pp_col0[%[[RA]]]
// ROWS: %pp_col0_out = seq.firreg
// ROWS: %pp_col0_sel = seq.firreg %pp_g0_wrow1
// ROWS: seq.firmem.write_port %pp_col0[%[[WA]]] = {{.*}} enable %pp_g0_wvalid0 mask
// A multi-pass row set: P = 3.
// P3LOG: state-in-ram-pp: Model: P 3, {{[0-9]+}} registers, {{[0-9]+}} bits in 3 rows x {{[0-9]+}} columns of 72 (block, 9-bit lanes); rows {{[1-9][0-9]*}} {{[1-9][0-9]*}}
// The early address: the memory banked, a bank's address registered at its pass (the stage before its
// read) and kept, its instance a stage register's copy.
// EALOG: state-in-ram-pp: Model: P 4, {{.*}} 1 banked reads, 16 bank address copies
// EA: %pp_ram_b0_addr = seq.compreg.ce {{.*}}, %hostClock, %pp_pass{{[0-9]}}_en{{[0-9]+}} {sv.attributes = [#sv.attribute<"keep" = "\22true\22">]}
// EA: %pp_ram_b0_inst = seq.compreg.ce {{.*}} {sv.attributes = [#sv.attribute<"keep" = "\22true\22">]}
// EA: comb.concat %pp_ram_b0_addr, %pp_ram_b0_inst
// UL: %pp_col0 = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <16 x 72, mask 9>
// C2LOG: Model x2 state gap {{[0-9]+}} stages 4
// FMLOG: Model x4 (ready pass) state gap {{[0-9]+}} stages 3
// FMLOG: state-in-ram-pp: Model: P 3,
// FMLOG-SAME: stage register bits (ready cone, {{[1-9][0-9]*}} ops)
// ROWSERR: state-in-ram-pp: Model: sir-rows is the paged form's
