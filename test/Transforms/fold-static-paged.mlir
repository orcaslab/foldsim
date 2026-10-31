// Paged register state (foldsim-fold-static `page`): four leaves on one copy
// keep their registers (100 bits) in one page RAM of 4 instances x 2 pages of
// 50 bits, read a page a host cycle into a fetch buffer the logic reads, and
// written back from a write-back register a page a host cycle; the state
// passes are a page apart. The 80-bit register is cut where its pages meet.
// The page word is cut into columns of whole 72-bit ports, each a RAM with
// its own controls (a region): at 72 region bits and p = 1, 72 + 28 bits.
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top})' 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page-memory-bits=0 page=Leaf:2})' 2>&1 | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page-memory-bits=0 page=*:3 storage=Leaf:pages=ultra})' 2>&1 | FileCheck %s --check-prefix=ULTRA
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page-memory-bits=0 page=Leaf:4 modulo})' 2>&1 | FileCheck %s --check-prefix=MOD
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page-memory-bits=0 page=Leaf:1 region-bits=72})' 2>&1 | FileCheck %s --check-prefix=COLS
// The leaves' memory (16 rows of 8 bits, read through a register) pages as
// rows with the registers by default (page-memory-bits): a register a row,
// the read a choice of the rows into the read register; the runs above page
// registers only (page-memory-bits=0).
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page=Leaf:2})' 2>&1 | FileCheck %s --check-prefix=ROWS
// ROWS: pages Leaf: p 2, 21 registers, 236 bits (with 1 memories' rows, 128 bits) (251 in 9-bit bytes, 16 enables); page RAMs 8 rows x 126 bits
// ROWS-NOT: seq.firmem {{.*}} <64 x 8
// ROWS-DAG: %mem_row15_fetched = seq.firreg
// ROWS-DAG: %mem_rdata_fetched = seq.firreg
// ROWS-DAG: hw.array_get {{%.+}}[{{%.+}}] : !hw.array<16xi8>, i4
// page-memories limits the rows to the memories it finds by name: the leaves'
// memory, `mem`, is found by `^me` and not by `^x`, which pages registers only.
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page=Leaf:2 page-memories=^me})' 2>&1 | FileCheck %s --check-prefix=ROWS
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page=Leaf:2 page-memories=^x})' 2>&1 | FileCheck %s --check-prefix=NOROWS
// NOROWS: pages Leaf: p 2, 4 registers, 100 bits; page RAMs 8 rows x 50 bits
// NOROWS: %mem = seq.firmem 1, 1, old, undefined : <64 x 8, mask 1>
// Registers a ready pass reads keep their stores (the leaves' output is the
// memory's read data, through two registers: a ready pass).
// RUN: foldsim-opt %S/Inputs/fold-static-storage-late.mlir --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page=Leaf:2})' 2>&1 | FileCheck %s --check-prefix=READY
// Off chip (storage <model>:pages=hbm): the page word's columns (here at
// most 32 bits) are virtual memories of {instance, page} rows; each
// position's page read and write come from the tables, and the copy's
// virtual memory ports follow its own. A row is a power of two of bits
// (the last column's 24 padded to 32, its padding's mask bytes off).
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page-memory-bits=0 page=Leaf:2 storage=Leaf:pages=hbm page-hbm-column-bits=32})' 2>&1 | FileCheck %s --check-prefix=HBM
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top storage=Leaf:regs=hbm})' 2>&1 | FileCheck %s --check-prefix=HBMREGS
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page=Nope:2})' 2>&1 | FileCheck %s --check-prefix=MODEL
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top page=Leaf:0})' 2>&1 | FileCheck %s --check-prefix=SPEC
// OFF: II = 6
// OFF-NOT: fetch3
// OFF-NOT: _pages
// CHECK: II = 8
// CHECK: Leaf x4 state gap 8 page 2
// CHECK-NEXT: state: 0 2 4 6
// CHECK: pages Leaf: p 2, 4 registers, 100 bits; page RAMs 8 rows x 50 bits in 1 columns, 1 ports, block; kept 0 registers, 0 bits; state passes 4 x 2 = 8, ports bound 8
// CHECK: hw.module @Leaf_static(in %hostClock : !seq.clock, in %state3 : i1, in %inst3 : i2, in %fetch3 : i1, in %finst3 : i2, in %d : i8, out q : i8)
// The page read address: the fetched instance's page 0 two host cycles on.
// CHECK: %ahead0_fetch2 = seq.firreg %fetch3
// CHECK: %ahead0_page_inst = seq.firreg [[PI:%.+]] clock
// CHECK: [[PI]] = comb.mux %fetch3, %finst3, %ahead0_page_inst : i2
// CHECK: %ahead0_page = seq.firreg [[PG:%.+]] clock
// CHECK: [[PG]] = comb.mux %fetch3, %false
// CHECK: [[RA:%.+]] = comb.concat %ahead0_page_inst, %ahead0_page : i2, i1
// The write-back registers, loaded at the state pass.
// CHECK: %narrow_writeback = seq.firreg [[NW:%.+]] clock %hostClock : i8
// CHECK: [[NW]] = comb.mux %region0_state, {{%.+}}, %narrow_writeback : i8
// CHECK: %wide_writeback = seq.firreg {{%.+}} clock %hostClock : i80
// CHECK: %region1_pages = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <8 x 50, mask 1>
// CHECK: [[RD:%.+]] = seq.firmem.read_port %region1_pages[[[RA]]]
// CHECK: %region1_page_out = seq.firreg [[RD]] clock %hostClock : i50
// CHECK: %region1_land0 = seq.firreg %region1_fetch1
// CHECK: %region1_land1 = seq.firreg %region1_land0
// Write-back: page j a host cycle + j after the state pass.
// CHECK: %region1_write0 = seq.firreg %region1_state
// CHECK: %region1_write1 = seq.firreg %region1_write0
// CHECK: %region1_write_inst = seq.firreg [[WI:%.+]] clock
// CHECK: [[WI]] = comb.mux %region1_state, %region1_inst, %region1_write_inst : i2
// CHECK: [[WA:%.+]] = comb.concat %region1_write_inst, %region1_write_page : i2, i1
// The fetch buffer, a part of a register per page: page 0's bits load at
// land0, page 1's at land1.
// CHECK: %narrow_fetched = seq.firreg [[NF:%.+]] clock %hostClock : i8
// CHECK: [[NFB:%.+]] = comb.extract %region1_page_out from 8 : (i50) -> i8
// CHECK: [[NF]] = comb.mux %region1_land0, [[NFB]], %narrow_fetched : i8
// CHECK: %wide_0_fetched = seq.firreg {{%.+}} clock %hostClock : i30
// CHECK: %wide_1_fetched = seq.firreg [[W1:%.+]] clock %hostClock : i50
// CHECK: [[W1]] = comb.mux %region1_land1, %region1_page_out, %wide_1_fetched : i50
// CHECK: [[P0:%.+]] = comb.concat {{%.+}}, %waddr_writeback, %narrow_writeback, %held_writeback : i30, i4, i8, i8
// CHECK: [[WE:%.+]] = comb.or %region1_write0, %region1_write1 : i1
// CHECK: seq.firmem.write_port %region1_pages[[[WA]]] = {{%.+}}, clock %hostClock enable [[WE]] : <8 x 50, mask 1>
// The simulator fetches each instance a page before its state pass: at
// positions 6 (frame before), 0, 2 and 4.
// CHECK: hw.module @foldsim_sim
// CHECK: hw.instance "Leaf_copy" @Leaf_static(hostClock: {{%.+}}: !seq.clock, state3: {{%.+}}: i1, inst3: {{%.+}}: i2, fetch3: {{%.+}}: i1, finst3: {{%.+}}: i2,
// ULTRA: pages Leaf: p 3, 4 registers, 100 bits; page RAMs 16 rows x 34 bits in 1 columns, 1 ports, ultra
// ULTRA: %region1_pages = seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <16 x 34, mask 1>
// MOD: II = 16
// MOD: state passes 4 x 4 = 16
// READY: Leaf x4 (ready pass) state gap 4 page 2
// READY: pages Leaf: p 2, 2 registers, 12 bits; page RAMs 8 rows x 6 bits in 1 columns, 1 ports, block; kept 2 registers, 88 bits
// COLS: pages Leaf: p 1, 4 registers, 100 bits; page RAMs 4 rows x 100 bits in 2 columns, 2 ports, block
// COLS: %region4_pages = seq.firmem {{.*}} : <4 x 72, mask 1>
// COLS: %region5_pages = seq.firmem {{.*}} : <4 x 28, mask 1>
// COLS: %wide_0_fetched = seq.firreg {{%.+}} clock %hostClock : i52
// COLS: %wide_1_fetched = seq.firreg {{%.+}} clock %hostClock : i28
// COLS: seq.firmem.write_port %region4_pages[%region4_write_inst] = {{%.+}}, clock %hostClock enable %region4_write0 : <4 x 72, mask 1>
// COLS: seq.firmem.write_port %region5_pages[%region5_write_inst] = {{%.+}}, clock %hostClock enable %region5_write0 : <4 x 28, mask 1>
// HBM: Leaf x4 state gap 8 page 2 off chip
// HBM: pages Leaf: p 2, 4 registers, 100 bits; off chip: 2 virtual memories of 8 rows x 56 bits (vm0-vm1), lead 2
// HBM: hw.module @Leaf_static(in %hostClock : !seq.clock, in %state3 : i1, in %inst3 : i2, in %pread3 : i1, in %pinst3 : i2, in %ppage3 : i1, in %pwrite3 : i1, in %pwpage3 : i1, in %d : i8, in %hostReset : i1, in %vm0_rsp_valid : i1, in %vm0_rsp_inst : i3, in %vm0_rsp_data : i32, in %vm1_rsp_valid : i1, in %vm1_rsp_inst : i3, in %vm1_rsp_data : i32, out q : i8, out vm0_rd_valid : i1, out vm0_rd_inst : i3, out vm0_rd_row : i1, out vm0_wr_valid : i1, out vm0_wr_inst : i3, out vm0_wr_row : i1, out vm0_wr_data : i32, out vm0_wr_mask : i4, out vm0_pending : i4, out vm1_rd_valid : i1, out vm1_rd_inst : i3, out vm1_rd_row : i1, out vm1_wr_valid : i1, out vm1_wr_inst : i3, out vm1_wr_row : i1, out vm1_wr_data : i32, out vm1_wr_mask : i4, out vm1_pending : i4)
// HBM: %region1_fetching = seq.firreg
// HBM: %region1_pages_pending = seq.firreg
// HBM: %region1_writeback_valid = seq.firreg
// HBM: foldsim.vmems = ["region1_pages:Leaf:3:1:32:4:1:8", "region2_pages:Leaf:3:1:32:4:1:8"]
// HBMREGS: storage: 'hbm' is for a model's pages, not 'regs'
// MODEL: page: no model 'Nope'
// SPEC: page: 'Leaf:0' is not <model>:<p>, p at least 1
module {
  hw.module @Leaf(in %clk : !seq.clock, in %d : i8, out q : i8) {
    %c-3_i4 = hw.constant -3 : i4
    %c1_i4 = hw.constant 1 : i4
    %c-1_i4 = hw.constant -1 : i4
    %mem = seq.firmem name "mem" 1, 1, old, undefined : <16 x 8, mask 1>
    %0 = comb.extract %wide from 0 : (i80) -> i72
    %1 = comb.extract %wide from 72 : (i80) -> i8
    %2 = comb.xor %d, %1 : i8
    %3 = comb.concat %0, %2 : i72, i8
    %wide = seq.firreg %3 clock %clk : i80
    %4 = comb.extract %wide from 40 : (i80) -> i8
    %5 = comb.extract %wide from 0 : (i80) -> i8
    %6 = comb.xor %d, %5 : i8
    %7 = comb.add %narrow, %d : i8
    %narrow = seq.firreg %7 clock %clk : i8
    %8 = comb.sub %c-1_i4, %waddr : i4
    seq.firmem.write_port %mem[%8] = %6, clock %clk : <16 x 8, mask 1>
    %9 = comb.add %waddr, %c1_i4 : i4
    %waddr = seq.firreg %9 clock %clk : i4
    %10 = comb.add %waddr, %c-3_i4 : i4
    %11 = comb.sub %c-1_i4, %10 : i4
    %12 = seq.firmem.read_port %mem[%11], clock %clk : <16 x 8, mask 1>
    %held = seq.firreg %12 clock %clk : i8
    %13 = comb.xor %held, %narrow, %4 : i8
    hw.output %13 : i8
  }
  hw.module private @Top_model(in %clk : !seq.clock, in %x : i8, in %l3_q : i8, out y : i8) {
    hw.output %l3_q : i8
  }
  hw.module @Top(in %clk : !seq.clock, in %x : i8, out y : i8) attributes {foldsim.netlist} {
    %l0.q = hw.instance "l0" @Leaf(clk: %clk: !seq.clock, d: %x: i8) -> (q: i8)
    %l1.q = hw.instance "l1" @Leaf(clk: %clk: !seq.clock, d: %l0.q: i8) -> (q: i8)
    %l2.q = hw.instance "l2" @Leaf(clk: %clk: !seq.clock, d: %l1.q: i8) -> (q: i8)
    %l3.q = hw.instance "l3" @Leaf(clk: %clk: !seq.clock, d: %l2.q: i8) -> (q: i8)
    %logic.y = hw.instance "logic" @Top_model(clk: %clk: !seq.clock, x: %x: i8, l3_q: %l3.q: i8) -> (y: i8)
    hw.output %logic.y : i8
  }
}

