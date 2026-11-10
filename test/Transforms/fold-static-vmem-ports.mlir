// A virtual memory's ports are the values its read and write take: here the
// registers `raddr`, `waddr` and `wdata`, which come after the memory in
// state (name) order, so they are replaced and erased after its ports are
// built. The copy's outputs must carry their replacements (the register's
// read from its store, or with pages its fetch), never the erased register:
// before the fix the ports were kept aside as bare values and the outputs
// named erased ops (a crash, or heap corruption on the 64-node datacenter).
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top virtual-memories=mem})' | FileCheck %s
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top virtual-memories=mem page=Leaf:2 page-memory-bits=0})' | FileCheck %s --check-prefix=PAGED
// The readiness a frame's position waits on (each virtual memory's instance at
// the next position has its read back) reads its slot tables from registers
// loaded a host cycle before (lookahead-tables, default on), not through the
// position's increment and the tables' decoders in the decision's cycle.
// Without an advance the position stays, and so does the register (its own
// value, not a second table of the position).
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top virtual-memories=mem})' | FileCheck %s --check-prefix=AHEAD
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-fold-static{module=Top virtual-memories=mem lookahead-tables=false})' | FileCheck %s --check-prefix=NOAHEAD
// AHEAD-LABEL: hw.module @foldsim_sim(
// AHEAD-DAG: %vready_ahead0 = seq.firreg {{%[^ ]+}} clock %hostClock reset sync %hostReset, {{%[^ ]+}} preset 1 : i1
// AHEAD-DAG: %vready_ahead1 = seq.firreg {{%[^ ]+}} clock %hostClock reset sync %hostReset, {{%[^ ]+}} preset 0 : i2
// AHEAD-DAG: = comb.mux {{%[^,]+}}, {{%[^,]+}}, %vready_ahead0 : i1
// AHEAD-DAG: = comb.mux {{%[^,]+}}, {{%[^,]+}}, %vready_ahead1 : i2
// NOAHEAD-LABEL: hw.module @foldsim_sim(
// NOAHEAD-NOT: vready_ahead
// CHECK-LABEL: hw.module @Leaf_static(
// CHECK-SAME: out vm0_rd_row : i4
// CHECK-SAME: out vm0_wr_row : i4, out vm0_wr_data : i8
// CHECK-DAG: %raddr_read = seq.firreg
// CHECK-DAG: %waddr_read = seq.firreg
// CHECK-DAG: %wdata_read = seq.firreg
// CHECK: hw.output {{%[^,]+}}, {{%[^,]+}}, {{%[^,]+}}, %raddr_read, {{%[^,]+}}, {{%[^,]+}}, %waddr_read, %wdata_read, {{%[^,]+}}, %mem_pending :
// PAGED-LABEL: hw.module @Leaf_static(
// PAGED-DAG: %raddr_fetched = seq.firreg
// PAGED-DAG: %waddr_fetched = seq.firreg
// PAGED-DAG: %wdata_fetched = seq.firreg
// PAGED: hw.output {{%[^,]+}}, {{%[^,]+}}, {{%[^,]+}}, %raddr_fetched, {{%[^,]+}}, {{%[^,]+}}, %waddr_fetched, %wdata_fetched, {{%[^,]+}}, %mem_pending :

module {
  hw.module @Leaf(in %clk : !seq.clock, in %d : i8, out q : i8) {
    %c1_i4 = hw.constant 1 : i4
    %true = hw.constant true
    %mem = seq.firmem name "mem" 1, 1, old, undefined : <16 x 8, mask 1>
    %0 = comb.extract %d from 0 : (i8) -> i4
    %1 = comb.extract %d from 4 : (i8) -> i4
    %2 = comb.extract %d from 0 : (i8) -> i1
    %raddr = seq.firreg %0 clock %clk : i4
    %waddr = seq.firreg %1 clock %clk : i4
    %wdata = seq.firreg %d clock %clk : i8
    seq.firmem.write_port %mem[%waddr] = %wdata, clock %clk enable %2 : <16 x 8, mask 1>
    %3 = seq.firmem.read_port %mem[%raddr], clock %clk enable %true : <16 x 8, mask 1>
    %4 = comb.xor %3, %d : i8
    hw.output %4 : i8
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
