// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=versal},foldsim-estimate-timing{output=- report= top=Top target=versal})' -o /dev/null | FileCheck %s

// A register bit that addresses a memory drives a pin of every LUT of it, a
// fanout that the route, not the logic levels, makes slow (Beacon's fanout
// term; foldsim/MemoryLoads.h, the model foldsim-replicate-controls bounds):
// the index of a 256 x 64 memory read asynchronously (distributed RAM however
// deep) drives 64 x 8 = 512 pins with its write address and 512 with its read
// address; its write-address end point and the end point after the read
// report it as the path's fanout. The same memory read through a register is
// block RAM: 2 pins a port.
// CHECK: levels,retimed,kind,start_kind,fanout,crossings,start_inst,start,end_inst,end,in_fanout
// CHECK-DAG: {{^}}0,0,wa,reg,1024,0,0,idx[0],0,mem.wa[0],0
// CHECK-DAG: {{^}}3,3,d,reg,1024,0,0,idx[0],0,q.d[0],0
// CHECK-DAG: {{^}}0,0,ra,reg,4,0,0,sidx[0],0,smem.ra[0],0
hw.module @Top(in %clk : !seq.clock, in %i : i8, in %si : i8, in %d : i64, out q : i64, out sq : i64) {
  %true = hw.constant true
  %idx = seq.firreg %i clock %clk : i8
  %mem = seq.firmem 0, 1, undefined, undefined : <256 x 64>
  seq.firmem.write_port %mem[%idx] = %d, clock %clk enable %true : <256 x 64>
  %rd = seq.firmem.read_port %mem[%idx], clock %clk : <256 x 64>
  %q = seq.firreg %rd clock %clk : i64
  %sidx = seq.firreg %si clock %clk : i8
  %smem = seq.firmem 1, 1, undefined, undefined : <256 x 64>
  seq.firmem.write_port %smem[%sidx] = %d, clock %clk enable %true : <256 x 64>
  %srd = seq.firmem.read_port %smem[%sidx], clock %clk enable %true : <256 x 64>
  %sq = seq.firreg %srd clock %clk : i64
  hw.output %q, %sq : i64, i64
}
