// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-replicate-controls{loads=16}))' | FileCheck %s

// Synthesis builds a memory read asynchronously (read latency 0) of LUTs
// however deep, so its address pins are its width times its depth in 32-entry
// LUTs: the 128-entry, 12-bit async memory's port is 48 loads a bit, over 16,
// and is split into three parts of 4 bits, each port with a replica of the
// control (the b4e 200 MHz miss: a 256 x 2048 async memory counted as block
// RAM, 57 loads, left its instance index on one net of 8,192 LUT pins).
// The same memory read a cycle later (latency 1) is block RAM: one pin per 36
// bits and 1024 entries, under 16, so neither it nor its control changes.
// A `ram_style = "block"` memory is block RAM whatever its latency.

// CHECK-LABEL: hw.module @Async
// CHECK: %idx = seq.firreg %inst2 clock %clk {sv.attributes = [#sv.attribute<"DONT_TOUCH" = "\22true\22">]} : i1
// CHECK: %idx_rep5 = seq.firreg
// CHECK-NOT: %idx_rep6
// CHECK: %amem_p0 = seq.firmem 0, 1, undefined, undefined : <128 x 4>
// CHECK: %amem_p1 = seq.firmem 0, 1, undefined, undefined : <128 x 4>
// CHECK: %amem_p2 = seq.firmem 0, 1, undefined, undefined : <128 x 4>
// CHECK-NOT: %amem_p3
hw.module @Async(in %clk: !seq.clock, in %inst2: i1, in %row: i6, in %d: i12,
                 out r: i12) {
  %true = hw.constant true
  %idx = seq.firreg %inst2 clock %clk {foldsim.control} : i1
  %amem = seq.firmem 0, 1, undefined, undefined : <128 x 12>
  %addr = comb.concat %row, %idx : i6, i1
  seq.firmem.write_port %amem[%addr] = %d, clock %clk enable %true : <128 x 12>
  %rd = seq.firmem.read_port %amem[%addr], clock %clk : <128 x 12>
  hw.output %rd : i12
}

// CHECK-LABEL: hw.module @Sync
// CHECK-NOT: DONT_TOUCH
// CHECK-NOT: _rep
// CHECK: %smem = seq.firmem 1, 1, undefined, undefined : <128 x 12>
hw.module @Sync(in %clk: !seq.clock, in %inst2: i1, in %row: i6, in %d: i12,
                out r: i12) {
  %true = hw.constant true
  %idx = seq.firreg %inst2 clock %clk {foldsim.control} : i1
  %smem = seq.firmem 1, 1, undefined, undefined : <128 x 12>
  %addr = comb.concat %row, %idx : i6, i1
  seq.firmem.write_port %smem[%addr] = %d, clock %clk enable %true : <128 x 12>
  %rd = seq.firmem.read_port %smem[%addr], clock %clk : <128 x 12>
  hw.output %rd : i12
}

// CHECK-LABEL: hw.module @Styled
// CHECK-NOT: DONT_TOUCH
// CHECK-NOT: _rep
// CHECK: %bmem = seq.firmem 0, 1, undefined, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <128 x 12>
hw.module @Styled(in %clk: !seq.clock, in %inst2: i1, in %row: i6, in %d: i12,
                  out r: i12) {
  %true = hw.constant true
  %idx = seq.firreg %inst2 clock %clk {foldsim.control} : i1
  %bmem = seq.firmem 0, 1, undefined, undefined {sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <128 x 12>
  %addr = comb.concat %row, %idx : i6, i1
  seq.firmem.write_port %bmem[%addr] = %d, clock %clk enable %true : <128 x 12>
  %rd = seq.firmem.read_port %bmem[%addr], clock %clk : <128 x 12>
  hw.output %rd : i12
}
