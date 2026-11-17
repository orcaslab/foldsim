// RUN: foldsim-opt %s --pass-pipeline='builtin.module(hw.module(foldsim-replicate-controls{loads=16}))' | FileCheck %s

// Synthesis builds a memory read asynchronously (read latency 0) of LUTs
// however deep, so its address pins are its width times its depth in 32-entry
// LUTs: the 128-entry, 12-bit async memory's port is 48 loads a bit, over 16
// (the b4e 200 MHz miss: a 256 x 2048 async memory counted as block RAM, 57
// loads, left its instance index on one net of 8,192 LUT pins).
// Such a memory is split by depth first, into parts of whole 64-entry LUT RAM
// banks, the high address bits selecting the read part and enabling the
// written one; then each part by bits (24 loads a bit, over 16: two parts of
// 6 bits). Split by bits alone, every part kept both banks, each bank with a
// write enable of its own: parts x banks control sets of few LUTs (the FSD
// 1,024 route failures, 11-17). Each part's port takes a replica.
// The same memory read a cycle later (latency 1) is block RAM: one pin per 36
// bits and 1024 entries, under 16, so neither it nor its control changes.
// A `ram_style = "block"` memory is block RAM whatever its latency.

// CHECK-LABEL: hw.module @Async
// CHECK: %idx = seq.firreg %inst2 clock %clk {sv.attributes = [#sv.attribute<"DONT_TOUCH" = "\22true\22">]} : i1
// CHECK: %idx_rep8 = seq.firreg
// CHECK-NOT: %idx_rep9
// CHECK: %amem_b0_p0 = seq.firmem 0, 1, undefined, undefined : <64 x 6>
// CHECK: %amem_b0_p1 = seq.firmem 0, 1, undefined, undefined : <64 x 6>
// CHECK: %amem_b1_p0 = seq.firmem 0, 1, undefined, undefined : <64 x 6>
// CHECK: %amem_b1_p1 = seq.firmem 0, 1, undefined, undefined : <64 x 6>
// CHECK-NOT: %amem_b2
// CHECK: %[[HI:.+]] = comb.extract %{{.+}} from 6 : (i7) -> i1
// CHECK: %[[B0:.+]] = comb.icmp eq %[[HI]], %false : i1
// CHECK: %[[EN0:.+]] = comb.and %true, %[[B0]] : i1
// CHECK: seq.firmem.write_port %amem_b0_p0[%{{.+}}] = %{{.+}}, clock %clk enable %[[EN0]] : <64 x 6>
// CHECK: seq.firmem.write_port %amem_b0_p1[%{{.+}}] = %{{.+}}, clock %clk enable %[[EN0]] : <64 x 6>
// CHECK: %[[A:.+]] = hw.array_create %{{.+}}, %{{.+}} : i12
// CHECK: %[[R:.+]] = hw.array_get %[[A]][%{{.+}}] : !hw.array<2xi12>, i1
// CHECK: hw.output %[[R]] : i12
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

// A part takes as many banks as its port allows: 512 x 2 is 32 loads a bit,
// over 16; 256 x 2 is 16: two parts of 256 entries, no bit split.
// CHECK-LABEL: hw.module @Banks
// CHECK: %dmem_b0 = seq.firmem 0, 1, undefined, undefined : <256 x 2>
// CHECK: %dmem_b1 = seq.firmem 0, 1, undefined, undefined : <256 x 2>
// CHECK-NOT: %dmem_b2
// CHECK-NOT: _p0
hw.module @Banks(in %clk: !seq.clock, in %inst2: i1, in %row: i8, in %d: i2,
                 out r: i2) {
  %true = hw.constant true
  %idx = seq.firreg %inst2 clock %clk {foldsim.control} : i1
  %dmem = seq.firmem 0, 1, undefined, undefined : <512 x 2>
  %addr = comb.concat %row, %idx : i8, i1
  seq.firmem.write_port %dmem[%addr] = %d, clock %clk enable %true : <512 x 2>
  %rd = seq.firmem.read_port %dmem[%addr], clock %clk : <512 x 2>
  hw.output %rd : i2
}
