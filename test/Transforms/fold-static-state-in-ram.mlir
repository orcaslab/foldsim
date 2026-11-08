// State in RAM (foldsim-fold-static `state-in-ram`, DESIGN.md of fold-sir-1108), on the leaves of
// fold-static-storage.mlir (four instances of Leaf on one copy: an 80-bit and an 8-bit register, a
// 4-bit address register, a register read from a memory, and a 16-entry memory `mem`). Written
// before the implementation, XFAIL until it lands (then drop it and tune the names).
// RUN: foldsim-opt %S/fold-static-storage.mlir --pass-pipeline='builtin.module(foldsim-fold-static{module=Top state-in-ram=Leaf})' 2> %t.log | FileCheck %s
// RUN: FileCheck %s --check-prefix=LOG < %t.log
// RUN: foldsim-opt %S/fold-static-storage.mlir --pass-pipeline='builtin.module(foldsim-fold-static{module=Top state-in-ram=Leaf:1000})' 2> %t.small.log | FileCheck %s --check-prefix=SMALL
// RUN: foldsim-opt %S/fold-static-storage.mlir --pass-pipeline='builtin.module(foldsim-fold-static{module=Top state-in-ram=Leaf storage=Leaf:sir=block})' | FileCheck %s --check-prefix=BLOCK
// P auto: the fewest passes that fill a column's depth, no more than one column needs (100 bits: 2),
// as pages with masks; the RAM by SLR group once scheduled.
// RUN: foldsim-opt %S/fold-static-storage.mlir --pass-pipeline='builtin.module(foldsim-fold-static{module=Top state-in-ram=Leaf::auto})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=AUTO
// AUTO: state-in-ram: Leaf: 100 bits/instance, 4 instances, 1 x 72 columns, P 2, block (4 registers, 8 rows, 1 RAMB36 a copy, 8 host cycles a copy's passes)
// AUTO: state-in-ram: Leaf: 0 hazard registers, 100 capture bits
// AUTO: state-in-ram: RAM by SLR group: -1: 1 RAMB36, 0 URAM288;
// AUTO: pages Leaf: p 2
// Off: the same netlist as the register stores of the storage test (no `sir`).
// RUN: foldsim-opt %S/fold-static-storage.mlir --pass-pipeline='builtin.module(foldsim-fold-static{module=Top})' | FileCheck %s --check-prefix=OFF
// A class below the minimum bits keeps its per-register stores.
// SMALL-NOT: _sir
// OFF-NOT: _sir
// The state packed in one RAM word per instance (80 + 8 + 4 + 8 bits, the registers written every
// pass in one run), read two host cycles ahead; the memory stays a memory of 16 x 4 entries.
// CHECK: %mem = seq.firmem 1, 1, old, undefined : <64 x 8, mask 1>
// CHECK: seq.firmem 1, 1, old, undefined {sv.attributes = [#sv.attribute<"ram_style" = "{{.*}}">]} : <4 x {{[0-9]+}}, mask {{[0-9]+}}>
// CHECK: seq.firmem.read_port %{{.*sir.*}}[%ahead0_inst2]
// CHECK-NOT: %wide = seq.firmem
// BLOCK: ram_style" = "\22block\22
// LOG: state-in-ram: Leaf: {{[0-9]+}} bits/instance, 4 instances, {{[0-9]+}} x 72 columns, P 1
