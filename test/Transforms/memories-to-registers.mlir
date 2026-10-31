// RUN: foldsim-opt %s --pass-pipeline="builtin.module(foldsim-memories-to-registers{memories=Cam:table,Cam:masked,Cam:late,Cam:missing,Other:table})" -split-input-file -verify-diagnostics | FileCheck %s

// A table matched entry by entry (a read at each constant address), read at
// one more address, and written at one: three registers; the constant reads
// are the registers (zero past the depth), the other a select among them
// (zero past the depth), and an entry loads the data when the write
// addresses it.
// CHECK-LABEL: hw.module @Cam
// CHECK-NOT: seq.firmem
// CHECK-DAG: %[[ZERO:.+]] = hw.constant 0 : i8
// CHECK-DAG: %table_0 = seq.firreg %[[N0:.+]] clock %clk : i8
// CHECK-DAG: %table_1 = seq.firreg %{{.+}} clock %clk : i8
// CHECK-DAG: %table_2 = seq.firreg %{{.+}} clock %clk : i8
// CHECK-DAG: %[[H0:.+]] = comb.icmp eq %wa, %c0_i2{{.*}} : i2
// CHECK-DAG: %[[E0:.+]] = comb.and %we, %[[H0]] : i1
// CHECK-DAG: %[[N0]] = comb.mux %[[E0]], %wd, %table_0 : i8
// CHECK-DAG: %[[A:.+]] = hw.array_create %[[ZERO]], %table_2, %table_1, %table_0 : i8
// CHECK-DAG: %[[R:.+]] = hw.array_get %[[A]][%ra]
// CHECK: hw.output %table_0, %table_1, %table_2, %[[ZERO]], %[[R]]
hw.module @Cam(in %clk : !seq.clock, in %we : i1, in %wa : i2, in %wd : i8, in %ra : i2,
               out m0 : i8, out m1 : i8, out m2 : i8, out m3 : i8, out r : i8) {
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 1 : i2
  %c2 = hw.constant 2 : i2
  %c3 = hw.constant 3 : i2
  %table = seq.firmem name "table" 0, 1, undefined, undefined : <3 x 8, mask 1>
  %0 = seq.firmem.read_port %table[%c0], clock %clk : <3 x 8, mask 1>
  %1 = seq.firmem.read_port %table[%c1], clock %clk : <3 x 8, mask 1>
  %2 = seq.firmem.read_port %table[%c2], clock %clk : <3 x 8, mask 1>
  %3 = seq.firmem.read_port %table[%c3], clock %clk : <3 x 8, mask 1>
  %4 = seq.firmem.read_port %table[%ra], clock %clk : <3 x 8, mask 1>
  seq.firmem.write_port %table[%wa] = %wd, clock %clk enable %we : <3 x 8, mask 1>
  hw.output %0, %1, %2, %3, %4 : i8, i8, i8, i8, i8
}

// -----

// A masked memory loads an entry a lane at a time; of two write ports the
// later one's write lands over the earlier one's.
// CHECK-LABEL: hw.module @Cam
// CHECK-NOT: seq.firmem
// CHECK-DAG: %masked_0 = seq.firreg %[[N:.+]] clock %clk : i8
// CHECK-DAG: %masked_1 = seq.firreg %{{.+}} clock %clk : i8
// CHECK-DAG: %[[HIT:.+]] = comb.and %we, %{{.+}} : i1
// CHECK-DAG: %[[HI:.+]] = comb.extract %wm from 1 : (i2) -> i1
// CHECK-DAG: %[[HIHIT:.+]] = comb.and %[[HI]], %[[HIT]] : i1
// CHECK-DAG: %[[DHI:.+]] = comb.extract %wd from 4 : (i8) -> i4
// CHECK-DAG: %[[OHI:.+]] = comb.extract %masked_0 from 4 : (i8) -> i4
// CHECK-DAG: %[[LHI:.+]] = comb.mux %[[HIHIT]], %[[DHI]], %[[OHI]] : i4
// CHECK-DAG: %[[P:.+]] = comb.concat %[[LHI]], %{{.+}} : i4, i4
// CHECK-DAG: %[[N]] = comb.mux %{{.+}}, %vd, %[[P]] : i8
// CHECK: hw.output %masked_0, %masked_1
hw.module @Cam(in %clk : !seq.clock, in %we : i1, in %wa : i1, in %wd : i8, in %wm : i2,
               in %ve : i1, in %va : i1, in %vd : i8, out m0 : i8, out m1 : i8) {
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %masked = seq.firmem name "masked" 0, 1, undefined, undefined : <2 x 8, mask 2>
  %0 = seq.firmem.read_port %masked[%c0], clock %clk : <2 x 8, mask 2>
  %1 = seq.firmem.read_port %masked[%c1], clock %clk : <2 x 8, mask 2>
  seq.firmem.write_port %masked[%wa] = %wd, clock %clk enable %we mask %wm : <2 x 8, mask 2>, i2
  seq.firmem.write_port %masked[%va] = %vd, clock %clk enable %ve : <2 x 8, mask 2>
  hw.output %0, %1 : i8, i8
}

// -----

// A memory read a cycle late stays one, with a warning; a memory not named
// (`other`) stays too, silently.
// CHECK-LABEL: hw.module @Cam
// CHECK: %late = seq.firmem 1, 1
// CHECK: %other = seq.firmem 0, 1
hw.module @Cam(in %clk : !seq.clock, in %we : i1, in %wa : i1, in %wd : i8, out m0 : i8, out m1 : i8) {
  %c0 = hw.constant 0 : i1
  // expected-warning @+1 {{memory 'late' stays a memory: its reads are registered (read latency 1)}}
  %late = seq.firmem name "late" 1, 1, undefined, undefined : <2 x 8, mask 1>
  %0 = seq.firmem.read_port %late[%c0], clock %clk : <2 x 8, mask 1>
  seq.firmem.write_port %late[%wa] = %wd, clock %clk enable %we : <2 x 8, mask 1>
  %other = seq.firmem name "other" 0, 1, undefined, undefined : <2 x 8, mask 1>
  %1 = seq.firmem.read_port %other[%c0], clock %clk : <2 x 8, mask 1>
  seq.firmem.write_port %other[%wa] = %wd, clock %clk enable %we : <2 x 8, mask 1>
  hw.output %0, %1 : i8, i8
}
