// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-index-class-inputs{module=Top verbose})' 2>&1 | FileCheck %s

// An arbiter reading its data at a grant, through two constant slices: per
// read an index output (the slice's low index plus the get's) and an element
// input; the netlist reads the array at the instance's index outputs.
// CHECK: foldsim-index-class-inputs: Arb.data_in: 4 x 8 b read at 2 index(es), 2 instance(s)
// CHECK: foldsim-index-class-inputs: 1 input(s), 2 read(s); instance input bits 64 -> 40
// CHECK-LABEL: hw.module @Arb(
// CHECK-SAME: in %g : i1, in %data_in_at0 : i8, in %data_in_at1 : i8, out out : i8, out data_in_index0 : i2, out data_in_index1 : i2)
// CHECK: %[[I0:.+]] = comb.concat %false, %g : i1, i1
// CHECK-NEXT: %[[X0:.+]] = comb.add %[[I0]], %c0_i2 : i2
// CHECK: %[[X1:.+]] = comb.add %{{.+}}, %c-2_i2 : i2
// CHECK-NEXT: comb.xor %data_in_at0, %data_in_at1
// CHECK-NEXT: hw.output %{{.+}}, %[[X0]], %[[X1]] : i8, i2, i2
hw.module @Arb(in %g : i1, in %data_in : !hw.array<4xi8>, out out : i8) {
  %c0 = hw.constant 0 : i2
  %c2 = hw.constant 2 : i2
  %lo = hw.array_slice %data_in[%c0] : (!hw.array<4xi8>) -> !hw.array<2xi8>
  %hi = hw.array_slice %data_in[%c2] : (!hw.array<4xi8>) -> !hw.array<2xi8>
  %a = hw.array_get %lo[%g] : !hw.array<2xi8>, i1
  %b = hw.array_get %hi[%g] : !hw.array<2xi8>, i1
  %x = comb.xor %a, %b : i8
  hw.output %x : i8
}

// Left alone: an array read as a whole (an output), one of 2 elements (under
// min-elements), one of 3 elements (not a power of two), one read at as many
// indices as it has elements, and one driven by a model's output.
// CHECK-LABEL: hw.module @Whole(in %d : !hw.array<4xi8>, out o : !hw.array<4xi8>)
hw.module @Whole(in %d : !hw.array<4xi8>, out o : !hw.array<4xi8>) {
  hw.output %d : !hw.array<4xi8>
}
// CHECK-LABEL: hw.module @Small(in %i : i1, in %d : !hw.array<2xi8>, out o : i8)
hw.module @Small(in %i : i1, in %d : !hw.array<2xi8>, out o : i8) {
  %x = hw.array_get %d[%i] : !hw.array<2xi8>, i1
  hw.output %x : i8
}
// CHECK-LABEL: hw.module @Odd(in %i : i2, in %d : !hw.array<3xi8>, out o : i8)
hw.module @Odd(in %i : i2, in %d : !hw.array<3xi8>, out o : i8) {
  %x = hw.array_get %d[%i] : !hw.array<3xi8>, i2
  hw.output %x : i8
}
// CHECK-LABEL: hw.module @All(in %i : i2, in %d : !hw.array<4xi8>, out o : i8)
hw.module @All(in %i : i2, in %d : !hw.array<4xi8>, out o : i8) {
  %c1 = hw.constant 1 : i2
  %c2 = hw.constant 2 : i2
  %c3 = hw.constant 3 : i2
  %j = comb.add %i, %c1 : i2
  %k = comb.add %i, %c2 : i2
  %l = comb.add %i, %c3 : i2
  %a = hw.array_get %d[%i] : !hw.array<4xi8>, i2
  %b = hw.array_get %d[%j] : !hw.array<4xi8>, i2
  %c = hw.array_get %d[%k] : !hw.array<4xi8>, i2
  %e = hw.array_get %d[%l] : !hw.array<4xi8>, i2
  %x = comb.xor %a, %b, %c, %e : i8
  hw.output %x : i8
}
// CHECK-LABEL: hw.module @Fed(in %i : i2, in %d : !hw.array<4xi8>, out o : i8)
hw.module @Fed(in %i : i2, in %d : !hw.array<4xi8>, out o : i8) {
  %x = hw.array_get %d[%i] : !hw.array<4xi8>, i2
  hw.output %x : i8
}

// CHECK-LABEL: hw.module @Top(
// CHECK: %[[A:.+]] = hw.bitcast %w : (i32) -> !hw.array<4xi8>
// CHECK-NEXT: %a0.out, %a0.data_in_index0, %a0.data_in_index1 = hw.instance "a0" @Arb(g: %g0: i1, data_in_at0: %[[E0:[0-9]+]]: i8, data_in_at1: %[[E1:[0-9]+]]: i8)
// CHECK-NEXT: %[[E0]] = hw.array_get %[[A]][%a0.data_in_index0]
// CHECK-NEXT: %[[E1]] = hw.array_get %[[A]][%a0.data_in_index1]
// CHECK: hw.instance "a1" @Arb(
hw.module @Top(in %g0 : i1, in %g1 : i1, in %i : i2, in %w : i32, in %v : i16, in %t : i24, out o0 : i8, out o1 : i8, out o2 : !hw.array<4xi8>, out o3 : i8, out o4 : i8, out o5 : i8, out o6 : i8) {
  %arr = hw.bitcast %w : (i32) -> !hw.array<4xi8>
  %a0.out = hw.instance "a0" @Arb(g: %g0: i1, data_in: %arr: !hw.array<4xi8>) -> (out: i8)
  %a1.out = hw.instance "a1" @Arb(g: %g1: i1, data_in: %arr: !hw.array<4xi8>) -> (out: i8)
  %w.o = hw.instance "w" @Whole(d: %arr: !hw.array<4xi8>) -> (o: !hw.array<4xi8>)
  %two = hw.bitcast %v : (i16) -> !hw.array<2xi8>
  %s.o = hw.instance "s" @Small(i: %g0: i1, d: %two: !hw.array<2xi8>) -> (o: i8)
  %three = hw.bitcast %t : (i24) -> !hw.array<3xi8>
  %d.o = hw.instance "d" @Odd(i: %i: i2, d: %three: !hw.array<3xi8>) -> (o: i8)
  %l.o = hw.instance "l" @All(i: %i: i2, d: %arr: !hw.array<4xi8>) -> (o: i8)
  %fed = hw.bitcast %w.o : (!hw.array<4xi8>) -> !hw.array<4xi8>
  %f.o = hw.instance "f" @Fed(i: %i: i2, d: %w.o: !hw.array<4xi8>) -> (o: i8)
  hw.output %a0.out, %a1.out, %w.o, %s.o, %d.o, %l.o, %f.o : i8, i8, !hw.array<4xi8>, i8, i8, i8, i8
}
