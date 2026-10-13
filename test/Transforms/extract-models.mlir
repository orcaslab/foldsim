// RUN: foldsim-opt %s --foldsim-extract-models="module=Parent models=Child" | FileCheck %s

hw.module @Child(in %clk : !seq.clock, in %reset : i1, in %id : i2, in %a : i8, in %unread : i8, out y : i8, out code : i4) {
  %c3_i3 = hw.constant 3 : i3
  %c0_i6 = hw.constant 0 : i6
  %idw = comb.concat %c0_i6, %id : i6, i2
  %twice = comb.add %a, %a : i8
  %sum = comb.add %twice, %idw : i8
  %y = seq.firreg %sum clock %clk : i8
  %bit = comb.extract %a from 0 : (i8) -> i1
  %next = comb.concat %bit, %c3_i3 : i1, i3
  %code = seq.firreg %next clock %clk : i4
  hw.output %y, %code : i8, i4
}

// The low bits of `code` are always 3, so only its top bit is a port. `a.a` is
// driven by the parent's logic, so it is a port; `b.a` is `a`'s output and
// `c.a` the parent's input, so both are connected directly. `id` is constant:
// a port of no bits, for the tokens only. `unread` is not read by `Child`: no
// port. The outputs the parent's logic does not read, `a.y` only wired to `b`,
// are channels of tokens too.
// CHECK-LABEL: hw.module @Parent(
// CHECK-SAME:    in %x : i8,
// CHECK-SAME:    in %a_y : i0, in %a_code_3_ : i1, in %b_y : i8, in %b_code : i0, in %c_y : i0, in %c_code_3_ : i1,
// CHECK-SAME:    out z : i8, out w : i4,
// CHECK-SAME:    out a_id : i0, out a_a : i8, out b_id : i0, out c_id : i0)
// CHECK-SAME:    attributes {foldsim.direct_inputs = ["b_a = a_y", "c_a = x"]}
// CHECK-NOT:   hw.instance
// CHECK-DAG:   [[C3:%.+]] = hw.constant 3 : i3
// CHECK-DAG:   [[A_CODE:%.+]] = comb.concat %a_code_3_, [[C3]] : i1, i3
// CHECK-DAG:   [[SUM:%.+]] = comb.add %x, %x : i8
// CHECK-DAG:   [[W:%.+]] = comb.xor [[A_CODE]], {{%.+}} : i4
// CHECK:       hw.output %b_y, [[W]], {{%.+}}, [[SUM]], {{%.+}}, {{%.+}} : i8, i4, i0, i8, i0, i0
hw.module @Parent(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z : i8, out w : i4) {
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %sum = comb.add %x, %x : i8
  %a.y, %a.code = hw.instance "a" @Child(clk: %clk: !seq.clock, reset: %reset: i1, id: %c1_i2: i2, a: %sum: i8, unread: %sum: i8) -> (y: i8, code: i4)
  %b.y, %b.code = hw.instance "b" @Child(clk: %clk: !seq.clock, reset: %reset: i1, id: %c2_i2: i2, a: %a.y: i8, unread: %sum: i8) -> (y: i8, code: i4)
  %c.y, %c.code = hw.instance "c" @Child(clk: %clk: !seq.clock, reset: %reset: i1, id: %c2_i2: i2, a: %x: i8, unread: %sum: i8) -> (y: i8, code: i4)
  %w = comb.xor %a.code, %c.code : i4
  hw.output %b.y, %w : i8, i4
}

// RUN: foldsim-opt %s --foldsim-extract-models="module=StructParent models=StructChild" | FileCheck %s --check-prefix=STRUCT

// A constant field of a struct is a constant bit too: only `v` is a port.
// STRUCT-LABEL: hw.module @StructParent(
// STRUCT-SAME:    in %c_o_3_0_ : i4,
// STRUCT-SAME:    out q : i8)
hw.module @StructChild(in %clk : !seq.clock, in %a : i4, out o : i8) {
  %c5_i4 = hw.constant 5 : i4
  %s = hw.struct_create (%c5_i4, %a) : !hw.struct<k: i4, v: i4>
  %b = hw.bitcast %s : (!hw.struct<k: i4, v: i4>) -> i8
  hw.output %b : i8
}
hw.module @StructParent(in %clk : !seq.clock, in %x : i4, out q : i8) {
  %c.o = hw.instance "c" @StructChild(clk: %clk: !seq.clock, a: %x: i4) -> (o: i8)
  hw.output %c.o : i8
}

// RUN: foldsim-opt %s --foldsim-extract-models="module=AggParent models=AggChild input-bits=c_o:3:2,c_o:1:0" | FileCheck %s --check-prefix=AGG
// RUN: not foldsim-opt %s --foldsim-extract-models="module=AggParent models=AggChild input-bits=c_o:1:0" 2>&1 | FileCheck %s --check-prefix=AGGERR

// The inputs are given: the bits they leave out are an aggregate constant.
// AGG-LABEL: hw.module @AggParent(
// AGG-SAME:    in %c_o_1_0_ : i2, in %c_o_3_2_ : i2,
// AGG:         [[O:%.+]] = comb.concat %c{{.*}}, %c_o_3_2_, %c_o_1_0_ : i4, i2, i2
// AGG:         hw.output [[O]] : i8
// The bits left out must be constants.
// AGGERR: error: bit 2 of 'c_o' is not a constant and is not in an input
hw.module @AggChild(in %clk : !seq.clock, in %a : i4, out o : i8) {
  %k = hw.aggregate_constant [2 : i2, 1 : i2] : !hw.struct<x: i2, y: i2>
  %s = hw.struct_create (%k, %a) : !hw.struct<k: !hw.struct<x: i2, y: i2>, v: i4>
  %b = hw.bitcast %s : (!hw.struct<k: !hw.struct<x: i2, y: i2>, v: i4>) -> i8
  hw.output %b : i8
}
hw.module @AggParent(in %clk : !seq.clock, in %x : i4, out q : i8) {
  %c.o = hw.instance "c" @AggChild(clk: %clk: !seq.clock, a: %x: i4) -> (o: i8)
  hw.output %c.o : i8
}

// RUN: foldsim-opt %s --foldsim-extract-models="module=Parent models=Child netlist" | FileCheck %s --check-prefix=NET

// As a netlist: the parent's logic is `Parent_model`, with no ports for tokens
// only, and `Parent` instantiates it beside the children, wired as before.
// NET-LABEL: hw.module private @Parent_model(
// NET-SAME:    in %x : i8, in %a_code_3_ : i1, in %b_y : i8, in %c_code_3_ : i1,
// NET-SAME:    out z : i8, out w : i4, out a_a : i8)
// NET-NOT:   attributes
// NET-LABEL: hw.module @Parent(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z : i8, out w : i4)
// NET-SAME:    attributes {foldsim.netlist}
// NET:         %logic.z, %logic.w, %logic.a_a = hw.instance "logic" @Parent_model(clk: %clk: !seq.clock, reset: %reset: i1, x: %x: i8, a_code_3_: [[ACODE:%.+]]: i1, b_y: %b.y: i8, c_code_3_: [[CCODE:%.+]]: i1)
// NET:         %a.y, %a.code = hw.instance "a" @Child(clk: %clk: !seq.clock, reset: {{%.+}}: i1, id: [[ID1:%.+]]: i2, a: %logic.a_a: i8, unread: {{%.+}}: i8)
// NET:         %b.y, %b.code = hw.instance "b" @Child(clk: %clk: !seq.clock, reset: {{%.+}}: i1, id: {{%.+}}: i2, a: %a.y: i8,
// NET:         %c.y, %c.code = hw.instance "c" @Child(clk: %clk: !seq.clock, reset: {{%.+}}: i1, id: {{%.+}}: i2, a: %x: i8,
// NET-DAG:     [[ACODE]] = comb.extract %a.code from 3 : (i4) -> i1
// NET-DAG:     [[CCODE]] = comb.extract %c.code from 3 : (i4) -> i1
// NET-DAG:     [[ID1]] = hw.constant 1 : i2
// NET:         hw.output %logic.z, %logic.w : i8, i4

// RUN: foldsim-opt %s --foldsim-extract-models="module=Mix models=Src,Dst netlist" | FileCheck %s --check-prefix=MIX

// In a netlist, an input wired from bits of several models' outputs and
// constants is connected to them directly, not through the parent's model; a
// bit an output always holds (`k`'s top two) is that constant.
// MIX-LABEL: hw.module private @Mix_model(in %clk : !seq.clock, in %x : i4, in %d_y : i8, out z : i8)
// MIX-LABEL: hw.module @Mix(
// MIX-DAG:     %p.y, %p.k = hw.instance "p" @Src(
// MIX-DAG:     %q.y, %q.k = hw.instance "q" @Src(
// MIX-DAG:     %d.y = hw.instance "d" @Dst(clk: %clk: !seq.clock, a: [[A:%[0-9]+]]: i8)
// MIX-DAG:     [[PY:%[0-9]+]] = comb.extract %p.y from 1 : (i4) -> i3
// MIX-DAG:     [[KC:%[-a-z0-9_]+]] = hw.constant -2 : i2
// MIX-DAG:     [[KL:%[0-9]+]] = comb.extract %q.k from 0 : (i4) -> i2
// MIX-DAG:     [[A]] = comb.concat %true, [[PY]], [[KC]], [[KL]] : i1, i3, i2, i2
hw.module @Src(in %clk : !seq.clock, in %a : i4, out y : i4, out k : i4) {
  %c2 = hw.constant -2 : i2
  %y = seq.firreg %a clock %clk : i4
  %lo = comb.extract %a from 0 : (i4) -> i2
  %r = seq.firreg %lo clock %clk : i2
  %k = comb.concat %c2, %r : i2, i2
  hw.output %y, %k : i4, i4
}
hw.module @Dst(in %clk : !seq.clock, in %a : i8, out y : i8) {
  %y = seq.firreg %a clock %clk : i8
  hw.output %y : i8
}
hw.module @Mix(in %clk : !seq.clock, in %x : i4, out z : i8) {
  %c1 = hw.constant 1 : i1
  %p.y, %p.k = hw.instance "p" @Src(clk: %clk: !seq.clock, a: %x: i4) -> (y: i4, k: i4)
  %q.y, %q.k = hw.instance "q" @Src(clk: %clk: !seq.clock, a: %x: i4) -> (y: i4, k: i4)
  %py = comb.extract %p.y from 1 : (i4) -> i3
  %kh = comb.extract %q.k from 2 : (i4) -> i2
  %kl = comb.extract %q.k from 0 : (i4) -> i2
  %in = comb.concat %c1, %py, %kh, %kl : i1, i3, i2, i2
  %d.y = hw.instance "d" @Dst(clk: %clk: !seq.clock, a: %in: i8) -> (y: i8)
  hw.output %d.y : i8
}

// RUN: foldsim-opt %s --foldsim-extract-models="module=Absorb models=Unit,Kept absorb-logic=4" | FileCheck %s --check-prefix=ABS

// With `absorb-logic`, logic that drives the same input of every instance is
// computed in the model: `en` comes from the model's own output and the
// parent's input `go`, so the model takes `go` instead; names do not make
// the instances' logic differ. `Kept` is also instantiated outside the
// parent, so its ports stay as they are.
// ABS-LABEL: hw.module @Unit(in %clk : !seq.clock, in %en : i1, in %d : i4, in %en_1 : i1, out busy : i1, out q : i4)
// ABS-NEXT:    %true = hw.constant true
// ABS-NEXT:    [[NB:%.+]] = comb.xor %r, %true : i1
// ABS-NEXT:    [[EN:%.+]] = comb.and [[NB]], %en_1 {{.*}}: i1
// ABS-NEXT:    %r = seq.firreg [[EN]] clock %clk : i1
// ABS-LABEL: hw.module @Kept(in %clk : !seq.clock, in %en : i1, out busy : i1)
// ABS-LABEL: hw.module @Absorb(
// ABS-SAME:    in %k_busy : i1, out z : i4, out b : i1, out k_en : i1)
// ABS-SAME:    foldsim.direct_inputs = ["u0_d = x", "u0_en_1 = go", "u1_d = u0_q", "u1_en_1 = go"]
hw.module @Unit(in %clk : !seq.clock, in %en : i1, in %d : i4, out busy : i1, out q : i4) {
  %r = seq.firreg %en clock %clk : i1
  %q = seq.firreg %d clock %clk : i4
  hw.output %r, %q : i1, i4
}
hw.module @Kept(in %clk : !seq.clock, in %en : i1, out busy : i1) {
  %r = seq.firreg %en clock %clk : i1
  hw.output %r : i1
}
hw.module @Absorb(in %clk : !seq.clock, in %go : i1, in %x : i4, out z : i4, out b : i1) {
  %true = hw.constant true
  %n0 = comb.xor %u0.busy, %true : i1
  %e0 = comb.and %n0, %go {sv.namehint = "u0_en"} : i1
  %u0.busy, %u0.q = hw.instance "u0" @Unit(clk: %clk: !seq.clock, en: %e0: i1, d: %x: i4) -> (busy: i1, q: i4)
  %n1 = comb.xor %u1.busy, %true : i1
  %e1 = comb.and %n1, %go {sv.namehint = "u1_en"} : i1
  %u1.busy, %u1.q = hw.instance "u1" @Unit(clk: %clk: !seq.clock, en: %e1: i1, d: %u0.q: i4) -> (busy: i1, q: i4)
  %nk = comb.xor %k.busy, %true : i1
  %ek = comb.and %nk, %go : i1
  %k.busy = hw.instance "k" @Kept(clk: %clk: !seq.clock, en: %ek: i1) -> (busy: i1)
  hw.output %u1.q, %k.busy : i4, i1
}
hw.module @Other(in %clk : !seq.clock, in %en : i1, out busy : i1) {
  %k.busy = hw.instance "k" @Kept(clk: %clk: !seq.clock, en: %en: i1) -> (busy: i1)
  hw.output %k.busy : i1
}

// RUN: foldsim-opt %s --foldsim-extract-models="module=PresetParent models=PresetChild" | FileCheck %s --check-prefix=PRESET

// A register always 0 after its first edge, but 1 before it, is not the
// constant 0: the parent takes it as an input. One preset to 0 is.
// PRESET-LABEL: hw.module @PresetParent(
// PRESET-SAME:    in %p_q : i1, out q : i1, out k : i1)
// PRESET:         hw.output %p_q, %false
hw.module @PresetChild(in %clk : !seq.clock, out q : i1, out k : i1) {
  %false = hw.constant false
  %r = seq.firreg %false clock %clk preset 1 : i1
  %s = seq.firreg %false clock %clk preset 0 : i1
  hw.output %r, %s : i1, i1
}
hw.module @PresetParent(in %clk : !seq.clock, out q : i1, out k : i1) {
  %p.q, %p.k = hw.instance "p" @PresetChild(clk: %clk: !seq.clock) -> (q: i1, k: i1)
  hw.output %p.q, %p.k : i1, i1
}
