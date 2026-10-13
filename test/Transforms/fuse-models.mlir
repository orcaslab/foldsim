// RUN: foldsim-opt %s --foldsim-fuse-models="module=Top groups=Req+Arb" | FileCheck %s
// RUN: not foldsim-opt %s --foldsim-fuse-models="module=Top groups=Req+Lone" 2>&1 | FileCheck %s --check-prefix=ERR

// Each `Req` exchanges bits with one `Arb`, whatever the instance order: `rdy`
// depends on `req` combinationally, and `y` on `rdy`. Fused, the path is inside
// one model, and the outputs only the other reads are not ports.
// CHECK-LABEL: hw.module private @Req_Arb(in %Req_id : i32, in %clk : !seq.clock, in %Req_x : i4, in %Arb_en : i1, out Req_y : i4, out Arb_q : i4)
// CHECK-DAG:     %r = seq.firreg %Req_x clock %clk : i4
// CHECK-DAG:     [[BIT:%.+]] = comb.extract %r from 0 : (i4) -> i1
// CHECK-DAG:     [[RDY:%.+]] = comb.and [[BIT]], %Arb_en : i1
// CHECK-DAG:     [[Y:%.+]] = comb.mux [[RDY]], %r, %{{.+}} : i4
// CHECK-DAG:     %q = seq.firreg %r clock %clk : i4
// CHECK:         hw.output [[Y]], %q : i4, i4
// CHECK-LABEL: hw.module @Top(
// CHECK:         %a0.Req_y, %a0.Arb_q = hw.instance "a0" @Req_Arb(Req_id: %c0_i32: i32, clk: %clk: !seq.clock, Req_x: %x0: i4, Arb_en: %en: i1)
// CHECK-NEXT:    %a1.Req_y, %a1.Arb_q = hw.instance "a1" @Req_Arb(Req_id: %c1_i32: i32, clk: %clk: !seq.clock, Req_x: %x1: i4, Arb_en: %en: i1)
// CHECK:         hw.output %a0.Req_y, %a1.Req_y, %a0.Arb_q, %a1.Arb_q, %l.q
// CHECK-NOT:   hw.module private @Req(
// CHECK-NOT:   hw.module private @Arb(

// ERR: error: cannot pair the instances of 'Req' (2) and 'Lone' (1)

hw.module private @Req(in %id : i32, in %clk : !seq.clock, in %x : i4, in %rdy : i1, out req : i4, out y : i4) {
  %r = seq.firreg %x clock %clk : i4
  %c0 = hw.constant 0 : i4
  %y = comb.mux %rdy, %r, %c0 : i4
  hw.output %r, %y : i4, i4
}
hw.module private @Arb(in %clk : !seq.clock, in %req : i4, in %en : i1, out rdy : i1, out q : i4) {
  %q = seq.firreg %req clock %clk : i4
  %b = comb.extract %req from 0 : (i4) -> i1
  %rdy = comb.and %b, %en : i1
  hw.output %rdy, %q : i1, i4
}
hw.module private @Lone(in %clk : !seq.clock, in %req : i4, out q : i4) {
  %q = seq.firreg %req clock %clk : i4
  hw.output %q : i4
}
hw.module @Top(in %clk : !seq.clock, in %x0 : i4, in %x1 : i4, in %en : i1, out y0 : i4, out y1 : i4, out q0 : i4, out q1 : i4, out q2 : i4) attributes {foldsim.netlist} {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %a0.req, %a0.y = hw.instance "a0" @Req(id: %c0_i32: i32, clk: %clk: !seq.clock, x: %x0: i4, rdy: %b0.rdy: i1) -> (req: i4, y: i4)
  %a1.req, %a1.y = hw.instance "a1" @Req(id: %c1_i32: i32, clk: %clk: !seq.clock, x: %x1: i4, rdy: %b1.rdy: i1) -> (req: i4, y: i4)
  %b1.rdy, %b1.q = hw.instance "b1" @Arb(clk: %clk: !seq.clock, req: %a1.req: i4, en: %en: i1) -> (rdy: i1, q: i4)
  %b0.rdy, %b0.q = hw.instance "b0" @Arb(clk: %clk: !seq.clock, req: %a0.req: i4, en: %en: i1) -> (rdy: i1, q: i4)
  %l.q = hw.instance "l" @Lone(clk: %clk: !seq.clock, req: %x0: i4) -> (q: i4)
  hw.output %a0.y, %a1.y, %b0.q, %b1.q, %l.q : i4, i4, i4, i4, i4
}
