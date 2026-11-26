// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-lane-scan{module=Top min-lanes=4},canonicalize,cse)' 2>&1 | FileCheck %s

// A round-robin grant over eight lanes, the only reduction in Top: from the
// pointer `ptr`, the first requesting lane at (ptr + i) % 8 (the `rot` scan;
// test/Driver/Inputs/lane-glue has it next to every other form). It is the
// only lane-scan test with a rotated chain that runs without Verilator.
// matchPatterns walks each rotated chain's muxes to collect their gathers;
// built with GCC (11.4, 13.3), that walk read the mux after its lifetime had
// ended (a for-condition variable used in the iteration expression) and
// foldsim-opt crashed on this input.
// Input: `foldsim --static --lane-scan Leaf` flat.mlir of
//   module Top: Leaf u[8] (req, gnt); always_comb for (i = 7..0)
//   { c = (ptr + i) % 8; if (req[c]) {gv = 1; g = c;} }; gnt[j] = gv & g == j;
//   ptr <= gv ? g + 1 : ptr; y = {g, gv}.

// CHECK: lane-scan: cell Top_lane_Leaf
// CHECK: lane-scan: Top lanes Leaf x8: {{.*}}scans ac 1 rot 1 (2 chains, 9 bits)

// The cell: the any-request OR (ac) and the rotated priority as two
// first-wins halves, lanes at or after the start (ptr % 8) and lanes before.
// CHECK-LABEL: hw.module @Top_lane_Leaf
// CHECK-SAME: in %lane : i3
// CHECK-SAME: in %ch0_in : i1, in %in_req : i1, in %ch1_in : i8
// CHECK: comb.or %ch0_in, %in_req : i1
// CHECK: %[[AT:.+]] = comb.icmp uge %{{.+}}, %{{.+}} : i4
// CHECK: comb.and %in_req, %[[AT]] : i1
// CHECK: comb.mux %{{.+}}, %lane, %{{.+}} : i3
// CHECK: comb.mux %{{.+}}, %lane, %{{.+}} : i3

// Top: the chain's end picks the half that found a lane, ptr when none did.
// CHECK-LABEL: hw.module @Top
// CHECK: %[[ANY:.+]] = comb.concat %[[G:.+]], %lane_Leaf_7.ch0_out : i3, i1
// CHECK: hw.instance "lane_Leaf_7" @Top_lane_Leaf
// CHECK: %[[BEFORE:.+]] = comb.mux %{{.+}}, %{{.+}}, %ptr : i3
// CHECK: %[[G]] = comb.mux %{{.+}}, %{{.+}}, %[[BEFORE]] : i3
// CHECK: hw.output %[[ANY]] : i4

module {
  hw.module @Leaf(in %clk : !seq.clock, in %rst : i1, in %x : i8, in %gnt : i1, out req : i1) {
    %c0_i7 = hw.constant 0 : i7
    %c5_i8 = hw.constant 5 : i8
    %0 = comb.mul %s, %c5_i8 : i8
    %1 = comb.concat %c0_i7, %gnt : i7, i1
    %2 = comb.add %0, %x, %1 : i8
    %3 = comb.mux %rst, %x, %2 : i8
    %s = seq.firreg %3 clock %clk : i8
    %4 = comb.extract %s from 0 : (i8) -> i1
    %5 = comb.extract %s from 3 : (i8) -> i1
    %6 = comb.xor %4, %5 : i1
    hw.output %6 : i1
  }
  hw.module @Top(in %clk : !seq.clock, in %rst : i1, in %x : i8, out y : i4) {
    %c1_i32 = hw.constant 1 : i32
    %c2_i32 = hw.constant 2 : i32
    %c3_i32 = hw.constant 3 : i32
    %c4_i32 = hw.constant 4 : i32
    %c5_i32 = hw.constant 5 : i32
    %c6_i32 = hw.constant 6 : i32
    %c7_i32 = hw.constant 7 : i32
    %c-1_i8 = hw.constant -1 : i8
    %c0_i24 = hw.constant 0 : i24
    %c0_i29 = hw.constant 0 : i29
    %c-1_i3 = hw.constant -1 : i3
    %c-2_i3 = hw.constant -2 : i3
    %c-3_i3 = hw.constant -3 : i3
    %c-4_i3 = hw.constant -4 : i3
    %c3_i3 = hw.constant 3 : i3
    %c2_i3 = hw.constant 2 : i3
    %c1_i3 = hw.constant 1 : i3
    %c0_i3 = hw.constant 0 : i3
    %c8_i32 = hw.constant 8 : i32
    %c7_i8 = hw.constant 7 : i8
    %c6_i8 = hw.constant 6 : i8
    %c5_i8 = hw.constant 5 : i8
    %c4_i8 = hw.constant 4 : i8
    %c3_i8 = hw.constant 3 : i8
    %c2_i8 = hw.constant 2 : i8
    %c1_i8 = hw.constant 1 : i8
    %0 = comb.concat %g_leaf_7.u.req, %g_leaf_6.u.req, %g_leaf_5.u.req, %g_leaf_4.u.req, %g_leaf_3.u.req, %g_leaf_2.u.req, %g_leaf_1.u.req, %g_leaf_0.u.req : i1, i1, i1, i1, i1, i1, i1, i1
    %g_leaf_0.u.req = hw.instance "g_leaf_0.u" @Leaf(clk: %clk: !seq.clock, rst: %rst: i1, x: %x: i8, gnt: %90: i1) -> (req: i1)
    %1 = comb.xor %x, %c1_i8 : i8
    %g_leaf_1.u.req = hw.instance "g_leaf_1.u" @Leaf(clk: %clk: !seq.clock, rst: %rst: i1, x: %1: i8, gnt: %92: i1) -> (req: i1)
    %2 = comb.xor %x, %c2_i8 : i8
    %g_leaf_2.u.req = hw.instance "g_leaf_2.u" @Leaf(clk: %clk: !seq.clock, rst: %rst: i1, x: %2: i8, gnt: %94: i1) -> (req: i1)
    %3 = comb.xor %x, %c3_i8 : i8
    %g_leaf_3.u.req = hw.instance "g_leaf_3.u" @Leaf(clk: %clk: !seq.clock, rst: %rst: i1, x: %3: i8, gnt: %96: i1) -> (req: i1)
    %4 = comb.xor %x, %c4_i8 : i8
    %g_leaf_4.u.req = hw.instance "g_leaf_4.u" @Leaf(clk: %clk: !seq.clock, rst: %rst: i1, x: %4: i8, gnt: %98: i1) -> (req: i1)
    %5 = comb.xor %x, %c5_i8 : i8
    %g_leaf_5.u.req = hw.instance "g_leaf_5.u" @Leaf(clk: %clk: !seq.clock, rst: %rst: i1, x: %5: i8, gnt: %100: i1) -> (req: i1)
    %6 = comb.xor %x, %c6_i8 : i8
    %g_leaf_6.u.req = hw.instance "g_leaf_6.u" @Leaf(clk: %clk: !seq.clock, rst: %rst: i1, x: %6: i8, gnt: %102: i1) -> (req: i1)
    %7 = comb.xor %x, %c7_i8 : i8
    %g_leaf_7.u.req = hw.instance "g_leaf_7.u" @Leaf(clk: %clk: !seq.clock, rst: %rst: i1, x: %7: i8, gnt: %104: i1) -> (req: i1)
    %8 = comb.concat %c0_i29, %ptr : i29, i3
    %9 = comb.add %8, %c7_i32 : i32
    %10 = comb.mods %9, %c8_i32 : i32
    %11 = comb.extract %10 from 8 : (i32) -> i24
    %12 = comb.icmp eq %11, %c0_i24 : i24
    %13 = comb.extract %10 from 0 : (i32) -> i8
    %14 = comb.mux %12, %13, %c-1_i8 : i8
    %15 = comb.shru %0, %14 : i8
    %16 = comb.extract %15 from 0 : (i8) -> i1
    %17 = comb.extract %10 from 0 : (i32) -> i3
    %18 = comb.mux %16, %17, %ptr : i3
    %19 = comb.add %8, %c6_i32 : i32
    %20 = comb.mods %19, %c8_i32 : i32
    %21 = comb.extract %20 from 8 : (i32) -> i24
    %22 = comb.icmp eq %21, %c0_i24 : i24
    %23 = comb.extract %20 from 0 : (i32) -> i8
    %24 = comb.mux %22, %23, %c-1_i8 : i8
    %25 = comb.shru %0, %24 : i8
    %26 = comb.extract %25 from 0 : (i8) -> i1
    %27 = comb.extract %20 from 0 : (i32) -> i3
    %28 = comb.mux %26, %27, %18 : i3
    %29 = comb.add %8, %c5_i32 : i32
    %30 = comb.mods %29, %c8_i32 : i32
    %31 = comb.extract %30 from 8 : (i32) -> i24
    %32 = comb.icmp eq %31, %c0_i24 : i24
    %33 = comb.extract %30 from 0 : (i32) -> i8
    %34 = comb.mux %32, %33, %c-1_i8 : i8
    %35 = comb.shru %0, %34 : i8
    %36 = comb.extract %35 from 0 : (i8) -> i1
    %37 = comb.extract %30 from 0 : (i32) -> i3
    %38 = comb.mux %36, %37, %28 : i3
    %39 = comb.add %8, %c4_i32 : i32
    %40 = comb.mods %39, %c8_i32 : i32
    %41 = comb.extract %40 from 8 : (i32) -> i24
    %42 = comb.icmp eq %41, %c0_i24 : i24
    %43 = comb.extract %40 from 0 : (i32) -> i8
    %44 = comb.mux %42, %43, %c-1_i8 : i8
    %45 = comb.shru %0, %44 : i8
    %46 = comb.extract %45 from 0 : (i8) -> i1
    %47 = comb.extract %40 from 0 : (i32) -> i3
    %48 = comb.mux %46, %47, %38 : i3
    %49 = comb.add %8, %c3_i32 : i32
    %50 = comb.mods %49, %c8_i32 : i32
    %51 = comb.extract %50 from 8 : (i32) -> i24
    %52 = comb.icmp eq %51, %c0_i24 : i24
    %53 = comb.extract %50 from 0 : (i32) -> i8
    %54 = comb.mux %52, %53, %c-1_i8 : i8
    %55 = comb.shru %0, %54 : i8
    %56 = comb.extract %55 from 0 : (i8) -> i1
    %57 = comb.extract %50 from 0 : (i32) -> i3
    %58 = comb.mux %56, %57, %48 : i3
    %59 = comb.add %8, %c2_i32 : i32
    %60 = comb.mods %59, %c8_i32 : i32
    %61 = comb.extract %60 from 8 : (i32) -> i24
    %62 = comb.icmp eq %61, %c0_i24 : i24
    %63 = comb.extract %60 from 0 : (i32) -> i8
    %64 = comb.mux %62, %63, %c-1_i8 : i8
    %65 = comb.shru %0, %64 : i8
    %66 = comb.extract %65 from 0 : (i8) -> i1
    %67 = comb.extract %60 from 0 : (i32) -> i3
    %68 = comb.mux %66, %67, %58 : i3
    %69 = comb.add %8, %c1_i32 : i32
    %70 = comb.mods %69, %c8_i32 : i32
    %71 = comb.extract %70 from 8 : (i32) -> i24
    %72 = comb.icmp eq %71, %c0_i24 : i24
    %73 = comb.extract %70 from 0 : (i32) -> i8
    %74 = comb.mux %72, %73, %c-1_i8 : i8
    %75 = comb.shru %0, %74 : i8
    %76 = comb.extract %75 from 0 : (i8) -> i1
    %77 = comb.extract %70 from 0 : (i32) -> i3
    %78 = comb.mux %76, %77, %68 : i3
    %79 = comb.mods %8, %c8_i32 : i32
    %80 = comb.extract %79 from 8 : (i32) -> i24
    %81 = comb.icmp eq %80, %c0_i24 : i24
    %82 = comb.extract %79 from 0 : (i32) -> i8
    %83 = comb.mux %81, %82, %c-1_i8 : i8
    %84 = comb.shru %0, %83 : i8
    %85 = comb.extract %84 from 0 : (i8) -> i1
    %86 = comb.extract %79 from 0 : (i32) -> i3
    %87 = comb.or %85, %76, %66, %56, %46, %36, %26, %16 : i1
    %88 = comb.mux %85, %86, %78 : i3
    %89 = comb.icmp eq %88, %c0_i3 : i3
    %90 = comb.and %87, %89 : i1
    %91 = comb.icmp eq %88, %c1_i3 : i3
    %92 = comb.and %87, %91 : i1
    %93 = comb.icmp eq %88, %c2_i3 : i3
    %94 = comb.and %87, %93 : i1
    %95 = comb.icmp eq %88, %c3_i3 : i3
    %96 = comb.and %87, %95 : i1
    %97 = comb.icmp eq %88, %c-4_i3 : i3
    %98 = comb.and %87, %97 : i1
    %99 = comb.icmp eq %88, %c-3_i3 : i3
    %100 = comb.and %87, %99 : i1
    %101 = comb.icmp eq %88, %c-2_i3 : i3
    %102 = comb.and %87, %101 : i1
    %103 = comb.icmp eq %88, %c-1_i3 : i3
    %104 = comb.and %87, %103 : i1
    %105 = comb.add %88, %c1_i3 : i3
    %106 = comb.mux %87, %105, %ptr : i3
    %107 = comb.mux %rst, %c0_i3, %106 : i3
    %ptr = seq.firreg %107 clock %clk : i3
    %108 = comb.concat %88, %87 : i3, i1
    hw.output %108 : i4
  }
}

