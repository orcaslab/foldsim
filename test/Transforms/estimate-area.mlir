// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-area{top=Top target=virtexuplusHBM lut-scale=1 output=- report=})' -o /dev/null | FileCheck %s --check-prefix=U55C
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-area{top=Top target=versal lut-scale=1 report=-})' -o /dev/null | FileCheck %s --check-prefix=V80
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-area{top=Top target=virtexuplusHBM lut-scale=1 memories=- report=})' -o /dev/null | FileCheck %s --check-prefix=MEM
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-area{top=Top target=virtexuplusHBM lut-scale=1 black-boxes=Leaf output=- report=})' -o /dev/null | FileCheck %s --check-prefix=STUB
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-area{target=virtexuplusHBM modules=- state-threads=1,4 report=})' -o /dev/null | FileCheck %s --check-prefix=STATE
// RUN: foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-area{top=Top target=virtexuplusHBM report=-})' -o /dev/null | FileCheck %s --check-prefix=SCALED
// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-estimate-area{top=Top target=agilex})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN

// The design: 16 adder LUTs, one truth table, 16 LUTs of a product by a
// constant, 6 of a 3 x 3 product zero-extended to 16 bits (a table per bit
// the product can have), 9.12 LUTs of a 64 x 8
// distributed RAM, and 36.48 of a 128 x 16 one with its 5.33-LUT multiplexer;
// xil_fma's 850 LUTs, 1400 flip-flops and 2 DSPs; 16 + 2 x 8 register bits;
// half a block each for 512 x 36 and its true dual-port twin, two for
// 1024 x 72, two UltraRAMs for 8192 x 72;
// one DSP48E2 for 16 x 16 and four for 33 x 33.
// U55C: depth,path,module,lut,lutram,ff,bram,uram,dsp,macros,total_lut,total_lutram,total_ff,total_bram,total_uram,total_dsp,total_macros
// U55C-NEXT: 0,,Top,44,46,16,3.0,2,5,0,894,46,1432,3.0,2,7,0
// U55C-NEXT: 1,l0,Leaf,0,0,8,0.0,0,0,0,0,0,8,0.0,0,0,0
// U55C-NEXT: 1,l1,Leaf,0,0,8,0.0,0,0,0,0,0,8,0.0,0,0,0
// U55C-NEXT: 1,fma,xil_fma,850,0,1400,0.0,0,2,0,850,0,1400,0.0,0,2,0

// On the V80, DSP58s are 27 x 24: 33 x 33 still takes four.
// V80: Top on versal: 940 LUTs (894 logic, 46 distributed RAM), 1432 flip-flops, 3.0 block RAMs, 2 UltraRAMs, 7 DSPs
// V80: logic: 39 LUTs mapped x 1.000 (lutScale) + 5 of memory multiplexers + 850 of vendor IP
// V80: mapped LUTs by op: comb.add 16 comb.mul 22 comb.truth_table 1
// V80: external modules: xil_fma x1

// Two read-write ports are one true dual-port block, not two copies.
// MEM: path,name,depth,width,reads,writes,read_writes,latency,style,placement,lut,lutram,ff,bram,uram,dsp,macros
// MEM-DAG: ,a64,64,8,1,1,0,0,-,lutram,0,9,0,0.0,0,0,0
// MEM-DAG: ,s512,512,36,1,1,0,1,-,bram,0,0,0,0.5,0,0,0
// MEM-DAG: ,b1024,1024,72,1,1,0,1,block,bram,0,0,0,2.0,0,0,0
// MEM-DAG: ,d128,128,16,1,1,0,1,distributed,lutram,5,36,0,0.0,0,0,0
// MEM-DAG: ,u8192,8192,72,1,1,0,1,ultra,uram,0,0,0,0.0,2,0,0
// MEM-DAG: ,t512,512,36,0,0,2,1,block,bram,0,0,0,0.5,0,0,0

// A harness stub's area is not the design's.
// STUB: 0,,Top,44,46,16,3.0,2,5,0,894,46,1416,3.0,2,7,0

// Per module: Leaf's 8 register bits stored for 4 threads are a 4 x 8
// distributed RAM, 4.56 LUTs; Top's leave out xil_fma's flip-flops.
// STATE: "Leaf": {
// STATE: "ff": 8,
// STATE: "4": {
// STATE-NEXT: "mem_bram": 0,
// STATE-NEXT: "mem_lutram": 0,
// STATE-NEXT: "reg_lutram": 4.5{{[0-9]+}}
// STATE: "Top": {
// STATE: "ip_ff": 1400,
// STATE: "4": {
// STATE-NEXT: "mem_bram": 12,
// STATE-NEXT: "mem_lutram": 227.7{{[0-9]+}},
// STATE-NEXT: "reg_lutram": 18.2{{[0-9]+}}

// By default the mapping's LUTs scale by the target's lutScale; the memory
// multiplexers and vendor IP's do not.
// SCALED: logic: 39 LUTs mapped x 0.337 (lutScale) + 5 of memory multiplexers + 850 of vendor IP

// UNKNOWN: no Beacon target 'agilex'; one of versal virtexuplus virtexuplusHBM zynquplus asap7

hw.module.extern @xil_fma(in %a : i32, out r : i32)

hw.module @Leaf(in %clk : !seq.clock, in %d : i8, out q : i8) {
  %r = seq.firreg %d clock %clk : i8
  hw.output %r : i8
}

hw.module @Top(in %clk : !seq.clock, in %a : i16, in %b : i16, in %x : i33,
               in %y : i33, in %en : i1, in %f : i32, in %a6 : i6,
               in %a7 : i7, in %a9 : i9, in %a10 : i10, in %a13 : i13,
               in %d8 : i8, in %d16 : i16, in %d36 : i36, in %d72 : i72,
               out p : i16, out p33 : i33, out s : i16, out t : i1,
               out q0 : i8, out q1 : i8, out r : i32, out m0 : i8,
               out m1 : i36, out m2 : i72, out m3 : i16, out m4 : i72,
               out k : i16, out n : i16, out m5 : i36) {
  %p = comb.mul %a, %b : i16
  %p33 = comb.mul %x, %y : i33
  %sum = comb.add %a, %b : i16
  %s = seq.firreg %sum clock %clk : i16
  %t = comb.truth_table %en, %en -> [false, false, false, true]
  %c0_i6 = hw.constant 0 : i6
  %c0_i13 = hw.constant 0 : i13
  %c3_i16 = hw.constant 3 : i16
  %x10 = comb.concat %c0_i6, %a10 : i6, i10
  %k = comb.mul %x10, %c3_i16 : i16
  %a3 = comb.extract %a from 0 : (i16) -> i3
  %b3 = comb.extract %b from 0 : (i16) -> i3
  %a16 = comb.concat %c0_i13, %a3 : i13, i3
  %b16 = comb.concat %c0_i13, %b3 : i13, i3
  %n = comb.mul %a16, %b16 : i16
  %q0 = hw.instance "l0" @Leaf(clk: %clk : !seq.clock, d: %d8 : i8) -> (q : i8)
  %q1 = hw.instance "l1" @Leaf(clk: %clk : !seq.clock, d: %d8 : i8) -> (q : i8)
  %r = hw.instance "fma" @xil_fma(a: %f : i32) -> (r : i32)

  %a64 = seq.firmem 0, 1, undefined, undefined {name = "a64"} : <64 x 8>
  seq.firmem.write_port %a64[%a6] = %d8, clock %clk enable %en : <64 x 8>
  %m0 = seq.firmem.read_port %a64[%a6], clock %clk : <64 x 8>

  %s512 = seq.firmem 1, 1, undefined, undefined {name = "s512"} : <512 x 36>
  seq.firmem.write_port %s512[%a9] = %d36, clock %clk enable %en : <512 x 36>
  %m1 = seq.firmem.read_port %s512[%a9], clock %clk : <512 x 36>

  %b1024 = seq.firmem 1, 1, undefined, undefined {name = "b1024", sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <1024 x 72>
  seq.firmem.write_port %b1024[%a10] = %d72, clock %clk enable %en : <1024 x 72>
  %m2 = seq.firmem.read_port %b1024[%a10], clock %clk : <1024 x 72>

  %d128 = seq.firmem 1, 1, undefined, undefined {name = "d128", sv.attributes = [#sv.attribute<"ram_style" = "\22distributed\22">]} : <128 x 16>
  seq.firmem.write_port %d128[%a7] = %d16, clock %clk enable %en : <128 x 16>
  %m3 = seq.firmem.read_port %d128[%a7], clock %clk : <128 x 16>

  %u8192 = seq.firmem 1, 1, undefined, undefined {name = "u8192", sv.attributes = [#sv.attribute<"ram_style" = "\22ultra\22">]} : <8192 x 72>
  seq.firmem.write_port %u8192[%a13] = %d72, clock %clk enable %en : <8192 x 72>
  %m4 = seq.firmem.read_port %u8192[%a13], clock %clk : <8192 x 72>

  %t512 = seq.firmem 1, 1, undefined, undefined {name = "t512", sv.attributes = [#sv.attribute<"ram_style" = "\22block\22">]} : <512 x 36>
  %m5 = seq.firmem.read_write_port %t512[%a9] = %d36 if %en, clock %clk enable %en : <512 x 36>
  %m6 = seq.firmem.read_write_port %t512[%a9] = %d36 if %en, clock %clk enable %en : <512 x 36>

  hw.output %p, %p33, %s, %t, %q0, %q1, %r, %m0, %m1, %m2, %m3, %m4, %k, %n, %m5 : i16, i33, i16, i1, i8, i8, i32, i8, i36, i72, i16, i72, i16, i16, i36
}
