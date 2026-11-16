/* verilator lint_off DECLFILENAME */
// syn2 (a32-S28C): a small synthetic target whose memories exercise every vmem feature S28 stage A folds, at once.
// Interface: clk, rst (a target input, high for the first cycles and on demand), x[7:0] in, y[7:0] out: tb_s28.sv's
// (RST_IN). Every memory has a registered read (a virtual-memory candidate) and no initializer; a row reads as its
// default until a valid bit (reset) says it was written, so the outputs are deterministic.
//   Top:  NSM x Sm (one copy per SLR group: --copies Sm:3 --slr-assign), an L2 with a 512-bit data RAM and a 64-bit
//         tag RAM read-first (SLR2), a 26-bit narrow RAM written every cycle at one row (read-during-write old data).
//   Sm:   Core + Cache.
//   Core: GPR, 4 banks of 1024-bit rows written in the same pass with byte enables (MW = 128: 2-beat entries, 4 adapters
//         on one local switch), read at addresses held across cycles (vmem-skip-held-reads) and written at the same
//         row it reads (read-first pairs); an accumulator.
//   Cache: a 512-bit data RAM and a 26-bit tag RAM (MW = 1), direct mapped, filled on a miss from Top's L2.
// W: the datapath width (the banks are 8 W wide).
module Core #(parameter W = 128, D = 32) (input logic clk, input logic rst, input logic [7:0] x,
    output logic req_v, output logic [15:0] req_a, input logic rsp_v, input logic [15:0] rsp_d,
    output logic [15:0] y);
  localparam A = $clog2(D);
  logic [8*W-1:0] gpr0 [D], gpr1 [D], gpr2 [D], gpr3 [D];
  logic [D-1:0] v0, v1, v2, v3;
  logic [A-1:0] ra, wa, rb, rc;
  logic [8*W-1:0] r0, r1, r2, r3;
  logic h0, h1, h2, h3;
  logic [15:0] acc;
  logic [8*W/8-1:0] be;
  wire [8*W-1:0] wd = {(8*W/16){acc ^ {8'(x), 8'(ra)}}};
  always_comb begin
    for (int i = 0; i < W; i++) be[i] = acc[i % 16] ^ x[i % 8];
  end
  // The read address changes only on x[2] (else it holds: a held read); the write goes to the row read.
  always_ff @(posedge clk) begin
    if (rst) ra <= '0; else if (x[2]) ra <= ra + A'(x[7:5]) + A'(1);
    // Banks 2 and 3 at addresses of their own (memories of their own); banks 0 and 1 share ra (one memory of
    // two lanes with --vmem-merge-lanes).
    if (rst) rb <= '0; else rb <= rb + A'(x[1:0]);
    if (rst) rc <= '0; else if (x[5]) rc <= ~ra;
    wa <= ra;
    r0 <= gpr0[ra]; r1 <= gpr1[ra]; r2 <= gpr2[rb]; r3 <= gpr3[rc];
    h0 <= v0[ra]; h1 <= v1[ra]; h2 <= v2[rb]; h3 <= v3[rc];
    for (int i = 0; i < W; i++) begin
      // (A row's first write after reset takes every byte: a valid row never shows bytes of before the reset.)
      if (x[0] && (be[i] || !v0[ra])) gpr0[ra][8*i +: 8] <= wd[8*i +: 8];
      if (x[1] && (be[i] || !v1[ra])) gpr1[ra][8*i +: 8] <= ~wd[8*i +: 8];
      if (x[3] && (be[i] || !v2[rb])) gpr2[rb][8*i +: 8] <= wd[8*i +: 8] + 8'(i);
      if (x[4] && (be[i] || !v3[rc])) gpr3[rc][8*i +: 8] <= wd[8*i +: 8] ^ 8'(i);
    end
  end
  always_ff @(posedge clk)
    if (rst) begin v0 <= '0; v1 <= '0; v2 <= '0; v3 <= '0; end
    else begin
      if (x[0]) v0[ra] <= 1'b1;
      if (x[1]) v1[ra] <= 1'b1;
      if (x[3]) v2[rb] <= 1'b1;
      if (x[4]) v3[rc] <= 1'b1;
    end
  function automatic logic [15:0] fold16(input logic [8*W-1:0] v);
    logic [15:0] s = '0;
    for (int i = 0; i < 8 * W / 16; i++) s ^= v[16*i +: 16] + 16'(i);
    return s;
  endfunction
  wire [15:0] g = (h0 ? fold16(r0) : 16'h1111) ^ (h1 ? fold16(r1) : 16'h2222) ^
                  (h2 ? fold16(r2) : 16'h3333) ^ (h3 ? fold16(r3) : 16'h4444);
  always_ff @(posedge clk)
    if (rst) acc <= 16'h5a5a; else acc <= {acc[14:0], acc[15]} ^ g ^ (rsp_v ? rsp_d : 16'h0);
  assign req_v = acc[0] ^ x[6];
  assign req_a = acc ^ 16'(wa);
  assign y = acc;
endmodule

module Cache #(parameter S = 64) (input logic clk, input logic rst,
    input logic req_v, input logic [15:0] req_a, output logic rsp_v, output logic [15:0] rsp_d,
    output logic mreq_v, output logic [15:0] mreq_a, input logic mrsp_v, input logic [15:0] mrsp_d);
  localparam I = $clog2(S);
  logic [511:0] data [S];
  logic [25:0] tag [S];
  logic [S-1:0] tv;
  logic hv, hvld; logic [15:0] ha; logic [511:0] hd; logic [25:0] ht;
  always_ff @(posedge clk) begin
    if (rst) hv <= 1'b0; else hv <= req_v;
    ha <= req_a;
    hd <= data[req_a[I-1:0]];
    ht <= tag[req_a[I-1:0]];
    hvld <= tv[req_a[I-1:0]];
    if (mrsp_v) begin
      data[mrsp_d[I-1:0]] <= {32{mrsp_d ^ 16'(mrsp_d[I-1:0])}};
      tag[mrsp_d[I-1:0]] <= {10'(mrsp_d[15:I]), mrsp_d};
    end
  end
  always_ff @(posedge clk) if (rst) tv <= '0; else if (mrsp_v) tv[mrsp_d[I-1:0]] <= 1'b1;
  wire hit = hv && hvld && ht[25:16] == 10'(ha[15:I]);
  assign rsp_v = hit | mrsp_v;
  assign rsp_d = hit ? hd[16 * ha[4:0] +: 16] : mrsp_d;
  assign mreq_v = hv && !hit;
  assign mreq_a = ha;
endmodule

module Sm (input logic clk, input logic rst, input logic [7:0] x, output logic [15:0] y,
    output logic mreq_v, output logic [15:0] mreq_a, input logic mrsp_v, input logic [15:0] mrsp_d);
  logic cv, rv; logic [15:0] ca, rd;
  Core core(.clk, .rst, .x, .req_v(cv), .req_a(ca), .rsp_v(rv), .rsp_d(rd), .y);
  Cache cache(.clk, .rst, .req_v(cv), .req_a(ca), .rsp_v(rv), .rsp_d(rd), .mreq_v, .mreq_a, .mrsp_v, .mrsp_d);
endmodule

module L2 #(parameter S = 256) (input logic clk, input logic rst,
    input logic req_v, input logic [15:0] req_a, output logic rsp_v, output logic [15:0] rsp_d);
  localparam I = $clog2(S);
  logic [511:0] data [S];
  logic [63:0] tg [S];
  logic [S-1:0] vld;
  logic v, hv; logic [15:0] a; logic [511:0] d; logic [63:0] t;
  always_ff @(posedge clk) begin
    if (rst) v <= 1'b0; else v <= req_v;
    a <= req_a;
    d <= data[req_a[I-1:0]];
    t <= tg[req_a[I-1:0]];
    hv <= vld[req_a[I-1:0]];
    // read-first: the row read this cycle is written this cycle too
    // (t only where its row was written: a never-written row's bits are no part of any value.)
    if (req_v) begin
      data[req_a[I-1:0]] <= {32{req_a ^ 16'(hv ? t : 64'h0)}};
      tg[req_a[I-1:0]] <= {req_a, (hv ? t[63:16] : 48'h0)};
    end
  end
  always_ff @(posedge clk) if (rst) vld <= '0; else if (req_v) vld[req_a[I-1:0]] <= 1'b1;
  assign rsp_v = v;
  assign rsp_d = (hv ? d[16 * a[4:0] +: 16] ^ t[15:0] : a * 16'd29) ^ a;
endmodule

module Top #(parameter NSM = 3) (input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [NSM-1:0] mv, sv;
  logic [15:0] ma [NSM], sy [NSM];
  logic [$clog2(NSM)-1:0] turn;
  logic bv, rv; logic [15:0] ba, rd;
  // A narrow RAM written every cycle at one row and read at another (read-during-write old data elsewhere).
  logic [25:0] nar [16];
  logic [15:0] nv;
  logic [25:0] nr; logic nh;
  always_ff @(posedge clk) begin
    if (rst) turn <= '0; else turn <= (turn == $clog2(NSM)'(NSM - 1)) ? '0 : turn + 1'b1;
    nr <= nar[x[3:0]];
    nh <= nv[x[3:0]];
    nar[x[7:4]] <= {x, rd, 2'(turn)};
  end
  always_ff @(posedge clk) if (rst) nv <= '0; else nv[x[7:4]] <= 1'b1;
  for (genvar s = 0; s < NSM; s++) begin : g_sm
    Sm sm(.clk, .rst, .x(x ^ 8'(s)), .y(sy[s]), .mreq_v(mv[s]), .mreq_a(ma[s]), .mrsp_v(sv[s]), .mrsp_d(rd));
    assign sv[s] = rv && (int'(turn) == s);
  end
  always_comb begin
    bv = 1'b0; ba = '0;
    for (int s = 0; s < NSM; s++) if (int'(turn) == s) begin bv = mv[s]; ba = ma[s]; end
  end
  L2 l2(.clk, .rst, .req_v(bv), .req_a(ba), .rsp_v(rv), .rsp_d(rd));
  always_comb begin
    automatic logic [15:0] acc = nh ? nr[15:0] : 16'h0f0f;
    for (int s = 0; s < NSM; s++) acc ^= sy[s];
    y = acc[7:0] ^ acc[15:8];
  end
endmodule
