/* verilator lint_off DECLFILENAME */
// a32-S26X-PH synthetic nested target ("tiny GPU"): a class tree with links and memories, for physical domains.
//   Top: NSM x Sm, NB x L2 bank, a request/response exchange between them (every SM reaches every bank).
//   Sm:  Core + Lsu + L1 (links Core<->Lsu<->L1, L1<->Top's banks).
//   Core: Alu + Rf (register file memory) + a pc/accumulator; Lsu: an address queue (memory).
//   L1: a direct-mapped data memory with tags; L2 bank: a data memory.
// Registers reset on rst; memories have no initializers: a row reads as a default until a valid bit (reset) says written, so the target's outputs are deterministic.
// W: datapath width (scales the design for out-of-context runs).
module Alu #(parameter W = 16) (input logic clk, input logic rst, input logic [W-1:0] a, input logic [W-1:0] b,
                                output logic [W-1:0] r);
  logic [W-1:0] acc;
  always_ff @(posedge clk) if (rst) acc <= '0; else acc <= acc + (a ^ {b[W/2-1:0], b[W-1:W/2]});
  assign r = acc ^ (a + b);
endmodule

module Rf #(parameter W = 16, D = 32) (input logic clk, input logic rst, input logic we, input logic [$clog2(D)-1:0] wa,
                                       input logic [W-1:0] wd, input logic [$clog2(D)-1:0] ra, output logic [W-1:0] rd);
  logic [W-1:0] mem [D];
  logic [D-1:0] vld;  // a row reads as its default until written (no memory initializers)
  always_ff @(posedge clk) begin
    if (we) mem[wa] <= wd;
    rd <= vld[ra] ? mem[ra] : W'(ra * 7 + 3);
  end
  always_ff @(posedge clk) if (rst) vld <= '0; else if (we) vld[wa] <= 1'b1;
endmodule

module Core #(parameter W = 16, D = 32) (input logic clk, input logic rst, input logic [W-1:0] x,
    output logic req_v, output logic [W-1:0] req_a, input logic rsp_v, input logic [W-1:0] rsp_d,
    output logic [W-1:0] y);
  localparam A = $clog2(D);
  logic [A-1:0] pc;
  logic [W-1:0] rd, r, last;
  Rf #(W, D) rf(.clk, .rst, .we(rsp_v), .wa(rsp_d[A-1:0] ^ pc), .wd(rsp_d ^ r), .ra(pc), .rd);
  Alu #(W) alu(.clk, .rst, .a(rd ^ x), .b(last), .r);
  always_ff @(posedge clk)
    if (rst) begin pc <= '0; last <= '0; end
    else begin pc <= pc + A'(1) + A'(x[0]); last <= rsp_v ? rsp_d : r; end
  assign req_v = r[0] | x[1];
  assign req_a = r;
  assign y = last ^ rd;
endmodule

module Lsu #(parameter W = 16, Q = 8) (input logic clk, input logic rst,
    input logic in_v, input logic [W-1:0] in_a, output logic out_v, output logic [W-1:0] out_a,
    input logic l1_v, input logic [W-1:0] l1_d, output logic rsp_v, output logic [W-1:0] rsp_d);
  localparam B = $clog2(Q);
  logic [W-1:0] q [Q];
  logic [B:0] wp, rp;
  wire full = (wp[B] != rp[B]) && (wp[B-1:0] == rp[B-1:0]);
  wire empty = wp == rp;
  always_ff @(posedge clk) begin
    if (rst) begin wp <= '0; rp <= '0; end
    else begin
      if (in_v && !full) begin q[wp[B-1:0]] <= in_a; wp <= wp + 1'b1; end
      if (!empty) rp <= rp + 1'b1;
    end
  end
  assign out_v = !empty;
  assign out_a = q[rp[B-1:0]];
  assign rsp_v = l1_v;
  assign rsp_d = l1_d + W'(rp);
endmodule

module L1 #(parameter W = 16, S = 64) (input logic clk, input logic rst,
    input logic req_v, input logic [W-1:0] req_a, output logic rsp_v, output logic [W-1:0] rsp_d,
    output logic mreq_v, output logic [W-1:0] mreq_a, input logic mrsp_v, input logic [W-1:0] mrsp_d);
  localparam I = $clog2(S);
  logic [W-1:0] data [S];
  logic [W-I-1:0] tag [S];
  logic [S-1:0] tv;  // tag valid bits (reset)
  logic hvld;
  logic hv; logic [W-1:0] ha, hd; logic [W-I-1:0] ht;
  always_ff @(posedge clk) begin
    if (rst) hv <= 1'b0; else hv <= req_v;
    ha <= req_a;
    hd <= data[req_a[I-1:0]];
    ht <= tag[req_a[I-1:0]];
    hvld <= tv[req_a[I-1:0]];
    if (mrsp_v) begin data[mrsp_d[I-1:0]] <= mrsp_d; tag[mrsp_d[I-1:0]] <= mrsp_d[W-1:I]; end
  end
  always_ff @(posedge clk) if (rst) tv <= '0; else if (mrsp_v) tv[mrsp_d[I-1:0]] <= 1'b1;
  wire hit = hv && hvld && ht == ha[W-1:I];
  assign rsp_v = hit | mrsp_v;
  assign rsp_d = hit ? hd : mrsp_d;
  assign mreq_v = hv && !hit;
  assign mreq_a = ha;
endmodule

module Sm #(parameter W = 16) (input logic clk, input logic rst, input logic [W-1:0] x, output logic [W-1:0] y,
    output logic mreq_v, output logic [W-1:0] mreq_a, input logic mrsp_v, input logic [W-1:0] mrsp_d);
  logic cv, lv, l1v, rv; logic [W-1:0] ca, la, l1d, rd;
  Core #(W) core(.clk, .rst, .x, .req_v(cv), .req_a(ca), .rsp_v(rv), .rsp_d(rd), .y);
  Lsu #(W) lsu(.clk, .rst, .in_v(cv), .in_a(ca), .out_v(lv), .out_a(la), .l1_v(l1v), .l1_d(l1d), .rsp_v(rv), .rsp_d(rd));
  L1 #(W) l1(.clk, .rst, .req_v(lv), .req_a(la), .rsp_v(l1v), .rsp_d(l1d), .mreq_v, .mreq_a, .mrsp_v, .mrsp_d);
endmodule

module L2 #(parameter W = 16, S = 256) (input logic clk, input logic rst,
    input logic req_v, input logic [W-1:0] req_a, output logic rsp_v, output logic [W-1:0] rsp_d);
  localparam I = $clog2(S);
  logic [W-1:0] data [S];
  logic [S-1:0] vld;  // a row reads as its default until written
  logic v; logic [W-1:0] a, d;
  always_ff @(posedge clk) begin
    if (rst) v <= 1'b0; else v <= req_v;
    a <= req_a;
    d <= vld[req_a[I-1:0]] ? data[req_a[I-1:0]] : W'(req_a[I-1:0]) * W'(29);
    if (req_v) data[req_a[I-1:0] ^ I'(1)] <= req_a ^ d;
  end
  always_ff @(posedge clk) if (rst) vld <= '0; else if (req_v) vld[req_a[I-1:0] ^ I'(1)] <= 1'b1;
  assign rsp_v = v;
  assign rsp_d = d ^ a;
endmodule

module Top #(parameter NSM = 4, NB = 2, W = 16) (input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [NSM-1:0] mv, sv;
  logic [W-1:0] ma [NSM], sd [NSM], sy [NSM];
  logic [NB-1:0] bv, rv;
  logic [W-1:0] ba [NB], rd [NB];
  logic [$clog2(NSM)-1:0] turn;
  always_ff @(posedge clk) if (rst) turn <= '0; else turn <= turn + 1'b1;
  for (genvar s = 0; s < NSM; s++) begin : g_sm
    Sm #(W) sm(.clk, .rst, .x({(W/8){x ^ 8'(s)}}), .y(sy[s]), .mreq_v(mv[s]), .mreq_a(ma[s]), .mrsp_v(sv[s]), .mrsp_d(sd[s]));
    // an SM's response: its bank's, when the bank serves it this cycle
    assign sv[s] = rv[s % NB] && (int'(turn) == s);
    assign sd[s] = rd[s % NB];
  end
  for (genvar k = 0; k < NB; k++) begin : g_l2
    L2 #(W) l2(.clk, .rst, .req_v(bv[k]), .req_a(ba[k]), .rsp_v(rv[k]), .rsp_d(rd[k]));
    // bank k takes the request of the SM whose turn it is (round robin over the SMs that map to it)
    always_comb begin
      bv[k] = 1'b0; ba[k] = '0;
      for (int s = 0; s < NSM; s++)
        if (s % NB == k && int'(turn) == s) begin bv[k] = mv[s]; ba[k] = ma[s]; end
    end
  end
  always_comb begin
    automatic logic [W-1:0] acc = '0;
    for (int s = 0; s < NSM; s++) acc ^= sy[s];
    y = acc[7:0] ^ acc[W-1:W-8];
  end
endmodule
