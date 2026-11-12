// Four leaves in a chain, each with virtual-memory rows wider than 1024 bits:
// a write-first 8 x 1,104 RAM (one write enable, read through a registered
// address, as a memory macro reads: a write shows in the next cycle's read),
// three write-first 8 x 400 lanes written at one address with their own
// enables (--vmem-merge-lanes: one 1,200-bit row of 150 byte enables), and a
// read-first 8 x 1,040 RAM (registered data). The output mixes bits of every
// row above bit 1,023.
module Ram #(parameter W = 8, parameter A = 4) (
  input  logic         clk,
  input  logic         R0_en,
  input  logic [A-1:0] R0_addr,
  output logic [W-1:0] R0_data,
  input  logic         W0_en,
  input  logic [A-1:0] W0_addr,
  input  logic [W-1:0] W0_data);
  logic [W-1:0] Memory [1 << A];
  logic [A-1:0] _R0_addr_d0;
  always_ff @(posedge clk) begin
    if (R0_en) _R0_addr_d0 <= R0_addr;
    if (W0_en) Memory[W0_addr] <= W0_data;
  end
  assign R0_data = Memory[_R0_addr_d0];
endmodule
module RamRF #(parameter W = 8, parameter A = 4) (
  input  logic         clk,
  input  logic         R0_en,
  input  logic [A-1:0] R0_addr,
  output logic [W-1:0] R0_data,
  input  logic         W0_en,
  input  logic [A-1:0] W0_addr,
  input  logic [W-1:0] W0_data);
  logic [W-1:0] Memory [1 << A];
  always_ff @(posedge clk) begin
    if (R0_en) R0_data <= Memory[R0_addr];
    if (W0_en) Memory[W0_addr] <= W0_data;
  end
endmodule
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  wire [2:0] ra = d[2:0] ^ {d[7], 2'd0};
  wire [2:0] wa = d[5:3];
  wire [1103:0] h;
  Ram #(.W(1104), .A(3)) whole (.clk, .R0_en(d[3] | d[7]), .R0_addr(wa ^ 3'd5), .R0_data(h),
                                .W0_en(d[2] | d[6]), .W0_addr(ra), .W0_data({{69{d + 8'd9}}, {69{d ^ 8'hc3}}}));
  wire [1199:0] wd = {{50{d ^ 8'h5a}}, {50{~d}}, {50{d + 8'd3}}};
  wire [1199:0] r;
  wire [2:0] be = {d[6] | d[1], d[2] ^ d[7], d[0] | d[4]};
  for (genvar i = 0; i < 3; ++i) begin : g_lane
    Ram #(.W(400), .A(3)) lane (.clk, .R0_en(d[1] | d[6]), .R0_addr(ra), .R0_data(r[i*400 +: 400]),
                                .W0_en((d[0] | d[5]) & be[i]), .W0_addr(wa), .W0_data(wd[i*400 +: 400]));
  end
  wire [1039:0] f;
  RamRF #(.W(1040), .A(3)) rf (.clk, .R0_en(d[4] | d[1]), .R0_addr(wa ^ ra), .R0_data(f),
                               .W0_en(d[7] | d[0]), .W0_addr(wa), .W0_data({130{d ^ 8'h96}}));
  assign q = h[7:0] ^ h[1103:1096] ^ r[7:0] ^ r[519:512] ^ r[1031:1024] ^ r[1199:1192] ^ f[1039:1032] ^ (f[1031:1024] << 1) ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
