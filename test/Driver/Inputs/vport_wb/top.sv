// Four leaves in a chain, each with a write-first target RAM of rows wider
// than two 512-bit beats (8 x 1,200, read through a registered address, as a
// memory macro reads: a write shows in the next cycle's read), written often
// at the row it reads, with three lane enables of 400 bits; and a read-first
// 16 x 8 RAM. A leaf writes and reads them from its input; its output mixes
// bits of every beat of the wide row.
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
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  wire [2:0] ra = d[2:0] ^ {d[7], 2'd0};
  wire [2:0] wa = d[5:3];
  wire [1199:0] wd = {{50{d ^ 8'h5a}}, {50{~d}}, {50{d + 8'd3}}};
  wire [1199:0] r;
  wire [2:0] be = {d[6] | d[1], d[2] ^ d[7], d[0] | d[4]};
  for (genvar i = 0; i < 3; ++i) begin : g_lane
    Ram #(.W(400), .A(3)) lane (.clk, .R0_en(d[1] | d[6]), .R0_addr(ra), .R0_data(r[i*400 +: 400]),
                                .W0_en((d[0] | d[5]) & be[i]), .W0_addr(wa), .W0_data(wd[i*400 +: 400]));
  end
  logic [7:0] mem [16];
  logic [7:0] s;
  always_ff @(posedge clk) begin
    if (d[3]) s <= mem[d[3:0]];
    if (d[2]) mem[d[7:4]] <= d ^ 8'h3c;
  end
  // (Deep logic from d to q: the chain's passes come apart, room for the
  // wide row's beats between them.)
  wire [15:0] m0 = d * (d ^ 8'h35);
  wire [15:0] m1 = m0 * (m0[7:0] | 8'h1);
  wire [15:0] m2 = m1 * (m1[15:8] ^ 8'h6b);
  wire [15:0] m3 = m2 * (m2[7:0] + 8'h11);
  assign q = r[7:0] ^ r[519:512] ^ (r[1031:1024] << 1) ^ r[1199:1192] ^ s ^ d ^ m3[11:4];
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
