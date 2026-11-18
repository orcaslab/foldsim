// The level's replicated instances holding classes (foldsim --class-auto N:OPS:deep): a generate loop of four
// arbiters per bank (as a cache's per-bank request crossbar), each with two slices of a class (Slice) and logic of
// its own that joins them; the arbiters become one nest: class of four instances holding the slices.
module Slice(input logic clk, input logic rst, input logic [7:0] a, input logic [7:0] b, output logic [7:0] o);
  logic t;
  always_ff @(posedge clk) if (rst) t <= 0; else t <= t ^ a[0] ^ b[1];
  assign o = t ? a : b;
endmodule

module Arb(input logic clk, input logic rst, input logic [3:0][7:0] in, output logic [7:0] out);
  logic [7:0] s0, s1, h;
  Slice sl0(.clk, .rst, .a(in[0]), .b(in[1]), .o(s0));
  Slice sl1(.clk, .rst, .a(in[2]), .b(in[3]), .o(s1));
  always_ff @(posedge clk) if (rst) h <= 8'h33; else h <= {h[6:0], h[7]} ^ s0 ^ (s1 + h);
  assign out = h;  // registered: the banks meet only through registers
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [3:0][7:0] o;
  for (genvar i = 0; i < 4; ++i) begin : g_bank
    Arb arb(.clk, .rst, .in({x + 8'(i), o[(i + 1) % 4], x ^ o[(i + 3) % 4], 8'(i * 29) ^ x}), .out(o[i]));
  end
  assign y = o[0] ^ o[1] ^ o[2] ^ o[3];
endmodule
