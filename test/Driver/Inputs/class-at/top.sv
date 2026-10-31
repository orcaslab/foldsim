// Two parameter sets of one module, under two instances of one parent.
module Leaf #(parameter W = 8) (input logic clk, input logic [W-1:0] d, output logic [W-1:0] q);
  always_ff @(posedge clk) q <= d;
endmodule

module Pair(input logic clk, input logic [7:0] x, output logic [7:0] y, output logic [3:0] z);
  logic [7:0] a;
  Leaf #(8) l0(.clk, .d(x), .q(a));
  Leaf #(8) l1(.clk, .d(a), .q(y));
  Leaf #(4) l2(.clk, .d(x[3:0]), .q(z));
endmodule

module Top(input logic clk, input logic [7:0] x, output logic [7:0] y, output logic [3:0] z,
           output logic [7:0] u, output logic [3:0] v);
  Pair p(.clk, .x, .y, .z);
  Pair q(.clk, .x, .y(u), .z(v));
endmodule
