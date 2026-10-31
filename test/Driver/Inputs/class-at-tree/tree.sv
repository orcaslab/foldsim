// A tree whose depth is a parameter: two slices and a smaller tree below a
// node, one leaf at the bottom; and a tree of one leaf.
module Leaf (input logic clk, input logic [7:0] d, output logic [7:0] q);
  always_ff @(posedge clk) q <= d;
endmodule

module Arb #(parameter N = 1) (input logic clk, input logic [7:0] d, output logic [7:0] q);
  if (N > 1) begin : g_tree
    logic [7:0] a, b;
    Leaf s0(.clk, .d(d), .q(a));
    Leaf s1(.clk, .d(~d), .q(b));
    Arb #(N / 2) t(.clk, .d(a ^ b), .q(q));
  end else begin : g_one
    Leaf l(.clk, .d, .q);
  end
endmodule

module Top(input logic clk, input logic [7:0] x, output logic [7:0] y, output logic [7:0] z);
  Arb #(4) tall(.clk, .d(x), .q(y));
  Arb #(1) flat(.clk, .d(x), .q(z));
endmodule
