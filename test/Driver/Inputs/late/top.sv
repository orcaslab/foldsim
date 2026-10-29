// Three leaves in a chain of late links: each leaf's output depends on its
// input in the same cycle.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] r;
  always_ff @(posedge clk) r <= d;
  assign q = d + r;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(y));
endmodule
