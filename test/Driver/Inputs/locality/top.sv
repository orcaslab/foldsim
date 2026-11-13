// Three leaves in a chain, each output late (it depends on its input in the same cycle), so with
// --zero-latency-links a reader takes it in its producer's host cycle: the link is a path of logic
// from one copy into the next, registered at neither end.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] r;
  always_ff @(posedge clk) r <= d + 8'd1;
  assign q = r ^ d;
endmodule

module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(y));
endmodule
