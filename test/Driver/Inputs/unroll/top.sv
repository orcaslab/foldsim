// A leaf whose input is a reduction over 1100 bit pairs in a procedural loop.
module Leaf(input logic clk, input logic d, output logic q);
  always_ff @(posedge clk) q <= d;
endmodule

module Top(input logic clk, input logic [1099:0] a, input logic [1099:0] b, output logic y);
  logic r;
  always_comb begin
    r = 1'b0;
    for (int i = 0; i < 1100; ++i)
      r |= a[i] & b[1099 - i];
  end
  Leaf l(.clk, .d(r), .q(y));
endmodule
