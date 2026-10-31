// A leaf per bit of a register made in nested generate loops: 4 x 4 = 16
// leaves, which slang counts as 4 x (4 + 1) = 20 steps of the inner loop.
module Leaf(input logic clk, input logic d, output logic q);
  always_ff @(posedge clk) q <= d;
endmodule

module Top(input logic clk, input logic [15:0] x, output logic [15:0] y);
  for (genvar i = 0; i < 4; ++i) begin : g_row
    for (genvar j = 0; j < 4; ++j) begin : g_col
      Leaf l(.clk, .d(x[i * 4 + j]), .q(y[i * 4 + j]));
    end
  end
endmodule
