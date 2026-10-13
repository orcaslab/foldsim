module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
`include "step.svh"
  always_ff @(posedge clk) q <= d + Step;
endmodule
