module Top(input logic clk, input logic [7:0] x, output logic [7:0] y, output logic [7:0] z);
`ifdef WITH_LEAF
  Leaf a(.clk, .d(x), .q(y));
  Leaf b(.clk, .d(y), .q(z));
`endif
endmodule
