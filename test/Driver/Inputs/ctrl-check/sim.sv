// Three instances of a 128-deep, 8-bit LUT memory: 2 write-enable control sets each (6 in all).
module mem_128x8(
  input clk
);
  (* ram_style = "distributed" *)
  reg [7:0] Memory[0:127];
endmodule

module foldsim_sim(
  input clk
);
  mem_128x8 m0 (.clk(clk));
  mem_128x8 m1 (.clk(clk));
  mem_128x8 m2 (.clk(clk));
endmodule
