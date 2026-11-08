// A leaf with a RAM of 4 rows read through an address register that loads
// only when its enable is set (a held read, as Driver/Inputs/vmem_held), and
// a late, deep output: the read data through two multiplies mixed with the
// input and the state. The held address and its enable come from the input
// and the state; the last stage's must reach the skip.
module Leaf(input logic clk, input logic [7:0] d, input logic [7:0] e,
            output logic [7:0] q);
  logic [7:0] ram [4];
  logic [1:0] raddr_q;
  logic [7:0] acc, r;
  always_comb r = ((ram[raddr_q] * 8'd29) + d) ^ (acc * 8'd13);
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:6]] <= d ^ e ^ acc;
    if (d[1] ^ acc[0]) raddr_q <= d[3:2] ^ acc[2:1];
    acc <= acc + r + e;
  end
  assign q = r * 8'd11;
endmodule
