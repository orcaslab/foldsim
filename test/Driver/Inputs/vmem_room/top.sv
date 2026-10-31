// Eight leaves side by side, each with a RAM of 16 rows read through a
// register and written every cycle: every state pass of the one copy issues
// a read and a write, in consecutive host cycles.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd;
  always_ff @(posedge clk) begin
    ram[d[7:4]] <= d;
    rd <= ram[d[3:0]];
  end
  assign q = rd ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] q [8];
  for (genvar i = 0; i < 8; ++i) begin : g_leaf
    Leaf l(.clk, .d(x ^ 8'(i * 37)), .q(q[i]));
  end
  assign y = q[0] ^ q[1] ^ q[2] ^ q[3] ^ q[4] ^ q[5] ^ q[6] ^ (q[7] + 8'd1);
endmodule
