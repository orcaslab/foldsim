// Sixteen leaves, each with a 256 x 8 memory (32 kbit threaded, past the
// 18-kbit floor of the state scratch's on-chip memories), a registered read.
module Leaf(input logic clk, input logic [15:0] d, output logic [7:0] q);
  logic [7:0] ram [256];
  logic [7:0] rd;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[15:8]] <= d[7:0];
    if (d[1]) rd <= ram[d[15:8] ^ 8'h5a];
  end
  assign q = rd ^ d[7:0];
endmodule
module Top(input logic clk, input logic [15:0] x, output logic [7:0] y);
  logic [15:0] w [17];
  assign w[0] = x;
  for (genvar i = 0; i < 16; i++) begin : g
    logic [7:0] q;
    Leaf l(.clk, .d(w[i]), .q(q));
    assign w[i + 1] = {w[i][15:8] + 8'd1, q};
  end
  assign y = w[16][7:0];
endmodule
