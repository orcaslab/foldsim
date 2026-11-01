// As top.sv (eight leaves), the leaves' state also taking a late input in the same cycle:
// the last leaf's output comes back into the first leaf's RAM writes (e), so
// the leaves' outputs (the read at the held address, mixed with the input)
// are written at ready passes, each a few host cycles before its state pass.
module Leaf(input logic clk, input logic [7:0] d, input logic [7:0] e,
            output logic [7:0] q);
  logic [7:0] ram [4];
  logic [1:0] raddr_q;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:6]] <= d ^ e;
    if (d[1]) raddr_q <= d[3:2];
  end
  assign q = ram[raddr_q] ^ d;
endmodule
// Eight leaves in the loop: the span from the first leaf's ready pass to its
// state pass sets the frame, with the state gap the schedule keeps after it.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] v [8];
  Leaf l0(.clk, .d(x), .e(v[7]), .q(v[0]));
  for (genvar i = 1; i < 8; i++) begin : g
    Leaf l(.clk, .d(v[i - 1]), .e(8'h0), .q(v[i]));
  end
  assign y = v[0];
endmodule
