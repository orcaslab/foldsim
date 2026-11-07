// As top.sv, the last leaf's output also back into the first leaf's state
// (e) in the same cycle: the first leaf's output is needed before its state
// pass, so the leaves' outputs are written at ready passes.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] v [4];
  Leaf l0(.clk, .d(x), .e(v[3]), .q(v[0]));
  for (genvar i = 1; i < 4; i++) begin : g
    Leaf l(.clk, .d(v[i - 1]), .e(8'h0), .q(v[i]));
  end
  assign y = v[0] ^ v[3];
endmodule
