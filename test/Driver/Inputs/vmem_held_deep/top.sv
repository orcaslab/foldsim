// Four leaves in a chain (leaf.sv, a held read): each leaf's late output feeds the next
// leaf's input in the same cycle, so each is written at its state pass and
// read by the next leaf's state pass within the frame.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] v [4];
  Leaf l0(.clk, .d(x), .e(x ^ 8'h5a), .q(v[0]));
  for (genvar i = 1; i < 4; i++) begin : g
    Leaf l(.clk, .d(v[i - 1]), .e(v[i - 1] ^ x), .q(v[i]));
  end
  assign y = v[3];
endmodule
