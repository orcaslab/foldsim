// 128 leaves in a chain, 8 parents of 16: leaf i's id is its parent's
// number (i / 16, shifted) XOR a child's constant (a power-of-two period).
module Top(input logic clk, reset, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] w [129];
  assign w[0] = x;
  for (genvar i = 0; i < 128; i++) begin : g
    Leaf l(.clk, .reset, .id(16'(((i / 16) << 4) ^ (((i % 16) * 40503 + 977) & 16'hffff))), .d(w[i]), .q(w[i + 1]));
  end
  assign y = w[128];
endmodule
