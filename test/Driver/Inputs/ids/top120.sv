// 120 leaves in a chain, 12 parents of 10: leaf i's id is its parent's
// number XOR a child's constant (a period of 10, not a power of two).
module Top(input logic clk, reset, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] w [121];
  assign w[0] = x;
  for (genvar i = 0; i < 120; i++) begin : g
    Leaf l(.clk, .reset, .id(16'((i / 10) ^ (((i % 10) * 40503 + 977) & 16'hffff))), .d(w[i]), .q(w[i + 1]));
  end
  assign y = w[120];
endmodule
