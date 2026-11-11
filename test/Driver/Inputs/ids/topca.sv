// 120 leaves in a chain in two runs of parents: 6 parents of 10, then 10 of
// 6. Leaf i's id is a child's constant plus its parent's number times a
// slope that depends on the child (3 or 12, then 5 or 3): no run of segments
// and no one F(parent) op G(child) gives it, two column-affine pieces do.
module Top(input logic clk, reset, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] w [121];
  assign w[0] = x;
  for (genvar i = 0; i < 120; i++) begin : g
    localparam int J = i < 60 ? i : i - 60;
    localparam int D = i < 60 ? 10 : 6;
    localparam int S = i < 60 ? ((J % D) < 5 ? 3 : 12) : ((J % D) < 3 ? 5 : 3);
    localparam int A = i < 60 ? (J % D) * 40503 + 977 : (J % D) * 23456 + 311;
    Leaf l(.clk, .reset, .id(16'((J / D) * S + A)), .d(w[i]), .q(w[i + 1]));
  end
  assign y = w[120];
endmodule
