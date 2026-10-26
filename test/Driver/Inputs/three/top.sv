module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(y));
endmodule
