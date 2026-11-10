// 128 leaves of Driver/Inputs/paged in a chain (state-scratch-fixed: the same
// scratch RAM at 64 and 128 instances).
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] w [129];
  assign w[0] = x;
  for (genvar i = 0; i < 128; i++) begin : g
    Leaf l(.clk, .d(w[i]), .q(w[i + 1]));
  end
  assign y = w[128];
endmodule
