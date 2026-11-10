// 32 leaves of Driver/Inputs/paged in a chain (state-scratch-fixed: the same
// scratch RAM at 32 and 64 instances).
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] w [33];
  assign w[0] = x;
  for (genvar i = 0; i < 32; i++) begin : g
    Leaf l(.clk, .d(w[i]), .q(w[i + 1]));
  end
  assign y = w[32];
endmodule
