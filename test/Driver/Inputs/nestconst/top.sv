// Four clusters of four leaves, each leaf's identity the cluster's input
// (a constant of the instance) plus its place: an early output of the
// cluster that is a constant of each instance. A leaf folds its identity
// into a word kept without reset from the first cycle (a memory written in
// reset would keep it too), so a first frame that read the identity
// unwritten would differ for good. y shows the leaf x[3:0] names.
module Leaf(input logic clk, input logic rst, input logic [7:0] id, output logic [15:0] pub);
  logic [7:0] seen;
  always_ff @(posedge clk) begin
    seen <= {seen[6:0], seen[7]} ^ id;
    if (rst) pub <= {id, ~id};
    else pub <= {pub[14:0], pub[15]} ^ {seen, id};
  end
endmodule
module Cluster(input logic clk, input logic rst, input logic [7:0] base, output logic [63:0] words);
  for (genvar i = 0; i < 4; ++i) begin : l
    Leaf lf(.clk, .rst, .id(base + 8'(i)), .pub(words[16 * i +: 16]));
  end
endmodule
module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [63:0] w [4];
  for (genvar g = 0; g < 4; ++g) begin : c
    Cluster cl(.clk, .rst, .base(8'(16 * g + 1)), .words(w[g]));
  end
  assign y = w[x[3:2]][16 * x[1:0] +: 8];
endmodule
