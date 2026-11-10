// A found-flag priority loop over eight Leaf lanes in the unfolded top (for
// foldsim-lane-scan's prefix stages): found_j = found_{j-1} | onehot_j,
// onehot_j = req_j & ~found_{j-1}, each lane's grant read back by its Leaf.
module Leaf(input logic clk, input logic rst, input logic [7:0] x, input logic [7:0] K, input logic gnt,
            output logic req);
  logic [7:0] s;
  always_ff @(posedge clk)
    if (rst) s <= K;
    else s <= (s * 8'd5 + x + K) ^ {7'd0, gnt};
  assign req = s[0] ^ s[3];
endmodule
module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  localparam int N = 8;
  logic req [N], gnt [N];
  for (genvar j = 0; j < N; ++j) begin : g
    Leaf u(.clk, .rst, .x(x ^ 8'(j)), .K(8'(3 * j + 1)), .gnt(gnt[j]), .req(req[j]));
  end
  logic found;
  logic [N-1:0] onehot;
  always_comb begin
    found = 1'b0;
    onehot = '0;
    for (int i = 0; i < N; ++i)
      if (req[i] && !found) begin
        onehot[i] = 1'b1;
        found = 1'b1;
      end
  end
  for (genvar j = 0; j < N; ++j) begin : g2
    assign gnt[j] = onehot[j];
  end
  assign y = {7'd0, found} ^ onehot;
endmodule
