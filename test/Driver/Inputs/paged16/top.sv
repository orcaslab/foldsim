// Sixteen leaves in a chain (Inputs/paged's Leaf), for streamed PP rows of a
// few instances behind a small scratch: the streamed instances' passes lie
// far apart in the frame (pp-beat-bits: each row's beats and drains fit
// between them). No reset: run with the registers and RAM rows zero at start
// (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd, r, ob;
  logic [159:0] acc;
  logic [12:0] cnt;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d ^ cnt[7:0];
    rd <= ram[d[3:0]];
    r <= d ^ ob;
    if (d[1]) acc <= {acc[151:0], d ^ rd} ^ {8'b0, acc[159:8]};
    cnt <= cnt + {5'b0, d};
    ob <= acc[7:0] ^ acc[159:152] ^ acc[87:80] ^ cnt[12:5];
  end
  assign q = rd ^ d ^ r;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] w [17];
  assign w[0] = x;
  Leaf l0(.clk, .d(w[0]), .q(w[1]));
  Leaf l1(.clk, .d(w[1]), .q(w[2]));
  Leaf l2(.clk, .d(w[2]), .q(w[3]));
  Leaf l3(.clk, .d(w[3]), .q(w[4]));
  Leaf l4(.clk, .d(w[4]), .q(w[5]));
  Leaf l5(.clk, .d(w[5]), .q(w[6]));
  Leaf l6(.clk, .d(w[6]), .q(w[7]));
  Leaf l7(.clk, .d(w[7]), .q(w[8]));
  Leaf l8(.clk, .d(w[8]), .q(w[9]));
  Leaf l9(.clk, .d(w[9]), .q(w[10]));
  Leaf l10(.clk, .d(w[10]), .q(w[11]));
  Leaf l11(.clk, .d(w[11]), .q(w[12]));
  Leaf l12(.clk, .d(w[12]), .q(w[13]));
  Leaf l13(.clk, .d(w[13]), .q(w[14]));
  Leaf l14(.clk, .d(w[14]), .q(w[15]));
  Leaf l15(.clk, .d(w[15]), .q(w[16]));
  assign y = w[16];
endmodule
