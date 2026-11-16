// Sixteen leaves in a chain (room-regions' dense case): each writes its input at the row of its high nibble every
// target cycle and reads the row of its low nibble, so a frame of the folded leaf issues a read and a write a host
// cycle, back to back: its controller's queue sees the most requests the room terms allow.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd;
  always_ff @(posedge clk) begin
    ram[d[7:4]] <= d;
    rd <= ram[d[3:0]];
  end
  assign q = rd ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] s [17];
  assign s[0] = x;
  Leaf l0(.clk, .d(s[0]), .q(s[1]));
  Leaf l1(.clk, .d(s[1]), .q(s[2]));
  Leaf l2(.clk, .d(s[2]), .q(s[3]));
  Leaf l3(.clk, .d(s[3]), .q(s[4]));
  Leaf l4(.clk, .d(s[4]), .q(s[5]));
  Leaf l5(.clk, .d(s[5]), .q(s[6]));
  Leaf l6(.clk, .d(s[6]), .q(s[7]));
  Leaf l7(.clk, .d(s[7]), .q(s[8]));
  Leaf l8(.clk, .d(s[8]), .q(s[9]));
  Leaf l9(.clk, .d(s[9]), .q(s[10]));
  Leaf l10(.clk, .d(s[10]), .q(s[11]));
  Leaf l11(.clk, .d(s[11]), .q(s[12]));
  Leaf l12(.clk, .d(s[12]), .q(s[13]));
  Leaf l13(.clk, .d(s[13]), .q(s[14]));
  Leaf l14(.clk, .d(s[14]), .q(s[15]));
  Leaf l15(.clk, .d(s[15]), .q(s[16]));
  assign y = s[16];
endmodule
