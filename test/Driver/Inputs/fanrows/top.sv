// fanrows: each reader instance takes its input from another port of its
// producer instance (reader (j, o) reads producer j's port o): the reader
// input has a source per producer port, each taken by some of the reader
// instances (a switch's per-output snapshot read by each output's arbiter).
module Prod (input clk, input rst, input [7:0] x, input [7:0] K,
    output [7:0] o0, output [7:0] o1, output [7:0] o2, output [7:0] o3);
  logic [7:0] s0, s1, s2, s3;
  always_ff @(posedge clk)
    if (rst) begin s0 <= 0; s1 <= 0; s2 <= 0; s3 <= 0; end
    else begin
      s0 <= s0 + (x ^ K); s1 <= s1 ^ (x + 8'd37); s2 <= s2 + (s0 ^ 8'd91); s3 <= {s3[6:0], s3[7]} ^ x ^ K;
    end
  assign o0 = s0; assign o1 = s1; assign o2 = s2; assign o3 = s3;
endmodule
module Rd (input clk, input rst, input [7:0] a, input [7:0] b, output [7:0] y);
  logic [7:0] acc;
  always_ff @(posedge clk) acc <= rst ? 8'd0 : (acc * 8'd3) + (a ^ {b[3:0], b[7:4]});
  assign y = acc;
endmodule
module Top (input clk, input rst, input [7:0] x, output [7:0] y);
  wire [7:0] p [2][4];
  Prod p0 (.clk, .rst, .x, .K(8'd5), .o0(p[0][0]), .o1(p[0][1]), .o2(p[0][2]), .o3(p[0][3]));
  Prod p1 (.clk, .rst, .x, .K(8'd9), .o0(p[1][0]), .o1(p[1][1]), .o2(p[1][2]), .o3(p[1][3]));
  wire [7:0] r [2][4];
  for (genvar j = 0; j < 2; ++j) begin : g_j
    for (genvar o = 0; o < 4; ++o) begin : g_o
      Rd rd (.clk, .rst, .a(p[j][o]), .b(p[j][(o + 1) % 4]), .y(r[j][o]));
    end
  end
  assign y = r[0][0] ^ r[0][1] ^ r[0][2] ^ r[0][3] ^ r[1][0] ^ r[1][1] ^ r[1][2] ^ r[1][3];
endmodule
