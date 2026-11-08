// A leaf with registers whose next value is another register's (a shift
// register with an enable, written in both orders) beside a deep cone, so
// that the staged copy replaces registers that are other registers' write
// data. No reset: run with the registers zero at start.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] s0, s1, s2, s3, acc;
  always_ff @(posedge clk) begin
    if (d[0]) s3 <= s2;
    if (d[1]) s2 <= s1;
    s1 <= s0;
    s0 <= d ^ acc;
    acc <= (acc * 8'd29 + s3) ^ ((s1 * 8'd13) + d);
    q <= acc ^ s3;
  end
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a ^ x), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
