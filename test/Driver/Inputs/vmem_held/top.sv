// Four leaves in a chain, each with a RAM of 4 rows read through an address
// register that loads only when bit 1 of its input is set (a registered read
// address with an enable, as SRAM macros and `if (en) raddr_q <= raddr` are):
// while it holds, the leaf keeps reading the same row, which its own writes
// (bit 0 set: its input at the row of bits 7:6) often change. Its output
// mixes the read with its input.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [4];
  logic [1:0] raddr_q;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:6]] <= d;
    if (d[1]) raddr_q <= d[3:2];
  end
  assign q = ram[raddr_q] ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
