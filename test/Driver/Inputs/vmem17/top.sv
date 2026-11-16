// Two leaves in a chain, each with a RAM of 2^17 rows (a 17-bit row field: wider than S28 A2's fixed 12 bits): a
// leaf writes its input at row {d[7:3], 9'b0, d[2:0]} when bit 0 is set and reads row {d[6:2], 9'b0, d[3:1]} (rows
// apart in their high bits: a 12-bit row field would alias them); its output mixes the read with its input.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [131072];
  logic [7:0] rd;
  always_ff @(posedge clk) begin
    if (d[0]) ram[{d[7:3], 9'b0, d[2:0]}] <= d;
    rd <= ram[{d[6:2], 9'b0, d[3:1]}];
  end
  assign q = rd ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(y));
endmodule
