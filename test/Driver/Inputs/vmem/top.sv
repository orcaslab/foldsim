// Four leaves in a chain, each with a RAM of 16 rows read through a register:
// a leaf writes its input at the row of its high nibble when bit 0 is set, and
// reads the row of its low nibble; its output mixes the read with its input.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d;
    rd <= ram[d[3:0]];
  end
  assign q = rd ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
