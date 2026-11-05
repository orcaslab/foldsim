// Four leaves in a chain, each with a RAM of 16 rows read through a register
// and registered outputs only (no late output), so that a leaf may be staged:
// a leaf writes its input mixed with its accumulator at the row of the
// input's high nibble when bit 0 is set, reads the row its low nibble and the
// accumulator name, and folds the read into the accumulator (a multiply deep
// enough for stages).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd, acc;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d ^ acc;
    rd <= ram[d[3:0] ^ acc[7:4]];
    acc <= (acc * 8'd29 + rd) ^ d;
    q <= acc ^ rd;
  end
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
