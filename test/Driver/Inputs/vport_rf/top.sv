// Four leaves in a chain, each with read-first target RAMs: a registered
// read and a write in the same clock, often of the same row (the read takes
// the row before the write: a read-first pair), one 16 x 8, one 8 x 600 (two
// beats a row).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] mem [16];
  logic [7:0] r;
  logic [599:0] wide [8];
  logic [599:0] w;
  wire [3:0] a = d[3:0] ^ {d[7], 3'd0};
  always_ff @(posedge clk) begin
    if (d[0] | d[5]) r <= mem[a];
    if (d[1] | d[6]) mem[a] <= d ^ 8'h3c;
    if (d[2] | d[4]) w <= wide[d[2:0]];
    if (d[3]) wide[d[2:0]] <= {d, {74{d ^ 8'h5a}}};
  end
  assign q = r ^ w[7:0] ^ w[599:592] ^ (w[519:512] << 1) ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
