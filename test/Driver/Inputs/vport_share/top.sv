// Two kinds of leaves in a chain (two classes), each with a read-first target
// RAM: Narrow's 16 x 8, Wide's 8 x 600 (two beats a row). Their passes fall at
// different positions: one static port can carry both (vport-share widens the
// narrow one's rows), where first fit gives the wide RAM a port of its own.
module Narrow(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] mem [16];
  logic [7:0] r;
  wire [3:0] a = d[3:0] ^ {d[7], 3'd0};
  always_ff @(posedge clk) begin
    if (d[0] | d[5]) r <= mem[a];
    if (d[1] | d[6]) mem[a] <= d ^ 8'h3c;
  end
  assign q = r ^ d;
endmodule
module Wide(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [599:0] wide [8];
  logic [599:0] w;
  always_ff @(posedge clk) begin
    if (d[2] | d[4]) w <= wide[d[2:0]];
    if (d[3]) wide[d[2:0]] <= {d, {74{d ^ 8'h5a}}};
  end
  assign q = w[7:0] ^ w[599:592] ^ (w[519:512] << 1) ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Narrow n0(.clk, .d(x), .q(a));
  Wide w0(.clk, .d(a), .q(b));
  Narrow n1(.clk, .d(b), .q(c));
  Wide w1(.clk, .d(c), .q(y));
endmodule
