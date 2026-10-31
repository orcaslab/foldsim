// Four leaves in a chain, each a queue of 16 bytes as IceNIC's buffers are
// built: the data read combinationally at a function of the head register
// (its complement, as `~1 - head`; a read at the register itself imports as a
// registered read already), written at the complement of the tail register
// (bit 0 of its input enqueues the input, bit 1 dequeues). The head and the
// tail meet often, so a row is read in the cycle it is written. Its output
// mixes the head's data with its input.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [3:0] head, tail;
  always_ff @(posedge clk) begin
    if (d[0]) begin ram[~tail] <= d; tail <= tail + 4'd1; end
    if (d[1]) head <= head + {3'd0, d[2]} + 4'd1;
  end
  assign q = ram[~head] ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
