// A queue whose head register is in a class of its own (Ptr, kept as an
// instance by a deeper fold): the queue's RAM asks for block RAM and is read
// combinationally at a function of that head, so its read can be made
// synchronous only with the head's registers beside it. The memory's write
// is at the complement of the tail register (bit 0 of the input enqueues it,
// bit 1 dequeues); the head and the tail meet often.
module Ptr(input logic clk, input logic [7:0] d, output logic [3:0] head);
  always_ff @(posedge clk)
    if (d[1]) head <= head + {3'd0, d[2]} + 4'd1;
endmodule
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  (* ram_style = "block" *) logic [7:0] ram [16];
  logic [3:0] head, tail;
  Ptr p(.clk, .d, .head);
  always_ff @(posedge clk)
    if (d[0]) begin ram[~tail] <= d; tail <= tail + 4'd1; end
  assign q = ram[~head] ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
