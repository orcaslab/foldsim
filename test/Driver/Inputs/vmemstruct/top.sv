// The vmem chain (Driver/Inputs/vmem) with each leaf's output a packed
// struct: a link between two copies in different SLR groups carries an
// aggregate output, read through a bitcast.
typedef struct packed {
  logic [3:0] hi;
  logic [3:0] lo;
} pair_t;
module Leaf(input logic clk, input logic [7:0] d, output pair_t q);
  logic [7:0] ram [16];
  logic [7:0] rd;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d;
    rd <= ram[d[3:0]];
  end
  assign q.hi = rd[7:4] ^ d[7:4];
  assign q.lo = rd[3:0] ^ d[3:0];
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  pair_t a, b, c, e;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(e));
  assign y = e;
endmodule
