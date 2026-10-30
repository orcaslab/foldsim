// Four leaves in a chain on one copy: each with a wide register, a narrow
// one and a memory with a registered read. With LATE the leaf's output is the
// read data (a ready pass: a state gap of 1), else a register of it (no ready
// pass: a state gap of at least 4).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [79:0] wide;
  logic [7:0] narrow;
  logic [7:0] mem [0:15];
  logic [3:0] waddr;
  logic [7:0] rdata, held;
  always_ff @(posedge clk) begin
    wide <= {wide[71:0], d ^ wide[79:72]};
    narrow <= narrow + d;
    mem[waddr] <= d ^ wide[7:0];
    waddr <= waddr + 4'd1;
    rdata <= mem[waddr - 4'd3];
    held <= rdata;
  end
`ifdef LATE
  assign q = rdata ^ narrow ^ wide[47:40];
`else
  assign q = held ^ narrow ^ wide[47:40];
`endif
endmodule

module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
