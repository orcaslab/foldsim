// As top.sv, with a read-first RAM: the read data is registered, read at the
// held address (bit 1 of the input loads it), so a write of the read row in
// the same cycle is seen only by the next read.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [4];
  logic [1:0] raddr_q;
  logic [7:0] rd;
  wire [1:0] raddr = d[1] ? d[3:2] : raddr_q;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:6]] <= d;
    raddr_q <= raddr;
    rd <= ram[raddr];
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
