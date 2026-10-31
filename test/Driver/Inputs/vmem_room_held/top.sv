// vmem_room's eight leaves with a registered read address that loads only
// when bit 0 of the input is set (--vmem-skip-held-reads applies), read first,
// and every cycle a write of the row read: no read is ever skipped, so every
// state pass issues a read and a write, a host cycle after the pass.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [3:0] raddr_q;
  logic [7:0] rd;
  wire [3:0] raddr = d[0] ? d[7:4] : raddr_q;
  always_ff @(posedge clk) begin
    ram[raddr] <= d;
    raddr_q <= raddr;
    rd <= ram[raddr];
  end
  assign q = rd ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] q [8];
  for (genvar i = 0; i < 8; ++i) begin : g_leaf
    Leaf l(.clk, .d(x ^ 8'(i * 37)), .q(q[i]));
  end
  assign y = q[0] ^ q[1] ^ q[2] ^ q[3] ^ q[4] ^ q[5] ^ q[6] ^ (q[7] + 8'd1);
endmodule
