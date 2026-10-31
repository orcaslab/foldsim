// Four leaves in a chain, each with memories read combinationally, the
// kind paged register state pages as rows (page-memory-bits): a 4-entry
// queue (its head read without a register, as a Chisel queue's), and a
// register file of 8 rows of 8 bits with two combinational read ports and a
// write port. The output is a register. No reset: run with the registers and
// rows zero at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] fifo [4];
  logic [7:0] rf [8];
  logic [1:0] wp, rp;
  logic [2:0] cnt;
  logic [7:0] r;
  wire push = d[0] && cnt != 3'd4;
  wire pop = d[1] && cnt != 3'd0;
  wire [7:0] head = fifo[rp];
  always_ff @(posedge clk) begin
    if (push) fifo[wp] <= d ^ 8'h5a;
    wp <= wp + {1'b0, push};
    rp <= rp + {1'b0, pop};
    cnt <= cnt + {2'b0, push} - {2'b0, pop};
    if (d[2]) rf[d[5:3]] <= head + d;
    r <= rf[d[7:5]] ^ rf[d[4:2]] ^ head;
  end
  assign q = r;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a ^ x), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c ^ a), .q(y));
endmodule
