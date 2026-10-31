// Four leaves in a chain, for paged register state: each leaf's output
// depends on its input in the same cycle (late), through a register `r` (so a
// ready pass reads it: it keeps its store) and the registered read of a RAM
// of 16 rows; its other registers page: a 40-bit shift-xor written only when
// bit 1 of its input is set (wider than a page at p >= 2, so cut across
// pages), a 13-bit sum and a byte mixing both. No reset: run with the
// registers and RAM rows zero at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd, r, ob;
  logic [39:0] acc;
  logic [12:0] cnt;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d ^ cnt[7:0];
    rd <= ram[d[3:0]];
    r <= d ^ ob;
    if (d[1]) acc <= {acc[31:0], d ^ rd} ^ {8'b0, acc[39:8]};
    cnt <= cnt + {5'b0, d};
    ob <= acc[7:0] ^ acc[39:32] ^ cnt[12:5];
  end
  assign q = rd ^ d ^ r;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
