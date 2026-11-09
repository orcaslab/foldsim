// Four leaves in a chain, for state in RAM with its memories' rows off chip
// (static-state-in-ram-rows.test): paged/top.sv with the RAM's registered
// read off the output's path (the output from a register `r` and the input,
// late), so the RAM pages as rows (no ready pass reads it); the other
// registers (a 160-bit shift-xor under an enable, a 13-bit sum, a byte mixing
// both) are state in RAM on chip. No reset: run with the registers and RAM
// rows zero at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd, r, ob;
  logic [159:0] acc;
  logic [12:0] cnt;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d ^ cnt[7:0];
    rd <= ram[d[3:0]];
    r <= d ^ ob ^ rd;
    if (d[1]) acc <= {acc[151:0], d ^ rd} ^ {8'b0, acc[159:8]};
    cnt <= cnt + {5'b0, d};
    ob <= acc[7:0] ^ acc[159:152] ^ acc[87:80] ^ cnt[12:5];
  end
  assign q = d ^ r;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
