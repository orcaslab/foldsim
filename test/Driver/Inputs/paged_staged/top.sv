// Four leaves in a chain, for staged and multicycle evaluation of paged
// register state kept off chip: each leaf's outputs are registers (no late
// output, so a leaf may be staged), its RAM of 16 rows is read through a
// register only (so it may be multicycle), and its next state hangs off a
// multiply deep enough for stages. Its registers page: a 160-bit shift-xor
// written only when bit 1 of its input or of the accumulator is set (cut
// across pages and columns), a 13-bit sum, the accumulator and the output.
// No reset: run with the registers and RAM rows zero at start
// (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd, acc;
  logic [159:0] wide;
  logic [12:0] cnt;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d ^ acc;
    rd <= ram[d[3:0] ^ acc[7:4]];
    acc <= (acc * 8'd29 + rd) ^ d ^ wide[7:0];
    if (d[1] | acc[0]) wide <= {wide[151:0], d ^ rd ^ acc} ^ {8'b0, wide[159:8]};
    cnt <= cnt + {5'b0, d} + {5'b0, acc * 8'd13};
    q <= acc ^ rd ^ wide[87:80] ^ cnt[12:5];
  end
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
