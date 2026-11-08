// A leaf whose output is late and deep: a RAM row read combinationally at an
// address from its input and state, through two multiplies, mixed with its
// input (no register between). Its input e reaches only its state. No reset:
// run with the registers and RAM rows zero at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, input logic [7:0] e,
            output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] acc, bcc, m, r;
  always_comb begin
    m = ram[d[3:0] ^ acc[3:0]];
    r = ((m * 8'd29) + d) ^ ((acc * 8'd13) + bcc);
  end
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= r ^ e;
    acc <= acc + r + e;
    bcc <= (bcc * 8'd7) ^ d ^ e;
  end
  assign q = (r * 8'd11) ^ bcc;
endmodule
