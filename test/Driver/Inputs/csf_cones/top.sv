// Ready-cone lanes (csf-ready-cones): four leaves in a chain of late links.
// Each leaf has a late output of logic alone (q: its input and registers; it
// moves to the cone model), a late output through a combinational memory read
// (m: it stays with the leaf, whose ready pass keeps it), and an early output
// (e: a register). No reset: run with the registers and RAM rows zero at
// start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q,
            output logic [7:0] m, output logic [7:0] e);
  logic [7:0] r, s;
  logic [15:0] w;
  logic [7:0] mem [16];
  always_ff @(posedge clk) begin
    r <= d ^ s;
    s <= s + r + {7'b0, d[7]};
    w <= {w[14:0], w[15] ^ d[3]} ^ {8'b0, r};
    if (d[0]) mem[d[7:4]] <= r ^ w[7:0];
  end
  assign q = (d + r) ^ s ^ w[15:8] ^ (d[2] ? w[7:0] : 8'h3c);
  assign m = mem[d[3:0]] ^ d;
  assign e = r ^ w[11:4];
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] q0, q1, q2, q3, m0, m1, m2, m3, e0, e1, e2, e3;
  Leaf l0(.clk, .d(x), .q(q0), .m(m0), .e(e0));
  Leaf l1(.clk, .d(q0 ^ x ^ e3), .q(q1), .m(m1), .e(e1));
  Leaf l2(.clk, .d(q1 ^ m0), .q(q2), .m(m2), .e(e2));
  Leaf l3(.clk, .d(q2 ^ e0), .q(q3), .m(m3), .e(e3));
  assign y = q3 ^ m2 ^ m3 ^ e1 ^ m1 ^ e2;
endmodule
