// A state-in-ram-pp class whose rows leave registers out (state-in-ram-pp-pack): per leaf, a 160-bit register on
// an enable (in the rows), a 48-bit product accumulated on another enable and a 32-bit product every cycle (deep
// next values: stored, packed into an instance's row of 72-bit columns), and a byte mixing them. Four leaves in
// a chain (a late output each). No reset: run with the registers zero at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [159:0] acc;
  logic [47:0] p1;
  logic [31:0] p2;
  logic [7:0] ob;
  always_ff @(posedge clk) begin
    if (d[1]) acc <= {acc[151:0], d ^ ob} ^ {8'b0, acc[159:8]};
    if (d[2]) p1 <= p1 * {40'h0, d} + 48'h9e3779b97f4a + {acc[39:0], d};
    p2 <= p2 * 32'h2c1b3c6d + {24'h0, d ^ p1[15:8]};
    ob <= acc[7:0] ^ p1[47:40] ^ p2[31:24] ^ p2[7:0];
  end
  assign q = d ^ ob;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
