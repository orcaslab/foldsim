// Sixteen leaves of Driver/Inputs/vmem in a chain: a class of 16 instances,
// so that a ring of fewer entries than instances is shared (vmem-rings); a
// leaf reads only when bit 1 is set, else its read register holds.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d;
    if (d[1]) rd <= ram[d[3:0]];
  end
  assign q = rd ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] w [17];
  assign w[0] = x;
  for (genvar i = 0; i < 16; i++) begin : g
    Leaf l(.clk, .d(w[i]), .q(w[i + 1]));
  end
  assign y = w[16];
endmodule
