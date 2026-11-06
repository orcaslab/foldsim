// The paged Leaf of Driver/Inputs/paged (a late output, a ready pass).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd, r, ob;
  logic [159:0] acc;
  logic [12:0] cnt;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d ^ cnt[7:0];
    rd <= ram[d[3:0]];
    r <= d ^ ob;
    if (d[1]) acc <= {acc[151:0], d ^ rd} ^ {8'b0, acc[159:8]};
    cnt <= cnt + {5'b0, d};
    ob <= acc[7:0] ^ acc[159:152] ^ acc[87:80] ^ cnt[12:5];
  end
  assign q = rd ^ d ^ r;
endmodule
// A chain of 40 Mid stages (each output a function of its input in the same
// cycle), eight Leafs fed from points along it, the output registered. Under
// --modulo the chain's later stages run in the next frame's slots, so a copy
// of interleaved Mid instances (one early, one late) has state passes more
// than a frame of absolute positions apart.
module Mid(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] s;
  always_ff @(posedge clk) s <= s + d;
  assign q = d ^ s;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] m [41];
  logic [7:0] o [8];
  assign m[0] = x;
  for (genvar k = 0; k < 40; k++) begin : g_mid
    Mid u(.clk, .d(m[k]), .q(m[k + 1]));
  end
  for (genvar i = 0; i < 8; i++) begin : g_leaf
    Leaf l(.clk, .d(m[5 * i]), .q(o[i]));
  end
  always_ff @(posedge clk) y <= o[0] ^ o[1] ^ o[2] ^ o[3] ^ o[4] ^ o[5] ^ o[6] ^ o[7];
endmodule
