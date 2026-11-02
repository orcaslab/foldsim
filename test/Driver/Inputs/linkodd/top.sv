// HBM link streams of a width not whole bytes (linkhbm with 29-bit words):
// sixteen units and sixteen readers in a ring, each reader reading its unit's
// word whole and a byte of a second unit's, so a stream's row is 29 + 8 bits
// or so, carried by a page engine beside the models' off-chip page columns.
module Unit(input logic clk, input logic rst, input logic [7:0] seed,
            input logic [28:0] in, output logic [28:0] out);
  always_ff @(posedge clk)
    if (rst) out <= {seed[4:0], {3{seed}}};
    else out <= {out[27:0], out[28]} ^ in ^ {21'b0, seed};
endmodule

module Reader(input logic clk, input logic rst, input logic [7:0] x,
              input logic [28:0] d, input logic [7:0] e, output logic [28:0] q);
  always_ff @(posedge clk)
    if (rst) q <= 0;
    else q <= q + d + {21'b0, x ^ e};
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [15:0][28:0] o, q;
  for (genvar i = 0; i < 16; ++i) begin : g
    Unit u(.clk, .rst, .seed(8'(i * 37 + 1)), .in(q[(i + 1) % 16]), .out(o[i]));
    Reader r(.clk, .rst, .x(x ^ 8'(i)), .d(o[i]), .e(o[(i + 5) % 16][15:8]), .q(q[i]));
  end
  assign y = q[0][7:0] ^ q[5][15:8] ^ q[10][23:16] ^ {3'b0, q[15][28:24]} ^ o[3][7:0];
endmodule
