// As linkhbm, but each unit's output also passes its input through (late: its
// readers take it in the frame it is written), so no HBM link stream may carry
// it: a stream lands its rows a frame on.
module Unit(input logic clk, input logic rst, input logic [7:0] seed,
            input logic [31:0] in, output logic [31:0] out);
  logic [31:0] r;
  always_ff @(posedge clk)
    if (rst) r <= {4{seed}};
    else r <= {r[30:0], r[31]} ^ in ^ {24'b0, seed};
  assign out = r ^ {in[15:0], in[31:16]};
endmodule

module Reader(input logic clk, input logic rst, input logic [7:0] x,
              input logic [31:0] d, input logic [7:0] e, output logic [31:0] q);
  always_ff @(posedge clk)
    if (rst) q <= 0;
    else q <= q + d + {24'b0, x ^ e};
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [15:0][31:0] o, q;
  for (genvar i = 0; i < 16; ++i) begin : g
    Unit u(.clk, .rst, .seed(8'(i * 37 + 1)), .in(q[(i + 1) % 16]), .out(o[i]));
    Reader r(.clk, .rst, .x(x ^ 8'(i)), .d(o[i]), .e(o[(i + 5) % 16][15:8]), .q(q[i]));
  end
  assign y = q[0][7:0] ^ q[5][15:8] ^ q[10][23:16] ^ q[15][31:24] ^ o[3][7:0];
endmodule
