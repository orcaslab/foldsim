// HBM link streams sharing page engines (--link-hbm-shift): sixteen of each
// of Unit, Vnit, Reader and Sink in a ring. Reader reads a Unit's and a
// Vnit's register whole (two streams into one reader class: their landings,
// and so their requests, at the same positions), Sink reads another Unit's
// (a second stream from Unit: its writes at the same positions as
// Unit -> Reader's); every link early, so each may be streamed.
module Unit(input logic clk, input logic rst, input logic [7:0] seed,
            input logic [31:0] in, output logic [31:0] out);
  always_ff @(posedge clk)
    if (rst) out <= {4{seed}};
    else out <= {out[30:0], out[31]} ^ in ^ {24'b0, seed};
endmodule

module Vnit(input logic clk, input logic rst, input logic [7:0] seed,
            input logic [31:0] in, output logic [31:0] out);
  always_ff @(posedge clk)
    if (rst) out <= {seed, ~seed, seed, ~seed};
    else out <= {out[23:0], out[31:24]} + (in ^ {seed, 24'b0});
endmodule

module Reader(input logic clk, input logic rst, input logic [7:0] x,
              input logic [31:0] d, input logic [31:0] e, output logic [31:0] q);
  always_ff @(posedge clk)
    if (rst) q <= 0;
    else q <= q + (d ^ e) + {24'b0, x};
endmodule

module Sink(input logic clk, input logic rst, input logic [7:0] x,
            input logic [31:0] d, output logic [31:0] s);
  always_ff @(posedge clk)
    if (rst) s <= 1;
    else s <= (s ^ d) + {x, 24'b0} + 1;
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [15:0][31:0] o, p, q, s;
  for (genvar i = 0; i < 16; ++i) begin : g
    Unit u(.clk, .rst, .seed(8'(i * 37 + 1)), .in(s[(i + 3) % 16]), .out(o[i]));
    Vnit v(.clk, .rst, .seed(8'(i * 11 + 5)), .in(q[(i + 1) % 16]), .out(p[i]));
    Reader r(.clk, .rst, .x(x ^ 8'(i)), .d(o[i]), .e(p[(i + 5) % 16]), .q(q[i]));
    Sink k(.clk, .rst, .x(x + 8'(i)), .d(o[(i + 7) % 16]), .s(s[i]));
  end
  assign y = q[0][7:0] ^ q[5][15:8] ^ q[10][23:16] ^ s[15][31:24] ^ p[3][7:0];
endmodule
