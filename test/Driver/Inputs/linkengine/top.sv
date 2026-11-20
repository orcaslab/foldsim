// Link engines: a reader class fed by two producer classes. Sixteen units,
// sixteen vnits and sixteen readers in a ring (as linkhbm): each reader reads
// a unit's 32-bit register whole, a byte of a second unit and 24 bits of a
// vnit (three rows a pass from two producer classes: one engine's three
// landings), and each unit and vnit reads a reader's register back (an engine
// each). Early links of the producers' entries; a frame long enough for a
// request to precede its landing by more than the decisions in flight.
module Unit(input logic clk, input logic rst, input logic [7:0] seed,
            input logic [31:0] in, output logic [31:0] out);
  always_ff @(posedge clk)
    if (rst) out <= {4{seed}};
    else out <= {out[30:0], out[31]} ^ in ^ {24'b0, seed};
endmodule

module Vnit(input logic clk, input logic rst, input logic [7:0] seed,
            input logic [15:0] in, output logic [23:0] out);
  always_ff @(posedge clk)
    if (rst) out <= {3{seed ^ 8'h5a}};
    else out <= {out[22:0], out[23] ^ out[7]} + {8'b0, in};
endmodule

module Reader(input logic clk, input logic rst, input logic [7:0] x,
              input logic [31:0] d, input logic [7:0] e, input logic [23:0] f,
              output logic [31:0] q);
  always_ff @(posedge clk)
    if (rst) q <= 0;
    else q <= q + d + {24'b0, x ^ e} + {f, 8'b0};
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [15:0][31:0] o, q;
  logic [15:0][23:0] v;
  for (genvar i = 0; i < 16; ++i) begin : g
    Unit u(.clk, .rst, .seed(8'(i * 37 + 1)), .in(q[(i + 1) % 16]), .out(o[i]));
    Vnit w(.clk, .rst, .seed(8'(i * 11 + 3)), .in(q[(i + 3) % 16][15:0]), .out(v[i]));
    Reader r(.clk, .rst, .x(x ^ 8'(i)), .d(o[i]), .e(o[(i + 5) % 16][15:8]),
             .f(v[(i + 7) % 16]), .q(q[i]));
  end
  assign y = q[0][7:0] ^ q[5][15:8] ^ q[10][23:16] ^ q[15][31:24] ^ o[3][7:0] ^ v[9][23:16];
endmodule
