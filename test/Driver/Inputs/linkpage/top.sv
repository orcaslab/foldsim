// Paged links: four units and four readers in a ring, each holding a 32-bit
// register that the other side's instance reads whole: the only source of
// that input, an early link of four entries (the producer's instances), so
// block or ultra RAM may hold it and `link-kinds` may page it.
module Unit(input logic clk, input logic rst, input logic [7:0] seed,
            input logic [31:0] in, output logic [31:0] out);
  always_ff @(posedge clk)
    if (rst) out <= {4{seed}};
    else out <= {out[30:0], out[31]} ^ in ^ {24'b0, seed};
endmodule

module Reader(input logic clk, input logic rst, input logic [7:0] x,
              input logic [31:0] d, output logic [31:0] q);
  always_ff @(posedge clk)
    if (rst) q <= 0;
    else q <= q + d + {24'b0, x};
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [3:0][31:0] o, q;
  for (genvar i = 0; i < 4; ++i) begin : g
    Unit u(.clk, .rst, .seed(8'(i * 37 + 1)), .in(q[(i + 1) % 4]), .out(o[i]));
    Reader r(.clk, .rst, .x(x ^ 8'(i)), .d(o[i]), .q(q[i]));
  end
  assign y = q[0][7:0] ^ q[1][15:8] ^ q[2][23:16] ^ q[3][31:24] ^ o[0][7:0];
endmodule
