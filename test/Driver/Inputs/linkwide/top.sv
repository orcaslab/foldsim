// As linkhbm, with each unit's word 13,000 bits wide (26 pages of a 512-bit
// stream row): more pages than a controller's queue holds.
module Unit(input logic clk, input logic rst, input logic [7:0] seed,
            input logic [31:0] in, output logic [12999:0] out);
  always_ff @(posedge clk)
    if (rst) out <= {1625{seed}};
    else out <= {out[12998:0], out[12999]} ^ {12968'b0, in ^ {24'b0, seed}};
endmodule

module Reader(input logic clk, input logic rst, input logic [7:0] x,
              input logic [12999:0] d, output logic [31:0] q);
  logic [31:0] f;
  always_comb begin
    f = 0;
    for (int k = 0; k < 406; ++k) f ^= d[32 * k +: 32];
  end
  always_ff @(posedge clk)
    if (rst) q <= 0;
    else q <= q + f + {24'b0, x};
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [12999:0] o [16];
  logic [31:0] q [16];
  for (genvar i = 0; i < 16; ++i) begin : g
    Unit u(.clk, .rst, .seed(8'(i * 37 + 1)), .in(q[(i + 1) % 16]), .out(o[i]));
    Reader r(.clk, .rst, .x(x ^ 8'(i)), .d(o[(i + 3) % 16]), .q(q[i]));
  end
  assign y = q[0][7:0] ^ q[5][15:8] ^ q[10][23:16] ^ q[15][31:24];
endmodule
