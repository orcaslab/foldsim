// A parent class whose passes are of two kinds: its state pass writes the
// early links to its two children (an HBM stream of several rows a pass:
// 160-bit channels in narrow rows), its ready pass a late output (an input of
// the parent combined with its state) that the root reads. Only the state
// pass writes the stream's pages: its ready pass needs no link-port window.
module Child(input logic clk, input logic rst, input logic [159:0] a, output logic [15:0] r);
  logic [15:0] s;
  always_comb begin
    s = 0;
    for (int k = 0; k < 10; ++k) s = s ^ a[16 * k +: 16];
  end
  always_ff @(posedge clk)
    if (rst) r <= 0;
    else r <= {r[14:0], r[15]} + s;
endmodule

module Parent(input logic clk, input logic rst, input logic [7:0] seed, input logic [15:0] x,
              output logic [15:0] late);
  logic [1:0][159:0] st;
  logic [1:0][15:0] rs;
  for (genvar i = 0; i < 2; ++i) begin : g
    Child c(.clk, .rst, .a(st[i]), .r(rs[i]));
    always_ff @(posedge clk)
      if (rst) st[i] <= {144'(i * 77 + 5), 8'(i), seed};
      else st[i] <= {st[i][158:0], st[i][159]} ^ {128'(rs[1 - i]), 16'(rs[i]), x};
  end
  assign late = x ^ st[0][15:0] ^ st[1][159:144];
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [3:0][15:0] l;
  logic [15:0] acc;
  for (genvar p = 0; p < 4; ++p) begin : g
    Parent u(.clk, .rst, .seed(8'(p * 41 + 3)), .x({x, 8'(p)} ^ acc), .late(l[p]));
  end
  always_ff @(posedge clk)
    if (rst) acc <= 0;
    else acc <= acc + (l[0] ^ l[1] ^ l[2] ^ l[3]);
  assign y = acc[7:0] ^ acc[15:8];
endmodule
