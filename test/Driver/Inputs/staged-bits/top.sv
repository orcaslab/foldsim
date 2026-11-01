// One producer port feeding a bit each to four readers: a switch keeps a
// 4-bit valid vector, each link instance reads its own bit (as fsd_switch's
// out_valid feeds fsd_link's in_valid in the datacenter). Staged links then
// stage the same producer port for every reader at the same position.
module Lnk(input logic clk, input logic rst, input logic v, input logic [7:0] d,
           output logic [7:0] q);
  always_ff @(posedge clk)
    if (rst) q <= 0;
    else if (v) q <= q + d;
    else q <= {q[6:0], q[7]};
endmodule

module Sw(input logic clk, input logic rst, input logic [7:0] x, input logic [3:0][7:0] back,
          output logic [3:0] v, output logic [7:0] d);
  always_ff @(posedge clk)
    if (rst) begin
      v <= 0;
      d <= 0;
    end else begin
      v <= x[3:0] ^ back[0][3:0] ^ back[3][7:4];
      d <= x ^ back[1] ^ back[2];
    end
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [3:0] v;
  logic [7:0] d;
  logic [3:0][7:0] q;
  Sw sw(.clk, .rst, .x, .back(q), .v, .d);
  for (genvar i = 0; i < 4; ++i) begin : g_lnk
    Lnk lnk(.clk, .rst, .v(v[i]), .d, .q(q[i]));
  end
  assign y = q[0] ^ q[1] ^ q[2] ^ q[3];
endmodule
