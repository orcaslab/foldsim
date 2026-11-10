// Ring links (link-rings) and early chain links in HBM (csf-mailbox-hbm-chains):
// eight groups of eight nodes. A node's `rout` is a register read by the next
// node's `rin` (an early chain link, a ring through every node: a group's
// `rout` is its last node's, read by the next group's first); its `zout` is
// combinational in `rin` and read by the next node's `zin` (a zero-latency
// chain: late links, each entry read a few host cycles after its write; the
// group's `zout`, its last node's, is a late output of the group too, read by
// the next group's first node: the frame's wrap). Registers start from their
// resets; the top closes both rings through its input.
module Node(input logic clk, input logic rst, input logic [7:0] x, input logic [7:0] id,
            input logic [7:0] rin, input logic [7:0] zin, output logic [7:0] rout,
            output logic [7:0] zout, output logic [7:0] acc);
  logic [7:0] s, r;
  always_ff @(posedge clk)
    if (rst) begin
      s <= id;
      r <= ~id;
    end else begin
      s <= s * 8'd5 + (zin ^ x) + id;
      r <= rin ^ {s[3:0], s[7:4]};
    end
  assign rout = r;
  assign zout = rin ^ s;
  assign acc = s;
endmodule

module Grp(input logic clk, input logic rst, input logic [7:0] x, input logic [7:0] id,
           input logic [7:0] rin, input logic [7:0] zin, output logic [7:0] rout,
           output logic [7:0] zout, output logic [7:0] acc);
  logic [7:0][7:0] c_rin, c_rout, c_zin, c_zout, c_acc;
  logic [7:0] a;
  for (genvar j = 0; j < 8; ++j) begin : g_node
    Node node(.clk, .rst, .x, .id(8'(id * 8 + j)), .rin(c_rin[j]), .zin(c_zin[j]), .rout(c_rout[j]),
              .zout(c_zout[j]), .acc(c_acc[j]));
    if (j == 0) begin : g_first
      assign c_rin[j] = rin;
      assign c_zin[j] = zin;
    end else begin : g_next
      assign c_rin[j] = c_rout[j - 1];
      assign c_zin[j] = c_zout[j - 1];
    end
  end
  always_ff @(posedge clk)
    if (rst) a <= id;
    else a <= {a[6:0], a[7]} ^ c_acc[0] ^ c_acc[1] ^ c_acc[2] ^ c_acc[3] ^ c_acc[4] ^ c_acc[5] ^ c_acc[6] ^ c_acc[7];
  assign rout = c_rout[7];
  assign zout = c_zout[7];
  assign acc = a;
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [7:0][7:0] g_rin, g_rout, g_zin, g_zout, g_acc;
  for (genvar k = 0; k < 8; ++k) begin : g_grp
    Grp grp(.clk, .rst, .x(x ^ 8'(k)), .id(8'(k)), .rin(g_rin[k]), .zin(g_zin[k]), .rout(g_rout[k]),
            .zout(g_zout[k]), .acc(g_acc[k]));
    if (k == 0) begin : g_first
      assign g_rin[k] = g_rout[7] ^ x;
      assign g_zin[k] = g_zout[7] ^ 8'h5a;
    end else begin : g_next
      assign g_rin[k] = g_rout[k - 1];
      assign g_zin[k] = g_zout[k - 1];
    end
  end
  logic [7:0] q;
  always_ff @(posedge clk)
    if (rst) q <= 8'h0;
    else q <= g_acc[0] ^ g_acc[1] ^ g_acc[2] ^ g_acc[3] ^ g_acc[4] ^ g_acc[5] ^ g_acc[6] ^ g_acc[7];
  assign y = q;
endmodule
