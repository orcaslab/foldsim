// Four leaves in a chain, each with five target RAMs read through a
// registered address, as a memory macro reads (a write shows in the next
// cycle's read: write-first): a 16 x 32 RAM with byte enables, built as a
// macro library builds it, four 8-bit lanes each with its own copy of the
// read-address register; two more 16 x 8 RAMs written at the same address,
// one read at the lanes' address (as good as a fifth lane) and one elsewhere;
// a 32 x 16 RAM; and an 8 x 26 RAM (not whole bytes). A leaf writes and reads
// them from its input; its output mixes the reads.
module Ram #(parameter W = 8, parameter A = 4) (
  input  logic         clk,
  input  logic         R0_en,
  input  logic [A-1:0] R0_addr,
  output logic [W-1:0] R0_data,
  input  logic         W0_en,
  input  logic [A-1:0] W0_addr,
  input  logic [W-1:0] W0_data);
  logic [W-1:0] Memory [1 << A];
  logic [A-1:0] _R0_addr_d0;
  always_ff @(posedge clk) begin
    if (R0_en) _R0_addr_d0 <= R0_addr;
    if (W0_en) Memory[W0_addr] <= W0_data;
  end
  assign R0_data = Memory[_R0_addr_d0];
endmodule
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  wire [3:0] ra = d[3:0] ^ {d[7], 3'd0};
  wire [3:0] wa = d[7:4];
  wire       ren = d[1] | d[6];
  wire       we = d[0] | d[5];
  wire [3:0] be = {d[2] ^ d[7], d[3], d[6] | d[4], ~d[1]};
  wire [31:0] wd = {d, ~d, d ^ 8'h5a, d + 8'd1};
  wire [7:0] r [4];
  for (genvar i = 0; i < 4; ++i) begin : g_lane
    Ram #(.W(8), .A(4)) lane (.clk, .R0_en(ren), .R0_addr(ra), .R0_data(r[i]),
                              .W0_en(we & be[i]), .W0_addr(wa), .W0_data(wd[i*8 +: 8]));
  end
  wire [7:0] p0, p1;
  Ram #(.W(8), .A(4)) pair0 (.clk, .R0_en(ren), .R0_addr(ra), .R0_data(p0),
                             .W0_en(d[2]), .W0_addr(wa), .W0_data(d ^ 8'h0f));
  Ram #(.W(8), .A(4)) pair1 (.clk, .R0_en(ren), .R0_addr(~ra), .R0_data(p1),
                             .W0_en(d[2]), .W0_addr(wa), .W0_data(d ^ 8'hf0));
  wire [15:0] r16;
  Ram #(.W(16), .A(5)) m16 (.clk, .R0_en(d[2]), .R0_addr({d[1], d[3:0]}), .R0_data(r16),
                            .W0_en(d[3]), .W0_addr({d[0], d[7:4]}), .W0_data({d, d ^ 8'h33}));
  wire [25:0] r26;
  Ram #(.W(26), .A(3)) m26 (.clk, .R0_en(1'b1), .R0_addr(d[2:0]), .R0_data(r26),
                            .W0_en(d[4]), .W0_addr(d[7:5]), .W0_data({d[1:0], d, d + 8'd7, d}));
  assign q = r[0] ^ {r[1][3:0], r[1][7:4]} ^ (r[2] + r[3]) ^ r16[7:0] ^ r16[15:8] ^ r26[7:0] ^ r26[25:18] ^ (p0 - p1) ^ d;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
// Eight leaves in a chain: one copy evaluates eight instances a frame.
module Top8(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] c [9];
  assign c[0] = x;
  for (genvar i = 0; i < 8; ++i) begin : g_leaf
    Leaf l(.clk, .d(c[i]), .q(c[i+1]));
  end
  assign y = c[8];
endmodule
