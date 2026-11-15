// The S26 hybrid (foldsim --state auto --state-home hybrid): a class whose
// state is flip-flops (a counter and a delay line), a small RAM (8 x 8 bits)
// and a large one (65,536 x 8 bits, read through a register), both inside a
// RAM module (so the memories have names the rules carry). Four leaves in a
// chain. The hybrid keeps the flip-flops and the small RAM on chip and makes
// the large RAM a virtual memory (rows in HBM, read on demand).
module dpram #(parameter D = 16, W = 8) (input logic clk, input logic we,
    input logic [$clog2(D)-1:0] wa, ra, input logic [W-1:0] wd, output logic [W-1:0] rd);
  logic [W-1:0] mem [D];
  always_ff @(posedge clk) begin
    if (we) mem[wa] <= wd;
    rd <= mem[ra];
  end
endmodule
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [5:0] cnt;
  logic [7:0] dl0, dl1;
  logic [7:0] small_rd;
  logic [7:0] big_rd;
  dpram #(.D(8), .W(8)) u_small(.clk, .we(d[0]), .wa(d[3:1]), .ra(d[6:4]), .wd(d ^ dl1), .rd(small_rd));
  dpram #(.D(65536), .W(8)) u_big(.clk, .we(d[1]), .wa({d, dl0}), .ra({dl0, d ^ {2'd0, cnt}}), .wd(d ^ dl1), .rd(big_rd));
  always_ff @(posedge clk) begin
    cnt <= cnt + {5'd0, d[2]} + 6'd1;
    dl0 <= d;
    dl1 <= dl0 ^ small_rd;
  end
  assign q = big_rd ^ small_rd ^ dl1;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
