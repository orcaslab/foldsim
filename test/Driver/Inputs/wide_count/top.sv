// A count over many inputs, for --compress-adds: each leaf counts its 72
// pending bits with a pairwise adder tree (a tree of two-input additions of
// zero-extended halves, as a population count is often written) and keeps a
// running total of what was counted minus what left, wrapping in 9 bits.
// Its output depends on its input in the same cycle (a late output). No
// reset: run with the registers zero at start (+verilator+rand+reset+0).
module PopTree #(parameter int N = 72, parameter int M = $clog2(N + 1)) (
    input logic [N-1:0] in, output logic [M-1:0] out);
  localparam int L = $clog2(N);
  localparam int P = 1 << L;
  logic [M-1:0] tmp [L+1][P];
  for (genvar i = 0; i < P; i++) begin : g_leaf
    if (i < N) begin : g_in
      assign tmp[0][i] = M'(in[i]);
    end else begin : g_pad
      assign tmp[0][i] = '0;
    end
  end
  for (genvar j = 0; j < L; j++) begin : g_level
    for (genvar i = 0; i < (P >> (j + 1)); i++) begin : g_node
      logic [j+1:0] res;
      assign res = (j + 2)'(tmp[j][2*i][j:0]) + (j + 2)'(tmp[j][2*i+1][j:0]);
      assign tmp[j+1][i] = M'(res);
    end
  end
  assign out = tmp[L][0];
endmodule
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  localparam int N = 72;
  logic [N-1:0] pend, gone;
  logic [6:0] cin, cout;
  logic [8:0] total;
  PopTree #(.N(N)) pc_in(.in(pend), .out(cin));
  PopTree #(.N(N)) pc_out(.in(gone), .out(cout));
  assign q = {1'b0, cin} ^ total[7:0] ^ d ^ {cout, total[8]};
  always_ff @(posedge clk) begin
    pend <= {pend[N-9:0], d} ^ (N'(d) << (d[5:3] * 9)) ^ {N{d[6]}} & {pend[0], pend[N-1:1]};
    gone <= pend & ({N{d[7]}} ^ {gone[N-2:0], d[2]});
    total <= total + 9'(cin) - 9'(cout) + 9'(d[1:0]);
  end
endmodule
// A chain of late outputs within a cycle.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a ^ x), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
