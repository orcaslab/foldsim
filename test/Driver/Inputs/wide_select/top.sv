// Selects over many inputs, for --prefix-negate and --balance-mux: each leaf
// answers the lowest of 72 pending bits (`pend & -pend`, a 72-bit negation)
// and takes the tag of the lowest of 20 mask bits in a loop that keeps the
// last match, under an enclosing condition (a priority chain on the false
// side of a mux that is itself the true value of another). Its output depends
// on its input in the same cycle (a late output). No reset: run with the
// registers zero at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  localparam int N = 72, M = 20;
  logic [N-1:0] pend, low;
  logic [M-1:0][7:0] tag;
  logic [M-1:0] mask;
  logic [7:0] pick, held, fold;
  always_comb begin
    low = pend & -pend;
    mask = M'(pend >> d[2:0]) ^ M'({d, d, d});
    pick = held;
    if (d[7] | d[0])
      for (int i = M - 1; i >= 0; i--)
        if (mask[i]) pick = tag[i];
    fold = '0;
    for (int i = 0; i < N / 8; i++) fold = fold ^ low[i * 8 +: 8];
  end
  assign q = pick ^ d ^ fold;
  always_ff @(posedge clk) begin
    pend <= ((pend & ~low) ^ ({N{d[1]}} & {pend[N-2:0], pend[N-1]}))
            | (N'({d, d[3:0]}) << (d[5:3] * 8)) | N'(d[6]);
    held <= pick ^ fold;
    if (d[4:0] < 5'(M)) tag[d[4:0]] <= d ^ held ^ low[N-1 -: 8];
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
