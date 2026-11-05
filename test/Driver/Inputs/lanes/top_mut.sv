// top.sv with lane 5 of each leaf mutated (a shift of 2, not 3): not of the
// other lanes' structure. Four leaves in a chain, as top.sv:
// eight lanes of one structure (an element of `acc`, a shared key, the lane's
// index as constants), whose results feed both the lane's register and a sum
// over the lanes (a fanout point: the sum reads the held results), and four
// counters as separate registers (a generate block). A registered RAM read,
// as Inputs/deep_mc's (a multicycle class reads no memory combinationally).
// No reset: run with the registers and RAM rows zero at start
// (+verilator+rand+reset+0). top_mut.sv: lane 5 differs (a shift of 2).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  localparam int N = 8;
  logic [15:0] acc [N];
  logic [15:0] nxt [N];
  logic [7:0] ram [16];
  logic [7:0] rd;
  logic [3:0] wp;
  logic [15:0] key, fold;
  always_comb begin
    key = {d, d ^ rd} ^ {4'b0, wp, 8'h5a};
    for (int i = 0; i < N; i++) begin
      logic [15:0] a, s;
      a = acc[i] ^ key;
      s = a + {acc[i][7:0], acc[i][15:8]} + 16'(i * 4099 + 1234);
      if (s[3:0] > d[3:0]) s = s - {8'b0, d};
      nxt[i] = (s << 1) ^ (i == 5 ? s >> 2 : s >> 3) ^ ((a ^ 16'(i * 7919 + 4321)) & {acc[i][3:0], key[11:0]});
    end
    fold = '0;
    for (int i = 0; i < N; i++) fold = fold + (nxt[i] ^ {acc[i][3:0], 12'(i)});
  end
  for (genvar g = 0; g < 4; g++) begin : g_cnt
    logic [11:0] c;
    always_ff @(posedge clk) c <= c + {4'b0, d} + (acc[g][11:0] ^ acc[g + 4][11:0]);
  end
  always_ff @(posedge clk) begin
    for (int i = 0; i < N; i++) if (d[i] | key[i]) acc[i] <= nxt[i];
    if (fold[0]) ram[fold[7:4]] <= fold[15:8];
    rd <= ram[d[3:0] ^ wp];
    wp <= wp + 4'(d[1]);
    q <= fold[7:0] ^ fold[15:8] ^ g_cnt[0].c[7:0] ^ g_cnt[3].c[11:4];
  end
endmodule
// The second leaf reads the first's output mixed with the netlist's input in
// the same cycle: a late link (from Top's own logic) into a split reader.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a ^ x), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
