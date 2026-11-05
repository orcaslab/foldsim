// Four leaves in a chain, for staged evaluation (--stage-levels): each leaf's
// output is a register (early), and its next state hangs off a deep cone: a
// scan of eight entries for the oldest (a compare and a choice each, in turn),
// whose result picks a row of a RAM read combinationally, the address of a
// RAM read through a register, the entry an element write takes and the
// enables of a sum and a RAM write. No reset: run with the registers and RAM
// rows zero at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [11:0] age [8];
  logic [7:0] val [8];
  logic [7:0] ram [16];
  logic [7:0] sram [16];
  logic [3:0] raddr_q;
  logic [2:0] wp;
  logic [15:0] acc;
  logic [7:0] rd, best, pick;
  logic [11:0] bage;
  logic [2:0] bi;
  always_comb begin
    bage = '0; bi = '0; best = '0;
    for (int i = 0; i < 8; i++) begin
      logic [11:0] a;
      a = age[i] ^ {4'b0, val[i]} ^ acc[11:0];
      if (a > bage || (a == bage && val[i] > best)) begin
        bage = a; bi = 3'(i); best = val[i];
      end
    end
    pick = ram[bage[3:0] ^ best[7:4]] ^ rd;
  end
  always_ff @(posedge clk) begin
    age[wp] <= age[wp] + {4'b0, d};
    val[bi] <= val[bi] ^ d ^ pick;
    if (d[0]) ram[d[7:4]] <= d ^ best;
    if (bage[0]) sram[{1'b0, bi} + {1'b0, d[2:0]}] <= pick;
    raddr_q <= bage[7:4] ^ best[3:0];
    wp <= wp + 3'(d[1]);
    if (d[2] | pick[0]) acc <= acc + {8'b0, pick} + {4'b0, bage};
    q <= pick ^ best ^ acc[15:8];
  end
  assign rd = sram[raddr_q];
endmodule
// The second leaf reads the first's output mixed with the netlist's input in
// the same cycle: a late link (from Top's own logic) into a staged reader.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a ^ x), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
