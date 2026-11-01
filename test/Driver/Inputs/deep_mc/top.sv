// Four leaves in a chain, for multicycle evaluation (--multicycle): as
// Inputs/deep's (an oldest-first scan of eight entries, each leaf's output a
// register), but both RAMs are read through registers at deep addresses (a
// multicycle class reads no memory combinationally), so the cone ends at
// registered reads' addresses and enables, an element write's entry, a RAM
// write's enable, address and data, and the registers' next values. No reset:
// run with the registers and RAM rows zero at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [11:0] age [8];
  logic [7:0] val [8];
  logic [7:0] ram [16];
  logic [7:0] sram [16];
  logic [2:0] wp;
  logic [15:0] acc;
  logic [7:0] rd, rr, best, pick;
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
    pick = rr ^ rd ^ bage[11:4];
  end
  always_ff @(posedge clk) begin
    age[wp] <= age[wp] + {4'b0, d};
    val[bi] <= val[bi] ^ d ^ pick;
    if (d[0] ^ bage[1]) ram[d[7:4] ^ best[3:0]] <= d ^ best;
    if (bage[0]) sram[{1'b0, bi} + {1'b0, d[2:0]}] <= pick;
    rd <= sram[bage[7:4] ^ best[3:0]];
    if (pick[1] | d[3]) rr <= ram[bage[3:0] ^ best[7:4]];
    wp <= wp + 3'(d[1]);
    if (d[2] | pick[0]) acc <= acc + {8'b0, pick} + {4'b0, bage};
    q <= pick ^ best ^ acc[15:8];
  end
endmodule
// The second leaf reads the first's output mixed with the netlist's input in
// the same cycle: a late link (from Top's own logic) into a multicycle reader.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a ^ x), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
