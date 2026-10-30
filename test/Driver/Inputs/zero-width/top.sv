// A packed table of one entry (a parameter sizes it to a single slot) indexed
// by a one-bit slot number: the import selects at a zero-width slice of it.
module Leaf (input wire clk, input wire reset, input wire slot, input wire set, input wire [2:0] v, output wire [2:0] st, output wire idle);
  reg [0:0][2:0] tstate;
  reg            s;
  always @(posedge clk) begin
    s <= slot;
    if (reset)
      tstate <= '0;
    else if (set)
      tstate[s] <= v;
  end
  assign st = tstate[s];
  assign idle = (tstate[s] == 3'd0);
endmodule
module Top (input wire clk, input wire reset, input wire slot, input wire set, input wire [2:0] v, output wire [2:0] st, output wire idle);
  Leaf leaf (.clk(clk), .reset(reset), .slot(slot), .set(set), .v(v), .st(st), .idle(idle));
endmodule
