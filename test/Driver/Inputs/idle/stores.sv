// Eight nodes in a chain behind a pulse source, for pp-idle-skip's stores
// (Driver/Inputs/idle/top_stores.sv): a node's token bit `t`, burst counter
// `n` and output token `o` are narrow registers of the wide register's
// group (each reads `w`, `w` reads them), so state-in-ram-pp keeps them
// apart from the rows, in stores. A token's arrival changes only `t` and `n`:
// the rows do not change, and the producer, whose burst ended the frame
// before with its token's toggle, does not either. A clean bit of the rows
// alone would let the node skip its burst. No reset: run with the registers
// zero at start (+verilator+rand+reset+0).
module SNode(input logic clk, input logic [7:0] a, output logic [7:0] q);
  logic t, o;
  logic [2:0] n;
  logic [255:0] w;
  always_ff @(posedge clk) begin
    if (a[0] != t) begin
      t <= a[0];
      n <= w[2:0] | 3'd4;
    end else if (n != 3'd0) begin
      n <= n - 3'd1;
      w <= {w[254:0], w[255] ^ w[3] ^ n[1] ^ o} ^ {248'b0, a};
      if (n == 3'd1) o <= ~o;
      if (&w[47:40]) t <= ~t;
    end
  end
  assign q = {w[6:0] ^ w[255:249], o};
endmodule
