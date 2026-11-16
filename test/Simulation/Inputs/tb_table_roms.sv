// tb_table_roms.sv: the module before (foldsim_sim) and after foldsim-table-roms (foldsim_sim_roms) side by side,
// random next indices and resets; every output compared in every cycle after the first reset.
module tb;
  logic clk = 0, rst = 1;
  logic [8:0] nx, ny;
  wire [7:0] a0, a1, d0, d1; wire [4:0] b0, b1; wire [2:0] c0, c1; wire [5:0] e0, e1;
  foldsim_sim u0(.hostClock(clk), .hostReset(rst), .nx(nx), .ny(ny), .a(a0), .b(b0), .c(c0), .d(d0), .e(e0));
  foldsim_sim_roms u1(.hostClock(clk), .hostReset(rst), .nx(nx), .ny(ny), .a(a1), .b(b1), .c(c1), .d(d1), .e(e1));
  integer cyc = 0, bad = 0;
  always #5 clk = ~clk;
  always @(negedge clk) begin
    nx <= $urandom; ny <= $urandom;
    rst <= (cyc < 3) || ($urandom % 50 == 0);
    if (cyc >= 2 && {a0, b0, c0, d0, e0} !== {a1, b1, c1, d1, e1}) begin
      if (bad < 3) $display("differ cycle %0d: %h %h", cyc, {a0, b0, c0, d0, e0}, {a1, b1, c1, d1, e1});
      bad = bad + 1;
    end
    cyc = cyc + 1;
    if (cyc == 5000) begin
      if (bad == 0) $display("MATCH 5000 cycles"); else $display("MISMATCH %0d cycles", bad);
      $finish;
    end
  end
endmodule
