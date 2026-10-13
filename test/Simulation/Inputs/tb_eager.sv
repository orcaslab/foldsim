// Drives `M` (eager-output.mlir) with `reset` in its second cycle only and
// `a` late, so that `y` fires ahead of each cycle; prints the first tokens of
// `y` and `z`.
module tb;
  localparam N = 8;
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  logic       rs_v [N] = '{0, 1, 0, 0, 0, 0, 0, 0};
  logic [7:0] a_v [N] = '{8'h10, 8'h01, 8'h02, 8'h04, 8'h08, 8'h10, 8'h20, 8'h40};
  int irs = 0, ia = 0, im = 0, iy = 0, iz = 0, cyc = 0;
  logic [7:0] y [N], z [N];
  wire rs_ready, a_ready, m_ready, y_valid, z_valid;
  wire [7:0] y_bits, z_bits;
  wire a_valid = ia < N && cyc >= 9;
  M_threaded_M dut(.hostClock(clk), .hostReset(rst),
    .reset_sink_valid(irs < N), .reset_sink_bits(rs_v[irs % N]), .reset_sink_ready(rs_ready),
    .a_sink_valid(a_valid), .a_sink_bits(a_v[ia % N]), .a_sink_ready(a_ready),
    .m_sink_valid(im < N), .m_sink_bits(8'hff), .m_sink_ready(m_ready),
    .y_source_ready(1'b1), .y_source_valid(y_valid), .y_source_bits(y_bits),
    .z_source_ready(1'b1), .z_source_valid(z_valid), .z_source_bits(z_bits));
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (cyc == 4) rst <= 0;
    if (!rst) begin
      if (irs < N && rs_ready) irs <= irs + 1;
      if (a_valid && a_ready) ia <= ia + 1;
      if (im < N && m_ready) im <= im + 1;
      if (y_valid && iy < N) begin y[iy] <= y_bits; iy <= iy + 1; end
      if (z_valid && iz < N) begin z[iz] <= z_bits; iz <= iz + 1; end
    end
    if (cyc == 60) begin
      $display("y 1 is z 1: %0d", y[1] == z[1]);
      for (int i = 2; i < 6; i++) $display("y %0d = %h, z %0d = %h", i, y[i], i, z[i]);
      $finish;
    end
  end
endmodule
