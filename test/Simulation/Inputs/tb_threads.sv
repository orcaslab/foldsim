// Drives four accumulators (reset-threads.mlir) after a reset of one host
// cycle, `reset` in their first cycle and `x` 1, 2, 3, ...; prints their
// first tokens.
module tb;
  localparam N = 8;
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int irs = 0, ix = 0, cyc = 0;
  int iy [4] = '{0, 0, 0, 0};
  logic [7:0] y [4][N];
  wire reset_ready, x_ready;
  wire [3:0] y_valid;
  wire [7:0] y_bits [4];
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .reset_valid(irs < N), .reset_bits(irs == 0), .reset_ready(reset_ready),
    .x_valid(ix < N), .x_bits(8'(ix + 1)), .x_ready(x_ready),
    .y0_ready(1'b1), .y0_valid(y_valid[0]), .y0_bits(y_bits[0]),
    .y1_ready(1'b1), .y1_valid(y_valid[1]), .y1_bits(y_bits[1]),
    .y2_ready(1'b1), .y2_valid(y_valid[2]), .y2_bits(y_bits[2]),
    .y3_ready(1'b1), .y3_valid(y_valid[3]), .y3_bits(y_bits[3]),
    .profile_finish());
  always @(posedge clk) begin
    cyc <= cyc + 1;
    rst <= 0;
    if (!rst) begin
      if (irs < N && reset_ready) irs <= irs + 1;
      if (ix < N && x_ready) ix <= ix + 1;
      for (int k = 0; k < 4; k++)
        if (y_valid[k] && iy[k] < N) begin
          y[k][iy[k]] <= y_bits[k];
          iy[k] <= iy[k] + 1;
        end
    end
    if (cyc == 200) begin
      for (int k = 0; k < 4; k++)
        $display("y%0d: %h %h %h %h %h", k, y[k][1], y[k][2], y[k][3], y[k][4], y[k][5]);
      $finish;
    end
  end
endmodule
