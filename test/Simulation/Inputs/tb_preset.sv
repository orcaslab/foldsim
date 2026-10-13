// Drives `x` 1, 2, 3, ... into preset.mlir's `Top`, whose state starts out
// random, after a reset of one host cycle; prints the first tokens of its
// outputs.
module tb;
  localparam N = 8;
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int ix = 0, cyc = 0;
  int i [5] = '{0, 0, 0, 0, 0};
  logic [3:0] o [5][N];
  wire x_ready;
  wire [4:0] valid;
  wire [3:0] q0, q1, t;
  wire f0, f1;
  wire [3:0] bits [5] = '{q0, q1, {3'b0, f0}, {3'b0, f1}, t};
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(ix < N), .x_bits(4'(ix + 1)), .x_ready(x_ready),
    .q0_ready(1'b1), .q0_valid(valid[0]), .q0_bits(q0),
    .q1_ready(1'b1), .q1_valid(valid[1]), .q1_bits(q1),
    .f0_ready(1'b1), .f0_valid(valid[2]), .f0_bits(f0),
    .f1_ready(1'b1), .f1_valid(valid[3]), .f1_bits(f1),
    .t_ready(1'b1), .t_valid(valid[4]), .t_bits(t),
    .profile_finish());
  always @(posedge clk) begin
    cyc <= cyc + 1;
    rst <= 0;
    if (!rst) begin
      if (ix < N && x_ready) ix <= ix + 1;
      for (int k = 0; k < 5; k++)
        if (valid[k] && i[k] < N) begin
          o[k][i[k]] <= bits[k];
          i[k] <= i[k] + 1;
        end
    end
    if (cyc == 200) begin
      for (int k = 0; k < 5; k++)
        $display("%0d: %h %h %h %h", k, o[k][0], o[k][1], o[k][2], o[k][3]);
      $finish;
    end
  end
endmodule
