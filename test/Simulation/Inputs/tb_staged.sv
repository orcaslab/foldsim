// Drives staged-tokens.mlir's simulator with `reset` in the first cycle and
// `x` 1, 2, 3, ..., offered and taken on pseudo-random host cycles; prints
// the tokens of `z0` and `z1` after the first (the registers before reset)
// and the host cycle the last one came on.
module tb;
  localparam N = 12;
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int irs = 0, ix = 0, cyc = 0, last = 0;
  int iz [2] = '{0, 0};
  logic [7:0] z [2][N];
  logic [15:0] lfsr = 16'hace1;
  wire reset_ready, x_ready;
  wire [1:0] z_valid;
  wire [7:0] z_bits [2];
  wire x_offer = lfsr[3] | lfsr[7];
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .reset_valid(irs < N), .reset_bits(irs == 0), .reset_ready(reset_ready),
    .x_valid(ix < N && x_offer), .x_bits(8'(ix + 1)), .x_ready(x_ready),
    .z0_ready(lfsr[0]), .z0_valid(z_valid[0]), .z0_bits(z_bits[0]),
    .z1_ready(lfsr[5]), .z1_valid(z_valid[1]), .z1_bits(z_bits[1]),
`ifdef BUBBLES
    .advance(adv),
`endif
    .profile_finish());
`ifdef BUBBLES
  localparam END = 4000;
  int bubble = 0;
  logic adv;
  initial if (!$value$plusargs("bubble=%d", bubble)) bubble = 0;
  always @(negedge clk) adv = $urandom % 100 >= bubble;
`else
  localparam END = 400;
`endif
  always @(posedge clk) begin
    cyc <= cyc + 1;
    rst <= 0;
    lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    if (!rst) begin
      if (irs < N && reset_ready) irs <= irs + 1;
      if (ix < N && x_offer && x_ready) ix <= ix + 1;
      if (z_valid[0] && lfsr[0] && iz[0] < N) begin
        z[0][iz[0]] <= z_bits[0];
        iz[0] <= iz[0] + 1;
        last <= cyc;
      end
      if (z_valid[1] && lfsr[5] && iz[1] < N) begin
        z[1][iz[1]] <= z_bits[1];
        iz[1] <= iz[1] + 1;
        last <= cyc;
      end
    end
    if (cyc == END) begin
      for (int k = 0; k < 2; k++) begin
        $write("z%0d:", k);
        for (int i = 1; i < N; i++) $write(" %h", z[k][i]);
        $write("\n");
      end
      $display("last token on host cycle %0d", last);
      $finish;
    end
  end
endmodule
