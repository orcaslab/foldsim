// Writes A1 to address 1, then C3 (low nibble only, for `s`) to address 1
// and B2 to address 2, each while reading the address written, then reads
// 1, 2, 1, ...; prints the first tokens of `r` and `s` of both instances.
module tb;
  localparam N = 8;
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  logic       we_v [N] = '{1, 1, 1, 0, 0, 0, 0, 0};
  logic [1:0] wm_v [N] = '{3, 1, 3, 0, 0, 0, 0, 0};
  logic [2:0] wa_v [N] = '{1, 1, 2, 0, 0, 0, 0, 0};
  logic [7:0] wd_v [N] = '{8'hA1, 8'hC3, 8'hB2, 0, 0, 0, 0, 0};
  logic [2:0] ra_v [N] = '{1, 1, 2, 1, 2, 1, 2, 1};
  int iwe = 0, iwm = 0, iwa = 0, iwd = 0, ira = 0, cyc = 0;
  int io [4] = '{0, 0, 0, 0};
  logic [7:0] o [4][N];
  wire we_ready, wmask_ready, waddr_ready, wdata_ready, raddr_ready;
  wire [3:0] o_valid;
  wire [7:0] o_bits [4];
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .we_valid(iwe < N), .we_bits(we_v[iwe % N]), .we_ready(we_ready),
    .wmask_valid(iwm < N), .wmask_bits(wm_v[iwm % N]), .wmask_ready(wmask_ready),
    .waddr_valid(iwa < N), .waddr_bits(wa_v[iwa % N]), .waddr_ready(waddr_ready),
    .wdata_valid(iwd < N), .wdata_bits(wd_v[iwd % N]), .wdata_ready(wdata_ready),
    .raddr_valid(ira < N), .raddr_bits(ra_v[ira % N]), .raddr_ready(raddr_ready),
    .r0_ready(1'b1), .r0_valid(o_valid[0]), .r0_bits(o_bits[0]),
    .s0_ready(1'b1), .s0_valid(o_valid[1]), .s0_bits(o_bits[1]),
    .r1_ready(1'b1), .r1_valid(o_valid[2]), .r1_bits(o_bits[2]),
    .s1_ready(1'b1), .s1_valid(o_valid[3]), .s1_bits(o_bits[3]),
    .profile_finish());
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (cyc == 1) rst <= 0;
    if (!rst) begin
      if (iwe < N && we_ready) iwe <= iwe + 1;
      if (iwm < N && wmask_ready) iwm <= iwm + 1;
      if (iwa < N && waddr_ready) iwa <= iwa + 1;
      if (iwd < N && wdata_ready) iwd <= iwd + 1;
      if (ira < N && raddr_ready) ira <= ira + 1;
      for (int k = 0; k < 4; k++)
        if (o_valid[k] && io[k] < N) begin o[k][io[k]] <= o_bits[k]; io[k] <= io[k] + 1; end
    end
    if (cyc == 100) begin
      for (int k = 0; k < 4; k++)
        $display("%s%0d: %h %h %h %h %h", k % 2 ? "s" : "r", k / 2,
                 o[k][1], o[k][2], o[k][3], o[k][4], o[k][5]);
      $finish;
    end
  end
endmodule
