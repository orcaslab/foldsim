// Writes A1 to address 1 and B2 to address 2 in the first two target cycles,
// reading addresses 0, 1, 2, 1, 2, ...; prints the first tokens of `r` and,
// with `TWO`, of `r1`, the same memory's other instance.
module tb;
  localparam N = 8;
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  logic       we_v [N] = '{1, 1, 0, 0, 0, 0, 0, 0};
  logic [2:0] wa_v [N] = '{1, 2, 0, 0, 0, 0, 0, 0};
  logic [7:0] wd_v [N] = '{8'hA1, 8'hB2, 0, 0, 0, 0, 0, 0};
  logic [2:0] ra_v [N] = '{0, 1, 2, 1, 2, 1, 2, 1};
  int iwe = 0, iwa = 0, iwd = 0, ira = 0, ir = 0, ir1 = 0, cyc = 0;
  logic [7:0] r [N], r1 [N];
  wire we_ready, waddr_ready, wdata_ready, raddr_ready, r_valid, r1_valid;
  wire [7:0] r_bits, r1_bits;
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .we_valid(iwe < N), .we_bits(we_v[iwe % N]), .we_ready(we_ready),
    .waddr_valid(iwa < N), .waddr_bits(wa_v[iwa % N]), .waddr_ready(waddr_ready),
    .wdata_valid(iwd < N), .wdata_bits(wd_v[iwd % N]), .wdata_ready(wdata_ready),
    .raddr_valid(ira < N), .raddr_bits(ra_v[ira % N]), .raddr_ready(raddr_ready),
    .r_ready(1'b1), .r_valid(r_valid), .r_bits(r_bits),
`ifdef TWO
    .r1_ready(1'b1), .r1_valid(r1_valid), .r1_bits(r1_bits),
`endif
    .profile_finish());
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (cyc == 1) rst <= 0;
    if (!rst) begin
      if (iwe < N && we_ready) iwe <= iwe + 1;
      if (iwa < N && waddr_ready) iwa <= iwa + 1;
      if (iwd < N && wdata_ready) iwd <= iwd + 1;
      if (ira < N && raddr_ready) ira <= ira + 1;
      if (r_valid && ir < N) begin r[ir] <= r_bits; ir <= ir + 1; end
`ifdef TWO
      if (r1_valid && ir1 < N) begin r1[ir1] <= r1_bits; ir1 <= ir1 + 1; end
`endif
    end
    if (cyc == 100) begin
      for (int i = 1; i < 6; i++) $display("r %0d = %h", i, r[i]);
`ifdef TWO
      for (int i = 1; i < 6; i++) $display("r1 %0d = %h", i, r1[i]);
`endif
      $finish;
    end
  end
endmodule
