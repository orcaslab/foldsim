// Runs the unthreaded target (`Top`, one target cycle per clock) and its
// simulator (`foldsim_sim`) on the same input tokens, the simulator with
// random stalls of every input and output channel, and compares each
// output's tokens from cycle W on. Each of the N instances takes inputs of
// its own, so a thread reading another's state shows; the target resets at
// the start, while every address is written, and again in mid-run.
// Prints MATCH or the first difference.
module tb;
  localparam N = 4, T = 128, W = 11;
  function automatic logic [31:0] mix(int t, int k);
    logic [31:0] x = 32'(t) * 32'h9e3779b9 ^ 32'(k) * 32'h85ebca6b;
    x ^= x >> 15; x *= 32'h2c1b3c6d; x ^= x >> 12;
    return x;
  endfunction
  function automatic logic stim_reset(int t); return t < 10 || (t >= 60 && t < 63); endfunction
  function automatic logic [3:0] stim_we(int t); return t < 8 ? 4'hf : mix(t, 1)[3:0]; endfunction
  function automatic logic [11:0] stim_waddr(int t);
    return t < 8 ? {4{3'(t)}} : mix(t, 2)[11:0];
  endfunction
  function automatic logic [31:0] stim_wdata(int t); return mix(t, 3); endfunction
  function automatic logic [11:0] stim_raddr(int t); return mix(t, 4)[11:0]; endfunction
  function automatic logic [3:0] stim_en(int t); return mix(t, 5)[3:0] | mix(t, 6)[3:0]; endfunction

  // The reference.
  logic rclk = 0, rreset;
  logic [3:0] rwe, ren;
  logic [11:0] rwaddr, rraddr;
  logic [31:0] rwdata;
  wire [15:0] ry [N];
  logic [15:0] ref_y [N][T];
  bit ref_done = 0;
  Top target(.clk(rclk), .reset(rreset), .we(rwe), .waddr(rwaddr), .wdata(rwdata),
    .raddr(rraddr), .en(ren), .y0(ry[0]), .y1(ry[1]), .y2(ry[2]), .y3(ry[3]));
  initial begin
    for (int t = 0; t < T; t++) begin
      rreset = stim_reset(t); rwe = stim_we(t); rwaddr = stim_waddr(t);
      rwdata = stim_wdata(t); rraddr = stim_raddr(t); ren = stim_en(t);
      #1;
      for (int k = 0; k < N; k++) ref_y[k][t] = ry[k];
      rclk = 1; #1; rclk = 0;
    end
    ref_done = 1;
  end

  // The simulator.
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int ir = 0, iwe = 0, iwa = 0, iwd = 0, ira = 0, ie = 0, cyc = 0;
  int iy [N] = '{default: 0};
  logic [15:0] sim_y [N][T];
  logic vr, vwe, vwa, vwd, vra, ve;
  logic [N-1:0] yr;
  wire rr, wer, war, wdr, rar, er;
  wire [N-1:0] yv;
  wire [15:0] yb [N];
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .reset_valid(vr), .reset_bits(stim_reset(ir)), .reset_ready(rr),
    .we_valid(vwe), .we_bits(stim_we(iwe)), .we_ready(wer),
    .waddr_valid(vwa), .waddr_bits(stim_waddr(iwa)), .waddr_ready(war),
    .wdata_valid(vwd), .wdata_bits(stim_wdata(iwd)), .wdata_ready(wdr),
    .raddr_valid(vra), .raddr_bits(stim_raddr(ira)), .raddr_ready(rar),
    .en_valid(ve), .en_bits(stim_en(ie)), .en_ready(er),
    .y0_ready(yr[0]), .y0_valid(yv[0]), .y0_bits(yb[0]),
    .y1_ready(yr[1]), .y1_valid(yv[1]), .y1_bits(yb[1]),
    .y2_ready(yr[2]), .y2_valid(yv[2]), .y2_bits(yb[2]),
    .y3_ready(yr[3]), .y3_valid(yv[3]), .y3_bits(yb[3]),
    .profile_finish());
  // Each channel stalls on a random half of the host cycles.
  always @(negedge clk) begin
    vr = ir < T && $urandom % 2; vwe = iwe < T && $urandom % 2;
    vwa = iwa < T && $urandom % 2; vwd = iwd < T && $urandom % 2;
    vra = ira < T && $urandom % 2; ve = ie < T && $urandom % 2;
    for (int k = 0; k < N; k++) yr[k] = $urandom % 2;
  end
  always @(posedge clk) begin
    cyc <= cyc + 1;
    rst <= 0;
    if (!rst) begin
      if (vr && rr) ir <= ir + 1;
      if (vwe && wer) iwe <= iwe + 1;
      if (vwa && war) iwa <= iwa + 1;
      if (vwd && wdr) iwd <= iwd + 1;
      if (vra && rar) ira <= ira + 1;
      if (ve && er) ie <= ie + 1;
      for (int k = 0; k < N; k++)
        if (yv[k] && yr[k] && iy[k] < T) begin
          sim_y[k][iy[k]] <= yb[k];
          iy[k] <= iy[k] + 1;
        end
    end
  end
  initial begin
    automatic bit ok = 1;
    automatic int done;
    do begin
      @(posedge clk);
      done = 1;
      for (int k = 0; k < N; k++) if (iy[k] < T) done = 0;
    end while (!done && cyc < 200000);
    wait (ref_done);
    if (!done) begin $display("TIMEOUT"); $finish; end
    for (int k = 0; k < N; k++)
      for (int t = W; t < T; t++)
        if (ok && sim_y[k][t] !== ref_y[k][t]) begin
          $display("MISMATCH y%0d cycle %0d: %h, target %h", k, t, sim_y[k][t], ref_y[k][t]);
          ok = 0;
        end
    if (ok) $display("MATCH %0d outputs x %0d cycles in %0d host cycles", N, T - W, cyc);
    $finish;
  end
endmodule
