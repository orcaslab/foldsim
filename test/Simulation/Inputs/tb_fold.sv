// Runs the unthreaded target (`Top`, one target cycle per clock) and its
// simulator (`foldsim_sim`) on the same input tokens, the simulator with
// random stalls of every input and output channel, and compares each
// output's tokens from cycle W on, once every stage of the model holds
// values of its inputs (enables and valids high, reset, in the first cycles).
// `N` instances; prints MATCH or the first difference.
`ifndef N
`define N 4
`endif
module tb;
  localparam N = `N, T = 96, W = 8;
  function automatic logic [31:0] mix(int t, int k);
    logic [31:0] x = 32'(t) * 32'h9e3779b9 ^ 32'(k) * 32'h85ebca6b;
    x ^= x >> 15; x *= 32'h2c1b3c6d; x ^= x >> 12;
    return x;
  endfunction
  function automatic logic stim_reset(int t); return t < 4; endfunction
  function automatic logic [7:0] stim_a(int t); return mix(t, 1)[7:0]; endfunction
  function automatic logic [N-1:0] stim_en(int t);
    return t < 8 ? '1 : mix(t, 2)[N-1:0] | mix(t, 4)[N-1:0];
  endfunction
  function automatic logic [N-1:0] stim_v(int t);
    return t < 8 ? '1 : mix(t, 3)[N-1:0];
  endfunction

  // The reference.
  logic rclk = 0, rreset;
  logic [7:0] ra;
  logic [N-1:0] ren, rv;
  wire [31:0] ry [N];
  logic [31:0] ref_y [N][T];
  bit ref_done = 0;
  Top target(.clk(rclk), .reset(rreset), .a(ra), .en(ren), .v(rv),
`ifdef N2
    .y0(ry[0]), .y1(ry[1])
`else
    .y0(ry[0]), .y1(ry[1]), .y2(ry[2]), .y3(ry[3])
`endif
  );
  initial begin
    for (int t = 0; t < T; t++) begin
      rreset = stim_reset(t); ra = stim_a(t); ren = stim_en(t); rv = stim_v(t);
      #1;
      for (int k = 0; k < N; k++) ref_y[k][t] = ry[k];
      rclk = 1; #1; rclk = 0;
    end
    ref_done = 1;
  end

  // The simulator.
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int ir = 0, ia = 0, ie = 0, iv = 0, cyc = 0;
  int iy [N] = '{default: 0};
  logic [31:0] sim_y [N][T];
  logic vr, va, ve, vv;
  logic [N-1:0] yr;
  wire rr, ar, er, vrdy;
  wire [N-1:0] yv;
  wire [31:0] yb [N];
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .reset_valid(vr), .reset_bits(stim_reset(ir)), .reset_ready(rr),
    .a_valid(va), .a_bits(stim_a(ia)), .a_ready(ar),
    .en_valid(ve), .en_bits(stim_en(ie)), .en_ready(er),
    .v_valid(vv), .v_bits(stim_v(iv)), .v_ready(vrdy),
`ifdef N2
    .y0_ready(yr[0]), .y0_valid(yv[0]), .y0_bits(yb[0]),
    .y1_ready(yr[1]), .y1_valid(yv[1]), .y1_bits(yb[1]),
`else
    .y0_ready(yr[0]), .y0_valid(yv[0]), .y0_bits(yb[0]),
    .y1_ready(yr[1]), .y1_valid(yv[1]), .y1_bits(yb[1]),
    .y2_ready(yr[2]), .y2_valid(yv[2]), .y2_bits(yb[2]),
    .y3_ready(yr[3]), .y3_valid(yv[3]), .y3_bits(yb[3]),
`endif
    .profile_finish());
  // Each channel stalls on a random half of the host cycles.
  always @(negedge clk) begin
    vr = ir < T && $urandom % 2; va = ia < T && $urandom % 2;
    ve = ie < T && $urandom % 2; vv = iv < T && $urandom % 2;
    for (int k = 0; k < N; k++) yr[k] = $urandom % 2;
  end
  always @(posedge clk) begin
    cyc <= cyc + 1;
    rst <= 0;
    if (!rst) begin
      if (vr && rr) ir <= ir + 1;
      if (va && ar) ia <= ia + 1;
      if (ve && er) ie <= ie + 1;
      if (vv && vrdy) iv <= iv + 1;
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
    end while (!done && cyc < 100000);
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
