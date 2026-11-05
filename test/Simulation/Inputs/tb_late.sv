// Runs the unthreaded target (`Top` of Driver/Inputs/late, one target cycle
// per clock) and its static simulator (`foldsim_sim`) on the same input
// tokens, the simulator's channels stalled on +stall=<percent> of the host
// cycles, and compares the output tokens from cycle W on (the leaves'
// registers have no reset). Prints MATCH or the first difference. With
// BUBBLES defined, `advance` is low on +bubble=<percent> of the host cycles
// (+burst=<n>: in runs of up to n).
module tb;
  localparam T = 200, W = 4;
  function automatic logic [7:0] stim_x(int t);
    logic [31:0] v = 32'(t) * 32'h9e3779b9;
    v ^= v >> 15; v *= 32'h2c1b3c6d; v ^= v >> 12;
    return v[7:0];
  endfunction
  logic rclk = 0;
  logic [7:0] rx;
  wire [7:0] ry;
  logic [7:0] ref_y [T];
  bit ref_done = 0;
  Top target(.clk(rclk), .x(rx), .y(ry));
  initial begin
    for (int t = 0; t < T; t++) begin
      rx = stim_x(t);
      #1; ref_y[t] = ry; rclk = 1; #1; rclk = 0;
    end
    ref_done = 1;
  end
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int ix = 0, iy = 0, cyc = 0;
  logic [7:0] sim_y [T];
  logic vx, yr;
  wire xr, yv;
  wire [7:0] yb;
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(vx), .x_bits(stim_x(ix)), .x_ready(xr),
    .y_ready(yr), .y_valid(yv), .y_bits(yb),
`ifdef BUBBLES
    .advance(adv),
`endif
    .profile_finish());
  int stall = 0, bubble = 0;
  logic adv;
  initial if (!$value$plusargs("stall=%d", stall)) stall = 0;
  initial if (!$value$plusargs("bubble=%d", bubble)) bubble = 0;
  // +burst=<n>: a bubble starts a run of up to n of them, as a host's long
  // stalls do.
  int burst = 1, brun = 0;
  initial if (!$value$plusargs("burst=%d", burst)) burst = 1;
  always @(negedge clk) begin
    vx = ix < T && ($urandom % 100 >= stall);
    yr = $urandom % 100 >= stall;
    if (brun > 0) begin
      adv = 0; brun--;
    end else begin
      adv = $urandom % 100 >= bubble;
      if (!adv && burst > 1) brun = $urandom % burst;
    end
  end
  always @(posedge clk) begin
    cyc <= cyc + 1; rst <= 0;
    if (!rst) begin
      if (vx && xr) ix <= ix + 1;
      if (yv && yr && iy < T) begin sim_y[iy] <= yb; iy <= iy + 1; end
    end
  end
  initial begin
    automatic bit ok = 1;
    do @(posedge clk); while (iy < T && cyc < 2000000);
    wait (ref_done);
    if (iy < T) begin $display("TIMEOUT at %0d tokens", iy); $finish; end
    for (int t = W; t < T; t++)
      if (ok && sim_y[t] !== ref_y[t]) begin
        $display("MISMATCH cycle %0d: %h, target %h", t, sim_y[t], ref_y[t]); ok = 0;
      end
    if (ok) $display("MATCH %0d cycles in %0d host cycles, H %0.2f", T - W, cyc, real'(cyc) / T);
    $finish;
  end
endmodule
