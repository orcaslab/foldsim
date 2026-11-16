// Runs the unthreaded target (`Top` of a Driver/Inputs design, one target
// cycle per clock, an 8-bit input x and output y) and its static simulator
// built with --vmem-controller s28 (fold-static vmem-s28) on the same input
// tokens, and compares the output tokens from cycle W on. The simulator's
// memories reach it through the S28 tree it instantiates (the units of
// foldsim_vmem_{adapter,lsw,csw}.sv); each of its region buses
// (s28r<g>_<j>_*, listed by S28_BUSES as `S28_BUS(g, j)) is served here by
// `s28_region_model`, a behavioural region controller:
//  * request words (619 bits {cls 6, mem 4, inst 12, row 16, beat 3, last 1,
//    we 1, wmask 64, wdata 512}) are taken when `req_rdy` (+stall=<percent>
//    of the host cycles low) and applied in arrival order: a write beat's
//    bytes under its mask; a read takes the row's beats as of its arrival;
//  * each read is answered +lo=<n> to +hi=<n> host cycles after it arrives,
//    in arrival order, its beats back to back (538-bit words {cls 6, mem 4,
//    inst 12, beat 3, last 1, data 512}); each row its memory's beats, from
//    +beats=<file> ($readmemh, entry {g 4, cls 6, mem 4}: the beats of region g's
//    class cls's memory mem; tb_s28_beats.py writes it from foldsim_sim.json),
//    else 1;
//  * `hold` is high in reset and on +hold=<percent> of the host cycles.
// Rows are zero at start (the target's with +verilator+rand+reset+0).
// With RST_IN the target has a reset input, high for its first two cycles
// (the simulator's `rst` channel), as in tb_reset.sv; with +rstevery=<n> also for two of every n cycles.
// Prints MATCH or the first difference; WORDS, the request words taken.
`ifndef S28_BUSES
`define S28_BUSES `S28_BUS(0, 0)
`endif

module s28_region_model #(parameter G = 0) (
    input logic clk, rst,
    input int cyc, lo, hi, stall, hold_pct,
    input logic req_v, input logic [618:0] req_word, input logic idle,
    output logic req_rdy, output logic hold,
    output logic rsp_v, output logic [537:0] rsp_word,
    output int words);
  logic [511:0] mem [bit [40:0]];
  typedef struct { int due; logic [33:0] hdr; int beats; logic [511:0] data [8]; } rd_t;
  rd_t q [$];
  int beat = 0;
  initial words = 0;
  logic [3:0] beats [1 << 14];
  initial begin
    automatic string f;
    for (int i = 0; i < (1 << 14); i++) beats[i] = 4'd1;
    if ($value$plusargs("beats=%s", f)) $readmemh(f, beats);
  end
  function automatic int beats_of(logic [5:0] cls, logic [3:0] m);
    return int'(beats[{4'(G), cls, m}]);
  endfunction
  always @(negedge clk) begin
    req_rdy = !rst && ($urandom % 100 >= stall);
    hold = rst || ($urandom % 100 < hold_pct);
    rsp_v = 0;
    if (!rst && q.size() != 0 && cyc >= q[0].due) begin
      rsp_v = 1;
      rsp_word = {q[0].hdr[33:12], 3'(beat), 1'(beat == q[0].beats - 1), q[0].data[beat]};
    end
  end
  always @(posedge clk) if (!rst) begin
    if (rsp_v) begin
      if (beat == q[0].beats - 1) begin beat = 0; void'(q.pop_front()); end
      else beat++;
    end
    if (req_v && req_rdy) begin
      automatic logic [5:0] c = req_word[618:613];
      automatic logic [3:0] m = req_word[612:609];
      automatic logic [11:0] i = req_word[608:597];
      automatic logic [15:0] r = req_word[596:581];
      automatic logic [2:0] b = req_word[580:578];
      automatic bit [40:0] key = {c, m, i, r, b};
      words++;
      if (req_word[576]) begin
        automatic logic [511:0] old = mem.exists(key) ? mem[key] : '0;
        for (int y = 0; y < 64; y++)
          if (req_word[512 + y]) old[8 * y +: 8] = req_word[8 * y +: 8];
        mem[key] = old;
      end else begin
        automatic rd_t rd;
        rd.hdr = {c, m, i, 12'b0};
        rd.beats = beats_of(c, m);
        for (int k = 0; k < rd.beats; k++) begin
          automatic bit [40:0] kk = {c, m, i, r, 3'(k)};
          rd.data[k] = mem.exists(kk) ? mem[kk] : '0;
        end
        rd.due = cyc + lo + int'($urandom % (hi - lo + 1));
        if (q.size() != 0 && rd.due < q[q.size() - 1].due) rd.due = q[q.size() - 1].due;
        q.push_back(rd);
      end
    end
  end
endmodule

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
  logic rrst;
`ifdef RST_IN
  Top target(.clk(rclk), .rst(rrst), .x(rx), .y(ry));
`else
  Top target(.clk(rclk), .x(rx), .y(ry));
`endif
  initial begin
    for (int t = 0; t < T; t++) begin
      rx = stim_x(t);
      rrst = t < 2 || (rstevery > 0 && t % rstevery < 2);
      #1; ref_y[t] = ry; rclk = 1; #1; rclk = 0;
    end
    ref_done = 1;
  end
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int ix = 0, iy = 0, ir = 0, cyc = 0;
  logic [7:0] sim_y [T];
  logic vx, yr, vr;
  wire xr, yv, rr;
  wire [7:0] yb;
  int stall = 0, lo = 1, hi = 1, mstall = 0, hold_pct = 0, rstevery = 0;
  initial if (!$value$plusargs("rstevery=%d", rstevery)) rstevery = 0;
  initial if (!$value$plusargs("stall=%d", stall)) stall = 0;
  initial if (!$value$plusargs("mstall=%d", mstall)) mstall = 0;
  initial if (!$value$plusargs("hold=%d", hold_pct)) hold_pct = 0;
  initial if (!$value$plusargs("lo=%d", lo)) lo = 1;
  initial if (!$value$plusargs("hi=%d", hi)) hi = lo;
`define S28_BUS(g, j) \
  wire s28r``g``_``j``_req_v, s28r``g``_``j``_idle, s28r``g``_``j``_req_rdy, s28r``g``_``j``_hold, s28r``g``_``j``_rsp_v; \
  wire [618:0] s28r``g``_``j``_req_word; wire [537:0] s28r``g``_``j``_rsp_word; int words_``g``_``j; \
  s28_region_model #(.G(g)) m_``g``_``j (.clk, .rst, .cyc, .lo, .hi, .stall(mstall), .hold_pct, \
    .req_v(s28r``g``_``j``_req_v), .req_word(s28r``g``_``j``_req_word), .idle(s28r``g``_``j``_idle), \
    .req_rdy(s28r``g``_``j``_req_rdy), .hold(s28r``g``_``j``_hold), .rsp_v(s28r``g``_``j``_rsp_v), \
    .rsp_word(s28r``g``_``j``_rsp_word), .words(words_``g``_``j));
  `S28_BUSES
`undef S28_BUS
`define S28_BUS(g, j) \
    .s28r``g``_``j``_req_v, .s28r``g``_``j``_req_word, .s28r``g``_``j``_idle, .s28r``g``_``j``_req_rdy, \
    .s28r``g``_``j``_hold, .s28r``g``_``j``_rsp_v, .s28r``g``_``j``_rsp_word,
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(vx), .x_bits(stim_x(ix)), .x_ready(xr),
`ifdef RST_IN
    .rst_valid(vr), .rst_bits(ir < 2 || (rstevery > 0 && ir % rstevery < 2)), .rst_ready(rr),
`endif
    .y_ready(yr), .y_valid(yv), .y_bits(yb),
    `S28_BUSES
    .profile_finish());
`undef S28_BUS
  always @(negedge clk) begin
    vx = ix < T && ($urandom % 100 >= stall);
    vr = ir < T && ($urandom % 100 >= stall);
    yr = $urandom % 100 >= stall;
  end
  always @(posedge clk) begin
    cyc <= cyc + 1; rst <= cyc < 4;
    if (!rst) begin
      if (vx && xr) ix <= ix + 1;
`ifdef RST_IN
      if (vr && rr) ir <= ir + 1;
`endif
      if (yv && yr && iy < T) begin sim_y[iy] <= yb; iy <= iy + 1; end
    end
  end
  initial begin
    automatic bit ok = 1;
    automatic int n = 0;
    do @(posedge clk); while (iy < T && cyc < 4000000);
    wait (ref_done);
    if (iy < T) begin $display("TIMEOUT at %0d tokens", iy); $finish; end
    for (int t = W; t < T; t++)
      if (ok && sim_y[t] !== ref_y[t]) begin
        $display("MISMATCH cycle %0d: %h, target %h", t, sim_y[t], ref_y[t]); ok = 0;
      end
    if (ok)
      $display("MATCH %0d cycles in %0d host cycles, H %0.2f", T - W, cyc, real'(cyc) / T);
`define S28_BUS(g, j) n += words_``g``_``j;
    `S28_BUSES
`undef S28_BUS
    $display("WORDS %0d", n);
    $finish;
  end
endmodule
