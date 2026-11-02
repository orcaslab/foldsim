// Folded port bundles (--fold-ports, Driver/Inputs/ports): runs the target
// (`Top`, one target cycle per clock) with a memory-like endpoint on each of
// its four bundles, and its static simulator (`foldsim_sim`) whose bundles
// are two record streams: each request record (req_chan, req_treset,
// req_bits) is one bundle's outputs of a target cycle, taken by that
// bundle's endpoint model, which then answers the next cycle's response
// record (Moore: a response of cycle t needs the requests before t only),
// +lo to +hi host cycles later (+hi up to hundreds: answers late), the
// bundles' answers offered in random order. Every bundle issues a record
// every target cycle (all busy). The streams and the target's channels are
// stalled on +stall=<percent> of the host cycles, `advance` low on
// +bubble=<percent> (BUBBLES). Prints MATCH or the first difference, and
// ORDER if a bundle's records come out of target-cycle order (a record of a
// bundle taken twice, or one missing). With SPEC (--spec-groups
// mp_<k>_ack=mp_<k>_rvalid:mp_<k>_rdata) each response field carries two
// copies, for a fire (rvalid and ack) of the cycle before or not: copy 1 the
// fired one, at [w +: w].
`include "foldsim_sim.vh"

module tb;
  localparam T = 200, W = 4, P = 4;
  function automatic logic [7:0] stim_x(int t);
    logic [31:0] v = 32'(t) * 32'h9e3779b9;
    v ^= v >> 15; v *= 32'h2c1b3c6d; v ^= v >> 12;
    return v[7:0];
  endfunction

  // The endpoint: a 16-word memory; its response the state after the
  // requests before, a function of the cycle's fire (rvalid and ack) as an
  // AXI read channel's.
  typedef struct { logic [15:0] mem [16]; logic rvalid; logic [15:0] rdata; } ep_t;
  // (Its response for a given fire, from the state before the cycle.)
  function automatic logic [16:0] answer(ref ep_t e, input logic avalid, input logic [7:0] addr,
                                         input logic [3:0] tag, input logic fire, input logic treset);
    if (treset) return '0;
    return {e.mem[addr[3:0]] + {15'b0, fire} + {12'b0, tag}, avalid ^ fire};
  endfunction
  function automatic logic [16:0] step(ref ep_t e, input logic avalid, input logic [7:0] addr,
                                       input logic [15:0] data, input logic ack, input logic [3:0] tag,
                                       input logic treset);
    logic fire = e.rvalid & ack;
    // The other fire's answer (a speculative copy).
    logic [16:0] other = answer(e, avalid, addr, tag, ~fire, treset);
    logic [16:0] now = answer(e, avalid, addr, tag, fire, treset);
    if (treset)
      for (int i = 0; i < 16; i++) e.mem[i] = '0;
    else if (avalid)
      e.mem[addr[3:0]] = data ^ {8'b0, addr} ^ {12'b0, tag};
    e.rvalid = now[0]; e.rdata = now[16:1];
    return other;
  endfunction
  function automatic void ep_init(ref ep_t e);
    for (int i = 0; i < 16; i++) e.mem[i] = 16'(i * 7);
    e.rvalid = 0; e.rdata = '0;
  endfunction

  // ---- The target with its endpoints ----
  logic rclk = 0, rrst;
  logic [7:0] rx;
  wire [7:0] ry;
  logic [7:0] ref_y [T];
  bit ref_done = 0;
  logic       r_rvalid [P];
  logic [15:0] r_rdata [P];
  wire        o_avalid [P];
  wire [7:0]  o_addr [P];
  wire [15:0] o_data [P];
  wire        o_ack [P];
  wire [3:0]  o_tag [P];
  ep_t rep [P];
  Top target(.clk(rclk), .rst(rrst), .x(rx), .y(ry),
    .mp_0_avalid(o_avalid[0]), .mp_0_addr(o_addr[0]), .mp_0_data(o_data[0]), .mp_0_ack(o_ack[0]), .mp_0_tag(o_tag[0]),
    .mp_0_rvalid(r_rvalid[0]), .mp_0_rdata(r_rdata[0]),
    .mp_1_avalid(o_avalid[1]), .mp_1_addr(o_addr[1]), .mp_1_data(o_data[1]), .mp_1_ack(o_ack[1]), .mp_1_tag(o_tag[1]),
    .mp_1_rvalid(r_rvalid[1]), .mp_1_rdata(r_rdata[1]),
    .mp_2_avalid(o_avalid[2]), .mp_2_addr(o_addr[2]), .mp_2_data(o_data[2]), .mp_2_ack(o_ack[2]), .mp_2_tag(o_tag[2]),
    .mp_2_rvalid(r_rvalid[2]), .mp_2_rdata(r_rdata[2]),
    .mp_3_avalid(o_avalid[3]), .mp_3_addr(o_addr[3]), .mp_3_data(o_data[3]), .mp_3_ack(o_ack[3]), .mp_3_tag(o_tag[3]),
    .mp_3_rvalid(r_rvalid[3]), .mp_3_rdata(r_rdata[3]));
  initial begin
    for (int p = 0; p < P; p++) begin
      ep_init(rep[p]); r_rvalid[p] = rep[p].rvalid; r_rdata[p] = rep[p].rdata;
    end
    for (int t = 0; t < T; t++) begin
      rx = stim_x(t);
      rrst = t < 2;
      #1; ref_y[t] = ry;
      for (int p = 0; p < P; p++)
        void'(step(rep[p], o_avalid[p], o_addr[p], o_data[p], o_ack[p], o_tag[p], rrst));
      rclk = 1; #1; rclk = 0;
      for (int p = 0; p < P; p++) begin r_rvalid[p] = rep[p].rvalid; r_rdata[p] = rep[p].rdata; end
    end
    ref_done = 1;
  end

  // ---- The simulator, its bundles behind the record streams ----
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int ix = 0, iy = 0, ir = 0, cyc = 0;
  logic [7:0] sim_y [T];
  logic vx, yr, vr, adv, qr, sv;
  logic [`FOLDSIM_PORTS_CW-1:0] sc;
  logic [`FOLDSIM_PORTS_RSP_W-1:0] sb;
  wire xr, yv, rr, qv, qt, sr;
  wire [7:0] yb;
  wire [`FOLDSIM_PORTS_CW-1:0] qc;
  wire [`FOLDSIM_PORTS_REQ_W-1:0] qb;
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(vx), .x_bits(stim_x(ix)), .x_ready(xr),
    .rst_valid(vr), .rst_bits(ir < 2), .rst_ready(rr),
    .y_ready(yr), .y_valid(yv), .y_bits(yb),
    .req_ready(qr), .req_valid(qv), .req_chan(qc), .req_treset(qt), .req_bits(qb),
    .rsp_ready(sr), .rsp_valid(sv), .rsp_chan(sc), .rsp_bits(sb),
`ifdef BUBBLES
    .advance(adv),
`endif
    .profile_finish());
  int stall = 0, bubble = 0, lo = 1, hi = 1;
  initial begin
    void'($value$plusargs("stall=%d", stall));
    void'($value$plusargs("bubble=%d", bubble));
    void'($value$plusargs("lo=%d", lo));
    void'($value$plusargs("hi=%d", hi));
  end
  // Each bundle's endpoint model, its records taken so far, and its answers
  // not yet taken (at most two: the run-ahead), each due at a host cycle.
  ep_t sep [P];
  int taken [P];
  logic [`FOLDSIM_PORTS_RSP_W-1:0] ans [P][$];
  int due [P][$];
  bit order_ok = 1;
  // A response record: the state's answer, and with SPEC the other fire's
  // as the other copy (the fired one copy 1).
  function automatic logic [`FOLDSIM_PORTS_RSP_W-1:0] pack(ref ep_t e, input logic [16:0] other,
                                                           input logic fired);
    logic [`FOLDSIM_PORTS_RSP_W-1:0] r = '0;
`ifdef SPEC
    logic [16:0] c1 = fired ? {e.rdata, e.rvalid} : other;
    logic [16:0] c0 = fired ? other : {e.rdata, e.rvalid};
    r[`FOLDSIM_PORTS_RSP_RVALID_LO +: 2] = {c1[0], c0[0]};
    r[`FOLDSIM_PORTS_RSP_RDATA_LO +: 32] = {c1[16:1], c0[16:1]};
`else
    r[`FOLDSIM_PORTS_RSP_RVALID_LO +: `FOLDSIM_PORTS_RSP_RVALID_W] = e.rvalid;
    r[`FOLDSIM_PORTS_RSP_RDATA_LO +: `FOLDSIM_PORTS_RSP_RDATA_W] = e.rdata;
`endif
    return r;
  endfunction
  initial
    for (int p = 0; p < P; p++) begin
      ep_init(sep[p]); taken[p] = 0;
      ans[p].push_back(pack(sep[p], {sep[p].rdata, sep[p].rvalid}, 0)); due[p].push_back(0);
    end
  int pick;
  always @(negedge clk) begin
    vx = ix < T && ($urandom % 100 >= stall);
    vr = ir < T && ($urandom % 100 >= stall);
    yr = $urandom % 100 >= stall;
    qr = $urandom % 100 >= stall;
    adv = $urandom % 100 >= bubble;
    // A bundle with an answer due, chosen at random.
    sv = 0; sc = '0; sb = '0;
    if ($urandom % 100 >= stall) begin
      automatic int start = $urandom % P;
      for (int i = 0; i < P && !sv; i++) begin
        automatic int p = (start + i) % P;
        if (ans[p].size() > 0 && due[p][0] <= cyc) begin
          sv = 1; sc = `FOLDSIM_PORTS_CW'(p); sb = ans[p][0];
        end
      end
    end
  end
  always @(posedge clk) begin
    cyc <= cyc + 1; rst <= 0;
    if (!rst) begin
      if (vx && xr) ix <= ix + 1;
      if (vr && rr) ir <= ir + 1;
      if (yv && yr && iy < T) begin sim_y[iy] <= yb; iy <= iy + 1; end
      if (sv && sr) begin
        void'(ans[sc].pop_front()); void'(due[sc].pop_front());
      end
      if (qv && qr) begin
        automatic int p = int'(qc);
        automatic logic fired = sep[p].rvalid & qb[`FOLDSIM_PORTS_REQ_ACK_LO];
        automatic logic [16:0] other;
        if (taken[p] >= T + 8) order_ok = 0;
        other = step(sep[p], qb[`FOLDSIM_PORTS_REQ_AVALID_LO], qb[`FOLDSIM_PORTS_REQ_ADDR_LO +: 8],
                     qb[`FOLDSIM_PORTS_REQ_DATA_LO +: 16], qb[`FOLDSIM_PORTS_REQ_ACK_LO],
                     qb[`FOLDSIM_PORTS_REQ_TAG_LO +: 4], qt);
        taken[p] = taken[p] + 1;
        ans[p].push_back(pack(sep[p], other, fired));
        due[p].push_back(cyc + lo + int'($urandom % (hi - lo + 1)));
      end
    end
  end
  initial begin
    automatic bit ok = 1;
    do @(posedge clk); while (iy < T && cyc < 400000);
    wait (ref_done);
    if (iy < T) begin $display("TIMEOUT at %0d tokens", iy); $finish; end
    for (int p = 0; p < P; p++)
      if (taken[p] + 1 < iy) begin
        if (ok) $display("ORDER: bundle %0d sent %0d records for %0d cycles", p, taken[p], iy);
        ok = 0;
      end
    if (!order_ok) begin $display("ORDER: a bundle sent too many records"); ok = 0; end
    for (int t = W; t < T; t++)
      if (ok && sim_y[t] !== ref_y[t]) begin
        $display("MISMATCH cycle %0d: %h, target %h", t, sim_y[t], ref_y[t]); ok = 0;
      end
    if (ok) $display("MATCH %0d cycles in %0d host cycles, H %0.2f", T - W, cyc, real'(cyc) / T);
    $finish;
  end
endmodule
