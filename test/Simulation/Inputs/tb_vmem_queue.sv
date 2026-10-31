// Runs the unthreaded target (`Top` of Driver/Inputs/vmem_room, one target
// cycle per clock) and its static simulator with the leaves' RAM virtual, on
// the same input tokens, and compares the output tokens from cycle W on. The
// virtual memory is served by `vmem_queue_model`, a controller whose request
// queue is exactly what its `free` advertises: QUEUE requests (a read or a
// write each), `free` = QUEUE minus the requests queued, registered (it counts
// a host cycle's requests from the next host cycle on, as foldsim_vmem's
// does). The queue serves one request a host cycle in request order, none on
// +full=<percent> of the host cycles (a same-cycle read and write of a row by
// the read-first order), and answers a read +lo=<n> to +hi=<n> host cycles
// after serving it. Prints MATCH or the first difference, and PROTOCOL if the
// simulator overflows the queue (it issued more than the room its `free`
// showed: VMEM_INTERFACE.md guarantee 2) or reads an instance whose read is
// outstanding. With BUBBLES, `advance` is low on +bubble=<percent> of the
// host cycles.
module vmem_queue_model #(parameter IW = 3, ROWS = 16, QUEUE = 16) (
    input logic clk, rst,
    input int cyc, lo, hi, full,
    output logic rsp_valid, output logic [IW-1:0] rsp_inst, output logic [7:0] rsp_data,
    output logic [15:0] free,
    input logic rd_valid, input logic [IW-1:0] rd_inst, input logic [3:0] rd_row,
    input logic wr_valid, input logic [IW-1:0] wr_inst, input logic [3:0] wr_row,
    input logic [7:0] wr_data, input logic wr_mask,
    output bit ok, output int peak);
  localparam INSTS = 1 << IW;
  typedef struct packed { logic wr; logic [IW-1:0] inst; logic [3:0] row; logic [7:0] data; logic mask; } req_t;
  logic [7:0] mem [INSTS][ROWS];
  req_t q [$];
  bit outstanding [INSTS], answered [INSTS];
  int due [INSTS];
  logic [7:0] data [INSTS];
  initial begin
    ok = 1; peak = 0;
    for (int i = 0; i < INSTS; i++) begin
      outstanding[i] = 0; answered[i] = 0;
      for (int r = 0; r < ROWS; r++) mem[i][r] = 0;
    end
  end
  always @(negedge clk) begin
    free = 16'(q.size() < QUEUE ? QUEUE - q.size() : 0);
    rsp_valid = 0;
    for (int i = 0; i < INSTS; i++)
      if (!rsp_valid && answered[i] && cyc >= due[i]) begin
        rsp_valid = 1; rsp_inst = IW'(i); rsp_data = data[i];
      end
  end
  always @(posedge clk) if (!rst) begin
    if (rsp_valid) begin outstanding[rsp_inst] = 0; answered[rsp_inst] = 0; end
    // One request served (queued a host cycle or more ago).
    if (q.size() != 0 && $urandom % 100 >= full) begin
      automatic req_t e = q.pop_front();
      if (e.wr) begin
        if (e.mask) mem[e.inst][e.row] = e.data;
      end else begin
        data[e.inst] = mem[e.inst][e.row];
        answered[e.inst] = 1;
        due[e.inst] = cyc + lo + int'($urandom % (hi - lo + 1));
      end
    end
    // This host cycle's requests, the read first (read-first RAM).
    if (rd_valid) begin
      if (outstanding[rd_inst]) begin
        if (ok) $display("PROTOCOL: read of instance %0d at host cycle %0d", rd_inst, cyc);
        ok = 0;
      end
      outstanding[rd_inst] = 1;
      q.push_back('{wr: 1'b0, inst: rd_inst, row: rd_row, data: 8'd0, mask: 1'b0});
    end
    if (wr_valid)
      q.push_back('{wr: 1'b1, inst: wr_inst, row: wr_row, data: wr_data, mask: wr_mask});
    if (q.size() > peak) peak = q.size();
    if (q.size() > QUEUE) begin
      if (ok) $display("PROTOCOL: the queue overflows (%0d requests in %0d entries) at host cycle %0d",
                       q.size(), QUEUE, cyc);
      ok = 0;
    end
  end
endmodule

module tb;
  localparam T = 300, W = 4, IW = 3;
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
  int ix = 0, iy = 0, cyc = 0, peak;
  logic [7:0] sim_y [T];
  logic vx, yr, adv;
  wire xr, yv;
  wire [7:0] yb;
  logic rsp_valid;
  logic [IW-1:0] rsp_inst;
  logic [7:0] rsp_data;
  logic [15:0] free;
  wire rd_valid, wr_valid, wr_mask;
  wire [IW-1:0] rd_inst, wr_inst;
  wire [3:0] rd_row, wr_row;
  wire [7:0] wr_data;
  bit model_ok;
  int full = 0, lo = 1, hi = 1, bubble = 0;
  initial if (!$value$plusargs("bubble=%d", bubble)) bubble = 0;
  initial if (!$value$plusargs("full=%d", full)) full = 0;
  initial if (!$value$plusargs("lo=%d", lo)) lo = 1;
  initial if (!$value$plusargs("hi=%d", hi)) hi = lo;
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(vx), .x_bits(stim_x(ix)), .x_ready(xr),
    .y_ready(yr), .y_valid(yv), .y_bits(yb),
`ifdef BUBBLES
    .advance(adv),
`endif
    .vm0_rsp_valid(rsp_valid), .vm0_rsp_inst(rsp_inst), .vm0_rsp_data(rsp_data), .vm0_free(free),
    .vm0_rd_valid(rd_valid), .vm0_rd_inst(rd_inst), .vm0_rd_row(rd_row),
    .vm0_wr_valid(wr_valid), .vm0_wr_inst(wr_inst), .vm0_wr_row(wr_row),
    .vm0_wr_data(wr_data), .vm0_wr_mask(wr_mask),
    .profile_finish());
  vmem_queue_model #(.IW(IW)) model(.clk, .rst, .cyc, .lo, .hi, .full,
    .rsp_valid, .rsp_inst, .rsp_data, .free, .rd_valid, .rd_inst, .rd_row,
    .wr_valid, .wr_inst, .wr_row, .wr_data, .wr_mask, .ok(model_ok), .peak);
  always @(negedge clk) begin
    vx = ix < T;
    yr = 1;
    adv = $urandom % 100 >= bubble;
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
    do @(posedge clk); while (iy < T && cyc < 400000);
    wait (ref_done);
    if (iy < T) begin $display("TIMEOUT at %0d tokens", iy); $finish; end
    for (int t = W; t < T; t++)
      if (ok && sim_y[t] !== ref_y[t]) begin
        $display("MISMATCH cycle %0d: %h, target %h", t, sim_y[t], ref_y[t]); ok = 0;
      end
    ok &= model_ok;
    if (ok)
      $display("MATCH %0d cycles in %0d host cycles, H %0.2f, queue peak %0d", T - W, cyc, real'(cyc) / T, peak);
    $finish;
  end
endmodule
