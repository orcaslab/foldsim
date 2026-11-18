// Runs the unthreaded target (`Top` of Driver/Inputs/vmem, one target cycle
// per clock) and its static simulator with the leaves' RAM virtual, on the
// same input tokens, and compares the output tokens from cycle W on. Each
// virtual memory port (`vm0`, and `vm1` with COPIES2, each copy's own; its
// instance index IW bits) is served by `vmem_model`: each instance's rows
// (zero at start, as the target's with +verilator+rand+reset+0), a read
// answered +lo=<n> to +hi=<n> host cycles after it is issued with the rows as
// of its issue (writes apply in request order, and a read of the row the same
// host cycle writes reads the old row: read-first; with WRITE_FIRST the new
// row), and `free` low on
// +full=<percent> of the host cycles. The channels stall on +stall=<percent>
// of the host cycles; with BUBBLES, `advance` is low on +bubble=<percent>.
// With +queue=<n> (n > 0) `free` is instead a request queue's room: every
// read and write the simulator issues takes an entry, and each host cycle the
// oldest entry leaves once it is +qlat=<n> host cycles old (default 4) and
// with +drain=<percent> chance (default 100); a request that finds the queue
// full prints OVERFLOW (the simulator's room terms must keep that from
// happening) and fails the run.
// With RNG_STREAMS each model and the channels draw from a random stream of
// their own (xorshift), not from the one $urandom: two netlists' host cycles
// then compare whatever order Verilator evaluates the models in (with COPIES2
// the shared stream's order between the two models follows the netlist's
// schedule, so equal decisions could still give different host cycles).
// Prints MATCH or the first difference, and PROTOCOL if the simulator issues
// a read of an instance whose read is still outstanding; then READS, the reads
// issued, and FEWER where they are at most +maxreads=<n>. ROWW: row bits.
`ifndef IW
`define IW 2
`endif
`ifndef ROWW
`define ROWW 4
`endif
module vmem_model #(parameter IW = 2, ROWS = 1 << `ROWW, SEED = 1) (
    input logic clk, rst,
    input int cyc, lo, hi, full, queue, qlat, drain,
    output logic rsp_valid, output logic [IW-1:0] rsp_inst, output logic [7:0] rsp_data,
    output logic [15:0] free,
    input logic rd_valid, input logic [IW-1:0] rd_inst, input logic [`ROWW-1:0] rd_row,
    input logic wr_valid, input logic [IW-1:0] wr_inst, input logic [`ROWW-1:0] wr_row,
    input logic [7:0] wr_data, input logic wr_mask,
    output bit ok, output int reads);
  localparam INSTS = 1 << IW;
`ifdef RNG_STREAMS
  int unsigned rs = 32'h9e3779b9 ^ SEED;
  function automatic int unsigned rnd(); rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; endfunction
`else
  function automatic int unsigned rnd(); return $urandom; endfunction
`endif
  logic [7:0] mem [INSTS][ROWS];
  bit outstanding [INSTS];
  int qage [$];
  int due [INSTS];
  logic [7:0] data [INSTS];
  initial begin
    ok = 1; reads = 0;
    for (int i = 0; i < INSTS; i++) begin
      outstanding[i] = 0;
      for (int r = 0; r < ROWS; r++) mem[i][r] = 0;
    end
  end
  always @(negedge clk) begin
    if (queue > 0) begin
      if (qage.size() != 0 && cyc - qage[0] >= qlat && int'(rnd() % 100) < drain)
        void'(qage.pop_front());
      free = 16'(queue - qage.size());
    end else
      free = (rnd() % 100 >= full) ? 16'd64 : 16'd0;
    // One due response a host cycle, any instance.
    rsp_valid = 0;
    for (int i = 0; i < INSTS; i++)
      if (!rsp_valid && outstanding[i] && cyc >= due[i]) begin
        rsp_valid = 1; rsp_inst = IW'(i); rsp_data = data[i];
      end
  end
  always @(posedge clk) if (!rst) begin
    if (rsp_valid) outstanding[rsp_inst] = 0;
`ifdef WRITE_FIRST
    if (wr_valid && wr_mask) mem[wr_inst][wr_row] = wr_data;
`endif
    if (queue > 0)
      for (int n = 0; n < int'(rd_valid) + int'(wr_valid); n++) begin
        if (qage.size() >= queue) begin
          if (ok) $display("OVERFLOW: a request into a full queue of %0d at host cycle %0d", queue, cyc);
          ok = 0;
        end
        qage.push_back(cyc);
      end
    if (rd_valid) begin
      reads++;
      if (outstanding[rd_inst]) begin
        if (ok) $display("PROTOCOL: read of instance %0d at host cycle %0d", rd_inst, cyc);
        ok = 0;
      end
      outstanding[rd_inst] = 1;
      due[rd_inst] = cyc + lo + int'(rnd() % (hi - lo + 1));
      data[rd_inst] = mem[rd_inst][rd_row];
    end
`ifndef WRITE_FIRST
    if (wr_valid && wr_mask) mem[wr_inst][wr_row] = wr_data;
`endif
  end
endmodule

`define VMEM_PORTS(k) \
    .vm``k``_rsp_valid(rsp_valid[k]), .vm``k``_rsp_inst(rsp_inst[k]), .vm``k``_rsp_data(rsp_data[k]), \
    .vm``k``_free(free[k]), \
    .vm``k``_rd_valid(rd_valid[k]), .vm``k``_rd_inst(rd_inst[k]), .vm``k``_rd_row(rd_row[k]), \
    .vm``k``_wr_valid(wr_valid[k]), .vm``k``_wr_inst(wr_inst[k]), .vm``k``_wr_row(wr_row[k]), \
    .vm``k``_wr_data(wr_data[k]), .vm``k``_wr_mask(wr_mask[k]),

module tb;
  localparam T = 200, W = 4;
`ifdef COPIES2
  localparam V = 2;
`else
  localparam V = 1;
`endif
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
  logic vx, yr, adv;
  wire xr, yv;
  wire [7:0] yb;
  logic rsp_valid [V];
  logic [`IW-1:0] rsp_inst [V];
  logic [7:0] rsp_data [V];
  logic [15:0] free [V];
  logic rd_valid [V], wr_valid [V], wr_mask [V];
  logic [`IW-1:0] rd_inst [V], wr_inst [V];
  logic [`ROWW-1:0] rd_row [V], wr_row [V];
  logic [7:0] wr_data [V];
  bit model_ok [V];
  int model_reads [V];
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(vx), .x_bits(stim_x(ix)), .x_ready(xr),
    .y_ready(yr), .y_valid(yv), .y_bits(yb),
`ifdef BUBBLES
    .advance(adv),
`endif
    `VMEM_PORTS(0)
`ifdef COPIES2
    `VMEM_PORTS(1)
`endif
    .profile_finish());
  int stall = 0, bubble = 0, full = 0, lo = 1, hi = 1, maxreads = -1;
  initial if (!$value$plusargs("maxreads=%d", maxreads)) maxreads = -1;
  initial if (!$value$plusargs("stall=%d", stall)) stall = 0;
  initial if (!$value$plusargs("bubble=%d", bubble)) bubble = 0;
  // +burst=<n>: a bubble starts a run of up to n of them, as a host's long
  // stalls do.
  int burst = 1, brun = 0;
  initial if (!$value$plusargs("burst=%d", burst)) burst = 1;
  initial if (!$value$plusargs("full=%d", full)) full = 0;
  int queue = 0, qlat = 4, drain = 100;
  initial if (!$value$plusargs("queue=%d", queue)) queue = 0;
  initial if (!$value$plusargs("qlat=%d", qlat)) qlat = 4;
  initial if (!$value$plusargs("drain=%d", drain)) drain = 100;
  initial if (!$value$plusargs("lo=%d", lo)) lo = 1;
  initial if (!$value$plusargs("hi=%d", hi)) hi = lo;
`ifdef RNG_STREAMS
  int unsigned ts = 32'h2545f491;
  function automatic int unsigned trnd(); ts ^= ts << 13; ts ^= ts >> 17; ts ^= ts << 5; return ts; endfunction
`else
  function automatic int unsigned trnd(); return $urandom; endfunction
`endif
  for (genvar k = 0; k < V; k++) begin : g_vm
    vmem_model #(.IW(`IW), .SEED(k + 7)) model(.clk, .rst, .cyc, .lo, .hi, .full, .queue, .qlat, .drain,
      .rsp_valid(rsp_valid[k]), .rsp_inst(rsp_inst[k]), .rsp_data(rsp_data[k]), .free(free[k]),
      .rd_valid(rd_valid[k]), .rd_inst(rd_inst[k]), .rd_row(rd_row[k]),
      .wr_valid(wr_valid[k]), .wr_inst(wr_inst[k]), .wr_row(wr_row[k]),
      .wr_data(wr_data[k]), .wr_mask(wr_mask[k]), .ok(model_ok[k]), .reads(model_reads[k]));
  end
  always @(negedge clk) begin
    vx = ix < T && (trnd() % 100 >= stall);
    yr = trnd() % 100 >= stall;
    if (brun > 0) begin
      adv = 0; brun--;
    end else begin
      adv = trnd() % 100 >= bubble;
      if (!adv && burst > 1) brun = trnd() % burst;
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
    do @(posedge clk); while (iy < T && cyc < 4000000);
    wait (ref_done);
    if (iy < T) begin $display("TIMEOUT at %0d tokens", iy); $finish; end
    for (int t = W; t < T; t++)
      if (ok && sim_y[t] !== ref_y[t]) begin
        $display("MISMATCH cycle %0d: %h, target %h", t, sim_y[t], ref_y[t]); ok = 0;
      end
    for (int k = 0; k < V; k++) ok &= model_ok[k];
    if (ok)
      $display("MATCH %0d cycles in %0d host cycles, H %0.2f", T - W, cyc, real'(cyc) / T);
    begin
      automatic int n = 0;
      for (int k = 0; k < V; k++) n += model_reads[k];
      $display("READS %0d", n);
      if (maxreads >= 0 && n <= maxreads) $display("FEWER");
    end
    $finish;
  end
endmodule
