// Runs the unthreaded target (`Top` of Driver/Inputs/vmem_multi, one target
// cycle per clock) and its static simulator with the leaves' RAMs virtual, on
// the same input tokens, and compares the output tokens from cycle W on. The
// simulator's virtual memories are bound (vm_decl.svh, vm_ports.svh from
// vmem_bind.py) to `vmem_group_model`s, one per kernel controller: each
// memory alone, or a group of a class copy's memories (--vmem-group) as the
// kernel's foldsim_vmem_group serves them. Each member's requests of a host
// cycle (a read and a write: one entry) wait in a FIFO of 8; its `free` is
// twice the free entries but one (a decision's requests come four host cycles
// after it, so five decisions' entries may follow a `free` it saw); one entry
// a host cycle leaves the group's FIFOs,
// round robin, and not at all on +full=<percent> of the host cycles (the
// controller out of room); it applies to the member's rows (a same-cycle
// read and write of a row by its read-during-write order, a write by its byte
// enables) and its read is answered +lo=<n> to +hi=<n> host cycles later, one
// answer a host cycle for the group. The channels stall on +stall=<percent>
// of the host cycles; with BUBBLES, `advance` is low on +bubble=<percent>.
// Prints MATCH or the first difference, and PROTOCOL if the simulator issues
// a read of a member's instance whose read is outstanding, or overflows a
// FIFO (it issued without the room its free showed).
module vmem_group_model #(parameter N = 1, parameter int MMW [N] = '{default: 1},
                          parameter int MDW [N] = '{default: 64}, parameter int RF [N] = '{default: 1},
                          parameter int MIW [N] = '{default: 4}, parameter int TW [N] = '{default: 0}) (
    input logic clk, rst,
    input int cyc, lo, hi, full,
    input logic rd_valid [N], input logic [7:0] rd_inst [N], input logic [7:0] rd_row [N],
    input logic wr_valid [N], input logic [7:0] wr_inst [N], input logic [7:0] wr_row [N],
    input logic [1023:0] wr_data [N], input logic [127:0] wr_mask [N],
    output logic rsp_valid [N], output logic [7:0] rsp_inst, output logic [1023:0] rsp_data,
    output logic [15:0] free [N],
    output bit ok);
  localparam DEPTH = 8;
  typedef struct packed { logic rd, wr; logic [7:0] inst, winst; logic [7:0] rrow, wrow; logic [1023:0] data; logic [127:0] mask; } ent_t;
  // Rows of up to 1024 bits; instances (the address's) below 32, rows below 64.
  logic [1023:0] mem [N][32][64];
  ent_t fifo [N][DEPTH];
  int head [N], cnt [N];
  bit outstanding [N][256], answered [N][256];
  int due [N][256];
  logic [1023:0] data [N][256];
  int last;
  initial begin
    ok = 1; last = 0;
    for (int m = 0; m < N; m++) begin
      head[m] = 0; cnt[m] = 0;
      for (int i = 0; i < 256; i++) begin
        outstanding[m][i] = 0; answered[m][i] = 0;
        if (i < 32) for (int r = 0; r < 64; r++) mem[m][i][r] = 0;
      end
    end
  end
  // A member with TW tag bits (a vmem-rings image, foldsim_vmem TAGW): its
  // instance index the low MIW - TW bits, the tag above it naming the request.
  function automatic int addr(int m, logic [7:0] i);
    return TW[m] != 0 ? int'(i) & ((1 << (MIW[m] - TW[m])) - 1) : int'(i);
  endfunction
  function automatic logic [1023:0] masked(int m, logic [1023:0] old, logic [1023:0] d, logic [127:0] mask);
    logic [1023:0] v = old;
    for (int b = 0; b < 128; b++)
      if (MMW[m] == 1 || (b < MMW[m] && mask[b])) v[b*8 +: 8] = d[b*8 +: 8];
    return v & ((1024'd1 << MDW[m]) - 1);
  endfunction
  always @(negedge clk) begin
    automatic bit any = 0;
    for (int m = 0; m < N; m++) begin
      free[m] = 16'(cnt[m] < DEPTH ? 2 * (DEPTH - 1 - cnt[m]) : 0);
      rsp_valid[m] = 0;
    end
    // One due answer a host cycle for the group.
    for (int m = 0; m < N; m++)
      for (int i = 0; i < 256; i++)
        if (!any && answered[m][i] && cyc >= due[m][i]) begin
          any = 1; rsp_valid[m] = 1; rsp_inst = 8'(i); rsp_data = data[m][i];
        end
  end
  always @(posedge clk) if (!rst) begin
    for (int m = 0; m < N; m++)
      if (rsp_valid[m]) begin outstanding[m][rsp_inst] = 0; answered[m][rsp_inst] = 0; end
    // One entry leaves (taken a host cycle or more ago), round robin.
    if ($urandom % 100 >= full) begin
      for (int j = 1; j <= N; j++) begin
        automatic int m = (last + j) % N;
        if (cnt[m] != 0) begin
          automatic ent_t e = fifo[m][head[m]];
          head[m] = (head[m] + 1) % DEPTH; cnt[m]--; last = m;
          if (e.rd && RF[m] != 0) data[m][e.inst] = mem[m][addr(m, e.inst)][e.rrow];
          if (e.wr) mem[m][addr(m, e.winst)][e.wrow] = masked(m, mem[m][addr(m, e.winst)][e.wrow], e.data, e.mask);
          if (e.rd && RF[m] == 0) data[m][e.inst] = mem[m][addr(m, e.inst)][e.rrow];
          if (e.rd) begin answered[m][e.inst] = 1; due[m][e.inst] = cyc + lo + int'($urandom % (hi - lo + 1)); end
          break;
        end
      end
    end
    // This host cycle's requests join the FIFOs.
    for (int m = 0; m < N; m++)
      if (rd_valid[m] || wr_valid[m]) begin
        automatic ent_t e;
        e.rd = rd_valid[m]; e.wr = wr_valid[m]; e.inst = rd_inst[m]; e.winst = wr_inst[m];
        e.rrow = rd_row[m]; e.wrow = wr_row[m]; e.data = wr_data[m]; e.mask = wr_mask[m];
        if (cnt[m] == DEPTH) begin
          if (ok) $display("PROTOCOL: member %0d's FIFO overflows at host cycle %0d", m, cyc);
          ok = 0;
        end else begin
          if (rd_valid[m] && outstanding[m][e.inst]) begin
            if (ok) $display("PROTOCOL: read of member %0d instance %0d at host cycle %0d", m, e.inst, cyc);
            ok = 0;
          end
          if (rd_valid[m]) outstanding[m][e.inst] = 1;
          fifo[m][(head[m] + cnt[m]) % DEPTH] = e; cnt[m]++;
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
`ifdef TOP8
  Top8 target(.clk(rclk), .x(rx), .y(ry));
`else
  Top target(.clk(rclk), .x(rx), .y(ry));
`endif
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
  int stall = 0, bubble = 0, full = 0, lo = 1, hi = 1;
  initial if (!$value$plusargs("stall=%d", stall)) stall = 0;
  initial if (!$value$plusargs("bubble=%d", bubble)) bubble = 0;
  initial if (!$value$plusargs("full=%d", full)) full = 0;
  initial if (!$value$plusargs("lo=%d", lo)) lo = 1;
  initial if (!$value$plusargs("hi=%d", hi)) hi = lo;
`include "vm_decl.svh"
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(vx), .x_bits(stim_x(ix)), .x_ready(xr),
    .y_ready(yr), .y_valid(yv), .y_bits(yb),
`ifdef BUBBLES
    .advance(adv),
`endif
`include "vm_ports.svh"
    .profile_finish());
  always @(negedge clk) begin
    vx = ix < T && ($urandom % 100 >= stall);
    yr = $urandom % 100 >= stall;
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
    for (int g = 0; g < G; g++) ok &= model_ok[g];
    if (ok)
      $display("MATCH %0d cycles in %0d host cycles, H %0.2f", T - W, cyc, real'(cyc) / T);
    $finish;
  end
endmodule
