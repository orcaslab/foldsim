// Runs the unthreaded target (`Top` of Driver/Inputs/paged, one target cycle
// per clock) and its static simulator with virtual memories on the same input
// tokens, and compares the output tokens from cycle W on. Each virtual memory
// (`vm0`, and `vm1` .. `vm5` with VM1 .. VM5), whose widths come from the
// simulator's foldsim_sim.vh (FOLDSIM_VMEM_<k>_IW, _AW, _DW, _MW: page words
// kept off chip with --storage <model>:pages=hbm, or a target RAM with
// --virtual-memories), is served by `vmem_bytes_model`: each instance's rows
// (zero at start, as the target's with +verilator+rand+reset+0), a read
// answered +lo=<n> to +hi=<n> host cycles after it is issued with the row as
// of its issue (writes apply in request order, by byte mask, and a read of the
// row the same host cycle writes reads the old row: read-first), and `free`
// low on +full=<percent> of the host cycles. The channels stall on
// +stall=<percent> of the host cycles; with BUBBLES, `advance` is low on
// +bubble=<percent>. Prints MATCH or the first difference, and PROTOCOL if
// the simulator issues a read of an instance whose read is still outstanding.
`include "foldsim_sim.vh"

module vmem_bytes_model #(parameter IW = 2, AW = 1, DW = 8, MW = 1) (
    input logic clk, rst,
    input int cyc, lo, hi, full,
    output logic rsp_valid, output logic [IW-1:0] rsp_inst, output logic [DW-1:0] rsp_data,
    output logic [15:0] free,
    input logic rd_valid, input logic [IW-1:0] rd_inst, input logic [AW-1:0] rd_row,
    input logic wr_valid, input logic [IW-1:0] wr_inst, input logic [AW-1:0] wr_row,
    input logic [DW-1:0] wr_data, input logic [MW-1:0] wr_mask,
    output bit ok);
  localparam INSTS = 1 << IW, ROWS = 1 << AW, G = DW / MW;
  logic [DW-1:0] mem [INSTS][ROWS];
  bit outstanding [INSTS];
  int due [INSTS];
  logic [DW-1:0] data [INSTS];
  initial begin
    ok = 1;
    for (int i = 0; i < INSTS; i++) begin
      outstanding[i] = 0;
      for (int r = 0; r < ROWS; r++) mem[i][r] = '0;
    end
  end
  always @(negedge clk) begin
    free = ($urandom % 100 >= full) ? 16'd64 : 16'd0;
    rsp_valid = 0;
    for (int i = 0; i < INSTS; i++)
      if (!rsp_valid && outstanding[i] && cyc >= due[i]) begin
        rsp_valid = 1; rsp_inst = IW'(i); rsp_data = data[i];
      end
  end
  always @(posedge clk) if (!rst) begin
    if (rsp_valid) outstanding[rsp_inst] = 0;
    if (rd_valid) begin
      if (outstanding[rd_inst]) begin
        if (ok) $display("PROTOCOL: read of instance %0d at host cycle %0d", rd_inst, cyc);
        ok = 0;
      end
      outstanding[rd_inst] = 1;
      due[rd_inst] = cyc + lo + int'($urandom % (hi - lo + 1));
      data[rd_inst] = mem[rd_inst][rd_row];
    end
    if (wr_valid)
      for (int k = 0; k < MW; k++)
        if (wr_mask[k]) mem[wr_inst][wr_row][k*G +: G] = wr_data[k*G +: G];
  end
endmodule

`define VM_MODEL(k) \
  logic vm``k``_rsp_valid, vm``k``_rd_valid, vm``k``_wr_valid; \
  logic [`FOLDSIM_VMEM_``k``_IW-1:0] vm``k``_rsp_inst, vm``k``_rd_inst, vm``k``_wr_inst; \
  logic [`FOLDSIM_VMEM_``k``_DW-1:0] vm``k``_rsp_data, vm``k``_wr_data; \
  logic [`FOLDSIM_VMEM_``k``_AW-1:0] vm``k``_rd_row, vm``k``_wr_row; \
  logic [`FOLDSIM_VMEM_``k``_MW-1:0] vm``k``_wr_mask; \
  logic [15:0] vm``k``_free; \
  bit vm``k``_ok; \
  vmem_bytes_model #(.IW(`FOLDSIM_VMEM_``k``_IW), .AW(`FOLDSIM_VMEM_``k``_AW), \
                     .DW(`FOLDSIM_VMEM_``k``_DW), .MW(`FOLDSIM_VMEM_``k``_MW)) model``k``( \
    .clk, .rst, .cyc, .lo, .hi, .full, .rsp_valid(vm``k``_rsp_valid), .rsp_inst(vm``k``_rsp_inst), \
    .rsp_data(vm``k``_rsp_data), .free(vm``k``_free), .rd_valid(vm``k``_rd_valid), \
    .rd_inst(vm``k``_rd_inst), .rd_row(vm``k``_rd_row), .wr_valid(vm``k``_wr_valid), \
    .wr_inst(vm``k``_wr_inst), .wr_row(vm``k``_wr_row), .wr_data(vm``k``_wr_data), \
    .wr_mask(vm``k``_wr_mask), .ok(vm``k``_ok));

`define VM_PORTS(k) \
    .vm``k``_rsp_valid(vm``k``_rsp_valid), .vm``k``_rsp_inst(vm``k``_rsp_inst), \
    .vm``k``_rsp_data(vm``k``_rsp_data), .vm``k``_free(vm``k``_free), \
    .vm``k``_rd_valid(vm``k``_rd_valid), .vm``k``_rd_inst(vm``k``_rd_inst), .vm``k``_rd_row(vm``k``_rd_row), \
    .vm``k``_wr_valid(vm``k``_wr_valid), .vm``k``_wr_inst(vm``k``_wr_inst), .vm``k``_wr_row(vm``k``_wr_row), \
    .vm``k``_wr_data(vm``k``_wr_data), .vm``k``_wr_mask(vm``k``_wr_mask),

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
  logic vx, yr, adv;
  wire xr, yv;
  wire [7:0] yb;
  int stall = 0, bubble = 0, full = 0, lo = 1, hi = 1;
  initial if (!$value$plusargs("stall=%d", stall)) stall = 0;
  initial if (!$value$plusargs("bubble=%d", bubble)) bubble = 0;
  initial if (!$value$plusargs("full=%d", full)) full = 0;
  initial if (!$value$plusargs("lo=%d", lo)) lo = 1;
  initial if (!$value$plusargs("hi=%d", hi)) hi = lo;
  `VM_MODEL(0)
`ifdef VM1
  `VM_MODEL(1)
`endif
`ifdef VM2
  `VM_MODEL(2)
`endif
`ifdef VM3
  `VM_MODEL(3)
`endif
`ifdef VM4
  `VM_MODEL(4)
`endif
`ifdef VM5
  `VM_MODEL(5)
`endif
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(vx), .x_bits(stim_x(ix)), .x_ready(xr),
    .y_ready(yr), .y_valid(yv), .y_bits(yb),
`ifdef BUBBLES
    .advance(adv),
`endif
    `VM_PORTS(0)
`ifdef VM1
    `VM_PORTS(1)
`endif
`ifdef VM2
    `VM_PORTS(2)
`endif
`ifdef VM3
    `VM_PORTS(3)
`endif
`ifdef VM4
    `VM_PORTS(4)
`endif
`ifdef VM5
    `VM_PORTS(5)
`endif
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
    ok &= vm0_ok;
`ifdef VM1
    ok &= vm1_ok;
`endif
`ifdef VM2
    ok &= vm2_ok;
`endif
`ifdef VM3
    ok &= vm3_ok;
`endif
`ifdef VM4
    ok &= vm4_ok;
`endif
`ifdef VM5
    ok &= vm5_ok;
`endif
    if (ok)
      $display("MATCH %0d cycles in %0d host cycles, H %0.2f", T - W, cyc, real'(cyc) / T);
    $finish;
  end
endmodule
