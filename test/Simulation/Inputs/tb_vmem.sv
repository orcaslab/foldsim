// Runs the unthreaded target (`Top` of Driver/Inputs/vmem, one target cycle
// per clock) and its static simulator with the leaves' RAM virtual, on the
// same input tokens, and compares the output tokens from cycle W on. Each
// virtual memory port (`vm0`, and `vm1` with COPIES2, each copy's own; its
// instance index IW bits) is served by `vmem_model`: each instance's rows
// (zero at start, as the target's with +verilator+rand+reset+0), a read
// answered +lo=<n> to +hi=<n> host cycles after it is issued with the rows as
// of its issue (writes apply in request order, and a read of the row the same
// host cycle writes reads the old row: read-first), and `free` low on
// +full=<percent> of the host cycles. The channels stall on +stall=<percent>
// of the host cycles; with BUBBLES, `advance` is low on +bubble=<percent>.
// Prints MATCH or the first difference, and PROTOCOL if the simulator issues
// a read of an instance whose read is still outstanding.
`ifndef IW
`define IW 2
`endif
module vmem_model #(parameter IW = 2, ROWS = 16) (
    input logic clk, rst,
    input int cyc, lo, hi, full,
    output logic rsp_valid, output logic [IW-1:0] rsp_inst, output logic [7:0] rsp_data,
    output logic [15:0] free,
    input logic rd_valid, input logic [IW-1:0] rd_inst, input logic [3:0] rd_row,
    input logic wr_valid, input logic [IW-1:0] wr_inst, input logic [3:0] wr_row,
    input logic [7:0] wr_data, input logic wr_mask,
    output bit ok);
  localparam INSTS = 1 << IW;
  logic [7:0] mem [INSTS][ROWS];
  bit outstanding [INSTS];
  int due [INSTS];
  logic [7:0] data [INSTS];
  initial begin
    ok = 1;
    for (int i = 0; i < INSTS; i++) begin
      outstanding[i] = 0;
      for (int r = 0; r < ROWS; r++) mem[i][r] = 0;
    end
  end
  always @(negedge clk) begin
    free = ($urandom % 100 >= full) ? 16'd64 : 16'd0;
    // One due response a host cycle, any instance.
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
    if (wr_valid && wr_mask) mem[wr_inst][wr_row] = wr_data;
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
  logic [3:0] rd_row [V], wr_row [V];
  logic [7:0] wr_data [V];
  bit model_ok [V];
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
  int stall = 0, bubble = 0, full = 0, lo = 1, hi = 1;
  initial if (!$value$plusargs("stall=%d", stall)) stall = 0;
  initial if (!$value$plusargs("bubble=%d", bubble)) bubble = 0;
  initial if (!$value$plusargs("full=%d", full)) full = 0;
  initial if (!$value$plusargs("lo=%d", lo)) lo = 1;
  initial if (!$value$plusargs("hi=%d", hi)) hi = lo;
  for (genvar k = 0; k < V; k++) begin : g_vm
    vmem_model #(.IW(`IW)) model(.clk, .rst, .cyc, .lo, .hi, .full,
      .rsp_valid(rsp_valid[k]), .rsp_inst(rsp_inst[k]), .rsp_data(rsp_data[k]), .free(free[k]),
      .rd_valid(rd_valid[k]), .rd_inst(rd_inst[k]), .rd_row(rd_row[k]),
      .wr_valid(wr_valid[k]), .wr_inst(wr_inst[k]), .wr_row(wr_row[k]),
      .wr_data(wr_data[k]), .wr_mask(wr_mask[k]), .ok(model_ok[k]));
  end
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
    for (int k = 0; k < V; k++) ok &= model_ok[k];
    if (ok)
      $display("MATCH %0d cycles in %0d host cycles, H %0.2f", T - W, cyc, real'(cyc) / T);
    $finish;
  end
endmodule
