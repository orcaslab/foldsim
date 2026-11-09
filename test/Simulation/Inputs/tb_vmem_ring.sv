// tb_vmem.sv for vmem-rings (Driver/Inputs/vmemring: 16 leaves, a read held
// when bit 1 is low): `vm0` the leaves' RAM (IW 4, 16 rows), `vm1` its image
// (a row per instance; IW1 instance bits, all but the lowest a tag: TAGW, not
// part of the address), each served by `vmem_model` as in
// tb_vmem.sv (+lo, +hi, +full, +stall, read first). Prints MATCH or the first
// difference, PROTOCOL on a read of an instance with one outstanding. The image's
// answers may take their own latency: +lo1=<n> +hi1=<n> (default +lo/+hi).
module vmem_model #(parameter IW = 2, AW = 4, TW = 0, ROWS = 1 << AW) (
    input logic clk, rst,
    input int cyc, lo, hi, full,
    output logic rsp_valid, output logic [IW-1:0] rsp_inst, output logic [7:0] rsp_data,
    output logic [15:0] free,
    input logic rd_valid, input logic [IW-1:0] rd_inst, input logic [AW-1:0] rd_row,
    input logic wr_valid, input logic [IW-1:0] wr_inst, input logic [AW-1:0] wr_row,
    input logic [7:0] wr_data, input logic wr_mask,
    output bit ok, output int reads);
  localparam INSTS = 1 << IW, AINSTS = 1 << (IW - TW);
  logic [7:0] mem [AINSTS][ROWS];
  bit outstanding [INSTS];
  int due [INSTS];
  logic [7:0] data [INSTS];
  initial begin
    ok = 1; reads = 0;
    for (int i = 0; i < INSTS; i++) begin
      outstanding[i] = 0;
      if (i < AINSTS) for (int r = 0; r < ROWS; r++) mem[i][r] = 0;
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
`ifdef WRITE_FIRST
    if (wr_valid && wr_mask) mem[wr_inst[IW-TW-1:0]][wr_row] = wr_data;
`endif
    if (rd_valid) begin
      reads++;
      if (outstanding[rd_inst]) begin
        if (ok) $display("PROTOCOL: read of instance %0d at host cycle %0d", rd_inst, cyc);
        ok = 0;
      end
      outstanding[rd_inst] = 1;
      due[rd_inst] = cyc + lo + int'($urandom % (hi - lo + 1));
      data[rd_inst] = mem[rd_inst[IW-TW-1:0]][rd_row];
    end
`ifndef WRITE_FIRST
    if (wr_valid && wr_mask) mem[wr_inst[IW-TW-1:0]][wr_row] = wr_data;
`endif
  end
endmodule

`ifndef IW1
`define IW1 4
`endif
`define VMEM_PORTS(k) \
    .vm``k``_rsp_valid(rsp_valid[k]), .vm``k``_rsp_inst(rsp_inst[k]), .vm``k``_rsp_data(rsp_data[k]), \
    .vm``k``_free(free[k]), \
    .vm``k``_rd_valid(rd_valid[k]), .vm``k``_rd_inst(rd_inst[k]), .vm``k``_rd_row(rd_row[k]), \
    .vm``k``_wr_valid(wr_valid[k]), .vm``k``_wr_inst(wr_inst[k]), .vm``k``_wr_row(wr_row[k]), \
    .vm``k``_wr_data(wr_data[k]), .vm``k``_wr_mask(wr_mask[k]),

module tb;
  localparam T = 200, W = 4;
  localparam V = 2;
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
  logic [7:0] rsp_inst [V];
  logic [7:0] rsp_data [V];
  logic [15:0] free [V];
  logic rd_valid [V], wr_valid [V], wr_mask [V];
  logic [7:0] rd_inst [V], wr_inst [V];
  logic [3:0] rd_row [V], wr_row [V];
  logic [7:0] wr_data [V];
  bit model_ok [V];
  int model_reads [V];
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(vx), .x_bits(stim_x(ix)), .x_ready(xr),
    .y_ready(yr), .y_valid(yv), .y_bits(yb),
`ifdef BUBBLES
    .advance(adv),
`endif
    .vm0_rsp_valid(rsp_valid[0]), .vm0_rsp_inst(rsp_inst[0][3:0]), .vm0_rsp_data(rsp_data[0]),
    .vm0_free(free[0]), .vm0_rd_valid(rd_valid[0]), .vm0_rd_inst(rd_inst[0][3:0]), .vm0_rd_row(rd_row[0]),
    .vm0_wr_valid(wr_valid[0]), .vm0_wr_inst(wr_inst[0][3:0]), .vm0_wr_row(wr_row[0]),
    .vm0_wr_data(wr_data[0]), .vm0_wr_mask(wr_mask[0]),
    .vm1_rsp_valid(rsp_valid[1]), .vm1_rsp_inst(rsp_inst[1][`IW1-1:0]), .vm1_rsp_data(rsp_data[1]),
    .vm1_free(free[1]), .vm1_rd_valid(rd_valid[1]), .vm1_rd_inst(rd_inst[1][`IW1-1:0]), .vm1_rd_row(rd_row[1]),
    .vm1_wr_valid(wr_valid[1]), .vm1_wr_inst(wr_inst[1][`IW1-1:0]), .vm1_wr_row(wr_row[1]),
    .vm1_wr_data(wr_data[1]), .vm1_wr_mask(wr_mask[1]),
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
  initial if (!$value$plusargs("lo=%d", lo)) lo = 1;
  initial if (!$value$plusargs("hi=%d", hi)) hi = lo;
  assign rd_inst[0][7:4] = 0; assign wr_inst[0][7:4] = 0;
  assign rd_inst[1][7:`IW1] = 0; assign wr_inst[1][7:`IW1] = 0;
  int lo1, hi1;
  initial begin
    if (!$value$plusargs("lo1=%d", lo1) && !$value$plusargs("lo=%d", lo1)) lo1 = 1;
    if (!$value$plusargs("hi1=%d", hi1) && !$value$plusargs("hi=%d", hi1)) hi1 = lo1;
    if (hi1 < lo1) hi1 = lo1;
  end
  for (genvar k = 0; k < V; k++) begin : g_vm
    localparam IWK = k ? `IW1 : 4;
    vmem_model #(.IW(IWK), .AW(4), .TW(k ? `IW1 - 1 : 0)) model(.clk, .rst, .cyc, .lo(k ? lo1 : lo), .hi(k ? hi1 : hi), .full,
      .rsp_valid(rsp_valid[k]), .rsp_inst(rsp_inst[k][IWK-1:0]), .rsp_data(rsp_data[k]), .free(free[k]),
      .rd_valid(rd_valid[k]), .rd_inst(rd_inst[k][IWK-1:0]), .rd_row(rd_row[k]),
      .wr_valid(wr_valid[k]), .wr_inst(wr_inst[k][IWK-1:0]), .wr_row(wr_row[k]),
      .wr_data(wr_data[k]), .wr_mask(wr_mask[k]), .ok(model_ok[k]), .reads(model_reads[k]));
  end
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
