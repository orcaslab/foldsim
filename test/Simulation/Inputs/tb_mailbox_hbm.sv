// tb_mailbox_hbm.sv: tb_linkhbm.sv with up to eight virtual memories (VM1 .. VM7), for the
// reader-indexed HBM mailboxes of csf-mailbox-hbm (one stream a mailbox row).
// tb_reset.sv (the target of Driver/Inputs/linkpage, a reset input) with
// virtual memories: the static simulator's HBM link streams (`link-kinds`
// kind `hbm`, each a virtual memory of INSTANCES 1 whose request tag is a
// landing slot), served as tb_pages.sv serves its: `vmem_bytes_model` (rows
// zero at start, a read answered +lo to +hi host cycles after it is issued,
// writes in request order, `free` low on +full percent of the host cycles),
// or with REAL_VMEM the kernel's foldsim_vmem.sv before an AXI memory model.
// Channels stalled on +stall, `advance` low on +bubble (BUBBLES). Prints
// MATCH or the first difference, and PROTOCOL for a read of a tag still
// outstanding, a write of a tag whose read is outstanding (the controller's
// guarantees hold per tag and instance), or a write of a row whose read is
// outstanding (the controller orders no write after an earlier read).
// VM1 .. VM7: more virtual memories. The tag bits (foldsim_vmem TAGW) are the
// instance port's top FOLDSIM_VMEM_<k>_TAGW bits.
`include "foldsim_sim.vh"

module vmem_bytes_model #(parameter IW = 2, AW = 1, DW = 8, MW = 1, INSTANCES = 1 << IW) (
    input logic clk, rst,
    input int cyc, lo, hi, full,
    output logic rsp_valid, output logic [IW-1:0] rsp_inst, output logic [DW-1:0] rsp_data,
    output logic [15:0] free,
    input logic rd_valid, input logic [IW-1:0] rd_inst, input logic [AW-1:0] rd_row,
    input logic wr_valid, input logic [IW-1:0] wr_inst, input logic [AW-1:0] wr_row,
    input logic [DW-1:0] wr_data, input logic [MW-1:0] wr_mask,
    output bit ok);
  // The room it offers (+room=<n>, default 64; the kernel's controller: 32).
  int room = 64;
  initial void'($value$plusargs("room=%d", room));
  localparam INSTS = 1 << IW, ROWS = 1 << AW, G = DW / MW;
  // INSTANCES 1: one image, the instance a tag (an HBM link stream).
  logic [DW-1:0] mem [INSTS][ROWS];
  function automatic int img(logic [IW-1:0] i); return INSTANCES == 1 ? 0 : int'(i); endfunction
  bit outstanding [INSTS];
  int due [INSTS];
  logic [DW-1:0] data [INSTS];
  logic [AW-1:0] row_of [INSTS];
  initial begin
    ok = 1;
    for (int i = 0; i < INSTS; i++) begin
      outstanding[i] = 0;
      for (int r = 0; r < ROWS; r++) mem[i][r] = '0;
    end
  end
  always @(negedge clk) begin
    free = ($urandom % 100 >= full) ? 16'(room) : 16'd0;
    rsp_valid = 0;
    for (int i = 0; i < INSTS; i++)
      if (!rsp_valid && outstanding[i] && cyc >= due[i]) begin
        rsp_valid = 1; rsp_inst = IW'(i); rsp_data = data[i];
      end
  end
  always @(posedge clk) if (!rst) begin
    if (rsp_valid) outstanding[rsp_inst] = 0;
    // (A read and a write in one host cycle: the write is checked against
    // the reads before it, as the controller orders such a pair.)
    if (wr_valid) begin
      if (outstanding[wr_inst]) begin
        if (ok) $display("PROTOCOL: write of tag %0d, whose read is outstanding, at host cycle %0d", wr_inst, cyc);
        ok = 0;
      end
      for (int i = 0; i < INSTS; i++)
        if (outstanding[i] && img(IW'(i)) == img(wr_inst) && row_of[i] == wr_row) begin
          if (ok) $display("PROTOCOL: write of row %0d, whose read (tag %0d) is outstanding, at host cycle %0d", wr_row, i, cyc);
          ok = 0;
        end
    end
    if (rd_valid) begin
      if (outstanding[rd_inst]) begin
        if (ok) $display("PROTOCOL: read of instance %0d at host cycle %0d", rd_inst, cyc);
        ok = 0;
      end
      outstanding[rd_inst] = 1;
      row_of[rd_inst] = rd_row;
      due[rd_inst] = cyc + lo + int'($urandom % (hi - lo + 1));
      data[rd_inst] = mem[img(rd_inst)][rd_row];
    end
    if ($test$plusargs("trace")) begin
      if (rd_valid) $display("%0d rd tag %0d row %0d -> %h", cyc, rd_inst, rd_row, mem[img(rd_inst)][rd_row]);
      if (wr_valid) $display("%0d wr tag %0d row %0d %h", cyc, wr_inst, wr_row, wr_data);
      if (rsp_valid) $display("%0d rsp tag %0d %h", cyc, rsp_inst, rsp_data);
    end
    if (wr_valid)
      for (int k = 0; k < MW; k++)
        if (wr_mask[k]) mem[img(wr_inst)][wr_row][k*G +: G] = wr_data[k*G +: G];
  end
endmodule


`ifdef REAL_VMEM
// The real controller (REAL_VMEM: the kernel's foldsim_vmem.sv on the command
// line) in front of an AXI memory: in order, each transaction answered +lo to
// +hi host cycles after it is taken, writes by byte strobe, unwritten lines
// zero.
module axi_mem_model #(parameter IDW = 4) (
    input logic clk, rst, input int cyc, lo, hi,
    input logic awvalid, output logic awready, input logic [63:0] awaddr,
    input logic [IDW-1:0] awid, input logic [7:0] awlen,
    input logic wvalid, output logic wready, input logic [511:0] wdata,
    input logic [63:0] wstrb, input logic wlast,
    output logic bvalid, input logic bready, output logic [IDW-1:0] bid,
    input logic arvalid, output logic arready, input logic [63:0] araddr,
    input logic [IDW-1:0] arid, input logic [7:0] arlen,
    output logic rvalid, input logic rready, output logic [511:0] rdata,
    output logic [IDW-1:0] rid, output logic rlast);
  logic [511:0] mem [longint];
  typedef struct { longint addr; int len; int id; int due; logic [511:0] data [16]; } txn_t;
  txn_t aw_q [$], b_q [$], ar_q [$];
  int wbeat = 0, rbeat = 0, last_due = 0;
  function automatic int next_due();
    int d = cyc + lo + int'($urandom % (hi - lo + 1));
    if (d < last_due) d = last_due;
    last_due = d;
    return d;
  endfunction
  assign awready = 1;
  assign arready = 1;
  assign wready = aw_q.size() > 0;
  always_comb begin
    bvalid = b_q.size() > 0 && b_q[0].due <= cyc;
    bid = b_q.size() > 0 ? IDW'(b_q[0].id) : '0;
    rvalid = ar_q.size() > 0 && ar_q[0].due <= cyc;
    rid = ar_q.size() > 0 ? IDW'(ar_q[0].id) : '0;
    rlast = ar_q.size() > 0 && rbeat == ar_q[0].len;
    rdata = ar_q.size() > 0 ? ar_q[0].data[rbeat] : '0;
  end
  // Handshakes sampled at the rising edge (the controller's outputs as they
  // were), applied at the falling edge, so this model's outputs never
  // change in the controller's edge.
  logic aw_f, w_f, b_f, ar_f, r_f, rst_f;
  logic [63:0] aw_a, ar_a; logic [7:0] aw_l, ar_l; logic [IDW-1:0] aw_i, ar_i;
  logic [511:0] w_d; logic [63:0] w_s; logic w_l;
  always @(posedge clk) begin
    rst_f <= rst;
    aw_f <= awvalid && awready; aw_a <= awaddr; aw_l <= awlen; aw_i <= awid;
    w_f <= wvalid && wready; w_d <= wdata; w_s <= wstrb; w_l <= wlast;
    b_f <= bvalid && bready;
    ar_f <= arvalid && arready; ar_a <= araddr; ar_l <= arlen; ar_i <= arid;
    r_f <= rvalid && rready;
  end
  always @(negedge clk) begin
    if (rst_f) begin
      aw_q.delete(); b_q.delete(); ar_q.delete(); wbeat = 0; rbeat = 0;
    end else begin
      if (b_f) void'(b_q.pop_front());
      if (r_f) begin
        if (rbeat == ar_q[0].len) begin void'(ar_q.pop_front()); rbeat = 0; end
        else rbeat++;
      end
      if (aw_f) begin automatic txn_t t; t.addr = aw_a; t.len = aw_l; t.id = aw_i; t.due = 0; aw_q.push_back(t); end
      if (w_f) begin
        automatic longint line = (aw_q[0].addr >> 6) + wbeat;
        automatic logic [511:0] old = mem.exists(line) ? mem[line] : '0;
        for (int k = 0; k < 64; k++) if (w_s[k]) old[k*8 +: 8] = w_d[k*8 +: 8];
        mem[line] = old;
        if (w_l) begin
          automatic txn_t t = aw_q.pop_front(); t.due = next_due(); b_q.push_back(t); wbeat = 0;
        end else wbeat++;
      end
      if (ar_f) begin
        // The read's data as of its acceptance.
        automatic txn_t t; t.addr = ar_a; t.len = ar_l; t.id = ar_i; t.due = next_due();
        for (int k = 0; k < 16; k++)
          t.data[k] = mem.exists((ar_a >> 6) + k) ? mem[(ar_a >> 6) + k] : '0;
        ar_q.push_back(t);
      end
    end
  end
endmodule

module vmem_real #(parameter IW = 2, AW = 1, DW = 8, MW = 1, INSTANCES = 4, TAGW = 0) (
    input logic clk, rst,
    input int cyc, lo, hi, full,
    output logic rsp_valid, output logic [IW-1:0] rsp_inst, output logic [DW-1:0] rsp_data,
    output logic [15:0] free,
    input logic rd_valid, input logic [IW-1:0] rd_inst, input logic [AW-1:0] rd_row,
    input logic wr_valid, input logic [IW-1:0] wr_inst, input logic [AW-1:0] wr_row,
    input logic [DW-1:0] wr_data, input logic [MW-1:0] wr_mask,
    output bit ok);
  wire awvalid, awready, wvalid, wready, wlast, bvalid, bready, arvalid, arready, rvalid, rready, rlast;
  wire [63:0] awaddr, araddr; wire [63:0] wstrb; wire [511:0] wdata, rdata;
  wire [3:0] awid, bid, arid, rid; wire [7:0] awlen, arlen;
  wire [5:0] free6; wire moving;
  foldsim_vmem #(.IW(IW - TAGW), .TAGW(TAGW), .AW(AW), .DW(DW), .MW(MW), .INSTANCES(INSTANCES)) ctl(
    .clk, .reset(rst), .rd_valid, .rd_inst, .rd_row, .wr_valid, .wr_inst, .wr_row, .wr_data, .wr_mask,
    .rsp_valid, .rsp_inst, .rsp_data, .free(free6), .base(64'h0), .zero(1'b1), .move(1'b0), .moving,
    .m_awvalid(awvalid), .m_awready(awready), .m_awaddr(awaddr), .m_awid(awid), .m_awlen(awlen),
    .m_wvalid(wvalid), .m_wready(wready), .m_wdata(wdata), .m_wstrb(wstrb), .m_wlast(wlast),
    .m_bvalid(bvalid), .m_bready(bready), .m_bid(bid),
    .m_arvalid(arvalid), .m_arready(arready), .m_araddr(araddr), .m_arid(arid), .m_arlen(arlen),
    .m_rvalid(rvalid), .m_rready(rready), .m_rdata(rdata), .m_rid(rid), .m_rlast(rlast));
  axi_mem_model mem(.clk, .rst, .cyc, .lo, .hi, .awvalid, .awready, .awaddr, .awid, .awlen,
    .wvalid, .wready, .wdata, .wstrb, .wlast, .bvalid, .bready, .bid, .arvalid, .arready, .araddr,
    .arid, .arlen, .rvalid, .rready, .rdata, .rid, .rlast);
  assign free = 16'(free6);
  localparam INSTS = 1 << IW;
  bit outstanding [INSTS];
  initial begin ok = 1; for (int i = 0; i < INSTS; i++) outstanding[i] = 0; end
  always @(posedge clk) if (!rst) begin
    if (rsp_valid) outstanding[rsp_inst] = 0;
    if (rd_valid) begin
      if (outstanding[rd_inst]) begin
        if (ok) $display("PROTOCOL: read of instance %0d at host cycle %0d", rd_inst, cyc);
        ok = 0;
      end
      outstanding[rd_inst] = 1;
    end
  end
endmodule
`define VMEM_MODULE vmem_real
`ifdef FOLDSIM_VMEM_TAGS
`define VMEM_INSTANCES(k) , .INSTANCES(`FOLDSIM_VMEM_``k``_INSTANCES), .TAGW(`FOLDSIM_VMEM_``k``_TAGW)
`else
`define VMEM_INSTANCES(k) , .INSTANCES(`FOLDSIM_VMEM_``k``_INSTANCES)
`endif
`else
`define VMEM_MODULE vmem_bytes_model
`define VMEM_INSTANCES(k) , .INSTANCES(`FOLDSIM_VMEM_``k``_INSTANCES)
`endif

`define VM_MODEL(k) \
  logic vm``k``_rsp_valid, vm``k``_rd_valid, vm``k``_wr_valid; \
  logic [`FOLDSIM_VMEM_``k``_IW-1:0] vm``k``_rsp_inst, vm``k``_rd_inst, vm``k``_wr_inst; \
  logic [`FOLDSIM_VMEM_``k``_DW-1:0] vm``k``_rsp_data, vm``k``_wr_data; \
  logic [`FOLDSIM_VMEM_``k``_AW-1:0] vm``k``_rd_row, vm``k``_wr_row; \
  logic [`FOLDSIM_VMEM_``k``_MW-1:0] vm``k``_wr_mask; \
  logic [15:0] vm``k``_free; \
  bit vm``k``_ok; \
  `VMEM_MODULE #(.IW(`FOLDSIM_VMEM_``k``_IW), .AW(`FOLDSIM_VMEM_``k``_AW), \
                 .DW(`FOLDSIM_VMEM_``k``_DW), .MW(`FOLDSIM_VMEM_``k``_MW) `VMEM_INSTANCES(k)) model``k``( \
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
  logic rrst;
  Top target(.clk(rclk), .rst(rrst), .x(rx), .y(ry));
  initial begin
    for (int t = 0; t < T; t++) begin
      rx = stim_x(t);
      rrst = t < 2;
      #1; ref_y[t] = ry; rclk = 1; #1; rclk = 0;
    end
    ref_done = 1;
  end
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int ix = 0, iy = 0, ir = 0, cyc = 0;
  logic [7:0] sim_y [T];
  logic vx, yr, vr, adv;
  wire xr, yv, rr;
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
`ifdef VM6
  `VM_MODEL(6)
`endif
`ifdef VM7
  `VM_MODEL(7)
`endif
  foldsim_sim dut(.hostClock(clk), .hostReset(rst),
    .x_valid(vx), .x_bits(stim_x(ix)), .x_ready(xr),
    .rst_valid(vr), .rst_bits(ir < 2), .rst_ready(rr),
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
`ifdef VM6
    `VM_PORTS(6)
`endif
`ifdef VM7
    `VM_PORTS(7)
`endif
    .profile_finish());
  always @(negedge clk) begin
    vx = ix < T && ($urandom % 100 >= stall);
    vr = ir < T && ($urandom % 100 >= stall);
    yr = $urandom % 100 >= stall;
    adv = $urandom % 100 >= bubble;
  end
  always @(posedge clk) begin
    cyc <= cyc + 1; rst <= 0;
    if (!rst) begin
      if (vx && xr) ix <= ix + 1;
      if (vr && rr) ir <= ir + 1;
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
`ifdef VM6
    ok &= vm6_ok;
`endif
`ifdef VM7
    ok &= vm7_ok;
`endif
    if (ok) $display("MATCH %0d cycles in %0d host cycles, H %0.2f", T - W, cyc, real'(cyc) / T);
    $finish;
  end
endmodule
