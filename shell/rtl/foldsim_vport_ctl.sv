// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A static port (foldsim_vport) behind foldsim_vmem's ports and contract, so
// the kernel can take it in a controller's place (foldsim --vmem-controller
// vport): the same parameters (those of a host cache, forwarding and the
// hazard table are accepted and unused) and the same guarantees per
// (tag, instance, row) -- a read observes every earlier write of its row
// (fence class 0: every write taken before it has its B), a read-first
// pair's read the row before its write (the write waits for the read), a
// write-first pair's read the row after it (the read waits for the write's
// B); answers tagged with the instance, here in request order.
//
// Row `row` of instance `inst` lives at base + ((row * INSTANCES + inst) <<
// log2(ROW_BYTES)), as in foldsim_vmem; ROW_BYTES of at least 64 (rows that
// share 64-byte lines are foldsim_vmem's alone). A row of BEATS beats is one
// burst; its answer's beats are put together into the row here.
//
// Known-zero rows: where KNOWN_ZERO is set and `zero` is high when a reset
// starts, the port writes zeros over the image's `BYTES` bytes after the
// reset's drain (no bit per row on chip); free returns after it. Without
// `zero` it behaves as foldsim_vmem without the bits.
//
// `moving` is high from `move` until the drain ends; the new base takes
// effect then. Telemetry: [0] reads taken, [1] writes taken, [5] host cycles
// with no room; the others read zero.
module foldsim_vport_ctl #(
    parameter IW             = 8,
    parameter AW             = 10,
    parameter DW             = 512,
    parameter MW             = DW / 8,
    parameter RDW_READ_FIRST = 1,
    parameter QUEUE          = 32,
    parameter AXI_ID_W       = 4,
    parameter OUTSTANDING    = 16,
    parameter INSTANCES      = 1 << IW,
    /* verilator lint_off UNUSEDPARAM */
    parameter CACHE_ROWS     = 0,
    parameter KZ_SHIFT       = 0,
    parameter HAZARD_BITS    = 8,
    parameter SEQ_W          = 16,
    parameter FORWARD        = 1,
    /* verilator lint_on UNUSEDPARAM */
    parameter KNOWN_ZERO     = 1,
    parameter ROW_BYTES      = (((DW + 7) / 8) > 64) ? (1 << $clog2((DW + 7) / 8)) : 64,
    parameter TAGW           = 0,
    parameter WRITES_OUT     = 128,
    parameter TELEMETRY      = 1,
    // The image's bytes (the zero fill's extent; the compiler's, from
    // foldsim_sim.json: its rows times INSTANCES times ROW_BYTES).
    parameter longint BYTES  = 0,
    localparam TEL_N         = 7,
    localparam TEL_W         = 48
) (
    input  wire                        clk,
    input  wire                        reset,

    input  wire                        rd_valid,
    input  wire [IW+TAGW-1:0]          rd_inst,
    input  wire [AW-1:0]               rd_row,

    input  wire                        wr_valid,
    input  wire [IW+TAGW-1:0]          wr_inst,
    input  wire [AW-1:0]               wr_row,
    input  wire [DW-1:0]               wr_data,
    input  wire [MW-1:0]               wr_mask,

    output reg                         rsp_valid,
    output reg  [IW+TAGW-1:0]          rsp_inst,
    output wire [DW-1:0]               rsp_data,

    output reg  [$clog2(QUEUE+1)-1:0]  free,

    input  wire [63:0]                 base,
    input  wire                        zero,
    input  wire                        move,
    output wire                        moving,

    output wire                        m_awvalid,
    input  wire                        m_awready,
    output wire [63:0]                 m_awaddr,
    output wire [AXI_ID_W-1:0]         m_awid,
    output wire [7:0]                  m_awlen,
    output wire                        m_wvalid,
    input  wire                        m_wready,
    output wire [511:0]                m_wdata,
    output wire [63:0]                 m_wstrb,
    output wire                        m_wlast,
    input  wire                        m_bvalid,
    output wire                        m_bready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [AXI_ID_W-1:0]         m_bid,
    /* verilator lint_on UNUSEDSIGNAL */
    output wire                        m_arvalid,
    input  wire                        m_arready,
    output wire [63:0]                 m_araddr,
    output wire [AXI_ID_W-1:0]         m_arid,
    output wire [7:0]                  m_arlen,
    input  wire                        m_rvalid,
    output wire                        m_rready,
    input  wire [511:0]                m_rdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [AXI_ID_W-1:0]         m_rid,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire                        m_rlast,

    output wire [TEL_N*TEL_W-1:0]      tel
);
    localparam DB    = (DW + 7) / 8 * 8;
    localparam RB    = $clog2(ROW_BYTES);
    localparam BEATS = (DB + 511) / 512;          // a row's data beats (its burst)
    localparam LW    = (BEATS > 1) ? $clog2(BEATS) : 1;
    localparam KB    = $clog2(INSTANCES);
    localparam POW2  = ((INSTANCES & (INSTANCES - 1)) == 0) ? 1 : 0;
    localparam OW    = AW + KB + RB - 6;       // a row's beat offset bits
    localparam IT    = IW + TAGW;
    localparam FW    = $clog2(QUEUE + 1);

`ifdef SIMULATION
    initial begin
        if (ROW_BYTES < 64 || (ROW_BYTES & (ROW_BYTES - 1)) != 0 || ROW_BYTES * 8 < DB)
            $error("foldsim_vport_ctl: ROW_BYTES must be a power of two of at least 64 and DW/8");
        if (BYTES % 64 != 0)
            $error("foldsim_vport_ctl: the image must be whole beats");
    end
`endif

    // A row's beat offset: (row * INSTANCES + inst) << (RB - 6).
    function automatic [OW-1:0] offset_of(input [AW-1:0] row, input [IW-1:0] inst);
        if (POW2 != 0)
            offset_of = ((OW'(row) << KB) | (OW'(inst) & OW'(INSTANCES - 1))) << (RB - 6);
        else
            offset_of = OW'((OW'(row) * OW'(INSTANCES) + OW'(inst)) << (RB - 6));
    endfunction

    // Inputs registered (the multiply, where INSTANCES is not a power of two,
    // has its cycle); a read and a write of one row of one instance in a
    // host cycle are a pair.
    reg            q_rd, q_wr, q_pair;
    reg  [OW-1:0]  q_raddr, q_waddr;
    reg  [IT-1:0]  q_rtag;
    reg  [DW-1:0]  q_wdata;
    reg  [MW-1:0]  q_wmask;
    always @(posedge clk) begin
        q_rd    <= rd_valid && !reset;
        q_wr    <= wr_valid && !reset;
        q_pair  <= rd_valid && wr_valid && rd_inst == wr_inst && rd_row == wr_row;
        q_raddr <= offset_of(rd_row, rd_inst[IW-1:0]);
        q_waddr <= offset_of(wr_row, wr_inst[IW-1:0]);
        q_rtag  <= rd_inst;
        q_wdata <= wr_data;
        q_wmask <= wr_mask;
    end

    wire [FW-1:0]  rd_free, wr_free;
    wire           p_rsp_valid, p_rsp_last;
    wire [IT-1:0]  p_rsp_tag;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [LW-1:0]  p_rsp_beat;            // (unused with one-beat rows)
    /* verilator lint_on UNUSEDSIGNAL */
    wire [511:0]   p_rsp_data;
    wire           busy;

    foldsim_vport #(
        .DW(DB), .MW(MW), .TW(IT), .OW(OW), .QUEUE(QUEUE), .OUTSTANDING(OUTSTANDING), .WRITES_OUT(WRITES_OUT),
        .ZERO_FILL(KNOWN_ZERO)
    ) port (
        .clk(clk), .reset(reset),
        .rd_valid(q_rd), .rd_addr(q_raddr), .rd_len(LW'(BEATS - 1)), .rd_tag(q_rtag), .rd_fence(4'd0),
        .rd_new(q_pair && (RDW_READ_FIRST == 0)),
        .wr_valid(q_wr), .wr_addr(q_waddr), .wr_len(LW'(BEATS - 1)), .wr_pair(q_pair && (RDW_READ_FIRST != 0)),
        .wr_data(DB'(q_wdata)), .wr_mask(q_wmask),
        .rsp_valid(p_rsp_valid), .rsp_tag(p_rsp_tag), .rsp_beat(p_rsp_beat), .rsp_last(p_rsp_last), .rsp_data(p_rsp_data),
        .rd_free(rd_free), .wr_free(wr_free),
        .base(base), .bytes(zero ? 64'(BYTES) : 64'd0), .move(move), .busy(busy),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awlen(m_awlen),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_araddr(m_araddr), .m_arlen(m_arlen), .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rdata(m_rdata),
        .m_rlast(m_rlast));
    assign m_awid = '0;
    assign m_arid = '0;
    assign moving = busy;

    // The answer's beats put together into its row.
    if (BEATS > 1) begin : g_rows
        /* verilator lint_off UNUSEDSIGNAL */
        reg [BEATS*512-1:0] row_q;
        /* verilator lint_on UNUSEDSIGNAL */
        always @(posedge clk)
            if (p_rsp_valid)
                row_q[32'(p_rsp_beat) * 512 +: 512] <= p_rsp_data;
        assign rsp_data = row_q[DW-1:0];
    end else begin : g_beat
        /* verilator lint_off UNUSEDSIGNAL */
        reg [511:0] row_q;
        /* verilator lint_on UNUSEDSIGNAL */
        always @(posedge clk)
            if (p_rsp_valid) row_q <= p_rsp_data;
        assign rsp_data = row_q[DW-1:0];
    end
    always @(posedge clk) begin
        rsp_valid <= p_rsp_valid && p_rsp_last;
        rsp_inst  <= p_rsp_tag;
    end

    // Room: the port's, less the requests in the input register.
    wire [FW-1:0] rd_room  = rd_free - FW'(q_rd);
    wire [FW-1:0] wr_room  = wr_free - FW'(q_wr);
    wire [FW-1:0] room_min = (rd_room < wr_room) ? rd_room : wr_room;
    always @(posedge clk) begin
        if (reset)
            free <= '0;
        else
            free <= room_min;
    end

    // Telemetry.
    reg [TEL_W-1:0] t_rd, t_wr, t_full;
    always @(posedge clk) begin
        if (reset) begin
            t_rd <= '0; t_wr <= '0; t_full <= '0;
        end else if (TELEMETRY != 0) begin
            t_rd   <= t_rd + TEL_W'(q_rd);
            t_wr   <= t_wr + TEL_W'(q_wr);
            t_full <= t_full + TEL_W'(free == 0);
        end
    end
    assign tel = {TEL_W'(0), t_full, TEL_W'(0), TEL_W'(0), TEL_W'(0), t_wr, t_rd};

`ifdef SIMULATION
    initial begin
        rsp_valid = 1'b0;
        free = '0;
        q_rd = 1'b0;
        q_wr = 1'b0;
    end
`endif
endmodule
