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

// One AXI4 master (a foldsim_vmem controller) on S device-memory masters, its
// rows interleaved over them (foldsim --stream-stripe S): so that one
// controller moves S ports' bandwidth.
//
// The region is 2^RW bytes at a multiple of 2^RW; a request's row is its
// offset in the region >> SHIFT (a burst is one row, of 2^SHIFT bytes or
// less, as foldsim_vmem's layout keeps it). Its stripe is the row's low
// log2(S) bits, and it goes to master `stripe` at PORT_BASE[stripe] |
// {row >> log2(S), offset within the row}: each master holds every S-th row,
// packed in 2^RW / S bytes at a multiple of that (a direct HBM port reaches
// its own pseudo-channel). Bit selects and an S-way select only: no adder.
//
// Responses go back in request order, whatever order the masters answer in:
// R from the master of the oldest read (a FIFO of the reads' stripes, popped
// at each burst's last beat), B from the master of the oldest write (a FIFO of
// the writes' stripes). The other masters wait (their ready low), which AXI
// allows; in order across all ids is in order per id. W follows AW: a FIFO of
// the writes' stripes, popped at each burst's last W beat.
//
// AR, AW and W leave from a register a master; R and B leave from one register
// each (a beat a host cycle: the controller's own master moves no more), so
// every valid towards a master and towards the controller is a register.
//
// Nothing here is reset: as the controllers do, it carries every transaction
// through a kernel reset to its end, from a power-up state of 0.
module foldsim_vmem_stripe #(
    // The masters: a power of two.
    parameter S           = 2,
    parameter ID_W        = 4,
    // log2 of a row's bytes (a burst stays within its row).
    parameter SHIFT       = 6,
    // Reads and writes in flight (FIFO depths: a power of two).
    parameter RDEPTH      = 64,
    parameter WDEPTH      = 64,
    // log2 of the region's bytes (its base a multiple of them).
    parameter RW          = 30,
    // Master j's base (a multiple of 2^RW / S): bits [64 j +: 64].
    parameter [64*S-1:0] PORT_BASE = '0
) (
    input  wire                  clk,

    input  wire                  s_awvalid,
    output wire                  s_awready,
    input  wire [63:0]           s_awaddr,
    input  wire [ID_W-1:0]       s_awid,
    input  wire [7:0]            s_awlen,
    input  wire                  s_wvalid,
    output wire                  s_wready,
    input  wire [511:0]          s_wdata,
    input  wire [63:0]           s_wstrb,
    input  wire                  s_wlast,
    output reg                   s_bvalid,
    input  wire                  s_bready,
    output reg  [ID_W-1:0]       s_bid,
    input  wire                  s_arvalid,
    output wire                  s_arready,
    input  wire [63:0]           s_araddr,
    input  wire [ID_W-1:0]       s_arid,
    input  wire [7:0]            s_arlen,
    output reg                   s_rvalid,
    input  wire                  s_rready,
    output reg  [511:0]          s_rdata,
    output reg  [ID_W-1:0]       s_rid,
    output reg                   s_rlast,

    output reg                   m_awvalid [S],
    input  wire                  m_awready [S],
    output reg  [63:0]           m_awaddr  [S],
    output reg  [ID_W-1:0]       m_awid    [S],
    output reg  [7:0]            m_awlen   [S],
    output reg                   m_wvalid  [S],
    input  wire                  m_wready  [S],
    output reg  [511:0]          m_wdata   [S],
    output reg  [63:0]           m_wstrb   [S],
    output reg                   m_wlast   [S],
    input  wire                  m_bvalid  [S],
    output wire                  m_bready  [S],
    input  wire [ID_W-1:0]       m_bid     [S],
    output reg                   m_arvalid [S],
    input  wire                  m_arready [S],
    output reg  [63:0]           m_araddr  [S],
    output reg  [ID_W-1:0]       m_arid    [S],
    output reg  [7:0]            m_arlen   [S],
    input  wire                  m_rvalid  [S],
    output wire                  m_rready  [S],
    input  wire [511:0]          m_rdata   [S],
    input  wire [ID_W-1:0]       m_rid     [S],
    input  wire                  m_rlast   [S]
);
    localparam SB  = (S > 1) ? $clog2(S) : 1;
    localparam RPW = $clog2(RDEPTH);
    localparam WPW = $clog2(WDEPTH);

    // A request's stripe and its address on that stripe's master.
    // (The stripe's bits: none with one master.)
    localparam SBE = (S > 1) ? SB : 0;
    localparam PW  = RW - SBE;
    function automatic [SB-1:0] stripe_of(input [63:0] a);
        /* verilator lint_off UNUSEDSIGNAL */
        reg [63:0] x;
        /* verilator lint_on UNUSEDSIGNAL */
        x = a;
        stripe_of = (S > 1) ? x[SHIFT +: SB] : '0;
    endfunction
    function automatic [63:0] addr_on(input [63:0] a);
        /* verilator lint_off UNUSEDSIGNAL */
        reg [63:0] x;
        /* verilator lint_on UNUSEDSIGNAL */
        reg [RW-1:0] off;
        reg [PW-1:0] packed_off;
        x = a;
        off = x[RW-1:0];
        // {the row's bits above the stripe, the offset in the row}
        packed_off = PW'(((off >> (SHIFT + SBE)) << SHIFT) | (off & RW'((64'd1 << SHIFT) - 64'd1)));
        addr_on = PORT_BASE[64 * stripe_of(a) +: 64] | {{(64 - PW){1'b0}}, packed_off};
    endfunction

    // Order FIFOs: the reads' stripes (R), the writes' stripes for W and for B.
    reg [SB-1:0]  ro_q [RDEPTH];
    reg [RPW-1:0] ro_h, ro_t;
    reg [RPW:0]   ro_n;
    reg [SB-1:0]  wo_q [WDEPTH];
    reg [WPW-1:0] wo_h, wo_t;
    reg [WPW:0]   wo_n;
    reg [SB-1:0]  bo_q [WDEPTH];
    reg [WPW-1:0] bo_h, bo_t;
    reg [WPW:0]   bo_n;
`ifdef SIMULATION
    initial begin
        ro_h = '0; ro_t = '0; ro_n = '0;
        wo_h = '0; wo_t = '0; wo_n = '0;
        bo_h = '0; bo_t = '0; bo_n = '0;
        s_bvalid = 1'b0; s_rvalid = 1'b0;
        for (integer j = 0; j < S; ++j) begin
            m_awvalid[j] = 1'b0; m_wvalid[j] = 1'b0; m_arvalid[j] = 1'b0;
        end
    end
`endif

    // ---- AR: into the stripe's register, its stripe into the read order.
    wire [SB-1:0] ar_j = stripe_of(s_araddr);
    wire          ar_room = !m_arvalid[ar_j] || m_arready[ar_j];
    assign s_arready = ar_room && (ro_n != RDEPTH[RPW:0]);
    wire          ar_take = s_arvalid && s_arready;

    // ---- R: the oldest read's master, a beat at a time.
    wire [SB-1:0] r_j     = ro_q[ro_h];
    wire          r_have  = (ro_n != '0);
    wire          r_room  = !s_rvalid || s_rready;
    wire          r_take  = r_have && r_room && m_rvalid[r_j];
    wire          r_done  = r_take && m_rlast[r_j];
    for (genvar j = 0; j < S; ++j) begin : g_rready
        assign m_rready[j] = r_have && r_room && (r_j == j[SB-1:0]);
    end

    // ---- AW: into the stripe's register, its stripe into the W and B orders.
    wire [SB-1:0] aw_j = stripe_of(s_awaddr);
    wire          aw_room = !m_awvalid[aw_j] || m_awready[aw_j];
    assign s_awready = aw_room && (wo_n != WDEPTH[WPW:0]) && (bo_n != WDEPTH[WPW:0]);
    wire          aw_take = s_awvalid && s_awready;

    // ---- W: to the oldest write's master (its AW taken: W may lead AW to
    // the master by the register, as AXI allows).
    wire [SB-1:0] w_j    = wo_q[wo_h];
    wire          w_have = (wo_n != '0);
    wire          w_room = !m_wvalid[w_j] || m_wready[w_j];
    assign s_wready = w_have && w_room;
    wire          w_take = s_wvalid && s_wready;
    wire          w_done = w_take && s_wlast;

    // ---- B: the oldest write's master.
    wire [SB-1:0] b_j    = bo_q[bo_h];
    wire          b_have = (bo_n != '0);
    wire          b_room = !s_bvalid || s_bready;
    wire          b_take = b_have && b_room && m_bvalid[b_j];
    for (genvar j = 0; j < S; ++j) begin : g_bready
        assign m_bready[j] = b_have && b_room && (b_j == j[SB-1:0]);
    end

    always @(posedge clk) begin
        // Masters' request registers.
        for (integer j = 0; j < S; ++j) begin
            if (m_arvalid[j] && m_arready[j])
                m_arvalid[j] <= 1'b0;
            if (m_awvalid[j] && m_awready[j])
                m_awvalid[j] <= 1'b0;
            if (m_wvalid[j] && m_wready[j])
                m_wvalid[j] <= 1'b0;
        end
        if (ar_take) begin
            m_arvalid[ar_j] <= 1'b1;
            m_araddr[ar_j]  <= addr_on(s_araddr);
            m_arid[ar_j]    <= s_arid;
            m_arlen[ar_j]   <= s_arlen;
            ro_q[ro_t]      <= ar_j;
            ro_t            <= ro_t + 1'b1;
        end
        if (aw_take) begin
            m_awvalid[aw_j] <= 1'b1;
            m_awaddr[aw_j]  <= addr_on(s_awaddr);
            m_awid[aw_j]    <= s_awid;
            m_awlen[aw_j]   <= s_awlen;
            wo_q[wo_t]      <= aw_j;
            wo_t            <= wo_t + 1'b1;
            bo_q[bo_t]      <= aw_j;
            bo_t            <= bo_t + 1'b1;
        end
        if (w_take) begin
            m_wvalid[w_j] <= 1'b1;
            m_wdata[w_j]  <= s_wdata;
            m_wstrb[w_j]  <= s_wstrb;
            m_wlast[w_j]  <= s_wlast;
        end
        if (w_done)
            wo_h <= wo_h + 1'b1;
        ro_n <= ro_n + {{RPW{1'b0}}, ar_take} - {{RPW{1'b0}}, r_done};
        wo_n <= wo_n + {{WPW{1'b0}}, aw_take} - {{WPW{1'b0}}, w_done};
        bo_n <= bo_n + {{WPW{1'b0}}, aw_take} - {{WPW{1'b0}}, b_take};
        // R and B registers towards the controller.
        if (s_rvalid && s_rready)
            s_rvalid <= 1'b0;
        if (r_take) begin
            s_rvalid <= 1'b1;
            s_rdata  <= m_rdata[r_j];
            s_rid    <= m_rid[r_j];
            s_rlast  <= m_rlast[r_j];
        end
        if (r_done)
            ro_h <= ro_h + 1'b1;
        if (s_bvalid && s_bready)
            s_bvalid <= 1'b0;
        if (b_take) begin
            s_bvalid <= 1'b1;
            s_bid    <= m_bid[b_j];
            bo_h     <= bo_h + 1'b1;
        end
    end
endmodule
