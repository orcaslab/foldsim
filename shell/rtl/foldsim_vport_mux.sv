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

// N static ports (foldsim_vport) on one AXI4 512-bit master, each port its
// own AXI id (its index): AXI answers each id in order, so R beats and B's go
// back by their id (the data fans out, only the valids are steered) and a
// port sees its own answers in its own order. AR and AW are taken round
// robin among the ports offering one, from registers; W follows AW's order
// (a FIFO of the ports whose AW went out; a burst's last beat moves on).
// Nothing is reset: as the ports, it carries every transaction to its end
// from a power-up state of 0. Ports are packed vectors (port k at
// [k*W +: W]).
module foldsim_vport_mux #(
    parameter N    = 7,
    // Writes in flight whose W has not finished (AW order FIFO depth).
    parameter WQ   = 16,
    localparam IW  = (N > 1) ? $clog2(N) : 1
) (
    input  wire                 clk,

    input  wire [N-1:0]         s_awvalid,
    output wire [N-1:0]         s_awready,
    input  wire [N*64-1:0]      s_awaddr,
    input  wire [N*8-1:0]       s_awlen,
    input  wire [N-1:0]         s_wvalid,
    output wire [N-1:0]         s_wready,
    input  wire [N*512-1:0]     s_wdata,
    input  wire [N*64-1:0]      s_wstrb,
    input  wire [N-1:0]         s_wlast,
    output wire [N-1:0]         s_bvalid,
    input  wire [N-1:0]         s_arvalid,
    output wire [N-1:0]         s_arready,
    input  wire [N*64-1:0]      s_araddr,
    input  wire [N*8-1:0]       s_arlen,
    output wire [N-1:0]         s_rvalid,
    output wire [511:0]         s_rdata,
    output wire                 s_rlast,

    output reg                  m_awvalid,
    input  wire                 m_awready,
    output reg  [63:0]          m_awaddr,
    output reg  [IW-1:0]        m_awid,
    output reg  [7:0]           m_awlen,
    output reg                  m_wvalid,
    input  wire                 m_wready,
    output reg  [511:0]         m_wdata,
    output reg  [63:0]          m_wstrb,
    output reg                  m_wlast,
    input  wire                 m_bvalid,
    input  wire [IW-1:0]        m_bid,
    output reg                  m_arvalid,
    input  wire                 m_arready,
    output reg  [63:0]          m_araddr,
    output reg  [IW-1:0]        m_arid,
    output reg  [7:0]           m_arlen,
    input  wire                 m_rvalid,
    input  wire [IW-1:0]        m_rid,
    input  wire [511:0]         m_rdata,
    input  wire                 m_rlast
);
    localparam QW = $clog2(WQ);

    // Round robin: the first port from `rr` on with its bit set, and whether any.
    function automatic [IW:0] rr_pick(input [N-1:0] v, input [IW-1:0] rr);
        reg [IW-1:0] p;
        reg          a;
        /* verilator lint_off UNUSEDSIGNAL */
        reg [31:0]   k;
        /* verilator lint_on UNUSEDSIGNAL */
        p = rr;
        a = 1'b0;
        for (int j = N - 1; j >= 0; j--) begin
            k = (32'(rr) + 32'(j)) % N;
            if (v[k]) begin p = IW'(k); a = 1'b1; end
        end
        rr_pick = {a, p};
    endfunction

    // ---- R and B: back by id. The ports take every beat (rready = 1).
    for (genvar k = 0; k < N; k++) begin : g_rsp
        assign s_rvalid[k] = m_rvalid && (m_rid == IW'(k));
        assign s_bvalid[k] = m_bvalid && (m_bid == IW'(k));
    end
    assign s_rdata = m_rdata;
    assign s_rlast = m_rlast;

    // ---- AR: round robin from a register.
    reg  [IW-1:0] ar_rr;
    reg  [IW-1:0] ar_pick;
    reg           ar_any;
    always @* {ar_any, ar_pick} = rr_pick(s_arvalid, ar_rr);
    wire ar_take = ar_any && (!m_arvalid || m_arready);
    for (genvar k = 0; k < N; k++) begin : g_arr
        assign s_arready[k] = ar_take && (ar_pick == IW'(k));
    end
    always @(posedge clk) begin
        if (m_arvalid && m_arready) m_arvalid <= 1'b0;
        if (ar_take) begin
            m_arvalid <= 1'b1;
            m_araddr  <= s_araddr[32'(ar_pick) * 64 +: 64];
            m_arlen   <= s_arlen[32'(ar_pick) * 8 +: 8];
            m_arid    <= ar_pick;
            ar_rr     <= (32'(ar_pick) == N - 1) ? '0 : ar_pick + 1'b1;
        end
    end

    // ---- AW: round robin from a register, while the W order FIFO has room.
    reg  [IW-1:0] wo_mem [WQ];
    reg  [QW:0]   wo_head, wo_tail;
    reg  [IW-1:0] aw_rr;
    reg  [IW-1:0] aw_pick;
    reg           aw_any;
    always @* {aw_any, aw_pick} = rr_pick(s_awvalid, aw_rr);
    wire wo_room = (wo_tail - wo_head) < (QW+1)'(WQ);
    wire aw_take = aw_any && wo_room && (!m_awvalid || m_awready);
    for (genvar k = 0; k < N; k++) begin : g_awr
        assign s_awready[k] = aw_take && (aw_pick == IW'(k));
    end
    // ---- W: from the port whose AW is oldest without its last beat.
    wire          wo_any = wo_head != wo_tail;
    wire [IW-1:0] w_src  = wo_mem[wo_head[QW-1:0]];
    wire          w_take = wo_any && s_wvalid[w_src] && (!m_wvalid || m_wready);
    for (genvar k = 0; k < N; k++) begin : g_wr
        assign s_wready[k] = w_take && (w_src == IW'(k));
    end
    always @(posedge clk) begin
        if (m_awvalid && m_awready) m_awvalid <= 1'b0;
        if (aw_take) begin
            m_awvalid <= 1'b1;
            m_awaddr  <= s_awaddr[32'(aw_pick) * 64 +: 64];
            m_awlen   <= s_awlen[32'(aw_pick) * 8 +: 8];
            m_awid    <= aw_pick;
            aw_rr     <= (32'(aw_pick) == N - 1) ? '0 : aw_pick + 1'b1;
            wo_mem[wo_tail[QW-1:0]] <= aw_pick;
            wo_tail   <= wo_tail + 1'b1;
        end
        if (m_wvalid && m_wready) m_wvalid <= 1'b0;
        if (w_take) begin
            m_wvalid <= 1'b1;
            m_wdata  <= s_wdata[32'(w_src) * 512 +: 512];
            m_wstrb  <= s_wstrb[32'(w_src) * 64 +: 64];
            m_wlast  <= s_wlast[w_src];
            if (s_wlast[w_src]) wo_head <= wo_head + 1'b1;
        end
    end

`ifdef SIMULATION
    initial begin
        m_awvalid = 1'b0; m_wvalid = 1'b0; m_arvalid = 1'b0;
        ar_rr = '0; aw_rr = '0; wo_head = '0; wo_tail = '0;
    end
`endif
endmodule
