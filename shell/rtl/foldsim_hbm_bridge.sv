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

// A minimal bridge from a 512-bit AXI4 master in the kernel's clock to a
// 256-bit AXI4 port in another clock (the shell's direct HBM ports: 256 bits
// at the static region's 400 MHz), in place of a generic SmartConnect: an
// asynchronous FIFO on each of the five channels and a 2:1 width change.
// AW and AR bursts double their beats; a W entry leaves as two beats (its
// low half first, the strobes halved likewise); two R beats make one 512-bit
// beat. Ids pass through. Nothing is reset but the FIFOs' pointers (at
// either side's reset, both sides held in it together by the kernel), as the
// AXI channels' valids must be.
module foldsim_hbm_bridge #(
    parameter IDW   = 4,
    parameter AWID  = 64,
    // log2 of each FIFO's entries.
    parameter LOGA  = 3,      // AW, AR, B
    parameter LOGD  = 4       // W, R
) (
    input  wire                 s_clk,
    input  wire                 s_reset,
    input  wire                 m_clk,
    input  wire                 m_reset,

    // The kernel's 512-bit master.
    input  wire                 s_awvalid,
    output wire                 s_awready,
    input  wire [AWID-1:0]      s_awaddr,
    input  wire [IDW-1:0]       s_awid,
    input  wire [7:0]           s_awlen,
    input  wire                 s_wvalid,
    output wire                 s_wready,
    input  wire [511:0]         s_wdata,
    input  wire [63:0]          s_wstrb,
    input  wire                 s_wlast,
    output wire                 s_bvalid,
    input  wire                 s_bready,
    output wire [IDW-1:0]       s_bid,
    input  wire                 s_arvalid,
    output wire                 s_arready,
    input  wire [AWID-1:0]      s_araddr,
    input  wire [IDW-1:0]       s_arid,
    input  wire [7:0]           s_arlen,
    output wire                 s_rvalid,
    input  wire                 s_rready,
    output wire [511:0]         s_rdata,
    output wire [IDW-1:0]       s_rid,
    output wire                 s_rlast,

    // The 256-bit port.
    output wire                 m_awvalid,
    input  wire                 m_awready,
    output wire [AWID-1:0]      m_awaddr,
    output wire [IDW-1:0]       m_awid,
    output wire [7:0]           m_awlen,
    output wire                 m_wvalid,
    input  wire                 m_wready,
    output wire [255:0]         m_wdata,
    output wire [31:0]          m_wstrb,
    output wire                 m_wlast,
    input  wire                 m_bvalid,
    output wire                 m_bready,
    input  wire [IDW-1:0]       m_bid,
    output wire                 m_arvalid,
    input  wire                 m_arready,
    output wire [AWID-1:0]      m_araddr,
    output wire [IDW-1:0]       m_arid,
    output wire [7:0]           m_arlen,
    input  wire                 m_rvalid,
    output wire                 m_rready,
    input  wire [255:0]         m_rdata,
    input  wire [IDW-1:0]       m_rid,
    input  wire                 m_rlast
);
    // (Bursts of up to 128 512-bit beats: their doubled lengths fit 8 bits.)
    // ---- AW and AR: {addr, id, len} across; the length doubled.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [AWID+IDW+7:0] aw_d, ar_d;          // (len bit 7: bursts of at most 128 beats)
    /* verilator lint_on UNUSEDSIGNAL */
    wire aw_v, ar_v;
    foldsim_afifo #(.W(AWID + IDW + 8), .LOG(LOGA)) aw_q (
        .w_clk(s_clk), .w_reset(s_reset), .w_valid(s_awvalid), .w_ready(s_awready),
        .w_data({s_awaddr, s_awid, s_awlen}),
        .r_clk(m_clk), .r_reset(m_reset), .r_valid(aw_v), .r_ready(m_awready), .r_data(aw_d));
    assign m_awvalid = aw_v;
    assign {m_awaddr, m_awid} = aw_d[AWID+IDW+7:8];
    assign m_awlen = {aw_d[6:0], 1'b1};
    foldsim_afifo #(.W(AWID + IDW + 8), .LOG(LOGA)) ar_q (
        .w_clk(s_clk), .w_reset(s_reset), .w_valid(s_arvalid), .w_ready(s_arready),
        .w_data({s_araddr, s_arid, s_arlen}),
        .r_clk(m_clk), .r_reset(m_reset), .r_valid(ar_v), .r_ready(m_arready), .r_data(ar_d));
    assign m_arvalid = ar_v;
    assign {m_araddr, m_arid} = ar_d[AWID+IDW+7:8];
    assign m_arlen = {ar_d[6:0], 1'b1};

    // ---- W: an entry leaves as two 256-bit beats.
    wire [576:0] w_d;
    wire         w_v;
    reg          w_half;
    foldsim_afifo #(.W(577), .LOG(LOGD)) w_q (
        .w_clk(s_clk), .w_reset(s_reset), .w_valid(s_wvalid), .w_ready(s_wready),
        .w_data({s_wlast, s_wstrb, s_wdata}),
        .r_clk(m_clk), .r_reset(m_reset), .r_valid(w_v), .r_ready(m_wready && w_half), .r_data(w_d));
    assign m_wvalid = w_v;
    assign m_wdata  = w_half ? w_d[511:256] : w_d[255:0];
    assign m_wstrb  = w_half ? w_d[575:544] : w_d[543:512];
    assign m_wlast  = w_half && w_d[576];
    always @(posedge m_clk)
        if (m_reset) w_half <= 1'b0;
        else if (m_wvalid && m_wready) w_half <= !w_half;

    // ---- B: the ids back.
    foldsim_afifo #(.W(IDW), .LOG(LOGA)) b_q (
        .w_clk(m_clk), .w_reset(m_reset), .w_valid(m_bvalid), .w_ready(m_bready), .w_data(m_bid),
        .r_clk(s_clk), .r_reset(s_reset), .r_valid(s_bvalid), .r_ready(s_bready), .r_data(s_bid));

    // ---- R: two beats make one; the pair's id and last are the second's.
    reg  [255:0] r_lo;
    reg          r_half;
    wire         r_room;
    assign m_rready = !r_half || r_room;
    always @(posedge m_clk)
        if (m_reset) r_half <= 1'b0;
        else if (m_rvalid && m_rready) begin
            r_half <= !r_half;
            if (!r_half) r_lo <= m_rdata;
        end
    foldsim_afifo #(.W(512 + IDW + 1), .LOG(LOGD)) r_q (
        .w_clk(m_clk), .w_reset(m_reset), .w_valid(m_rvalid && r_half), .w_ready(r_room),
        .w_data({m_rlast, m_rid, m_rdata, r_lo}),
        .r_clk(s_clk), .r_reset(s_reset), .r_valid(s_rvalid), .r_ready(s_rready),
        .r_data({s_rlast, s_rid, s_rdata}));
`ifdef SIMULATION
    initial begin w_half = 1'b0; r_half = 1'b0; end
`endif
endmodule

// An asynchronous FIFO of 2^LOG entries: distributed RAM written in w_clk and
// read asynchronously into a register in r_clk, Gray-coded pointers crossing
// through two flops. First-word fall-through: r_valid with r_data held.
module foldsim_afifo #(
    parameter W   = 32,
    parameter LOG = 4
) (
    input  wire          w_clk,
    input  wire          w_reset,
    input  wire          w_valid,
    output wire          w_ready,
    input  wire [W-1:0]  w_data,
    input  wire          r_clk,
    input  wire          r_reset,
    output reg           r_valid,
    input  wire          r_ready,
    output reg  [W-1:0]  r_data
);
    localparam N = 1 << LOG;
    reg  [W-1:0] mem [N];
    reg  [LOG:0] wp, wg, rp, rg;              // binary and Gray pointers
    (* ASYNC_REG = "TRUE" *) reg [LOG:0] wg_r1, wg_r2;   // write Gray in r_clk
    (* ASYNC_REG = "TRUE" *) reg [LOG:0] rg_w1, rg_w2;   // read Gray in w_clk
    function automatic [LOG:0] gray(input [LOG:0] b);
        gray = b ^ (b >> 1);
    endfunction
    // Full: the write pointer a lap ahead of the (synchronized) read pointer.
    assign w_ready = !(wg == {~rg_w2[LOG:LOG-1], rg_w2[LOG-2:0]});
    wire push = w_valid && w_ready;
    always @(posedge w_clk) begin
        if (push) mem[wp[LOG-1:0]] <= w_data;
        if (w_reset) begin
            wp <= '0; wg <= '0;
        end else if (push) begin
            wp <= wp + 1'b1;
            wg <= gray(wp + 1'b1);
        end
        rg_w1 <= rg; rg_w2 <= rg_w1;
    end
    // Read side: the next entry loads the output register when it is empty
    // or taken.
    wire empty = (rg == wg_r2);
    wire load  = !empty && (!r_valid || r_ready);
    always @(posedge r_clk) begin
        wg_r1 <= wg; wg_r2 <= wg_r1;
        if (r_reset) begin
            rp <= '0; rg <= '0; r_valid <= 1'b0;
        end else begin
            if (load) begin
                r_data  <= mem[rp[LOG-1:0]];
                rp      <= rp + 1'b1;
                rg      <= gray(rp + 1'b1);
                r_valid <= 1'b1;
            end else if (r_ready)
                r_valid <= 1'b0;
        end
    end
`ifdef SIMULATION
    initial begin
        wp = '0; wg = '0; rp = '0; rg = '0; r_valid = 1'b0;
        wg_r1 = '0; wg_r2 = '0; rg_w1 = '0; rg_w2 = '0;
    end
`endif
endmodule
