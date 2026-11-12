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

// A static port: one AXI4 512-bit master that carries every device-memory
// request the schedule placed on it (page rows, link-stream rows, virtual
// memories' rows, ring images), in the order the simulator issues them.
//
// The schedule places at most one read and one write on a port a position,
// so the port has a single request input of each kind and no arbitration.
// Requests carry their address (the simulator forms it: the layout gives
// every member a power-of-two stride, so an address is a concatenation of
// fields and a base), their beats and a tag the answer returns.
//
// One AXI id: AXI returns its reads in order, so the answers leave in the
// order of the reads, beat by beat (`rsp_*`, one 512-bit beat a cycle, its
// tag and its beat number), and nothing reorders them. Writes leave in order
// too; a row of k beats is held whole in the write queue and sent as k beats.
//
// Ordering AXI does not give, the port gives with two counters and a fence
// each, instead of a per-row hazard table:
//  - read after write: a read whose fence class is c (rd_fence) leaves only
//    once every write taken before (now - c * 2^FS) host cycles has its B.
//    The schedule proves, per member, that a read's row was last written at
//    least that long before (class 0: any write; the class's distance is a
//    lower bound the compiler checks over the frame). A ring of the write
//    count, sampled every 2^FS cycles, holds the counts of the window.
//  - write after read (a read-first pair, wr_pair): the write leaves once
//    every read taken before it has its last beat back.
// A late answer is a bubble in the simulator, never a wrong row.
//
// Room: `free` counts the requests the port may still take (QUEUE less those
// taken and not yet on AXI). The simulator issues nothing when free is 0.
//
// Reset: no request is taken; every request taken before it still goes to
// memory; then (ZERO_FILL) the port writes zeros over its region
// [base, base + bytes) in 4 KiB bursts, so that every row reads as zero, as
// the target's RAMs after a reset; free returns once memory has answered
// every transaction. A region move (`move`) is a reset without the fill
// whose new base takes effect once the old region's transactions are done.
module foldsim_vport #(
    // Write rows: data bits (a multiple of 8, up to 4096) and mask bits
    // (DW/8 byte enables, or 1 for whole rows).
    parameter DW          = 512,
    parameter MW          = DW / 8,
    // Tag bits a read carries back.
    parameter TW          = 8,
    // Beat-offset bits of an address within the port's region (bytes / 64).
    parameter OW          = 28,
    // Request-queue entries (reads and writes each).
    parameter QUEUE       = 32,
    // Reads in flight on AXI (requests).
    parameter OUTSTANDING = 64,
    // Writes on AXI waiting for their B.
    parameter WRITES_OUT  = 64,
    // Fence: classes (a power of two) and the sample interval's log2.
    parameter FCLASSES    = 16,
    parameter FS          = 8,
    parameter ZERO_FILL   = 1,
    localparam BEATS      = (DW + 511) / 512,
    localparam LW         = (BEATS > 1) ? $clog2(BEATS) : 1,
    localparam FW         = $clog2(QUEUE + 1),
    localparam FCW        = $clog2(FCLASSES)
) (
    input  wire                 clk,
    input  wire                 reset,

    input  wire                 rd_valid,
    input  wire [OW-1:0]        rd_addr,
    input  wire [LW-1:0]        rd_len,      // beats - 1
    input  wire [TW-1:0]        rd_tag,
    input  wire [FCW-1:0]       rd_fence,
    // The read sees the write taken in its own cycle (a write-first RAM's
    // same-row pair): it waits for that write's B too.
    input  wire                 rd_new,

    input  wire                 wr_valid,
    input  wire [OW-1:0]        wr_addr,
    input  wire [LW-1:0]        wr_len,
    input  wire                 wr_pair,
    input  wire [DW-1:0]        wr_data,
    input  wire [MW-1:0]        wr_mask,

    output reg                  rsp_valid,
    output reg  [TW-1:0]        rsp_tag,
    output reg  [LW-1:0]        rsp_beat,
    output reg                  rsp_last,
    output reg  [511:0]         rsp_data,

    output reg  [FW-1:0]        rd_free,
    output reg  [FW-1:0]        wr_free,

    input  wire [63:0]          base,
    input  wire [63:0]          bytes,
    input  wire                 move,
    output wire                 busy,

    output reg                  m_awvalid,
    input  wire                 m_awready,
    output reg  [63:0]          m_awaddr,
    output reg  [7:0]           m_awlen,
    output reg                  m_wvalid,
    input  wire                 m_wready,
    output reg  [511:0]         m_wdata,
    output reg  [63:0]          m_wstrb,
    output reg                  m_wlast,
    input  wire                 m_bvalid,
    output wire                 m_bready,
    output reg                  m_arvalid,
    input  wire                 m_arready,
    output reg  [63:0]          m_araddr,
    output reg  [7:0]           m_arlen,
    input  wire                 m_rvalid,
    output wire                 m_rready,
    input  wire [511:0]         m_rdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire                 m_rlast
    /* verilator lint_on UNUSEDSIGNAL */
);
    localparam QW  = $clog2(QUEUE);
    localparam OUW = $clog2(OUTSTANDING);
    localparam CW  = 24;                       // request counters (wrap-safe compares)
    localparam DB  = DW / 8;
    localparam GROUP = DB / MW;

    // ---- read queue: {addr, len, tag, fence, write count at intake}
    localparam RQW = OW + LW + TW + FCW + CW;
    reg  [RQW-1:0] rq_mem [QUEUE];
    reg  [QW:0]    rq_head, rq_tail;
    // ---- write queue: {addr, len, pair, read count at intake} + the row
    localparam WQW = OW + LW + 1 + 4 + CW;
    reg  [WQW-1:0] wq_mem   [QUEUE];
    reg  [DW-1:0]  wq_data  [QUEUE];
    reg  [MW-1:0]  wq_mask  [QUEUE];
    reg  [QW:0]    wq_head, wq_tail;

    // Counters, never reset (a kernel reset carries every transaction on AXI
    // to its end): writes taken / B's back; reads taken into the queue / on
    // AR / last beats back. Answers to reads from before a reset (`stale`)
    // are dropped.
    reg  [CW-1:0]  w_taken, w_done, r_in, r_taken, r_done;
    reg  [CW-1:0]  stale;
    reg  [CW-1:0]  w_aw;
    // Resets so far (mod 16): a pair taken before a reset has no read to wait for.
    reg  [3:0]     epoch;                    // AWs on AXI (writes and fill bursts)
    // The write count sampled every 2^FS cycles, FCLASSES of them.
    reg  [CW-1:0]  w_ring [FCLASSES];
    reg  [FS-1:0]  tick;
    reg  [FCW-1:0] ring_at;

    reg  [63:0]    base_q;
    reg            draining, filling;
    reg  [63:0]    fill_at;

`ifdef SIMULATION
    // Power-up state (the FPGA's registers start at 0; draining from the start).
    initial begin
        rq_head = '0;
        rq_tail = '0;
        wq_head = '0;
        wq_tail = '0;
        w_taken = '0;
        w_done = '0;
        r_in = '0;
        r_taken = '0;
        r_done = '0;
        stale = '0;
        w_aw = '0;
        epoch = '0;
        tick = '0;
        ring_at = '0;
        draining = 1'b1;
        filling = 1'b0;
        of_head = '0;
        of_tail = '0;
        r_beat = '0;
        w_busy = 1'b0;
        fill_beat = '0;
        fill_w = 1'b0;
        m_arvalid = 1'b0;
        m_awvalid = 1'b0;
        m_wvalid = 1'b0;
        rsp_valid = 1'b0;
        rd_free = '0;
        wr_free = '0;
        base_q = '0;
        fill_at = '0;
        for (int i = 0; i < FCLASSES; i++) w_ring[i] = '0;
    end
`endif

    wire take_rd = rd_valid && !draining && !reset;
    wire take_wr = wr_valid && !draining && !reset;

    // Intake.
    always @(posedge clk) begin
        if (take_rd)
            rq_mem[rq_tail[QW-1:0]] <= {rd_addr, rd_len, rd_tag, rd_fence, w_taken + CW'(rd_new && take_wr)};
        if (take_wr) begin
            wq_mem[wq_tail[QW-1:0]]  <= {wr_addr, wr_len, wr_pair, epoch, r_in + CW'(take_rd)};
            wq_data[wq_tail[QW-1:0]] <= wr_data;
            wq_mask[wq_tail[QW-1:0]] <= wr_mask;
        end
    end

    // ---- AR: the head read, once its fence holds.
    wire [RQW-1:0] rh = rq_mem[rq_head[QW-1:0]];
    wire [OW-1:0]  rh_addr  = rh[RQW-1 -: OW];
    wire [LW-1:0]  rh_len   = rh[RQW-OW-1 -: LW];
    wire [TW-1:0]  rh_tag   = rh[RQW-OW-LW-1 -: TW];
    wire [FCW-1:0] rh_fence = rh[CW +: FCW];
    wire [CW-1:0]  rh_wcnt  = rh[CW-1:0];
    // Writes that must have their B: class 0, every write taken before the
    // read; class c, the count c samples ago (no later than the read's own).
    wire [CW-1:0]  need_c   = w_ring[ring_at - rh_fence];
    wire [CW-1:0]  need     = (rh_fence == 0) ? rh_wcnt
                            : (($signed(need_c - rh_wcnt) < 0) ? need_c : rh_wcnt);
    wire           r_fence_ok = $signed(w_done - need) >= 0;
    wire           r_room   = (r_taken - r_done) < OUTSTANDING;
    wire           ar_go    = (rq_head != rq_tail) && r_fence_ok && r_room && (!m_arvalid || m_arready);

    // In-flight reads: their tags and beats, in order.
    reg  [TW+LW-1:0] of_mem [OUTSTANDING];
    reg  [OUW:0]     of_head, of_tail;
    reg  [LW-1:0]    r_beat;

    always @(posedge clk) begin
        if (ar_go) begin
            m_araddr <= base_q + 64'({rh_addr, 6'b0});
            m_arlen  <= 8'(rh_len);
            of_mem[of_tail[OUW-1:0]] <= {rh_tag, rh_len};
        end
    end

    // ---- R: straight to the simulator, a beat a cycle.
    assign m_rready = 1'b1;
    wire [TW+LW-1:0] oh = of_mem[of_head[OUW-1:0]];
    wire r_last_beat = m_rvalid && (r_beat == oh[LW-1:0]);
    always @(posedge clk) begin
        rsp_valid <= m_rvalid && (stale == 0);
        rsp_tag   <= oh[TW+LW-1 -: TW];
        rsp_beat  <= r_beat;
        rsp_last  <= r_last_beat;
        rsp_data  <= m_rdata;
    end

    // ---- AW/W: the head write once its pair's reads are back; its beats
    // from the held row.
    wire [WQW-1:0] wh = wq_mem[wq_head[QW-1:0]];
    wire [OW-1:0]  wh_addr = wh[WQW-1 -: OW];
    wire [LW-1:0]  wh_len  = wh[WQW-OW-1 -: LW];
    wire           wh_pair = wh[CW+4];
    wire [3:0]     wh_ep   = wh[CW +: 4];
    wire [CW-1:0]  wh_rcnt = wh[CW-1:0];
    wire           w_pair_ok = !wh_pair || (wh_ep != epoch) || ($signed(r_done - wh_rcnt) >= 0);
    wire           w_room  = (w_aw - w_done) < WRITES_OUT;
    reg            w_busy;                // a row's beats are leaving
    reg  [LW-1:0]  w_beat;
    reg  [QW-1:0]  w_slot;
    wire           wq_any  = (wq_head != wq_tail);
    wire           aw_go   = wq_any && !w_busy && w_pair_ok && w_room
                          && (!m_awvalid || m_awready);
    wire           w_step  = w_busy && (!m_wvalid || m_wready);

    // A beat of the held row, its mask's bytes.
    wire [DW-1:0]  wrow = wq_data[w_slot];
    wire [MW-1:0]  wmsk = wq_mask[w_slot];
    reg  [511:0]   beat_d;
    reg  [63:0]    beat_s;
    always @* begin
        beat_d = '0;
        beat_s = '0;
        for (int i = 0; i < 512; i++)
            if (32'(w_beat) * 512 + i < DW)
                beat_d[i] = wrow[32'(w_beat) * 512 + i];
        for (int i = 0; i < 64; i++)
            if (32'(w_beat) * 64 + i < DB)
                beat_s[i] = wmsk[(32'(w_beat) * 64 + i) / GROUP];
    end

    // ---- zero fill (after a reset's drain): 4 KiB bursts of zeros.
    reg  [5:0]     fill_beat;
    reg            fill_w;                // a fill burst's beats are leaving
    wire           fill_aw = filling && draining && !w_busy && !fill_w && !wq_any && w_room
                          && (!m_awvalid || m_awready) && (fill_at < bytes);

    assign m_bready = 1'b1;
    assign busy     = draining;

    always @(posedge clk) begin
        // samples of the write count
        tick <= tick + 1'b1;
        if (&tick) begin
            ring_at <= ring_at + 1'b1;
            w_ring[ring_at + 1'b1] <= w_taken;
        end
        if (take_rd) begin rq_tail <= rq_tail + 1'b1; r_in <= r_in + 1'b1; end
        if (take_wr) begin wq_tail <= wq_tail + 1'b1; w_taken <= w_taken + 1'b1; end
        // AR
        if (m_arvalid && m_arready) m_arvalid <= 1'b0;
        if (ar_go) begin
            m_arvalid <= 1'b1;
            rq_head   <= rq_head + 1'b1;
            of_tail   <= of_tail + 1'b1;
            r_taken   <= r_taken + 1'b1;
        end
        // R
        if (m_rvalid) begin
            r_beat <= r_last_beat ? '0 : r_beat + 1'b1;
            if (r_last_beat) begin
                of_head <= of_head + 1'b1;
                r_done  <= r_done + 1'b1;
                if (stale != 0) stale <= stale - 1'b1;
            end
        end
        // AW / W
        if (m_awvalid && m_awready) m_awvalid <= 1'b0;
        if (aw_go) begin
            m_awvalid <= 1'b1;
            m_awaddr  <= base_q + 64'({wh_addr, 6'b0});
            m_awlen   <= 8'(wh_len);
            w_busy    <= 1'b1;
            w_beat    <= '0;
            w_slot    <= wq_head[QW-1:0];
            w_aw      <= w_aw + 1'b1;
        end else if (fill_aw) begin
            w_aw      <= w_aw + 1'b1;
            m_awvalid <= 1'b1;
            m_awaddr  <= base_q + fill_at;
            m_awlen   <= 8'd63;
            fill_w    <= 1'b1;
            fill_beat <= '0;
            fill_at   <= fill_at + 64'd4096;
            w_taken   <= w_taken + 1'b1;
        end
        if (m_wvalid && m_wready) m_wvalid <= 1'b0;
        if (w_step) begin
            m_wvalid <= 1'b1;
            m_wdata  <= beat_d;
            m_wstrb  <= beat_s;
            m_wlast  <= (w_beat == wh_len);
            w_beat   <= w_beat + 1'b1;
            if (w_beat == wh_len) begin
                w_busy  <= 1'b0;
                wq_head <= wq_head + 1'b1;
            end
        end else if (fill_w && (!m_wvalid || m_wready)) begin
            m_wvalid  <= 1'b1;
            m_wdata   <= '0;
            m_wstrb   <= '1;
            m_wlast   <= &fill_beat;
            fill_beat <= fill_beat + 1'b1;
            if (&fill_beat) fill_w <= 1'b0;
        end
        if (m_bvalid) w_done <= w_done + 1'b1;
        // Drain (reset / move): done once every queue is empty and every
        // transaction answered; then (reset) the fill; then free.
        if (draining && rq_head == rq_tail && wq_head == wq_tail && !w_busy && !fill_w
                && r_taken == r_done && w_aw == w_done && !m_arvalid && !m_awvalid && !m_wvalid
                && !(filling && fill_at < bytes)) begin
            filling  <= 1'b0;
            draining <= 1'b0;
            base_q   <= base;
        end
        rd_free <= draining ? '0 : FW'(QUEUE - 32'(rq_tail - rq_head) - (take_rd ? 1 : 0) + (ar_go ? 1 : 0));
        wr_free <= draining ? '0 : FW'(QUEUE - 32'(wq_tail - wq_head) - (take_wr ? 1 : 0)
                                       + ((w_step && w_beat == wh_len) ? 1 : 0));
        // Reset: drop the reads not yet on AR (and their answers to come),
        // keep the writes (they reach memory before the fill); a move keeps
        // everything and only drains.
        if (reset) begin
            rq_head   <= rq_tail;
            r_in      <= r_taken + CW'(ar_go);
            epoch     <= epoch + 1'b1;
            stale     <= r_taken + CW'(ar_go) - r_done - CW'(r_last_beat);
            draining  <= 1'b1;
            filling   <= (ZERO_FILL != 0);
            fill_at   <= '0;
            base_q    <= base;
            rd_free   <= '0;
            wr_free   <= '0;
        end else if (move) begin
            draining  <= 1'b1;
        end
    end
endmodule
