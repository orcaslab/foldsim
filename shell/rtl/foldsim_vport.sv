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
//    once every write taken (c+1) * 2^FS host cycles or more before it has
//    its B (class 0: every write taken before it; with rd_new, its own
//    cycle's write too). The schedule proves, per member, that a read's row
//    was last written at least that long before (positions only stretch
//    under bubbles, so host cycles bound them from below). A ring of the
//    write count, sampled every 2^FS cycles, holds the counts of the window;
//    the count a read waits for is decided as it is taken.
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
    // Read-after-write by a table of write numbers instead of fence classes
    // (0: classes): 2^HZ entries, indexed by a hash of the row (its beat
    // offset without its KS low bits, which a row's alignment keeps zero),
    // each holding the number of the newest write taken of a row hashing
    // there; a read waits for that write's B while it is in flight (rows
    // that share an entry only make a read wait longer).
    parameter HZ          = 0,
    parameter KS          = 0,
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
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [63:0]          bytes,       // sampled at reset
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire                 move,
    output wire                 busy,
    // Telemetry: this cycle the head read waits on its fence or AXI room;
    // the head write waits (pair, room, AW); a read is on AXI.
    output wire [2:0]           stat,

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
    // Request counters: wrap-safe compares over at most QUEUE + OUTSTANDING
    // + WRITES_OUT requests, and the fence window's writes.
    localparam CW  = 16;
    localparam DB  = DW / 8;
    localparam GROUP = DB / MW;

    // ---- read queue: {addr, len, tag, the write count it waits for}
    localparam HB  = (HZ > 0) ? HZ : 1;
    localparam RQW = OW + LW + TW + CW;
    reg  [RQW-1:0] rq_mem [QUEUE];
    reg  [QW:0]    rq_head, rq_tail;
    // ---- write queue: {addr, len, pair, epoch, read count at intake} + the row
    localparam WQW = OW + LW + 1 + 4 + CW;
    reg  [WQW-1:0] wq_mem   [QUEUE];
    reg  [DW-1:0]  wq_data  [QUEUE];
    reg  [MW-1:0]  wq_mask  [QUEUE];
    reg  [QW:0]    wq_head, wq_tail;

    // Counters, never reset (a kernel reset carries every transaction on AXI
    // to its end): writes taken / AWs on AXI / B's back; reads taken into the
    // queue / on AR / last beats back, and on AXI. Answers to reads from
    // before a reset (`stale`) are dropped.
    reg  [CW-1:0]  w_taken, w_aw, w_done, r_in, r_taken, r_done, r_out, stale;
    // Resets so far (mod 16): a pair taken before a reset has no read to wait for.
    reg  [3:0]     epoch;
    // The write count sampled every 2^FS cycles, FCLASSES of them.
    reg  [CW-1:0]  w_ring [FCLASSES];
    reg  [FS-1:0]  tick;
    reg  [FCW-1:0] ring_at;

    reg  [63:0]    base_q;
    reg            draining, filling;
    // The zero fill: the next burst's offset and the bytes left (40 bits: a
    // region below 1 TiB), counted down so no compare reaches across them.
    reg  [39:0]    fill_at, fill_left;

    // The head read, out of the queue's memory into registers.
    reg            h_valid;
    reg  [OW-1:0]  h_addr;
    reg  [LW-1:0]  h_len;
    reg  [TW-1:0]  h_tag;
    reg  [CW-1:0]  h_need;
    // Writes taken and not yet answered (w_taken - w_done), registered.
    reg  [CW-1:0]  w_inflight;
    // The head write's fields likewise; its row stays in the queue's memory.
    reg            g_valid;
    reg  [OW-1:0]  g_addr;
    reg  [LW-1:0]  g_len;
    reg            g_pair;
    reg  [3:0]     g_ep;
    reg  [CW-1:0]  g_rcnt;
    reg  [QW-1:0]  g_slot;

    // In-flight reads: their tags and beats, in order.
    reg  [TW+LW-1:0] of_mem [OUTSTANDING];
    reg  [OUW:0]     of_head, of_tail;
    reg  [LW-1:0]    r_beat;

    reg            w_busy;                  // a row's beats are leaving
    reg  [LW-1:0]  w_beat;
    reg  [QW-1:0]  w_slot;
    reg  [LW-1:0]  w_len;
    reg  [5:0]     fill_beat, fill_last;
    // A fill burst's beats: up to 64 (4 KiB), fewer for the region's last
    // bytes (whole beats: a region is whole beats).
    wire           fill_more = fill_left != 40'd0;
    wire [5:0]     fill_n    = (|fill_left[39:12]) ? 6'd63 : 6'(fill_left[11:6] - 6'd1);
    reg            fill_w;                  // a fill burst's beats are leaving

`ifdef SIMULATION
    // Power-up state (the FPGA's registers start at 0; draining from the start).
    initial begin
        rq_head = '0; rq_tail = '0; wq_head = '0; wq_tail = '0;
        w_taken = '0; w_aw = '0; w_done = '0; r_in = '0; r_taken = '0; r_done = '0; r_out = '0; stale = '0;
        epoch = '0; tick = '0; ring_at = '0; draining = 1'b1; filling = 1'b0; base_q = '0; fill_at = '0; fill_left = '0;
        h_valid = 1'b0; p_valid = 1'b0; g_valid = 1'b0; w_inflight = '0;
        for (int i = 0; i < (1 << ((HZ > 0) ? HZ : 1)); i++) hz_tab[i] = '0;
        r_fence_ok = 1'b0; r_room = 1'b0; of_head = '0; of_tail = '0; r_beat = '0;
        w_busy = 1'b0; w_beat = '0; fill_beat = '0; fill_last = '0; fill_w = 1'b0;
        m_arvalid = 1'b0; m_awvalid = 1'b0; m_wvalid = 1'b0; rsp_valid = 1'b0; rd_free = '0; wr_free = '0;
        for (int i = 0; i < FCLASSES; i++) w_ring[i] = '0;
    end
`endif

    wire take_rd = rd_valid && !draining && !reset;
    wire take_wr = wr_valid && !draining && !reset;

    // Intake.
    always @(posedge clk) begin
        if (take_rd)
            rq_mem[rq_tail[QW-1:0]] <= {rd_addr, rd_len, rd_tag, in_need};
        if (take_wr) begin
            wq_mem[wq_tail[QW-1:0]]  <= {wr_addr, wr_len, wr_pair, epoch, r_in + CW'(take_rd)};
            wq_data[wq_tail[QW-1:0]] <= wr_data;
            wq_mask[wq_tail[QW-1:0]] <= wr_mask;
        end
    end

    // Writes a read must see have their B, decided as it is taken: class 0,
    // every write taken before it (and its own cycle's, rd_new); class c,
    // those of the count sampled c samples ago, taken at least (c+1) 2^FS
    // host cycles before it (no later than its own count).
    wire [CW-1:0]  in_wcnt  = w_taken + CW'(rd_new && take_wr);
    wire [CW-1:0]  in_c     = w_ring[ring_at - rd_fence];
    wire [CW-1:0]  in_need_c = (rd_fence == 0) ? in_wcnt
                             : (($signed(in_c - in_wcnt) < 0) ? in_c : in_wcnt);
    // The write-number table (HZ > 0): a row's entry is its beat offset above
    // KS folded to HB bits. A write taken leaves its number (w_taken after
    // it) at the clock edge, so a read taken a cycle or more later reads it;
    // a write-first pair's read takes its own cycle's write's number.
    function automatic [HB-1:0] bucket_of(input [OW-1:0] a);
        bucket_of = '0;
        for (int i = KS; i < OW; i++)
            bucket_of[(i - KS) % HB] = bucket_of[(i - KS) % HB] ^ a[i];
    endfunction
    localparam NB = 1 << HB;
    reg  [CW-1:0]  hz_tab [NB];
    wire [HB-1:0]  in_bucket = bucket_of(rd_addr);
    wire [HB-1:0]  wr_bucket = bucket_of(wr_addr);
    wire [CW-1:0]  in_need_b = (rd_new && take_wr) ? w_taken + 1'b1 : hz_tab[in_bucket];
    always @(posedge clk)
        if (HZ > 0 && take_wr)
            hz_tab[wr_bucket] <= w_taken + 1'b1;
    wire [CW-1:0]  in_need   = (HZ > 0) ? in_need_b : in_need_c;

    // ---- AR: the head read, once its fence holds.
    wire [RQW-1:0] rh = rq_mem[rq_head[QW-1:0]];
    wire [OW-1:0]  rh_addr  = rh[RQW-1 -: OW];
    wire [LW-1:0]  rh_len   = rh[RQW-OW-1 -: LW];
    wire [TW-1:0]  rh_tag   = rh[RQW-OW-LW-1 -: TW];
    wire [CW-1:0]  rh_need  = rh[CW-1:0];
    // Two reads at the head, out of the queue's memory into registers: the
    // next (p) behind the head (h), so the head's fence is always computed
    // from registers (never through the queue's read), a cycle late (B's
    // only add: conservative). A next read moves up as the head leaves.
    reg            r_fence_ok;
    reg            p_valid;
    reg  [OW-1:0]  p_addr;
    reg  [LW-1:0]  p_len;
    reg  [TW-1:0]  p_tag;
    reg  [CW-1:0]  p_need;
    // (A difference's sign bit, not a compare after it.)
    // Table numbers (HZ > 0): the write is in flight while its number is in
    // (w_done, w_done + w_inflight] (a number left long ago is outside it).
    function automatic fence_ok(input [CW-1:0] need);
        /* verilator lint_off UNUSEDSIGNAL */
        reg [CW-1:0] dw;
        /* verilator lint_on UNUSEDSIGNAL */
        dw = w_done - need;
        fence_ok = (HZ > 0) ? ((need - w_done - 1'b1) >= w_inflight) : !dw[CW-1];
    endfunction
    // Room for another read on AXI, registered (one AR a cycle: a margin of one).
    reg            r_room;
    wire           ar_go    = h_valid && r_fence_ok && r_room && (!m_arvalid || m_arready) && !reset;
    wire           rq_any   = rq_head != rq_tail;
    wire           promote  = p_valid && (!h_valid || ar_go);
    wire           h_load   = rq_any && (!p_valid || promote);     // into p

    always @(posedge clk) begin
        r_fence_ok <= promote ? fence_ok(p_need) : fence_ok(h_need);
        r_room     <= r_out < CW'(OUTSTANDING - 1);
        if (h_load) begin
            p_addr   <= rh_addr;
            p_len    <= rh_len;
            p_tag    <= rh_tag;
            p_need   <= rh_need;
        end
        if (promote) begin
            h_addr   <= p_addr;
            h_len    <= p_len;
            h_tag    <= p_tag;
            h_need   <= p_need;
        end
        if (ar_go) begin
            m_araddr <= base_q + 64'({h_addr, 6'b0});
            m_arlen  <= 8'(h_len);
            of_mem[of_tail[OUW-1:0]] <= {h_tag, h_len};
        end
    end

    // ---- R: straight to the simulator, a beat a cycle.
    assign m_rready = 1'b1;
    wire [TW+LW-1:0] oh = of_mem[of_head[OUW-1:0]];
    wire r_last_beat = m_rvalid && (r_beat == oh[LW-1:0]);
    always @(posedge clk) begin
        rsp_valid <= m_rvalid && (stale == 0) && !reset;
        rsp_tag   <= oh[TW+LW-1 -: TW];
        rsp_beat  <= r_beat;
        rsp_last  <= r_last_beat;
        rsp_data  <= m_rdata;
    end

    // ---- AW/W: the head write once its pair's reads are back; its beats
    // from the held row.
    wire [WQW-1:0] wh = wq_mem[wq_head[QW-1:0]];
    wire           w_pair_ok = !g_pair || (g_ep != epoch) || ($signed(r_done - g_rcnt) >= 0);
    wire           w_room  = (w_aw - w_done) < CW'(WRITES_OUT);
    wire           wq_any  = wq_head != wq_tail;
    // (The next row's AW goes out in the cycle the current row's last beat
    // leaves: rows back to back, a beat a cycle.)
    wire           w_last_now = w_busy && (!m_wvalid || m_wready) && (w_beat == w_len);
    wire           aw_go   = g_valid && (!w_busy || w_last_now) && !fill_w && w_pair_ok && w_room && (!m_awvalid || m_awready);
    wire           g_load  = wq_any && (!g_valid || aw_go);
    wire           w_step  = w_busy && (!m_wvalid || m_wready);
    wire           w_end   = w_step && (w_beat == w_len);

    // A beat of the held row, its mask's bytes.
    wire [DW-1:0]  wrow = wq_data[w_slot];
    wire [MW-1:0]  wmsk = wq_mask[w_slot];
    // The row padded to whole beats, its mask as byte enables.
    wire [BEATS*512-1:0] wrow_p = (BEATS*512)'(wrow);
    wire [BEATS*64-1:0]  wbytes;
    for (genvar j = 0; j < BEATS * 64; j++) begin : g_bytes
        if (j < DB) begin : g_in
            assign wbytes[j] = wmsk[j / GROUP];
        end else begin : g_pad
            assign wbytes[j] = 1'b0;
        end
    end
    wire [511:0]   beat_d = wrow_p[32'(w_beat) * 512 +: 512];
    wire [63:0]    beat_s = wbytes[32'(w_beat) * 64 +: 64];

    // ---- zero fill (after a reset's drain): 4 KiB bursts of zeros.
    wire           fill_aw = filling && draining && !g_valid && !wq_any && !w_busy && !fill_w && w_room
                          && (!m_awvalid || m_awready) && fill_more;

    assign m_bready = 1'b1;
    assign busy     = draining;
    assign stat     = {r_out != 0, g_valid && !aw_go && !w_busy && !fill_w, h_valid && !ar_go};

    always @(posedge clk) begin
        if (g_load) begin
            g_addr <= wh[WQW-1 -: OW];
            g_len  <= wh[WQW-OW-1 -: LW];
            g_pair <= wh[CW+4];
            g_ep   <= wh[CW +: 4];
            g_rcnt <= wh[CW-1:0];
            g_slot <= wq_head[QW-1:0];
        end
    end

    always @(posedge clk) begin
        // samples of the write count
        tick <= tick + 1'b1;
        if (&tick) begin
            ring_at <= ring_at + 1'b1;
            w_ring[ring_at + 1'b1] <= w_taken;
        end
        if (take_rd) begin rq_tail <= rq_tail + 1'b1; r_in <= r_in + 1'b1; end
        if (take_wr) begin wq_tail <= wq_tail + 1'b1; w_taken <= w_taken + 1'b1; end
        // the read head and the one behind it
        if (h_load) rq_head <= rq_head + 1'b1;
        if (h_load) p_valid <= 1'b1;
        else if (promote) p_valid <= 1'b0;
        if (promote) h_valid <= 1'b1;
        else if (ar_go) h_valid <= 1'b0;
        // AR
        if (m_arvalid && m_arready) m_arvalid <= 1'b0;
        if (ar_go) begin
            m_arvalid <= 1'b1;
            of_tail   <= of_tail + 1'b1;
            r_taken   <= r_taken + 1'b1;
        end
        r_out <= r_out + CW'(ar_go) - CW'(r_last_beat);
        // R
        if (m_rvalid) begin
            r_beat <= r_last_beat ? '0 : r_beat + 1'b1;
            if (r_last_beat) begin
                of_head <= of_head + 1'b1;
                r_done  <= r_done + 1'b1;
                if (stale != 0) stale <= stale - 1'b1;
            end
        end
        // the write head
        if (g_load) wq_head <= wq_head + 1'b1;
        if (g_load) g_valid <= 1'b1;
        else if (aw_go) g_valid <= 1'b0;
        // AW / W
        if (m_awvalid && m_awready) m_awvalid <= 1'b0;
        if (aw_go) begin
            m_awvalid <= 1'b1;
            m_awaddr  <= base_q + 64'({g_addr, 6'b0});
            m_awlen   <= 8'(g_len);
            w_busy    <= 1'b1;
            w_beat    <= '0;
            w_len     <= g_len;
            w_slot    <= g_slot;
            w_aw      <= w_aw + 1'b1;
        end else if (fill_aw) begin
            m_awvalid <= 1'b1;
            m_awaddr  <= base_q + 64'(fill_at);
            m_awlen   <= 8'(fill_n);
            fill_w    <= 1'b1;
            fill_beat <= '0;
            fill_last <= fill_n;
            fill_at   <= fill_at + 40'({7'(fill_n) + 7'd1, 6'b0});
            fill_left <= fill_left - 40'({7'(fill_n) + 7'd1, 6'b0});
            w_taken   <= w_taken + 1'b1;
            w_aw      <= w_aw + 1'b1;
        end
        if (m_wvalid && m_wready) m_wvalid <= 1'b0;
        if (w_step) begin
            m_wvalid <= 1'b1;
            m_wdata  <= beat_d;
            m_wstrb  <= beat_s;
            m_wlast  <= (w_beat == w_len);
            w_beat   <= w_end ? '0 : w_beat + 1'b1;
            if (w_end && !aw_go) w_busy <= 1'b0;
        end else if (fill_w && (!m_wvalid || m_wready)) begin
            m_wvalid  <= 1'b1;
            m_wdata   <= '0;
            m_wstrb   <= '1;
            m_wlast   <= fill_beat == fill_last;
            fill_beat <= fill_beat + 1'b1;
            if (fill_beat == fill_last) fill_w <= 1'b0;
        end
        if (m_bvalid) w_done <= w_done + 1'b1;
        w_inflight <= w_inflight + CW'(take_wr || fill_aw) - CW'(m_bvalid);
        // Drain (reset / move): done once every queue is empty and every
        // transaction answered; then (reset) the fill; then free.
        if (draining && !rq_any && !h_valid && !p_valid && !wq_any && !g_valid && !w_busy && !fill_w
                && r_out == 0 && w_aw == w_done && !m_arvalid && !m_awvalid && !m_wvalid
                && !(filling && fill_more)) begin
            filling  <= 1'b0;
            draining <= 1'b0;
            base_q   <= base;
        end
        // Room, from registers (a cycle conservative): the queue's entries,
        // its head register, this cycle's intake.
        rd_free <= draining ? '0 : FW'(QUEUE - 32'(rq_tail - rq_head) - 32'(h_valid) - 32'(p_valid) - 32'(take_rd));
        wr_free <= draining ? '0 : FW'(QUEUE - 32'(wq_tail - wq_head) - 32'(g_valid) - 32'(take_wr));
        // Reset: drop the reads not yet on AR (and the answers to those on
        // AXI), keep the writes (they reach memory before the fill); a move
        // keeps everything and only drains.
        if (reset) begin
            rq_head   <= rq_tail;
            h_valid   <= 1'b0;
            p_valid   <= 1'b0;
            stale     <= r_out - CW'(r_last_beat);
            r_in      <= r_taken;
            epoch     <= epoch + 1'b1;
            draining  <= 1'b1;
            filling   <= (ZERO_FILL != 0);
            fill_at   <= '0;
            fill_left <= (ZERO_FILL != 0) ? 40'(bytes) : 40'd0;
            base_q    <= base;
            rd_free   <= '0;
            wr_free   <= '0;
        end else if (move) begin
            draining  <= 1'b1;
        end
    end
endmodule
