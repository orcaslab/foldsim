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

// A static port: one AXI4 master that carries every device-memory request
// the schedule placed on it (page rows, link-stream rows, virtual memories'
// rows, ring images), in the order the simulator issues them.
//
// The schedule places at most one read and one write on a port a position,
// so the port has a single request input of each kind and no arbitration.
// Requests carry their address (a beat offset in the port's region: the
// simulator forms it), their beats and a tag the answer returns.
//
// One AXI id: AXI returns its reads in order, so the answers leave in the
// order of the reads, beat by beat (`rsp_*`, one 512-bit beat a cycle, its
// tag and its beat number), and nothing reorders them. Writes leave in order
// too; a row is held whole in the write queue and sent as its beats.
//
// Ordering AXI does not give, the port gives with counters and a fence each,
// instead of a per-row hazard table:
//  - read after write: a read whose fence class is c (rd_fence) leaves only
//    once every write taken (c+1) * 2^FS host cycles or more before it has
//    its B (class 0: every write taken before it; with rd_new, its own
//    cycle's write too). Or (HZ > 0) a table of write numbers by a hash of
//    the row: the read waits for the B of the newest write of its entry.
//  - write after read (a read-first pair, wr_pair): the write leaves once
//    every read taken up to its cycle has its last beat back.
// A late answer is a bubble in the simulator, never a wrong row.
//
// Bursts (MAX_BURST > 1): while the head request cannot leave for want of
// AXI credit (AXI_RD_OUT / AXI_WR_OUT bursts in flight, 16 on the shell's
// HBM ports) or channel room, the next request joins it when its beats
// follow the previous request's in memory within one 4 KiB page (decided as
// it is taken: a bit a request), the burst stays within MAX_BURST beats, and
// the joining request's own fence (or pair) already holds. Answers, fences and pairs stay per request; only the AXI
// transactions are fewer.
//
// Ring copies (COPY): a read with rd_copy sends its answer's beats into the
// port's own write queue, as a write of rd_copy_addr, instead of to the
// simulator; it answers once, tag only (rsp_copied), when the copy's last
// beat is in the queue. The copy's write slot is taken with the read (wr_free
// counts it), so the copy leaves in request order and never waits behind a
// later write; ahead of its own cycle's write, or (rd_new: the copy is of the
// row that write makes) just after it. A read with rd_after_copies leaves only once every copy taken
// before it has its B (a ring prefetch of the image the copies write).
//
// Room: rd_free / wr_free count the requests the port may still take (a copy
// read takes a read and a write). The simulator issues nothing past them.
//
// Reset: no request is taken; reads not yet on AXI are dropped and answers
// to those on AXI are not delivered; every write taken before it (and every
// copy that answered) still goes to memory; then, where ZERO_FILL is set and
// `zero` is high, the port writes zeros over its region [base, base + bytes)
// in 4 KiB bursts so that every row reads zero -- except while the region is
// clean (`zero` rose since the last write taken: the host cleared it); free
// returns once memory has answered every transaction. A region move (`move`)
// only drains; the new base takes effect once nothing is in flight.
//
// Clocks (ASYNC): the AXI side runs in m_clk at 256 bits (the shell's direct
// HBM ports: 256 bits at the static region's clock); the write queue's rows
// are read there half a beat at a time (the queue is the clock crossing), the
// read beats cross in a FIFO, requests in small FIFOs, B's and the queue's
// release as Gray counters. Without ASYNC the AXI side is 512 bits in clk.
// Nothing on the AXI side is reset (a kernel reset carries every AXI
// transaction to its end); FPGA registers start at zero.
module foldsim_vport #(
    // Write rows: data bits (a multiple of 8) and mask bits (DW/8 byte
    // enables, or fewer: a bit for DW/8/MW bytes).
    parameter DW           = 512,
    parameter MW           = DW / 8,
    // Tag bits a read carries back.
    parameter TW           = 8,
    // Beat-offset bits of an address within the port's region (bytes / 64).
    parameter OW           = 28,
    // Beats a read (a copied read: at most a write row's beats).
    parameter LMAX         = (DW + 511) / 512,
    // Request-queue entries (reads and writes each).
    parameter QUEUE        = 32,
    // Read requests in flight (between the head and their last beat; two
    // fewer than the power of two at or above it).
    parameter OUTSTANDING  = 32,
    // Write bursts on AXI waiting for their B where AXI_WR_OUT is 0.
    parameter WRITES_OUT   = 64,
    // AXI transactions in flight (0: OUTSTANDING reads / WRITES_OUT writes).
    parameter AXI_RD_OUT   = 16,
    parameter AXI_WR_OUT   = 16,
    // Beats a merged burst (1: no merging).
    parameter MAX_BURST    = 8,
    // Fence: classes (a power of two) and the sample interval's log2.
    parameter FCLASSES     = 16,
    parameter FS           = 8,
    parameter ZERO_FILL    = 1,
    // Read-after-write by a table of write numbers instead of fence classes
    // (0: classes): 2^HZ entries, indexed by a hash of the row (its beat
    // offset without its KS low bits), each the number of the newest write
    // taken of a row hashing there.
    parameter HZ           = 0,
    parameter KS           = 0,
    // The write number a read waits for comes in with it (rd_need, 1) from a
    // table its sender keeps (a block RAM looked up a host cycle ahead,
    // foldsim_vport_ctl), instead of the port's own (0). Numbers are this
    // port's: a write taken in a host cycle is number w_count + 1 of that
    // cycle (fill bursts count too; copies do not).
    parameter NEED_IN      = 0,
    // Ring copies (rd_copy, rd_after_copies).
    parameter COPY         = 0,
    // AXI side in m_clk at 256 bits.
    parameter ASYNC        = 0,
    // `base` is aligned to the region's span (2^(OW+6) bytes): addresses
    // are base's high bits and the offset, no adder.
    parameter BASE_ALIGNED = 0,
    localparam BEATS       = (DW + 511) / 512,
    localparam LW          = (BEATS > 1) ? $clog2(BEATS) : 1,
    localparam LRW         = (LMAX > 1) ? $clog2(LMAX) : 1,
    localparam FW          = $clog2(QUEUE + 1),
    localparam FCW         = $clog2(FCLASSES),
    localparam AXW         = (ASYNC != 0) ? 256 : 512
) (
    input  wire                 clk,
    input  wire                 reset,

    input  wire                 rd_valid,
    input  wire [OW-1:0]        rd_addr,
    input  wire [LRW-1:0]       rd_len,      // beats - 1
    input  wire [TW-1:0]        rd_tag,
    input  wire [FCW-1:0]       rd_fence,
    // The read sees the write taken in its own cycle (a write-first RAM's
    // same-row pair): it waits for that write's B too.
    input  wire                 rd_new,
    input  wire                 rd_copy,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [OW-1:0]        rd_copy_addr,   // (COPY)
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire                 rd_after_copies,

    input  wire                 wr_valid,
    input  wire [OW-1:0]        wr_addr,
    input  wire [LW-1:0]        wr_len,
    input  wire                 wr_pair,
    input  wire [DW-1:0]        wr_data,
    input  wire [MW-1:0]        wr_mask,

    output reg                  rsp_valid,
    output reg  [TW-1:0]        rsp_tag,
    output reg  [LRW-1:0]       rsp_beat,
    output reg                  rsp_last,
    output reg                  rsp_copied,
    output reg  [511:0]         rsp_data,

    output reg  [FW-1:0]        rd_free,
    output reg  [FW-1:0]        wr_free,

    input  wire [63:0]          base,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [63:0]          bytes,       // sampled at reset (whole beats)
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire                 zero,
    input  wire                 move,
    output wire                 busy,
    // Telemetry: this cycle the head read waits (fence, credit, channel);
    // the head write waits; a read is on AXI.
    output wire [2:0]           stat,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [15:0]          rd_need,     // (NEED_IN)
    /* verilator lint_on UNUSEDSIGNAL */
    output wire [15:0]          w_count,

    /* verilator lint_off UNUSEDSIGNAL */
    input  wire                 m_clk,       // (ASYNC)
    /* verilator lint_on UNUSEDSIGNAL */
    output reg                  m_awvalid,
    input  wire                 m_awready,
    output reg  [63:0]          m_awaddr,
    output reg  [7:0]           m_awlen,
    output reg                  m_wvalid,
    input  wire                 m_wready,
    output reg  [AXW-1:0]       m_wdata,
    output reg  [AXW/8-1:0]     m_wstrb,
    output reg                  m_wlast,
    input  wire                 m_bvalid,
    output wire                 m_bready,
    output reg                  m_arvalid,
    input  wire                 m_arready,
    output reg  [63:0]          m_araddr,
    output reg  [7:0]           m_arlen,
    input  wire                 m_rvalid,
    output wire                 m_rready,
    input  wire [AXW-1:0]       m_rdata,
    input  wire                 m_rlast
);
    localparam QW    = $clog2(QUEUE);
    localparam OFD   = (OUTSTANDING > 32) ? (1 << $clog2(OUTSTANDING)) : 32;
    localparam OFW   = $clog2(OFD);
    // Request counters: wrap-safe compares over the requests in the port and
    // the fence window's writes; copies (fewer) on CCW bits.
    localparam CW    = 16;
    localparam CCW   = 10;
    localparam RDB   = (AXI_RD_OUT != 0) ? AXI_RD_OUT : OUTSTANDING;
    localparam WRB   = (AXI_WR_OUT != 0) ? AXI_WR_OUT : WRITES_OUT;
    localparam MB    = (MAX_BURST > 1) ? 1 : 0;
    localparam CP    = (COPY != 0) ? 1 : 0;
    // A burst's request counts ride with it to its B where bursts merge
    // requests or carry copies.
    localparam BQ    = (MB != 0 || CP != 0) ? 1 : 0;
    localparam BQW   = (WRB > 1) ? $clog2(WRB) : 1;
    localparam NRW   = $clog2(MAX_BURST + 1);
    localparam BLW   = 6;                       // a burst's beats - 1 (64: a 4 KiB fill burst)
    localparam CRW   = BQW + 2;                 // B counter (Gray) bits
    localparam ACW   = $clog2(RDB + 1) + 1;     // bursts on AXI
    localparam BCW   = $clog2(WRB + 1) + 1;
    localparam DB    = DW / 8;
    localparam GROUP = DB / MW;
    // Writes' bursts trimmed to their beats with a byte enabled (see intake).
    localparam TRIM  = (BEATS > 1 && MW > 1 && MB == 0) ? 1 : 0;
    // Copied beats in a beat queue of their own (COPY with rows of several
    // beats): 512 bits whatever the row, instead of a second write source on
    // every beat of the row memory (see the copy beats below).
    localparam CB    = (CP != 0 && BEATS > 1) ? 1 : 0;
    localparam CBL   = 5;                       // its entries' log2 (a LUTRAM's depth)
    localparam H     = 512 / AXW;               // AXI beats a 512-bit beat
    localparam HB_   = (HZ > 0) ? HZ : 1;
    // Read-after-write by write numbers (the port's table or the sender's).
    localparam TAB   = (HZ > 0 || NEED_IN != 0) ? 1 : 0;
    localparam NB    = (HZ > 0 && NEED_IN == 0) ? (1 << HB_) : 1;
    localparam LOGA  = 3;                       // request FIFOs (ASYNC)
    localparam LOGR  = 4;                       // read beats (ASYNC)

    // =================================================================
    // Kernel side (clk)
    // =================================================================
    reg            draining;
    wire           drained;                    // every queue empty, every transaction answered
    /* verilator lint_off UNUSEDSIGNAL */
    wire           cw_en, cw_done, cq_adv;     // a copied beat / a copy's last beat / a copy slot leaves (COPY)
    /* verilator lint_on UNUSEDSIGNAL */
    wire           rb_valid, rb_last;          // read beats from the AXI side
    wire [511:0]   rb_data;
    reg            s_busy;                     // the AXI side streams a burst's rows
    wire           take_rd = rd_valid && !draining && !reset;
    wire           take_wr = wr_valid && !draining && !reset;
    wire           take_cp = take_rd && rd_copy && (CP != 0);

    // Counters, never reset: writes taken / B'd;
    // reads taken / done (last beat); copies taken / B'd. Answers to reads
    // from before a reset (`stale`) are dropped.
    reg  [CW-1:0]  w_taken, w_done, w_inflight, r_in, r_done;
    reg  [OFW:0]   stale;
    reg  [CCW-1:0] c_in, c_done;
    // Resets so far (mod 16): a pair taken before a reset has no read to wait for.
    reg  [3:0]     epoch;
    // The write count sampled every 2^FS cycles, FCLASSES of them.
    reg  [CW-1:0]  w_ring [FCLASSES];
    reg  [FS-1:0]  tick;
    reg  [FCW-1:0] ring_at;

    reg  [63:0]    base_q;
    reg            fill_go, clean, z_q;
    // The zero fill: the next burst's beat offset and the beats left.
    reg  [OW-1:0]  fill_at;
    reg  [OW:0]    fill_left;
    wire           fill_more = (ZERO_FILL != 0) && fill_left != '0;
    wire [BLW-1:0] fill_n    = (|fill_left[OW:6]) ? BLW'(63) : BLW'(fill_left[5:0] - 6'd1);

    // ---- read queue: {addr, len, tag, need, copy, after, copies before,
    // contiguous with the read before it in one page}
    // (The fields of a request as below, RQF bits; the queue keeps only the
    // fields its features use: copies' with COPY, the contiguity bit with
    // MAX_BURST > 1.)
    localparam RQF = OW + LRW + TW + CW + 3 + CCW;
    localparam RQW = OW + LRW + TW + CW + ((COPY != 0) ? 2 + CCW : 0) + ((MAX_BURST > 1) ? 1 : 0);
    reg  [RQW-1:0] rq_mem [QUEUE];
    reg  [QW:0]    rq_head, rq_tail;
    // ---- write queue: meta {addr, len, pair, epoch, reads taken by it};
    // rows in the data memory below (read on the AXI side).
    localparam WQF = OW + LW + 1 + 4 + CW + 1;
    localparam WQW = WQF - ((MAX_BURST > 1) ? 0 : 1);
    // (TRIM: the write's first beat in its row, stored above the fields.)
    localparam WQS = WQW + ((TRIM != 0) ? LW : 0);
    reg  [WQS-1:0] wq_mem [QUEUE];
    reg  [QW:0]    wq_head, wq_tail;
    // Slots the AXI side has read (its rows may be overwritten), in clk.
    wire [QW:0]    w_rel;
    // A write's slot: after a copy read's slot taken in its cycle -- unless
    // the read sees that write (rd_new, a write-first pair): its copy then
    // waits for the write's B, so the copy slot goes after the write (ahead
    // of it the copy would hold the write back in the queue: a deadlock).
    wire           cp_late = take_cp && take_wr && rd_new;
    wire [QW-1:0]  ws = wq_tail[QW-1:0] + QW'(take_cp && !cp_late);
    /* verilator lint_off UNUSEDSIGNAL */
    wire [QW-1:0]  cs = wq_tail[QW-1:0] + QW'(cp_late);   // a copy read's slot (COPY)
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- fences
    wire [CW-1:0]  in_wcnt   = w_taken + CW'(rd_new && take_wr);
    wire [CW-1:0]  in_c      = w_ring[ring_at - rd_fence];
    wire [CW-1:0]  in_need_c = (rd_fence == 0) ? in_wcnt
                             : (($signed(in_c - in_wcnt) < 0) ? in_c : in_wcnt);
    function automatic [HB_-1:0] bucket_of(input [OW-1:0] a);
        bucket_of = '0;
        for (int i = KS; i < OW; i++)
            bucket_of[(i - KS) % HB_] = bucket_of[(i - KS) % HB_] ^ a[i];
    endfunction
    wire [CW-1:0]  in_tab;                     // the port's own table's number for the read
    if (NB > 1) begin : g_tab
        reg  [CW-1:0]  hz_tab [NB];
        always @(posedge clk)
            if (take_wr)
                hz_tab[bucket_of(wr_addr)] <= w_taken + 1'b1;
        assign in_tab = hz_tab[bucket_of(rd_addr)];
`ifdef SIMULATION
        initial for (int i = 0; i < NB; i++) hz_tab[i] = '0;
`endif
    end else begin : g_no_tab
        assign in_tab = '0;
    end
    wire [CW-1:0]  in_need_b = (NEED_IN != 0) ? rd_need : (rd_new && take_wr) ? w_taken + 1'b1 : in_tab;
    assign w_count = 16'(w_taken);
    wire [CW-1:0]  in_need   = (TAB != 0) ? in_need_b : in_need_c;
    // (A difference's sign bit, not a compare after it.) Table numbers: the
    // write is in flight while its number is in (w_done, w_done + w_inflight].
    function automatic fence_ok(input [CW-1:0] need);
        /* verilator lint_off UNUSEDSIGNAL */
        reg [CW-1:0] dw;
        /* verilator lint_on UNUSEDSIGNAL */
        dw = w_done - need;
        fence_ok = (TAB != 0) ? ((need - w_done - 1'b1) >= w_inflight) : !dw[CW-1];
    endfunction
    function automatic after_ok(input af, input [CCW-1:0] cn);
        /* verilator lint_off UNUSEDSIGNAL */
        reg [CCW-1:0] dc;
        /* verilator lint_on UNUSEDSIGNAL */
        dc = c_done - cn;
        after_ok = !af || !dc[CCW-1];
    endfunction

    // A request's fields in the read queue and back.
    function automatic [RQW-1:0] rq_pack(input [RQF-1:0] f);
        if (CP != 0 && MB != 0) rq_pack = RQW'(f);
        else if (CP != 0)       rq_pack = RQW'(f >> 1);
        else if (MB != 0)       rq_pack = RQW'({f[RQF-1:CCW+3], f[0]});
        else                    rq_pack = RQW'(f >> (CCW + 3));
    endfunction
    function automatic [RQF-1:0] rq_unpack(input [RQW-1:0] q);
        if (CP != 0 && MB != 0) rq_unpack = RQF'(q);
        else if (CP != 0)       rq_unpack = RQF'({q, 1'b0});
        else if (MB != 0)       rq_unpack = RQF'({q[RQW-1:1], {(CCW + 2){1'b0}}, q[0]});
        else                    rq_unpack = RQF'({q, {(CCW + 3){1'b0}}});
    endfunction

    // ---- write beats (TRIM: rows of several beats with byte enables, no
    // merging): a write's burst covers only its beats from the first to the
    // last with a byte enabled (none: its first beat, no byte written), so a
    // row written in part -- a member's slice of a wide row -- spends W beats
    // on its part only. Its row keeps its slot whole; the streamer starts at
    // the first beat (kept with the row) and ends at the last.
    wire [BEATS-1:0] wr_any;
    for (genvar j = 0; j < BEATS; j++) begin : g_any
        wire [63:0] by;
        for (genvar b = 0; b < 64; b++) begin : g_by
            if (64 * j + b < DB) begin : g_in
                assign by[b] = wr_mask[(64 * j + b) / GROUP];
            end else begin : g_pad
                assign by[b] = 1'b0;
            end
        end
        if (j == 0) begin : g_j0
            assign wr_any[j] = |by;
        end else begin : g_jn
            assign wr_any[j] = (|by) && (LW'(j) <= wr_len);
        end
    end
    function automatic [LW-1:0] first_of(input [BEATS-1:0] a);
        first_of = '0;
        for (int j = BEATS - 1; j >= 0; j--) if (a[j]) first_of = LW'(j);
    endfunction
    function automatic [LW-1:0] last_of(input [BEATS-1:0] a);
        last_of = '0;
        for (int j = 0; j < BEATS; j++) if (a[j]) last_of = LW'(j);
    endfunction
    wire [LW-1:0]  wt_first = (TRIM != 0) ? first_of(wr_any) : '0;
    wire [LW-1:0]  wt_last  = (TRIM != 0) ? last_of(wr_any) : wr_len;
    wire [OW-1:0]  wt_addr  = wr_addr + OW'(wt_first);
    wire [LW-1:0]  wt_len   = wt_last - wt_first;

    // ---- intake. A request may join a burst behind the one taken before it
    // (a copy slot between two writes parts them) when its beats follow that
    // one's within its 4 KiB page (requests never cross a page).
    reg  [OW-1:0]  r_end, w_end;               // the last request's next beat
    reg            w_end_ok;
    wire           rd_cont = (MB != 0) && rd_addr == r_end && rd_addr[5:0] != 6'd0;
    wire           wr_cont = (MB != 0) && w_end_ok && !take_cp && wr_addr == w_end && wr_addr[5:0] != 6'd0;
    always @(posedge clk) begin
        if (take_rd) begin
            rq_mem[rq_tail[QW-1:0]] <= rq_pack({rd_addr, rd_len, rd_tag, in_need, take_cp,
                                                rd_after_copies && (CP != 0), c_in, rd_cont});
            r_end <= rd_addr + OW'(rd_len) + 1'b1;
        end
        if (take_wr) begin
            wq_mem[ws] <= WQS'({wt_first, WQW'({wt_addr, wt_len, wr_pair, epoch, r_in + CW'(take_rd), wr_cont} >> (WQF - WQW))});
            w_end <= wr_addr + OW'(wr_len) + 1'b1;
        end
        if (take_wr && !cp_late) w_end_ok <= 1'b1;
        else if (take_cp) w_end_ok <= 1'b0;
    end

    // ---- copy slots: {slot, copy address, beats - 1}, in slot order. The
    // head's entry is filled once its read's last beat is in the queue; after
    // a reset the unfilled ones are void (written with no byte enabled).
    wire           cs_head, cs_filled, cs_void;        // wq head is a copy slot / filled / void
    wire [OW-1:0]  cs_addr;
    wire [LW-1:0]  cs_len;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [QW-1:0]  cw_slot;                            // the slot the copied beats go to (COPY)
    /* verilator lint_on UNUSEDSIGNAL */
    if (CP != 0) begin : g_cq
        reg  [QW+OW+LW-1:0] cq_mem [QUEUE];
        reg  [QW-1:0]       cq_slot [QUEUE];
        reg  [QW:0]         cq_head, cq_tail, cq_fill;
        reg                 cq_cancel;
        always @(posedge clk) begin
            if (take_cp) begin
                cq_mem[cq_tail[QW-1:0]]  <= {cs, rd_copy_addr, rd_len[LW-1:0]};
                cq_slot[cq_tail[QW-1:0]] <= cs;
                cq_tail <= cq_tail + 1'b1;
            end
            if (cq_adv) cq_head <= cq_head + 1'b1;
            cq_fill <= cq_fill + (QW+1)'(cw_done) + (QW+1)'(cq_adv && cs_void);
            if (reset) cq_cancel <= 1'b1;
            else if (draining && !fill_go && drained) cq_cancel <= 1'b0;
        end
        wire [QW+OW+LW-1:0] ch = cq_mem[cq_head[QW-1:0]];
        assign cs_head   = (cq_head != cq_tail) && (wq_head[QW-1:0] == ch[OW+LW +: QW]);
        assign cs_filled = cq_head != cq_fill;
        assign cs_void   = cq_cancel && !cs_filled;
        assign cs_addr   = ch[LW +: OW];
        assign cs_len    = ch[LW-1:0];
        assign cw_slot   = cq_slot[cq_fill[QW-1:0]];
`ifdef SIMULATION
        initial begin cq_head = '0; cq_tail = '0; cq_fill = '0; cq_cancel = 1'b0; end
`endif
    end else begin : g_no_cq
        assign cs_head = 1'b0; assign cs_filled = 1'b0; assign cs_void = 1'b0;
        assign cs_addr = '0; assign cs_len = '0; assign cw_slot = '0;
    end

    // ---- read issue: the next read (p) behind the head (h); h is the AXI
    // burst being formed (its first request's fence; a joining request's own
    // fence held when it joined). Fences are computed from registers, a
    // cycle late (B's only add: conservative).
    wire [RQF-1:0] rh = rq_unpack(rq_mem[rq_head[QW-1:0]]);
    wire [OW-1:0]  rh_addr = rh[RQF-1 -: OW];
    wire [LRW-1:0] rh_len  = rh[RQF-OW-1 -: LRW];
    wire [TW-1:0]  rh_tag  = rh[RQF-OW-LRW-1 -: TW];
    wire [CW-1:0]  rh_need = rh[CCW+3 +: CW];
    wire           rh_cp   = rh[CCW+2];
    wire           rh_af   = rh[CCW+1];
    wire [CCW-1:0] rh_cn   = rh[CCW:1];
    wire           rh_ct   = rh[0];
    reg            p_valid, p_ok, p_cp, p_af, p_ct;
    reg  [OW-1:0]  p_addr;
    reg  [LRW-1:0] p_len;
    reg  [TW-1:0]  p_tag;
    reg  [CW-1:0]  p_need;
    reg  [CCW-1:0] p_cn;
    reg            h_valid, h_ok, h_af, h_cp;
    reg  [TW-1:0]  h_tag;
    reg  [OW-1:0]  h_addr;
    reg  [BLW-1:0] h_blen;
    reg  [CW-1:0]  h_need;
    reg  [CCW-1:0] h_cn;
    reg            r_room, of_room;
    wire           ar_rdy;
    // In-flight reads: their tags, beats and copy flags, in order.
    localparam OQW = TW + LRW + ((COPY != 0) ? 1 : 0);
    (* ram_style = "distributed" *) reg  [OQW-1:0] of_mem [OFD];
    reg  [OFW:0]    of_head, of_tail, of_iss;
    reg  [LRW-1:0]  r_beat;
    reg  [ACW-1:0]  ar_cnt;                    // read bursts on AXI
    wire           ar_go   = h_valid && h_ok && r_room && ar_rdy && !reset;
    wire           rq_any  = rq_head != rq_tail;
    // (Bursts: a request takes its in-flight entry as it joins the head, and
    // a reset drops those not issued; without bursts the head is one request,
    // entered as it issues, and the AXI credit bounds the entries.)
    wire           promote = p_valid && (!h_valid || ar_go) && (of_room || MB == 0);
    wire [BLW:0]   r_sum   = (BLW+1)'(h_blen) + (BLW+1)'(p_len) + 1'b1;
    wire           merge   = (MB != 0) && h_valid && p_valid && !ar_go && !(r_room && ar_rdy) && p_ok && of_room
                          && p_ct && (r_sum < (BLW+1)'(MAX_BURST));
    wire           h_load  = rq_any && (!p_valid || promote || merge);

    always @(posedge clk) begin
        h_ok <= promote ? (fence_ok(p_need) && after_ok(p_af, p_cn)) : (fence_ok(h_need) && after_ok(h_af, h_cn));
        p_ok <= !h_load && fence_ok(p_need) && after_ok(p_af, p_cn);
        r_room  <= ar_cnt < ACW'(RDB - 1);
        of_room <= (of_tail - of_head) < (OFW+1)'(OFD - 2);
        if (h_load) begin
            p_addr  <= rh_addr;
            p_ct    <= rh_ct;
            p_len   <= rh_len;
            p_tag   <= rh_tag;
            p_need  <= rh_need;
            p_cp    <= rh_cp;
            p_af    <= rh_af;
            p_cn    <= rh_cn;
        end
        if (promote) begin
            h_addr <= p_addr;
            h_tag  <= p_tag;
            h_cp   <= p_cp;
            h_blen <= BLW'(p_len);
            h_need <= p_need;
            h_af   <= p_af;
            h_cn   <= p_cn;
        end else if (merge) begin
            h_blen <= BLW'(r_sum);
        end
        if ((MB != 0) ? (promote || merge) : ar_go)
            of_mem[of_tail[OFW-1:0]] <= (MB != 0) ? OQW'({p_tag, p_len, p_cp} >> ((COPY != 0) ? 0 : 1))
                                                  : OQW'({h_tag, LRW'(h_blen), h_cp} >> ((COPY != 0) ? 0 : 1));
    end

    // ---- read beats (from the AXI side): to the simulator, or (a copy)
    // into the write queue -- not in a cycle the simulator writes (the
    // queue's one write port): the beat waits.
    wire [TW+LRW:0] oh;
    if (COPY != 0) begin : g_oh_cp
        assign oh = of_mem[of_head[OFW-1:0]];
    end else begin : g_oh
        assign oh = {of_mem[of_head[OFW-1:0]], 1'b0};
    end
    wire [TW-1:0]  oh_tag = oh[TW+LRW -: TW];
    wire [LRW-1:0] oh_len = oh[LRW:1];
    wire           oh_cp  = oh[0] && (CP != 0) && (stale == 0);
    wire           cb_full;                    // (CB) the copy beat queue is full
    wire           rb_ready = !(oh_cp && ((CB != 0) ? cb_full : take_wr));
    wire           rb_fire  = rb_valid && rb_ready;
    wire           r_last_beat = rb_fire && (r_beat == oh_len);
    assign cw_en   = rb_fire && oh_cp && !reset;
    assign cw_done = cw_en && (r_beat == oh_len);
    always @(posedge clk) begin
        rsp_valid  <= rb_fire && (stale == 0) && !reset && (!oh[0] || (CP == 0) || r_last_beat);
        rsp_copied <= oh[0] && (CP != 0);
        rsp_tag    <= oh_tag;
        rsp_beat   <= r_beat;
        rsp_last   <= r_last_beat;
        rsp_data   <= rb_data;
    end

    // ---- write issue: the next write (pw) behind the head (g), as reads.
    // A copy slot is taken from the copy entries once filled (void after a
    // reset: no byte written); a write's pair condition held when it joined.
    wire [WQS-1:0] whs = wq_mem[wq_head[QW-1:0]];
    wire [WQF-1:0] wh = WQF'({whs[WQW-1:0], 1'b0} >> ((MAX_BURST > 1) ? 1 : 0));
    wire [LW-1:0]  wh_first = (TRIM != 0) ? whs[WQS-1 -: LW] : '0;
    wire           wq_any = wq_head != wq_tail;
    wire [OW-1:0]  wl_addr = cs_head ? cs_addr : wh[WQF-1 -: OW];
    wire [LW-1:0]  wl_len  = cs_head ? cs_len  : wh[WQF-OW-1 -: LW];
    reg            pw_valid, pw_ok, pw_pair, pw_cp, pw_vd, pw_ct;
    reg  [OW-1:0]  pw_addr;
    reg  [LW-1:0]  pw_len;
    reg  [3:0]     pw_ep;
    reg  [CW-1:0]  pw_rcnt;
    // The write the head takes next: without merging (MB == 0) the queue's
    // head itself, so a write reaches AW a host cycle sooner (no pre-write
    // stage, as 3dde835's port); with merging the pre-write stage, where the
    // next write waits to join the head's burst.
    wire [OW-1:0]  px_addr  = (MB != 0) ? pw_addr  : wl_addr;
    wire [LW-1:0]  px_len   = (MB != 0) ? pw_len   : wl_len;
    wire           px_pair  = (MB != 0) ? pw_pair  : !cs_head && wh[CW+5];
    wire [3:0]     px_ep    = (MB != 0) ? pw_ep    : wh[CW+1 +: 4];
    wire [CW-1:0]  px_rcnt  = (MB != 0) ? pw_rcnt  : wh[CW:1];
    wire           px_cp    = (MB != 0) ? pw_cp    : cs_head && !cs_void;
    wire           px_vd    = (MB != 0) ? pw_vd    : cs_head && cs_void;
    wire [LW-1:0]  px_first = cs_head ? '0 : wh_first;   // (TRIM: no merging)
    reg            g_valid, g_ok, g_pair, g_vd, g_cpf;
    reg  [OW-1:0]  g_addr;
    reg  [BLW-1:0] g_blen;
    reg  [LW-1:0]  g_first;
    reg  [NRW-1:0] g_nreq, g_ncp;
    reg  [3:0]     g_ep;
    reg  [CW-1:0]  g_rcnt;
    reg            w_room;
    wire           aw_rdy;
    reg  [BCW-1:0] bq_cnt;                     // write bursts on AXI
    function automatic pair_ok(input pr, input [3:0] ep, input [CW-1:0] rc);
        /* verilator lint_off UNUSEDSIGNAL */
        reg [CW-1:0] dr;
        /* verilator lint_on UNUSEDSIGNAL */
        dr = r_done - rc;
        pair_ok = !pr || (ep != epoch) || !dr[CW-1];
    endfunction
    // (Without merging the head's pair check is its registers' now, as
    // 3dde835's: a write-after-read pair leaves the host cycle its reads are
    // back, not a host cycle later.)
    wire           g_okx     = (MB != 0) ? g_ok : pair_ok(g_pair, g_ep, g_rcnt);
    wire           aw_go     = g_valid && g_okx && w_room && aw_rdy;
    wire           pw_can    = wq_any && (!cs_head || cs_filled || cs_void);
    wire           px_valid  = (MB != 0) ? pw_valid : pw_can;
    wire           w_promote = px_valid && (!g_valid || aw_go);
    wire [BLW:0]   w_sum     = (BLW+1)'(g_blen) + (BLW+1)'(pw_len) + 1'b1;
    wire           w_merge   = (MB != 0) && g_valid && pw_valid && !aw_go && !(w_room && aw_rdy) && pw_ok
                            && pw_ct && (w_sum < (BLW+1)'(MAX_BURST));
    wire           pw_load   = (MB != 0) ? pw_can && (!pw_valid || w_promote || w_merge) : w_promote;
    assign cq_adv  = pw_load && cs_head;

    always @(posedge clk) begin
        g_ok  <= w_promote ? pair_ok(px_pair, px_ep, px_rcnt) : pair_ok(g_pair, g_ep, g_rcnt);
        pw_ok <= !pw_load && pair_ok(pw_pair, pw_ep, pw_rcnt);
        w_room <= bq_cnt < BCW'(WRB - 1);
        if (pw_load) begin
            pw_addr  <= wl_addr;
            pw_len   <= wl_len;
            pw_pair  <= !cs_head && wh[CW+5];
            pw_ep    <= wh[CW+1 +: 4];
            pw_rcnt  <= wh[CW:1];
            pw_ct    <= !cs_head && wh[0];
            pw_cp    <= cs_head && !cs_void;
            pw_vd    <= cs_head && cs_void;
        end
        if (w_promote) begin
            g_addr <= px_addr;
            g_blen <= BLW'(px_len);
            g_first <= px_first;
            g_pair <= px_pair;
            g_ep   <= px_ep;
            g_rcnt <= px_rcnt;
            g_vd   <= px_vd;
            g_cpf  <= px_cp;
            g_nreq <= NRW'(!px_cp && !px_vd);
            g_ncp  <= NRW'(px_cp || px_vd);
        end else if (w_merge) begin
            g_blen <= BLW'(w_sum);
            g_nreq <= g_nreq + NRW'(!pw_cp);
            g_ncp  <= g_ncp + NRW'(pw_cp);
        end
    end

    // ---- the zero fill (after a reset's drain): 4 KiB bursts of zeros.
    wire           fill_aw = fill_go && fill_more && !g_valid && w_room && aw_rdy;

    // ---- bursts to the AXI side: {offset, beats - 1, fill, no byte enabled};
    // their request counts wait here for their B.
    wire           d_push = aw_go || fill_aw;
    // (Selected by the fill phase, a register: no write head exists then.)
    // (TRIM: the row's first beat above them, for the streamer.)
    // (CB: above it, a filled copy's flag: its beats come from the copy beat queue.)
    localparam DTB = OW + BLW + 2;
    localparam DLT = (TRIM != 0) ? LW : 0;
    localparam DFW = DTB + DLT + ((CB != 0) ? 1 : 0);
    wire [DFW-1:0] d_data = fill_go ? DFW'({fill_at, fill_n, 1'b1, 1'b0})
                                    : DFW'({g_addr, g_blen, 1'b0, g_vd && (CP != 0)})
                                      | ((TRIM != 0) ? DFW'(g_first) << DTB : '0)
                                      | ((CB != 0) ? DFW'(g_cpf) << (DTB + DLT) : '0);
    wire           b_ev;                       // a B (in clk)
    wire [NRW-1:0] b_nreq, b_ncp;
    if (BQ != 0) begin : g_bq
        reg  [2*NRW-1:0] bq_mem [1 << BQW];
        reg  [BQW-1:0]   bq_head, bq_tail;
        always @(posedge clk) begin
            if (d_push) begin
                bq_mem[bq_tail] <= fill_go ? {NRW'(1), NRW'(0)} : {g_nreq, g_ncp};
                bq_tail <= bq_tail + 1'b1;
            end
            if (b_ev) bq_head <= bq_head + 1'b1;
        end
        assign {b_nreq, b_ncp} = bq_mem[bq_head];
`ifdef SIMULATION
        initial begin bq_head = '0; bq_tail = '0; end
`endif
    end else begin : g_no_bq
        assign b_nreq = NRW'(1);
        assign b_ncp  = '0;
    end

    // ---- drain: every queue empty and every transaction answered.
    assign         drained = !rq_any && !p_valid && !h_valid && !wq_any && !pw_valid && !g_valid
                          && ar_cnt == 0 && bq_cnt == 0 && of_head == of_tail;
    assign busy = draining;
    assign stat = {ar_cnt != 0, g_valid && !aw_go, h_valid && !ar_go};

    always @(posedge clk) begin
        // samples of the write count
        tick <= tick + 1'b1;
        if (&tick) begin
            ring_at <= ring_at + 1'b1;
            // (The count after this cycle: a write taken in the sample's
            // cycle is in it, so a write taken (c+1) 2^FS host cycles before
            // a class-c read is always waited for, as the class promises.)
            w_ring[ring_at + 1'b1] <= w_taken + CW'(take_wr);
        end
        // the region clean: `zero` rose since the last write taken
        if (ZERO_FILL != 0) begin
            z_q <= zero;
            if (take_wr || take_cp) clean <= 1'b0;
            else if (zero && !z_q) clean <= 1'b1;
        end
        if (take_rd) begin rq_tail <= rq_tail + 1'b1; r_in <= r_in + 1'b1; end
        if (take_cp) c_in <= c_in + 1'b1;
        wq_tail <= wq_tail + (QW+1)'(take_wr) + (QW+1)'(take_cp);
        if (take_wr) w_taken <= w_taken + 1'b1;
        // reads
        if (h_load) rq_head <= rq_head + 1'b1;
        if (h_load) p_valid <= 1'b1;
        else if (promote || merge) p_valid <= 1'b0;
        if (promote) h_valid <= 1'b1;
        else if (ar_go) h_valid <= 1'b0;
        if ((MB != 0) ? (promote || merge) : ar_go) of_tail <= of_tail + 1'b1;
        if (ar_go) of_iss <= of_tail;
        ar_cnt <= ar_cnt + ACW'(ar_go) - ACW'(rb_fire && rb_last);
        if (rb_fire) begin
            r_beat <= r_last_beat ? '0 : r_beat + 1'b1;
            if (r_last_beat) begin
                of_head <= of_head + 1'b1;
                r_done  <= r_done + 1'b1;
                if (stale != 0) stale <= stale - 1'b1;
            end
        end
        // writes
        if (pw_load) wq_head <= wq_head + 1'b1;
        if (pw_load && MB != 0) pw_valid <= 1'b1;
        else if (w_promote || w_merge) pw_valid <= 1'b0;
        if (w_promote) g_valid <= 1'b1;
        else if (aw_go) g_valid <= 1'b0;
        if (fill_aw) begin
            fill_at   <= fill_at + OW'(fill_n) + 1'b1;
            fill_left <= fill_left - (OW+1)'(fill_n) - 1'b1;
            w_taken   <= w_taken + 1'b1;
        end
        bq_cnt <= bq_cnt + BCW'(d_push) - BCW'(b_ev);
        if (b_ev) begin
            w_done <= w_done + CW'(b_nreq);
            c_done <= c_done + CCW'(b_ncp);
        end
        w_inflight <= w_inflight + CW'(take_wr || fill_aw) - (b_ev ? CW'(b_nreq) : CW'(0));
        // Drain (reset / move) done: the new base; then (reset) the fill.
        if (draining && drained && !fill_go) begin
            base_q <= base;
            if (fill_more) fill_go <= 1'b1;
            else draining <= 1'b0;
        end
        if (fill_go && !fill_more && drained) begin
            fill_go  <= 1'b0;
            draining <= 1'b0;
        end
        // Room, from registers (a cycle conservative).
        rd_free <= draining ? '0 : FW'(QUEUE - 32'((QW+1)'(rq_tail - rq_head)) - 32'(h_valid) - 32'(p_valid) - 32'(take_rd));
        // (A slot returns once the AXI side has read its row: in clk the
        // release pointer is the streamer's read pointer itself, exact and
        // cheaper than counting the head stages.)
        wr_free <= draining ? '0 : FW'(QUEUE - 32'((QW+1)'(wq_tail - w_rel)) - 32'(take_wr) - 32'(take_cp));
        // Reset: drop the reads not yet on AXI (and the answers to those on
        // it), keep the writes; a move keeps everything and only drains.
        if (reset) begin
            rq_head   <= rq_tail;
            h_valid   <= 1'b0;
            p_valid   <= 1'b0;
            if (MB != 0) of_tail <= of_iss;
            stale     <= (OFW+1)'(((MB != 0) ? of_iss : of_tail) - of_head) - (OFW+1)'(r_last_beat);
            r_in      <= r_done + CW'((OFW+1)'(((MB != 0) ? of_iss : of_tail) - of_head));
            epoch     <= epoch + 1'b1;
            draining  <= 1'b1;
            fill_go   <= 1'b0;
            fill_at   <= '0;
            fill_left <= (ZERO_FILL != 0 && zero && !clean) ? bytes[OW+6:6] : '0;
            rd_free   <= '0;
            wr_free   <= '0;
        end else if (move) begin
            draining  <= 1'b1;
        end
    end

    // =================================================================
    // AXI side (m_clk where ASYNC, else clk)
    // =================================================================
    wire           mclk = (ASYNC != 0) ? m_clk : clk;
    function automatic [63:0] addr_of(input [OW-1:0] off);
        if (BASE_ALIGNED != 0)
            addr_of = (base_q & ~((64'd1 << (OW + 6)) - 64'd1)) | 64'({off, 6'b0});
        else
            addr_of = base_q + 64'({off, 6'b0});
    endfunction

    // ---- AR
    if (ASYNC != 0) begin : g_ar_async
        wire               v;
        wire [OW+BLW-1:0]  d;
        wire               take = v && (!m_arvalid || m_arready);
        foldsim_vport_fifo #(.W(OW + BLW), .LOG(LOGA)) q (
            .w_clk(clk), .w_valid(ar_go), .w_ready(ar_rdy), .w_data({h_addr, h_blen}),
            .r_clk(m_clk), .r_valid(v), .r_ready(take), .r_data(d));
        always @(posedge m_clk) begin
            if (m_arvalid && m_arready) m_arvalid <= 1'b0;
            if (take) begin
                m_arvalid <= 1'b1;
                m_araddr  <= addr_of(d[BLW +: OW]);
                m_arlen   <= {1'b0, d[BLW-1:0], 1'b1};
            end
        end
    end else begin : g_ar_sync
        assign ar_rdy = !m_arvalid || m_arready;
        always @(posedge clk) begin
            if (m_arvalid && m_arready) m_arvalid <= 1'b0;
            if (ar_go) begin
                m_arvalid <= 1'b1;
                m_araddr  <= addr_of(h_addr);
                m_arlen   <= 8'(h_blen);
            end
        end
    end

    // ---- R: (ASYNC) two 256-bit beats make one, across in a FIFO.
    if (ASYNC != 0) begin : g_r_async
        reg  [255:0] r_lo;
        reg          r_half;
        wire         room;
        assign m_rready = !r_half || room;
        always @(posedge m_clk)
            if (m_rvalid && m_rready) begin
                r_half <= !r_half;
                if (!r_half) r_lo <= m_rdata[255:0];
            end
        foldsim_vport_fifo #(.W(513), .LOG(LOGR)) q (
            .w_clk(m_clk), .w_valid(m_rvalid && r_half), .w_ready(room), .w_data({m_rlast, m_rdata[255:0], r_lo}),
            .r_clk(clk), .r_valid(rb_valid), .r_ready(rb_ready), .r_data({rb_last, rb_data}));
`ifdef SIMULATION
        initial r_half = 1'b0;
`endif
    end else begin : g_r_sync
        assign rb_valid = m_rvalid;
        assign rb_data  = 512'(m_rdata);
        assign rb_last  = m_rlast;
        assign m_rready = rb_ready;
    end

    // ---- B: counted (ASYNC: a Gray counter read in clk).
    assign m_bready = 1'b1;
    if (ASYNC != 0) begin : g_b_async
        wire [CRW-1:0] got;
        reg  [CRW-1:0] seen;
        /* verilator lint_off UNUSEDSIGNAL */
        wire [CRW-1:0] mine;
        /* verilator lint_on UNUSEDSIGNAL */
        foldsim_vport_gcnt #(.N(CRW)) c (.a_clk(m_clk), .inc(m_bvalid), .a_cnt(mine), .b_clk(clk), .b_cnt(got));
        assign b_ev = got != seen;
        always @(posedge clk) if (b_ev) seen <= seen + 1'b1;
`ifdef SIMULATION
        initial seen = '0;
`endif
    end else begin : g_b_sync
        assign b_ev = m_bvalid;
    end

    // ---- AW and W. A burst's descriptor is taken by the AW channel and the
    // W streamer; the streamer reads the queue's rows at its release pointer
    // (rp), a beat (row_beat) and an AXI beat (half) at a time.
    /* verilator lint_off UNUSEDSIGNAL */
    wire              dv, d_pop;
    reg               d_aw, d_w;               // taken by AW / by W (ASYNC)
    wire [QW:0]       s_rp;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [DFW-1:0]    dd;
    wire [DFW+LW:0]   ddx = (DFW+LW+1)'(dd);
    wire [LW-1:0]     dd_first = (TRIM != 0) ? ddx[DTB +: LW] : '0;
    wire              dd_cp    = (CB != 0) && ddx[DTB + DLT];
    reg               s_fill, s_nostrb, s_cp;
    reg  [BLW-1:0]    s_left;
    reg  [LW-1:0]     s_rb;
    reg               s_half;
    wire              s_beat_end  = s_busy && (!m_wvalid || m_wready) && (H == 1 || s_half);
    wire              s_can       = !s_busy || (s_beat_end && s_left == '0);
    wire              aw_can      = !m_awvalid || m_awready;
    wire              aw_take, s_take;
    if (ASYNC != 0) begin : g_d_async
        foldsim_vport_fifo #(.W(DFW), .LOG(LOGA)) q (
            .w_clk(clk), .w_valid(d_push), .w_ready(aw_rdy), .w_data(d_data),
            .r_clk(m_clk), .r_valid(dv), .r_ready(d_pop), .r_data(dd));
        assign aw_take = dv && !d_aw && aw_can;
        assign s_take  = dv && !d_w && s_can;
        assign d_pop   = (d_aw || aw_take) && (d_w || s_take);
        always @(posedge m_clk) begin
            d_aw <= d_pop ? 1'b0 : (d_aw || aw_take);
            d_w  <= d_pop ? 1'b0 : (d_w || s_take);
        end
    end else begin : g_d_sync
        assign aw_rdy  = aw_can && s_can;
        assign dv      = d_push;
        assign dd      = d_data;
        assign aw_take = d_push;
        assign s_take  = d_push;
        assign d_pop   = d_push;
    end
    // (In clk with a base to add: with merging the write head's address is
    // added as it is promoted from the pre-write stage, the fill's kept
    // whole, so no adder precedes m_awaddr; without merging the head comes
    // straight from the queue's memory, so its address is added from the
    // head's register as it leaves -- the queue read and the adder in one
    // host cycle would be too deep.)
    reg  [63:0]       g_addr64, f_addr64;
    always @(posedge clk) begin
        if (w_promote && MB != 0) g_addr64 <= addr_of(px_addr);
        if (draining && drained && !fill_go) f_addr64 <= base;
        else if (fill_aw) f_addr64 <= f_addr64 + 64'd4096;      // (every burst but the last is 4 KiB)
    end
    wire [63:0]       aw_addr = (ASYNC != 0 || BASE_ALIGNED != 0) ? addr_of(dd[BLW+2 +: OW])
                              : fill_go ? f_addr64 : (MB != 0) ? g_addr64 : addr_of(g_addr);
    always @(posedge mclk) begin
        if (m_awvalid && m_awready) m_awvalid <= 1'b0;
        if (aw_take) begin
            m_awvalid <= 1'b1;
            m_awaddr  <= aw_addr;
            m_awlen   <= (H > 1) ? {1'b0, dd[2 +: BLW], 1'b1} : 8'(dd[2 +: BLW]);
        end
    end

    // The queue's rows: a memory per 512-bit beat of a row (the first also
    // holds the mask and the row's beats), written in clk by the simulator's
    // writes and the copied beats, read in mclk by the streamer.
    // Row stage (RS: in clk, without merging): the streamer copies the row it
    // takes into a register and streams the beats from there, so the row's
    // slot is free as the burst is taken -- not after its last beat, which
    // waits on W (a multi-beat row, or W held behind other masters' bursts).
    // The queue then holds QUEUE rows besides the one on W (a slot more room
    // under write pressure, for a row of flops). With merging the release
    // stays late on purpose (room returning later lets writes merge); ASYNC
    // streams in m_clk from the queue (the clock crossing).
    localparam RS  = (ASYNC == 0 && MB == 0) ? 1 : 0;
    /* verilator lint_off UNUSEDSIGNAL */
    wire           rs_inc;                     // (RS) a row taken by the streamer
    /* verilator lint_on UNUSEDSIGNAL */
    wire [MW-1:0]  row_mask;
    wire [LW-1:0]  row_len;
    wire [511:0]   sub_rd [BEATS];
    for (genvar j = 0; j < BEATS; j++) begin : g_sub
        localparam SWJ = (j < BEATS - 1) ? 512 : DW - 512 * (BEATS - 1);
        localparam XJ  = (j == 0) ? MW + ((BEATS > 1) ? LW : 0) : 0;
        reg  [SWJ+XJ-1:0] mem [QUEUE];
        wire [SWJ+XJ-1:0] din_w;
        if (j == 0 && BEATS > 1) begin : g_x
            assign din_w = {wt_last, wr_mask, wr_data[SWJ-1:0]};
        end else if (j == 0) begin : g_m
            assign din_w = {wr_mask, wr_data[SWJ-1:0]};
        end else begin : g_d
            assign din_w = wr_data[512*j +: SWJ];
        end
        if (CP != 0 && CB == 0) begin : g_cw
            // (A copied beat: its row's beats and a full mask with the first.)
            wire [SWJ+XJ-1:0] din_c;
            if (j == 0 && BEATS > 1) begin : g_x
                assign din_c = {LW'(oh_len), {MW{1'b1}}, rb_data[SWJ-1:0]};
            end else if (j == 0) begin : g_m
                assign din_c = {{MW{1'b1}}, rb_data[SWJ-1:0]};
            end else begin : g_d
                assign din_c = rb_data[SWJ-1:0];
            end
            wire we = take_wr || (cw_en && 32'(r_beat) == j);
            always @(posedge clk) if (we) mem[take_wr ? ws : cw_slot] <= take_wr ? din_w : din_c;
        end else begin : g_w
            always @(posedge clk) if (take_wr) mem[ws] <= din_w;
        end
        wire [SWJ+XJ-1:0] rd0 = mem[s_rp[QW-1:0]];
        wire [SWJ+XJ-1:0] rd;
        if (RS != 0) begin : g_rs
            // (Loaded whenever the streamer may take: when it takes, the row
            // at the release pointer is the one taken.)
            reg [SWJ+XJ-1:0] q;
            always @(posedge clk) if (s_can) q <= rd0;
            assign rd = q;
        end else begin : g_rq
            assign rd = rd0;
        end
        assign sub_rd[j] = 512'(rd[SWJ-1:0]);
        if (j == 0) begin : g_x0
            assign row_mask = rd[SWJ +: MW];
            if (BEATS > 1) begin : g_l
                assign row_len = rd[SWJ+MW +: LW];            // its last beat
            end else begin : g_l1
                assign row_len = '0;
            end
        end
    end
    // A beat's byte enables, from the row's mask.
    wire [BEATS*64-1:0] wbytes;
    for (genvar j = 0; j < BEATS * 64; j++) begin : g_bytes
        if (j < DB) begin : g_in
            assign wbytes[j] = row_mask[j / GROUP];
        end else begin : g_pad
            assign wbytes[j] = 1'b0;
        end
    end
    wire [511:0]   beat_d;
    if (BEATS > 1) begin : g_bd
        assign beat_d = sub_rd[s_rb];
    end else begin : g_bd1
        assign beat_d = sub_rd[0];
    end
    wire [63:0]    beat_s0 = wbytes[32'(s_rb) * 64 +: 64];
    // ---- copy beats (CB): copied beats queue here in the order they come
    // (the copies' order: answers come in read order), not in the row
    // memory; a filled copy's burst streams them from the queue's head with
    // every byte of its row enabled. The queue is not reserved: when it is
    // full the next copied beat waits on R (it drains: the copies in it are
    // ahead of every later copy in the write queue, and nothing they wait for
    // comes after them on R). After a reset a copy's beats not all in (a void
    // copy) are dropped from it.
    wire [511:0]   cb_d;
    wire [63:0]    cb_s;
    if (CB != 0) begin : g_cb
        (* ram_style = "distributed" *) reg [511:0] cb_mem [1 << CBL];
        reg  [CBL:0]   cb_w, cb_c;                 // beats in (clk), whole copies in
        /* verilator lint_off UNUSEDSIGNAL */
        wire [CBL:0]   cb_r;                       // beats streamed (mclk)
        /* verilator lint_on UNUSEDSIGNAL */
        wire [CBL:0]   cb_rc;                      // the same, seen in clk
        always @(posedge clk) begin
            if (cw_en) cb_mem[cb_w[CBL-1:0]] <= rb_data;
            if (reset) cb_w <= cb_c;
            else if (cw_en) cb_w <= cb_w + 1'b1;
            if (cw_done) cb_c <= cb_w + 1'b1;
        end
        assign cb_full = (cb_w - cb_rc) == (CBL+1)'(1 << CBL);
        assign cb_d = cb_mem[cb_r[CBL-1:0]];
        wire cb_inc = s_beat_end && s_cp;
        if (ASYNC != 0) begin : g_cba
            foldsim_vport_gcnt #(.N(CBL + 1)) c (.a_clk(m_clk), .inc(cb_inc), .a_cnt(cb_r), .b_clk(clk), .b_cnt(cb_rc));
        end else begin : g_cbs
            reg [CBL:0] r;
            always @(posedge clk) r <= r + (CBL+1)'(cb_inc);
            assign cb_r  = r;
            assign cb_rc = r;
`ifdef SIMULATION
            initial r = '0;
`endif
        end
        // (A row's bytes, every one enabled: a copy's.)
        wire [BEATS*64-1:0] cbytes;
        for (genvar j = 0; j < BEATS * 64; j++) begin : g_cby
            assign cbytes[j] = (j < DB);
        end
        assign cb_s = cbytes[32'(s_rb) * 64 +: 64];
`ifdef SIMULATION
        initial begin cb_w = '0; cb_c = '0; end
`endif
    end else begin : g_no_cb
        assign cb_full = 1'b0;
        assign cb_d = '0;
        assign cb_s = '0;
    end
    wire [511:0]   beat_dx = s_cp ? cb_d : beat_d;
    wire [63:0]    beat_s  = s_cp ? cb_s : beat_s0;
    // (A copy's burst is one row: it ends with the burst.)
    wire           row_end = !s_nostrb && !s_cp && (s_rb == row_len);
    // The AXI beat of the 512-bit beat (ASYNC: its half).
    wire [AXW-1:0]   ax_d;
    wire [AXW/8-1:0] ax_s;
    if (H > 1) begin : g_half
        assign ax_d = s_half ? beat_dx[511:256] : beat_dx[255:0];
        assign ax_s = s_half ? beat_s[63:32] : beat_s[31:0];
    end else begin : g_full
        assign ax_d = beat_dx;
        assign ax_s = beat_s;
    end
    // A row ends at its last beat (or the burst's, for a void row); then the
    // release pointer moves (its row is in the output register).
    wire           s_inc = s_beat_end && !s_fill && (row_end || s_left == '0);
    always @(posedge mclk) begin
        if (m_wvalid && m_wready) m_wvalid <= 1'b0;
        if (s_busy && (!m_wvalid || m_wready)) begin
            m_wvalid <= 1'b1;
            m_wdata  <= s_fill ? '0 : ax_d;
            m_wstrb  <= s_fill ? '1 : (s_nostrb ? '0 : ax_s);
            m_wlast  <= s_left == '0 && (H == 1 || s_half);
            s_half   <= (H > 1) && !s_half;
        end
        if (s_beat_end) begin
            s_left <= s_left - 1'b1;
            if (s_left == '0) s_busy <= 1'b0;
            if (!s_fill) s_rb <= s_inc ? '0 : s_rb + 1'b1;
        end
        if (s_take) begin
            s_busy   <= 1'b1;
            s_left   <= dd[2 +: BLW];
            s_fill   <= dd[1];
            s_nostrb <= dd[0];
            s_rb     <= dd_first;
            s_cp     <= dd_cp;
        end
    end
    // The release pointer (rows read), read in clk.
    if (ASYNC != 0) begin : g_rel_async
        assign rs_inc = 1'b0;
        foldsim_vport_gcnt #(.N(QW + 1)) c (.a_clk(m_clk), .inc(s_inc), .a_cnt(s_rp), .b_clk(clk), .b_cnt(w_rel));
    end else begin : g_rel_sync
        reg [QW:0] rp;
        // (RS: a row is released as the streamer takes it into its register;
        // a fill burst holds none. With merging, after its last beat.)
        assign rs_inc = (RS != 0) && s_take && !dd[1];
        always @(posedge clk) rp <= rp + (QW+1)'((RS != 0) ? rs_inc : s_inc);
        assign s_rp  = rp;
        // (The room the simulator sees is registered from w_rel: with RS a
        // slot counts free a host cycle after the take, as the AW leaves.)
        assign w_rel = rp;
`ifdef SIMULATION
        initial rp = '0;
`endif
    end

`ifdef SIMULATION
    // Power-up state (the FPGA's registers start at 0; draining from the start).
    initial begin
        rq_head = '0; rq_tail = '0; wq_head = '0; wq_tail = '0;
        w_taken = '0; w_done = '0; w_inflight = '0; r_in = '0; r_done = '0; stale = '0;
        c_in = '0; c_done = '0; epoch = '0; tick = '0; ring_at = '0; draining = 1'b1; fill_go = 1'b0;
        clean = 1'b0; z_q = 1'b0; base_q = '0; fill_at = '0; fill_left = '0;
        r_end = '0; w_end = '0; w_end_ok = 1'b0;
        p_valid = 1'b0; h_valid = 1'b0; p_ok = 1'b0; h_ok = 1'b0; r_room = 1'b0; of_room = 1'b0;
        pw_valid = 1'b0; g_valid = 1'b0; pw_ok = 1'b0; g_ok = 1'b0; w_room = 1'b0;
        of_head = '0; of_tail = '0; of_iss = '0; r_beat = '0; ar_cnt = '0; bq_cnt = '0;
        for (int i = 0; i < FCLASSES; i++) w_ring[i] = '0;
        m_arvalid = 1'b0; m_awvalid = 1'b0; m_wvalid = 1'b0; rsp_valid = 1'b0; rsp_copied = 1'b0;
        rd_free = '0; wr_free = '0;
        d_aw = 1'b0; d_w = 1'b0; s_busy = 1'b0; s_left = '0; s_rb = '0; s_half = 1'b0;
        s_fill = 1'b0; s_nostrb = 1'b0; s_cp = 1'b0;
    end
    // Without bursts the in-flight table holds the reads on AXI: the credit fits it.
    initial if (MB == 0 && RDB > OFD) $error("foldsim_vport: %0d reads on AXI, %0d in-flight entries", RDB, OFD);
    // The contract's preconditions.
    always @(posedge clk) begin
        if (take_rd && 32'(rd_addr[5:0]) + 32'(rd_len) > 63)
            $error("foldsim_vport: a read crosses a 4 KiB page (offset %0h, %0d beats)", rd_addr, rd_len + 1);
        if (take_wr && 32'(wr_addr[5:0]) + 32'(wr_len) > 63)
            $error("foldsim_vport: a write crosses a 4 KiB page (offset %0h, %0d beats)", wr_addr, wr_len + 1);
        if (take_cp && (32'(rd_len) >= BEATS || 32'(rd_copy_addr[5:0]) + 32'(rd_len) > 63))
            $error("foldsim_vport: a copied read longer than a write row or crossing a page");
        if (take_rd && 32'((QW+1)'(rq_tail - rq_head)) + 32'(p_valid) + 32'(h_valid) >= QUEUE)
            $error("foldsim_vport: a read taken with no room");
        if ((take_wr || take_cp) && 32'((QW+1)'(wq_tail - w_rel)) + 32'(take_wr) + 32'(take_cp) > QUEUE)
            $error("foldsim_vport: a write (or copy) taken with no room");
    end
`endif
endmodule

/* verilator lint_off DECLFILENAME */
// A FIFO across two clocks: distributed RAM written in w_clk, read
// asynchronously into a register in r_clk, Gray pointers through two flops.
// First-word fall-through. Not reset: both sides start empty at power-up.
module foldsim_vport_fifo #(
    parameter W   = 32,
    parameter LOG = 3
) (
    input  wire          w_clk,
    input  wire          w_valid,
    output wire          w_ready,
    input  wire [W-1:0]  w_data,
    input  wire          r_clk,
    output reg           r_valid,
    input  wire          r_ready,
    output reg  [W-1:0]  r_data
);
    localparam N = 1 << LOG;
    reg  [W-1:0] mem [N];
    reg  [LOG:0] wp, wg, rp, rg;
    (* ASYNC_REG = "TRUE" *) reg [LOG:0] wg_r1, wg_r2;
    (* ASYNC_REG = "TRUE" *) reg [LOG:0] rg_w1, rg_w2;
    function automatic [LOG:0] gray(input [LOG:0] b);
        gray = b ^ (b >> 1);
    endfunction
    assign w_ready = !(wg == {~rg_w2[LOG:LOG-1], rg_w2[LOG-2:0]});
    wire push = w_valid && w_ready;
    always @(posedge w_clk) begin
        if (push) begin
            mem[wp[LOG-1:0]] <= w_data;
            wp <= wp + 1'b1;
            wg <= gray(wp + 1'b1);
        end
        rg_w1 <= rg; rg_w2 <= rg_w1;
    end
    wire empty = (rg == wg_r2);
    wire load  = !empty && (!r_valid || r_ready);
    always @(posedge r_clk) begin
        wg_r1 <= wg; wg_r2 <= wg_r1;
        if (load) begin
            r_data  <= mem[rp[LOG-1:0]];
            rp      <= rp + 1'b1;
            rg      <= gray(rp + 1'b1);
            r_valid <= 1'b1;
        end else if (r_ready)
            r_valid <= 1'b0;
    end
`ifdef SIMULATION
    initial begin
        wp = '0; wg = '0; rp = '0; rg = '0; r_valid = 1'b0;
        wg_r1 = '0; wg_r2 = '0; rg_w1 = '0; rg_w2 = '0;
    end
`endif
endmodule

// A counter in a_clk (incremented by `inc`), read in b_clk through a Gray
// register and two flops. Not reset: both sides start at zero.
module foldsim_vport_gcnt #(
    parameter N = 6
) (
    input  wire          a_clk,
    input  wire          inc,
    output reg  [N-1:0]  a_cnt,
    input  wire          b_clk,
    output wire [N-1:0]  b_cnt
);
    reg  [N-1:0] g;
    (* ASYNC_REG = "TRUE" *) reg [N-1:0] g1, g2;
    always @(posedge a_clk)
        if (inc) begin
            a_cnt <= a_cnt + 1'b1;
            g     <= (a_cnt + 1'b1) ^ ((a_cnt + 1'b1) >> 1);
        end
    always @(posedge b_clk) begin g1 <= g; g2 <= g1; end
    for (genvar i = 0; i < N; i++) begin : g_bin
        assign b_cnt[i] = ^g2[N-1:i];
    end
`ifdef SIMULATION
    initial begin a_cnt = '0; g = '0; g1 = '0; g2 = '0; end
`endif
endmodule
/* verilator lint_on DECLFILENAME */
