// tb_vport: the kernel's static port (shell/rtl/foldsim_vport.sv) alone, driven
// as a folded simulator drives it -- at most a read and a write a host cycle,
// within the room rd_free/wr_free showed; an instance issues nothing while its
// read is pending -- against an AXI4 memory that keeps its rows late and reads
// them early: a write reaches memory only at its B (1 to +hi memory cycles
// after its last W beat, in order), a read takes memory's rows as its AR is
// taken (answered 1 to +hi cycles on, in order: one id). So a read that
// reached AR before the B of an earlier write of its row would answer the old
// row: every answer is checked against a reference RAM. Reads carry fence
// class 0 (any earlier write) or, with +far, the class whose distance the
// driver keeps (it then never reads a row written within that distance). A
// read and a write of one row in one host cycle form a read-first pair
// (wr_pair): the read answers the old row. +stall (percent) holds the
// channels' ready low at random. +resets=N resets the port N times with
// requests in flight; after each, every row reads zero (the port's zero
// fill). +clean: before each reset the driver stops, waits for the port and
// memory to go idle, clears memory itself and raises `zero` (the host's
// clear): the port must then not fill (no AW until it shows room again).
// +wr=P: percent of issues with a write (50); +cycles=N: run length.
//
// Version 2 of the port: ASYNC (the AXI side in its own clock, +sp=/+mp= the
// two half periods in ps: any ratio, 256-bit beats), bursts (MAX_BURST, with
// AXI_RD_OUT / AXI_WR_OUT transactions in flight: checked never exceeded;
// +seq=P percent of the requests sweep the instances so that rows follow
// each other in memory; +merge=1 fails unless fewer ARs than reads reached
// memory, +wmerge=1 fewer AWs than writes) and ring copies (COPY, +copy=P percent of reads copy their row into
// an image row instead of answering, +copy/2 percent read image rows with
// rd_after_copies: they must see every earlier copy). Every AXI burst is
// checked to stay in its 4 KiB page; every answer's beats arrive once, in
// order. Prints PASS or the first mismatches.
`timescale 1ps/1ps
module tb;
    parameter IW = 3, AW = 4, DW = 1040, QUEUE = 16, OUTSTANDING = 16, WRITES_OUT = 8, CYCLES = 30000, FS = 4, HZ = 0;
    parameter ASYNC = 0, COPY = 0, MAX_BURST = 1, AXI_RD_OUT = 0, AXI_WR_OUT = 0, BASE_ALIGNED = 0;
    localparam INST = 1 << IW, ROWS = 1 << AW, MW = DW / 8;
    localparam BEATS = (DW + 511) / 512, LW = (BEATS > 1) ? $clog2(BEATS) : 1;
    // A row's slot in memory: a power of two beats (rows never cross a 4 KiB page).
    localparam STRIDE = 1 << $clog2(BEATS);
    localparam FW = $clog2(QUEUE + 1), OW = 20, TW = IW, FCW = 4;
    localparam AXW = ASYNC ? 256 : 512, AXB = AXW / 8;
    localparam longint BASE = BASE_ALIGNED ? 64'h0000_0040_0000_0000 : 64'h1234_5000;
    // The rows, then the image rows the copies write.
    localparam longint BYTES = 2 * longint'(INST) * ROWS * STRIDE * 64;
    reg clk = 0, m_clk = 0, reset = 1, zero = 1;
    integer sp = 1000, mp = 1000;
    initial begin
        void'($value$plusargs("sp=%d", sp));
        void'($value$plusargs("mp=%d", mp));
    end
    always #(sp) clk = ~clk;
    always #(mp) m_clk = ~m_clk;
    wire aclk = ASYNC ? m_clk : clk;

    reg rd_valid = 0, wr_valid = 0, wr_pair = 0, rd_copy = 0, rd_after = 0;
    reg [OW-1:0] rd_addr = 0, wr_addr = 0, rd_copy_addr = 0;
    reg [LW-1:0] rd_len = 0, wr_len = 0;
    reg [TW-1:0] rd_tag = 0;
    reg [FCW-1:0] rd_fence = 0;
    reg [DW-1:0] wr_data = 0;
    reg [MW-1:0] wr_mask = 0;
    wire rsp_valid, rsp_last, rsp_copied, busy;
    wire [TW-1:0] rsp_tag;
    wire [LW-1:0] rsp_beat;
    wire [511:0] rsp_data;
    wire [FW-1:0] rd_free, wr_free;
    wire [2:0] stat;
    wire m_awvalid, m_wvalid, m_wlast, m_bready, m_arvalid, m_rready;
    reg m_awready = 0, m_wready = 0, m_bvalid = 0, m_arready = 0, m_rvalid = 0, m_rlast = 0;
    wire [63:0] m_awaddr, m_araddr;
    wire [7:0] m_awlen, m_arlen;
    wire [AXW-1:0] m_wdata;
    wire [AXB-1:0] m_wstrb;
    reg [AXW-1:0] m_rdata = 0;

    foldsim_vport #(.DW(DW), .MW(MW), .TW(TW), .OW(OW), .QUEUE(QUEUE), .OUTSTANDING(OUTSTANDING),
        .WRITES_OUT(WRITES_OUT), .AXI_RD_OUT(AXI_RD_OUT), .AXI_WR_OUT(AXI_WR_OUT), .MAX_BURST(MAX_BURST),
        .FCLASSES(16), .FS(FS), .HZ(HZ), .COPY(COPY), .ASYNC(ASYNC), .BASE_ALIGNED(BASE_ALIGNED)) dut (
        .clk(clk), .reset(reset),
        .rd_valid(rd_valid), .rd_addr(rd_addr), .rd_len(rd_len), .rd_tag(rd_tag), .rd_fence(rd_fence), .rd_new(1'b0),
        .rd_copy(rd_copy), .rd_copy_addr(rd_copy_addr), .rd_after_copies(rd_after),
        .wr_valid(wr_valid), .wr_addr(wr_addr), .wr_len(wr_len), .wr_pair(wr_pair), .wr_data(wr_data), .wr_mask(wr_mask),
        .rsp_valid(rsp_valid), .rsp_tag(rsp_tag), .rsp_beat(rsp_beat), .rsp_last(rsp_last), .rsp_copied(rsp_copied),
        .rsp_data(rsp_data), .rd_free(rd_free), .wr_free(wr_free), .base(BASE), .bytes(BYTES), .zero(zero),
        .move(1'b0), .busy(busy), .stat(stat), .m_clk(m_clk),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awlen(m_awlen),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_araddr(m_araddr), .m_arlen(m_arlen), .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rdata(m_rdata),
        .m_rlast(m_rlast));

    reg stop = 0;
    integer hi = 40, stall = 0, far = 0, resets = 0, seq = 0, copy = 0, merge = 0, wmerge = 0, clean = 0, wr = 50, cycles = CYCLES;
    integer errors = 0, reads = 0, answered = 0, writes = 0, pairs = 0, copies = 0, img_reads = 0;
    integer ar_beats = 0, aw_beats = 0, ars = 0, aws = 0, ar_out = 0, aw_out = 0, ar_max = 0, aw_max = 0, fill_aws = 0;
    reg [BEATS*512-1:0] ram [INST][ROWS];
    reg [BEATS*512-1:0] img [INST][ROWS];
    reg [BEATS*512-1:0] expect_row [INST];
    reg [BEATS*512-1:0] got [INST];
    reg          pending [INST];
    reg          is_copy [INST];
    integer      beat_at [INST];
    integer      last_k [INST];
    // When each row was last written (host cycle), for +far's reads.
    longint      wrote_at [INST][ROWS];
    reg [AXW-1:0] mem [longint];
    function automatic [AXW-1:0] line(input longint a);
        line = mem.exists(a) ? mem[a] : '0;
    endfunction

    // ---- AXI memory: writes at their B, reads at their AR, one id, in order ----
    typedef struct { longint addr; int beats; } burst_t;
    burst_t aw_q[$];
    reg [AXW-1:0] wbuf[$];
    reg [AXB-1:0] sbuf[$];
    int     wdone = 0;
    longint b_at[$];
    longint b_addr[$];
    int     b_beats[$];
    reg [AXW-1:0] b_data[$][$];
    reg [AXB-1:0] b_strb[$][$];
    reg [AXW-1:0] r_data[$];
    reg           r_last[$];
    longint       r_at[$];
    longint now = 0, mnow = 0;
    always @(posedge clk) now <= now + 1;
    // An AXI burst stays within its 4 KiB page.
    task automatic page_check(input string ch, input longint a, input int beats);
        if ((a % 4096) + longint'(beats) * AXB > 4096 || (a % AXB) != 0) begin
            if (errors < 5) $display("FAIL: %s burst at %h of %0d beats leaves its 4 KiB page", ch, a, beats);
            errors = errors + 1;
        end
    endtask

    always @(posedge aclk) begin
        mnow <= mnow + 1;
        m_awready <= ($urandom % 100) >= stall;
        m_wready  <= ($urandom % 100) >= stall;
        m_arready <= ($urandom % 100) >= stall;
        if (m_awvalid & m_awready) begin
            aw_q.push_back('{longint'(m_awaddr), int'(m_awlen) + 1});
            page_check("AW", longint'(m_awaddr), int'(m_awlen) + 1);
            aws = aws + 1; aw_out = aw_out + 1;
            if (!stop && int'(m_awlen) + 1 > aw_beats) aw_beats = int'(m_awlen) + 1;
            // (A burst of 64 512-bit beats is the zero fill's.)
            if (int'(m_awlen) + 1 > ((MAX_BURST > BEATS) ? MAX_BURST : BEATS) * (512 / AXW) && int'(m_awlen) + 1 != 64 * (512 / AXW)) begin
                if (errors < 5) $display("FAIL: an AW of %0d beats (bursts of at most %0d 512-bit beats)", int'(m_awlen) + 1, MAX_BURST);
                errors = errors + 1;
            end
            if (stop) fill_aws = fill_aws + 1;
            if (aw_out > aw_max) aw_max = aw_out;
        end
        if (m_wvalid & m_wready) begin
            wbuf.push_back(m_wdata);
            sbuf.push_back(m_wstrb);
            if (m_wlast) wdone = wdone + 1;
        end
        if (aw_q.size() && wdone) begin
            burst_t b;
            automatic reg [AXW-1:0] d[$];
            automatic reg [AXB-1:0] s[$];
            longint t;
            b = aw_q.pop_front();
            wdone = wdone - 1;
            t = mnow + 1 + longint'($urandom % hi);
            if (b_at.size() && t < b_at[$]) t = b_at[$];
            for (int k = 0; k < b.beats; ++k) begin
                d.push_back(wbuf.pop_front());
                s.push_back(sbuf.pop_front());
            end
            b_at.push_back(t); b_addr.push_back(b.addr); b_beats.push_back(b.beats);
            b_data.push_back(d); b_strb.push_back(s);
        end
        if (m_bvalid & m_bready) begin m_bvalid <= 0; aw_out = aw_out - 1; end
        if ((~m_bvalid | m_bready) && b_at.size() && b_at[0] <= mnow) begin
            for (int k = 0; k < b_beats[0]; ++k) begin
                longint a;
                reg [AXW-1:0] l;
                a = (b_addr[0] - BASE) / AXB + k;
                l = line(a);
                for (int j = 0; j < AXB; ++j)
                    if (b_strb[0][k][j]) l[j*8 +: 8] = b_data[0][k][j*8 +: 8];
                mem[a] = l;
            end
            void'(b_at.pop_front()); void'(b_addr.pop_front()); void'(b_beats.pop_front());
            void'(b_data.pop_front()); void'(b_strb.pop_front());
            m_bvalid <= 1;
        end
        if (m_arvalid & m_arready) begin
            longint t;
            page_check("AR", longint'(m_araddr), int'(m_arlen) + 1);
            if (int'(m_arlen) + 1 > ((MAX_BURST > BEATS) ? MAX_BURST : BEATS) * (512 / AXW)) begin
                if (errors < 5) $display("FAIL: an AR of %0d beats (bursts of at most %0d 512-bit beats)", int'(m_arlen) + 1, MAX_BURST);
                errors = errors + 1;
            end
            ars = ars + 1; ar_out = ar_out + 1;
            if (int'(m_arlen) + 1 > ar_beats) ar_beats = int'(m_arlen) + 1;
            if (ar_out > ar_max) ar_max = ar_out;
            t = mnow + 1 + longint'($urandom % hi);
            if (r_at.size() && t < r_at[$]) t = r_at[$];
            for (int k = 0; k <= int'(m_arlen); ++k) begin
                r_data.push_back(line((m_araddr - BASE) / AXB + k));
                r_last.push_back(k == int'(m_arlen));
                r_at.push_back(t);
            end
        end
        if (m_rvalid && m_rready) begin
            m_rvalid <= 0;
            if (m_rlast) ar_out = ar_out - 1;
        end
        if ((!m_rvalid || m_rready) && r_at.size() && r_at[0] <= mnow && ($urandom % 100) >= stall) begin
            m_rvalid <= 1;
            m_rdata <= r_data.pop_front();
            m_rlast <= r_last.pop_front();
            void'(r_at.pop_front());
        end
        if (AXI_RD_OUT != 0 && ar_out > AXI_RD_OUT) begin
            if (errors < 5) $display("FAIL: %0d reads on AXI (at most %0d)", ar_out, AXI_RD_OUT);
            errors = errors + 1;
        end
        if (AXI_WR_OUT != 0 && aw_out > AXI_WR_OUT) begin
            if (errors < 5) $display("FAIL: %0d writes on AXI (at most %0d)", aw_out, AXI_WR_OUT);
            errors = errors + 1;
        end
    end

    // ---- The simulator ----
    function automatic [BEATS*512-1:0] merge_row(input [BEATS*512-1:0] old, input [DW-1:0] d, input [MW-1:0] m);
        merge_row = old;
        for (int j = 0; j < MW; ++j)
            if (m[j]) merge_row[j*8 +: 8] = d[j*8 +: 8];
    endfunction
    function automatic [OW-1:0] addr_of(input int i, input int r);
        addr_of = OW'((r * INST + i) * STRIDE);
    endfunction
    function automatic [OW-1:0] img_of(input int i, input int k);
        img_of = OW'(((ROWS + k) * INST + i) * STRIDE);
    endfunction
    int seq_i = 0, seq_r = 0;
    always @(posedge clk) begin
        if (rsp_valid) begin
            if (!pending[rsp_tag]) begin
                if (errors < 5) $display("FAIL: an answer for instance %0d with no read", rsp_tag);
                errors = errors + 1;
            end else if (rsp_copied) begin
                if (!is_copy[rsp_tag] || !rsp_last) begin
                    if (errors < 5) $display("FAIL: instance %0d: a copied answer for a plain read", rsp_tag);
                    errors = errors + 1;
                end
                answered <= answered + 1;
                pending[rsp_tag] = 0;
            end else begin
                if (is_copy[rsp_tag]) begin
                    if (errors < 5) $display("FAIL: instance %0d: a copied read answered with data", rsp_tag);
                    errors = errors + 1;
                end
                if (int'(rsp_beat) != beat_at[rsp_tag]) begin
                    if (errors < 5) $display("FAIL: instance %0d: beat %0d where %0d was due", rsp_tag, rsp_beat, beat_at[rsp_tag]);
                    errors = errors + 1;
                end
                beat_at[rsp_tag] = beat_at[rsp_tag] + 1;
                got[rsp_tag][32'(rsp_beat) * 512 +: 512] = rsp_data;
                if (rsp_last) begin
                    answered <= answered + 1;
                    if (got[rsp_tag][DW-1:0] !== expect_row[rsp_tag][DW-1:0]) begin
                        if (errors < 5) $display("FAIL: instance %0d answered %h, expected %h", rsp_tag, got[rsp_tag][DW-1:0],
                                                 expect_row[rsp_tag][DW-1:0]);
                        errors = errors + 1;
                    end
                    pending[rsp_tag] = 0;
                end
            end
        end
        rd_valid <= 0; wr_valid <= 0;
        // (free does not yet count the last cycle's requests: room for both, and a copy's write.)
        if (!reset && !stop && rd_free >= 3 && wr_free >= 5 && ($urandom % 100) < 70) begin
            int i, r, w, k;
            reg dr, dw, sq, cp, ir, iw;
            sq = ($urandom % 100) < seq;
            i = int'($urandom % INST);
            r = int'($urandom % ((AW > 2) ? 4 : ROWS));
            w = int'($urandom % ((AW > 2) ? 4 : ROWS));
            k = int'($urandom % 4);
            dr = 1'($urandom % 2);
            dw = ($urandom % 100) < wr;
            if ($urandom % 4 == 0) begin r = int'($urandom % ROWS); w = int'($urandom % ROWS); end
            if ($urandom % 3 == 0) w = r;
            if (sq) begin
                // A sweep: the next instance, the sweep's row (rows follow each other in memory); it
                // moves on once that instance issues.
                i = (seq_i + 1) % INST; r = seq_r; w = seq_r;
            end
            cp = COPY && dr && ($urandom % 100) < copy;
            ir = COPY && dr && !cp && ($urandom % 100) < copy / 2;
            // (Image reads mostly chase the instance's last copy: rd_after_copies orders them.)
            if (ir && ($urandom % 4) != 0) k = last_k[i];
            iw = COPY && dw && ($urandom % 100) < copy / 4;
            // +far: a read with fence class 2 only of rows not written in the last 2 * 2^FS cycles (+ slack).
            if (far && dr && !ir && (now - wrote_at[i][r]) < 3 * (1 << FS)) dr = 0;
            // (+far image reads: class 2 covers no copy; rd_after_copies orders them. Image rows are
            // written by copies only then.)
            if (far) iw = 0;
            if (!pending[i] && (dr || dw)) begin
                reg [DW-1:0] d;
                reg [MW-1:0] m;
                if (sq) begin
                    seq_i = i;
                    if (i == INST - 1) seq_r = (seq_r + 1) % 4;
                end
                m = ($urandom % 2) ? '1 : MW'({$urandom, $urandom, $urandom, $urandom, $urandom});
                for (int q = 0; q < DW; q += 32) d[q +: 32] = $urandom;
                if (dr) begin
                    rd_valid <= 1; rd_len <= LW'(BEATS - 1); rd_tag <= TW'(i);
                    rd_addr <= ir ? img_of(i, k) : addr_of(i, r);
                    rd_fence <= far ? 4'd2 : 4'd0;
                    rd_copy <= cp; rd_copy_addr <= img_of(i, k); rd_after <= ir;
                    pending[i] = 1; is_copy[i] = cp; beat_at[i] = 0; reads <= reads + 1;
                    expect_row[i] = ir ? img[i][k] : ram[i][r];
                    if (cp) begin img[i][k] = ram[i][r]; copies <= copies + 1; last_k[i] = k; end
                    if (ir) img_reads <= img_reads + 1;
                end
                if (dw) begin
                    wr_valid <= 1; wr_len <= LW'(BEATS - 1); wr_data <= d; wr_mask <= m;
                    if (iw) begin
                        // A write of an image row (after any copy into it taken before).
                        wr_addr <= img_of(i, k);
                        wr_pair <= dr && ir;
                        img[i][k] = merge_row(img[i][k], d, m);
                    end else begin
                        wr_addr <= addr_of(i, w);
                        wr_pair <= dr && !ir && (w == r);
                        if (dr && !ir && w == r) pairs <= pairs + 1;
                        ram[i][w] = merge_row(ram[i][w], d, m);
                        wrote_at[i][w] = now;
                    end
                    writes <= writes + 1;
                end
            end
        end
    end

    task automatic clear_ref();
        for (int i = 0; i < INST; ++i) begin
            pending[i] = 0;
            for (int r = 0; r < ROWS; ++r) begin ram[i][r] = '0; img[i][r] = '0; wrote_at[i][r] = -1000000; end
        end
    endtask

    initial begin
        void'($value$plusargs("hi=%d", hi));
        void'($value$plusargs("stall=%d", stall));
        void'($value$plusargs("far=%d", far));
        void'($value$plusargs("resets=%d", resets));
        void'($value$plusargs("seq=%d", seq));
        void'($value$plusargs("copy=%d", copy));
        void'($value$plusargs("merge=%d", merge));
        void'($value$plusargs("wmerge=%d", wmerge));
        void'($value$plusargs("clean=%d", clean));
        void'($value$plusargs("wr=%d", wr));
        void'($value$plusargs("cycles=%d", cycles));
        for (int i = 0; i < INST; ++i) begin is_copy[i] = 0; beat_at[i] = 0; last_k[i] = 0; end
        clear_ref();
        // Memory starts clear and `zero` high (the runtime's clear): no fill at power-up.
        repeat (4) @(posedge clk);
        reset <= 0;
        for (int k = 0; k <= resets; ++k) begin
            wait (rd_free == FW'(QUEUE));
            repeat (cycles / (resets + 1)) @(posedge clk);
            if (k < resets) begin
                if (clean) begin
                    // The host's clear: idle, memory cleared, `zero` raised; then the reset (no fill).
                    @(negedge clk) stop = 1;
                    wait (rd_free == FW'(QUEUE) && wr_free == FW'(QUEUE));
                    repeat (40 * hi + 200) @(posedge clk);
                    if (ar_out != 0 || aw_out != 0 || aw_q.size() != 0) begin
                        $display("FAIL: the port not idle %0d %0d", ar_out, aw_out);
                        errors = errors + 1;
                    end
                    mem.delete();
                    zero = 0;
                    repeat (3) @(posedge clk);
                    zero = 1;
                    repeat (3) @(posedge clk);
                    fill_aws = 0;
                end
                // A reset (with requests in flight unless +clean): answers after it are not delivered; then
                // all rows read zero.
                @(negedge clk) reset = 1;
                repeat (2) @(posedge clk);
                @(negedge clk) reset = 0;
                clear_ref();
                if (clean) begin
                    wait (rd_free == FW'(QUEUE));
                    if (fill_aws != 0) begin
                        $display("FAIL: %0d fill bursts after a reset of a clean region", fill_aws);
                        errors = errors + 1;
                    end
                    @(negedge clk) stop = 0;
                end
            end
        end
        stop = 1;
        repeat (40 * hi + 2000) @(posedge clk);
        for (int i = 0; i < INST; ++i)
            if (pending[i]) begin
                $display("FAIL: instance %0d's read never answered", i);
                errors = errors + 1;
            end
        $display("%0d reads (%0d answered), %0d writes, %0d same-row pairs, %0d copies, %0d image reads, hi %0d, stall %0d, far %0d, resets %0d",
                 reads, answered, writes, pairs, copies, img_reads, hi, stall, far, resets);
        $display("AXI: %0d ARs, %0d AWs, at most %0d reads / %0d writes in flight, bursts of up to %0d / %0d beats", ars, aws,
                 ar_max, aw_max, ar_beats, aw_beats);
        if (merge && ars >= reads) begin
            $display("FAIL: no read burst merged (%0d ARs for %0d reads)", ars, reads);
            errors = errors + 1;
        end
        if (wmerge && aws >= writes) begin
            $display("FAIL: no write burst merged (%0d AWs for %0d writes)", aws, writes);
            errors = errors + 1;
        end
        if (errors == 0 && reads + writes > 40) $display("PASS");
        else $display("FAILED: %0d", errors);
        $finish;
    end
endmodule
