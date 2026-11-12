// tb_vport: the kernel's static port (shell/rtl/foldsim_vport.sv) alone, driven
// as a folded simulator drives it -- at most a read and a write a host cycle,
// within the room rd_free/wr_free showed; an instance issues nothing while its
// read is pending -- against an AXI4 memory that keeps its rows late and reads
// them early: a write reaches memory only at its B (1 to +hi host cycles after
// its last W beat, in order), a read takes memory's rows as its AR is taken
// (answered 1 to +hi host cycles on, in order: one id). So a read that reached
// AR before the B of an earlier write of its row would answer the old row:
// every answer is checked against a reference RAM. Reads carry fence class 0
// (any earlier write) or, with +far, the class whose distance the driver keeps
// (it then never reads a row written within that distance). A read and a write
// of one row in one host cycle form a read-first pair (wr_pair): the read
// answers the old row. +stall (percent) holds the channels' ready low at
// random. +resets=N resets the port N times with requests in flight; after
// each, every row reads zero (the port's zero fill). Prints PASS or the first
// mismatches.
`timescale 1ns/1ps
module tb;
    parameter IW = 3, AW = 4, DW = 1040, QUEUE = 16, OUTSTANDING = 16, WRITES_OUT = 8, CYCLES = 30000, FS = 4, HZ = 0;
    localparam INST = 1 << IW, ROWS = 1 << AW, MW = DW / 8;
    localparam BEATS = (DW + 511) / 512, LW = (BEATS > 1) ? $clog2(BEATS) : 1;
    localparam FW = $clog2(QUEUE + 1), OW = 20, TW = IW, FCW = 4;
    // A row's beats in memory: rows BEATS beats apart, instances first.
    localparam longint BASE = 64'h1000_0000;
    localparam longint BYTES = longint'(INST) * ROWS * BEATS * 64;
    reg clk = 0, reset = 1;
    always #1 clk = ~clk;

    reg rd_valid = 0, wr_valid = 0, wr_pair = 0;
    reg [OW-1:0] rd_addr = 0, wr_addr = 0;
    reg [LW-1:0] rd_len = 0, wr_len = 0;
    reg [TW-1:0] rd_tag = 0;
    reg [FCW-1:0] rd_fence = 0;
    reg [DW-1:0] wr_data = 0;
    reg [MW-1:0] wr_mask = 0;
    wire rsp_valid, rsp_last, busy;
    wire [TW-1:0] rsp_tag;
    wire [LW-1:0] rsp_beat;
    wire [511:0] rsp_data;
    wire [FW-1:0] rd_free, wr_free;
    wire m_awvalid, m_wvalid, m_wlast, m_bready, m_arvalid, m_rready;
    reg m_awready = 0, m_wready = 0, m_bvalid = 0, m_arready = 0, m_rvalid = 0, m_rlast = 0;
    wire [63:0] m_awaddr, m_araddr;
    wire [7:0] m_awlen, m_arlen;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    reg [511:0] m_rdata = 0;

    foldsim_vport #(.DW(DW), .MW(MW), .TW(TW), .OW(OW), .QUEUE(QUEUE), .OUTSTANDING(OUTSTANDING),
        .WRITES_OUT(WRITES_OUT), .FCLASSES(16), .FS(FS), .HZ(HZ)) dut (
        .clk(clk), .reset(reset),
        .rd_valid(rd_valid), .rd_addr(rd_addr), .rd_len(rd_len), .rd_tag(rd_tag), .rd_fence(rd_fence),
        .wr_valid(wr_valid), .wr_addr(wr_addr), .wr_len(wr_len), .wr_pair(wr_pair), .wr_data(wr_data), .wr_mask(wr_mask),
        .rsp_valid(rsp_valid), .rsp_tag(rsp_tag), .rsp_beat(rsp_beat), .rsp_last(rsp_last), .rsp_data(rsp_data),
        .rd_free(rd_free), .wr_free(wr_free), .base(BASE), .bytes(BYTES), .move(1'b0), .busy(busy),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awlen(m_awlen),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_araddr(m_araddr), .m_arlen(m_arlen), .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rdata(m_rdata),
        .m_rlast(m_rlast));

    reg stop = 0;
    integer hi = 40, stall = 0, far = 0, resets = 0, errors = 0, reads = 0, answered = 0, writes = 0, pairs = 0;
    reg [BEATS*512-1:0] ram [INST][ROWS];
    reg [BEATS*512-1:0] expect_row [INST];
    reg [BEATS*512-1:0] got [INST];
    reg          pending [INST];
    // When each row was last written (host cycle), for +far's reads.
    longint      wrote_at [INST][ROWS];
    reg [511:0] mem [longint];
    function automatic [511:0] line(input longint a);
        line = mem.exists(a) ? mem[a] : '0;
    endfunction

    // ---- AXI memory: writes at their B, reads at their AR, one id, in order ----
    typedef struct { longint addr; int beats; } burst_t;
    burst_t aw_q[$];
    reg [511:0] wbuf[$];
    reg [63:0]  sbuf[$];
    int     wdone = 0;
    longint b_at[$];
    longint b_addr[$];
    int     b_beats[$];
    reg [511:0] b_data[$][$];
    reg [63:0]  b_strb[$][$];
    reg [511:0] r_data[$];
    reg         r_last[$];
    longint     r_at[$];
    longint now = 0;
    always @(posedge clk) now <= now + 1;

    always @(posedge clk) begin
        m_awready <= ($urandom % 100) >= stall;
        m_wready  <= ($urandom % 100) >= stall;
        m_arready <= ($urandom % 100) >= stall;
        if (m_awvalid & m_awready) aw_q.push_back('{longint'(m_awaddr), int'(m_awlen) + 1});
        if (m_wvalid & m_wready) begin
            wbuf.push_back(m_wdata);
            sbuf.push_back(m_wstrb);
            if (m_wlast) wdone = wdone + 1;
        end
        if (aw_q.size() && wdone) begin
            burst_t b;
            automatic reg [511:0] d[$];
            automatic reg [63:0]  s[$];
            longint t;
            b = aw_q.pop_front();
            wdone = wdone - 1;
            t = now + 1 + longint'($urandom % hi);
            if (b_at.size() && t < b_at[$]) t = b_at[$];
            for (int k = 0; k < b.beats; ++k) begin
                d.push_back(wbuf.pop_front());
                s.push_back(sbuf.pop_front());
            end
            b_at.push_back(t); b_addr.push_back(b.addr); b_beats.push_back(b.beats);
            b_data.push_back(d); b_strb.push_back(s);
        end
        if (m_bvalid & m_bready) m_bvalid <= 0;
        if ((~m_bvalid | m_bready) && b_at.size() && b_at[0] <= now) begin
            for (int k = 0; k < b_beats[0]; ++k) begin
                longint a;
                reg [511:0] l;
                a = (b_addr[0] - BASE) / 64 + k;
                l = line(a);
                for (int j = 0; j < 64; ++j)
                    if (b_strb[0][k][j]) l[j*8 +: 8] = b_data[0][k][j*8 +: 8];
                mem[a] = l;
            end
            void'(b_at.pop_front()); void'(b_addr.pop_front()); void'(b_beats.pop_front());
            void'(b_data.pop_front()); void'(b_strb.pop_front());
            m_bvalid <= 1;
        end
        if (m_arvalid & m_arready) begin
            longint t;
            t = now + 1 + longint'($urandom % hi);
            if (r_at.size() && t < r_at[$]) t = r_at[$];
            for (int k = 0; k <= int'(m_arlen); ++k) begin
                r_data.push_back(line((m_araddr - BASE) / 64 + k));
                r_last.push_back(k == int'(m_arlen));
                r_at.push_back(t);
            end
        end
        m_rvalid <= 0;
        if (r_at.size() && r_at[0] <= now && ($urandom % 100) >= stall) begin
            m_rvalid <= 1;
            m_rdata <= r_data.pop_front();
            m_rlast <= r_last.pop_front();
            void'(r_at.pop_front());
        end
    end

    // ---- The simulator ----
    function automatic [BEATS*512-1:0] merge(input [BEATS*512-1:0] old, input [DW-1:0] d, input [MW-1:0] m);
        merge = old;
        for (int j = 0; j < MW; ++j)
            if (m[j]) merge[j*8 +: 8] = d[j*8 +: 8];
    endfunction
    function automatic [OW-1:0] addr_of(input int i, input int r);
        addr_of = OW'((r * INST + i) * BEATS);
    endfunction
    always @(posedge clk) begin
        if (rsp_valid) begin
            got[rsp_tag][32'(rsp_beat) * 512 +: 512] = rsp_data;
            if (!pending[rsp_tag]) begin
                $display("FAIL: an answer for instance %0d with no read", rsp_tag);
                errors = errors + 1;
            end else if (rsp_last) begin
                answered <= answered + 1;
                if (got[rsp_tag][DW-1:0] !== expect_row[rsp_tag][DW-1:0]) begin
                    if (errors < 5) $display("FAIL: instance %0d answered %h, expected %h", rsp_tag, got[rsp_tag][DW-1:0],
                                             expect_row[rsp_tag][DW-1:0]);
                    errors = errors + 1;
                end
                pending[rsp_tag] = 0;
            end
        end
        rd_valid <= 0; wr_valid <= 0;
        // (free does not yet count the last cycle's requests: room for both.)
        if (!reset && !stop && rd_free >= 3 && wr_free >= 3 && ($urandom % 100) < 70) begin
            int i, r, w;
            reg dr, dw;
            i = int'($urandom % INST);
            r = int'($urandom % ((AW > 2) ? 4 : ROWS));
            w = int'($urandom % ((AW > 2) ? 4 : ROWS));
            dr = 1'($urandom % 2);
            dw = 1'($urandom % 2);
            if ($urandom % 4 == 0) begin r = int'($urandom % ROWS); w = int'($urandom % ROWS); end
            if ($urandom % 3 == 0) w = r;
            // +far: a read with fence class 2 only of rows not written in the last 2 * 2^FS cycles (+ slack).
            if (far && dr && (now - wrote_at[i][r]) < 3 * (1 << FS)) dr = 0;
            if (!pending[i] && (dr || dw)) begin
                reg [DW-1:0] d;
                reg [MW-1:0] m;
                m = ($urandom % 2) ? '1 : MW'({$urandom, $urandom, $urandom, $urandom, $urandom});
                for (int k = 0; k < DW; k += 32) d[k +: 32] = $urandom;
                if (dr) begin
                    rd_valid <= 1; rd_addr <= addr_of(i, r); rd_len <= LW'(BEATS - 1); rd_tag <= TW'(i);
                    rd_fence <= far ? 4'd2 : 4'd0;
                    pending[i] = 1; reads <= reads + 1;
                    expect_row[i] = ram[i][r];
                end
                if (dw) begin
                    wr_valid <= 1; wr_addr <= addr_of(i, w); wr_len <= LW'(BEATS - 1); wr_data <= d; wr_mask <= m;
                    wr_pair <= dr && (w == r);
                    if (dr && w == r) pairs <= pairs + 1;
                    ram[i][w] = merge(ram[i][w], d, m);
                    wrote_at[i][w] = now;
                    writes <= writes + 1;
                end
            end
        end
    end

    initial begin
        void'($value$plusargs("hi=%d", hi));
        void'($value$plusargs("stall=%d", stall));
        void'($value$plusargs("far=%d", far));
        void'($value$plusargs("resets=%d", resets));
        for (int i = 0; i < INST; ++i) begin
            pending[i] = 0;
            for (int r = 0; r < ROWS; ++r) begin ram[i][r] = '0; wrote_at[i][r] = -1000000; end
        end
        // Memory starts dirty: the zero fill must clear it.
        for (longint a = 0; a < BYTES / 64; ++a) mem[a] = {16{32'hdead_beef}};
        repeat (4) @(posedge clk);
        reset <= 0;
        for (int k = 0; k <= resets; ++k) begin
            wait (rd_free == FW'(QUEUE));
            repeat (CYCLES / (resets + 1)) @(posedge clk);
            if (k < resets) begin
                // A reset with requests in flight: answers after it are not delivered; then all rows read zero.
                @(negedge clk) reset = 1;
                repeat (2) @(posedge clk);
                @(negedge clk) reset = 0;
                for (int i = 0; i < INST; ++i) begin
                    pending[i] = 0;
                    for (int r = 0; r < ROWS; ++r) begin ram[i][r] = '0; wrote_at[i][r] = -1000000; end
                end
            end
        end
        stop = 1;
        repeat (20 * hi + 2000) @(posedge clk);
        for (int i = 0; i < INST; ++i)
            if (pending[i]) begin
                $display("FAIL: instance %0d's read never answered", i);
                errors = errors + 1;
            end
        $display("%0d reads (%0d answered), %0d writes, %0d same-row pairs, hi %0d, stall %0d, far %0d, resets %0d", reads,
                 answered, writes, pairs, hi, stall, far, resets);
        if (errors == 0 && reads > 40) $display("PASS");
        else $display("FAILED: %0d", errors);
        $finish;
    end
endmodule
