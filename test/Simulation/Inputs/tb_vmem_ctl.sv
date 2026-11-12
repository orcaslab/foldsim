// tb_vmem_ctl: the kernel's virtual-memory controller (shell/rtl/foldsim_vmem.sv)
// alone, driven as a folded simulator drives it -- an instance issues nothing
// while its read is pending, every request within the room `free` showed --
// against an AXI4 memory that keeps its rows late and reads them early: a
// write reaches memory only at its B (sent 1 to +hi host cycles after its
// last W beat, in order), and a read takes memory's rows as its AR is taken
// (answered 1 to +hi host cycles on, the bursts of the two ids interleaved).
// So a read that reached AR before the B of an earlier write of its row
// answers the old row: every answer is checked against a reference RAM
// (a read and a write of one row in one host cycle by RDW_READ_FIRST). +stall
// (percent) holds the channels' ready low at random. Prints PASS or the first
// mismatches.
`timescale 1ns/1ps
module tb;
    parameter IW = 3, AW = 4, DW = 1040, MW = 1, RDW_READ_FIRST = 1, QUEUE = 16, OUTSTANDING = 16,
              CACHE_ROWS = 0, KNOWN_ZERO = 0, FORWARD = 0, WRITES_OUT = 8, ROW_BYTES = 0, CYCLES = 30000, KZ_SHIFT = 0;
    localparam INST = 1 << IW, ROWS = 1 << AW, DB = (DW + 7) / 8 * 8, BYTES = DB / 8;
    localparam RB = ROW_BYTES ? ROW_BYTES : ((BYTES > 64) ? (1 << $clog2(BYTES)) : 64);
    localparam FW = $clog2(QUEUE + 1), GROUP = BYTES / MW;
    reg clk = 0, reset = 1;
    always #1 clk = ~clk;

    reg rd_valid = 0, wr_valid = 0;
    reg [IW-1:0] rd_inst, wr_inst;
    reg [AW-1:0] rd_row, wr_row;
    reg [DW-1:0] wr_data;
    reg [MW-1:0] wr_mask;
    wire rsp_valid;
    wire [IW-1:0] rsp_inst;
    wire [DW-1:0] rsp_data;
    wire [FW-1:0] free;
    wire m_awvalid, m_wvalid, m_wlast, m_bready, m_arvalid, m_rready, moving;
    reg m_awready = 0, m_wready = 0, m_bvalid = 0, m_arready = 0, m_rvalid = 0, m_rlast = 0;
    wire [63:0] m_awaddr, m_araddr;
    wire [0:0] m_awid, m_arid;
    reg [0:0] m_bid = 0, m_rid = 0;
    wire [7:0] m_awlen, m_arlen;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    reg [511:0] m_rdata;
    wire [7*48-1:0] tel;

    // (-DVPORT_CTL: the static port in the controller's place, foldsim_vport_ctl,
    // its zero fill over the image.)
`ifdef VPORT_CTL
    foldsim_vport_ctl #(.BYTES(longint'(INST) * ROWS * RB), .IW(IW),
`else
    foldsim_vmem #(.IW(IW),
`endif .AW(AW), .DW(DW), .MW(MW), .RDW_READ_FIRST(RDW_READ_FIRST), .QUEUE(QUEUE), .AXI_ID_W(1),
        .OUTSTANDING(OUTSTANDING), .INSTANCES(INST), .CACHE_ROWS(CACHE_ROWS), .KNOWN_ZERO(KNOWN_ZERO), .ROW_BYTES(RB),
        .WRITES_OUT(WRITES_OUT), .FORWARD(FORWARD), .KZ_SHIFT(KZ_SHIFT)) dut (
        .clk(clk), .reset(reset), .rd_valid(rd_valid), .rd_inst(rd_inst), .rd_row(rd_row), .wr_valid(wr_valid),
        .wr_inst(wr_inst), .wr_row(wr_row), .wr_data(wr_data), .wr_mask(wr_mask), .rsp_valid(rsp_valid),
        .rsp_inst(rsp_inst), .rsp_data(rsp_data), .free(free), .base(64'h1000_0000), .zero(1'b1), .move(1'b0),
        .moving(moving), .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awid(m_awid),
        .m_awlen(m_awlen), .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb),
        .m_wlast(m_wlast), .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bid(m_bid), .m_arvalid(m_arvalid),
        .m_arready(m_arready), .m_araddr(m_araddr), .m_arid(m_arid), .m_arlen(m_arlen), .m_rvalid(m_rvalid),
        .m_rready(m_rready), .m_rdata(m_rdata), .m_rid(m_rid), .m_rlast(m_rlast), .tel(tel));

    reg stop = 0;
    integer hi = 40, stall = 0, errors = 0, reads = 0, answered = 0, writes = 0, pairs = 0, hazards = 0;
    // The reference RAM, and each instance's pending read's expected row.
    reg [DB-1:0] ram [INST][ROWS];
    reg [DB-1:0] expect_row [INST];
    reg          pending [INST];
    // Memory: 64-byte lines by line address (from the base), zero where absent.
    reg [511:0] mem [longint];
    function automatic [511:0] line(input longint a);
        line = mem.exists(a) ? mem[a] : '0;
    endfunction

    // ---- AXI: writes to memory at their B, reads from memory at their AR ----
    typedef struct { longint addr; int beats; } burst_t;
    burst_t aw_q[$];
    reg [511:0] wbuf[$];
    reg [63:0]  sbuf[$];
    int     wdone = 0;   // whole W bursts not yet paired with their AW
    longint b_at[$];     // a write's B time, in order
    longint b_addr[$];
    int     b_beats[$];
    reg [511:0] b_data[$][$];
    reg [63:0]  b_strb[$][$];
    // Per id: every beat read (at AR), its burst's last, and when it may go.
    reg [511:0] r_data[2][$];
    reg         r_last[2][$];
    longint     r_at[2][$];
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
        // A burst whose AW and whole W are in (W may come before its AW): its
        // B (and its effect on memory) 1..hi cycles on, in order.
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
                a = (b_addr[0] - 64'h1000_0000) / 64 + k;
                l = line(a);
                for (int j = 0; j < 64; ++j)
                    if (b_strb[0][k][j]) l[j*8 +: 8] = b_data[0][k][j*8 +: 8];
                mem[a] = l;
            end
            void'(b_at.pop_front()); void'(b_addr.pop_front()); void'(b_beats.pop_front());
            void'(b_data.pop_front()); void'(b_strb.pop_front());
            m_bvalid <= 1; m_bid <= 0;
        end
        if (m_arvalid & m_arready) begin
            automatic reg [511:0] d[$];
            longint t;
            t = now + 1 + longint'($urandom % hi);
            for (int k = 0; k <= int'(m_arlen); ++k) d.push_back(line((m_araddr - 64'h1000_0000) / 64 + k));
            if (r_at[m_arid].size() && t < r_at[m_arid][$]) t = r_at[m_arid][$];
            for (int k = 0; k < d.size(); ++k) begin
                r_data[m_arid].push_back(d[k]);
                r_last[m_arid].push_back(k == d.size() - 1);
                r_at[m_arid].push_back(t);
            end
        end
        if (m_rvalid & m_rready) m_rvalid <= 0;
        if (~m_rvalid | m_rready) begin
            int id;
            id = int'($urandom % 2);
            if (!(r_at[id].size() && r_at[id][0] <= now)) id = 1 - id;
            if (r_at[id].size() && r_at[id][0] <= now && ($urandom % 100) >= stall) begin
                m_rvalid <= 1; m_rid <= id[0];
                m_rdata <= r_data[id].pop_front();
                m_rlast <= r_last[id].pop_front();
                void'(r_at[id].pop_front());
            end
        end
    end

    // ---- The simulator: requests within the room free showed ----
    function automatic [DB-1:0] merge(input [DB-1:0] old, input [DB-1:0] d, input [MW-1:0] m);
        merge = old;
        for (int j = 0; j < BYTES; ++j)
            if (m[j / GROUP]) merge[j*8 +: 8] = d[j*8 +: 8];
    endfunction
    always @(posedge clk) begin
        if (rsp_valid) begin
            answered <= answered + 1;
            if (!pending[rsp_inst]) begin
                $display("FAIL: an answer for instance %0d with no read", rsp_inst);
                errors = errors + 1;
            end else if (rsp_data !== expect_row[rsp_inst][DW-1:0]) begin
                if (errors < 5) $display("FAIL: instance %0d answered %h, expected %h", rsp_inst, rsp_data, expect_row[rsp_inst][DW-1:0]);
                errors = errors + 1;
            end
            pending[rsp_inst] = 0;
        end
        rd_valid <= 0; wr_valid <= 0;
        // (free does not yet count the last cycle's requests: room for both.)
        if (!reset && !stop && free >= 4 && ($urandom % 100) < 70) begin
            reg [IW-1:0] i;
            reg [AW-1:0] r, w;
            reg dr, dw;
            // (Assigned here: a declaration's initializer runs once.)
            i = IW'($urandom);
            r = AW'($urandom % ((AW > 2) ? 4 : ROWS));
            w = AW'($urandom % ((AW > 2) ? 4 : ROWS));
            dr = 1'($urandom % 2);
            dw = 1'($urandom % 2);
            if ($urandom % 4 == 0) begin r = AW'($urandom); w = AW'($urandom); end
            if ($urandom % 3 == 0) w = r;
            if (!pending[i] && (dr || dw)) begin
                reg [DB-1:0] d;
                reg [MW-1:0] m;
                m = (MW == 1) ? MW'(1) : (($urandom % 2) ? '1 : MW'({$urandom, $urandom, $urandom, $urandom, $urandom}));
                for (int k = 0; k < DB; k += 32) d[k +: 32] = $urandom;
                d = DB'(d[DW-1:0]);
                if (dr) begin
                    rd_valid <= 1; rd_inst <= i; rd_row <= r;
                    pending[i] = 1; reads <= reads + 1;
                    expect_row[i] = (dw && w == r && !RDW_READ_FIRST) ? merge(ram[i][r], d, m) : ram[i][r];
                    if (dw && w == r) pairs <= pairs + 1;
                end
                if (dw) begin
                    wr_valid <= 1; wr_inst <= i; wr_row <= w; wr_data <= d[DW-1:0]; wr_mask <= m;
                    ram[i][w] = merge(ram[i][w], d, m);
                    writes <= writes + 1;
                end
            end
        end
    end

    initial begin
        void'($value$plusargs("hi=%d", hi));
        void'($value$plusargs("stall=%d", stall));
        for (int i = 0; i < INST; ++i) begin
            pending[i] = 0;
            for (int r = 0; r < ROWS; ++r) ram[i][r] = '0;
        end
        repeat (4) @(posedge clk);
        reset <= 0;
        // The reset's drain: free comes back at QUEUE.
        wait (free == FW'(QUEUE));
        repeat (CYCLES) @(posedge clk);
        stop = 1;
        repeat (20 * hi + 2000) @(posedge clk);
        for (int i = 0; i < INST; ++i)
            if (pending[i]) begin
                $display("FAIL: instance %0d's read never answered", i);
                errors = errors + 1;
            end
        $display("%0d reads (%0d answered), %0d writes, %0d same-row pairs, hi %0d, stall %0d", reads, answered, writes,
                 pairs, hi, stall);
        if (errors == 0 && reads > 100) $display("PASS");
        else $display("FAILED: %0d", errors);
        $finish;
    end
endmodule
