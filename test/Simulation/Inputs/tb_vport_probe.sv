// tb_vport_probe: request-to-answer latency of one virtual-memory
// controller (foldsim_vmem, or -DVPORT_CTL foldsim_vport_ctl) for fixed
// access patterns, against the Verilator emulator's memory model: every
// channel always ready, a read's beats LAT host cycles after its AR (in order,
// a beat a cycle), a write's B LAT host cycles after its last W beat (in
// order). Memory takes a write at its B (stricter than the emulator, which
// takes it at W: a controller correct here is correct there).
// Patterns (+mode=):
//   0 cold reads: one read, its answer, 50 idle cycles, again (new rows)
//   1 RAW same row: write row r, read row r +d cycles later
//   2 RAW other row: write row r, read row r^1 +d cycles later
//   3 read-first pairs: read+write row r in one cycle; the next pair of row r
//     +d cycles after the answer (an RTL memory at its state pass)
//   4 streams: NI instances, each a read and a write of its own row every
//     +d cycles while not pending (a class copy's members)
// Prints the average latency and checks every answer against a reference RAM;
// +maxavg=N fails a run whose average latency exceeds N host cycles.
`timescale 1ns/1ps
module tb;
    parameter IW = 3, AW = 6, DW = 1040, MW = 1, RDW_READ_FIRST = 1, QUEUE = 32, OUTSTANDING = 128,
              KNOWN_ZERO = 1, FORWARD = 1, WRITES_OUT = 128, ROW_BYTES = 0, KZ_SHIFT = 0, HZ = 11;
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
    reg m_bvalid = 0, m_rvalid = 0, m_rlast = 0;
    wire m_awready = 1, m_wready = 1, m_arready = 1;
    wire [63:0] m_awaddr, m_araddr;
    wire [3:0] m_awid, m_arid;
    reg [3:0] m_bid = 0, m_rid = 0;
    wire [7:0] m_awlen, m_arlen;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    reg [511:0] m_rdata;
    wire [7*48-1:0] tel;

`ifdef VPORT_CTL
    foldsim_vport_ctl #(.BYTES(longint'(INST) * ROWS * RB), .HZ(HZ), .IW(IW),
`else
    foldsim_vmem #(.IW(IW),
`endif .AW(AW), .DW(DW), .MW(MW), .RDW_READ_FIRST(RDW_READ_FIRST), .QUEUE(QUEUE), .AXI_ID_W(4),
        .OUTSTANDING(OUTSTANDING), .INSTANCES(INST), .CACHE_ROWS(0), .KNOWN_ZERO(KNOWN_ZERO), .ROW_BYTES(RB),
        .WRITES_OUT(WRITES_OUT), .FORWARD(FORWARD), .KZ_SHIFT(KZ_SHIFT)) dut (
        .clk(clk), .reset(reset), .rd_valid(rd_valid), .rd_inst(rd_inst), .rd_row(rd_row), .wr_valid(wr_valid),
        .wr_inst(wr_inst), .wr_row(wr_row), .wr_data(wr_data), .wr_mask(wr_mask), .rsp_valid(rsp_valid),
        .rsp_inst(rsp_inst), .rsp_data(rsp_data), .free(free), .base(64'h1000_0000), .zero(1'b0), .move(1'b0),
        .moving(moving), .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awid(m_awid),
        .m_awlen(m_awlen), .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb),
        .m_wlast(m_wlast), .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bid(m_bid), .m_arvalid(m_arvalid),
        .m_arready(m_arready), .m_araddr(m_araddr), .m_arid(m_arid), .m_arlen(m_arlen), .m_rvalid(m_rvalid),
        .m_rready(m_rready), .m_rdata(m_rdata), .m_rid(m_rid), .m_rlast(m_rlast), .tel(tel));

    integer lat = 100, mode = 0, d = 0, n = 200, ni = 8, errors = 0, maxavg = 0;
    reg [DB-1:0] ram [INST][ROWS];
    reg [DB-1:0] expect_row [INST];
    reg          pending [INST];
    longint      issued_at [INST], last_ans [INST];
    longint      lat_sum = 0, lat_n = 0, lat_max = 0;
    reg [511:0] mem [longint];
    function automatic [511:0] line(input longint a);
        line = mem.exists(a) ? mem[a] : '0;
    endfunction

    typedef struct { longint addr; int beats; } burst_t;
    burst_t aw_q[$];
    reg [511:0] wbuf[$];
    reg [63:0]  sbuf[$];
    int     wdone = 0;
    longint b_at[$], b_addr[$];
    int     b_beats[$];
    reg [511:0] b_data[$][$];
    reg [63:0]  b_strb[$][$];
    reg [511:0] r_data[$];
    reg         r_last[$];
    reg [3:0]   r_id[$];
    longint     r_at[$];
    longint now = 0;
    always @(posedge clk) now <= now + 1;

    always @(posedge clk) begin
        if (m_awvalid) aw_q.push_back('{longint'(m_awaddr), int'(m_awlen) + 1});
        if (m_wvalid) begin
            wbuf.push_back(m_wdata);
            sbuf.push_back(m_wstrb);
            if (m_wlast) wdone = wdone + 1;
        end
        if (aw_q.size() && wdone) begin
            burst_t b;
            automatic reg [511:0] dd[$];
            automatic reg [63:0]  s[$];
            b = aw_q.pop_front();
            wdone = wdone - 1;
            for (int k = 0; k < b.beats; ++k) begin
                dd.push_back(wbuf.pop_front());
                s.push_back(sbuf.pop_front());
            end
            b_at.push_back(now + lat); b_addr.push_back(b.addr); b_beats.push_back(b.beats);
            b_data.push_back(dd); b_strb.push_back(s);
        end
        m_bvalid <= 0;
        if (b_at.size() && b_at[0] <= now) begin
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
            m_bvalid <= 1;
        end
        if (m_arvalid) begin
            for (int k = 0; k <= int'(m_arlen); ++k) begin
                r_data.push_back(line((m_araddr - 64'h1000_0000) / 64 + k));
                r_last.push_back(k == int'(m_arlen));
                r_id.push_back(m_arid);
                r_at.push_back(now + lat);
            end
        end
        m_rvalid <= 0;
        if (r_at.size() && r_at[0] <= now) begin
            m_rvalid <= 1;
            m_rdata <= r_data.pop_front();
            m_rlast <= r_last.pop_front();
            m_rid <= r_id.pop_front();
            void'(r_at.pop_front());
        end
    end

    function automatic [DB-1:0] merge(input [DB-1:0] old, input [DB-1:0] dd, input [MW-1:0] m);
        merge = old;
        for (int j = 0; j < BYTES; ++j)
            if (m[j / GROUP]) merge[j*8 +: 8] = dd[j*8 +: 8];
    endfunction
    always @(posedge clk) begin
        if (rsp_valid) begin
            longint l;
            if (!pending[rsp_inst]) begin
                $display("FAIL: an answer for instance %0d with no read", rsp_inst);
                errors = errors + 1;
            end else if (rsp_data !== expect_row[rsp_inst][DW-1:0]) begin
                if (errors < 5) $display("FAIL: instance %0d answered wrong", rsp_inst);
                errors = errors + 1;
            end
            pending[rsp_inst] = 0;
            l = now - issued_at[rsp_inst];
            lat_sum += l; lat_n += 1; if (l > lat_max) lat_max = l;
            last_ans[rsp_inst] = now;
        end
    end

    task automatic issue(input bit dr, input bit dw, input [IW-1:0] i, input [AW-1:0] r, input [AW-1:0] w);
        reg [DB-1:0] dd;
        for (int k = 0; k < DB; k += 32) dd[k +: 32] = $urandom;
        dd = DB'(dd[DW-1:0]);
        rd_valid <= dr; wr_valid <= dw;
        if (dr) begin
            rd_inst <= i; rd_row <= r; pending[i] = 1; issued_at[i] = now + 1;
            expect_row[i] = (dw && w == r && !RDW_READ_FIRST) ? merge(ram[i][r], dd, '1) : ram[i][r];
        end
        if (dw) begin
            wr_inst <= i; wr_row <= w; wr_data <= dd[DW-1:0]; wr_mask <= '1;
            ram[i][w] = merge(ram[i][w], dd, '1);
        end
        @(posedge clk);
        #0.5 rd_valid = 0; wr_valid = 0;
    endtask

    initial begin
        void'($value$plusargs("lat=%d", lat));
        void'($value$plusargs("mode=%d", mode));
        void'($value$plusargs("d=%d", d));
        void'($value$plusargs("n=%d", n));
        void'($value$plusargs("ni=%d", ni));
        void'($value$plusargs("maxavg=%d", maxavg));
        for (int i = 0; i < INST; ++i) begin
            pending[i] = 0; last_ans[i] = 0;
            for (int r = 0; r < ROWS; ++r) ram[i][r] = '0;
        end
        repeat (4) @(posedge clk);
        reset <= 0;
        @(posedge clk);
        wait (free == FW'(QUEUE));
        @(posedge clk);
        for (int k = 0; k < n; ++k) begin
            case (mode)
                0: begin
                    issue(1, 0, 0, AW'(k), 0);
                    wait (!pending[0]); repeat (50) @(posedge clk);
                end
                1, 2: begin
                    issue(0, 1, 0, 0, AW'(2 * k));
                    repeat (d) @(posedge clk);
                    issue(1, 0, 0, AW'(2 * k + (mode == 2)), 0);
                    wait (!pending[0]); repeat (lat + 300) @(posedge clk);
                end
                3: begin
                    issue(1, 1, 0, AW'(k % 4), AW'(k % 4));
                    wait (!pending[0]); repeat (d) @(posedge clk);
                end
                4: begin
                    // one instance a cycle in turn (a copy's threads), when not pending
                    for (int i = 0; i < ni; ++i) begin
                        if (!pending[i] && free >= 10) issue(1, 1, IW'(i), AW'(k % 8), AW'(k % 8));
                        else @(posedge clk);
                    end
                    repeat (d) @(posedge clk);
                end
            endcase
        end
        // Every read answered, or 200,000 host cycles (a hang).
        for (int t = 0; t < 200000; ++t) begin
            automatic bit any = 0;
            for (int i = 0; i < INST; ++i) any |= pending[i];
            if (!any) break;
            @(posedge clk);
        end
        repeat (4 * lat + 500) @(posedge clk);
        for (int i = 0; i < INST; ++i)
            if (pending[i]) begin $display("FAIL: instance %0d never answered", i); errors++; end
        $display("mode %0d d %0d lat %0d: %0d answers, avg latency %0d.%0d, max %0d, %0d host cycles", mode, d, lat, lat_n,
                 lat_sum / (lat_n ? lat_n : 1), (lat_sum * 10 / (lat_n ? lat_n : 1)) % 10, lat_max, now);
        if (maxavg != 0 && lat_sum > longint'(maxavg) * lat_n) begin
            $display("FAIL: average latency above %0d", maxavg);
            errors++;
        end
        if (errors == 0) $display("PASS"); else $display("FAILED: %0d", errors);
        $finish;
    end
endmodule
