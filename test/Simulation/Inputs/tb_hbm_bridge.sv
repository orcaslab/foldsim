// tb_hbm_bridge: foldsim_hbm_bridge between a 512-bit AXI4 master in one clock
// and a 256-bit memory in another (+sp=/+mp= the two half periods in ps, so
// any ratio), every handshake stalled at random (+stall percent). The master
// writes random bursts (1-4 beats, random strobes, random ids) and reads rows
// whose writes have all had their B; the memory applies W beats in AW order
// and answers reads in order 1 to +hi of its cycles late. Every read beat is
// checked against the master's shadow image (updated at each B). Prints
// PASS or the first mismatches.
`timescale 1ps/1ps
module tb;
    localparam ROWS = 64, IDW = 2;
    reg s_clk = 0, m_clk = 0, s_reset = 1, m_reset = 1;
    integer sp = 2500, mp = 1250, stall = 20, hi = 20, n_ops = 3000;
    initial begin
        void'($value$plusargs("sp=%d", sp));
        void'($value$plusargs("mp=%d", mp));
        void'($value$plusargs("stall=%d", stall));
        void'($value$plusargs("hi=%d", hi));
        void'($value$plusargs("ops=%d", n_ops));
    end
    always #(sp) s_clk = ~s_clk;
    always #(mp) m_clk = ~m_clk;

    reg s_awvalid = 0, s_wvalid = 0, s_bready = 0, s_arvalid = 0, s_rready = 0, s_wlast = 0;
    wire s_awready, s_wready, s_bvalid, s_arready, s_rvalid, s_rlast;
    reg [63:0] s_awaddr = 0, s_araddr = 0;
    reg [IDW-1:0] s_awid = 0, s_arid = 0;
    wire [IDW-1:0] s_bid, s_rid;
    reg [7:0] s_awlen = 0, s_arlen = 0;
    reg [511:0] s_wdata = 0;
    reg [63:0] s_wstrb = 0;
    wire [511:0] s_rdata;
    wire m_awvalid, m_wvalid, m_wlast, m_bready, m_arvalid, m_rready;
    reg m_awready = 0, m_wready = 0, m_bvalid = 0, m_arready = 0, m_rvalid = 0, m_rlast = 0;
    wire [63:0] m_awaddr, m_araddr;
    wire [IDW-1:0] m_awid, m_arid;
    reg [IDW-1:0] m_bid = 0, m_rid = 0;
    wire [7:0] m_awlen, m_arlen;
    wire [255:0] m_wdata;
    wire [31:0] m_wstrb;
    reg [255:0] m_rdata = 0;

    foldsim_hbm_bridge #(.IDW(IDW)) dut (.*);

    // ---- the 256-bit memory, in m_clk ----
    reg [255:0] mem [2 * ROWS];
    typedef struct { longint addr; int beats; int id; } burst_t;
    burst_t awq[$], arq[$];
    int     wbeat = 0;
    reg [255:0] rq_d[$]; reg rq_l[$]; int rq_id[$]; longint rq_at[$];
    longint mcyc = 0;
    int bq[$];
    reg [255:0] wbuf_d[$]; reg [31:0] wbuf_s[$]; reg wbuf_l[$];
    always @(posedge m_clk) begin
        mcyc <= mcyc + 1;
        m_awready <= ($urandom % 100) >= stall;
        m_wready  <= ($urandom % 100) >= stall;
        m_arready <= ($urandom % 100) >= stall;
        if (m_awvalid && m_awready) awq.push_back('{longint'(m_awaddr), int'(m_awlen) + 1, int'(m_awid)});
        if (m_wvalid && m_wready) begin
            wbuf_d.push_back(m_wdata); wbuf_s.push_back(m_wstrb); wbuf_l.push_back(m_wlast);
        end
        // W may come before its AW (AXI allows it): beats apply once their AW is in.
        while (awq.size() && wbuf_d.size()) begin
            automatic longint a = awq[0].addr / 32 + wbeat;
            automatic reg [255:0] d = wbuf_d.pop_front(); automatic reg [31:0] st = wbuf_s.pop_front(); automatic reg l = wbuf_l.pop_front();
            for (int j = 0; j < 32; j++) if (st[j]) mem[a % (2 * ROWS)][j*8 +: 8] = d[j*8 +: 8];
            wbeat = wbeat + 1;
            if ((wbeat == awq[0].beats) != l) begin $display("FAIL: wlast at beat %0d of %0d", wbeat, awq[0].beats); end
            if (wbeat == awq[0].beats) begin bq.push_back(awq[0].id); void'(awq.pop_front()); wbeat = 0; end
        end
        if (m_bvalid && m_bready) m_bvalid <= 0;
        if ((!m_bvalid || m_bready) && bq.size() && ($urandom % 100) >= stall) begin
            m_bvalid <= 1; m_bid <= IDW'(bq.pop_front());
        end
        if (m_arvalid && m_arready) begin
            automatic longint t = mcyc + 1 + longint'($urandom % hi);
            if (rq_at.size() && t < rq_at[$]) t = rq_at[$];
            for (int k = 0; k <= int'(m_arlen); k++) begin
                rq_d.push_back(mem[(m_araddr / 32 + k) % (2 * ROWS)]);
                rq_l.push_back(k == int'(m_arlen)); rq_id.push_back(int'(m_arid)); rq_at.push_back(t);
            end
        end
        if (m_rvalid && m_rready) m_rvalid <= 0;
        if ((!m_rvalid || m_rready) && rq_at.size() && rq_at[0] <= mcyc && ($urandom % 100) >= stall) begin
            m_rvalid <= 1; m_rdata <= rq_d.pop_front(); m_rlast <= rq_l.pop_front(); m_rid <= IDW'(rq_id.pop_front());
            void'(rq_at.pop_front());
        end
    end

    // ---- the 512-bit master, in s_clk ----
    reg [511:0] shadow [ROWS];
    int pending [ROWS];          // writes of a row without their B
    int wleft[$];                // AWs issued: W beats left to send
    reg [511:0] wq_d[$]; reg [63:0] wq_s[$];
    int bwait[$][$];             // per write, its rows (in AW order)
    reg [511:0] bdata[$][$]; reg [63:0] bstrb[$][$];
    int rexp_row[$]; int rexp_id[$]; int rexp_last[$];
    int errors = 0, writes = 0, reads = 0, beats_ok = 0, ops = 0;
    always @(posedge s_clk) begin
        if (!s_reset) begin
            s_bready <= ($urandom % 100) >= stall;
            s_rready <= ($urandom % 100) >= stall;
            // AW
            if (s_awvalid && s_awready) s_awvalid <= 0;
            if (s_arvalid && s_arready) s_arvalid <= 0;
            if ((!s_awvalid || s_awready) && ops < n_ops && wq_d.size() < 16 && bwait.size() < 24 && ($urandom % 3) == 0) begin
                automatic int row = int'($urandom % (ROWS - 4)), len = int'($urandom % 4);
                automatic int rows[$];
                automatic bit free_rows = 1;
                for (int k = 0; k <= len; k++) if (pending[row + k] >= 1000) free_rows = 0;
                if (free_rows) begin
                automatic reg [511:0] ds[$];
                automatic reg [63:0] ss[$];
                s_awvalid <= 1; s_awaddr <= 64'(row) * 64; s_awlen <= 8'(len); s_awid <= IDW'($urandom);
                for (int k = 0; k <= len; k++) begin
                    reg [511:0] d; reg [63:0] st;
                    for (int b = 0; b < 16; b++) d[b*32 +: 32] = $urandom;
                    st = ($urandom % 2) ? '1 : {$urandom, $urandom};
                    wq_d.push_back(d); wq_s.push_back(st); ds.push_back(d); ss.push_back(st);
                    rows.push_back(row + k); pending[row + k] = pending[row + k] + 1;
                end
                wleft.push_back(len + 1);
                bwait.push_back(rows); bdata.push_back(ds); bstrb.push_back(ss);
                writes = writes + 1; ops = ops + 1;
                end
            end
            // AR: a row with no write in flight
            if ((!s_arvalid || s_arready) && ops < n_ops && ($urandom % 3) == 0) begin
                automatic int row = int'($urandom % (ROWS - 4)), len = int'($urandom % 4);
                automatic bit ok = 1;
                for (int k = 0; k <= len; k++) if (pending[row + k] != 0) ok = 0;
                if (ok) begin
                    automatic reg [IDW-1:0] id = IDW'($urandom);
                    s_arvalid <= 1; s_araddr <= 64'(row) * 64; s_arlen <= 8'(len); s_arid <= id;
                    for (int k = 0; k <= len; k++) begin
                        rexp_row.push_back(row + k); rexp_id.push_back(int'(id)); rexp_last.push_back(k == len);
                        pending[row + k] = pending[row + k] + 1000;   // no write until it is read
                    end
                    reads = reads + 1; ops = ops + 1;
                end
            end
            // W, in AW order
            if (s_wvalid && s_wready) s_wvalid <= 0;
            if ((!s_wvalid || s_wready) && wleft.size() && wq_d.size() && ($urandom % 100) >= stall) begin
                automatic int left = wleft.pop_front() - 1;
                s_wvalid <= 1; s_wdata <= wq_d.pop_front(); s_wstrb <= wq_s.pop_front();
                s_wlast <= left == 0;
                if (left != 0) wleft.push_front(left);
            end
            // B: the oldest write's rows take its data
            if (s_bvalid && s_bready) begin
                for (int k = 0; k < bwait[0].size(); k++) begin
                    automatic int r = bwait[0][k];
                    for (int j = 0; j < 64; j++) if (bstrb[0][k][j]) shadow[r][j*8 +: 8] = bdata[0][k][j*8 +: 8];
                    pending[r] = pending[r] - 1;
                end
                void'(bwait.pop_front()); void'(bdata.pop_front()); void'(bstrb.pop_front());
            end
            // R
            if (s_rvalid && s_rready) begin
                if (rexp_row.size() == 0) begin $display("FAIL: an R beat with no read"); errors++; end
                else begin
                    automatic int r = rexp_row.pop_front(), id = rexp_id.pop_front(), l = rexp_last.pop_front();
                    if (s_rdata !== shadow[r] || int'(s_rid) != id || int'(s_rlast) != l) begin
                        if (errors < 5) $display("FAIL: row %0d id %0d/%0d last %0d/%0d data %s", r, s_rid, id, s_rlast, l,
                                                 (s_rdata === shadow[r]) ? "ok" : "wrong");
                        errors++;
                    end else beats_ok++;
                    pending[r] = pending[r] - 1000;
                end
            end
        end
    end

    initial begin
        for (int r = 0; r < ROWS; r++) begin shadow[r] = '0; pending[r] = 0; end
        for (int r = 0; r < 2 * ROWS; r++) mem[r] = '0;
        repeat (5) @(posedge s_clk);
        @(negedge m_clk) m_reset = 0;
        @(negedge s_clk) s_reset = 0;
        wait (ops >= n_ops);
        // Drain: every write's B and every read beat back (or a timeout).
        for (int c = 0; c < 400000 && (rexp_row.size() || bwait.size()); c++) @(posedge s_clk);
        repeat (100) @(posedge s_clk);
        if (rexp_row.size() || bwait.size()) begin $display("FAIL: %0d read beats, %0d writes outstanding", rexp_row.size(), bwait.size()); errors++; end
        $display("%0d writes, %0d reads, %0d read beats checked, sp %0d mp %0d stall %0d", writes, reads, beats_ok, sp, mp, stall);
        if (errors == 0 && beats_ok > 100) $display("PASS"); else $display("FAILED: %0d", errors);
        $finish;
    end
endmodule
