// tb_vmem_share.sv: foldsim_vmem_share (N controllers on one AXI master) under random traffic. Each controller
// issues single-beat reads and writes to addresses of its own ({k, sequence}), holding each request until taken;
// the master takes AR/AW/W at random, answers reads in order with the address as data and writes with a B. Checked:
// every read answer reaches its controller with its own address, in order per id; every W carries its AW's address;
// every B reaches the controller that wrote; nothing is lost (all answered at the end). Printed: the grant trace's
// checksum (cycle and controller of each AR/AW take), equal for two arbiters that grant alike cycle by cycle.
`timescale 1ns/1ps
module tb;
    parameter N = 21, ID_W = 1, CYCLES = 20000;
    localparam NID = 1 << ID_W;
    reg clk = 0;
    always #1 clk = ~clk;
    reg               s_awvalid [N], s_wvalid [N], s_wlast [N], s_bready [N], s_arvalid [N], s_rready [N];
    reg  [63:0]       s_awaddr [N], s_araddr [N], s_wstrb [N];
    reg  [ID_W-1:0]   s_awid [N], s_arid [N];
    reg  [7:0]        s_awlen [N], s_arlen [N];
    reg  [511:0]      s_wdata [N];
    wire              s_awready [N], s_wready [N], s_bvalid [N], s_arready [N], s_rvalid [N], s_rlast [N];
    wire [ID_W-1:0]   s_bid [N], s_rid [N];
    wire [511:0]      s_rdata [N];
    wire              m_awvalid, m_wvalid, m_wlast, m_bready, m_arvalid, m_rready;
    wire [63:0]       m_awaddr, m_araddr, m_wstrb;
    wire [ID_W-1:0]   m_awid, m_arid;
    wire [7:0]        m_awlen, m_arlen;
    wire [511:0]      m_wdata;
    reg               m_awready = 0, m_wready = 0, m_bvalid = 0, m_arready = 0, m_rvalid = 0, m_rlast = 1;
    reg  [ID_W-1:0]   m_bid = 0, m_rid = 0;
    reg  [511:0]      m_rdata = 0;
    foldsim_vmem_share #(.N(N), .ID_W(ID_W), .RDEPTH(N * 4), .WDEPTH(N * 4)) dut (.clk(clk),
        .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr), .s_awid(s_awid), .s_awlen(s_awlen),
        .s_wvalid(s_wvalid), .s_wready(s_wready), .s_wdata(s_wdata), .s_wstrb(s_wstrb), .s_wlast(s_wlast),
        .s_bvalid(s_bvalid), .s_bready(s_bready), .s_bid(s_bid), .s_arvalid(s_arvalid), .s_arready(s_arready),
        .s_araddr(s_araddr), .s_arid(s_arid), .s_arlen(s_arlen), .s_rvalid(s_rvalid), .s_rready(s_rready),
        .s_rdata(s_rdata), .s_rid(s_rid), .s_rlast(s_rlast),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awid(m_awid), .m_awlen(m_awlen),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bid(m_bid), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_araddr(m_araddr), .m_arid(m_arid), .m_arlen(m_arlen), .m_rvalid(m_rvalid), .m_rready(m_rready),
        .m_rdata(m_rdata), .m_rid(m_rid), .m_rlast(m_rlast));
    integer issue = 40, mready = 60, rgo = 50, errors = 0, cyc = 0;
    longint trace = 0, nr = 0, nw = 0, nrd = 0, nb = 0;
    int unsigned rseq [N], wseq [N], wpend [N];
    longint rq [N][NID][$];           // each controller's reads in flight, per id
    longint mq [$]; reg [ID_W-1:0] mqid [$];   // the master's reads, in AR order
    longint awq [$]; reg [ID_W-1:0] awid_q [$]; // AWs taken, waiting for their W
    longint wq [$];                             // W's taken, waiting for their AW
    int bown [$]; reg [ID_W-1:0] bq_id [$];     // B's to send: the writer
    initial begin
        void'($value$plusargs("issue=%d", issue));
        void'($value$plusargs("mready=%d", mready));
        for (int k = 0; k < N; k++) begin
            s_awvalid[k] = 0; s_wvalid[k] = 0; s_wlast[k] = 1; s_bready[k] = 1; s_arvalid[k] = 0; s_rready[k] = 1;
            s_awlen[k] = 0; s_arlen[k] = 0; s_wstrb[k] = '1; rseq[k] = 0; wseq[k] = 0; wpend[k] = 0;
            s_awaddr[k] = 0; s_araddr[k] = 0; s_awid[k] = 0; s_arid[k] = 0; s_wdata[k] = 0;
        end
    end
    always @(posedge clk) begin
        cyc <= cyc + 1;
        // Master side.
        if (m_arvalid && m_arready) begin
            mq.push_back(m_araddr); mqid.push_back(m_arid);
            trace = trace * 31 + cyc * 64 + longint'(m_araddr >> 32); nr++;
        end
        if (m_awvalid && m_awready) begin
            awq.push_back(m_awaddr); awid_q.push_back(m_awid);
            trace = trace * 37 + cyc * 64 + longint'(m_awaddr >> 32); nw++;
        end
        // (W may come before its AW is taken, AXI allows it: W's and AW's matched in order.)
        if (m_wvalid && m_wready) begin
            wq.push_back(m_wdata[63:0]);
            if (!m_wlast) begin errors++; if (errors < 5) $display("FAIL: a W without wlast"); end
        end
        while (wq.size() != 0 && awq.size() != 0) begin
            if (wq[0] != awq[0]) begin errors++; if (errors < 5) $display("FAIL: W %h for AW %h", wq[0], awq[0]); end
            bown.push_back(int'(awq[0] >> 32)); bq_id.push_back(awid_q[0]);
            void'(wq.pop_front()); void'(awq.pop_front()); void'(awid_q.pop_front());
        end
        if (m_rvalid && m_rready) m_rvalid <= 0;
        if ((!m_rvalid || m_rready) && mq.size() != 0 && ($urandom % 100) < rgo) begin
            m_rvalid <= 1; m_rdata <= 512'(mq[0]); m_rid <= mqid[0]; m_rlast <= 1;
            void'(mq.pop_front()); void'(mqid.pop_front());
        end
        if (m_bvalid && m_bready) m_bvalid <= 0;
        if ((!m_bvalid || m_bready) && bq_id.size() != 0 && ($urandom % 100) < rgo) begin
            m_bvalid <= 1; m_bid <= bq_id[0]; void'(bq_id.pop_front()); void'(bown.pop_front());
        end
        m_arready <= ($urandom % 100) < mready; m_awready <= ($urandom % 100) < mready; m_wready <= ($urandom % 100) < mready;
        // Controllers.
        for (int k = 0; k < N; k++) begin
            if (s_rvalid[k] && s_rready[k]) begin
                if (rq[k][s_rid[k]].size() == 0 || s_rdata[k][63:0] != rq[k][s_rid[k]][0]) begin
                    errors++; if (errors < 5) $display("FAIL: controller %0d got read data %h (id %0d)", k, s_rdata[k][63:0], s_rid[k]);
                end else void'(rq[k][s_rid[k]].pop_front());
                nrd++;
            end
            if (s_bvalid[k] && s_bready[k]) begin
                if (wpend[k] == 0) begin errors++; if (errors < 5) $display("FAIL: controller %0d got a B it did not wait for", k); end
                else wpend[k]--;
                nb++;
            end
            if (s_arvalid[k] && s_arready[k]) begin
                rq[k][s_arid[k]].push_back(s_araddr[k]); s_arvalid[k] <= 0;
            end else if (!s_arvalid[k] && cyc < CYCLES && ($urandom % 100) < issue) begin
                reg [ID_W-1:0] id; id = ID_W'($urandom);
                s_arvalid[k] <= 1; s_arid[k] <= id; s_araddr[k] <= {32'(k), 32'(rseq[k])}; rseq[k] <= rseq[k] + 1;
            end
            if (s_awvalid[k] && s_awready[k]) begin
                s_awvalid[k] <= 0; wpend[k]++;
            end else if (!s_awvalid[k] && !s_wvalid[k] && cyc < CYCLES && ($urandom % 100) < issue) begin
                s_awvalid[k] <= 1; s_awid[k] <= ID_W'($urandom); s_awaddr[k] <= {32'(k), 32'(wseq[k] | 32'h8000_0000)};
                s_wvalid[k] <= 1; s_wdata[k] <= 512'({32'(k), 32'(wseq[k] | 32'h8000_0000)}); wseq[k] <= wseq[k] + 1;
            end
            if (s_wvalid[k] && s_wready[k]) s_wvalid[k] <= 0;
        end
        if (cyc == CYCLES + 4000) begin
            for (int k = 0; k < N; k++) begin
                if (wpend[k] != 0 || s_awvalid[k] || s_wvalid[k] || s_arvalid[k]) begin errors++; if (errors < 5) $display("FAIL: controller %0d left %0d writes", k, wpend[k]); end
                for (int i = 0; i < NID; i++) if (rq[k][i].size() != 0) begin errors++; if (errors < 5) $display("FAIL: controller %0d left reads", k); end
            end
            $display("reads %0d writes %0d answered %0d/%0d trace %h", nr, nw, nrd, nb, trace);
            if (errors == 0 && nr > 1000 && nw > 1000) $display("PASS"); else $display("FAIL: %0d errors", errors);
            $finish;
        end
    end
endmodule
