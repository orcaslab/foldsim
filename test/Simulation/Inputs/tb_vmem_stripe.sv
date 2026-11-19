// tb_vmem_stripe.sv: foldsim_vmem_stripe against S AXI slave models (random ready, random latency, answers in any
// order across masters) and a flat reference image. Phase 1 writes every row (bursts of 1-4 beats, random ids) and
// waits for each B; phase 2 reads them back in random order with up to 16 in flight; phase 3 writes the upper half
// while reading the lower half. Checks: every R beat's data and id against the issue order (the stripe answers in
// request order), B count and ids, each slave's rows packed (address within its rows' span). Prints PASS or FAIL.
`ifndef S
`define S 4
`endif
module tb;
    localparam S = `S, ID_W = 4, SHIFT = 8, ROWS = 256, BEATS = 4;
    localparam [63:0] RB = 64'h40_0000_0000;
    localparam SB = S > 1 ? $clog2(S) : 1;
    function automatic [64*S-1:0] bases(); for (int j = 0; j < S; j++) bases[64*j +: 64] = 64'h1000_0000 * (j + 1); endfunction  // multiples of 2^16 / S
    reg clk = 0; always #5 clk = ~clk;
    int seed = 1, stall = 30, maxlat = 40;
    logic s_awvalid = 0, s_wvalid = 0, s_arvalid = 0, s_bready = 1, s_rready = 1;
    logic [63:0] s_awaddr, s_araddr; logic [ID_W-1:0] s_awid, s_arid; logic [7:0] s_awlen, s_arlen;
    logic [511:0] s_wdata; logic [63:0] s_wstrb = '1; logic s_wlast;
    wire s_awready, s_wready, s_arready, s_bvalid, s_rvalid, s_rlast; wire [ID_W-1:0] s_bid, s_rid; wire [511:0] s_rdata;
    wire m_awvalid [S], m_wvalid [S], m_arvalid [S], m_bready [S], m_rready [S], m_wlast [S];
    wire [63:0] m_awaddr [S], m_araddr [S], m_wstrb [S]; wire [ID_W-1:0] m_awid [S], m_arid [S];
    wire [7:0] m_awlen [S], m_arlen [S]; wire [511:0] m_wdata [S];
    logic m_awready [S], m_wready [S], m_arready [S], m_bvalid [S], m_rvalid [S], m_rlast [S];
    logic [ID_W-1:0] m_bid [S], m_rid [S]; logic [511:0] m_rdata [S];
    foldsim_vmem_stripe #(.S(S), .ID_W(ID_W), .SHIFT(SHIFT), .RDEPTH(32), .WDEPTH(32), .RW(16),
                          .PORT_BASE(bases())) dut(.*);
    // Slaves (axi_slave below): memories by their own address; answers after a random latency.
    bit fail = 0; int cyc = 0;
    always @(posedge clk) begin cyc <= cyc + 1; if (cyc > 3000000) begin $display("FAIL: timeout"); $finish; end end
    function automatic void err(string m); if (!fail) $display("FAIL: %s at %0d", m, cyc); fail = 1; endfunction
    bit sfail [S];
    for (genvar j = 0; j < S; j++) begin : g_slave
        axi_slave #(.ID_W(ID_W), .LO(64'h1000_0000 * (j + 1)), .HI(64'h1000_0000 * (j + 1) + (ROWS / S) * (1 << SHIFT))) sl(
            .clk, .cyc, .stall, .maxlat,
            .awvalid(m_awvalid[j]), .awready(m_awready[j]), .awaddr(m_awaddr[j]), .awid(m_awid[j]), .awlen(m_awlen[j]),
            .wvalid(m_wvalid[j]), .wready(m_wready[j]), .wdata(m_wdata[j]), .wlast(m_wlast[j]),
            .bvalid(m_bvalid[j]), .bready(m_bready[j]), .bid(m_bid[j]),
            .arvalid(m_arvalid[j]), .arready(m_arready[j]), .araddr(m_araddr[j]), .arid(m_arid[j]), .arlen(m_arlen[j]),
            .rvalid(m_rvalid[j]), .rready(m_rready[j]), .rdata(m_rdata[j]), .rid(m_rid[j]), .rlast(m_rlast[j]), .fail(sfail[j]));
    end
    always @(posedge clk) for (int j = 0; j < S; j++) if (sfail[j]) err($sformatf("slave %0d", j));
    // Driver and reference.
    logic [511:0] ref_img [longint];
    typedef struct { logic [ID_W-1:0] id; logic [63:0] a; int beats; } exp_t;
    exp_t exp_r [$]; logic [ID_W-1:0] exp_b [$];
    int rb = 0, nr = 0, nw = 0, nb = 0;
    // Handshakes sampled at the edge (before the DUT's registers move), read by the driver a moment after it.
    bit aw_hs, w_hs, ar_hs;
    always @(posedge clk) begin aw_hs <= s_awvalid && s_awready; w_hs <= s_wvalid && s_wready; ar_hs <= s_arvalid && s_arready; end
    always @(posedge clk) begin
        if (s_rvalid && s_rready) begin
            if (!exp_r.size()) err("R with no read");
            else begin
                if (s_rid !== exp_r[0].id) err($sformatf("R id %0d, expected %0d", s_rid, exp_r[0].id));
                if (s_rdata !== ref_img[exp_r[0].a + 64 * rb]) err($sformatf("R data at %h beat %0d", exp_r[0].a, rb));
                if (s_rlast !== (rb == exp_r[0].beats - 1)) err("RLAST off");
                rb++; if (rb == exp_r[0].beats) begin rb = 0; void'(exp_r.pop_front()); nr++; end
            end
        end
        if (s_bvalid && s_bready) begin
            if (!exp_b.size()) err("B with no write"); else begin automatic logic [ID_W-1:0] e = exp_b.pop_front(); if (s_bid !== e) err($sformatf("B id %0d, expected %0d (B %0d)", s_bid, e, nb)); end
            nb++;
        end
    end
    always @(negedge clk) begin s_rready = $urandom % 100 >= stall; s_bready = $urandom % 100 >= stall; end
    task automatic write_row(int row, int beats);
        logic [ID_W-1:0] id = ID_W'($urandom);
        @(negedge clk); s_awvalid = 1; s_awaddr = RB + (64'(row) << SHIFT); s_awid = id; s_awlen = 8'(beats - 1);
        exp_b.push_back(id);
        do begin @(posedge clk); #1; end while (!aw_hs); @(negedge clk); s_awvalid = 0;
        for (int x = 0; x < beats; x++) begin
            logic [511:0] d = {16{$urandom}};
            s_wvalid = 1; s_wdata = d; s_wlast = x == beats - 1;
            ref_img[RB + (64'(row) << SHIFT) + 64 * x] = d;
            do begin @(posedge clk); #1; end while (!w_hs); @(negedge clk);
        end
        s_wvalid = 0; nw++;
    endtask
    task automatic read_row(int row, int beats);
        exp_t e; e.id = ID_W'($urandom); e.a = RB + (64'(row) << SHIFT); e.beats = beats;
        @(negedge clk); s_arvalid = 1; s_araddr = e.a; s_arid = e.id; s_arlen = 8'(beats - 1);
        exp_r.push_back(e);
        do begin @(posedge clk); #1; end while (!ar_hs); @(negedge clk); s_arvalid = 0;
    endtask
    initial begin
        void'($value$plusargs("seed=%d", seed)); void'($value$plusargs("stall=%d", stall)); void'($value$plusargs("lat=%d", maxlat));
        void'($urandom(seed));
        for (int r = 0; r < ROWS; r++) write_row(r, BEATS);
        while (exp_b.size()) @(posedge clk);
        for (int k = 0; k < 4 * ROWS; k++) begin read_row($urandom % ROWS, 1 + $urandom % BEATS); while (exp_r.size() > 16) @(posedge clk); end
        while (exp_r.size()) @(posedge clk);
        fork
            for (int k = 0; k < ROWS; k++) write_row(ROWS / 2 + $urandom % (ROWS / 2), 1 + $urandom % BEATS);
            for (int k = 0; k < 2 * ROWS; k++) begin read_row($urandom % (ROWS / 2), 1 + $urandom % BEATS); while (exp_r.size() > 16) @(posedge clk); end
        join
        while (exp_r.size() || exp_b.size()) begin @(posedge clk); if (cyc > 2000000) begin err("timeout"); break; end end
        if (nb != nw) err("B count");
        $display("%s: S %0d, %0d reads, %0d writes", fail ? "FAIL" : "PASS", S, nr, nw);
        $finish;
    end
endmodule

module axi_slave #(parameter ID_W = 4, parameter [63:0] LO = 0, HI = 0) (
    input logic clk, input int cyc, stall, maxlat,
    input logic awvalid, output logic awready, input logic [63:0] awaddr, input logic [ID_W-1:0] awid, input logic [7:0] awlen,
    input logic wvalid, output logic wready, input logic [511:0] wdata, input logic wlast,
    output logic bvalid, input logic bready, output logic [ID_W-1:0] bid,
    input logic arvalid, output logic arready, input logic [63:0] araddr, input logic [ID_W-1:0] arid, input logic [7:0] arlen,
    output logic rvalid, input logic rready, output logic [511:0] rdata, output logic [ID_W-1:0] rid, output logic rlast,
    output bit fail);
    logic [511:0] mem [longint];
    int r_due [$], r_beats [$], b_due [$], aw_beats [$], w_last [$];
    logic [ID_W-1:0] r_id [$], b_id [$], aw_id [$];
    logic [63:0] r_a [$], aw_a [$];
    logic [511:0] w_d [$];
    int rbeat = 0;
    initial fail = 0;
    always @(negedge clk) begin
        arready = $urandom % 100 >= stall; awready = $urandom % 100 >= stall; wready = $urandom % 100 >= stall;
        rvalid = 0; bvalid = 0;
        if (r_due.size() && r_due[0] <= cyc && $urandom % 100 >= stall) begin
            rvalid = 1; rid = r_id[0]; rlast = rbeat == r_beats[0] - 1;
            rdata = mem.exists(r_a[0] + 64 * rbeat) ? mem[r_a[0] + 64 * rbeat] : '0;
        end
        if (b_due.size() && b_due[0] <= cyc && $urandom % 100 >= stall) begin bvalid = 1; bid = b_id[0]; end
    end
    always @(posedge clk) begin
        if (arvalid && arready) begin
            if (araddr < LO || araddr >= HI) fail = 1;
            r_due.push_back(cyc + 1 + $urandom % maxlat); r_id.push_back(arid); r_a.push_back(araddr); r_beats.push_back(arlen + 1);
        end
        if (rvalid && rready) begin
            rbeat++;
            if (rbeat == r_beats[0]) begin rbeat = 0; void'(r_due.pop_front()); void'(r_id.pop_front()); void'(r_a.pop_front()); void'(r_beats.pop_front()); end
        end
        if (awvalid && awready) begin
            if (awaddr < LO || awaddr >= HI) fail = 1;
            aw_id.push_back(awid); aw_a.push_back(awaddr); aw_beats.push_back(awlen + 1);
        end
        if (wvalid && wready) begin w_d.push_back(wdata); w_last.push_back(wlast); end
        // (W may come before its AW, as AXI allows: beats wait for their AW.)
        while (aw_id.size() && w_d.size()) begin
            mem[aw_a[0]] = w_d.pop_front(); aw_a[0] += 64; aw_beats[0]--;
            if ((aw_beats[0] == 0) != (w_last.pop_front() != 0)) fail = 1;
            if (aw_beats[0] == 0) begin
                b_due.push_back(cyc + 1 + $urandom % maxlat); b_id.push_back(aw_id.pop_front());
                void'(aw_a.pop_front()); void'(aw_beats.pop_front());
            end
        end
        if (bvalid && bready) begin void'(b_due.pop_front()); void'(b_id.pop_front()); end
    end
endmodule
