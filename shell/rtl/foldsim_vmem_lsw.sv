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

`include "foldsim_vmem_s28.vh"

// foldsim_vmem_lsw: the local switch of one class (S28): its N type
// adapters' entries onto one request bus of 619-bit words, one a cycle.
//
// Buffer. A class copy evaluates one instance a host cycle but may access
// all its memories in it (the operand collector: 4 GPR banks), so up to N
// entries arrive in a cycle. Each adapter has its own bank of FIFO entries
// (a memory with one write port: a buffer shared by the N adapters would
// need N write ports, and its room rule could not hold, see below). An
// entry is the adapter's read and write of a host cycle, kept whole, each
// with its own instance (a page column's read and write-back are of
// different {instance, page}); the
// drain cuts it into words: a read is one word, a write ceil(DW/512) beats,
// in RDW_READ_FIRST's order (a read-first pair's read first).
//
// Room. Each adapter's `free` is the native rule of foldsim_vmem_group:
// twice its bank's free entries less one, less one entry more for the
// adapter's register (an entry issued in a host cycle reaches the bank a
// cycle later). The simulator takes free >= 10 (+6 with room-regions) as
// room. It is 0 while the region controller holds the tree (`hold`: a
// reset's drain, a move). [Why not one shared buffer: a class's N memories
// are N vm<k> ports, each deciding room on its own `free`; a shared buffer
// of E free entries could promise each port at most E/N of them, and the
// rule needs 6 per port (9 with room-regions) before any port issues: a
// 16-entry buffer shared by 4 GPR banks never shows room.]
//
// Drain. Round robin over the adapters with an entry, from the one after
// the last granted; a granted adapter keeps the grant until its entry's last
// word, so an entry's words stay together. One word a cycle, registered,
// held while the central switch is not ready.
//
// Responses: the controller's beats for this class, registered, to the
// adapter of their `mem`.
module foldsim_vmem_lsw #(
    parameter N            = 1,
    // This class's number in its region (the words' `cls`).
    parameter CLS          = 0,
    // The adapters' widest instance, row, data and mask bits.
    parameter IW           = 8,
    parameter AW           = 10,
    parameter DW           = 512,
    parameter MW           = DW / 8,
    // Each adapter's row, data and mask bits (MDW/8 byte enables or 1),
    // and its read-write order: adapter i's in bits [i*32 +: 32].
    parameter logic [N*32-1:0] MAW = {N{32'(AW)}},
    parameter logic [N*32-1:0] MDW = {N{32'(DW)}},
    parameter logic [N*32-1:0] MMW = {N{32'(MW)}},
    parameter logic [N*32-1:0] RDW_READ_FIRST = {N{32'd1}},
    // Entries per adapter bank, a power of two of at least 8.
    parameter FIFO         = 16,
    localparam GW          = $clog2(FIFO + 1)
) (
    input  wire                   clk,

    // The adapters' entries, adapter i's in field i of each port.
    input  wire [N-1:0]           e_valid,
    input  wire [N-1:0]           e_rd,
    input  wire [N-1:0]           e_wr,
    input  wire [N*IW-1:0]        e_rinst,
    input  wire [N*IW-1:0]        e_winst,
    input  wire [N*AW-1:0]        e_rrow,
    input  wire [N*AW-1:0]        e_wrow,
    input  wire [N*DW-1:0]        e_wdata,
    input  wire [N*MW-1:0]        e_wmask,
    output reg  [N*16-1:0]        e_free,

    // To the central switch.
    output reg                    o_valid,
    output reg  [`S28_REQ_W-1:0]  o_word,
    input  wire                   o_ready,
    // From it: the region's hold; to it: nothing buffered here.
    input  wire                   hold,
    output reg                    idle,

    // Responses for this class, and to the adapters.
    input  wire                   r_valid,
    input  wire [`S28_RSP_W-1:0]  r_word,
    output reg  [N-1:0]           a_valid,
    output reg  [IW-1:0]          a_inst,
    output reg  [`S28_BEAT_W-1:0] a_beat,
    output reg                    a_last,
    output reg  [511:0]           a_data
);
    localparam PW = $clog2(FIFO);
    // Words of an entry: a read and up to 8 write beats.
    localparam XW = 4;
    function automatic int MAW_(input int i); MAW_ = int'(MAW[i*32 +: 32]); endfunction
    function automatic int MDW_(input int i); MDW_ = int'(MDW[i*32 +: 32]); endfunction
    function automatic int MMW_(input int i); MMW_ = int'(MMW[i*32 +: 32]); endfunction

`ifdef SIMULATION
    initial begin
        if (FIFO < 8 || (FIFO & (FIFO - 1)) != 0)
            $error("foldsim_vmem_lsw: FIFO must be a power of two of at least 8");
        if (N > (1 << `S28_MEM_W) || CLS >= (1 << `S28_CLS_W) || IW > `S28_INST_W || AW > `S28_ROW_W)
            $error("foldsim_vmem_lsw: N %0d / CLS %0d / IW %0d / AW %0d past the word's fields", N, CLS, IW, AW);
        for (integer m = 0; m < N; ++m) begin
            if (MDW_(m) < 1 || MDW_(m) > DW || MAW_(m) < 1 || MAW_(m) > AW)
                $error("foldsim_vmem_lsw: adapter %0d's data or row bits out of range", m);
            if (MMW_(m) != 1 && MMW_(m) * 8 != MDW_(m))
                $error("foldsim_vmem_lsw: adapter %0d's mask must be 1 or MDW/8 byte enables", m);
            if ((MDW_(m) + 7) / 8 * 8 > 8 * 512)
                $error("foldsim_vmem_lsw: adapter %0d's row past 8 beats", m);
        end
    end
`endif

    // ---- The banks ----
    wire [N-1:0]              nonempty;
    wire [N-1:0]              pop;
    wire [`S28_REQ_W-1:0]     word  [N];
    wire                      wlast [N];
    reg  [XW-1:0]             widx;          // the granted entry's next word
    reg  [N-1:0]              g_oh;          // the adapter granted last (one-hot)
    reg                       busy;          // inside an entry of adapter g

    for (genvar m = 0; m < N; ++m) begin : g_bank
        localparam A  = MAW_(m);
        localparam D  = MDW_(m);
        localparam K  = MMW_(m);
        localparam RF = RDW_READ_FIRST[m*32 +: 32];
        wire          ev = e_valid[m];
        wire          erd = e_rd[m];
        wire          ewr = e_wr[m];
        wire [IW-1:0] eri = e_rinst[m*IW +: IW];
        wire [IW-1:0] ewi = e_winst[m*IW +: IW];
        wire [AW-1:0] err = e_rrow[m*AW +: AW];
        wire [AW-1:0] ewr_row = e_wrow[m*AW +: AW];
        wire [DW-1:0] ewd = e_wdata[m*DW +: DW];
        wire [MW-1:0] ewm = e_wmask[m*MW +: MW];
        localparam DB = (D + 7) / 8 * 8;
        localparam B  = (DB + 511) / 512;
        localparam EW = 2 + 2 * IW + 2 * A + D + K;
        reg  [EW-1:0] mem [FIFO];
        reg  [PW-1:0] wp, rp;
        reg  [GW-1:0] cnt;
`ifdef SIMULATION
        initial begin
            wp  = '0;
            rp  = '0;
            cnt = '0;
        end
`endif
        wire [GW-1:0] cnt_next = cnt + GW'(ev) - GW'(pop[m]);
        wire [GW-1:0] cnt_in   = cnt + GW'(ev);
        always @(posedge clk) begin
            if (ev) begin
                mem[wp] <= {erd, ewr, eri, ewi, err[A-1:0], ewr_row[A-1:0], ewd[D-1:0], ewm[K-1:0]};
                wp <= wp + 1'b1;
            end
            if (pop[m])
                rp <= rp + 1'b1;
            cnt <= cnt_next;
            // Twice the free entries, less one, less the adapter's register's
            // (the entries after this cycle's push; its pop not counted: the
            // room shows a cycle late, never early).
            e_free[m*16 +: 16] <= (hold || cnt_in >= GW'(FIFO - 2)) ? '0 : 16'(2 * (FIFO - 2 - 32'(cnt_in)));
        end
        assign nonempty[m] = (cnt != '0);

        // The head entry's word `widx` (0 unless this adapter is granted).
        wire [EW-1:0]  h = mem[rp];
        wire           h_rd, h_wr;
        wire [IW-1:0]  h_rinst, h_winst;
        wire [A-1:0]   h_rrow, h_wrow;
        assign {h_rd, h_wr, h_rinst, h_winst, h_rrow, h_wrow} = h[EW-1 -: 2 + 2 * IW + 2 * A];
        wire [B*512-1:0] h_data = (B*512)'(h[K +: D]);
        wire [B*64-1:0]  h_mask;
        if (K == 1) begin : g_row_en
            assign h_mask = {(B*64){h[0]}};
        end else begin : g_bytes
            assign h_mask = (B*64)'(h[K-1:0]);
        end
        wire [XW-1:0]  x       = (busy && g_oh[m]) ? widx : '0;
        wire [XW-1:0]  nw      = XW'(h_rd) + (h_wr ? XW'(B) : '0);
        wire           is_rd   = h_rd && ((RF != 0) ? (x == '0) : (x == XW'(h_wr ? B : 0)));
        wire [XW-1:0]  wbeat   = ((RF != 0) ? x - XW'(h_rd) : x);
        wire [2:0]     beat    = is_rd ? 3'd0 : wbeat[2:0];
        assign wlast[m] = (x == nw - 1'b1);
        if (B > 1) begin : g_slice
            assign word[m] = {`S28_CLS_W'(CLS), `S28_MEM_W'(m), `S28_INST_W'(is_rd ? h_rinst : h_winst),
                              `S28_ROW_W'(is_rd ? h_rrow : h_wrow), beat, wlast[m], ~is_rd,
                              h_mask[32'(beat) * 64 +: 64], h_data[32'(beat) * 512 +: 512]};
        end else begin : g_one
            assign word[m] = {`S28_CLS_W'(CLS), `S28_MEM_W'(m), `S28_INST_W'(is_rd ? h_rinst : h_winst),
                              `S28_ROW_W'(is_rd ? h_rrow : h_wrow), beat, wlast[m], ~is_rd,
                              h_mask, h_data};
        end
        /* verilator lint_off UNUSEDSIGNAL */
        wire unused = (|wbeat) | (|ewd) | (|err) | (|ewr_row) | (|ewm);
        /* verilator lint_on UNUSEDSIGNAL */

`ifdef SIMULATION
        always @(posedge clk) begin
            if (ev && cnt == GW'(FIFO) && !pop[m])
                $error("foldsim_vmem_lsw: adapter %0d's bank overflows", m);
            if (ev && !erd && !ewr)
                $error("foldsim_vmem_lsw: adapter %0d's empty entry", m);
        end
`endif
    end

    // ---- Grant (one-hot): the entry in progress, else round robin over
    // the adapters with an entry from the one after the last granted
    // (after_q, a register: the adapters above it) ----
    reg  [N-1:0] after_q;
    function automatic [N-1:0] lowest(input [N-1:0] v);
        for (integer i = 0; i < N; ++i)
            lowest[i] = v[i] & ~(|(v & N'((N'(1) << i) - 1'b1)));
    endfunction
    function automatic [N-1:0] above(input [N-1:0] v);
        for (integer i = 0; i < N; ++i)
            above[i] = |(v & N'((N'(1) << i) - 1'b1));
    endfunction
    wire [N-1:0] hi     = nonempty & after_q;
    wire [N-1:0] pick   = (|hi) ? lowest(hi) : lowest(nonempty);
    wire [N-1:0] sel_oh = busy ? g_oh : pick;
    wire         go     = (!o_valid || o_ready) && (busy || (|nonempty));
    reg  [`S28_REQ_W-1:0] sel_word;
    reg                   sel_last;
    always @* begin
        sel_word = '0;
        sel_last = 1'b0;
        for (integer m = 0; m < N; ++m) begin
            sel_word = sel_word | ({`S28_REQ_W{sel_oh[m]}} & word[m]);
            sel_last = sel_last | (sel_oh[m] & wlast[m]);
        end
    end
    for (genvar m = 0; m < N; ++m) begin : g_pop
        assign pop[m] = go && sel_last && sel_oh[m];
    end

`ifdef SIMULATION
    initial begin
        o_valid = 1'b0;
        busy    = 1'b0;
        g_oh    = '0;
        after_q = '0;
        widx    = '0;
        idle    = 1'b1;
        a_valid = '0;
    end
`endif
    always @(posedge clk) begin
        if (go) begin
            o_word  <= sel_word;
            g_oh    <= sel_oh;
            after_q <= above(sel_oh);
            busy    <= !sel_last;
            widx    <= sel_last ? '0 : ((busy ? widx : '0) + 1'b1);
        end
        if (go)
            o_valid <= 1'b1;
        else if (o_ready)
            o_valid <= 1'b0;
    end

    // Nothing here or in the adapters' registers.
    always @(posedge clk)
        idle <= !(|nonempty) && !o_valid && !(|e_valid) && !busy;

    // ---- Responses ----
    always @(posedge clk) begin
        for (integer m = 0; m < N; ++m)
            a_valid[m] <= r_valid && r_word[`S28_R_MEM] == `S28_MEM_W'(m);
        a_inst <= IW'(r_word[`S28_R_INST]);
        a_beat <= r_word[`S28_R_BEAT];
        a_last <= r_word[`S28_R_LAST];
        a_data <= r_word[`S28_R_DATA];
    end
    /* verilator lint_off UNUSEDSIGNAL */
    wire [`S28_RSP_W-1:0] unused_r = r_word;
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
