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

// foldsim_vmem_csw: a central switch (S28): K request buses (the local
// switches of one domain subtree, or the central switches of one region at
// the region controller's input) onto one, inside one SLR.
//  * Each input lands in a 2-entry skid buffer; its `ready` is that buffer's
//    room, a register (no ready path crosses the switch).
//  * Round robin over the inputs with a word, from the one after the last
//    granted; a granted input keeps the grant until a word's `last` (a local
//    switch entry stays together). One word a cycle, registered.
//  * Responses: registered, to the input whose classes hold the word's `cls`
//    (ICLSM[k]: bit c set if input k carries class c).
//  * `hold` goes down registered; `idle` comes up registered: every input
//    idle, no word here.
module foldsim_vmem_csw #(
    parameter K                 = 2,
    // The words' memory, instance and row fields (foldsim_vmem_s28.vh: per design).
    parameter MEM_W  = 4,
    parameter INST_W = 16,
    parameter ROW_W  = 12,
    // Input k's classes in bits [k*64 +: 64].
    parameter logic [K*64-1:0] ICLSM = '1
) (
    input  wire                   clk,

    input  wire [K-1:0]            i_valid,
    input  wire [K*`S28_REQ_W-1:0] i_word,
    output wire [K-1:0]            i_ready,
    input  wire [K-1:0]            i_idle,
    output reg                    i_hold,

    output reg                    o_valid,
    output reg  [`S28_REQ_W-1:0]  o_word,
    input  wire                   o_ready,
    input  wire                   o_hold,
    output reg                    o_idle,

    input  wire                   r_valid,
    input  wire [`S28_RSP_W-1:0]  r_word,
    output reg  [K-1:0]           ri_valid,
    output reg  [`S28_RSP_W-1:0]  ri_word
);

    // ---- Skid buffers ----
    wire [K-1:0]           nonempty, pop;
    wire [`S28_REQ_W-1:0]  head [K];
    wire                   head_last [K];
    for (genvar k = 0; k < K; ++k) begin : g_skid
        reg  [`S28_REQ_W-1:0] s0, s1;   // s0 the head
        reg  [1:0]            n;
`ifdef SIMULATION
        initial n = '0;
`endif
        assign i_ready[k]  = (n != 2'd2);
        assign nonempty[k] = (n != 2'd0);
        assign head[k]      = s0;
        assign head_last[k] = s0[`S28_Q_LAST];
        wire push = i_valid[k] && (n != 2'd2);
        wire [`S28_REQ_W-1:0] iw = i_word[k*`S28_REQ_W +: `S28_REQ_W];
        always @(posedge clk) begin
            case ({push, pop[k]})
            2'b10: begin
                if (n == 2'd0) s0 <= iw; else s1 <= iw;
                n <= n + 1'b1;
            end
            2'b01: begin
                s0 <= s1;
                n  <= n - 1'b1;
            end
            2'b11: begin
                if (n == 2'd1) s0 <= iw; else begin s0 <= s1; s1 <= iw; end
            end
            default: ;
            endcase
        end
    end

    // ---- Grant (one-hot): the input in progress, else round robin over the
    // inputs with a word from the one after the last granted (after_q, a
    // register: the inputs above it) ----
    reg  [K-1:0] g_oh, after_q;
    reg          busy;
    function automatic [K-1:0] lowest(input [K-1:0] v);
        for (integer i = 0; i < K; ++i)
            lowest[i] = v[i] & ~(|(v & K'((K'(1) << i) - 1'b1)));
    endfunction
    function automatic [K-1:0] above(input [K-1:0] v);
        for (integer i = 0; i < K; ++i)
            above[i] = |(v & K'((K'(1) << i) - 1'b1));
    endfunction
    wire [K-1:0] hi     = nonempty & after_q;
    wire [K-1:0] pick   = (|hi) ? lowest(hi) : lowest(nonempty);
    wire [K-1:0] sel_oh = busy ? g_oh : pick;
    wire         go     = (!o_valid || o_ready) && (busy ? |(g_oh & nonempty) : |nonempty);
    assign pop = {K{go}} & sel_oh;
    reg  [`S28_REQ_W-1:0] word;
    reg                   word_last;
    always @* begin
        word = '0;
        word_last = 1'b0;
        for (integer k = 0; k < K; ++k) begin
            word      = word | ({`S28_REQ_W{sel_oh[k]}} & head[k]);
            word_last = word_last | (sel_oh[k] & head_last[k]);
        end
    end

`ifdef SIMULATION
    initial begin
        o_valid = 1'b0;
        busy    = 1'b0;
        g_oh    = '0;
        after_q = '0;
        o_idle  = 1'b1;
        i_hold  = 1'b1;
        ri_valid = '0;
    end
`endif
    always @(posedge clk) begin
        if (go) begin
            o_word  <= word;
            g_oh    <= sel_oh;
            after_q <= above(sel_oh);
            busy    <= !word_last;
        end
        if (go)
            o_valid <= 1'b1;
        else if (o_ready)
            o_valid <= 1'b0;
    end

    // ---- Hold down, idle up ----
    always @(posedge clk) begin
        i_hold <= o_hold;
        o_idle <= (&i_idle) && !(|nonempty) && !o_valid && !busy;
    end

    // ---- Responses ----
    always @(posedge clk) begin
        for (integer k = 0; k < K; ++k)
            ri_valid[k] <= r_valid && ICLSM[k*64 + 32'(r_word[`S28_R_CLS])];
        ri_word <= r_word;
    end
endmodule
