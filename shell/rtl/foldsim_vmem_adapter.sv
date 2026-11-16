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

// foldsim_vmem_adapter: the type adapter of one virtual memory of one class
// (S28). It keeps the simulator's native vm<k> port (VMEM_INTERFACE.md) and
// does no address arithmetic and no AXI:
//  * request: a host cycle's read and write of the memory leave registered,
//    as one entry, to the class's local switch, each with its own instance
//    (a page column reads the next {instance, page} while it writes the
//    previous one back); the local switch keeps the entries, cuts them into
//    512-bit beats and reports the room (`free`, passed through here);
//  * response: the region controller's beats of a read, in order and back to
//    back, put together into the row; `rsp_valid` with the last one. Nothing
//    is answered while `reset` is high.
// Lanes (LANES = K > 1, a power of two): K instances share one row of K lanes
// of ceil(DW/8) bytes each, instance i in lane i mod K of row i / K (the
// region addresses the row by inst / K). A write goes to its lane only: its
// data shifted there, its byte enables that lane's (a whole-row mask, MW 1,
// enables the lane's bytes), never a read-modify-write; an answer gives the
// lane of its instance. Two lanes of a row share its address, so the region
// orders them as writes of one row (each waits for the other's B).
module foldsim_vmem_adapter #(
    parameter IW    = 8,
    parameter AW    = 10,
    parameter DW    = 512,
    parameter MW    = DW / 8,
    parameter LANES = 1,
    // The words' memory, instance and row fields (foldsim_vmem_s28.vh: per design;
    // the memory field is the local switch's).
    /* verilator lint_off UNUSEDPARAM */
    parameter MEM_W  = 4,
    /* verilator lint_on UNUSEDPARAM */
    parameter INST_W = 16,
    parameter ROW_W  = 12,
    localparam DB   = (DW + 7) / 8 * 8,
    localparam RDW  = (LANES > 1) ? LANES * DB : DW,       // the row's data bits
    localparam RMW  = (LANES > 1) ? LANES * DB / 8 : MW,   // the row's mask bits
    localparam BEATS = ((RDW + 7) / 8 * 8 + 511) / 512
) (
    input  wire                   clk,
    input  wire                   reset,

    // The simulator's port (the controller side of vm<k>).
    input  wire                   rd_valid,
    input  wire [IW-1:0]          rd_inst,
    input  wire [AW-1:0]          rd_row,
    input  wire                   wr_valid,
    input  wire [IW-1:0]          wr_inst,
    input  wire [AW-1:0]          wr_row,
    input  wire [DW-1:0]          wr_data,
    input  wire [MW-1:0]          wr_mask,
    output reg                    rsp_valid,
    output reg  [IW-1:0]          rsp_inst,
    output wire [DW-1:0]          rsp_data,
    output wire [15:0]            free,

    // To the local switch: the entry, registered; its room for this memory.
    output reg                    e_valid,
    output reg                    e_rd,
    output reg                    e_wr,
    output reg  [IW-1:0]          e_rinst,
    output reg  [IW-1:0]          e_winst,
    output reg  [AW-1:0]          e_rrow,
    output reg  [AW-1:0]          e_wrow,
    output reg  [RDW-1:0]         e_wdata,
    output reg  [RMW-1:0]         e_wmask,
    input  wire [15:0]            e_free,

    // From the local switch: this memory's response beats.
    input  wire                   r_valid,
    input  wire [IW-1:0]          r_inst,
    input  wire [`S28_BEAT_W-1:0] r_beat,
    input  wire                   r_last,
    input  wire [511:0]           r_data
);
    localparam LW = (LANES > 1) ? $clog2(LANES) : 1;
`ifdef SIMULATION
    initial begin
        if (IW > `S28_INST_W || AW > `S28_ROW_W || BEATS > (1 << `S28_BEAT_W))
            $error("foldsim_vmem_adapter: IW %0d / AW %0d / %0d beats past the word's fields", IW, AW, BEATS);
        if (MW != 1 && MW * 8 != DW)
            $error("foldsim_vmem_adapter: MW must be 1 or DW/8");
        if (LANES < 1 || (LANES & (LANES - 1)) != 0 || (LANES > 1 && $clog2(LANES) > IW))
            $error("foldsim_vmem_adapter: LANES must be a power of two within the instance bits");
        e_valid   = 1'b0;
        rsp_valid = 1'b0;
    end
`endif

    // The write's lane: its data and byte enables moved there.
    wire [RDW-1:0] lane_data;
    wire [RMW-1:0] lane_mask;
    if (LANES > 1) begin : g_lanes
        wire [LW-1:0]   wl = wr_inst[LW-1:0];
        wire [DB/8-1:0] bm = (MW == 1) ? {(DB/8){wr_mask[0]}} : (DB/8)'(wr_mask);
        assign lane_data = RDW'(DB'(wr_data)) << (32'(wl) * DB);
        assign lane_mask = RMW'(bm) << (32'(wl) * (DB / 8));
    end else begin : g_row
        assign lane_data = wr_data;
        assign lane_mask = wr_mask;
    end
    always @(posedge clk) begin
        e_valid <= rd_valid | wr_valid;
        e_rd    <= rd_valid;
        e_wr    <= wr_valid;
        e_rinst <= rd_inst;
        e_winst <= wr_inst;
        e_rrow  <= rd_row;
        e_wrow  <= wr_row;
        e_wdata <= lane_data;
        e_wmask <= lane_mask;
    end
    assign free = e_free;

    // The row, beat by beat; the answer, its instance's lane.
    /* verilator lint_off UNUSEDSIGNAL */
    reg [BEATS*512-1:0] row_q;
    /* verilator lint_on UNUSEDSIGNAL */
    if (BEATS > 1) begin : g_beats
        always @(posedge clk)
            if (r_valid)
                row_q[32'(r_beat) * 512 +: 512] <= r_data;
    end else begin : g_beat
        /* verilator lint_off UNUSEDSIGNAL */
        wire [`S28_BEAT_W-1:0] unused_beat = r_beat;
        /* verilator lint_on UNUSEDSIGNAL */
        always @(posedge clk)
            if (r_valid)
                row_q <= r_data;
    end
    always @(posedge clk) begin
        rsp_valid <= r_valid & r_last & ~reset;
        rsp_inst  <= r_inst;
    end
    if (LANES > 1) begin : g_rsp_lane
        assign rsp_data = DW'(row_q >> (32'(rsp_inst[LW-1:0]) * DB));
    end else begin : g_rsp_row
        assign rsp_data = row_q[DW-1:0];
    end
endmodule
