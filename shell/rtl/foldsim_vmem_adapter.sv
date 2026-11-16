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
//  * request: a host cycle's read and write of the memory (one instance's)
//    leave registered, as one entry, to the class's local switch, which keeps
//    the entries, cuts them into 512-bit beats and reports the room (`free`,
//    passed through here);
//  * response: the region controller's beats of a read, in order and back to
//    back, put together into the row; `rsp_valid` with the last one. Nothing
//    is answered while `reset` is high (the controller answers nothing from
//    before a reset after it either).
// The simulator keeps its per-instance pending bits; nothing here needs them.
module foldsim_vmem_adapter #(
    parameter IW = 8,
    parameter AW = 10,
    parameter DW = 512,
    parameter MW = DW / 8,
    localparam BEATS = ((DW + 7) / 8 * 8 + 511) / 512
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
    output reg  [IW-1:0]          e_inst,
    output reg  [AW-1:0]          e_rrow,
    output reg  [AW-1:0]          e_wrow,
    output reg  [DW-1:0]          e_wdata,
    output reg  [MW-1:0]          e_wmask,
    input  wire [15:0]            e_free,

    // From the local switch: this memory's response beats.
    input  wire                   r_valid,
    input  wire [IW-1:0]          r_inst,
    input  wire [`S28_BEAT_W-1:0] r_beat,
    input  wire                   r_last,
    input  wire [511:0]           r_data
);
`ifdef SIMULATION
    initial begin
        if (IW > `S28_INST_W || AW > `S28_ROW_W || BEATS > (1 << `S28_BEAT_W))
            $error("foldsim_vmem_adapter: IW %0d / AW %0d / %0d beats past the word's fields", IW, AW, BEATS);
        if (MW != 1 && MW * 8 != DW)
            $error("foldsim_vmem_adapter: MW must be 1 or DW/8");
        e_valid   = 1'b0;
        rsp_valid = 1'b0;
    end
`endif

    // A read and a write of a host cycle are one instance's (the contract).
    always @(posedge clk) begin
        e_valid <= rd_valid | wr_valid;
        e_rd    <= rd_valid;
        e_wr    <= wr_valid;
        e_inst  <= wr_valid ? wr_inst : rd_inst;
        e_rrow  <= rd_row;
        e_wrow  <= wr_row;
        e_wdata <= wr_data;
        e_wmask <= wr_mask;
    end
    assign free = e_free;

    // The row, beat by beat.
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
    assign rsp_data = row_q[DW-1:0];
endmodule
