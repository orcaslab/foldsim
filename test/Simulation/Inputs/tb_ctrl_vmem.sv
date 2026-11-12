// tb_ctrl_vmem.sv: foldsim_ctrl's virtual-memory space registers over AXI-Lite: VMEM_BYTES (low word) and
// VMEM_BYTES_HI (high word) for a space of VMEM_BYTES bytes (a parameter: past 4 GiB too), VMEM_BASE_LO/HI as written.
`timescale 1ns/1ps
`include "foldsim_regs.vh"
module tb;
    parameter [63:0] VMEM_BYTES = 64'h1_7628_8000;
    reg clk = 0, reset = 1;
    always #1 clk = ~clk;
    reg aw = 0, w = 0, ar = 0; reg [11:0] awaddr = 0, araddr = 0; reg [31:0] wdata = 0;
    wire awready, wready, arready, bvalid, rvalid; wire [31:0] rdata;
    foldsim_ctrl #(.VMEM_BYTES(VMEM_BYTES)) dut (
        .clk(clk), .reset(reset),
        .s_awvalid(aw), .s_awready(awready), .s_awaddr(awaddr), .s_wvalid(w), .s_wready(wready), .s_wdata(wdata),
        .s_bvalid(bvalid), .s_bready(1'b1), .s_arvalid(ar), .s_arready(arready), .s_araddr(araddr),
        .s_rvalid(rvalid), .s_rready(1'b1), .s_rdata(rdata),
        .mem_base(), .mem_clock(), .mem_latency(), .mem_queue(), .mem_geom(), .mem_wmark(), .mem_timing(), .mem_refresh(),
        .mem_refi(), .mem_rfc(), .bubble(), .vmem_base(), .vmem_on(), .vmem_zero(), .vmem_move(), .vmem_moving(1'b0),
        .vmem_tel_sel(), .vmem_tel('0), .mem_stats('0), .mem_axi('0), .mem_dbg('0), .mem_idle(1'b1), .mem_error(1'b0),
        .mem_stall(1'b0), .profile_finish('0), .mem_reset_valid(), .mem_reset_ready(1'b1), .mem_reset_bits(),
        .reset_valid(), .reset_ready(1'b1), .reset_bits(), .dcr_req_valid_valid(), .dcr_req_valid_ready(1'b1),
        .dcr_req_valid_bits(), .dcr_req_rw_valid(), .dcr_req_rw_ready(1'b1), .dcr_req_rw_bits(), .dcr_req_addr_valid(),
        .dcr_req_addr_ready(1'b1), .dcr_req_addr_bits(), .dcr_req_data_valid(), .dcr_req_data_ready(1'b1),
        .dcr_req_data_bits(), .start_valid(), .start_ready(1'b1), .start_bits(), .dcr_rsp_valid_valid(1'b0),
        .dcr_rsp_valid_ready(), .dcr_rsp_valid_bits(1'b0), .dcr_rsp_data_valid(1'b0), .dcr_rsp_data_ready(),
        .dcr_rsp_data_bits('0), .busy_valid(1'b0), .busy_ready(), .busy_bits(1'b0));
    task automatic rd(input [11:0] a, output [31:0] d);
        @(posedge clk); ar <= 1; araddr <= a;
        do @(posedge clk); while (!arready); ar <= 0;
        while (!rvalid) @(posedge clk);
        d = rdata;
    endtask
    task automatic wr(input [11:0] a, input [31:0] d);
        @(posedge clk); aw <= 1; w <= 1; awaddr <= a; wdata <= d;
        do @(posedge clk); while (!(awready && wready)); aw <= 0; w <= 0;
        while (!bvalid) @(posedge clk);
    endtask
    integer err = 0;
    initial begin
        reg [31:0] lo, hi, blo, bhi;
        repeat (8) @(posedge clk); reset <= 0; repeat (4) @(posedge clk);
        rd(`FOLDSIM_REG_VMEM_BYTES, lo); rd(`FOLDSIM_REG_VMEM_BYTES_HI, hi);
        $display("VMEM_BYTES %h_%h", hi, lo);
        if ({hi, lo} != VMEM_BYTES) begin $display("FAIL: VMEM_BYTES %h_%h, expected %h", hi, lo, VMEM_BYTES); err = err + 1; end
        wr(`FOLDSIM_REG_VMEM_BASE_LO, 32'h4000_0000); wr(`FOLDSIM_REG_VMEM_BASE_HI, 32'h0000_0001);
        rd(`FOLDSIM_REG_VMEM_BASE_LO, blo); rd(`FOLDSIM_REG_VMEM_BASE_HI, bhi);
        if ({bhi, blo} != 64'h1_4000_0000) begin $display("FAIL: VMEM_BASE %h_%h", bhi, blo); err = err + 1; end
        if (err == 0) $display("PASS");
        $finish;
    end
    initial begin #200000; $display("FAIL: timeout"); $finish; end
endmodule
