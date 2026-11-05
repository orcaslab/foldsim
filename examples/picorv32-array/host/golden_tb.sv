`timescale 1 ns / 1 ps
// golden_tb: the unfolded pico_v80 driven as the host drives the kernel (pico_host.cpp): reset, the CSR writes of
// load.hex, go, then the cycles until running falls, then the CSR reads 0..7. Prints the golden the host checks:
// key=value lines (hex). +load=<load.hex> +writes=<n> [+maxcyc=<n>].
module golden_tb;
  logic clk = 0, rst = 1, req_v = 0, req_w = 0, go = 0;
  logic [11:0] req_a = 0;
  logic [31:0] req_d = 0;
  wire rsp_v, running;
  wire [31:0] rsp_d;
  logic [43:0] load [0:65535];
  int writes = 0, maxcyc = 10000000, n;
  string load_f;
  pico_v80 dut(.clk, .rst, .host_req_valid(req_v), .host_req_write(req_w), .host_req_addr(req_a),
    .host_req_data(req_d), .go, .host_rsp_valid(rsp_v), .host_rsp_data(rsp_d), .running,
    .m_axi_mem_0_awready(1'b0), .m_axi_mem_0_wready(1'b0), .m_axi_mem_0_bvalid(1'b0), .m_axi_mem_0_bid('0),
    .m_axi_mem_0_bresp('0), .m_axi_mem_0_arready(1'b0), .m_axi_mem_0_rvalid(1'b0), .m_axi_mem_0_rdata('0),
    .m_axi_mem_0_rlast(1'b0), .m_axi_mem_0_rid('0), .m_axi_mem_0_rresp('0),
    .m_axi_mem_0_awvalid(), .m_axi_mem_0_awaddr(), .m_axi_mem_0_awid(), .m_axi_mem_0_awlen(), .m_axi_mem_0_wvalid(),
    .m_axi_mem_0_wdata(), .m_axi_mem_0_wstrb(), .m_axi_mem_0_wlast(), .m_axi_mem_0_bready(), .m_axi_mem_0_arvalid(),
    .m_axi_mem_0_araddr(), .m_axi_mem_0_arid(), .m_axi_mem_0_arlen(), .m_axi_mem_0_rready());
  task automatic tick; #1 clk = 1; #1 clk = 0; endtask
  initial begin
    if (!$value$plusargs("load=%s", load_f) || !$value$plusargs("writes=%d", writes)) $fatal(1, "+load= +writes=");
    void'($value$plusargs("maxcyc=%d", maxcyc));
    $readmemh(load_f, load, 0, writes - 1);
    repeat (64) tick();
    rst = 0;
    for (int i = 0; i < writes; i++) begin
      req_v = 1; req_w = 1; req_a = load[i][43:32]; req_d = load[i][31:0]; tick();
    end
    req_v = 0; req_w = 0;
    go = 1; tick(); go = 0;
    n = 0;
    do begin tick(); n++; end while (running && n < maxcyc);
    if (running) $fatal(1, "still running after %0d cycles", maxcyc);
    for (int a = 0; a < 8; a++) begin
      req_v = 1; req_w = 0; req_a = 12'(a); tick(); req_v = 0; #0;
      // the response is registered: valid on the cycle after the read
      if (!rsp_v) $fatal(1, "no response to read %0d", a);
      case (a)
        0: $display("checksum=%08h", rsp_d);
        1: $display("trap=%08h", rsp_d);
        2: $display("tokens=%08h", rsp_d);
        3: $display("cycles=%08h", rsp_d);
        default: $display("last%0d=%08h", a - 4, rsp_d);
      endcase
    end
    $finish;
  end
endmodule
