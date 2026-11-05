`timescale 1 ns / 1 ps
// pico_v80: pico_array behind the foldsim kernel's control endpoint, the design's V80 binding (as Vortex's
// foldsim_target.sv is Vortex's). Its ports carry their own names; pico_v80.toml's [platform.control] binds them to
// the endpoint's roles (reset, start, busy, the CSR request and response).
//   host_req (write): addr[11:10] the tile, addr[9:0] the word: loads the tile's RAM while the cores are held.
//   go:               releases the cores (they run from address 0 until each traps).
//   host_req (read):  answered the next cycle on host_rsp: addr 0 the checksum of every output token, 1 the trapped
//                     tiles, 2 the output tokens, 3 the target cycles since go, 4 + t tile t's last output token.
//   running:          the cores run and not every tile has trapped.
// The kernel binds at least one AXI4 memory port; the array has no memory outside its tiles, so m_axi_mem_0 is
// idle (no request ever issued, every response taken).
module pico_v80 #(
  parameter N      = 4,
  parameter ID_W   = 4,
  parameter ADDR_W = 64,
  parameter DATA_W = 512
) (
  input  logic                clk,
  input  logic                rst,
  input  logic                host_req_valid,
  input  logic                host_req_write,
  input  logic [11:0]         host_req_addr,
  input  logic [31:0]         host_req_data,
  input  logic                go,
  output logic                host_rsp_valid,
  output logic [31:0]         host_rsp_data,
  output logic                running,
  // AXI4 device-memory master 0 (idle)
  output logic                m_axi_mem_0_awvalid,
  input  logic                m_axi_mem_0_awready,
  output logic [ADDR_W-1:0]   m_axi_mem_0_awaddr,
  output logic [ID_W-1:0]     m_axi_mem_0_awid,
  output logic [7:0]          m_axi_mem_0_awlen,
  output logic                m_axi_mem_0_wvalid,
  input  logic                m_axi_mem_0_wready,
  output logic [DATA_W-1:0]   m_axi_mem_0_wdata,
  output logic [DATA_W/8-1:0] m_axi_mem_0_wstrb,
  output logic                m_axi_mem_0_wlast,
  input  logic                m_axi_mem_0_bvalid,
  output logic                m_axi_mem_0_bready,
  input  logic [ID_W-1:0]     m_axi_mem_0_bid,
  input  logic [1:0]          m_axi_mem_0_bresp,
  output logic                m_axi_mem_0_arvalid,
  input  logic                m_axi_mem_0_arready,
  output logic [ADDR_W-1:0]   m_axi_mem_0_araddr,
  output logic [ID_W-1:0]     m_axi_mem_0_arid,
  output logic [7:0]          m_axi_mem_0_arlen,
  input  logic                m_axi_mem_0_rvalid,
  output logic                m_axi_mem_0_rready,
  input  logic [DATA_W-1:0]   m_axi_mem_0_rdata,
  input  logic                m_axi_mem_0_rlast,
  input  logic [ID_W-1:0]     m_axi_mem_0_rid,
  input  logic [1:0]          m_axi_mem_0_rresp
);
  logic                run;
  logic [31:0]         cycles, tokens;
  logic [N-1:0]        out_valid, trap;
  logic [N-1:0][31:0]  out_data, last;
  logic [31:0]         checksum;
  wire load = host_req_valid && host_req_write && !run;
  pico_array #(.N(N)) array (
    .clk(clk), .resetn(run), .load_en(load), .load_tile(8'(host_req_addr[11:10])),
    .load_addr(host_req_addr[9:0]), .load_data(host_req_data),
    .out_valid(out_valid), .out_data(out_data), .trap(trap), .checksum(checksum)
  );
  always_ff @(posedge clk) begin
    if (rst) begin
      run            <= 1'b0;
      cycles         <= '0;
      tokens         <= '0;
      last           <= '0;
      host_rsp_valid <= 1'b0;
      host_rsp_data  <= '0;
    end else begin
      if (go) run <= 1'b1;
      if (run) cycles <= cycles + 1;
      for (int t = 0; t < N; t++)
        if (out_valid[t]) last[t] <= out_data[t];
      tokens <= tokens + 32'($countones(out_valid));
      host_rsp_valid <= host_req_valid && !host_req_write;
      case (host_req_addr)
        12'd0:   host_rsp_data <= checksum;
        12'd1:   host_rsp_data <= 32'(trap);
        12'd2:   host_rsp_data <= tokens;
        12'd3:   host_rsp_data <= cycles;
        default: host_rsp_data <= host_req_addr - 12'd4 < 12'(N) ? last[host_req_addr - 12'd4] : '0;
      endcase
    end
  end
  assign running = run && !(&trap);
  // The idle memory master.
  assign m_axi_mem_0_awvalid = 1'b0;
  assign m_axi_mem_0_awaddr  = '0;
  assign m_axi_mem_0_awid    = '0;
  assign m_axi_mem_0_awlen   = '0;
  assign m_axi_mem_0_wvalid  = 1'b0;
  assign m_axi_mem_0_wdata   = '0;
  assign m_axi_mem_0_wstrb   = '0;
  assign m_axi_mem_0_wlast   = 1'b0;
  assign m_axi_mem_0_bready  = 1'b1;
  assign m_axi_mem_0_arvalid = 1'b0;
  assign m_axi_mem_0_araddr  = '0;
  assign m_axi_mem_0_arid    = '0;
  assign m_axi_mem_0_arlen   = '0;
  assign m_axi_mem_0_rready  = 1'b1;
  wire unused = &{1'b0, m_axi_mem_0_awready, m_axi_mem_0_wready, m_axi_mem_0_bvalid, m_axi_mem_0_bid,
                  m_axi_mem_0_bresp, m_axi_mem_0_arready, m_axi_mem_0_rvalid, m_axi_mem_0_rdata,
                  m_axi_mem_0_rlast, m_axi_mem_0_rid, m_axi_mem_0_rresp};
endmodule
