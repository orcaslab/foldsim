`timescale 1 ns / 1 ps
// pico_array: N PicoRV32 tiles (YosysHQ/picorv32, upstream/picorv32.v, unmodified), each with a private 4 KiB
// RAM, in a ring of mailboxes: tile i's sends land in tile i+1's mailbox. A second, non-Vortex design for
// foldsim (gen_1105, task D): the tiles repeat, so they fold as one class.
//
// Tile address map: 0x0000-0x0fff RAM (byte writes, one-cycle read); 0x1000_0000 OUT (write: an output
// token); 0x1000_0004 ID (read: the tile's index); 0x1000_0008 MBOX (write: send to the next tile; read: the
// last word received, 0 before any). The RAMs are loaded through `load_*` while the cores are in reset.
module pico_tile #(parameter WORDS = 1024) (
  input  logic        clk,
  input  logic        resetn,
  input  logic [7:0]  tile_id,
  input  logic        load_en,
  input  logic [9:0]  load_addr,
  input  logic [31:0] load_data,
  input  logic [31:0] rx_data,
  output logic        out_valid,
  output logic [31:0] out_data,
  output logic        tx_valid,
  output logic [31:0] tx_data,
  output logic        trap
);
  logic        mem_valid, mem_instr, mem_ready;
  logic [31:0] mem_addr, mem_wdata, mem_rdata, ram_q, io_q;
  logic [3:0]  mem_wstrb;
  logic        io_sel;
  picorv32 #(
    .ENABLE_COUNTERS(0), .ENABLE_COUNTERS64(0), .ENABLE_MUL(1), .ENABLE_DIV(1), .BARREL_SHIFTER(1),
    .PROGADDR_RESET(32'h0), .STACKADDR(32'h1000)
  ) cpu (
    .clk(clk), .resetn(resetn), .trap(trap),
    .mem_valid(mem_valid), .mem_instr(mem_instr), .mem_ready(mem_ready), .mem_addr(mem_addr),
    .mem_wdata(mem_wdata), .mem_wstrb(mem_wstrb), .mem_rdata(mem_rdata),
    .mem_la_read(), .mem_la_write(), .mem_la_addr(), .mem_la_wdata(), .mem_la_wstrb(),
    .pcpi_valid(), .pcpi_insn(), .pcpi_rs1(), .pcpi_rs2(),
    .pcpi_wr(1'b0), .pcpi_rd(32'h0), .pcpi_wait(1'b0), .pcpi_ready(1'b0),
    .irq(32'h0), .eoi(), .trace_valid(), .trace_data()
  );
  logic [31:0] ram [0:WORDS-1];
  wire        acc    = resetn && mem_valid && !mem_ready;
  wire        in_ram = mem_addr < 4 * WORDS;
  wire [9:0]  waddr  = load_en ? load_addr : mem_addr[11:2];
  wire [3:0]  wen    = load_en ? 4'hf : (acc && in_ram ? mem_wstrb : 4'h0);
  wire [31:0] wdata  = load_en ? load_data : mem_wdata;
  always_ff @(posedge clk) begin
    for (int i = 0; i < 4; i++)
      if (wen[i]) ram[waddr][8*i +: 8] <= wdata[8*i +: 8];
    ram_q <= ram[mem_addr[11:2]];
  end
  always_ff @(posedge clk) begin
    if (!resetn) begin
      mem_ready <= 1'b0;
      out_valid <= 1'b0;
      tx_valid  <= 1'b0;
      io_sel    <= 1'b0;
    end else begin
      mem_ready <= acc;
      io_sel    <= !in_ram;
      out_valid <= acc && mem_addr == 32'h1000_0000 && |mem_wstrb;
      tx_valid  <= acc && mem_addr == 32'h1000_0008 && |mem_wstrb;
      if (acc && mem_addr == 32'h1000_0000 && |mem_wstrb) out_data <= mem_wdata;
      if (acc && mem_addr == 32'h1000_0008 && |mem_wstrb) tx_data <= mem_wdata;
      io_q <= mem_addr == 32'h1000_0004 ? {24'h0, tile_id} : mem_addr == 32'h1000_0008 ? rx_data : 32'h0;
    end
  end
  assign mem_rdata = io_sel ? io_q : ram_q;
endmodule

module pico_array #(parameter N = 4) (
  input  logic          clk,
  input  logic          resetn,
  input  logic          load_en,
  input  logic [7:0]    load_tile,
  input  logic [9:0]    load_addr,
  input  logic [31:0]   load_data,
  output logic [N-1:0]  out_valid,
  output logic [N-1:0][31:0] out_data,
  output logic [N-1:0]  trap,
  output logic [31:0]   checksum
);
  logic [N-1:0]        tx_valid;
  logic [N-1:0][31:0]  tx_data, mbox;
  for (genvar g = 0; g < N; g++) begin : g_tiles
    pico_tile tile (
      .clk(clk), .resetn(resetn), .tile_id(8'(g)),
      .load_en(load_en && load_tile == 8'(g)), .load_addr(load_addr), .load_data(load_data),
      .rx_data(mbox[g]), .out_valid(out_valid[g]), .out_data(out_data[g]),
      .tx_valid(tx_valid[g]), .tx_data(tx_data[g]), .trap(trap[g])
    );
  end
  // The ring: tile g's sends land in tile g+1's mailbox; a running checksum of every output token.
  always_ff @(posedge clk) begin
    if (!resetn) begin
      mbox     <= '0;
      checksum <= '0;
    end else begin
      for (int g = 0; g < N; g++)
        if (tx_valid[g]) mbox[(g + 1) % N] <= tx_data[g];
      checksum <= checksum_next(checksum, out_valid, out_data);
    end
  end
  function automatic logic [31:0] checksum_next(logic [31:0] c, logic [N-1:0] v, logic [N-1:0][31:0] d);
    for (int g = 0; g < N; g++)
      if (v[g]) c = {c[30:0], c[31]} ^ d[g] ^ 32'(g);
    return c;
  endfunction
endmodule
