// vport_axi_model: the AXI4 memory behind one static port (foldsim_vport,
// SYNC: 512-bit beats, one id) in the folded-simulator testbenches. It keeps
// rows late and reads them early, as tb_vport's: a write reaches memory only
// at its B (lo to hi host cycles after its last W beat, B's in order); a read
// takes memory's rows as its AR is taken and answers lo to hi cycles on, in
// order, a beat a cycle. So a read the port lets reach AR before the B of an
// earlier write of its row answers the old row (the fence is checked by the
// answers). `stall` (percent) holds each ready low at random. Memory starts
// zero (the host's cleared region).
module vport_axi_model (
    input  logic        clk, rst,
    input  int          cyc, lo, hi, stall,
    input  logic        awvalid, output logic awready, input logic [63:0] awaddr, input logic [7:0] awlen,
    input  logic        wvalid,  output logic wready,  input logic [511:0] wdata, input logic [63:0] wstrb,
    input  logic        wlast,
    output logic        bvalid,  input logic bready,
    input  logic        arvalid, output logic arready, input logic [63:0] araddr, input logic [7:0] arlen,
    output logic        rvalid,  input logic rready, output logic [511:0] rdata, output logic rlast);
  logic [511:0] mem [longint];
  typedef struct { longint at; int len; } aw_t;
  typedef struct { longint at; logic [511:0] d; logic [63:0] s; } beat_t;
  typedef struct { int due; beat_t beats [$]; } wr_t;
  typedef struct { int due; logic [511:0] d; bit last; } rb_t;
  aw_t aws [$];
  beat_t cur [$];
  wr_t bq [$];
  rb_t rq [$];
  function automatic int lat();
    return lo + int'($urandom % (hi - lo + 1));
  endfunction
  function automatic logic [511:0] rd(longint a);
    return mem.exists(a) ? mem[a] : '0;
  endfunction
  always @(negedge clk) begin
    awready = ($urandom % 100) >= stall;
    wready  = aws.size() != 0 && ($urandom % 100) >= stall;
    arready = ($urandom % 100) >= stall;
    bvalid  = bq.size() != 0 && cyc >= bq[0].due;
    rvalid  = rq.size() != 0 && cyc >= rq[0].due && ($urandom % 100) >= stall;
    rdata   = rq.size() != 0 ? rq[0].d : '0;
    rlast   = rq.size() != 0 && rq[0].last;
  end
  always @(posedge clk) begin
    if (rvalid && rready) void'(rq.pop_front());
    if (bvalid && bready) begin
      automatic wr_t w = bq.pop_front();
      foreach (w.beats[i]) begin
        automatic logic [511:0] v = rd(w.beats[i].at);
        for (int b = 0; b < 64; b++)
          if (w.beats[i].s[b]) v[b*8 +: 8] = w.beats[i].d[b*8 +: 8];
        mem[w.beats[i].at] = v;
      end
    end
    if (arvalid && arready) begin
      automatic int due = cyc + lat();
      for (int i = 0; i <= int'(arlen); i++) begin
        automatic rb_t r;
        r.due = due; r.d = rd(longint'(araddr >> 6) + i); r.last = i == int'(arlen);
        rq.push_back(r);
      end
    end
    if (wvalid && wready) begin
      automatic beat_t x;
      x.at = aws[0].at + cur.size(); x.d = wdata; x.s = wstrb;
      cur.push_back(x);
      if (wlast) begin
        automatic wr_t w;
        if (cur.size() != aws[0].len + 1)
          $display("PROTOCOL: AXI write of %0d beats, AW said %0d", cur.size(), aws[0].len + 1);
        w.due = cyc + lat(); w.beats = cur;
        bq.push_back(w);
        cur.delete();
        void'(aws.pop_front());
      end
    end
    if (awvalid && awready) begin
      automatic aw_t a;
      a.at = longint'(awaddr >> 6); a.len = int'(awlen);
      aws.push_back(a);
    end
  end
endmodule
