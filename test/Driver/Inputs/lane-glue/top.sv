// Per-child glue in the unfolded top over eight Leaf instances (lanes), for
// foldsim-lane-scan: per-lane logic (a pending bit, a masked request vector,
// a per-lane grant decode of a same-cycle grant), and every reduction form
// the pass turns into a scan: XOR of the lanes' data (ac), a popcount (ac),
// AND / OR / parity of packed lane vectors (vec), a lowest-set encoder loop
// (prio), the lowest set bit and a one-hot AND-OR mux of the data (carry +
// ac), a round-robin grant from a pointer, `(ptr + i) % N` (rot), the
// granted lane's data (select) and a bit of a lane vector at the pointer
// (bitsel), and a per-lane register (a lane vector of state, moved into the
// cells). Leaf's data is late (it reads the frame's input).
module Leaf(input logic clk, input logic rst, input logic [7:0] x, input logic [7:0] K, input logic gnt,
            input logic [7:0] bc, output logic req, output logic [7:0] data, output logic done);
  logic [7:0] s;
  always_ff @(posedge clk)
    if (rst) s <= K;
    else s <= (s * 8'd5 + x + K + {7'd0, gnt}) ^ bc;
  assign req = s[0] ^ s[3];
  assign data = s ^ {x[3:0], x[7:4]};
  assign done = s[7] & s[2];
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  localparam int N = 8;
  logic req [N], done [N], gnt [N];
  logic [7:0] data [N];
  logic [7:0] bc;
  for (genvar j = 0; j < N; ++j) begin : g_leaf
    Leaf u(.clk, .rst, .x(x ^ 8'(j)), .K(8'(3 * j + 1)), .gnt(gnt[j]), .bc, .req(req[j]), .data(data[j]),
           .done(done[j]));
  end
  logic [N-1:0] rv, dv, mask, pq;
  for (genvar j = 0; j < N; ++j) begin : g_vec
    assign rv[j] = req[j] & ~done[j];
    assign dv[j] = done[j];
  end
  wire [N-1:0] m = rv & mask;
  logic [7:0] xr;
  logic [3:0] pc;
  always_comb begin
    xr = '0;
    pc = '0;
    for (int j = 0; j < N; ++j) begin
      xr ^= data[j];
      pc += 4'(rv[j]);
    end
  end
  wire alld = &dv, anyr = |m, par = ^rv;
  logic [2:0] enc;
  logic encv;
  always_comb begin
    enc = '0;
    encv = 1'b0;
    for (int i = N - 1; i >= 0; i--)
      if (m[i]) begin
        enc = 3'(i);
        encv = 1'b1;
      end
  end
  wire [N-1:0] oh = m & ~(m - N'(1));
  logic [7:0] ohd;
  always_comb begin
    ohd = '0;
    for (int j = 0; j < N; ++j) ohd |= {8{oh[j]}} & data[j];
  end
  logic [2:0] ptr, g;
  logic gv;
  always_comb begin
    gv = 1'b0;
    g = ptr;
    for (int i = N - 1; i >= 0; i--) begin
      automatic int c = (int'(ptr) + i) % N;
      if (rv[c]) begin
        gv = 1'b1;
        g = 3'(c);
      end
    end
  end
  wire [7:0] gd = data[g];
  for (genvar j = 0; j < N; ++j) begin : g_gnt
    assign gnt[j] = gv & (g == 3'(j));
  end
  wire rp = rv[ptr];
  always_ff @(posedge clk)
    if (rst) begin
      mask <= '1;
      pq <= '0;
      ptr <= '0;
      bc <= '0;
    end else begin
      mask <= mask ^ {rv[N-2:0], rv[N-1]};
      pq <= rv ^ dv;
      ptr <= gv ? g + 3'd1 : ptr;
      bc <= xr ^ gd ^ ohd;
    end
  assign y = xr ^ {pc, enc, encv} ^ {alld, anyr, par, rp, g, gv} ^ gd ^ ohd ^ {7'd0, &pq};
endmodule
