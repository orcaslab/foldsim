// Port-bundle glue in the unfolded top over eight Leaf lanes (for
// foldsim-lane-scan): each Leaf has two ports (an unpacked array of data and a
// packed valid pair: elements of one lane's arrays are the lane's); a
// suffix chain over the lanes (c_j = c_{j+1} & (a_j | b_j), a cell over the
// lanes in reverse order); the lowest pending lane isolated as `p & -p` (a
// carry chain with a carry of 1); and a buffer register of {onehot, index,
// tag} (split at the onehot's ends: the onehot is per-lane state), whose
// onehot bit j is lane j's grant.
module Leaf(input logic clk, input logic rst, input logic [7:0] x, input logic [7:0] K, input logic go,
            input logic gnt, output logic [7:0] pd [2], output logic [1:0] pv, output logic a, output logic b);
  logic [7:0] s;
  always_ff @(posedge clk)
    if (rst) s <= K;
    else s <= (s * 8'd5 + x + K) ^ {6'd0, go, gnt};
  assign pd[0] = s ^ x;
  assign pd[1] = {s[3:0], s[7:4]};
  assign pv = {s[2] ^ s[6], s[1] | s[4]};
  assign a = s[0] ^ s[3];
  assign b = s[1] & s[5];
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  localparam int N = 8;
  logic a [N], b [N], go [N], gnt [N];
  logic [7:0] pd [N][2];
  logic [1:0] pv [N];
  for (genvar j = 0; j < N; ++j) begin : g
    Leaf u(.clk, .rst, .x(x ^ 8'(j)), .K(8'(3 * j + 1)), .go(go[j]), .gnt(gnt[j]), .pd(pd[j]), .pv(pv[j]),
           .a(a[j]), .b(b[j]));
  end
  // suffix chain
  logic c;
  logic [N-1:0] seen;
  always_comb begin
    c = 1'b1;
    for (int i = N - 1; i >= 0; i--) begin
      seen[i] = c;
      c = c & (a[i] | b[i]);
    end
  end
  // port pick per lane: port 1 when its valid is set
  logic [N-1:0] pend;
  logic [7:0] tagj [N];
  for (genvar j = 0; j < N; ++j) begin : g2
    assign go[j] = seen[j];
    assign pend[j] = pv[j][0] | pv[j][1];
    assign tagj[j] = pv[j][1] ? pd[j][1] : pd[j][0];
  end
  // lowest pending lane, its index and tag
  wire [N-1:0] oh = pend & -pend;
  logic [2:0] idx;
  logic [7:0] tag;
  always_comb begin
    idx = '0;
    tag = '0;
    for (int j = 0; j < N; ++j)
      if (oh[j]) begin
        idx = 3'(j);
        tag = tagj[j];
      end
  end
  // a buffer of {onehot, index, tag}
  logic [N+3+8-1:0] buf_r;
  logic buf_v;
  always_ff @(posedge clk)
    if (rst) begin
      buf_v <= 1'b0;
      buf_r <= '0;
    end else begin
      buf_v <= |pend;
      if (|pend) buf_r <= {oh, idx, tag};
    end
  for (genvar j = 0; j < N; ++j) begin : g3
    assign gnt[j] = buf_v & buf_r[11 + j];
  end
  assign y = {7'd0, c} ^ seen ^ buf_r[7:0] ^ {4'd0, buf_v, buf_r[10:8]};
endmodule
