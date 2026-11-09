// Fan-in links for the reader-indexed mailboxes (csf-mailbox-builder): eight
// sources each read by one of two sinks, four a sink, so each source
// instance's outputs land in one sink row, a lane per sink input. A source's
// `valid` and `tag` are registers (early links), its `data` mixes in the
// frame's input (a late link); a sink's `ready` (read back by its four
// sources, a link that fans out) is a register too. Registers start from
// their resets.
module Src(input logic clk, input logic rst, input logic [7:0] x, input logic [7:0] K, input logic ready,
           output logic valid, output logic [2:0] tag, output logic [11:0] data);
  logic [11:0] s;
  always_ff @(posedge clk)
    if (rst) s <= {4'h0, K};
    else if (ready || !s[0]) s <= s * 12'd5 + {4'h0, x} + {4'h0, K};
  assign valid = s[0] ^ s[5];
  assign tag = s[11:9];
  assign data = s ^ {x, x[3:0]};
endmodule

module Sink(input logic clk, input logic rst, input logic [3:0] valid, input logic [3:0][2:0] tag,
            input logic [3:0][11:0] data, output logic [3:0] ready, output logic [15:0] acc);
  logic [15:0] a;
  logic [3:0] r;
  always_ff @(posedge clk)
    if (rst) begin
      a <= 16'h0;
      r <= 4'hf;
    end else begin
      a <= a * 16'd3 + (valid[0] ? {1'b0, tag[0], data[0]} : 16'h0) +
           (valid[1] ? {1'b0, tag[1], data[1]} : 16'h0) +
           (valid[2] ? {data[2], tag[2], 1'b1} : 16'h0) +
           (valid[3] ? {4'h0, data[3]} ^ {13'h0, tag[3]} : 16'h0);
      r <= a[3:0] ^ {r[2:0], r[3]};
    end
  assign ready = r;
  assign acc = a;
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] valid, ready;
  logic [7:0][2:0] tag;
  logic [7:0][11:0] data;
  logic [1:0][15:0] acc;
  for (genvar j = 0; j < 8; ++j) begin : g_src
    Src src(.clk, .rst, .x(x ^ 8'(j)), .K(8'(2 * j + 1)), .ready(ready[j]), .valid(valid[j]), .tag(tag[j]),
            .data(data[j]));
  end
  for (genvar k = 0; k < 2; ++k) begin : g_sink
    Sink sink(.clk, .rst, .valid(valid[4*k +: 4]), .tag(tag[4*k +: 4]), .data(data[4*k +: 4]),
              .ready(ready[4*k +: 4]), .acc(acc[k]));
  end
  assign y = acc[0][7:0] ^ acc[1][15:8] ^ acc[1][7:0];
endmodule
