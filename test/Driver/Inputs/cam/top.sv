// Four associative tables in a chain: each matches its input against every
// entry each cycle (a read per entry, as a cache MSHR's address table) and
// writes it at a rotating pointer when its low bits are not zero (an enable
// independent of the table, so that circt-verilog makes the table a memory);
// its output mixes the match vector with an entry read at an address from
// the input and a count of the hits.
module Cam(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] tags [0:15];
  logic [15:0] valid, hits;
  logic [3:0] wp;
  logic [7:0] seen;
  wire write = d[1:0] != 2'd0;
  for (genvar i = 0; i < 16; i++) begin : g_match
    assign hits[i] = valid[i] && tags[i] == d;
  end
  always @(posedge clk) begin
    if (write)
      tags[wp] <= d;
  end
  always_ff @(posedge clk) begin
    if (write) begin
      valid[wp] <= 1'b1;
      wp <= wp + 4'd1;
    end
    seen <= seen + {7'd0, hits != 16'd0};
  end
  assign q = hits[15:8] ^ hits[7:0] ^ tags[d[5:2]] ^ seen;
endmodule

module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Cam c0(.clk, .d(x & 8'h3f), .q(a));
  Cam c1(.clk, .d(a & 8'h1f), .q(b));
  Cam c2(.clk, .d(b & 8'h0f), .q(c));
  Cam c3(.clk, .d(c & 8'h07), .q(y));
endmodule
