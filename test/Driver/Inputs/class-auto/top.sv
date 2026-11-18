// The level's replicated instances (foldsim --class-auto): a generate loop of four counters (per port, as an AXI
// adapter's tag buffers or a cache's per-bank queues) left outside the classes, beside two instances of a class.
// Each counter keeps state and mixes its input with its neighbour's output.
module Ctr(input logic clk, input logic rst, input logic [7:0] in, input logic [7:0] nb, output logic [7:0] out);
  logic [7:0] q, r;
  always_ff @(posedge clk)
    if (rst) begin q <= 0; r <= 8'h5a; end
    else begin
      q <= q + (in ^ nb) + {7'b0, r[0]};
      r <= {r[6:0], r[7] ^ r[5] ^ q[1]};
    end
  assign out = q ^ {r[3:0], r[7:4]};
endmodule

module Leaf(input logic clk, input logic rst, input logic [7:0] a, output logic [7:0] b);
  logic [7:0] s;
  always_ff @(posedge clk) if (rst) s <= 0; else s <= (s << 1) ^ a ^ {s[7], 7'b0};
  assign b = s + a;
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [3:0][7:0] o;
  logic [7:0] l0, l1;
  for (genvar i = 0; i < 4; ++i) begin : g_ctr
    Ctr ctr(.clk, .rst, .in(x + 8'(i * 37)), .nb(o[(i + 1) % 4]), .out(o[i]));
  end
  Leaf leaf0(.clk, .rst, .a(o[0] ^ o[1]), .b(l0));
  Leaf leaf1(.clk, .rst, .a(o[2] ^ o[3]), .b(l1));
  assign y = l0 ^ l1 ^ o[3];
endmodule
