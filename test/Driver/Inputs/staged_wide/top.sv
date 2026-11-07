// A leaf with wide arithmetic a stage cannot end inside unless it is split
// (splitForStages): a 53 x 53 product (106 bits), 106-bit sums and a
// difference, unsigned and signed 64-bit ordering compares, and variable
// shifts of 127 bits by a 127-bit amount (as Rocket's FPU has), over state
// fed by the input. No reset: run with the registers zero at start.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [52:0] a, b;
  logic [105:0] p, s;
  logic [63:0] u, v;
  logic [126:0] x, sh;
  logic [7:0] f;
  always_ff @(posedge clk) begin
    a <= {a[44:0], d} ^ {b[7:0], a[52:8]};
    b <= {b[45:0], d[6:0]} + a;
    p <= a * b;
    s <= p + {s[97:0], d} - {d, p[105:8]};
    u <= {u[55:0], d} ^ p[63:0];
    v <= {v[55:0], ~d} ^ s[105:42];
    x <= {x[118:0], d} ^ {p, s[20:0]};
    sh <= ($signed(x) >>> {120'b0, d[6:0] ^ s[6:0]}) ^ (x << s[126 - 106 +: 1] * x[126:0]) ^ (x >> p);
    f <= {f[3:0], u < v, $signed(u) < $signed(v), u >= s[63:0], $signed(v) > $signed(p[63:0])};
  end
  assign q = p[7:0] ^ s[105:98] ^ sh[126:119] ^ sh[7:0] ^ f ^ u[63:56];
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a ^ x), .q(b));
  Leaf l2(.clk, .d(b), .q(y));
endmodule
