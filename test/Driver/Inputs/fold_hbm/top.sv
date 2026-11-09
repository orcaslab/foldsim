// Folded pipelines' stores in HBM (fold-store-hbm): per leaf, a pipeline of
// five stages on an enable (its store four slots deep) and one of four on
// another (three slots), their last stages read with an accumulator that is
// no pipeline; each stage after the first maps zero to zero, so that the
// target's zero registers at start are what the folded stores hold. Four leaves in a chain (a late output each).
// No reset: run with the registers zero at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic en, en2;
  logic [7:0] s1, s2, s3, s4, s5, t1, t2, t3, t4, acc;
  assign en = d[1] | acc[0];
  assign en2 = d[2] ^ acc[3];
  always_ff @(posedge clk) begin
    if (en) begin
      s1 <= (d * 8'd3) ^ acc;
      s2 <= s1 ^ {1'b0, s1[7:1]};
      s3 <= s2 * 8'd5;
      s4 <= {s3[3:0], s3[7:4]} ^ s3;
      s5 <= s4 * 8'd3;
    end
    if (en2) begin
      t1 <= d + {acc[3:0], acc[7:4]};
      t2 <= t1 * 8'd7;
      t3 <= t2 ^ {t2[6:0], 1'b0};
      t4 <= {t3[0], t3[7:1]} * 8'd9;
    end
    acc <= acc + s5 + (d ^ t4);
  end
  assign q = s5 ^ acc ^ d ^ t4;
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] a, b, c;
  Leaf l0(.clk, .d(x), .q(a));
  Leaf l1(.clk, .d(a ^ x), .q(b));
  Leaf l2(.clk, .d(b), .q(c));
  Leaf l3(.clk, .d(c), .q(y));
endmodule
