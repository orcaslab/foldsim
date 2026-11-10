// Folded pipelines' stores streamed (fold-store-stream): sixteen leaves, each
// with four pipelines on enables of their own, their stores from one bit to
// over 32 kbit (an entry per slot and instance): a 1-bit pipeline of four
// stages (store 3 x 16 x 1), the 8-bit ones of five and four stages, and a
// 600-bit one of five stages (store 4 x 16 x 600 = 38,400 bits). Each stage
// after the first maps zero to zero, so that the target's zero registers at
// start are what the folded stores hold. No reset: run with the registers zero
// at start (+verilator+rand+reset+0).
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic en, en2, en3, en4;
  logic [7:0] s1, s2, s3, s4, s5, t1, t2, t3, t4, acc;
  logic b1, b2, b3, b4;
  logic [599:0] w1, w2, w3, w4, w5;
  assign en = d[1] | acc[0];
  assign en2 = d[2] ^ acc[3];
  assign en3 = d[5] | acc[6];
  assign en4 = d[0] ^ acc[1] ^ d[7];
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
    if (en3) begin
      b1 <= ^(d ^ acc);
      b2 <= b1;
      b3 <= b2;
      b4 <= b3;
    end
    if (en4) begin
      w1 <= {75{d ^ acc}} ^ {600{acc[2]}};
      w2 <= w1 ^ {w1[598:0], 1'b0};
      w3 <= {w2[299:0], w2[599:300]} ^ w2;
      w4 <= w3 ^ {w3[588:0], 11'd0};
      w5 <= {w4[0], w4[599:1]};
    end
    acc <= acc + s5 + (d ^ t4) + {7'd0, b4} + (w5[7:0] ^ w5[599:592] ^ w5[307:300]);
  end
  assign q = s5 ^ acc ^ d ^ t4 ^ {7'd0, b4} ^ w5[455:448];
endmodule
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] q [16];
  logic [7:0] r;
  for (genvar i = 0; i < 16; i++) begin : g
    Leaf l(.clk, .d(x ^ 8'(i * 29) ^ (i == 0 ? 8'd0 : r)), .q(q[i]));
  end
  always_ff @(posedge clk) r <= q[0] ^ q[1] ^ q[2] ^ q[3] ^ q[4] ^ q[5] ^ q[6] ^ q[7] ^ q[8] ^ q[9] ^ q[10] ^ q[11] ^ q[12] ^ q[13] ^ q[14] ^ q[15];
  assign y = r ^ q[3];
endmodule
