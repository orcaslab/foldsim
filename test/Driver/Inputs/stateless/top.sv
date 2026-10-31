// Stateless classes: two SMs, each with four per-lane address units (pure
// logic, no clock) between its registered lane operands and its
// accumulator, as an LSU's AGUs sit between its dispatch registers and its
// request queue. A unit's operands depend on the SM's registers alone; its
// address feeds the SM's next state and, through the SM, its operands of
// the next cycle. The SMs' accumulators meet in the top's output. Each SM's
// lane port reads its own units alone (two of the eight), and the top reads
// each SM's register alone (one of the two): links a compact fold keeps
// small. Registers start from their resets.
module Agu(input logic [15:0] base, input logic [15:0] off, input logic [1:0] op,
           output logic [15:0] addr, output logic misal);
  logic [15:0] sum;
  always_comb begin
    case (op)
      2'd0: sum = base + off;
      2'd1: sum = base - off;
      2'd2: sum = base + {off[13:0], 2'b00};
      default: sum = {base[7:0], off[15:8]};
    endcase
  end
  // A window remapped, as a stack interleave remaps its own.
  assign addr = sum[15:12] == 4'hf ? {sum[15:12], sum[3:0], sum[11:4]} : sum;
  assign misal = sum[0];
endmodule

module Sm(input logic clk, input logic rst, input logic [7:0] seed, input logic [7:0] x,
          output logic [15:0] acc);
  logic [3:0][15:0] b, o;
  logic [3:0][1:0] op;
  logic [3:0][15:0] a;
  logic [3:0] m;
  for (genvar i = 0; i < 4; ++i) begin : g_lane
    Agu agu(.base(b[i]), .off(o[i]), .op(op[i]), .addr(a[i]), .misal(m[i]));
    always_ff @(posedge clk)
      if (rst) begin
        b[i] <= 16'(i) ^ {8'h0, seed};
        o[i] <= 0;
        op[i] <= 0;
      end else begin
        b[i] <= {x, x ^ seed} + 16'(i * 13);
        o[i] <= a[i] ^ {seed, x};
        op[i] <= x[2*i +: 2] ^ op[i] ^ {m[i], acc[i]};
      end
  end
  always_ff @(posedge clk)
    if (rst) acc <= 0;
    else acc <= acc + (a[0] ^ a[1] ^ a[2] ^ a[3]) + {12'b0, m};
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [15:0] acc0, acc1;
  Sm sm0(.clk, .rst, .seed(8'h00), .x, .acc(acc0));
  Sm sm1(.clk, .rst, .seed(8'h5a), .x(~x), .acc(acc1));
  assign y = acc0[7:0] ^ acc0[15:8] ^ acc1[7:0] ^ acc1[15:8];
endmodule
