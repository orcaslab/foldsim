// Eight nodes in a chain behind a pulse source, for pp-idle-skip: a node's
// registers change only where an input changed (a register of each input's
// last value), so between the source's pulses the nodes go idle one after
// the other. Node k reads node k - 1 and node k - 2 (registered outputs; the
// first two a constant instead), the source reads the input token every
// sixteenth cycle, and a sink mixes two nodes into the output. No reset: run
// with the registers zero at start (+verilator+rand+reset+0). top.sv's Top has
// no reset input (tb_pages.sv), top_rst.sv's one for the source's counter
// (tb_linkhbm.sv).
module Src(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] q);
  logic [3:0] cnt;
  always_ff @(posedge clk) begin
    cnt <= rst ? 4'd0 : cnt + 4'd1;
    if (cnt == 4'd0) q <= x;
  end
endmodule
module Node(input logic clk, input logic [7:0] a, input logic [7:0] b, output logic [7:0] q);
  logic [7:0] la, lb;
  logic [159:0] acc;
  logic [12:0] cnt;
  always_ff @(posedge clk) begin
    la <= a;
    lb <= b;
    if (a != la || b != lb) begin
      acc <= {acc[151:0], a ^ b} ^ {8'b0, acc[159:8]};
      cnt <= cnt + {5'b0, a};
    end
    q <= acc[7:0] ^ acc[159:152] ^ acc[87:80] ^ cnt[12:5] ^ a;
  end
endmodule
module Sink(input logic clk, input logic [7:0] d, input logic [7:0] e, output logic [7:0] y);
  always_ff @(posedge clk) y <= d ^ {e[3:0], e[7:4]};
endmodule
