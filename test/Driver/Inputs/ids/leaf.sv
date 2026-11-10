// A leaf of a chain whose instances each take a constant identifier (`id`, a
// parent's number joined with a child's constant, as a generated hierarchy
// numbers its cores): ids, the per-instance constants a fold reads by
// instance (slot-segments: a separable decoder, no table of them).
module Leaf(input logic clk, reset, input logic [15:0] id, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] r;
  logic [15:0] acc;
  always_ff @(posedge clk) begin
    acc <= reset ? id : {acc[14:0], acc[15]} ^ {8'b0, d} ^ id;
    r <= d ^ acc[7:0] ^ acc[15:8];
  end
  assign q = r ^ id[15:8];
endmodule
