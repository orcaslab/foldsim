// The leaf of Driver/Inputs/paged (see there), for fixedscratch's chains.
module Leaf(input logic clk, input logic [7:0] d, output logic [7:0] q);
  logic [7:0] ram [16];
  logic [7:0] rd, r, ob;
  logic [159:0] acc;
  logic [12:0] cnt;
  always_ff @(posedge clk) begin
    if (d[0]) ram[d[7:4]] <= d ^ cnt[7:0];
    rd <= ram[d[3:0]];
    r <= d ^ ob;
    if (d[1]) acc <= {acc[151:0], d ^ rd} ^ {8'b0, acc[159:8]};
    cnt <= cnt + {5'b0, d};
    ob <= acc[7:0] ^ acc[159:152] ^ acc[87:80] ^ cnt[12:5];
  end
  assign q = rd ^ d ^ r;
endmodule
