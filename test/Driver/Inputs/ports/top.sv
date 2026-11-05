// Folded port bundles (--fold-ports): four bundles mp_<k>_* to memory-like
// endpoints outside the target, one Unit a bundle. A bundle's outputs:
// avalid, addr, data from the unit, ack from its response input in the same
// cycle (late: through the top's model), and tag from a register of the top
// itself (early: a token of the next cycle); its inputs: rvalid, rdata (the
// endpoint's state: Moore).
module Unit(input logic clk, input logic rst, input logic [7:0] seed, input logic [7:0] x,
            input logic rvalid, input logic [15:0] rdata,
            output logic avalid, output logic [7:0] addr, output logic [15:0] data,
            output logic ack, output logic [15:0] acc);
  logic [15:0] s;
  always_ff @(posedge clk)
    if (rst) begin s <= {seed, ~seed}; acc <= 0; end
    else begin
      s <= {s[14:0], s[15] ^ s[13] ^ s[12] ^ s[10]} ^ {8'b0, x};
      acc <= acc + (rvalid ? rdata : 16'h0101);
    end
  assign avalid = s[0] | s[3];
  assign addr = s[11:4] ^ seed;
  assign data = s ^ acc;
  assign ack = rvalid ^ s[1];
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y,
           output logic mp_0_avalid, output logic [7:0] mp_0_addr, output logic [15:0] mp_0_data, output logic mp_0_ack, output logic [3:0] mp_0_tag,
           input logic mp_0_rvalid, input logic [15:0] mp_0_rdata,
           output logic mp_1_avalid, output logic [7:0] mp_1_addr, output logic [15:0] mp_1_data, output logic mp_1_ack, output logic [3:0] mp_1_tag,
           input logic mp_1_rvalid, input logic [15:0] mp_1_rdata,
           output logic mp_2_avalid, output logic [7:0] mp_2_addr, output logic [15:0] mp_2_data, output logic mp_2_ack, output logic [3:0] mp_2_tag,
           input logic mp_2_rvalid, input logic [15:0] mp_2_rdata,
           output logic mp_3_avalid, output logic [7:0] mp_3_addr, output logic [15:0] mp_3_data, output logic mp_3_ack, output logic [3:0] mp_3_tag,
           input logic mp_3_rvalid, input logic [15:0] mp_3_rdata);
  logic [3:0][15:0] acc;
  Unit u0(.clk, .rst, .seed(8'd1), .x, .rvalid(mp_0_rvalid), .rdata(mp_0_rdata), .avalid(mp_0_avalid),
          .addr(mp_0_addr), .data(mp_0_data), .ack(mp_0_ack), .acc(acc[0]));
  Unit u1(.clk, .rst, .seed(8'd38), .x, .rvalid(mp_1_rvalid), .rdata(mp_1_rdata), .avalid(mp_1_avalid),
          .addr(mp_1_addr), .data(mp_1_data), .ack(mp_1_ack), .acc(acc[1]));
  Unit u2(.clk, .rst, .seed(8'd75), .x, .rvalid(mp_2_rvalid), .rdata(mp_2_rdata), .avalid(mp_2_avalid),
          .addr(mp_2_addr), .data(mp_2_data), .ack(mp_2_ack), .acc(acc[2]));
  Unit u3(.clk, .rst, .seed(8'd112), .x, .rvalid(mp_3_rvalid), .rdata(mp_3_rdata), .avalid(mp_3_avalid),
          .addr(mp_3_addr), .data(mp_3_data), .ack(mp_3_ack), .acc(acc[3]));
  logic [3:0] tag [4];
  always_ff @(posedge clk)
    for (int k = 0; k < 4; ++k)
      tag[k] <= rst ? 4'(k) : tag[k] + x[3:0] + 4'(k);
  assign mp_0_tag = tag[0];
  assign mp_1_tag = tag[1];
  assign mp_2_tag = tag[2];
  assign mp_3_tag = tag[3];
  assign y = acc[0][7:0] ^ acc[1][15:8] ^ acc[2][7:0] ^ acc[3][15:8];
endmodule
