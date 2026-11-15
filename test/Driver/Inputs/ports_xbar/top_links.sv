// top_links.sv (--index-class-inputs --index-class-links): as top.sv, with
// four more arbiters (one class with a0, a1) whose arrays are OTHER
// CLASSES' outputs, read through their links: a2 the units' {data, ack}
// (an early and a late output of four producer instances), a3 the same
// rotated (every instance its own array), a4 four bit ranges of registers
// of the top's own model (one producer instance), a5 a mix of a response,
// units' outputs and a constant.
// ---- top.sv's header:
// Driver/Inputs/ports with a response crossbar (--index-class-inputs): the
// four bundles' responses {rdata, rvalid} are an array that two port
// arbiters (`Arb`, one class) each take whole and read at a grant they
// compute in the same cycle (round robin from a pointer register, over the
// valid bits): `data_in[grant]`, through two constant slices (the low and the
// high pair) -- two reads of four elements. The arbiters' outputs feed the
// units' state and `y` (late: through the top's model).
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

module Arb(input logic clk, input logic rst, input logic [1:0] id, input logic ready_out,
           input logic [3:0] valid_in, input logic [3:0][16:0] data_in,
           output logic valid_out, output logic [16:0] data_out, output logic [1:0] sel_out);
  logic [1:0] ptr, grant;
  logic [1:0][16:0] lo, hi;
  always_comb begin
    grant = ptr;
    for (int i = 3; i >= 0; --i)
      if (valid_in[2'(ptr + 2'(i))])
        grant = 2'(ptr + 2'(i));
  end
  assign lo = data_in[1:0];
  assign hi = data_in[3:2];
  assign valid_out = |valid_in;
  assign data_out = grant[1] ? hi[grant[0]] : lo[grant[0]];
  assign sel_out = grant;
  always_ff @(posedge clk)
    if (rst) ptr <= id;
    else if (valid_out & ready_out) ptr <= grant + 2'd1;
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
  logic [3:0][16:0] rsp;
  logic [4:0] av;
  logic [4:0][16:0] ad;
  logic [4:0][1:0] as;
  logic [15:0] ud [4];
  logic ua [4], uk [4];
  logic [3:0][16:0] ua1, ua2, treg, mix;
  logic [16:0] ad5;
  always_ff @(posedge clk)
    for (int k = 0; k < 4; ++k)
      treg[k] <= rst ? 17'(k * 5) : {treg[k][15:0], treg[k][16]} ^ {9'b0, x} ^ ad[0];
  for (genvar k = 0; k < 4; ++k) begin : g_ua
    assign ua1[k] = {ud[k], uk[k]};
    assign ua2[k] = {ud[(k + 1) % 4], uk[(k + 3) % 4]};
  end
  assign mix = {17'h1abcd, {ud[2], uk[1]}, {mp_1_rdata, mp_1_rvalid}, {ud[0], ua[3]}};
  assign rsp = {{mp_3_rdata, mp_3_rvalid}, {mp_2_rdata, mp_2_rvalid},
                {mp_1_rdata, mp_1_rvalid}, {mp_0_rdata, mp_0_rvalid}};
  Arb a0(.clk, .rst, .id(2'd0), .ready_out(x[0]),
         .valid_in({mp_3_rvalid, mp_2_rvalid, mp_1_rvalid, mp_0_rvalid}), .data_in(rsp),
         .valid_out(av[0]), .data_out(ad[0]), .sel_out(as[0]));
  Arb a1(.clk, .rst, .id(2'd2), .ready_out(x[1]),
         .valid_in({mp_3_rvalid, mp_2_rvalid, mp_1_rvalid, mp_0_rvalid}), .data_in(rsp),
         .valid_out(av[1]), .data_out(ad[1]), .sel_out(as[1]));
  Arb a2(.clk, .rst, .id(2'd1), .ready_out(x[2]),
         .valid_in({ua[3], ua[2], ua[1], ua[0]}), .data_in(ua1),
         .valid_out(av[2]), .data_out(ad[2]), .sel_out(as[2]));
  Arb a3(.clk, .rst, .id(2'd3), .ready_out(x[3] ^ uk[0]),
         .valid_in({uk[3], ua[2], uk[1], ua[0]}), .data_in(ua2),
         .valid_out(av[3]), .data_out(ad[3]), .sel_out(as[3]));
  Arb a4(.clk, .rst, .id(2'd2), .ready_out(x[4]),
         .valid_in(treg[1][3:0] | x[7:4]), .data_in(treg),
         .valid_out(av[4]), .data_out(ad[4]), .sel_out(as[4]));
  Arb a5(.clk, .rst, .id(2'd0), .ready_out(x[5]),
         .valid_in({x[6], uk[1], mp_1_rvalid, ua[3]}), .data_in(mix),
         .valid_out(), .data_out(ad5), .sel_out());
  Unit u0(.clk, .rst, .seed(8'd1), .x(x ^ ad[0][8:1]), .rvalid(mp_0_rvalid), .rdata(mp_0_rdata), .avalid(ua[0]),
          .addr(mp_0_addr), .data(ud[0]), .ack(uk[0]), .acc(acc[0]));
  Unit u1(.clk, .rst, .seed(8'd38), .x(x ^ ad[1][16:9]), .rvalid(mp_1_rvalid), .rdata(mp_1_rdata), .avalid(ua[1]),
          .addr(mp_1_addr), .data(ud[1]), .ack(uk[1]), .acc(acc[1]));
  Unit u2(.clk, .rst, .seed(8'd75), .x(x ^ {as[0], as[1], ad[0][4:1]}), .rvalid(mp_2_rvalid), .rdata(mp_2_rdata), .avalid(ua[2]),
          .addr(mp_2_addr), .data(ud[2]), .ack(uk[2]), .acc(acc[2]));
  Unit u3(.clk, .rst, .seed(8'd112), .x(x ^ ad[2][16:9] ^ ad[4][8:1]), .rvalid(mp_3_rvalid), .rdata(mp_3_rdata), .avalid(ua[3]),
          .addr(mp_3_addr), .data(ud[3]), .ack(uk[3]), .acc(acc[3]));
  logic [3:0] tag [4];
  always_ff @(posedge clk)
    for (int k = 0; k < 4; ++k)
      tag[k] <= rst ? 4'(k) : tag[k] + x[3:0] + 4'(k);
  assign {mp_3_avalid, mp_2_avalid, mp_1_avalid, mp_0_avalid} = {ua[3], ua[2], ua[1], ua[0]};
  assign mp_0_data = ud[0]; assign mp_1_data = ud[1]; assign mp_2_data = ud[2]; assign mp_3_data = ud[3];
  assign mp_0_ack = uk[0]; assign mp_1_ack = uk[1]; assign mp_2_ack = uk[2]; assign mp_3_ack = uk[3];
  assign mp_0_tag = tag[0];
  assign mp_1_tag = tag[1];
  assign mp_2_tag = tag[2];
  assign mp_3_tag = tag[3];
  assign y = acc[0][7:0] ^ acc[1][15:8] ^ acc[2][7:0] ^ acc[3][15:8] ^ ad[0][16:9] ^ ad[1][8:1] ^
             {av[1:0], as[0], as[1], ad[0][0], ad[1][0]} ^ ad[2][8:1] ^ ad[3][16:9] ^ ad[4][7:0] ^
             {as[2], as[3], as[4], av[4:3]} ^ ad5[9:2] ^ {7'b0, ^treg[2]};
endmodule
