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
  logic [1:0] av;
  logic [1:0][16:0] ad;
  logic [1:0][1:0] as;
  assign rsp = {{mp_3_rdata, mp_3_rvalid}, {mp_2_rdata, mp_2_rvalid},
                {mp_1_rdata, mp_1_rvalid}, {mp_0_rdata, mp_0_rvalid}};
  Arb a0(.clk, .rst, .id(2'd0), .ready_out(x[0] ^ mp_0_ack ^ mp_3_ack),
         .valid_in({mp_3_rvalid, mp_2_rvalid, mp_1_rvalid, mp_0_rvalid}), .data_in(rsp),
         .valid_out(av[0]), .data_out(ad[0]), .sel_out(as[0]));
  Arb a1(.clk, .rst, .id(2'd2), .ready_out(x[1] ^ mp_1_ack ^ ad[0][3] ^ as[0][1]),
         .valid_in({mp_3_rvalid, mp_2_rvalid, mp_1_rvalid, mp_0_rvalid}), .data_in(rsp),
         .valid_out(av[1]), .data_out(ad[1]), .sel_out(as[1]));
  Unit u0(.clk, .rst, .seed(8'd1), .x(x ^ ad[0][8:1]), .rvalid(mp_0_rvalid), .rdata(mp_0_rdata), .avalid(mp_0_avalid),
          .addr(mp_0_addr), .data(mp_0_data), .ack(mp_0_ack), .acc(acc[0]));
  Unit u1(.clk, .rst, .seed(8'd38), .x(x ^ ad[1][16:9]), .rvalid(mp_1_rvalid), .rdata(mp_1_rdata), .avalid(mp_1_avalid),
          .addr(mp_1_addr), .data(mp_1_data), .ack(mp_1_ack), .acc(acc[1]));
  Unit u2(.clk, .rst, .seed(8'd75), .x(x ^ {as[0], as[1], ad[0][4:1]}), .rvalid(mp_2_rvalid), .rdata(mp_2_rdata), .avalid(mp_2_avalid),
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
  assign y = acc[0][7:0] ^ acc[1][15:8] ^ acc[2][7:0] ^ acc[3][15:8] ^ ad[0][16:9] ^ ad[1][8:1] ^
             {av, as[0], as[1], ad[0][0], ad[1][0]};
endmodule
