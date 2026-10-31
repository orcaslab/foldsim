// A bus broadcast: the top's registered request goes, the same value, to
// each of four lanes, which also see their own select; each lane keeps a
// mixing accumulator. The top's model then has four outputs driven by one
// value, which shared links read as one. The lanes' accumulators come back
// to the top, read by one input each.
module Lane(input logic clk, input logic rst, input logic [31:0] bus, input logic sel,
            input logic [7:0] peer, output logic [15:0] acc);
  always_ff @(posedge clk)
    if (rst) acc <= 0;
    else if (sel) acc <= acc + bus[15:0] ^ bus[31:16] ^ {8'h0, peer};
    else acc <= {acc[14:0], acc[15]} ^ bus[15:0];
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [31:0] bus;
  logic [3:0] sel;
  logic [3:0][15:0] acc;
  logic [15:0] peers;
  for (genvar i = 0; i < 4; ++i) begin : g_lane
    Lane lane(.clk, .rst, .bus, .sel(sel[i]), .peer(peers[8*(i/2) +: 8]), .acc(acc[i]));
  end
  always_ff @(posedge clk)
    peers <= rst ? 16'h0 : {~x ^ acc[0][15:8], x + acc[2][7:0]};
  always_ff @(posedge clk)
    if (rst) begin
      bus <= 0;
      sel <= 0;
    end else begin
      bus <= {bus[23:0], x} ^ {16'h0, acc[0] ^ acc[3]};
      sel <= x[3:0] ^ acc[1][3:0];
    end
  assign y = acc[0][7:0] ^ acc[1][15:8] ^ acc[2][7:0] ^ acc[3][15:8];
endmodule
