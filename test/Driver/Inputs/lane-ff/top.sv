// For foldsim-lane-scan over 32 Leaf lanes: a find-first over a packed
// array written procedurally (the front end's form of VX_find_first: node
// values stored into packed vectors and read back), and a procedural AND
// tree whose result is compared to all ones.
module VX_find_first #(
    parameter N       = 1,
    parameter DATAW   = 1,
    parameter REVERSE = 0 
) (
    input  wire [N-1:0][DATAW-1:0] data_in,
    input  wire [N-1:0]            valid_in,
    output wire [DATAW-1:0]        data_out,
    output wire                    valid_out
);
    localparam LOGN = $clog2(N);
    localparam TL   = (1 << LOGN) - 1;
    localparam TN   = (1 << (LOGN+1)) - 1;
    reg             valid_r;
    reg [DATAW-1:0] data_r;
    
    
    always @(*) begin : g_scan
        reg [TN-1:0]       s_n;
        reg [TN*DATAW-1:0] d_n;
        s_n = '0;
        d_n = '0;
        for (integer i = 0; i < N; ++i) begin
            s_n[TL+i] = REVERSE ? valid_in[N-1-i] : valid_in[i];
            d_n[(TL+i)*DATAW +: DATAW] = REVERSE ? data_in[N-1-i] : data_in[i];
        end
        for (integer k = TL-1; k >= 0; --k) begin
            s_n[k] = s_n[2*k+2] | s_n[2*k+1];
            d_n[k*DATAW +: DATAW] = s_n[2*k+1] ? d_n[(2*k+1)*DATAW +: DATAW] : d_n[(2*k+2)*DATAW +: DATAW];
        end
        valid_r = s_n[0];
        data_r  = d_n[0 +: DATAW];
    end
    assign valid_out = valid_r;
    assign data_out  = data_r;
endmodule
module and_tree #(parameter N = 1) (input wire [N-1:0] in, output wire all);
  localparam TL = (1 << $clog2(N)) - 1, TN = 2 * TL + 1;
  reg r;
  always @(*) begin : g
    reg [TN-1:0] s;
    s = '1;
    for (integer i = 0; i < N; ++i) s[TL+i] = in[i];
    for (integer k = TL-1; k >= 0; --k) s[k] = s[2*k+1] & s[2*k+2];
    r = s[0];
  end
  assign all = r;
endmodule
module Leaf(input logic clk, input logic rst, input logic [7:0] x, input logic [7:0] K,
            output logic v, output logic a, output logic [7:0] d);
  logic [7:0] s;
  always_ff @(posedge clk) s <= rst ? K : s * 8'd5 + x + K;
  assign v = s[0] & s[3];
  assign a = s[1] | s[2] | s[5];
  assign d = s ^ x;
endmodule
module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  localparam N = 32;
  logic [N-1:0] valid, any; logic [N-1:0][7:0] data; logic [N-1:0][4:0] idx;
  for (genvar j = 0; j < N; ++j) begin : gi assign idx[j] = 5'(j); end
  for (genvar j = 0; j < N; ++j) begin : g
    Leaf u(.clk, .rst, .x(x ^ 8'(j)), .K(8'(3*j+1)), .v(valid[j]), .a(any[j]), .d(data[j]));
  end
  logic [4:0] f; logic fv, all;
  VX_find_first #(.N(N), .DATAW(5), .REVERSE(0)) ff(.data_in(idx), .valid_in(valid), .data_out(f), .valid_out(fv));
  and_tree #(.N(N)) at(.in(any), .all(all));
  logic [7:0] yr;
  always_ff @(posedge clk) yr <= {fv, all, 1'b0, f};
  assign y = yr;
endmodule
