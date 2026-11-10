// For foldsim-lane-scan over 32 Leaf lanes: a registered vector of the
// lanes' fire bits beside two parent bits (added in the parent), its
// population count by an adder tree whose widths grow a level at a time (no
// add can overflow), and a sum of 4-bit lane values the same way.
module popcount #(parameter N = 1, parameter M = $clog2(N+1)) (
    input  wire [N-1:0] data_in,
    output wire [M-1:0] data_out
);
    localparam PN = 1 << $clog2(N);
    localparam LOGPN = $clog2(PN);
    wire [M-1:0] tmp [LOGPN-1:0][PN-1:0];
    for (genvar j = 0; j < LOGPN; ++j) begin : g_lvl
        localparam D = j + 1;
        localparam Q = (D < LOGPN) ? (D + 1) : M;
        for (genvar i = 0; i < (1 << (LOGPN-j-1)); ++i) begin : g_node
            localparam l = i * 2;
            localparam r = i * 2 + 1;
            wire [Q-1:0] res;
            if (j == 0) begin : g_leaf
                if (r < N) begin : g_two
                    assign res = data_in[l] + data_in[r];
                end else if (l < N) begin : g_one
                    assign res = 2'(data_in[l]);
                end else begin : g_none
                    assign res = 2'b0;
                end
            end else begin : g_inner
                assign res = D'(tmp[j-1][l]) + D'(tmp[j-1][r]);
            end
            assign tmp[j][i] = M'(res);
        end
    end
    assign data_out = tmp[LOGPN-1][0];
endmodule
module Leaf(input logic clk, input logic rst, input logic [7:0] x, input logic [7:0] K,
            output logic v, output logic [3:0] d);
  logic [7:0] s;
  always_ff @(posedge clk) s <= rst ? K : s * 8'd5 + x + K;
  assign v = s[0] ^ s[5];
  assign d = s[6:3];
endmodule
module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  localparam N = 32;
  logic [N-1:0] v; logic [N-1:0][3:0] d;
  for (genvar j = 0; j < N; ++j) begin : g
    Leaf u(.clk, .rst, .x(x ^ 8'(j)), .K(8'(3*j+1)), .v(v[j]), .d(d[j]));
  end
  // The fire bits in a register beside two parent bits.
  logic [N-1:0] fire;
  for (genvar j = 0; j < N; ++j) begin : g_fire
    assign fire[j] = v[j] & ~x[j % 8];
  end
  logic [N+1:0] fire_r;
  always_ff @(posedge clk) fire_r <= rst ? '0 : {x[6:5], fire};
  logic [5:0] cnt;
  popcount #(.N(N+2)) pc (.data_in(fire_r), .data_out(cnt));
  // A sum of the lanes' 4-bit values: a tree of 9-bit adds.
  logic [8:0] sum;
  always_comb begin
    sum = '0;
    for (int j = 0; j < N; ++j) sum = sum + 9'(d[j]);
  end
  logic [7:0] yr;
  always_ff @(posedge clk) yr <= {cnt[5:3] ^ cnt[2:0], 5'(sum) ^ 5'(sum >> 5)} ^ {2'b0, cnt};
  assign y = yr;
endmodule
