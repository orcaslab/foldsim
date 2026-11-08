// A table of 2^17 rows x 11 bits (1,441,792 bits) and one of 2^17 x 7 (917,504): the first past Vivado's variable
// limit (Synth 8-4556), the second within it; a memory of 2^20 x 8 bits (a RAM, exempt).
module big(input [16:0] i, output [10:0] a, output [6:0] b);
  wire [131071:0][10:0] t11 = '{default: 11'h0};
  wire [131071:0][6:0] t7 = '{default: 7'h0};
  reg [7:0] mem [0:1048575];
  assign a = t11[i];
  assign b = t7[i];
endmodule
