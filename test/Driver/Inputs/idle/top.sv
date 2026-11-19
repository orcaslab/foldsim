// Driver/Inputs/idle: Top without a reset input (nodes.sv).
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] s, n0, n1, n2, n3, n4, n5, n6, n7;
  Src src(.clk, .rst(1'b0), .x, .q(s));
  Node u0(.clk, .a(s), .b(8'h5a), .q(n0));
  Node u1(.clk, .a(n0), .b(8'h33), .q(n1));
  Node u2(.clk, .a(n1), .b(n0), .q(n2));
  Node u3(.clk, .a(n2), .b(n1), .q(n3));
  Node u4(.clk, .a(n3), .b(n2), .q(n4));
  Node u5(.clk, .a(n4), .b(n3), .q(n5));
  Node u6(.clk, .a(n5), .b(n4), .q(n6));
  Node u7(.clk, .a(n6), .b(n5), .q(n7));
  Sink k(.clk, .d(n7), .e(n4), .y);
endmodule
