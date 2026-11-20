// Driver/Inputs/idle: Top without a reset input (stores.sv, nodes.sv's Src
// and Sink).
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] s, n0, n1, n2, n3, n4, n5, n6, n7;
  Src src(.clk, .rst(1'b0), .x, .q(s));
  SNode u0(.clk, .a(s), .q(n0));
  SNode u1(.clk, .a(n0), .q(n1));
  SNode u2(.clk, .a(n1), .q(n2));
  SNode u3(.clk, .a(n2), .q(n3));
  SNode u4(.clk, .a(n3), .q(n4));
  SNode u5(.clk, .a(n4), .q(n5));
  SNode u6(.clk, .a(n5), .q(n6));
  SNode u7(.clk, .a(n6), .q(n7));
  Sink k(.clk, .d(n7), .e(n4), .y);
endmodule
