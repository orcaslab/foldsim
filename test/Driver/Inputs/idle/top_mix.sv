// Driver/Inputs/idle: eight Nodes (nodes.sv) behind a pulse source, then eight SNodes
// (stores.sv), SNode k reading Node 7 - k, for pp-idle-skip across SLR groups: with the
// two classes in different groups, each SNode reads a Node's clean bits across them. No
// reset input.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] s, n0, n1, n2, n3, n4, n5, n6, n7, m0, m1, m2, m3, m4, m5, m6, m7;
  Src src(.clk, .rst(1'b0), .x, .q(s));
  Node u0(.clk, .a(s), .b(8'h5a), .q(n0));
  Node u1(.clk, .a(n0), .b(8'h33), .q(n1));
  Node u2(.clk, .a(n1), .b(n0), .q(n2));
  Node u3(.clk, .a(n2), .b(n1), .q(n3));
  Node u4(.clk, .a(n3), .b(n2), .q(n4));
  Node u5(.clk, .a(n4), .b(n3), .q(n5));
  Node u6(.clk, .a(n5), .b(n4), .q(n6));
  Node u7(.clk, .a(n6), .b(n5), .q(n7));
  SNode s0(.clk, .a(n7), .q(m0));
  SNode s1(.clk, .a(n6), .q(m1));
  SNode s2(.clk, .a(n5), .q(m2));
  SNode s3(.clk, .a(n4), .q(m3));
  SNode s4(.clk, .a(n3), .q(m4));
  SNode s5(.clk, .a(n2), .q(m5));
  SNode s6(.clk, .a(n1), .q(m6));
  SNode s7(.clk, .a(n0), .q(m7));
  Sink k(.clk, .d(m7), .e(n4), .y);
endmodule
