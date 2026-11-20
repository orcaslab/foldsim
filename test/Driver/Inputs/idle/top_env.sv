// Driver/Inputs/idle: eight nodes in a chain that each read the input `x`
// (nodes.sv), for pp-idle-skip's inputs of the netlist: a node is idle while
// `x` and its producer are unchanged (tb_pages.sv with STIM_X_SLOW holds a
// token several target cycles). Top without a reset input.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] n0, n1, n2, n3, n4, n5, n6, n7;
  Node u0(.clk, .a(x), .b(8'h5a), .q(n0));
  Node u1(.clk, .a(n0), .b(x), .q(n1));
  Node u2(.clk, .a(n1), .b(x), .q(n2));
  Node u3(.clk, .a(n2), .b(x), .q(n3));
  Node u4(.clk, .a(n3), .b(x), .q(n4));
  Node u5(.clk, .a(n4), .b(x), .q(n5));
  Node u6(.clk, .a(n5), .b(x), .q(n6));
  Node u7(.clk, .a(n6), .b(x), .q(n7));
  Sink k(.clk, .d(n7), .e(n4), .y);
endmodule
