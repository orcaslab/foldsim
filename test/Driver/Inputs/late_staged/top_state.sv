// Four leaves (leaf.sv) on the netlist's input, their late outputs mixed into
// the netlist's output in the same cycle: each written at its state pass (no
// ready pass), from the last stage when staged.
module Top(input logic clk, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] v [4];
  for (genvar i = 0; i < 4; i++) begin : g
    Leaf l(.clk, .d(x ^ 8'(i * 37)), .e(x), .q(v[i]));
  end
  assign y = v[0] ^ v[1] ^ v[2] ^ v[3];
endmodule
