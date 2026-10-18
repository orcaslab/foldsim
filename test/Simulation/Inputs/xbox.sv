// The black box of the folding tests: q is f(d) of four cycles of clk with
// ce high before, f(d) = 7 d + 1.
module XBox(input clk, input ce, input [7:0] d, input d_valid,
            output [7:0] q, output q_valid);
  logic [7:0] p [4];
  always @(posedge clk)
    if (ce) begin
      p[0] <= 8'(d * 7 + 1);
      for (int i = 1; i < 4; i++) p[i] <= p[i - 1];
    end
  assign q = p[3];
  assign q_valid = d_valid;
endmodule
