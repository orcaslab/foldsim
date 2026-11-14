// Block RAM reads whose address registers sit across class boundaries in
// both directions and more than one level away (all of Inner, Mid, Ram and
// Leaf are classes kept as instances):
// - Leaf.ram is read at a function of Mid's output, itself Inner's register
//   through Mid (an output of a class nested in a class);
// - Ram.ram is read at its input `ra` (Leaf's register `rp`, reset
//   synchronously, plus Mid's output) combined with Ram's own register: an
//   input from the parent, whose next value the parent must supply, and
//   through it the nested output again.
module Inner(input logic clk, input logic reset, input logic [7:0] d,
             output logic [3:0] head);
  always_ff @(posedge clk)
    if (reset) head <= 4'd3;
    else if (d[1]) head <= head + {3'd0, d[2]} + 4'd1;
endmodule
module Mid(input logic clk, input logic reset, input logic [7:0] d,
           output logic [3:0] head);
  logic [3:0] h;
  Inner i(.clk, .reset, .d, .head(h));
  assign head = h ^ 4'd5;
endmodule
module Ram(input logic clk, input logic [7:0] d, input logic [3:0] ra,
           output logic [7:0] q);
  (* ram_style = "block" *) logic [7:0] ram [16];
  logic [3:0] tail;
  always_ff @(posedge clk)
    if (d[0]) begin ram[tail] <= d; tail <= tail + 4'd3; end
  assign q = ram[ra ^ {tail[1:0], 2'd0}];
endmodule
module Leaf(input logic clk, input logic reset, input logic [7:0] d,
            output logic [7:0] q);
  (* ram_style = "block" *) logic [7:0] ram [16];
  logic [3:0] head, tail, rp;
  logic [7:0] q2;
  Mid m(.clk, .reset, .d, .head);
  always_ff @(posedge clk) begin
    if (d[0]) begin ram[~tail] <= d; tail <= tail + 4'd1; end
    if (reset) rp <= 4'd9;
    else if (d[3]) rp <= rp + 4'd3;
  end
  Ram r(.clk, .d(d ^ 8'h3c), .ra(rp + head), .q(q2));
  assign q = ram[~head] ^ d ^ q2;
endmodule
module Top(input logic clk, input logic reset, input logic [7:0] x,
           output logic [7:0] y);
  logic [7:0] a, b;
  Leaf l0(.clk, .reset, .d(x), .q(a));
  Leaf l1(.clk, .reset, .d(a), .q(b));
  Leaf l2(.clk, .reset, .d(b), .q(y));
endmodule
