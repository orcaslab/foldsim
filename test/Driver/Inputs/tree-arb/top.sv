// A round-robin arbiter tree over eight requesters, as VX_stream_arb builds
// one past its fanout: slices of two with a buffered output, then a smaller
// tree of the slices' outputs, one arbiter at the bottom. The innermost
// arbiter feeds a bank whose ready depends on the request it is offered, so
// readies run combinationally from the bank through every level back to the
// requesters. Registers start from their initializers.
module Buf(input logic clk, input logic rst, input logic valid_in, input logic [7:0] data_in, output logic ready_in,
           output logic valid_out, output logic [7:0] data_out, input logic ready_out);
  logic v;
  logic [7:0] d;
  assign ready_in = !v || ready_out;
  always_ff @(posedge clk)
    if (rst) v <= 1'b0;
    else if (ready_in) begin
      v <= valid_in;
      d <= data_in;
    end
  assign valid_out = v;
  assign data_out = d;
endmodule

module Rr2(input logic clk, input logic rst, input logic [1:0] valid_in, input logic [1:0][7:0] data_in,
           output logic [1:0] ready_in, output logic valid_out, output logic [7:0] data_out,
           input logic ready_out);
  logic last;
  wire pick = valid_in[1] && (!valid_in[0] || !last);
  assign valid_out = |valid_in;
  assign data_out = data_in[pick];
  assign ready_in = {ready_out && pick, ready_out && !pick};
  always_ff @(posedge clk)
    if (rst) last <= 1'b0;
    else if (valid_out && ready_out) last <= pick;
endmodule

module Slice(input logic clk, input logic rst, input logic [1:0] valid_in, input logic [1:0][7:0] data_in,
             output logic [1:0] ready_in, output logic valid_out, output logic [7:0] data_out,
             input logic ready_out);
  logic v, r;
  logic [7:0] d;
  Rr2 rr(.clk, .rst, .valid_in, .data_in, .ready_in, .valid_out(v), .data_out(d), .ready_out(r));
  Buf out_buf(.clk, .rst, .valid_in(v), .data_in(d), .ready_in(r), .valid_out, .data_out, .ready_out);
endmodule

module Arb #(parameter N = 8) (input logic clk, input logic rst, input logic [N-1:0] valid_in,
                               input logic [N-1:0][7:0] data_in, output logic [N-1:0] ready_in,
                               output logic valid_out, output logic [7:0] data_out,
                               input logic ready_out);
  if (N > 2) begin : g_fanout
    logic [N/2-1:0] v, r;
    logic [N/2-1:0][7:0] d;
    for (genvar s = 0; s < N / 2; ++s) begin : g_slices
      Slice slice(.clk, .rst, .valid_in(valid_in[2*s +: 2]), .data_in(data_in[2*s +: 2]),
                  .ready_in(ready_in[2*s +: 2]), .valid_out(v[s]), .data_out(d[s]),
                  .ready_out(r[s]));
    end
    Arb #(N / 2) join_arb(.clk, .rst, .valid_in(v), .data_in(d), .ready_in(r), .valid_out,
                          .data_out, .ready_out);
  end else begin : g_arbiter
    Rr2 rr(.clk, .rst, .valid_in, .data_in, .ready_in, .valid_out, .data_out, .ready_out);
  end
endmodule

module Bank(input logic clk, input logic rst, input logic valid, input logic [7:0] data, output logic ready,
            output logic [7:0] acc);
  logic [7:0] a;
  logic t;
  assign ready = t || data[0];
  always_ff @(posedge clk)
    if (rst) begin
      t <= 1'b0;
      a <= 8'h0;
    end else begin
      t <= ~t;
      if (valid && ready) a <= a * 8'd3 + data;
    end
  assign acc = a;
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [7:0] req;
  logic [7:0] rdy;
  logic [7:0][7:0] data;
  for (genvar i = 0; i < 8; ++i) begin : g_req
    assign data[i] = x ^ 8'(i * 37);
  end
  logic v, r;
  logic [7:0] d;
  Arb #(8) arb(.clk, .rst, .valid_in(req), .data_in(data), .ready_in(rdy), .valid_out(v),
               .data_out(d), .ready_out(r));
  Bank bank(.clk, .rst, .valid(v), .data(d), .ready(r), .acc(y));
  always_ff @(posedge clk) req <= rst ? 8'h0 : (req & ~rdy) | x;
endmodule
