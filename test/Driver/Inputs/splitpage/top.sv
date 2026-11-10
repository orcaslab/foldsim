// A parent's wide outputs to its children (csf-mailbox-hbm-split): four Par of
// eight Kid each. A Par register `cw` goes to every child rotated (input i: a
// port of the parent per child), a Par register `s0` to its first child's s;
// the other children's s is the previous sibling's registered o (an early
// chain). Child 0 then takes 16 bits of its parent, the others 8: with rows of
// 8 bits the stream sliced by reader splits in two. Registers start from their
// resets.
module Kid(input logic clk, input logic rst, input logic [7:0] id, input logic [7:0] i,
           input logic [7:0] s, output logic [7:0] o);
  logic [7:0] st;
  always_ff @(posedge clk)
    if (rst) begin
      st <= id;
      o <= ~id;
    end else begin
      st <= st * 8'd3 + (i ^ id);
      o <= s ^ {st[2:0], st[7:3]};
    end
endmodule

module Par(input logic clk, input logic rst, input logic [7:0] x, input logic [7:0] id,
           output logic [7:0] acc);
  logic [7:0] cw, s0, a;
  logic [7:0][7:0] k_s, k_o;
  for (genvar j = 0; j < 8; ++j) begin : g_kid
    Kid kid(.clk, .rst, .id(8'(id * 8 + j)), .i(8'({cw, cw} >> j)),
            .s(k_s[j]), .o(k_o[j]));
    if (j == 0) begin : g_first
      assign k_s[j] = s0;
    end else begin : g_next
      assign k_s[j] = k_o[j - 1];
    end
  end
  always_ff @(posedge clk)
    if (rst) begin
      cw <= id;
      s0 <= ~id;
      a <= 8'h0;
    end else begin
      cw <= cw + x + k_o[7];
      s0 <= s0 ^ {cw[6:0], cw[7]};
      a <= {a[6:0], a[7]} ^ k_o[7] ^ k_o[3];
    end
  assign acc = a;
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [3:0][7:0] p_acc;
  for (genvar k = 0; k < 4; ++k) begin : g_par
    Par par(.clk, .rst, .x(x ^ 8'(k)), .id(8'(k)), .acc(p_acc[k]));
  end
  logic [7:0] q;
  always_ff @(posedge clk)
    if (rst) q <= 8'h0;
    else q <= p_acc[0] ^ p_acc[1] ^ p_acc[2] ^ p_acc[3];
  assign y = q;
endmodule
