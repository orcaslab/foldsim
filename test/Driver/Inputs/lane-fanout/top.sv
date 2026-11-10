// For foldsim-lane-scan over 32 Leaf lanes: a 1 -> N fan-out arbiter (a
// round-robin over the ready outputs from a pointer, the last output's
// compare dropped as always true; two priority encoders whose procedural
// find-first trees the canonicalizer merges, their compares rewritten per
// lane into several forms; a lock), out_valid[o] = valid && sel == o and
// ready = out_ready[sel].
module find_first #(
    parameter N       = 1,
    parameter DATAW   = 1,
    parameter REVERSE = 0 
) (
    input  wire [N-1:0][DATAW-1:0] data_in,
    input  wire [N-1:0]            valid_in,
    output wire [DATAW-1:0]        data_out,
    output wire                    valid_out
);
    localparam LOGN = $clog2(N);
    localparam TL   = (1 << LOGN) - 1;
    localparam TN   = (1 << (LOGN+1)) - 1;
    reg             valid_r;
    reg [DATAW-1:0] data_r;
    
    
    always @(*) begin : g_scan
        reg [TN-1:0]       s_n;
        reg [TN*DATAW-1:0] d_n;
        s_n = '0;
        d_n = '0;
        for (integer i = 0; i < N; ++i) begin
            s_n[TL+i] = REVERSE ? valid_in[N-1-i] : valid_in[i];
            d_n[(TL+i)*DATAW +: DATAW] = REVERSE ? data_in[N-1-i] : data_in[i];
        end
        for (integer k = TL-1; k >= 0; --k) begin
            s_n[k] = s_n[2*k+2] | s_n[2*k+1];
            d_n[k*DATAW +: DATAW] = s_n[2*k+1] ? d_n[(2*k+1)*DATAW +: DATAW] : d_n[(2*k+2)*DATAW +: DATAW];
        end
        valid_r = s_n[0];
        data_r  = d_n[0 +: DATAW];
    end
    assign valid_out = valid_r;
    assign data_out  = data_r;
endmodule
module lzc #(
    parameter N       = 2,
    parameter REVERSE = 0,  
    parameter LOGN    = (((N) > 1) ? $clog2(N) : 1)
) (
    input  wire [N-1:0]    data_in,
    output wire [LOGN-1:0] data_out,
    output wire            valid_out
);
    if (N == 1) begin : g_passthru
        
        assign data_out  = '0;
        assign valid_out = data_in;
    end else begin : g_lzc
        wire [N-1:0][LOGN-1:0] indices;
        for (genvar i = 0; i < N; ++i) begin : g_indices
            assign indices[i] = REVERSE ? LOGN'(i) : LOGN'(N-1-i);
        end
        find_first #(
            .N       (N),
            .DATAW   (LOGN),
            .REVERSE (!REVERSE)
        ) find_first (
            .valid_in  (data_in),
            .data_in   (indices),
            .data_out  (data_out),
            .valid_out (valid_out)
        );
    end
endmodule
module prio_enc #(
    parameter N       = 1,
    parameter REVERSE = 0, 
    parameter MODEL   = 1,
    parameter LN      = (((N) > 1) ? $clog2(N) : 1)
) (
    input  wire [N-1:0]  data_in,
    output wire [N-1:0]  onehot_out,
    output wire [LN-1:0] index_out,
    output wire          valid_out
);
    if (REVERSE) begin : g_msb
        if (N == 1) begin : g_n1
            assign onehot_out = data_in;
            assign index_out  = '0;
            assign valid_out  = data_in;
        end else if (N == 2) begin : g_n2
            assign onehot_out = {data_in[1], data_in[0] & ~data_in[1]};
            assign index_out  = data_in[1];
            assign valid_out  = (| data_in);
        end else if (MODEL != 0) begin : g_model1
            
            wire [N-1:0] rev_in  = {<<{data_in}};
            wire [N-1:0] rev_low = rev_in & -rev_in;
            assign onehot_out = {<<{rev_low}};
            wire [N-1:0][LN-1:0] indices;
            for (genvar i = 0; i < N; ++i) begin : g_indices
                assign indices[i] = LN'(i);
            end
            find_first #(
                .N (N),
                .DATAW (LN),
                .REVERSE (1)
            ) find_first (
                .valid_in  (data_in),
                .data_in   (indices),
                .data_out  (index_out),
                .valid_out (valid_out)
            );
        end else begin : g_model0
            reg [LN-1:0] index_w;
            reg [N-1:0]  onehot_w;
            always @(*) begin
                index_w  = 'x;
                onehot_w = 'x;
                for (integer i = 0; i < N-1; ++i) begin
                    if (data_in[i]) begin
                        index_w  = LN'(i);
                        onehot_w = N'(1) << i;
                    end
                end
            end
            assign index_out  = index_w;
            assign onehot_out = onehot_w;
            assign valid_out  = (| data_in);
        end
    end else begin: g_lsb
        if (N == 1) begin : g_n1
            assign onehot_out = data_in;
            assign index_out  = '0;
            assign valid_out  = data_in;
        end else if (N == 2) begin : g_n2
            assign onehot_out = {data_in[1] && ~data_in[0], data_in[0]};
            assign index_out  = ~data_in[0];
            assign valid_out  = (| data_in);
        end else if (MODEL == 1) begin : g_model1
            
            assign onehot_out = data_in & -data_in;
            lzc #(
                .N       (N),
                .REVERSE (1)
            ) lzc (
                .data_in   (data_in),
                .data_out  (index_out),
                .valid_out (valid_out)
            );
        end else if (MODEL == 2) begin : g_model2
            wire [N-1:0] scan_lo;
            VX_scan #(
                .N  (N),
                .OP ("|")
            ) scan (
                .data_in  (data_in),
                .data_out (scan_lo)
            );
            assign onehot_out = scan_lo & {(~scan_lo[N-2:0]), 1'b1};
            lzc #(
                .N       (N),
                .REVERSE (1)
            ) lzc (
                .data_in  (data_in),
                .data_out (index_out),
                .valid_out(valid_out)
            );
        end else if (MODEL == 3) begin : g_model3
            assign onehot_out = data_in & -data_in;
            lzc #(
                .N       (N),
                .REVERSE (1)
            ) lzc (
                .data_in   (data_in),
                .data_out  (index_out),
                .valid_out (valid_out)
            );
        end else begin : g_model0
            reg [LN-1:0] index_w;
            reg [N-1:0]  onehot_w;
            always @(*) begin
                index_w  = 'x;
                onehot_w = 'x;
                for (integer i = N-1; i >= 0; --i) begin
                    if (data_in[i]) begin
                        index_w  = LN'(i);
                        onehot_w = N'(1) << i;
                    end
                end
            end
            assign index_out  = index_w;
            assign onehot_out = onehot_w;
            assign valid_out  = (| data_in);
        end
    end
endmodule

module Leaf(input logic clk, input logic rst, input logic [7:0] K, input logic v,
            input logic [7:0] d, output logic ready, output logic [7:0] q);
  logic [7:0] s;
  always_ff @(posedge clk) s <= rst ? K : (v && ready) ? s ^ d : s * 8'd5 + K;
  assign ready = s[2] | s[6];
  assign q = s;
endmodule
module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  localparam N = 32, SEL_O = 5;
  logic [N-1:0] out_ready, out_valid; logic [N-1:0][7:0] q;
  wire m_valid = x[0] | x[3];
  wire [7:0] m_data = x;
  for (genvar o = 0; o < N; ++o) begin : g
    Leaf u(.clk, .rst, .K(8'(5*o+3)), .v(out_valid[o]), .d(m_data), .ready(out_ready[o]), .q(q[o]));
  end
  reg lock_v; reg [SEL_O-1:0] lock_sel, rr_ptr;
  wire [SEL_O-1:0] dest_sel = m_data[6:2];
  wire [SEL_O-1:0] rr_base = rr_ptr;
  wire [N-1:0] rr_upper;
  for (genvar o = 0; o < N; ++o) begin : g_rr_upper
    if (o == (1 << SEL_O) - 1) begin : g_last
      assign rr_upper[o] = out_ready[o];
    end else begin : g_cmp
      assign rr_upper[o] = out_ready[o] && (rr_base <= SEL_O'(o));
    end
  end
  wire [SEL_O-1:0] rr_upper_idx, rr_any_idx;
  wire rr_upper_valid, rr_any_valid;
  prio_enc #(.N(N)) rr_upper_pe (.data_in(rr_upper), .onehot_out(), .index_out(rr_upper_idx), .valid_out(rr_upper_valid));
  prio_enc #(.N(N)) rr_any_pe (.data_in(out_ready), .onehot_out(), .index_out(rr_any_idx), .valid_out(rr_any_valid));
  wire [SEL_O-1:0] rr_sel = rr_upper_valid ? rr_upper_idx : (rr_any_valid ? rr_any_idx : rr_ptr);
  wire [SEL_O-1:0] sel_new = m_data[7] ? dest_sel : rr_sel;
  wire [SEL_O-1:0] sel = lock_v ? lock_sel : sel_new;
  for (genvar o = 0; o < N; ++o) begin : g_sel
    assign out_valid[o] = m_valid && (sel == SEL_O'(o));
  end
  wire m_ready = out_ready[sel];
  wire m_fire = m_valid && m_ready;
  wire [SEL_O-1:0] rr_next = (sel == SEL_O'(N - 1)) ? '0 : (sel + SEL_O'(1));
  always @(posedge clk) begin
    if (rst) begin lock_v <= 0; lock_sel <= 0; rr_ptr <= 0; end
    else if (m_fire) begin
      lock_v <= ~m_data[1]; lock_sel <= sel;
      if (m_data[1]) rr_ptr <= rr_next;
    end
  end
  logic [7:0] yr;
  always_ff @(posedge clk) yr <= {m_ready, lock_v, 1'b0, sel} ^ q[x[4:0]];
  assign y = yr;
endmodule
