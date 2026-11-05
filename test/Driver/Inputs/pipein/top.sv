// Per-lane pipeline registers in front of a lane class: each lane's operand is
// selected from a shared operand bank (a dynamic offset plus the lane's own
// constant) into a register of the top, and a second register of the top
// holds its value unless the lane's enable bit is set (a register reading
// itself); the first is 32 bits, its halves two lane inputs. Both registers and their selects reach one lane input alone, so
// `--absorb-exclusive Lane --absorb-registers` moves them into the lane.
// MUTANT (top_mut.sv): lane 2's select takes another bank entry.
module Lane(input logic clk, input logic rst, input logic [15:0] opnd, input logic [15:0] hi,
            input logic [15:0] held,
            output logic [15:0] acc);
  always_ff @(posedge clk)
    if (rst) acc <= 0;
    else acc <= {acc[14:0], acc[15]} ^ opnd + held - hi;
endmodule

module Top(input logic clk, input logic rst, input logic [7:0] x, output logic [7:0] y);
  logic [3:0][15:0] bank;
  logic [1:0] off;
  logic [3:0][31:0] pipe;
  logic [3:0][15:0] hold, acc;
  for (genvar i = 0; i < 4; ++i) begin : g_lane
`ifdef MUTANT
    localparam int K = (i == 2) ? 3 : i;
`else
    localparam int K = i;
`endif
    always_ff @(posedge clk) begin
      pipe[i] <= {bank[3 - K] + {8'h0, x}, x[i] ? bank[off + 2'(unsigned'(K))] : bank[K] ^ {8'h0, x}};
      if (rst) hold[i] <= 0;
      else if (x[4 + i]) hold[i] <= bank[K] + {x, x};
    end
    Lane lane(.clk, .rst, .opnd(pipe[i][15:0]), .hi(pipe[i][31:16]), .held(hold[i]), .acc(acc[i]));
  end
  always_ff @(posedge clk)
    if (rst) begin
      bank <= 0;
      off <= 0;
    end else begin
      bank <= {bank[2:0], {x, acc[0][7:0] ^ acc[3][15:8]}};
      off <= x[1:0] ^ acc[1][1:0];
    end
  assign y = acc[0][7:0] ^ acc[1][15:8] ^ acc[2][7:0] ^ acc[3][15:8];
endmodule
