// The folded simulator of top.sv against Top itself: the native design runs
// first, then the simulator's y tokens are compared with its y, cycle by cycle.
module tb;
  localparam int N = 64;
  function automatic logic [7:0] X(int k);
    return 8'(k * 37 + 5) ^ 8'(k >> 2);
  endfunction

  logic nclk = 0, done = 0;
  logic [7:0] nx, ny;
  logic [7:0] expected [N];
  Top native(.clk(nclk), .x(nx), .y(ny));
  initial begin
    for (int k = 0; k < N; k++) begin
      nx = X(k);
      #1 expected[k] = ny;
      nclk = 1;
      #1 nclk = 0;
    end
    done = 1;
  end

  logic clk = 0, rst = 1;
  always #1 clk = ~clk;
  int n = 0, m = 0, bad = 0;
  logic x_ready, y_valid;
  logic [7:0] y_bits;
  foldsim_sim dut(.hostClock(clk), .hostReset(rst), .profile_finish(),
                  .x_ready, .x_valid(!rst && n < N), .x_bits(X(n)),
                  .y_ready(1'b1), .y_valid, .y_bits);
  initial begin
    wait (done);
    repeat (4) @(posedge clk);
    rst <= 0;
  end
  always @(posedge clk) if (!rst) begin
    if (n < N && x_ready)
      n <= n + 1;
    if (y_valid) begin
      if (y_bits !== expected[m]) begin
        bad = bad + 1;
        $display("y %0d = %h, native %h", m, y_bits, expected[m]);
      end
      m <= m + 1;
      if (m == N - 1) begin
        $display("tokens %0d mismatches %0d", N, bad);
        $finish;
      end
    end
  end
  initial begin
    #100000 $display("timeout at %0d tokens", m);
    $finish;
  end
endmodule
