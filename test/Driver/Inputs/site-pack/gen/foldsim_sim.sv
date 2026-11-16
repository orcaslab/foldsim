// A generated simulator's memories (foldsim_site_check.py packing test): class A in SLR group 1 (SLR0), two copies of
// class B in group 0 (SLR1).
module mem_a_65536x72(input clk);
  (* ram_style = "ultra" *) reg [71:0] mem[0:65535];
endmodule
module mem_c_16384x72(input clk);
  (* ram_style = "ultra", cascade_height = 2 *) reg [71:0] mem[0:16383];
endmodule
module mem_b_32768x72(input clk);
  (* ram_style = "ultra" *) reg [71:0] mem[0:32767];
endmodule
module A_static(input clk);
  mem_a_65536x72 m0 (.clk(clk));
  mem_c_16384x72 m1 (.clk(clk));
endmodule
module B_inner(input clk);
  mem_b_32768x72 m2 (.clk(clk));
endmodule
module B_static(input clk);
  B_inner i (.clk(clk));
endmodule
module B_copy1_static(input clk);
  B_inner i (.clk(clk));
endmodule
module foldsim_sim_slr1(input clk);
  A_static A_copy (.clk(clk));
endmodule
module foldsim_sim_slr0(input clk);
  B_static B_copy (.clk(clk));
  B_copy1_static B_copy1 (.clk(clk));
endmodule
module foldsim_sim(input clk);
  foldsim_sim_slr0 g0 (.clk(clk));
  foldsim_sim_slr1 g1 (.clk(clk));
endmodule
