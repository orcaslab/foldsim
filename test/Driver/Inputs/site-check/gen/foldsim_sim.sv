// A generated simulator's memories (foldsim_site_check.py test): a class in SLR group 1 (SLR0), the level.
module big_65536x32(input clk);
  (* ram_style = "ultra" *) reg [31:0] mem[0:65535];
endmodule
module bram_262144x9(input clk);
  (* ram_style = "block" *) reg [8:0] mem[0:262143];
endmodule
module small_4096x144(input clk);
  (* ram_style = "ultra" *) reg [143:0] mem[0:4095];
endmodule
module Leaf_static(input clk);
  big_65536x32 m0 (.clk(clk));
  bram_262144x9 m1 (.clk(clk));
endmodule
module Top_model_static(input clk);
  small_4096x144 m2 (.clk(clk));
endmodule
module foldsim_sim_slr1(input clk);
  Leaf_static Leaf_copy (.clk(clk));
endmodule
module foldsim_sim_slr0(input clk);
  Top_model_static Top_model_copy (.clk(clk));
endmodule
module foldsim_sim(input clk);
  foldsim_sim_slr0 g0 (.clk(clk));
  foldsim_sim_slr1 g1 (.clk(clk));
endmodule
