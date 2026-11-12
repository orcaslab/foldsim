#!/usr/bin/env python3
"""afu_masters.py <foldsim_afu.sv.in> <kernel.vh> <N> <out.sv>: the kernel top's sharing of the virtual-memory
masters, its own text (from `localparam VMEM_M` to the masters' wires) in a module that prints, for each master j,
the controllers it takes in order (the g_vmem_master loop's G and K), for vmem-slr-masters.test."""
import re, sys

src, vh, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
text = open(src).read()
m = re.search(r'^    localparam VMEM_M = `FOLDSIM_VMEM_MASTERS;\n.*?(?=^    wire\s+vmm_awvalid)', text, re.M | re.S)
if not m:
    sys.exit('afu_masters.py: no virtual-memory masters section in ' + src)
loop = m.group(0)
open(out, 'w').write(f'''`include "{vh}"
module tb;
    localparam VMEM_N = {n};
{loop}
    for (genvar j = 0; j < VMEM_M; ++j) begin : g_m
        localparam G = vmem_sharers(j);
        for (genvar i = 0; i < G; ++i) begin : g_c
            localparam K = vmem_sharer(j, i);
            initial #(j * 100 + i + 1) $display("master %0d sharer %0d: controller %0d of %0d", j, i, K, G);
        end
    end
    initial #10000 $finish;
endmodule
''')
