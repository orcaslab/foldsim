#!/usr/bin/env python3
"""trace.py <vmem_ports.py dir> <foldsim_kernel.vh> <foldsim_sim.vh> [mutant]: an emulator AXI trace of S28 region
masters as the region controller maps lin addresses (vmem_ports.s28_pc: the address each master sends), every grain
of every region read once; a linear region's (PC_LOG 0) at the runtime's base 0x41_0000_0000 + lin. `mutant`: the SLR2
region's PC stride taken as 4 (the SLR0/1 map) instead of its own; `lowbase`: a linear region's base without its high word
(a controller that took the base before the runtime wrote VMEM_BASE_HI)."""
import re
import sys
sys.path.insert(0, sys.argv[1])
import vmem_ports as vp  # noqa: E402
kvh, svh = open(sys.argv[2]).read(), open(sys.argv[3]).read()
mutant = len(sys.argv) > 4 and sys.argv[4] == 'mutant'
lowbase = len(sys.argv) > 4 and sys.argv[4] == 'lowbase'
regions, _ = vp.s28_regions(sys.argv[3])
mp = {int(r): (int(p), int(m)) for p, r, m in re.findall(r'`f\((\d+), (\d+), (\d+)\)',
                                                        re.search(r'MASTER_REPEAT\(f,s\) (.*)', kvh).group(1))}
get = lambda r, k: int(re.search(rf'^`define FOLDSIM_S28_R{r}_{k} (\d+)', kvh, re.M).group(1))
port = {(r, m): p for p, (r, m) in ((p, rm) for p, rm in
                                    ((int(p), (int(r), int(m))) for p, r, m in re.findall(
                                        r'`f\((\d+), (\d+), (\d+)\)', re.search(r'MASTER_REPEAT\(f,s\) (.*)', kvh).group(1))))}
t = 0
for x in regions:
    r = x['r']
    b = dict(nm=get(r, 'NM'), pc_log=get(r, 'PC_LOG'), pc_first=get(r, 'PC_FIRST'), pc_stride=get(r, 'PC_STRIDE'),
             grain_log=get(r, 'GRAIN_LOG'))
    if mutant and x['slr'] == 'SLR2':
        b['pc_stride'] = 4
    for lin in range(x['lin_base'], x['lin_base'] + x['bytes'], 1 << b['grain_log']):
        m, pc, share = vp.s28_pc(b, lin)
        a = 0x40_0000_0000 + (pc << 30) + (512 << 20) + share if pc is not None else \
            (0 if lowbase else 0x41_0000_0000) + lin
        print(f'{t} {port[(r, m)]}:AR {a:x}')
        t += 1
