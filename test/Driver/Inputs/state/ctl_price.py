# ctl_price.py <the foldsim driver, beside the modules> <scratch dir>: the kernel's controllers priced routed
# (foldsim_storage ctl_cost, calibrated on the routed vtA-vtG / fsd64h / B4 images), the platform's way into device
# memory (platform_cost: the masters' shares and SLASH's sc_mem tree), and a generation's controllers as built
# (foldsim_degree read_built, vmem_controllers / vmem_bram with --vmem-group copy).
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_storage as st
import foldsim_degree as fd
r = lambda c: ' '.join(f'{k} {v:.0f}' for k, v in c.items())
# routed measurements: B4's L2-data controller 2,826 + 1,169, 16 RAMB36; vt's grouped node controller 2,472 + 551,
# 19; a page engine 1,259-1,391 + 1,356-1,381, none; a 64-bit stream 1,205-1,270 + 363; fsd64h's 4096-bit row
# 4,860-4,891 + 8,784, 0.5.
print('B4 vmem:', r(st.ctl_cost('vmem', 512, 512, 1, 4 << 11)))
print('vt group:', r(st.ctl_cost('vmem', 64, 256, 2, 8 << 13)))
print('engine:', r(st.ctl_cost('engine', 512)))
print('stream:', r(st.ctl_cost('link', 64)))
print('wide:', r(st.ctl_cost('vmem', 4096, 0, 1, 16 << 4)))
# The known-zero bits: 2,368-deep banks stay distributed, 4,096-deep ones take a RAMB36 each.
print('kz 37888:', st.ctl_cost('vmem', 80, 0, 1, kz=37888)['bram'], 'kz 65536:', st.ctl_cost('vmem', 80, 0, 1, kz=65536)['bram'])
# Masters: vtC (1 memory port, 11 controllers: 12 masters, 4 sc_mem of 2 inputs, 7,407 LUTs each routed);
# fsd64h (1 port, 40 controllers on 31 masters: 31 shares, 8 sc_mem of 4); a 16-port target with no controller
# already has 8 sc_mem of 2; 32 ports and 60 controllers: on the ports' masters.
for p, n, m in ((1, 11, None), (1, 40, 31), (16, 0, None), (8, 0, None), (32, 60, None)):
    print(f'platform {p} {n} {m}:', r(st.platform_cost(p, n, m)))
# As built: a generation's json (two grouped virtual memories of a copy, a stream, an engine), FORWARD 0.
d = sys.argv[2]
os.makedirs(d, exist_ok=True)
g = lambda **k: dict(dict(size=1, inst_width=3, instances=8, row_width=13, data_width=64, mask_width=1,
                          known_zero=1, kz_shift=0, cache_rows=0), **k)
json.dump(dict(vmem_groups=[g(**{'class': 'node', 'size': 2, 'cache_rows': 256}),
                            g(**{'class': '__link0', 'instances': 1, 'row_width': 3, 'data_width': 256, 'known_zero': 0}),
                            g(**{'class': '__pages0', 'instances': 1, 'row_width': 5, 'data_width': 512, 'known_zero': 0})]),
          open(d + '/foldsim_sim.json', 'w'))
open(d + '/foldsim_sim.vh', 'w').write('`define FOLDSIM_VMEM_FORWARD 0\n')
open(d + '/foldsim_kernel.vh', 'w').write('`define FOLDSIM_MEM_PORTS 1\n')
b = fd.read_built(d + '/foldsim_sim.json')
print(f"built: {len(b['controllers'])} {sorted(b['counts'].items())} forward {b['forward']} ports {b['mem_ports']} "
      f"logic {b['logic']:.0f} lutram {b['lutram']:.0f} bram {b['bram']:.0f}")
# A copy's virtual memories as controllers: one a memory, or one a row size with --vmem-group copy.
mems = [('a', 4096, 64, 1, 1, 1, None, 8, True), ('b', 1024, 32, 1, 1, 1, None, 4, True),
        ('c', 512, 512, 1, 1, 1, None, 64, True)]
print('controllers:', len(fd.vmem_controllers(mems, 8)), len(fd.vmem_controllers(mems, 8, 'copy', 8)),
      [(v['data_width'], v['row_width'], v['members']) for v in fd.vmem_controllers(mems, 8, 'copy', 8)])
