# kernel.py <the foldsim tools dir> <scratch dir>: foldsim_sll_check.kernel_resources on small netlists: a folded
# memory endpoint (port bundles) with a page engine, a group of two memories and a single one; four unfolded
# memory ports and no controller; the 11-07 flat term (a region without the kernel fit).
import json, os, sys
sys.path.insert(0, sys.argv[1])
import foldsim_sll_check as sc, foldsim_storage as st
region = json.load(open(os.path.join(sys.argv[1], 'foldsim_v80_slr.json')))
d = sys.argv[2]
def vm(cls, dw):
    return {'class': cls, 'data_width': dw, 'row_width': 8, 'inst_width': 4, 'cache_rows': 0, 'instances': 1}
cases = {
    'fold': {'port_bundles': [{'name': 'm_axi_mem_'}], 'ports': [],
             'vmem': [vm('__pages0', 512), vm('A', 512), vm('A', 512), vm('B', 64)],
             'vmem_groups': [dict(vm('__pages0', 512), members=[0], size=1), dict(vm('A', 512), members=[1, 2], size=2),
                             dict(vm('B', 64), members=[3], size=1)]},
    'ports': {'ports': [{'name': f'm_axi_mem_{k}_awvalid', 'width': 1} for k in range(4)], 'vmem': []},
}
for name, j in cases.items():
    os.makedirs(os.path.join(d, name), exist_ok=True)
    json.dump(j, open(os.path.join(d, name, 'foldsim_sim.json'), 'w'))
    kr = sc.kernel_resources(os.path.join(d, name), region)
    print(name, {k: round(v) for k, v in kr['parts'].items()}, round(kr['lut']), round(kr['bram'], 1))
flat = dict(region, kernel={'lut': 155000, 'ff': 65000})
kr = sc.kernel_resources(os.path.join(d, 'fold'), flat)
print('flat', round(kr['lut']), round(kr['ff']))
