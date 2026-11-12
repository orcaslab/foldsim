# kernel.py <the foldsim tools dir> <scratch dir>: foldsim_sll_check.kernel_resources on small netlists: a folded
# memory endpoint (port bundles) with a page engine, a group of two memories and a single one; four unfolded
# memory ports and no controller; a memory and two static ports; the 11-07 flat term (a region without the kernel fit).
import json, os, sys
sys.path.insert(0, sys.argv[1])
import foldsim_sll_check as sc, foldsim_storage as st
region = json.load(open(os.path.join(sys.argv[1], 'foldsim_v80_slr.json')))
d = sys.argv[2]
print('scales', dict(sorted(region['kernel']['ctl_lut_scale'].items())), region['kernel']['ctl_bram_scale'])
def vm(cls, dw):
    return {'class': cls, 'data_width': dw, 'row_width': 8, 'inst_width': 4, 'cache_rows': 0, 'instances': 1}
cases = {
    'fold': {'port_bundles': [{'name': 'm_axi_mem_'}], 'ports': [],
             'vmem': [vm('__pages0', 512), vm('A', 512), vm('A', 512), vm('B', 64)],
             'vmem_groups': [dict(vm('__pages0', 512), members=[0], size=1), dict(vm('A', 512), members=[1, 2], size=2),
                             dict(vm('B', 64), members=[3], size=1)]},
    'ports': {'ports': [{'name': f'm_axi_mem_{k}_awvalid', 'width': 1} for k in range(4)], 'vmem': []},
    # Static ports (vport-engines) beside a memory: each a page engine's controller of its row width.
    'vports': {'port_bundles': [{'name': 'm_axi_mem_'}], 'ports': [], 'vmem': [vm('B', 64)],
               'vports': [dict(DW=432, OW=19), dict(DW=2720, OW=25)]},
}
for name, j in cases.items():
    os.makedirs(os.path.join(d, name), exist_ok=True)
    json.dump(j, open(os.path.join(d, name, 'foldsim_sim.json'), 'w'))
    kr = sc.kernel_resources(os.path.join(d, name), region)
    if name == 'vports':
        kr_v = kr
        b = st.ctl_cost('vmem', 64)
        kr_b = 1.6 * (b['logic'] + b['lutram'])
    print(name, {k: round(v) for k, v in kr['parts'].items()}, round(kr['lut']), round(kr['bram'], 1))
flat = dict(region, kernel={'lut': 155000, 'ff': 65000})
kr = sc.kernel_resources(os.path.join(d, 'fold'), flat)
print('flat', round(kr['lut']), round(kr['ff']))
# The static ports in the SLR check's controllers (vmem_controllers: after the memory, page engine p's members).
cs = sc.vmem_controllers(os.path.join(d, 'vports'), '    page engine 1 2720 bits, tag 10, row 22: C:pps0_rows link:D>E')
print('vport ctrls', [(c['name'], c['cls'], c['port_bits'], sorted(c['members'].items())) for c in cs])
e = st.ctl_cost('engine', 432)['logic'] + st.ctl_cost('engine', 432)['lutram'] + st.ctl_cost('engine', 2720)['logic'] + st.ctl_cost('engine', 2720)['lutram']
print('vport term = engine scale x ctl_cost', round(0.77 * e) == round(kr_v['parts']['3 controllers'] - kr_b))
