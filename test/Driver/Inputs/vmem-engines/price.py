# price.py <the foldsim driver, beside the modules> <scratch dir>: the virtual-memory controllers' SLLs per boundary
# as foldsim_sll_check.vmem_slls places them (--vmem-engines placer / nmu / local), over three SLRs, against the V80
# compute shell's memory NoC ports (all HBM ports in SLR2) and a shell with an HBM port in every SLR; a page
# engine's copies from the schedule's `page engine` line (vmem_controllers); the link term's port bits from the
# sll estimate's `wires` lines when the schedule has no `A -> B` lines (port_links).
import json, os, sys
here = os.path.dirname(os.path.realpath(sys.argv[1]))
sys.path.insert(0, here)
import foldsim_sll_check as sc
names = ['SLR2', 'SLR1', 'SLR0']
# A wide copy (2,048-bit ports: 16,440 bits) in SLR1, a narrow one (1,050) in SLR0, a page engine whose copies
# the assignment does not name (spread as the simulator level: half SLR2, half SLR0).
ctrls = [dict(name='G0', cls='wide', port_bits=16440), dict(name='G1', cls='narrow', port_bits=1050),
         dict(name='G2', cls='__pages0', port_bits=1050)]
shares = {'wide': {'lut': [0.0, 1.0, 0.0]}, 'narrow': {'lut': [0.0, 0.0, 1.0]}}
top = [{'lut': 50.0}, {'lut': 0.0}, {'lut': 50.0}]
v80 = json.load(open(os.path.join(here, 'foldsim_v80_slr.json')))['noc']
per_slr = dict(v80, ports=[dict(name=f'hbm_{s}', slr=s, targets=['HBM']) for s in names])
for title, noc in (('v80', v80), ('per-slr', per_slr)):
    for mode in ('placer', 'nmu', 'local'):
        tot, rows = sc.vmem_slls(ctrls, shares, top, names, noc, mode)
        print(f'{title} {mode}: ' + ' '.join(f'{x:.0f}' for x in tot) + ' | '
              + ' '.join(f'{r["name"]}:{r["slr"]}' for r in rows))
# A page engine carrying a PP row chunk of `wide` and a link stream narrow -> wide: its ports go where those are.
d = sys.argv[2]
os.makedirs(d, exist_ok=True)
json.dump({'vmem': [dict(**{'class': '__pages0'}, data_width=512, row_width=10, inst_width=8)],
           'vmem_groups': [dict(**{'class': '__pages0'}, members=[0])]}, open(os.path.join(d, 'foldsim_sim.json'), 'w'))
st = '  page engines: 1 engines, 0 page columns\n    page engine 0 512 bits, tag 9, row 10: wide:pps0_rows link:narrow>wide\n'
c = sc.vmem_controllers(d, st)
print('engine members:', c[0]['cls'], c[0]['port_bits'], sorted(c[0]['members'].items()))
tot, rows = sc.vmem_slls(c, shares, top, names, v80, 'placer')
print('engine placed:', rows[0]['slr'], ' '.join(f'{x:.0f}' for x in tot))
print('port links from wires:', sc.port_links('  II = 4\n', {('wide', 'narrow'): 96}))
print('port links from A -> B:', sc.port_links('    wide -> narrow: 2 links, 64 bits\n', {('wide', 'narrow'): 96}))
# --vmem-engines slr: each controller in its members' main SLR (page engines of one SLR group, foldsim
# --vmem-slr-groups), the masters shared per SLR (the binding's HBM ports, 8 on the V80, split by the controllers'
# port bits, at least one an SLR with controllers): only the masters and the off-SLR share of a member cross.
tot, rows = sc.vmem_slls(ctrls, shares, top, names, v80, 'slr')
print('v80 slr: ' + ' '.join(f'{x:.0f}' for x in tot) + ' | ' + ' '.join(f'{r["name"]}:{r["slr"]}' for r in rows
      if r['cls']) + ' | masters ' + ' '.join(f'{k}:{v}' for r in rows if not r['cls'] for k, v in r['masters'].items()))
tot, rows = sc.vmem_slls(ctrls, shares, top, names, v80, 'slr', 3)
print('v80 slr 3 masters: ' + ' '.join(f'{x:.0f}' for x in tot))
print('slr_masters:', sc.slr_masters([1, 1, 0], 8), sc.slr_masters([100, 1, 1], 2), sc.slr_masters([0, 0, 0], 8))
