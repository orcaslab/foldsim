#!/usr/bin/env python3
"""fit.py: the calm4 whole-design calibration -> foldsim tools/foldsim/calibrations/calm4-20261103.json. Measured: syn/
hier2.rpt (a100_lite_m4 synthesis only, image mode); priced: price_ab3.json (the same netlist as built, uncalibrated).
Per family: logic factor = measured logic / ((copy logic + store select logic) x SYNTH_PER_ROUTED); LUTRAM factor =
measured LUTRAM + SRL / (store LUTRAM x SYNTH_PER_ROUTED), or, where the model prices no LUTRAM, an additive LUTRAM per
copy. Top level: (measured logic of the simulator's own level / S - the ports' queues) / link register bits. Links:
measured link-memory LUTRAM / priced. Virtual-memory controllers: measured vmem_<k> / the HBM kind's priced costs.
Fixed: the control endpoint (no estimate)."""
import json, re, os
S = 1.038
# The calm4 synthesis build (its syn/hier2.rpt), under the calibration work directory.
C = os.path.join(os.environ.get('FOLDSIM_CAL_DIR', 'lk2'), 'calm4')
rows, order = {}, []
for line in open(C + '/syn/hier2.rpt'):
    f = line.split('|')
    if len(f) < 12 or not f[3].strip().isdigit():
        continue
    v = [int(x) for x in f[3:11]]
    n = f[1].strip(); ind = len(f[1]) - len(f[1].lstrip())
    order.append((ind, n, v))
    rows.setdefault(n, v)
d = json.load(open(C + '/price_ab3.json'))['res']
copies = {k.split('#')[0]: v for k, v in d['copies'].items()}
FAM = [tuple(x) for x in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'vortex_families.json')))['families']]  # Vortex's, as data
fams = []
for name, rx in FAM:
    ml = mr = gl = gr = 0.0; n = 0
    for c, v in copies.items():
        if not re.search(rx, c) or c + '_copy' not in rows:
            continue
        m = rows[c + '_copy']; n += 1
        ml += v['logic'] + v['store_lut']; mr += v['store_lutram']
        gl += m[1]; gr += m[2] + m[3]
    if not n:
        continue
    e = dict(family=name, match=rx, logic=round(gl / (ml * S), 3) if ml else 1.0, copies=n,
             measured=dict(logic=gl, lutram=gr), priced=dict(logic=round(ml), lutram=round(mr)))
    if mr >= 0.05 * gr or gr < 200:
        e['lutram'] = round(gr / (mr * S), 3) if mr else 1.0
    else:
        e['lutram'] = 1.0; e['lutram_add'] = round((gr - mr * S) / S / n)
    fams.append(e)
links_bits = d['link_ff']
tlb = (rows['(sim)'][1] / S - 4 * 3500.0) / links_bits
i = [k for k, r in enumerate(order) if r[1] == 'sim'][0]
lk_lutram = 0
for ind, n, v in order[i + 1:]:
    if ind <= 3: break
    if ind == 5 and not n.endswith('_copy') and n != '(sim)': lk_lutram += v[2] + v[3]
# The links' memories Vivado leaves directly in the simulator (no child instance in the report): its LUTRAM, block
# RAM beyond its children's.
simv = rows['sim']; kid = [0] * 8
for ind, n, v in order[i + 1:]:
    if ind <= 3: break
    if ind == 5: kid = [a + b for a, b in zip(kid, v)]
lk_lutram += simv[2] - kid[2]
link_bram_measured = (simv[5] - kid[5]) + (simv[6] - kid[6]) / 2
link_f = lk_lutram / (d['link_lutram'] * S)
vm = [rows['vmem_0'], rows['vmem_1']]
vml, vmr, vmb = sum(v[1] for v in vm), sum(v[2] + v[3] for v in vm), sum(v[5] + v[6] / 2 for v in vm)
cal = dict(name='calm4-20261103',
           source='st32_1031/lk2/calm4: a100_lite_m4 (a100/config 9dcc73c2a), gen/m4 II 512, v7 kernel V7B; Vivado '
                  '2025.2 synthesis only, image mode (RuntimeOptimized, hierarchy kept, global retiming), V80; priced '
                  'as built (foldsim_degree state --kernel-terms --links-as-built --no-page)',
           synth_per_routed=S, families=fams, rest=0.0,
           links_measured=dict(lutram=lk_lutram, ramb36=link_bram_measured, priced_lutram=d['link_lutram'],
                               priced_ramb36=d['link_bram']), top_logic_per_link_bit=round(tlb, 3), link_lutram=round(link_f, 3),
           vmem_ctl=dict(logic=round(vml / (d['vmem_ctl']['lut'] * S), 3), lutram=round(vmr / (d['vmem_ctl']['lutram'] * S), 3),
                         measured=dict(logic=vml, lutram=vmr, ramb36=vmb), priced=d['vmem_ctl']),
           fixed=dict(logic=round(rows['ctrl'][1] / S), lutram=round((rows['ctrl'][2] + rows['ctrl'][3]) / S), ramb36=0,
                      what='the control endpoint (foldsim_ctrl)'),
           measured_total=dict(logic=rows['foldsim_afu'][1], lutram=rows['foldsim_afu'][2] + rows['foldsim_afu'][3],
                               ff=rows['foldsim_afu'][4], ramb36=rows['foldsim_afu'][5], ramb18=rows['foldsim_afu'][6],
                               uram=rows['foldsim_afu'][7]))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calm4-20261103.json')
json.dump(cal, open(out, 'w'), indent=1)
for e in fams: print(f"{e['family']:14s} n {e['copies']:3d} logic {e['logic']:6.3f} lutram {e['lutram']:6.3f} add {e.get('lutram_add', 0)}")
print('top', cal['top_logic_per_link_bit'], 'links', cal['link_lutram'], 'vmem', cal['vmem_ctl']['logic'], cal['vmem_ctl']['lutram'], 'fixed', cal['fixed'])
