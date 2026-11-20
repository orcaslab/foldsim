#!/usr/bin/env python3
"""fit3.py: the three-point whole-design calibration -> calibrations/calm4b4s4-20261104.json (fit2.py + point 3).

Point 1, calm4: a100_lite_m4 (9dcc73c2a), gen/m4, Vivado synthesis only (hier2.rpt); model LUTs x SYNTH_PER_ROUTED.
Point 2, B4: a100_lite_m4 A3 (bfd8b6ea4), gen/m4p (calibrated policy), the routed V80 image (calm4a3/img,
report_utilization_foldsim.txt, -hierarchical); model LUTs as they are (routed).
Each priced as built, uncalibrated (foldsim_degree state --kernel-terms --links-as-built --no-page).

Every factor is pooled over both points: measured sum / priced sum, each point's price scaled to its own measure
(x1.038 synthesized, x1 routed). The terms are fit_calm4.py's: per family logic and LUTRAM factor (or an additive
LUTRAM per copy where the model prices none), the top level's logic per link register bit, the links' LUTRAM, the
virtual-memory controllers, the control endpoint as a fixed term.

Point 3 (11-04), s4x: the small GPU (2 cores, IW=FW=SFU=4, TCU, VM, FPU_TYPE_STD; m1tel/cflags_s4.txt), m1tel/gen/s4x
(foldsim fold-seq-room 054231b, B4's flags, L2 data in HBM), the routed V80 image m1tel/img_s4x at 50 MHz (WNS +1.239);
priced as built uncalibrated (m1tel/price_s4x_unc.json). The 21 %-pessimistic price of the small fold under calm4b4.
The virtual-memory controller factors pool only the points whose price has them (calm4; B4's and s4x's prices carry
none): fit2 pooled every point's measured controllers over calm4's price alone.
B4's price is calm4a3/price_m4p_unc.json (m4p_unc.argv.json, current code, 1,029,124 LUTs): price_m4p.json was
overwritten at 18:39 by a calibrated pass (check.py writes to its argv's --out) and no longer reproduces fit2's input.
calm4's price_ab3.json reproduces exactly (calm4/price_unc.json)."""
import json, re, os

S_SYN = 1.038
# The calibration builds' work directory (calm4/, calm4a3/, ...).
L = os.environ.get('FOLDSIM_CAL_DIR', 'lk2')
POINTS = [
    dict(name='calm4 (synthesis)', rpt=L + '/calm4/syn/hier2.rpt', price=L + '/calm4/price_ab3.json', s=S_SYN,
         afu='foldsim_afu'),
    dict(name='B4 (routed image)', rpt=L + '/calm4a3/img/bin/foldsim.vbin.prj/report_utilization_foldsim.txt',
         price=L + '/calm4a3/price_m4p_unc.json', s=1.0, afu='foldsim_afu_0'),
    dict(name='s4x (routed image)', rpt=L + '/m1tel/img_s4x/bin/foldsim.vbin.prj/report_utilization_foldsim.txt',
         price=L + '/m1tel/price_s4x_unc.json', s=1.0, afu='foldsim_afu_0'),
]
FAM = [tuple(x) for x in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'vortex_families.json')))['families']]  # Vortex's, as data


def parse(path):
    """[(indent, name, [total, logic, lutram, srl, ff, ramb36, ramb18, uram])] from a hierarchical report, either
    layout (synthesis: values from column 3; image: PR attribute and PPLOCs first, values from column 5)."""
    order = []
    for line in open(path):
        f = line.split('|')
        if len(f) < 12:
            continue
        num = [re.sub(r'\(.*\)', '', x).strip() for x in f[2:]]
        k = next((j for j, x in enumerate(num) if x.isdigit()), None)
        if k is None or len(num) < k + 8 or not all(x.isdigit() for x in num[k:k + 8]):
            continue
        n = f[1].strip()
        order.append((len(f[1]) - len(f[1].lstrip()), n, [int(x) for x in num[k:k + 8]]))
    return order


def point(p):
    order = parse(p['rpt'])
    rows = {}
    for ind, n, v in order:
        rows.setdefault(n, v)
    d = json.load(open(p['price']))['res']
    i = [k for k, r in enumerate(order) if r[1] == 'sim'][0]
    sim_ind = order[i][0]
    kids = [(n, v) for ind, n, v in order[i + 1:] if ind == sim_ind + 2]
    end = next((k for k in range(i + 1, len(order)) if order[k][0] <= sim_ind), len(order))
    kids = [(n, v) for ind, n, v in order[i + 1:end] if ind == sim_ind + 2]
    lk_lutram = sum(v[2] + v[3] for n, v in kids if not n.endswith('_copy') and n != '(sim)')
    kid = [sum(v[j] for n, v in kids) for j in range(8)]
    simv = rows['sim']
    lk_lutram += simv[2] - kid[2]
    copies = {k.split('#')[0]: v for k, v in d['copies'].items()}
    vm = [rows['vmem_0'], rows['vmem_1']]
    return dict(rows=rows, d=d, copies=copies, s=p['s'], lk_lutram=lk_lutram,
                link_bram=(simv[5] - kid[5]) + (simv[6] - kid[6]) / 2,
                vml=sum(v[1] for v in vm), vmr=sum(v[2] + v[3] for v in vm), vmb=sum(v[5] + v[6] / 2 for v in vm),
                ctrl=rows['ctrl'], afu=rows[p['afu']], top=rows['(sim)'][1])


P = [point(p) for p in POINTS]
fams = []
for name, rx in FAM:
    ml = mr = gl = gr = 0.0
    n = 0
    per = []
    for q in P:
        a = dict(ml=0.0, mr=0.0, gl=0, gr=0, n=0)
        for c, v in q['copies'].items():
            if not re.search(rx, c) or c + '_copy' not in q['rows']:
                continue
            m = q['rows'][c + '_copy']
            a['n'] += 1
            a['ml'] += (v['logic'] + v['store_lut']) * q['s']
            a['mr'] += v['store_lutram'] * q['s']
            a['gl'] += m[1]
            a['gr'] += m[2] + m[3]
        per.append(a)
        ml += a['ml']; mr += a['mr']; gl += a['gl']; gr += a['gr']; n += a['n']
    if not n:
        continue
    e = dict(family=name, match=rx, logic=round(gl / ml, 3) if ml else 1.0, copies=n,
             points=[dict(logic=round(a['gl'] / a['ml'], 3) if a['ml'] else None,
                          lutram=round(a['gr'] / a['mr'], 3) if a['mr'] else None) for a in per])
    if mr >= 0.05 * gr or gr < 200:
        e['lutram'] = round(gr / mr, 3) if mr else 1.0
    else:
        # additive LUTRAM per copy, in routed LUTs: each point's excess over its priced LUTRAM, at its own scale
        e['lutram'] = 1.0
        e['lutram_add'] = round(sum((a['gr'] - a['mr']) / q['s'] for a, q in zip(per, P)) / n)
    fams.append(e)

tl_m = sum(q['top'] / q['s'] - 4 * 3500.0 for q in P)
tl_p = sum(q['d']['link_ff'] for q in P)
link_f = sum(q['lk_lutram'] for q in P) / sum(q['d']['link_lutram'] * q['s'] for q in P)
VP = [q for q in P if q['d']['vmem_ctl']['lut'] > 0]  # the points whose price has controllers
vm_l = sum(q['vml'] for q in VP) / sum(q['d']['vmem_ctl']['lut'] * q['s'] for q in VP)
vm_r = sum(q['vmr'] for q in VP) / sum(q['d']['vmem_ctl']['lutram'] * q['s'] for q in VP)
fx_l = sum(q['ctrl'][1] / q['s'] for q in P) / len(P)
fx_r = sum((q['ctrl'][2] + q['ctrl'][3]) / q['s'] for q in P) / len(P)
cal = dict(name='calm4b4s4-20261104',
           source='three points pooled: calm4 (a100_lite_m4 9dcc73c2a, gen/m4, Vivado synthesis only, image mode), '
                  'B4 (a100_lite_m4 A3 bfd8b6ea4, gen/m4p, routed V80 image at 50 MHz, WNS +1.253) and s4x (the small '
                  'GPU, m1tel/gen/s4x, foldsim 054231b, routed V80 image at 50 MHz, WNS +1.239); each priced as built '
                  'uncalibrated (foldsim_degree state --kernel-terms --links-as-built --no-page); controller factors '
                  'over the points whose price has controllers (calm4)',
           synth_per_routed=S_SYN, families=fams, rest=0.0,
           top_logic_per_link_bit=round(tl_m / tl_p, 3), link_lutram=round(link_f, 3),
           vmem_ctl=dict(logic=round(vm_l, 3), lutram=round(vm_r, 3)),
           fixed=dict(logic=round(fx_l), lutram=round(fx_r), ramb36=0, what='the control endpoint (foldsim_ctrl)'),
           points=[dict(name=p['name'], measured_total=dict(logic=q['afu'][1], lutram=q['afu'][2] + q['afu'][3],
                                                             ff=q['afu'][4], ramb36=q['afu'][5], ramb18=q['afu'][6],
                                                             uram=q['afu'][7]), measure_scale=q['s'])
                   for p, q in zip(POINTS, P)])
out = L + '/foldsim/tools/foldsim/calibrations/calm4b4s4-20261104.json'
json.dump(cal, open(out, 'w'), indent=1)
for e in fams:
    pts = ' '.join(f"[{x['logic']}, {x['lutram']}]" for x in e['points'])
    print(f"{e['family']:14s} n {e['copies']:3d} logic {e['logic']:6.3f} lutram {e['lutram']:6.3f} "
          f"add {e.get('lutram_add', 0):6}  per point {pts}")
print('top', cal['top_logic_per_link_bit'], 'links', cal['link_lutram'], 'vmem', cal['vmem_ctl'], 'fixed', cal['fixed'])
