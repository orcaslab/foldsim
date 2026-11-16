#!/usr/bin/env python3
"""foldsim_fitdb.py: the fit calibration database. Every synthesis or placement report of a generated
simulator, set against the same netlist's count, so that a fit verdict carries error bars measured on the
designs built so far, and every new build makes them tighter.

  count  <gen dir> [--json F]
        The counted resources of each class copy of a generated simulator (foldsim_sim.mlir) and of the
        simulator level: LUTs (logic by foldsim_cost, distributed RAM, black boxes), distributed-RAM LUTs as
        foldsim_cost prices them (`lutram`) and as the primitives hold them at the least (`lutram_lb`: RAM32M16
        / RAM64M8, 8 LUTs for 14 / 7 bit columns a read port, no read select), flip-flops (register bits),
        URAM288 and RAMB36 by each memory's ram_style, DSPs. Counted, not estimated: the only estimate in it is
        foldsim_cost's logic LUTs. The simulator level's link memories and registers are counted with the copy
        that reads them (the compiler places a link inside its reader's SLR group).
  ingest --db DB --name N <gen dir> <report> [--stage synth|placed] [--family F]
        One build into the database (a JSON line per build; the same name and stage again replaces it): the
        count of its netlist and, from Vivado's hierarchical utilization report (report_utilization
        -hierarchical) of that netlist, each `<copy>_copy` instance's Total LUTs, Logic LUTs, LUTRAMs, SRLs, FFs,
        RAMB36 (+ RAMB18 / 2), URAM, DSP Blocks, the simulator's (`sim`) and the AFU's (`foldsim_afu_0`; the
        kernel beside the simulator is the AFU less the simulator).
  fit --db DB [--out F] [--stage synth] [--min-luts 20000]
        Measured per counted, per resource: over each design's whole simulator (`design`: its copies, links and
        level), over single copies (their own modules) of at least --min-luts counted LUTs (`copy`), the
        simulator level with the links (`level`), and the kernel's measured
        resources. For each: n, min, median, max, and a leave-one-design-out check (does each design fall
        within the range the others give?). foldsim_fit_bound.py reads the result.

The database holds no device or design knowledge: any design folded by foldsim and synthesized for any
target can be ingested; a fit is per --stage and --family (all families by default).
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foldsim_cost  # noqa: E402

RES = ('lut', 'logic', 'lutram', 'lutram_lb', 'lutram_forced', 'ff', 'uram', 'bram', 'dsp')
# What a counted resource is measured as (report columns, see measured()).
MEASURED_AS = dict(lut='lut', logic='logic', lutram='lutram', lutram_lb='lutram', lutram_forced='lutram', ff='ff',
                   uram='uram',
                   bram='bram', dsp='dsp')
URAM_DEPTH, URAM_WIDTH = 4096, 72
# The level's array reads priced (foldsim_cost.array_reads); FOLDSIM_COST_LEVEL_ARRAY_READS=0 leaves them out, as
# before 11-11.
ARRAY_READS = os.environ.get('FOLDSIM_COST_LEVEL_ARRAY_READS', '1') != '0'
# The level's other logic at its synthesized share (foldsim_cost.LEVEL_LOGIC_SYNTH, 11-11);
# FOLDSIM_COST_LEVEL_SCALE=1 counts it as parsed, as before.
LEVEL_SCALE = float(os.environ.get('FOLDSIM_COST_LEVEL_SCALE', foldsim_cost.LEVEL_LOGIC_SYNTH))


def lutram_primitives(d, w, r):
    """Distributed-RAM LUTs a memory of depth d, width w and r read ports takes at the least: Versal's
    RAM32M16 holds 14 bit columns 32 deep in 8 LUTs, RAM64M8 7 columns 64 deep (deeper: a column per 64
    entries); one column per read port (the write port shared). The read select over columns (depth > 64)
    is logic, left out: a floor, not a price."""
    r = max(r, 1)
    if d <= 32:
        return w * r * 8 / 14
    return w * math.ceil(d / 64) * r * 8 / 7


def memory(rec, built=True):
    """{lutram, lutram_lb, uram, bram, mlogic} of one foldsim_cost memory record as Vivado builds it by its ram_style
    (foldsim_sll_check.mem_ram: URAM288 in 72-bit columns of 4,096 rows, a read port beyond the first of a
    memory without a read-write port a copy; else foldsim_cost.mem_parts's block or distributed RAM, its read select
    and write enables in `mlogic`). `built=False`: as the compiler's planner prices it (foldsim_cost.mem_cost: the
    driver's state-scratch check, foldsim_scratch_built)."""
    _, d, w, r, wr, rl, st, mk, *_ = rec
    out = dict(lutram=0.0, lutram_lb=0.0, lutram_forced=0.0, uram=0.0, bram=0.0, mlogic=0.0)
    if st == 'ultra':
        if built:
            # (A deep one without its cascade_height at Vivado's default build: foldsim_cost.uram_count.)
            out['uram'] = foldsim_cost.uram_count(d, w, r, wr, mk, rec[9] if len(rec) > 9 else None)
        else:
            copies = max(1, r if wr else math.ceil(r / 2))
            out['uram'] = math.ceil(w / URAM_WIDTH) * math.ceil(d / URAM_DEPTH) * copies
        return out
    # (`mlogic`: the distributed RAM's read select and write enables, logic LUTs as Vivado reports them.)
    lr, lg, br = foldsim_cost.mem_parts(d, w, r, rl, st, mk) if built else \
        (*foldsim_cost.mem_cost(d, w, r, rl, st, mk)[:1], 0.0, foldsim_cost.mem_cost(d, w, r, rl, st, mk)[1])
    out['bram'] = br
    out['mlogic'] = lg
    if lr:
        out['lutram'] = lr
        out['lutram_lb'] = lutram_primitives(d, w, r)
        # Read combinationally and deeper than 64: no block RAM (a synchronous read) can hold it, and flops
        # would take 64 times the sites: distributed RAM is the least it can be built as.
        if rl == 0 and d > 64:
            out['lutram_forced'] = out['lutram_lb']
    return out


def new():
    return {r: 0.0 for r in RES}


def add(acc, other, k=1.0):
    for r in RES:
        acc[r] += k * other.get(r, 0.0)
    return acc


def subtree(mods, name, stop, acc):
    """A module and its children but the `stop` modules (other copies): logic LUTs, memories, flops, black
    boxes (their LUTs, flops and DSPs, foldsim_cost.BLACKBOX)."""
    m = mods[name]
    lg = sum(m.lut.values())
    acc['logic'] += lg
    acc['lut'] += lg
    acc['ff'] += m.ff
    acc['dsp'] += m.dsp
    acc['bram'] += m.bram
    for rec in m.mems:
        mm = memory(rec)
        add(acc, mm)
        acc['lut'] += mm['lutram'] + mm['mlogic']
        acc['logic'] += mm['mlogic']
    for child, _ in m.insts:
        if child in stop:
            continue
        base = child.split('__')[0]
        if child in foldsim_cost.BLACKBOX or base in foldsim_cost.BLACKBOX:
            l, f, d = foldsim_cost.BLACKBOX.get(child, foldsim_cost.BLACKBOX.get(base))
            acc['lut'] += l
            acc['logic'] += l
            acc['ff'] += f
            acc['dsp'] += d
        elif child in mods:
            subtree(mods, child, stop, acc)
    return acc


def sim_path(gen):
    gen = gen.rstrip('/')
    for p in (os.path.join(gen, 'gen', 'foldsim_sim.mlir'), os.path.join(gen, 'foldsim_sim.mlir')):
        if os.path.exists(p):
            return p
    sys.exit(f'{gen}: no foldsim_sim.mlir (keep the netlist in the generation)')


def count(gen):
    """{'copies': {copy: counted, its own module's}, 'links': {copy: the link memories and registers read into
    it}, 'level': the simulator level's rest, 'sim': the whole simulator} of a generated simulator."""
    mods, _ = foldsim_cost.parse(sim_path(gen), count=True)
    statics = {m for m in mods if m.endswith('_static')}
    if 'foldsim_sim' not in mods:
        sys.exit(f'{gen}: no foldsim_sim module')
    copies = {s[:-len('_static')]: subtree(mods, s, statics - {s}, new()) for s in sorted(statics)}
    # The simulator's level: @foldsim_sim and the SLR group modules under it (modules that are not copies).
    level = ['foldsim_sim'] + [ch for ch, _ in mods['foldsim_sim'].insts if ch in mods and ch not in statics]
    # (--physical-domains: the domain modules, foldsim_sim_dom_<class>, nested in the level and the SLR groups' modules,
    # are the level's too: their links charged to their readers, their controls the level's rest, as without them.)
    for lv in level:
        level += [ch for ch, _ in mods[lv].insts if ch.startswith('foldsim_sim_dom_') and ch in mods and ch not in level]
    names = sorted(copies, key=len, reverse=True)

    def owner(n):
        # A link memory or register `<producer>_<port>_to_<reader>_<port>` is placed with its reader.
        return next((c for c in names if f'_to_{c}_' in n or n.endswith(f'_to_{c}')), None) or \
            next((c for c in names if n.startswith(c + '_')), None)
    rest = new()
    links = {c: new() for c in copies}
    regs = {}
    # The level's array reads (hw.array_get: the link streams' word selects, tables, reader selects), priced on what
    # each array is built of (foldsim_cost.array_reads): foldsim_cost.parse prices every one as a select of all
    # its entries, which the r16c rule left out wholesale.
    reads = foldsim_cost.array_reads(sim_path(gen), level) if ARRAY_READS else {}
    for lv in level:
        m = mods[lv]
        # The level's array reads are its link registers' reads by instance, built in the registers' read
        # selects: its other logic only (foldsim_sll_check.netlist_resources).
        lg = LEVEL_SCALE * sum(v for k, v in m.lut.items() if k != 'array_get') + reads.get(lv, (0.0, {}))[0]
        rest['logic'] += lg
        rest['lut'] += lg
        for rec in m.mems:
            mm = memory(rec)
            o = owner(rec[0])
            tgt = links[o] if o else rest
            add(tgt, mm)
            tgt['lut'] += mm['lutram'] + mm['mlogic']
            tgt['logic'] += mm['mlogic']
        for child, _ in m.insts:
            if child not in statics and child not in level and child in mods:
                subtree(mods, child, statics, rest)
        regs[lv] = m
    # Level registers by name (foldsim_sll_check.level_registers).
    cur = None
    for line in open(sim_path(gen)):
        t = line.strip()
        if t.startswith('hw.module'):
            mm = re.match(r'hw\.module(?:\s+private)?\s+@([\w$.]+)\(', t)
            cur = mm.group(1) if mm and mm.group(1) in regs else None
            continue
        if cur and ' = seq.' in t:
            lhs, rhs = t.split(' = ', 1)
            if rhs.split()[0] in foldsim_cost.REGS:
                w = foldsim_cost.width(foldsim_cost.result_type(rhs))
                o = owner(lhs.strip().lstrip('%'))
                (links[o] if o else rest)['ff'] += w
    sim = add(new(), rest)
    for c in copies:
        add(add(sim, copies[c]), links[c])
    return dict(copies=copies, links=links, level=rest, sim=sim)


COLS = {'Total LUTs': 'lut', 'Logic LUTs': 'logic', 'LUTRAMs': 'lutram', 'SRLs': 'srl', 'FFs': 'ff',
        'RAMB36': 'ramb36', 'RAMB18': 'ramb18', 'URAM': 'uram', 'DSP Blocks': 'dsp'}


def report_rows(path):
    """[(instance, module, {column: value})] of a hierarchical utilization report, columns by header name
    (with or without the PR Attribute / PPLOC columns), RAMB18 folded into `bram` as half a RAMB36."""
    cols, rows, state = None, [], None
    for line in open(path):
        if line.startswith('| Design State'):
            state = line.split(':', 1)[1].strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells and cells[0] == 'Instance':
            cols = {i: COLS[c] for i, c in enumerate(cells) if c in COLS}
            continue
        if not cols or len(cells) < max(cols) + 1:
            continue
        vals = {}
        for i, k in cols.items():
            m = re.match(r'(\d+)', cells[i])
            vals[k] = int(m.group(1)) if m else 0
        vals['bram'] = vals.pop('ramb36', 0) + vals.pop('ramb18', 0) / 2
        rows.append((cells[0], cells[1], vals))
    return rows, state


def measured(path, copies):
    """{'copies': {copy: measured}, 'sim': measured, 'afu': measured, 'state': report state}: the first
    `<copy>_copy` row of each counted copy, the first `sim` and `foldsim_afu_0` rows."""
    rows, state = report_rows(path)
    out = dict(copies={}, sim=None, afu=None, state=state)
    for inst, mod, v in rows:
        if inst.endswith('_copy') and inst[:-len('_copy')] in copies:
            out['copies'].setdefault(inst[:-len('_copy')], v)
        elif inst == 'sim' and out['sim'] is None:
            out['sim'] = v
        elif inst == 'foldsim_afu_0' and out['afu'] is None:
            out['afu'] = v
    return out


def ingest(a):
    c = count(a.gen)
    m = measured(a.report, c['copies'])
    if not m['copies'] and not m['sim']:
        sys.exit(f'{a.report}: neither a <copy>_copy instance of {a.gen}\'s copies nor the simulator (sim)')
    rec = dict(name=a.name, stage=a.stage, family=a.family, gen=os.path.abspath(a.gen),
               report=os.path.abspath(a.report), report_state=m['state'], date=time.strftime('%F %T'),
               copies={k: dict(count=c['copies'][k], links=c['links'][k], meas=m['copies'].get(k)) for k in c['copies']},
               # The level as measured (the simulator less its copies' instances) holds the links too.
               level=dict(count=add(add(new(), c['level']), {r: sum(v[r] for v in c['links'].values()) for r in RES})),
               sim=dict(count=c['sim'], meas=m['sim']), afu=dict(meas=m['afu']))
    if m['sim'] and m['afu']:
        rec['kernel'] = dict(meas={k: m['afu'][k] - m['sim'][k] for k in m['afu']})
    if m['sim']:
        cm = {k: sum(v['meas'][k] for v in rec['copies'].values() if v['meas']) for k in m['sim']}
        rec['level']['meas'] = {k: m['sim'][k] - cm[k] for k in m['sim']}
    db = [json.loads(l) for l in open(a.db)] if os.path.exists(a.db) else []
    db = [r for r in db if not (r['name'] == a.name and r['stage'] == a.stage)] + [rec]
    with open(a.db, 'w') as f:
        for r in db:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    nm = sum(1 for v in rec['copies'].values() if v['meas'])
    print(f'{a.name} ({a.stage}, {m["state"]}): {nm} of {len(rec["copies"])} copies measured; '
          + ', '.join(f'{r} {ratio_s(sim_meas(rec, r), sim_count(rec, r)) if m["sim"] else "-"}'
                      for r in ('lut', 'lutram', 'lutram_forced', 'ff', 'uram', 'bram')) + ' (simulator measured/counted)')


def sim_count(rec, r):
    return rec['sim']['count'][r]


def sim_meas(rec, r):
    return rec['sim']['meas'][MEASURED_AS[r]]


def sum_count(rec, r):
    return sum(v['count'][r] for v in rec['copies'].values() if v['meas'])


def sum_meas(rec, r):
    return sum(v['meas'][MEASURED_AS[r]] for v in rec['copies'].values() if v['meas'])


def ratio_s(m, c):
    return f'{m / c:.3f}' if c else '-'


# Student's t, one-sided 95 %, by degrees of freedom (1..10, then 1.645 + 2.4 / df).
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}


def stats(xs):
    """n, min, median, max of (label, ratio) pairs, and the one-sided 95 % prediction bounds of one more design
    (mean -/+ t x sd x sqrt(1 + 1/n)), never inside the observed range: `lo` and `hi`, what a bound uses."""
    if not xs:
        return None
    vals = sorted(v for _, v in xs)
    n = len(vals)
    out = dict(n=n, min=vals[0], median=statistics.median(vals), max=vals[-1],
               lo_by=min(xs, key=lambda x: x[1])[0], hi_by=max(xs, key=lambda x: x[1])[0])
    if n >= 3:
        mu, sd = statistics.mean(vals), statistics.stdev(vals)
        t = T95.get(n - 1, 1.645 + 2.4 / (n - 1))
        w = t * sd * math.sqrt(1 + 1 / n)
        out.update(mean=mu, sd=sd, lo=min(vals[0], mu - w), hi=max(vals[-1], mu + w))
    else:
        out.update(lo=vals[0], hi=vals[-1])
    return out


def fit(a):
    db = [json.loads(l) for l in open(a.db)]
    db = [r for r in db if r['stage'] == a.stage and (not a.family or r['family'] == a.family)
          and r['sim']['count']['lut'] >= a.min_design_luts]
    out = dict(format=2, stage=a.stage, family=a.family or 'all', designs=[r['name'] for r in db], min_luts=a.min_luts,
               design={}, copy={}, level={}, kernel={}, loo={})
    # Each design's copies together, measured per counted (a resource the design counts at least 1 % of its
    # total of, so that a design without URAM does not give URAM a ratio).
    for r in RES:
        xs = [(d['name'], sim_meas(d, r) / sim_count(d, r)) for d in db if d['sim'].get('meas') and sim_count(d, r) > 0
              and (r not in ('uram', 'bram', 'dsp') or sim_count(d, r) >= 10)]
        out['design'][r] = stats(xs)
        cs = [(f'{d["name"]}:{k}', v['meas'][MEASURED_AS[r]] / v['count'][r]) for d in db
              for k, v in d['copies'].items() if v['meas'] and v['count']['lut'] >= a.min_luts and v['count'][r] > 0
              and (r not in ('uram', 'bram', 'dsp') or v['count'][r] >= 4)]
        out['copy'][r] = stats(cs)
        ls = [(d['name'], d['level']['meas'][MEASURED_AS[r]] / d['level']['count'][r]) for d in db
              if d['level'].get('meas') and d['level']['count'][r] >= (1000 if r in ('lut', 'logic', 'ff') else 10)]
        out['level'][r] = stats(ls)
    for k in ('lut', 'lutram', 'ff', 'bram', 'uram', 'dsp'):
        ks = [(d['name'], d['kernel']['meas'][k]) for d in db if d.get('kernel')]
        out['kernel'][k] = stats(ks)
    # Leave one design out: the range of the others' design ratios against the design's own.
    for d in db:
        rest = [x for x in db if x['name'] != d['name']]
        row = {}
        for r in ('lut', 'lutram', 'ff', 'uram', 'bram'):
            if not d['sim'].get('meas') or sim_count(d, r) <= 0:
                continue
            xs = [(x['name'], sim_meas(x, r) / sim_count(x, r)) for x in rest
                  if x['sim'].get('meas') and sim_count(x, r) > 0]
            if not xs:
                continue
            own = sim_meas(d, r) / sim_count(d, r)
            if own <= 0:
                continue
            st = stats(xs)
            row[r] = dict(own=own, lo=st['lo'], hi=st['hi'], within=st['lo'] <= own <= st['hi'],
                          err_mid=st['median'] / own - 1)
        out['loo'][d['name']] = row
    js = json.dumps(out, indent=1, sort_keys=True)
    if a.out:
        open(a.out, 'w').write(js + '\n')
    lines = [f'fit {a.stage} over {len(db)} designs ({", ".join(out["designs"])}); measured per counted:']
    for scope in ('design', 'copy', 'level'):
        for r in RES:
            s = out[scope][r]
            if s:
                lines.append(f'  {scope:6} {r:13} n {s["n"]:3}  min {s["min"]:.3f} ({s["lo_by"]})  median '
                             f'{s["median"]:.3f}  max {s["max"]:.3f} ({s["hi_by"]})  bounds [{s["lo"]:.3f}, {s["hi"]:.3f}]')
    for k, s in out['kernel'].items():
        if s:
            lines.append(f'  kernel {k:9} n {s["n"]:3}  min {s["min"]:,.0f} ({s["lo_by"]})  median {s["median"]:,.0f}  '
                         f'max {s["max"]:,.0f} ({s["hi_by"]})')
    for n, row in out['loo'].items():
        lines.append(f'  leave {n} out: ' + ', '.join(
            f'{r} {v["own"]:.3f} in [{v["lo"]:.3f}, {v["hi"]:.3f}]{"" if v["within"] else " OUTSIDE"} '
            f'(median {100 * v["err_mid"]:+.0f} %)' for r, v in row.items()))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('count')
    c.add_argument('gen')
    c.add_argument('--json')
    i = sub.add_parser('ingest')
    i.add_argument('--db', required=True)
    i.add_argument('--name', required=True)
    i.add_argument('--stage', default='synth', choices=('synth', 'placed'))
    i.add_argument('--family', default='')
    i.add_argument('gen')
    i.add_argument('report')
    f = sub.add_parser('fit')
    f.add_argument('--db', required=True)
    f.add_argument('--out')
    f.add_argument('--stage', default='synth', choices=('synth', 'placed'))
    f.add_argument('--family', default='')
    f.add_argument('--min-luts', type=float, default=20000)
    f.add_argument('--min-design-luts', type=float, default=100000,
                   help='designs below this many counted LUTs are left out (a tiny design\'s ratios are not a large one\'s)')
    a = ap.parse_args()
    if a.cmd == 'count':
        c = count(a.gen)
        js = json.dumps(c, indent=1, sort_keys=True)
        if a.json:
            open(a.json, 'w').write(js + '\n')
        t = c['sim']
        print(f'{os.path.basename(a.gen.rstrip("/"))}: {len(c["copies"])} copies; simulator counted: '
              + ', '.join(f'{r} {t[r]:,.0f}' for r in RES))
    elif a.cmd == 'ingest':
        ingest(a)
    else:
        fit(a)


if __name__ == '__main__':
    main()
