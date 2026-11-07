#!/usr/bin/env python3
"""foldsim_sll_check.py <gen dir> [--log <driver log>] [--slr-region v80|<json>] [--copy-cap SLR]
                       [--resources netlist|schedule] [--foldsim-opt <bin>] [--out <dir>]

Whether an existing netlist places over the region's SLRs, without regenerating it.

Each class copy's resources: with `--resources netlist` (the default where the generation's foldsim_sim.mlir is
there) its LUTs, flip-flops, URAM288 and RAMB36 counted on the generated netlist itself (foldsim_cost over the
copy's module `<copy>_static` and the stores and page RAMs below it; a memory's RAM by its ram_style, as Vivado
builds it: URAM288 by 72-bit columns of 4,096 rows, RAMB36 by foldsim_cost.mem_cost), LUTs and flops scaled to
synthesized ones by the region's netlist_lut_scale / netlist_ff_scale, and the simulator level's link memories
counted with the copy whose output they hold (a link memory is its producer's output register, placed at the
producer); with `--resources schedule` its LUT sites only, from the class estimates of the generation's models
and its schedule (foldsim_degree.copy_sites: paged copies in page RAMs).

The checks, in order:
  1. every copy fits whole, in each resource, the `--copy-cap` SLR's capacity times the region's margin (SLR a
     name, `smallest` or `largest`; default largest: a copy past every SLR must be cut by the placer, its wide
     state and store ports then crossing SLLs at a cost not known here: r16c, 11-07, a slice copy of 517 URAM
     against SLRs of 496 / 511 / 204, failed partitioning at 23,987 SLLs);
  2. some assignment of the copies to the SLRs keeps every SLR within its capacity in each resource (the
     generation's logged fold pipeline -- the `+ ... foldsim-opt <flat.mlir> ... --pass-pipeline=...` line of the
     generation log, <gen dir>.gen.log by default -- run again schedule-only with sll-estimate over the copies'
     LUT sites and, as further resources, their flops, URAM and RAMB36);
  3. the link bits over the worst boundary of that assignment, whole copies, as SLLs (times the region's
     sll_per_link_bit, calibrated on a placed build: the placer splits a large copy where its own cut is small, a
     wide link memory is read a word at a time) against the budget (sll_per_boundary x sll_margin; past it a
     warning) and the boundary's SLLs (past them a refusal).

Logic (LUTs, flops) is checked against each SLR's capacity times the region's margin, memories (URAM288, RAMB36)
against its whole count (a copy's RAM primitives, each on narrow ports, can be placed apart).

Exits 0 (OK, or WARN), 1 if a check refuses, 2 on error. Report in <out>/sll.txt (the estimate)
and on stdout.
"""
import argparse, json, math, os, re, shlex, subprocess, sys
from itertools import takewhile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foldsim_cost  # noqa: E402
import foldsim_degree  # noqa: E402

RESOURCES = ('lut', 'ff', 'uram', 'bram')
# A Versal CLB: 8 LUT6 and 16 flip-flops.
FFS_PER_LUT = 2
URAM_DEPTH, URAM_WIDTH = 4096, 72


def split_top(pipeline):
    """The top-level passes of builtin.module(...), braces and parentheses respected."""
    body = pipeline[pipeline.index('(') + 1:pipeline.rindex(')')]
    parts, depth, cur = [], 0, ''
    for ch in body:
        if ch in '({':
            depth += 1
        elif ch in ')}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return parts


def mem_ram(rec):
    """(URAM288, RAMB36) of a foldsim_cost memory record as Vivado builds it: a
    ram_style ultra memory in 72-bit columns of 4,096 rows (Vivado puts a
    small one in a whole URAM too), else foldsim_cost.mem_cost's block RAM."""
    _, d, w, r, wr, rl, st, *_ = rec
    if st == 'ultra':
        # A read port beyond the first is a copy (foldsim_storage.kind_cost).
        copies = max(1, r if wr else math.ceil(r / 2))
        return math.ceil(w / URAM_WIDTH) * math.ceil(d / URAM_DEPTH) * copies, 0.0
    return 0.0, foldsim_cost.mem_cost(d, w, r, rl, st)[1]


def subtree(mods, name, stop, acc):
    """A module's LUTs (logic, black boxes, distributed RAM), flops and RAM with
    its children, but not the `stop` modules (other copies)."""
    m = mods[name]
    acc['lut'] += sum(m.lut.values())
    acc['ff'] += m.ff
    for rec in m.mems:
        u, b = mem_ram(rec)
        acc['uram'] += u
        acc['bram'] += b
        _, d, w, r, _, rl, st, *_ = rec
        acc['lut'] += foldsim_cost.mem_cost(d, w, r, rl, st)[0] if st != 'ultra' else 0.0
    for child, _ in m.insts:
        if child in stop:
            continue
        base = child.split('__')[0]
        if child in foldsim_cost.BLACKBOX or base in foldsim_cost.BLACKBOX:
            l, f, _ = foldsim_cost.BLACKBOX.get(child, foldsim_cost.BLACKBOX.get(base))
            acc['lut'] += l
            acc['ff'] += f
        elif child in mods:
            subtree(mods, child, stop, acc)
    return acc


def top_registers(sim):
    """(name, bits) of each register of the simulator level (@foldsim_sim)."""
    inside = False
    for line in open(sim):
        t = line.strip()
        if t.startswith('hw.module'):
            inside = bool(re.match(r'hw\.module(?:\s+private)?\s+@foldsim_sim\(', t))
            continue
        if inside and ' = seq.' in t:
            lhs, rhs = t.split(' = ', 1)
            if rhs.split()[0] in foldsim_cost.REGS:
                yield lhs.strip().lstrip('%'), foldsim_cost.width(foldsim_cost.result_type(rhs))


def netlist_resources(sim, copies, region):
    """{copy: {lut, ff, uram, bram}} of a generated simulator (`sim`, its
    foldsim_sim.mlir) for the schedule's `copies`, LUTs and flops scaled to
    synthesized ones (region netlist_lut_scale / netlist_ff_scale), and the
    simulator level's rest ('__top__': its logic and the link memories no
    copy's name prefixes)."""
    mods, _ = foldsim_cost.parse(sim)
    statics = {f'{c}_static' for c in copies if f'{c}_static' in mods}
    out = {}
    for c in copies:
        acc = dict(lut=0.0, ff=0.0, uram=0.0, bram=0.0)
        if f'{c}_static' in mods:
            subtree(mods, f'{c}_static', statics - {f'{c}_static'}, acc)
        out[c] = acc
    top = mods.get('foldsim_sim')
    rest = dict(lut=0.0, ff=0.0, uram=0.0, bram=0.0)
    if top is not None:
        # The simulator level's array reads are its link registers' reads by
        # instance, which synthesis builds in the registers' read selects (r16c:
        # 2.23M LUTs counted as array_get, the whole simulator level 129k LUTs
        # synthesized besides its copies): its other logic only.
        rest['lut'] += sum(v for k, v in top.lut.items() if k != 'array_get')
        # A link memory or register is its producer's output register: the copy
        # whose name (the longest) prefixes the memory's or register's.
        names = sorted(copies, key=len, reverse=True)
        for reg, w in top_registers(sim):
            owner = next((c for c in names if reg.startswith(c + '_')), None)
            (out[owner] if owner else rest)['ff'] += w
        for rec in top.mems:
            u, b = mem_ram(rec)
            _, d, w, r, _, rl, st, *_ = rec
            lr = foldsim_cost.mem_cost(d, w, r, rl, st)[0] if st != 'ultra' else 0.0
            owner = next((c for c in names if rec[0].startswith(c + '_')), None)
            dst = out[owner] if owner else rest
            dst['uram'] += u
            dst['bram'] += b
            dst['lut'] += lr
        for child, _ in top.insts:
            if child not in statics and child in mods:
                subtree(mods, child, statics, rest)
    ls, fs = region.get('netlist_lut_scale', 1.0), region.get('netlist_ff_scale', 1.0)
    for v in list(out.values()) + [rest]:
        v['lut'] *= ls
        v['ff'] *= fs
    out['__top__'] = rest
    return out


def slr_caps(region):
    """Per SLR (in order), its capacity in each resource: LUTs and flops times
    the margin (logic placed whole must route), URAM and RAMB36 whole (a
    copy's RAM primitives, each on narrow ports, are placed where they fit:
    b4e's opc copy of 1,024 RAMB36, 98 % of SLR1's and past its margin,
    placed and routed, 11-07)."""
    m = region.get('margin', 1.0)
    return [dict(name=s['name'], lut=m * s['luts'], ff=m * FFS_PER_LUT * s['luts'],
                 uram=s.get('uram', 0), bram=s.get('ramb36', 0)) for s in region['slrs']]


def fmt(r, x):
    return f'{x:,.1f}' if r in ('uram', 'bram') else f'{x:,.0f}'


def lut_equivalent(res, caps):
    """A copy's binding resource in LUTs: each resource as the share of the
    region's it takes, times the region's LUTs."""
    tot = {r: sum(c[r] for c in caps) for r in RESOURCES}
    return max(res[r] / tot[r] * tot['lut'] for r in RESOURCES if tot[r])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gen')
    ap.add_argument('--log')
    ap.add_argument('--slr-region', default='v80')
    ap.add_argument('--copy-cap', default='any', metavar='SLR',
                    help='where every class copy must fit whole: any (default: some SLR of the region), or the '
                         'SLR named, smallest or largest (by LUTs)')
    ap.add_argument('--resources', choices=('netlist', 'schedule'), default=None,
                    help='netlist (default where foldsim_sim.mlir is there) or schedule')
    ap.add_argument('--foldsim-opt', default=os.path.join(HERE, 'foldsim-opt'))
    ap.add_argument('--out')
    ap.add_argument('--register-state-entries', type=int, default=None)
    a = ap.parse_args()
    gen = os.path.abspath(a.gen.rstrip('/'))
    gdir = os.path.join(gen, 'gen') if os.path.isdir(os.path.join(gen, 'gen')) else gen
    log = a.log or gen + '.gen.log'
    out = os.path.abspath(a.out or gen + '.sll')
    os.makedirs(out, exist_ok=True)
    region_path = os.path.join(HERE, 'foldsim_v80_slr.json') if a.slr_region == 'v80' else a.slr_region
    region = json.load(open(region_path))
    spl = region['sites_per_lut']
    caps = slr_caps(region)
    names = [c['name'] for c in caps]
    budget = int(region.get('sll_margin', 1.0) * region.get('sll_per_boundary', 0))
    name = os.path.basename(gen)
    sim = os.path.join(gdir, 'foldsim_sim.mlir')
    mode = a.resources or ('netlist' if os.path.exists(sim) else 'schedule')

    line = next((l for l in open(log) if 'foldsim-fold-static{' in l and '--pass-pipeline=' in l), None)
    if not line:
        sys.exit(f'no fold pipeline line in {log}')
    # The trace line is unquoted (the pipeline's options hold spaces): the input is the token after foldsim-opt,
    # the pipeline everything from --pass-pipeline= to the final -o.
    flat = line.lstrip('+ ').split()[1]
    m = re.search(r'--pass-pipeline=[\'"]?(builtin\.module\(.*\))[\'"]? -o \S+\s*$', line)
    if not m:
        sys.exit(f'no --pass-pipeline=builtin.module(...) -o in the fold line of {log}')
    passes = split_top(m.group(1))
    fi = next(i for i, p in enumerate(passes) if p.startswith('foldsim-fold-static{'))
    fold = passes[fi][len('foldsim-fold-static{'):-1]
    # Options this check sets itself, or that would write the generation's outputs.
    fold = ' '.join(o for o in shlex.split(fold) if not re.match(
        r'(report|slr-groups|slr-capacity|slr-capacities|slr-sites|slr-assign|slr-sll-budget|sll-estimate|schedule-only)\b', o))
    models = os.path.join(out, 'models.mlir')
    run = lambda cmd: subprocess.run(cmd, check=True, capture_output=True, text=True)
    if not os.path.exists(flat) and os.path.exists(os.path.join(gdir, 'models.mlir')):
        models = os.path.join(gdir, 'models.mlir')
    else:
        run([a.foldsim_opt, flat, '--mlir-print-op-on-diagnostic=false',
             f'--pass-pipeline=builtin.module({",".join(passes[:fi])})', '-o', models])
    # The class copies and their instances: the generation's own schedule (same fold options).
    stext = open(os.path.join(gdir, 'schedule.txt')).read()
    ii, classes, _ = foldsim_degree.parse_schedule(stext)
    info = foldsim_degree.parse_classes(stext)

    top_share = None
    if mode == 'netlist':
        res = netlist_resources(sim, list(classes), region)
        top_share = res.pop('__top__')
    else:
        cal = json.load(open(os.path.join(HERE, 'foldsim_v80.json')))
        entries = a.register_state_entries
        if entries is None:
            m = re.search(r'register-state-entries=(\d+)', fold)
            entries = int(m.group(1)) if m else 2
        of = {c: re.sub(r'_copy\d+$', '', c) for c in classes}
        est = foldsim_degree.class_estimates(models, sorted(set(of.values())))
        res = {c: dict(lut=foldsim_degree.copy_sites(cal, est[of[c]], of[c], n, entries, info.get(c), ii) / spl,
                       ff=0.0, uram=0.0, bram=0.0) for c, n in classes.items()}
    # The simulator level's own logic and unowned memories spread over the region: each SLR keeps its share free.
    free = [dict(c) for c in caps]
    if top_share:
        tot = {r: sum(c[r] for c in caps) for r in RESOURCES}
        for c in free:
            for r in RESOURCES:
                if tot[r]:
                    c[r] -= top_share[r] * c[r] / tot[r]

    lines = [f'{name}: SLRs {",".join(names)}, resources from the {mode}'
             + (f' (LUT x{region.get("netlist_lut_scale", 1.0)}, FF x{region.get("netlist_ff_scale", 1.0)})'
                if mode == 'netlist' else '') + f'; margin {region.get("margin", 1.0)}; SLL budget {budget} a '
             f'boundary ({region.get("sll_per_boundary")} x {region.get("sll_margin")})']
    for c in caps:
        lines.append(f'  {c["name"]} capacity: ' + ', '.join(f'{r} {fmt(r, c[r])}' for r in RESOURCES))
    if top_share:
        lines.append('  simulator level (spread): ' + ', '.join(f'{r} {fmt(r, top_share[r])}' for r in RESOURCES))
    # 1. Each copy whole within an SLR: any of them, or the one named.
    if a.copy_cap == 'any':
        allowed = caps
    elif a.copy_cap in ('smallest', 'largest'):
        allowed = [(min if a.copy_cap == 'smallest' else max)(caps, key=lambda c: c['lut'])]
    else:
        allowed = [c for c in caps if c['name'] == a.copy_cap]
        if not allowed:
            sys.exit(f'--copy-cap: no SLR {a.copy_cap} in the region')
    fill = lambda c, s: max(res[c][r] / s[r] if s[r] else (math.inf if res[c][r] else 0.0) for r in RESOURCES)
    big = sorted(res, key=lambda c: -lut_equivalent(res[c], caps))
    over = []
    lines.append(f'  copy cap: {a.copy_cap} (' + ', '.join(s['name'] for s in allowed) + ')')
    for c in big[:6]:
        lines.append(f'    {c}: ' + ', '.join(f'{r} {fmt(r, res[c][r])}' for r in RESOURCES)
                     + '; of each SLR\'s capacity (binding resource): '
                     + ', '.join(f'{s["name"]} {100 * fill(c, s):.0f} %' for s in caps))
    for c in res:
        if all(fill(c, s) > 1 + 1e-9 for s in allowed):
            s = min(allowed, key=lambda s: fill(c, s))
            over.append((c, s['name'], [r for r in RESOURCES if res[c][r] > s[r] + 1e-9]))
    # 2. and 3. An assignment within every SLR's capacity in each resource (fold-static sll-estimate: LUT sites,
    # then flops, URAM and RAMB36 in tenths as further resources); its SLRs then shown resource by resource.
    def vec(x, sites):
        return '/'.join([str(max(0, round(sites)))] + [str(max(0, round(10 * x[r] if r != 'ff' else x[r])))
                                                      for r in RESOURCES[1:]])
    report = os.path.join(out, 'sll.txt')
    opts = (f'{fold} schedule-only report={report} sll-estimate slr-sll-budget={budget} slr-capacities='
            + ','.join(vec(c, spl * c['lut']) for c in free) + ' slr-sites='
            + ','.join(f'{c}:{vec(res[c], max(1, spl * res[c]["lut"]))}' for c in classes))
    run([a.foldsim_opt, models, f'--pass-pipeline=builtin.module(foldsim-fold-static{{{opts}}})', '-o', os.devnull])
    text = open(report).read()
    est_lines = text[text.index('  sll estimate:'):].split('\n')
    sec = [est_lines[0]] + list(takewhile(lambda l: l.startswith('    '), est_lines[1:]))
    m = re.search(r'\(worst (\d+), total (\d+)\)', '\n'.join(sec))
    worst = int(m.group(1)) if m else None
    lines += sec
    assign_bad = []
    if worst is not None:
        for g, s in enumerate(free):
            mm = re.search(rf'^    slr{g} \(\d+ sites\):(.*)$', '\n'.join(sec), re.M)
            members = mm.group(1).split() if mm else []
            use = {r: sum(res[c][r] for c in members) for r in RESOURCES}
            lines.append(f'    slr{g} = {s["name"]}: ' + ', '.join(
                f'{r} {fmt(r, use[r])} / {fmt(r, s[r])}' for r in RESOURCES))
            bad = [r for r in RESOURCES if use[r] > s[r] + 1e-9]
            if bad:
                assign_bad.append((s['name'], bad))
    largest = big[0]
    lines.append(f'{name}: largest class copy {largest}: ' + ', '.join(
        f'{s["name"]} {100 * max(res[largest][r] / s[r] if s[r] else 0 for r in RESOURCES):.0f} %' for s in caps)
        + ' of the SLR\'s capacity (its binding resource)')
    if over:
        verdict = 'REFUSE: ' + '; '.join(f'{c} fits no SLR whole (the nearest, {s}: over its {"/".join(r)})'
                                         for c, s, r in over) + \
            ' (a copy past its SLR is cut by the placer, its SLLs unknown: r16c failed so)'
    elif worst is None:
        verdict = 'REFUSE: no assignment of the copies within the SLRs\' capacities'
    elif assign_bad:
        verdict = 'REFUSE: the assignment found is over ' + '; '.join(
            f'{s}\'s {"/".join(r)}' for s, r in assign_bad)
    else:
        # The SLLs the worst boundary takes as placed: its whole-copy link bits
        # times the region's sll_per_link_bit (a placed build's SLLs over its
        # estimate: the placer splits the copies where their own cut is small
        # and a wide link reads a word at a time); without it, the link bits.
        per = region.get('sll_per_link_bit')
        sll = worst * per if per else worst
        lines.append(f'{name}: worst boundary {worst} whole-copy link bits' + (
            f' -> {sll:,.0f} SLLs projected (x{per})' if per else '') +
            f'; budget {budget}, the boundary\'s {region.get("sll_per_boundary")}')
        if sll > region.get('sll_per_boundary', budget):
            verdict = f'REFUSE: {sll:,.0f} SLLs projected past the boundary\'s {region.get("sll_per_boundary")}'
        elif sll > budget:
            verdict = f'WARN: {sll:,.0f} SLLs projected over the {budget} budget'
        else:
            verdict = f'OK: {sll:,.0f} SLLs projected within the {budget} budget'
    lines.append(f'{name}: {verdict}')
    print('\n'.join(lines))
    with open(report, 'a') as f:
        f.write('\n'.join(lines) + '\n')
    sys.exit(1 if verdict.startswith('REFUSE') else 0)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr[-3000:] if e.stderr else str(e))
        sys.exit(2)
