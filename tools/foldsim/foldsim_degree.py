#!/usr/bin/env python3
"""Choosing the fold degree: how many copies each class is folded into.

A class of n instances folded into k copies (k a power of two dividing n)
costs k copies of its logic and k stores of its state, each n/k instances deep,
and the frame it allows. For every choice of k per class:

  II      exact: the scheduler's (foldsim-fold-static schedule-only);
  H now   max(II, the last output's slot + 8): the frame's round trip through
          the memory endpoint's tokens, until the endpoint is scheduled in it;
  LUTs    the IR's estimate per class (foldsim_cost.py), calibrated per class
          against Vivado's hierarchical utilization of a build of the design;
  period  a fit of the routed period against the largest copy's LUTs, over
          the calibrated builds (constant with one build, none without).

A simulated cycle takes II x period ns (H now x period today). Reports the
Pareto front of ns per simulated cycle against LUTs and picks the fastest
point within a LUT budget. The model ranks and prunes; it does not promise a
clock: the period term is the weak one (retiming and routing dominate it).

usage:
  foldsim_degree.py calibrate --models <models.mlir> --schedule <schedule.txt>
      --util <routed_util_hier.rpt> --timing <routed_timing_summary.rpt>
      --period <ns> --name <build> --out <calibration.json>
  foldsim.py ... --static --copies auto [--lut-budget N] [--degree-calibration F]
"""
import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import foldsim_cost as cost  # noqa: E402


def parse_schedule(text):
    """II, each class's instance count, and the last output's slot."""
    ii = int(re.search(r'II = (\d+)', text).group(1))
    classes = {m.group(1): int(m.group(2))
               for m in re.finditer(r'^  (\S+) x(\d+)', text, re.M)}
    slots = [int(s) for s in re.findall(r'^  output \S+ @(\d+)', text, re.M)]
    return ii, classes, max(slots) if slots else 0


def class_estimates(models, names):
    """Per class, one instance: logic LUTs (black boxes included), register
    bits and the memories, from the models' IR."""
    mods, _ = cost.parse(models)
    out = {}
    for name in names:
        acc = dict(lut=0.0, ff=0, mems=[])

        def visit(n):
            m = mods[n]
            acc['lut'] += sum(m.lut.values())
            acc['ff'] += m.ff
            acc['mems'] += [list(r) for r in m.mems]
            for child, _ in m.insts:
                base = child.split('__')[0]
                if child in cost.BLACKBOX or base in cost.BLACKBOX:
                    acc['lut'] += cost.BLACKBOX.get(child, cost.BLACKBOX.get(base))[0]
                elif child in mods:
                    visit(child)
        if name in mods:
            visit(name)
        out[name] = acc
    return out


def state_lutram(est, threads):
    """Distributed-RAM LUTs of a copy's state at `threads` instances: its
    registers as memories of one entry per instance (flip-flops for one), and
    its memories `threads` times deeper."""
    luts = 0.0 if threads == 1 else cost.mem_cost(threads, est['ff'], 1, 0)[0]
    bram = 0.0
    for (_, d, w, r, _, rl, st) in est['mems']:
        lr, br = cost.mem_cost(d * threads, w, r, rl, st)
        luts += lr
        bram += br
    return luts, bram


def read_util(path):
    """Measured LUTs of each class's copies (logic, distributed RAM) and of the
    rest, from a hierarchical utilization report of a foldsim_afu build."""
    copies, rest = {}, 0
    for line in open(path):
        f = line.split('|')
        if len(f) < 7 or not line.startswith('|'):
            continue
        name = f[1].rstrip()
        depth = len(name) - len(name.lstrip())
        try:
            lut, logic, lutram = int(f[3]), int(f[4]), int(f[5])
        except ValueError:
            continue
        name = name.strip()
        if depth == 1:
            total = lut
        if depth != 5:
            continue
        m = re.match(r'(.+?)(_copy\d+)?_copy$', name)
        if m:
            c = copies.setdefault(m.group(1), dict(copies=0, logic=0, lutram=0, lut=0))
            c['copies'] += 1
            c['logic'] += logic
            c['lutram'] += lutram
            c['lut'] += lut
    rest = total - sum(c['lut'] for c in copies.values())
    return copies, rest, total


def read_wns(path):
    text = open(path).read()
    m = re.search(r'Design Timing Summary.*?\n(?:.*\n){5}\s*(-?\d+\.\d+)', text)
    return float(m.group(1))


def calibrate(a):
    ii, classes, _ = parse_schedule(open(a.schedule).read())
    est = class_estimates(a.models, classes)
    copies, rest, total = read_util(a.util)
    period = a.period - read_wns(a.timing)
    cal = json.load(open(a.out)) if os.path.exists(a.out) else {}
    build = dict(name=a.name, luts=total, rest=rest, period=period, classes={})
    for name, n in classes.items():
        base = re.sub(r'_copy\d+$', '', name)
        m = copies.get(base)
        if not m or name != base:
            continue
        k = m['copies']
        t = n
        e = est[name]
        lr, _ = state_lutram(e, t)
        build['classes'][name] = dict(
            copies=k, threads=t, logic=m['logic'] / k, lutram=m['lutram'] / k,
            est_logic=e['lut'], est_lutram=lr,
            kappa=(m['logic'] / k) / e['lut'] if e['lut'] else 1.0,
            lam=(m['lutram'] / k) / lr if lr else 1.0)
    build['max_copy'] = max(m['lut'] / m['copies'] for m in copies.values())
    cal.setdefault('builds', [b for b in cal.get('builds', []) if b['name'] != a.name])
    cal['builds'] = [b for b in cal['builds'] if b['name'] != a.name] + [build]
    json.dump(cal, open(a.out, 'w'), indent=1)
    for name, c in build['classes'].items():
        print(f"{a.name} {name}: logic {c['logic']:.0f} (est {c['est_logic']:.0f}, "
              f"x{c['kappa']:.2f}), lutram {c['lutram']:.0f} (est {c['est_lutram']:.0f}, "
              f"x{c['lam']:.2f})")
    print(f"{a.name}: {total} LUTs, rest {rest}, period {period:.3f} ns, "
          f"largest copy {build['max_copy']:.0f}")


def factors(cal, name, est_lut):
    """A class's calibration: from the build whose estimate of it is nearest
    (the same design at another degree, else the nearest size), else the
    median over every class and build, else none."""
    builds = (cal or {}).get('builds', [])
    near = [(abs(math.log(c['est_logic'] / est_lut)), c) for b in builds
            for n, c in b['classes'].items() if n == name and c['est_logic'] and est_lut]
    if near:
        c = min(near, key=lambda x: x[0])[1]
        return c['kappa'], c['lam']
    ks = sorted(c['kappa'] for b in builds for c in b['classes'].values())
    ls = sorted(c['lam'] for b in builds for c in b['classes'].values())
    if ks:
        return ks[len(ks) // 2], ls[len(ls) // 2]
    return 1.0, 1.0


def rest_of(cal, est_total):
    """The LUTs outside the copies (memory endpoint, links, control) of the
    build nearest in estimated size."""
    builds = (cal or {}).get('builds', [])
    if not builds:
        return 0.0
    def size(b):
        return sum(c['est_logic'] * c['copies'] for c in b['classes'].values())
    return min(builds, key=lambda b: abs(math.log(max(size(b), 1) / max(est_total, 1))))['rest']


def period_fit(cal):
    """Least squares of period = p0 + p1 log2(largest copy LUTs), p1 >= 0."""
    pts = [(math.log2(b['max_copy']), b['period']) for b in (cal or {}).get('builds', [])]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0][1], 0.0, pts
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    p1 = max(0.0, sum((x - mx) * (y - my) for x, y in pts) / sxx) if sxx else 0.0
    return my - p1 * mx, p1, pts


def pow2_divisors(n):
    return [k for k in (1 << j for j in range(n.bit_length())) if n % k == 0]


def evaluate(models, base_schedule, fold_options, foldsim_opt, workdir, cal,
             jobs=4, budget=0):
    """Every choice of copies per class: II, LUTs, period, ns per cycle; the
    Pareto front, and the fastest point within `budget` LUTs (0: none)."""
    _, classes, _ = parse_schedule(base_schedule)
    est = class_estimates(models, classes)
    rest = rest_of(cal, sum(est[c]['lut'] for c in classes))
    names = [c for c, n in classes.items() if n > 1]
    choices = [pow2_divisors(classes[c]) for c in names]
    os.makedirs(workdir, exist_ok=True)
    fit = period_fit(cal)

    def one(ks):
        copies = ','.join(f'{c}:{k}' for c, k in zip(names, ks) if k > 1)
        tag = '_'.join(str(k) for k in ks) or 'base'
        report = os.path.join(workdir, f'schedule_{tag}.txt')
        opts = fold_options + f' report={report} schedule-only'
        if copies:
            opts += f' copies={copies}'
        r = subprocess.run([foldsim_opt, models,
                            f'--pass-pipeline=builtin.module(foldsim-fold-static{{{opts}}})',
                            '-o', os.devnull], capture_output=True, text=True)
        if r.returncode:
            return dict(copies=copies or '-', error=r.stderr.strip().splitlines()[-1:])
        ii, _, out_slot = parse_schedule(open(report).read())
        luts = bram = 0.0
        largest = 0.0
        for c, n in classes.items():
            k = dict(zip(names, ks)).get(c, 1)
            kappa, lam = factors(cal, c, est[c]['lut'])
            lr, br = state_lutram(est[c], n // k)
            per_copy = kappa * est[c]['lut'] + lam * lr
            luts += k * per_copy
            bram += k * br
            largest = max(largest, per_copy)
        luts += rest
        h_now = max(ii, out_slot + 8)
        res = dict(copies=copies or '-', ii=ii, h_now=h_now, luts=round(luts),
                   bram=round(bram, 1), largest_copy=round(largest))
        if fit:
            p0, p1, _ = fit
            period = p0 + p1 * math.log2(max(largest, 1.0))
            res.update(period=round(period, 3), mhz=round(1000 / period, 1),
                       ns=round(ii * period, 1), ns_now=round(h_now * period, 1))
        return res

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        points = list(pool.map(one, itertools.product(*choices)))
    good = [p for p in points if 'ii' in p]
    key = 'ns' if fit else 'ii'
    front = []
    for p in sorted(good, key=lambda p: (p['luts'], p[key])):
        if not front or p[key] < front[-1][key]:
            front.append(p)
    within = [p for p in good if not budget or p['luts'] <= budget]
    pick = min(within, key=lambda p: (p[key], p['luts'])) if within else None
    return dict(classes=classes, points=points, front=front, pick=pick,
                budget=budget, fit=fit and dict(p0=fit[0], p1=fit[1], points=fit[2]))


def table(result):
    lines = []
    fit = result['fit']
    cols = ['copies', 'ii', 'h_now', 'luts', 'bram', 'largest_copy'] + \
        (['mhz', 'ns', 'ns_now'] if fit else [])
    lines.append(' '.join(f'{c:>12}' if c != 'copies' else f'{c:<60}' for c in cols))
    front = {id(p) for p in result['front']}
    for p in sorted(result['points'], key=lambda p: p.get('luts', 0)):
        if 'error' in p:
            lines.append(f"{p['copies']:<60} error: {' '.join(p['error'])}")
            continue
        mark = (' *' if id(p) in front else '') + (' <- pick' if p is result['pick'] else '')
        lines.append(' '.join(f'{p[c]:>12}' if c != 'copies' else f'{p[c]:<60}'
                              for c in cols) + mark)
    lines.append('* Pareto front (ns per simulated cycle against LUTs); '
                 f"budget {result['budget'] or 'none'}")
    if fit:
        lines.append(f"period = {fit['p0']:.3f} + {fit['p1']:.3f} log2(largest copy LUTs) ns, "
                     f"over {len(fit['points'])} build(s)")
    else:
        lines.append('no calibration: LUTs uncalibrated, no period term (ranked by II)')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('calibrate', help='add a build to a calibration file')
    c.add_argument('--models', required=True)
    c.add_argument('--schedule', required=True)
    c.add_argument('--util', required=True)
    c.add_argument('--timing', required=True)
    c.add_argument('--period', type=float, required=True)
    c.add_argument('--name', required=True)
    c.add_argument('--out', required=True)
    e = sub.add_parser('explore', help='the front of a models IR')
    e.add_argument('--models', required=True)
    e.add_argument('--module', required=True)
    e.add_argument('--foldsim-opt', default='foldsim-opt')
    e.add_argument('--workdir', required=True)
    e.add_argument('--calibration', default='')
    e.add_argument('--lut-budget', type=int, default=0)
    e.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    if a.cmd == 'calibrate':
        return calibrate(a)
    cal = json.load(open(a.calibration)) if a.calibration else None
    fold = f'module={a.module} sim=foldsim_sim'
    base = os.path.join(a.workdir, 'schedule_base.txt')
    os.makedirs(a.workdir, exist_ok=True)
    r = subprocess.run([a.foldsim_opt, a.models,
                        f'--pass-pipeline=builtin.module(foldsim-fold-static{{{fold} '
                        f'report={base} schedule-only}})', '-o', os.devnull],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    result = evaluate(a.models, open(base).read(), fold, a.foldsim_opt, a.workdir, cal,
                      a.jobs, a.lut_budget)
    print(table(result))
    json.dump(result, open(os.path.join(a.workdir, 'degree.json'), 'w'), indent=1,
              default=str)


if __name__ == '__main__':
    main()
