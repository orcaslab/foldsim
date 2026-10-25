#!/usr/bin/env python3
"""Choosing the fold degree: how many copies each class is folded into.

A class of n instances folded into k copies (k a power of two dividing n)
costs k copies of its logic and k stores of its state, each n/k instances deep,
and the frame it allows. For every choice of k per class:

  II      exact: the scheduler's (foldsim-fold-static schedule-only);
  LUTs    the IR's estimate per class (foldsim_cost.py), calibrated per class
          against Vivado's hierarchical utilization of a build of the design;
  sites   the LUT sites the placer needs, distributed RAM weighted, against
          the V80 dynamic region (fits);
  period  a fit of the routed period against the largest copy's LUTs and,
          with a logic screen of every build, the netlist's depth.

A simulated cycle takes II x period ns: since the memory endpoint is in the
frame, H = II. Reports the Pareto front of ns per simulated cycle against LUTs,
the fastest point that places within a LUT budget, and the unfolded target's
area (what an unfolded simulator needs at the least). The model ranks and
prunes; it does not promise a clock: retiming and routing dominate the period.

usage:
  foldsim_degree.py calibrate --models <models.mlir> --schedule <schedule.txt>
      --util <routed_util_hier.rpt> --timing <routed_timing_summary.rpt>
      --period <ns> --name <build> --out <calibration.json> [--screen <dir>]
  foldsim_degree.py fit --calibration <calibration.json>
  foldsim_degree.py explore --models <models.mlir> --module <top> --workdir <dir>
      [--calibration F] [--screen <dir>]
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


# The V80 dynamic region's LUT sites as the placer counts them, and what a LUT
# of distributed RAM costs there in sites: a RAM64M8 claims its whole SLICEM
# however few bits it holds. Fitted to the one full image that failed to
# place (1,094,665 logic + 547,062 LUTRAM LUTs needed 3,326,213 sites of
# 3,269,043); images at 1.9M sites placed.
PLACE_SITES = 3269043
LUTRAM_SITES = 4.08
# The V80's block RAMs (RAMB36E5).
BRAM_SITES = 3741
# Placement is not routing: keep a margin for a design that also routes.
PLACE_MARGIN = 0.9


def placement(logic, lutram, bram=0.0):
    """(LUT sites the placer needs, whether they and the block RAMs fit the
    region with margin)."""
    sites = logic + LUTRAM_SITES * lutram
    return sites, sites <= PLACE_MARGIN * PLACE_SITES and bram <= PLACE_MARGIN * BRAM_SITES


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
            totals = dict(logic=logic, lutram=lutram)
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
    return copies, rest, total, totals


def read_wns(path):
    text = open(path).read()
    m = re.search(r'Design Timing Summary.*?\n(?:.*\n){5}\s*(-?\d+\.\d+)', text)
    return float(m.group(1))


def calibrate(a):
    ii, classes, _ = parse_schedule(open(a.schedule).read())
    est = class_estimates(a.models, classes)
    copies, rest, total, totals = read_util(a.util)
    period = a.period - read_wns(a.timing)
    cal = json.load(open(a.out)) if os.path.exists(a.out) else {}
    build = dict(name=a.name, luts=total, rest=rest, period=period, classes={}, **totals)
    if a.screen:
        build['depth'] = screen_depth(a.screen)
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


def area(cal, est, classes, degree, rest):
    """LUTs of one fold degree ({class: copies}, 1 when absent): logic and
    distributed RAM apart, the sites they take, and the largest copy."""
    logic = lutram = bram = largest = 0.0
    for c, n in classes.items():
        k = degree.get(c, 1)
        kappa, lam = factors(cal, c, est[c]['lut'])
        lr, br = state_lutram(est[c], n // k)
        logic += k * kappa * est[c]['lut']
        lutram += k * lam * lr
        bram += k * br
        largest = max(largest, kappa * est[c]['lut'] + lam * lr)
    logic += rest
    sites, fits = placement(logic, lutram, bram)
    return dict(luts=round(logic + lutram), logic=round(logic), lutram=round(lutram),
                sites=round(sites), fits=fits, bram=round(bram, 1), largest_copy=round(largest))


# The screen's measure of a netlist's depth for the period fit: log2 of the end
# points at least this many levels deep. A count, not the deepest path: the
# few deepest are what Vivado retimes or optimizes away (a modulo by a
# constant), a unit deep throughout is not.
DEPTH_LEVELS = 14


def screen_depth(screen):
    """Depth of a screened netlist (ci/logic_screen.py screen <dir>)."""
    import csv
    deep = sum(1 for r in csv.DictReader(open(f'{screen}/endpoints.csv'))
               if int(r['levels']) >= DEPTH_LEVELS)
    return round(math.log2(max(deep, 1)), 3)


def lstsq(rows, ys):
    """Least squares with the normal equations (few points, few terms)."""
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[p], b[i], b[p] = a[p], a[i], b[p], b[i]
        if abs(a[i][i]) < 1e-12:
            return None
        for r in range(n):
            if r != i:
                f = a[r][i] / a[i][i]
                a[r] = [x - f * y for x, y in zip(a[r], a[i])]
                b[r] -= f * b[i]
    return [b[i] / a[i][i] for i in range(n)]


TERMS = {'const': (), 'size': ('size',), 'size+depth': ('size', 'depth')}


def fit_row(terms, largest, depth):
    return [1.0] + [math.log2(max(largest, 1.0)) if t == 'size' else depth for t in terms]


def fit_terms(builds, terms):
    """Least squares of the period over the builds, and each build's
    leave-one-out error (empty when the builds are too few)."""
    rows = [fit_row(terms, b['max_copy'], b.get('depth', 0.0)) for b in builds]
    ys = [b['period'] for b in builds]
    coef = lstsq(rows, ys)
    if coef is None:
        return None
    loo = {}
    if len(builds) > len(coef) + 1:
        for i, b in enumerate(builds):
            c = lstsq(rows[:i] + rows[i + 1:], ys[:i] + ys[i + 1:])
            if c is None:
                return dict(coef=coef, terms=terms, loo={}, rms=None)
            loo[b['name']] = round(sum(x * y for x, y in zip(c, rows[i])) - ys[i], 3)
    rms = math.sqrt(sum(e * e for e in loo.values()) / len(loo)) if loo else None
    return dict(coef=coef, terms=terms, loo=loo, rms=rms and round(rms, 3))


def period_fit(cal, use_depth=True):
    """The period fit over the calibrated builds: of a constant, the largest
    copy's size, and size plus the screen's depth (with `use_depth` and a
    screen of every build), the one with the least leave-one-out error; size
    alone when the builds are too few to tell."""
    builds = (cal or {}).get('builds', [])
    if not builds:
        return None
    names = ['const', 'size'] + (['size+depth'] if use_depth and all(
        'depth' in b for b in builds) else [])
    fits = [f for f in (fit_terms(builds, TERMS[n]) for n in names) if f]
    scored = [f for f in fits if f['rms'] is not None]
    if scored:
        return min(scored, key=lambda f: f['rms'])
    return next((f for f in fits if f['terms'] == TERMS['size']), fits[0])


def fit_period(fit, largest, depth):
    """A candidate's period from the fit."""
    return sum(x * y for x, y in zip(fit['coef'], fit_row(fit['terms'], largest, depth or 0.0)))


def pow2_divisors(n):
    return [k for k in (1 << j for j in range(n.bit_length())) if n % k == 0]


def evaluate(models, base_schedule, fold_options, foldsim_opt, workdir, cal,
             jobs=4, budget=0, depth=None):
    """Every choice of copies per class: II, LUTs, whether it places, period
    and ns per cycle; the Pareto front, and the fastest point that places and
    stays within `budget` LUTs (0: none). `depth` is the design's screen depth
    (screen_depth), for a fit that has the depth term."""
    _, classes, _ = parse_schedule(base_schedule)
    est = class_estimates(models, classes)
    rest = rest_of(cal, sum(est[c]['lut'] for c in classes))
    names = [c for c, n in classes.items() if n > 1]
    choices = [pow2_divisors(classes[c]) for c in names]
    os.makedirs(workdir, exist_ok=True)
    fit = period_fit(cal, use_depth=depth is not None)

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
        ii, _, _ = parse_schedule(open(report).read())
        res = dict(copies=copies or '-', ii=ii,
                   **area(cal, est, classes, dict(zip(names, ks)), rest))
        if fit:
            period = fit_period(fit, res['largest_copy'], depth)
            res.update(period=round(period, 3), mhz=round(1000 / period, 1),
                       ns=round(ii * period, 1))
        return res

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        points = list(pool.map(one, itertools.product(*choices)))
    good = [p for p in points if 'ii' in p]
    key = 'ns' if fit else 'ii'
    front = []
    for p in sorted(good, key=lambda p: (p['luts'], p[key])):
        if not front or p[key] < front[-1][key]:
            front.append(p)
    within = [p for p in good if p['fits'] and (not budget or p['luts'] <= budget)]
    pick = min(within, key=lambda p: (p[key], p['luts'])) if within else None
    # Every instance its own copy and its state in flip-flops: the target's
    # own area, which an unfolded simulator (FireSim) needs at the least.
    unfolded = area(cal, est, classes, dict(classes), rest)
    return dict(classes=classes, points=points, front=front, pick=pick, budget=budget,
                unfolded=unfolded, depth=depth, fit=fit)


def table(result):
    lines = []
    fit = result['fit']
    cols = ['copies', 'ii', 'luts', 'lutram', 'sites', 'fits', 'bram', 'largest_copy'] + \
        (['mhz', 'ns'] if fit else [])
    lines.append(' '.join(f'{c:>12}' if c != 'copies' else f'{c:<60}' for c in cols))
    front = {id(p) for p in result['front']}
    for p in sorted(result['points'], key=lambda p: p.get('luts', 0)):
        if 'error' in p:
            lines.append(f"{p['copies']:<60} error: {' '.join(p['error'])}")
            continue
        mark = (' *' if id(p) in front else '') + (' <- pick' if p is result['pick'] else '')
        lines.append(' '.join(f'{str(p[c]):>12}' if c != 'copies' else f'{p[c]:<60}'
                              for c in cols) + mark)
    u = result['unfolded']
    lines.append(f"{'unfolded (every instance its own copy)':<60} {'-':>12} {u['luts']:>12} "
                 f"{u['lutram']:>12} {u['sites']:>12} {str(u['fits']):>12} {u['bram']:>12}")
    lines.append('* Pareto front (ns per simulated cycle against LUTs); '
                 f"budget {result['budget'] or 'none'}; fits: {PLACE_MARGIN:.0%} of "
                 f'{PLACE_SITES} V80 LUT sites (a LUTRAM LUT {LUTRAM_SITES} sites) and of '
                 f'{BRAM_SITES} RAMB36')
    if fit:
        c = fit['coef']
        terms = ' + '.join([f'{c[0]:.3f}'] + [
            f'{x:.3f} ' + ('log2(largest copy LUTs)' if t == 'size' else
                           f'log2(end points >= {DEPTH_LEVELS} levels) [{result["depth"]}]')
            for x, t in zip(c[1:], fit['terms'])])
        lines.append(f'period = {terms} ns; leave-one-out RMS {fit["rms"]} ns {fit["loo"] or ""}')
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
    c.add_argument('--screen', default='',
                   help='the logic screen of the build\'s netlist (ci/logic_screen.py screen)')
    e = sub.add_parser('explore', help='the front of a models IR')
    e.add_argument('--models', required=True)
    e.add_argument('--module', required=True)
    e.add_argument('--foldsim-opt', default='foldsim-opt')
    e.add_argument('--workdir', required=True)
    e.add_argument('--calibration', default='')
    e.add_argument('--lut-budget', type=int, default=0)
    e.add_argument('--jobs', type=int, default=4)
    e.add_argument('--screen', default='',
                   help='the logic screen of the design\'s netlist, for the depth term')
    f = sub.add_parser('fit', help='the period fit of a calibration and its leave-one-out error')
    f.add_argument('--calibration', required=True)
    a = ap.parse_args()
    if a.cmd == 'fit':
        cal = json.load(open(a.calibration))
        for name, terms in TERMS.items():
            if 'depth' in terms and not all('depth' in b for b in cal['builds']):
                continue
            print(name, fit_terms(cal['builds'], terms))
        print('chosen', period_fit(cal)['terms'] or 'const')
        return 0
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
                      a.jobs, a.lut_budget, screen_depth(a.screen) if a.screen else None)
    print(table(result))
    json.dump(result, open(os.path.join(a.workdir, 'degree.json'), 'w'), indent=1,
              default=str)


if __name__ == '__main__':
    main()
