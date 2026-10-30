#!/usr/bin/env python3
"""Choosing the fold degree: how many copies each class is folded into.

A class of n instances folded into k copies costs k copies of its logic and
k stores of its state, each about n/k instances deep (the first n mod k one
deeper), and the frame it allows: the deepest copy's. For every choice of k per class:

  II      exact: the scheduler's (foldsim-fold-static schedule-only);
  LUTs    the IR's estimate per class (foldsim_cost.py), calibrated per class
          against Vivado's hierarchical utilization of a build of the design;
  sites   the LUT sites the placer needs, distributed RAM weighted, against
          the V80 dynamic region (fits);
  period  a fit of the routed period against the largest copy's LUTs and,
          with a logic screen of every build, the netlist's depth;
  storage with --storage auto, a kind per state store (flip-flops,
          distributed, block or ultra RAM) against the region's sites,
          RAMB36 and URAM288 (foldsim_storage.py); fixed: the compiler's
          default rules against the device's RAMB36;
  routes  the routability term (foldsim_storage.py): the router's congestion
          fitted on the V80 builds against LUTs and block RAMs and the SLRs
          they spread over (the region's, fewer with fewer SLR groups); a pick
          must route as well as place (--no-routability: place only).

A simulated cycle takes II x period ns: since the memory endpoint is in the
frame, H = II. Reports the Pareto front of ns per simulated cycle against LUTs,
the fastest point that places within a LUT budget, and the unfolded target's
area (what an unfolded simulator needs at the least). The model ranks and
prunes; it does not promise a clock: retiming and routing dominate the period.

Every choice is scheduled while there are few; lanes folded as classes (256
instances each) make tens of thousands, and those are searched (evaluate).

usage:
  foldsim_degree.py calibrate --models <models.mlir> --schedule <schedule.txt>
      --util <routed_util_hier.rpt> --timing <routed_timing_summary.rpt>
      --period <ns> --name <build> --out <calibration.json> [--screen <dir>]
  foldsim_degree.py fit --calibration <calibration.json>
  foldsim_degree.py explore --models <models.mlir> --module <top> --workdir <dir>
      [--calibration F] [--screen <dir>] [--storage fixed|auto] [--no-routability]
      [--no-uram] [--slr-groups] [--schedules <dir> [--cached-only]]
  foldsim.py ... --static --copies auto [--storage auto] [--lut-budget N]
      [--degree-calibration F]
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
import foldsim_storage as storage  # noqa: E402


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
        acc = dict(lut=0.0, ff=0, mems=[], regw={})

        def visit(n):
            m = mods[n]
            acc['lut'] += sum(m.lut.values())
            acc['ff'] += m.ff
            acc['mems'] += [list(r) for r in m.mems]
            for w, k in m.regw.items():
                acc['regw'][w] = acc['regw'].get(w, 0) + k
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


def state_lutram(est, threads, form='lutram'):
    """Distributed-RAM LUTs of a copy's state at `threads` instances: its
    registers as memories of one entry per instance (flip-flops for one, a
    ring or a store in flops), and its memories `threads` times deeper, an
    entry for each instance index (a power of two)."""
    threads = 1 << (threads - 1).bit_length() if threads > 1 else 1
    luts = 0.0 if form != 'lutram' else cost.mem_cost(threads, est['ff'], 1, 0)[0]
    bram = 0.0
    for (_, d, w, r, _, rl, st, *_) in est['mems']:
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
    text = open(a.schedule).read()
    ii, classes, _ = parse_schedule(text)
    sched = parse_classes(text)
    entries = a.register_state_entries
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
        form, bypass = state_form(t, sched.get(name), ii, entries)
        lr, _ = state_lutram(e, t, form)
        sl = state_logic(cal, form, bypass, e['ff'])
        build['classes'][name] = dict(
            copies=k, threads=t, logic=m['logic'] / k, lutram=m['lutram'] / k,
            est_logic=e['lut'], est_lutram=lr, form=form, state_logic=sl,
            kappa=max(m['logic'] / k - sl, 0) / e['lut'] if e['lut'] else 1.0,
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
    (the same design at another degree, else the nearest size), the latest
    such build of a design built more than once (the compiler it reflects),
    else the median over every class and build, else none."""
    builds = (cal or {}).get('builds', [])
    near = [(round(abs(math.log(c['est_logic'] / est_lut)), 6), -i, c)
            for i, b in enumerate(builds) for n, c in b['classes'].items()
            if n == name and c['est_logic'] and est_lut]
    if near:
        c = min(near, key=lambda x: x[:2])[2]
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


def area(cal, est, classes, degree, rest, sched=None, ii=0, entries=2, store='fixed',
         route=False, uram=True, slr_groups=False):
    """LUTs of one fold degree ({class: copies}, 1 when absent): logic and
    distributed RAM apart, the sites they take, and the largest copy. With the
    degree's schedule report (parse_classes) each copy's state takes the form
    the compiler gives it (state_form). `store`: 'fixed', the compiler's
    default storage (block RAM by Vivado's inference, the rest distributed),
    against the whole device's RAMB36; 'auto', a kind per store
    (storage_area) against the V80 dynamic region. `route`: whether a point
    must also route (the routability term, foldsim_storage.congestion).
    `uram`: whether the storage may use UltraRAM. `slr_groups`: whether each
    copy is kept whole in one SLR (foldsim --slr-groups), which confines the
    fold to as many SLRs as it has copies."""
    slrs = storage.route_slrs(sum(degree.get(c, 1) for c in classes) if slr_groups else 0)
    if store == 'auto':
        res = storage_area(cal, est, classes, degree, rest, sched, ii, entries, route, uram,
                           slrs)
    else:
        res = fixed_area(cal, est, classes, degree, rest, sched, ii, entries)
    if route:
        c = storage.fold_congestion(res['luts'], res['bram'], res.get('uram', 0), slrs)
        res['congestion'] = round(c, 2)
        res['routes'] = c <= storage.ROUTE_LIMIT
    return res


def fixed_area(cal, est, classes, degree, rest, sched=None, ii=0, entries=2):
    logic = lutram = bram = largest = 0.0
    for c, n in classes.items():
        k = degree.get(c, 1)
        kappa, lam = factors(cal, c, est[c]['lut'])
        for j in range(k):
            t = copy_threads(n, k, j)
            info = (sched or {}).get(c if j == 0 else f'{c}_copy{j}')
            form, bypass = state_form(t, info, ii, entries) if sched is not None else \
                ('one' if t == 1 else 'lutram', False)
            lr, br = state_lutram(est[c], t, form)
            copy_logic = kappa * est[c]['lut'] + state_logic(cal, form, bypass, est[c]['ff'])
            logic += copy_logic
            lutram += lam * lr
            bram += br
            largest = max(largest, copy_logic + lam * lr)
    logic += rest
    sites, fits = placement(logic, lutram, bram)
    return dict(luts=round(logic + lutram), logic=round(logic), lutram=round(lutram),
                sites=round(sites), fits=fits, bram=round(bram, 1), largest_copy=round(largest))


def storage_area(cal, est, classes, degree, rest, sched, ii, entries, route, uram=True,
                 slrs=None):
    """area() with a kind chosen per store (foldsim_storage.assign) against
    the V80 dynamic region: its sites, RAMB36 and URAM288 and, with `route`,
    the LUTs that route over `slrs` SLRs (the region's by default). Every
    copy of a class takes its stores' kinds (the compiler's `storage` rules
    are per class); a class's register stores choose by width, and only
    where every copy of several instances keeps them in distributed RAM (no
    flops or ring) with a state gap of 4."""
    logic = rest
    units, copy_logic, base_lutram = [], {}, 0.0
    for c, n in classes.items():
        k = degree.get(c, 1)
        kappa, lam = factors(cal, c, est[c]['lut'])
        forms = []
        for j in range(k):
            t = copy_threads(n, k, j)
            info = (sched or {}).get(c if j == 0 else f'{c}_copy{j}')
            form, bypass = state_form(t, info, ii, entries) if sched is not None else \
                ('one' if t == 1 else 'lutram', False)
            cl = kappa * est[c]['lut'] + state_logic(cal, form, bypass, est[c]['ff'])
            logic += cl
            copy_logic[(c, j)] = cl
            forms.append((j, t, form, info['gap'] if info else None))
        deep = [f for f in forms if f[2] == 'lutram']
        regs_free = all(f[2] in ('lutram', 'one') for f in forms)
        for w, cnt in sorted(est[c]['regw'].items()):
            per = []
            for j, t, _, gap in deep:
                per.append((j, dict(depth=1 << (t - 1).bit_length(), width=w, rports=1, wports=1,
                                    rlat=0, mask=1, reg=True, gap=gap, threads=t), cnt))
            if per:
                units.append(unit(c, f'regs>={w}', per, lam, None if regs_free else 'distributed'))
        mems = {}
        for (name, d, w, r, wr, rl, st, mask, named) in est[c]['mems']:
            mems.setdefault((name, named, d, w, r, wr, rl, st, mask), 0)
            mems[(name, named, d, w, r, wr, rl, st, mask)] += 1
        for (name, named, d, w, r, wr, rl, st, mask), cnt in mems.items():
            per = []
            for j, t, _, gap in forms:
                th = 1 << (t - 1).bit_length() if t > 1 else 1
                per.append((j, dict(depth=d * th, width=w, rports=r, wports=wr, rlat=rl,
                                    mask=mask, reg=False, gap=gap, threads=t), cnt))
            # Vivado's inference at the deepest copy's depth, or the hint.
            th = max(1 << (t - 1).bit_length() if t > 1 else 1 for _, t, _, _ in forms)
            default = 'block' if cost.mem_cost(d * th, w, r, rl, st)[1] else 'distributed'
            # A memory the compiler's rules can name: named, and its name
            # free of the pass options' separators.
            nameable = named and not re.search(r'[,=:{}\s]', name)
            u = unit(c, name, per, lam, None if nameable else default)
            u['default'] = default
            units.append(u)
    if not uram:
        for u in units:
            u['costs'].pop('ultra', None)
    caps = {p: storage.MARGIN * storage.REGION[p] for p in ('sites', 'bram', 'uram')}
    base = dict(sites=logic, luts=logic)
    choice, used = storage.assign(units, base, caps,
                                  dict(luts=storage.route_luts(slrs)) if route else None)
    fits = all(used[p] <= caps[p] + 1e-9 for p in ('sites', 'bram', 'uram'))
    lutram = bram = uram = 0.0
    per_copy = dict(copy_logic)
    rules, floor = [], 0
    for u, ch in zip(units, choice):
        (kind, _), = ch.items()     # one store a unit: never split
        for j, c in u['per'][kind]:
            lutram += c['lutram']
            bram += c['bram']
            uram += c['uram']
            logic += c['lut']
            floor = max(floor, c['ii_floor'])
            per_copy[(u['class'], j)] += c['lut'] + c['lutram']
        u['kind'] = kind
    for c in classes:
        rules += storage_rules(c, [u for u in units if u['class'] == c])
    sites = logic + storage.LUTRAM_SITES * lutram
    return dict(luts=round(logic + lutram), logic=round(logic), lutram=round(lutram),
                sites=round(sites), fits=fits, bram=round(bram, 1), uram=uram,
                largest_copy=round(max(per_copy.values())), storage=','.join(rules),
                **({'ii_floor': floor} if floor > ii else {}))


def unit(cls, store, per, lam, fixed):
    """A store of a class, every copy's: its cost per kind (None where some
    copy cannot take it), each copy's apart for the largest copy."""
    costs, by_copy = {}, {}
    for kind in storage.KINDS:
        total = dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0, ii_floor=0)
        parts = []
        for j, st, cnt in per:
            c = storage.kind_cost(st, kind)
            if c is None:
                total = None
                break
            c = dict(c, lutram=lam * c['lutram'])
            c = {k: (v if k == 'ii_floor' else v * cnt) for k, v in c.items()}
            parts.append((j, c))
            for key in total:
                total[key] = max(total[key], c[key]) if key == 'ii_floor' else total[key] + c[key]
        if total is not None:
            costs[kind] = total
            by_copy[kind] = parts
    return dict(costs=costs, per=by_copy, count=1, fixed=fixed, store=store, **{'class': cls})


def storage_rules(cls, units):
    """The compiler's `storage` rules of a class's choice: its memories whose
    kind is not the default, and its register stores by width (each width
    whose kind differs from the narrower's: the largest bound applies)."""
    rules = []
    regs = sorted((int(u['store'].split('>=')[1]), u['kind']) for u in units
                  if u['store'].startswith('regs>=') and not u['fixed'])
    current = 'distributed'
    for w, kind in regs:
        if kind != current:
            rules.append(f'{cls}:regs>={w}={kind}')
            current = kind
    for u in units:
        if not u['store'].startswith('regs>=') and not u['fixed'] and u['kind'] != u['default']:
            rules.append(f"{cls}:{u['store']}={u['kind']}")
    return rules


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


def firesim_factors(cal):
    """FireSim's LUTs against our estimate of the unfolded target, logic and
    distributed RAM apart: the least over the FireSim builds calibrated, so a
    projection is what FireSim needs at the least."""
    fs = (cal or {}).get('firesim', [])
    if not fs:
        return None
    return dict(logic=min(b['logic'] / b['est_logic'] for b in fs),
                lutram=min(b['lutram'] / max(b['est_lutram'], 1) for b in fs),
                builds=[b['name'] for b in fs])


def firesim_area(cal, unfolded):
    """FireSim's projected LUTs and sites for a design: the unfolded target
    scaled by the FireSim factors (None without FireSim builds)."""
    f = firesim_factors(cal)
    if not f:
        return None
    logic, lutram = f['logic'] * unfolded['logic'], f['lutram'] * unfolded['lutram']
    sites, fits = placement(logic, lutram, unfolded['bram'])
    return dict(luts=round(logic + lutram), lutram=round(lutram), sites=round(sites), fits=fits)


def calibrate_firesim(a):
    """Add a FireSim build of a design: its LUTs from Vivado's utilization
    (the kernel's row) and its period, against the unfolded target's estimate
    from the design's models IR (calibrated as the Foldsim builds are)."""
    row = [a.luts, a.logic, a.lutram] if a.luts else None
    cols = None
    for line in ([] if row else open(a.util)):
        f = [x.strip() for x in line.split('|')]
        if 'Total LUTs' in f:
            cols = [f.index(c) for c in ('Total LUTs', 'Logic LUTs', 'LUTRAMs')]
        elif cols and len(f) > max(cols) and f[1] == a.instance:
            row = [int(f[c].split('(')[0]) for c in cols]
            break
    if row is None:
        sys.exit(f'no row {a.instance} in {a.util}')
    cal = json.load(open(a.out)) if os.path.exists(a.out) else {}
    _, classes, _ = parse_schedule(open(a.schedule).read())
    est = class_estimates(a.models, classes)
    unf = area(cal, est, classes, dict(classes), rest_of(cal, sum(e['lut'] for e in est.values())))
    b = dict(name=a.name, luts=row[0], logic=row[1], lutram=row[2], period=a.period,
             est_logic=unf['logic'], est_lutram=unf['lutram'])
    cal['firesim'] = [x for x in cal.get('firesim', []) if x['name'] != a.name] + [b]
    json.dump(cal, open(a.out, 'w'), indent=1)
    print(f"{a.name}: {row[0]} LUTs (logic {row[1]}, lutram {row[2]}) against the unfolded "
          f"target's {unf['logic']} + {unf['lutram']}: x{row[1] / unf['logic']:.2f} logic, "
          f"x{row[2] / max(unf['lutram'], 1):.2f} lutram, period {a.period} ns")


TERMS = {'const': (), 'size': ('size',), 'size+depth': ('size', 'depth'),
         'copies': ('copies',), 'span': ('span',), 'size+copies': ('size', 'copies'),
         'size+span': ('size', 'span')}


def features(largest, depth, copies, luts):
    """A design's terms for the period fit: log2 of its largest copy's LUTs,
    the screen's depth, its copies over every class, and the share of the
    V80's placement capacity its LUTs take (how far it spreads over the dies)."""
    return dict(size=math.log2(max(largest, 1.0)), depth=depth or 0.0,
                copies=float(copies), span=luts / PLACE_SITES)


def build_features(b):
    return features(b['max_copy'], b.get('depth', 0.0),
                    sum(c['copies'] for c in b['classes'].values()), b['luts'])


def fit_row(terms, feats):
    return [1.0] + [feats[t] for t in terms]


def fit_terms(builds, terms):
    """Least squares of the period over the builds, and each build's
    leave-one-out error (empty when the builds are too few)."""
    rows = [fit_row(terms, build_features(b)) for b in builds]
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
    # A design is priced within the range the builds cover: no term is
    # extrapolated past them.
    span = {t: (min(build_features(b)[t] for b in builds),
                max(build_features(b)[t] for b in builds)) for t in terms}
    return dict(coef=coef, terms=terms, loo=loo, rms=rms and round(rms, 3), range=span)


def period_fit(cal, use_depth=True):
    """The period fit over the calibrated builds: of a constant, the largest
    copy's size, and size plus the screen's depth (with `use_depth` and a
    screen of every build), the one with the least leave-one-out error; size
    alone when the builds are too few to tell. A build marked
    `"period_fit": false` calibrates area only."""
    builds = [b for b in (cal or {}).get('builds', []) if b.get('period_fit', True)]
    if not builds:
        return None
    names = [n for n, t in TERMS.items()
             if 'depth' not in t or (use_depth and all('depth' in b for b in builds))]
    fits = [f for f in (fit_terms(builds, TERMS[n]) for n in names) if f]
    scored = [f for f in fits if f['rms'] is not None]
    if scored:
        return min(scored, key=lambda f: f['rms'])
    return next((f for f in fits if f['terms'] == TERMS['size']), fits[0])


def fit_period(fit, feats):
    """A candidate's period from the fit, each term held to the builds' range."""
    held = {t: min(max(feats[t], lo), hi) for t, (lo, hi) in fit.get('range', {}).items()}
    return sum(x * y for x, y in zip(fit['coef'], fit_row(fit['terms'], {**feats, **held})))


def parse_classes(text):
    """Each class of a schedule report (a copy of a class its own): its
    instances, the state gap, and each instance's ready and state slots."""
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r'^  (\S+) x(\d+)( \(ready pass\))?(?: state gap (\d+))?', line)
        if m:
            cur = out[m.group(1)] = dict(n=int(m.group(2)), ready=[], state=[],
                                         gap=int(m.group(4)) if m.group(4) else None)
            continue
        m = re.match(r'^    (ready|state): ([\d ]+)$', line)
        if m and cur is not None:
            cur[m.group(1)] = [int(x) for x in m.group(2).split()]
    return out


def ring_order(info, ii):
    """Whether a copy keeps its registers in a ring (foldsim-fold-static's
    ringOrder): every pass inside the frame, and each instance's passes after
    the state pass of the instance before it, ending with its own."""
    if info['n'] < 2 or len(info['state']) != info['n']:
        return False
    passes = sorted([(s, 1, i) for i, s in enumerate(info['state'])] +
                    [(s, 0, i) for i, s in enumerate(info['ready'])])
    if any(not 0 <= s < ii for s, _, _ in passes) or \
            len({s for s, _, _ in passes}) != len(passes):
        return False
    start = max(n + 1 for n, (_, pass_, _) in enumerate(passes) if pass_ == 1)
    current = None
    for n in range(len(passes)):
        _, pass_, inst = passes[(start + n) % len(passes)]
        current = inst if current is None else current
        if inst != current:
            return False
        if pass_ == 1:
            current = None
    return current is None


def state_form(threads, info, ii, entries):
    """How a copy of `threads` instances keeps its registers: 'one' (a flop),
    'ring' (a flop per instance, rotated), 'flops' (a store in flops: read
    select) or 'lutram'; and whether its state bypass is built (gap below 3)."""
    bypass = info is not None and info['gap'] is not None and info['gap'] < 3
    if threads == 1:
        return 'one', bypass
    if threads <= entries:
        if info is not None and ring_order(info, ii):
            return 'ring', bypass
        return 'flops', bypass
    return 'lutram', bypass


# Logic LUTs per register bit of a copy's state beyond its one-instance logic,
# by form, without and with the state bypass: the read select and the bypass
# muxes. Routed, the mid slice copy of two instances in distributed RAM with
# the bypass (m_rt4c8, 57.5k logic LUTs) against the small one of one instance
# (s_rt4ro, 34.3k): 23.2k over 25,764 bits (0.90). In flops, synthesized
# alone, 1.7k fewer (0.83). A ring has no select of its registers, but its
# read buffers keep theirs: synthesized alone 49.0k, x0.92 routed (the
# m_rt4c8 copy: 57.5k routed, 62.2k synthesized) 45.1k, 10.8k over the one
# instance's (0.42). Without the bypass a flop store's select is a 2:1 mux
# (0.5, two to a LUT); distributed RAM's read port is the select (0).
# Overridden by a calibration's 'state_logic'.
STATE_LOGIC = dict(one=(0.0, 0.0), ring=(0.0, 0.42), flops=(0.5, 0.83), lutram=(0.0, 0.90))


def state_logic(cal, form, bypass, bits):
    table = dict(STATE_LOGIC, **{k: tuple(v) for k, v in (cal or {}).get('state_logic', {}).items()})
    return table[form][1 if bypass else 0] * bits


def copy_threads(n, k, j):
    """Instances of copy j of a class of n folded into k copies."""
    return n // k + (1 if j < n % k else 0)


def degrees(n):
    """The copy counts worth trying for a class of n instances: the fewest
    copies for each depth of the deepest copy, which bounds the frame."""
    ks, seen = [], set()
    for k in range(1, n + 1):
        deepest = -(-n // k)
        if deepest not in seen:
            seen.add(deepest)
            ks.append(k)
    return ks


def deepest(n, k):
    return -(-n // k)


def by_depth(sizes, choices):
    """One choice per frame depth: every class on the fewest copies whose
    deepest copy is no deeper. The frame is bounded by the deepest copy of
    any class, so a class folded finer than the others only adds copies."""
    depths = sorted({deepest(n, k) for n, ks in zip(sizes, choices) for k in ks})
    return list(dict.fromkeys(
        tuple(next(k for k in ks if deepest(n, k) <= d) for n, ks in zip(sizes, choices))
        for d in depths))


def neighbours(point, choices):
    """The choices one degree finer or coarser in a single class."""
    for i, k in enumerate(point):
        j = choices[i].index(k)
        for m in (j - 1, j + 1):
            if 0 <= m < len(choices[i]):
                yield point[:i] + (choices[i][m],) + point[i + 1:]


def pareto(points, key):
    front = []
    for p in sorted(points, key=lambda p: (p['luts'], p[key])):
        if not front or p[key] < front[-1][key]:
            front.append(p)
    return front


def evaluate(models, base_schedule, fold_options, foldsim_opt, workdir, cal,
             jobs=4, budget=0, depth=None, max_points=256, store='fixed', route=False,
             cache=None, cached_only=False, uram=True, slr_groups=False):
    """Every choice of copies per class: II, LUTs, whether it places, period
    and ns per cycle; the Pareto front, and the fastest point that places and
    stays within `budget` LUTs (0: none). `depth` is the design's screen depth
    (screen_depth), for a fit that has the depth term. `store` and `route`:
    the storage choice and the routability term (area); with `route` the
    pick must also route; `slr_groups`: each copy whole in one SLR (area).
    `cache`: a directory of schedule reports of the same models and options
    (an earlier explore's workdir), read instead of scheduling; with
    `cached_only`, its choices are the only ones priced.

    Past `max_points` choices (lanes folded as classes: 256 instances each),
    every choice is too many to schedule; the search then schedules one choice
    per frame depth (by_depth), then the neighbours of the front until the
    front stops changing or `max_points` are scheduled."""
    _, classes, _ = parse_schedule(base_schedule)
    m = re.search(r'register-state-entries=(\d+)', fold_options)
    entries = int(m.group(1)) if m else 2
    est = class_estimates(models, classes)
    rest = rest_of(cal, sum(est[c]['lut'] for c in classes))
    names = [c for c, n in classes.items() if n > 1]
    choices = [degrees(classes[c]) for c in names]
    os.makedirs(workdir, exist_ok=True)
    fit = period_fit(cal, use_depth=depth is not None)

    def one(ks):
        copies = ','.join(f'{c}:{k}' for c, k in zip(names, ks) if k > 1)
        tag = '_'.join(str(k) for k in ks) or 'base'
        report = os.path.join(workdir, f'schedule_{tag}.txt')
        cached = cache and os.path.join(cache, f'schedule_{tag}.txt')
        if cached and os.path.exists(cached):
            report = cached
        else:
            opts = fold_options + f' report={report} schedule-only'
            if copies:
                opts += f' copies={copies}'
            r = subprocess.run([foldsim_opt, models,
                                f'--pass-pipeline=builtin.module(foldsim-fold-static{{{opts}}})',
                                '-o', os.devnull], capture_output=True, text=True)
            if r.returncode:
                return dict(copies=copies or '-', error=r.stderr.strip().splitlines()[-1:])
        text = open(report).read()
        ii, _, _ = parse_schedule(text)
        res = dict(copies=copies or '-', ii=ii,
                   **area(cal, est, classes, dict(zip(names, ks)), rest,
                          parse_classes(text), ii, entries, store, route, uram, slr_groups))
        if fit:
            period = fit_period(fit, features(res['largest_copy'], depth, sum(ks),
                                              res['luts']))
            res.update(period=round(period, 3), mhz=round(1000 / period, 1),
                       ns=round(ii * period, 1))
        return res

    key = 'ns' if fit else 'ii'
    space = math.prod(len(ks) for ks in choices)
    done = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        def run(batch):
            done.update(zip(batch, pool.map(one, batch)))
        if cached_only:
            tags = [f[len('schedule_'):-len('.txt')] for f in sorted(os.listdir(cache))
                    if f.startswith('schedule_') and f != 'schedule_base.txt']
            run([tuple(int(k) for k in t.split('_')) for t in tags
                 if len(t.split('_')) == len(names)])
        elif space <= max_points:
            run(list(itertools.product(*choices)))
        else:
            sizes = [classes[c] for c in names]
            run(by_depth(sizes, choices)[:max_points])
            while len(done) < max_points:
                point = {id(p): ks for ks, p in done.items()}
                front = pareto([p for p in done.values() if 'ii' in p], key)
                batch = list(dict.fromkeys(
                    q for p in front for q in neighbours(point[id(p)], choices)
                    if q not in done))
                if not batch:
                    break
                run(batch[:max_points - len(done)])
    points = list(done.values())
    good = [p for p in points if 'ii' in p]
    front = pareto(good, key)
    within = [p for p in good if p['fits'] and (not budget or p['luts'] <= budget)
              and p.get('routes', True)]
    pick = min(within, key=lambda p: (p[key], p['luts'])) if within else None
    # Every instance its own copy and its state in flip-flops: the target's
    # own area, which an unfolded simulator (FireSim) needs at the least.
    unfolded = area(cal, est, classes, dict(classes), rest)
    return dict(classes=classes, points=points, front=front, pick=pick, budget=budget,
                unfolded=unfolded, firesim=firesim_area(cal, unfolded), depth=depth, fit=fit,
                space=space, store=store, route=route, uram=uram, slr_groups=slr_groups)


def table(result):
    lines = []
    fit = result['fit']
    store, route = result.get('store', 'fixed'), result.get('route', False)
    cols = ['copies', 'ii', 'luts', 'lutram', 'sites', 'fits', 'bram'] + \
        (['uram'] if store == 'auto' else []) + ['largest_copy'] + \
        (['congestion', 'routes'] if route else []) + (['mhz', 'ns'] if fit else [])
    lines.append(' '.join(f'{c:>12}' if c != 'copies' else f'{c:<60}' for c in cols))
    front = {id(p) for p in result['front']}
    for p in sorted(result['points'], key=lambda p: p.get('luts', 0)):
        if 'error' in p:
            lines.append(f"{p['copies']:<60} error: {' '.join(p['error'])}")
            continue
        mark = (' *' if id(p) in front else '') + (' <- pick' if p is result['pick'] else '')
        if p.get('ii_floor'):
            mark += f" (II >= {p['ii_floor']} with paging)"
        lines.append(' '.join(f'{str(p[c]):>12}' if c != 'copies' else f'{p[c]:<60}'
                              for c in cols) + mark)
    u = result['unfolded']
    lines.append(f"{'unfolded (every instance its own copy)':<60} {'-':>12} {u['luts']:>12} "
                 f"{u['lutram']:>12} {u['sites']:>12} {str(u['fits']):>12} {u['bram']:>12}")
    fs = result.get('firesim')
    if fs:
        lines.append(f"{'FireSim, projected (at least: unfolded x FireSim builds)':<60} {'-':>12} "
                     f"{fs['luts']:>12} {fs['lutram']:>12} {fs['sites']:>12} {str(fs['fits']):>12}")
    space = result.get('space', len(result['points']))
    if space > len(result['points']):
        lines.append(f"searched {len(result['points'])} of {space} choices: one per frame depth, "
                     'then the neighbours of the front')
    if store == 'auto':
        r = storage.REGION
        lines.append('* Pareto front (ns per simulated cycle against LUTs); '
                     f"budget {result['budget'] or 'none'}; storage chosen per store; fits: "
                     f"{storage.MARGIN:.0%} of the V80 region's {r['sites']} LUT sites (a LUTRAM "
                     f"LUT {storage.LUTRAM_SITES} sites), {r['bram']} RAMB36 and "
                     + (f"{r['uram']} URAM288" if result.get('uram', True) else 'no URAM288'))
        pick = result['pick']
        if pick and pick.get('storage'):
            lines.append(f"storage of the pick: {pick['storage']}")
    else:
        lines.append('* Pareto front (ns per simulated cycle against LUTs); '
                     f"budget {result['budget'] or 'none'}; fits: {PLACE_MARGIN:.0%} of "
                     f'{PLACE_SITES} V80 LUT sites (a LUTRAM LUT {LUTRAM_SITES} sites) and of '
                     f'{BRAM_SITES} RAMB36')
    if route:
        per = storage.ROUTE_PER
        slrs = storage.route_slrs()
        groups = ', fewer if a choice has fewer copies (SLR groups)' \
            if result.get('slr_groups') else ''
        lines.append(f'routes: congestion = {storage.ROUTE_A} L + {storage.ROUTE_B} L (L / S) % '
                     f"(L = {storage.SYNTH_PER_ROUTED} LUTs + {per['bram']} RAMB36 + "
                     f"{per['uram']} URAM288, in millions; S = the region's {slrs:g} SLRs"
                     f'{groups}; foldsim_storage.py fit-route) <= {storage.ROUTE_LIMIT} %: a load '
                     f'of at most {storage.route_luts():.0f} LUTs')
    if fit:
        c = fit['coef']
        label = dict(size='log2(largest copy LUTs)',
                     depth=f'log2(end points >= {DEPTH_LEVELS} levels) [{result["depth"]}]',
                     copies='copies', span='share of the V80\'s LUT sites')
        held = fit.get('range', {})
        terms = ' + '.join([f'{c[0]:.3f}'] + [
            f'{x:.3f} {label[t]}' + (f' (held to {held[t][0]:g}..{held[t][1]:g})' if t in held else '')
            for x, t in zip(c[1:], fit['terms'])])
        lines.append(f'period = {terms} ns; leave-one-out RMS {fit["rms"]} ns {fit["loo"] or ""}')
    else:
        lines.append('no calibration: LUTs uncalibrated, no period term (ranked by II)')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('calibrate-firesim', help='add a FireSim build of a design')
    g.add_argument('--models', required=True)
    g.add_argument('--schedule', required=True,
                   help='a schedule report of the design (any fold degree): its classes')
    g.add_argument('--util', default='', help="FireSim's report_utilization_firesim.txt")
    g.add_argument('--luts', type=int, default=0, help='instead of --util: total LUTs')
    g.add_argument('--logic', type=int, default=0)
    g.add_argument('--lutram', type=int, default=0)
    g.add_argument('--instance', default='top_i')
    g.add_argument('--period', type=float, required=True, help='the routed period, ns')
    g.add_argument('--name', required=True)
    g.add_argument('--out', required=True)
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
    c.add_argument('--register-state-entries', type=int, default=2,
                   help="the build's foldsim --register-state-entries (0 for builds before it)")
    e = sub.add_parser('explore', help='the front of a models IR')
    e.add_argument('--models', required=True)
    e.add_argument('--module', required=True)
    e.add_argument('--foldsim-opt', default='foldsim-opt')
    e.add_argument('--workdir', required=True)
    e.add_argument('--calibration', default='')
    e.add_argument('--lut-budget', type=int, default=0)
    e.add_argument('--jobs', type=int, default=4)
    e.add_argument('--max-points', type=int, default=256,
                   help='schedule every choice up to this many, else search (evaluate)')
    e.add_argument('--screen', default='',
                   help='the logic screen of the design\'s netlist, for the depth term')
    e.add_argument('--storage', choices=('fixed', 'auto'), default='fixed',
                   help="fixed: the compiler's default storage; auto: a kind per store "
                        '(foldsim_storage.py), as foldsim --storage auto emits it')
    e.add_argument('--routability', action=argparse.BooleanOptionalAction, default=True,
                   help='a pick must also route (foldsim_storage.congestion)')
    e.add_argument('--uram', action=argparse.BooleanOptionalAction, default=True,
                   help='with --storage auto: whether stores may take UltraRAM')
    e.add_argument('--slr-groups', action='store_true',
                   help='each copy kept whole in one SLR, as foldsim --slr-groups '
                        '(the routability term)')
    e.add_argument('--schedules', default='',
                   help='a directory of schedule reports of the same models and options '
                        '(an earlier explore\'s workdir), read instead of scheduling')
    e.add_argument('--cached-only', action='store_true',
                   help='with --schedules: price its choices only, schedule none')
    e.add_argument('--fold-options', default='',
                   help='more foldsim-fold-static options, as the driver gives them '
                        '(nest, modulo, ...)')
    f = sub.add_parser('fit', help='the period fit of a calibration and its leave-one-out error')
    f.add_argument('--calibration', required=True)
    a = ap.parse_args()
    if a.cmd == 'fit':
        cal = json.load(open(a.calibration))
        builds = [b for b in cal['builds'] if b.get('period_fit', True)]
        for name, terms in TERMS.items():
            if 'depth' in terms and not all('depth' in b for b in builds):
                continue
            print(name, fit_terms(builds, terms))
        print('chosen', period_fit(cal)['terms'] or 'const')
        return 0
    if a.cmd == 'calibrate-firesim':
        return calibrate_firesim(a)
    if a.cmd == 'calibrate':
        return calibrate(a)
    cal = json.load(open(a.calibration)) if a.calibration else None
    fold = f'module={a.module} sim=foldsim_sim'
    if a.fold_options:
        fold += ' ' + a.fold_options
    base = os.path.join(a.workdir, 'schedule_base.txt')
    os.makedirs(a.workdir, exist_ok=True)
    if a.schedules and os.path.exists(os.path.join(a.schedules, 'schedule_base.txt')):
        base = os.path.join(a.schedules, 'schedule_base.txt')
    else:
        r = subprocess.run([a.foldsim_opt, a.models,
                            f'--pass-pipeline=builtin.module(foldsim-fold-static{{{fold} '
                            f'report={base} schedule-only}})', '-o', os.devnull],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr)
    result = evaluate(a.models, open(base).read(), fold, a.foldsim_opt, a.workdir, cal,
                      a.jobs, a.lut_budget, screen_depth(a.screen) if a.screen else None,
                      a.max_points, a.storage, a.routability, a.schedules or None,
                      a.cached_only, a.uram, a.slr_groups)
    print(table(result))
    json.dump(result, open(os.path.join(a.workdir, 'degree.json'), 'w'), indent=1,
              default=str)


if __name__ == '__main__':
    main()
