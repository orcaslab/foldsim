#!/usr/bin/env python3
"""Beacon's routing terms: a timing estimate per end point from its logic levels, its path's largest fanout and
whether it crosses an SLR, calibrated on a placed V80 build.

Beacon (foldsim-estimate-timing) counts logic levels the way Vivado does, and its rule of thumb (at most 18 levels at
200 MHz) assumes short nets. On a folded design two things a level count cannot see set the period instead
(b4e at 200 MHz, 11-07: 90 % of each failing path was route, at 6-7 levels):

- fanout: a fold's slot controls (instance index, state pass) address and enable every LUT of a region's stores, nets
  of thousands of pins (the `fanout` column, which counts a memory port's control bit as every pin of the mapped
  memory, foldsim/MemoryLoads.h);
- SLR crossings: a path between SLR groups, or inside a group that does not fit its SLR and so spills over a
  boundary, pays a super long line, the clock skew between SLRs and the inter-SLR compensation.

  foldsim_beacon_terms.py fit <sample.tsv> -o <calibration.json>
      least squares of a placed build's path delays (requirement minus slack) on levels, log2 fanout and SLR changes:
      one row per path, Vivado's columns (slack, levels, max_fanout, slr_changes; see f12_1107/cal/sample.tcl).
  foldsim_beacon_terms.py score <screen dir> --calibration <json> [--schedule schedule.txt --slr-region <json|v80>]
                                [--period-ns 5.0] [--top N]
      each end point of a Beacon screen (endpoints.csv, instances.csv) estimated in ns and in equivalent levels
      ((ns - intercept) / ns per level); an end point crosses if its start and end are in different SLR groups
      (foldsim_slr<g>) or its group's LUT sites (schedule.txt) exceed its SLR's capacity (--slr-region: group g is
      the g-th SLR when the floorplan names SLRs; otherwise, groups being placed freely, the largest SLR's).
"""
import argparse, csv, json, math, os, re, sys, collections


def features(levels, fanout, crosses):
    return [1.0, float(levels), math.log2(max(1, fanout)), float(crosses)]


def fit(a):
    rows = list(csv.DictReader(open(a.sample), delimiter='\t'))
    X, y = [], []
    for r in rows:
        if r['max_fanout'] in ('', '-'):
            continue
        X.append(features(int(r['levels']), int(r['max_fanout']), int(r['slr_changes']) > 0))
        y.append(a.period_ns - float(r['slack']))
    # Normal equations, 4 x 4.
    n = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) for j in range(n)] for i in range(n)]
    b = [sum(x[i] * t for x, t in zip(X, y)) for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for r in range(n):
            if r != c and A[c][c]:
                f = A[r][c] / A[c][c]
                A[r] = [u - f * v for u, v in zip(A[r], A[c])]
                b[r] -= f * b[c]
    coef = [b[i] / A[i][i] for i in range(n)]
    pred = [sum(c * x for c, x in zip(coef, xs)) for xs in X]
    res = [t - p for t, p in zip(y, pred)]
    mean = sum(y) / len(y)
    r2 = 1 - sum(e * e for e in res) / sum((t - mean) ** 2 for t in y)
    out = {'about': f'fit of {len(X)} paths of {a.sample} (requirement {a.period_ns} ns minus slack)',
           'intercept_ns': coef[0], 'ns_per_level': coef[1], 'ns_per_log2_fanout': coef[2], 'ns_slr_crossing': coef[3],
           'r2': r2, 'rmse_ns': math.sqrt(sum(e * e for e in res) / len(res)), 'paths': len(X)}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))


def slr_capacities(path):
    if path == 'v80':
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')
    region = json.load(open(path))
    return [(s['name'], region['margin'] * s['luts'] * region['sites_per_lut']) for s in region['slrs']]


def score(a):
    cal = json.load(open(a.calibration))
    c0, cl, cf, cx = (cal[k] for k in ('intercept_ns', 'ns_per_level', 'ns_per_log2_fanout', 'ns_slr_crossing'))
    group_of_inst = {}
    for r in csv.DictReader(open(os.path.join(a.screen, 'instances.csv'))):
        m = re.match(r'foldsim_slr(\d+)(/|$)', r['instance'])
        group_of_inst[r['id']] = int(m.group(1)) if m else None
    straddles = set()
    if a.schedule and a.slr_region:
        caps = slr_capacities(a.slr_region)
        named = bool(a.named)
        for m in re.finditer(r'^    slr(\d+) \((\d+) sites\)', open(a.schedule).read(), re.M):
            g, sites = int(m.group(1)), int(m.group(2))
            cap = caps[g][1] if named else max(c for _, c in caps)
            if sites > cap:
                straddles.add(g)
        print(f'groups over their SLR\'s capacity ({"named SLRs" if named else "largest SLR"}): '
              f'{sorted(straddles) or "none"}')
    worst = []
    over = 0
    by_unit = collections.Counter()
    n = 0
    for r in csv.DictReader(open(os.path.join(a.screen, 'endpoints.csv'))):
        n += 1
        lv, fo = int(r['levels']), int(r['fanout'] or 0)
        gs, ge = group_of_inst.get(r['start_inst']), group_of_inst.get(r['end_inst'])
        crosses = (gs is not None and ge is not None and gs != ge) or (ge in straddles)
        ns = c0 + cl * lv + cf * math.log2(max(1, fo)) + cx * crosses
        if ns > a.period_ns:
            over += 1
            by_unit[re.sub(r'\[\d+\]', '', r['end']).rsplit('.', 1)[0][:100]] += 1
        worst.append((ns, lv, fo, crosses, r['start'], r['end']))
        if len(worst) > 4 * a.top:
            worst = sorted(worst, reverse=True)[:a.top]
    worst = sorted(worst, reverse=True)[:a.top]
    eq = lambda ns: (ns - c0) / cl
    print(f'{n} end points; calibration {a.calibration}: {c0:.3f} + {cl:.3f}/level + {cf:.3f}/log2 fanout + '
          f'{cx:.3f} crossing ns (r2 {cal.get("r2", 0):.2f})')
    print(f'worst estimate {worst[0][0]:.2f} ns ({eq(worst[0][0]):.1f} equivalent levels) against {a.period_ns} ns '
          f'({eq(a.period_ns):.1f} levels): {"MISS" if worst[0][0] > a.period_ns else "fits"}; '
          f'{over} end points over the period')
    for ns, lv, fo, x, s, e in worst:
        print(f'  {ns:6.2f} ns  {lv:2d} levels  fanout {fo:6d}  {"cross" if x else "     "}  {s[:70]} -> {e[:90]}')
    if by_unit:
        print('end points over the period by end register:')
        for u, k in by_unit.most_common(12):
            print(f'  {k:7d} {u}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit')
    f.add_argument('sample')
    f.add_argument('-o', '--out', required=True)
    f.add_argument('--period-ns', type=float, default=5.0)
    s = sub.add_parser('score')
    s.add_argument('screen')
    s.add_argument('--calibration', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_timing.json'))
    s.add_argument('--schedule')
    s.add_argument('--slr-region')
    s.add_argument('--named', action='store_true', help='the floorplan assigns group g to the g-th SLR')
    s.add_argument('--period-ns', type=float, default=5.0)
    s.add_argument('--top', type=int, default=15)
    a = ap.parse_args()
    fit(a) if a.cmd == 'fit' else score(a)


if __name__ == '__main__':
    main()
