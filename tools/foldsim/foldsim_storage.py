#!/usr/bin/env python3
"""Storage of folded state on the V80, and the routability of a fold.

Storage. A fold makes, per copy of a class, a store of each of its
registers (one entry per instance, read combinationally into a read
register) and of each of its memories (the target's depth times the
instances). Each store can live in the kinds of the V80:

  kind (ram_style)  capacity per primitive   ports              read
  registers         one flop a bit           any                any
  distributed       a LUT: 64 x 1 (SLICEM)   1 write, a LUT per  any
                                             read port
  block             RAMB36: 512 x 72 ...     2 (a write and a    registered
                    32K x 1, or half (18K)   read, SDP)
  ultra             URAM288: 4K x 72         2 (as block)       registered

A store takes a kind only if the kind meets its ports and read latency: a
register store reads combinationally, so it takes block or ultra RAM only
read a host cycle early (foldsim-fold-static `storage`), which needs a state
gap of 4; a memory read combinationally takes neither. Read ports beyond
the kind's are copies of the RAM (each a write and a read), as Vivado builds
them; write ports beyond the kind's would need slots of several host cycles
(paging, not built), whose II floor is reported (`ii_floor`).

`assign` chooses a kind per store against the V80 dynamic region: first the
cheapest kind of each store, pricing every resource by its share of the
region (a URAM is 1/1211 of the URAMs, a LUT site 1/3,269,043 of the
sites), then, while a resource is over its capacity, the moves that relieve
it at the least price, never pushing another over its own. A point that
still exceeds a capacity does not place. Once placed, the same repair
brings the routing load (below) under its limit if it can, never at
placement's cost; a point it cannot bring under does not route.

Routability. The router's global routing utilization (Vivado's initial
estimate, `Global Vertical/Horizontal Routing Utilization`) against the
design's LUTs, fitted over the V80 OOC builds and images (`fit-route`,
foldsim_v80_route.csv: the OOC builds of fame5_work/static/ooc2x2 by
`route-points`, and the images img_int, img_fo and img_ii8 from their
impl_1 runme.log and place_report_utilization_0.rpt):

  congestion = A * L + B * L**2   (percent, L in millions)
  L = LUTs + 24 RAMB36 + 26 URAM288

A block RAM's routing in LUTs is its signals over a LUT6's 7 pins: a RAMB36
at 72 bits has about 165 (72 in, 72 out, two addresses, enables), a URAM288
about 180. The builds cannot fit these weights (their block RAMs hardly
vary), so they are set, not fitted; without them, state moved out of LUTs
into block RAM would look free of routing.

The fit of the builds that routed reaches 23.8 % (f_k2, 1.18M LUTs); that
of those that failed starts at 33.0 % (o8_t8sf, 1.43M); a fold routes when
its estimate is below the middle, 28.4 % (a load of 1.33M). The builds count
synthesized LUTs (placed for the images); the fold-degree model estimates
routed ones (its calibration's), which are 3.8 % fewer (the median over the
24 OOC builds that routed, 0.7-9.0 %), so a fold's LUTs are scaled by 1.038
first: it routes up to a load of 1.28M in the model's LUTs.

usage:
  foldsim_storage.py fit-route [--points foldsim_v80_route.csv]
  foldsim_storage.py route-points <build dir>...   (rows for the points file)
"""
import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import foldsim_cost as cost  # noqa: E402

# The SLASH dynamic region of the V80 (pblock_slash): LUT sites as the placer
# counts them (foldsim_degree.PLACE_SITES), RAMB36 and URAM288.
REGION = dict(sites=3269043, bram=2415, uram=1211)
# What a LUT of distributed RAM costs in sites (foldsim_degree.LUTRAM_SITES).
LUTRAM_SITES = 4.08
# Placement is not routing: the margin kept on every capacity.
MARGIN = 0.9

KINDS = ('registers', 'distributed', 'block', 'ultra')
# RAMB36E5 in simple dual port (a write and a read port): depth x width, and
# its half, RAMB18E5 (0.5 of a RAMB36).
BRAM36 = ((512, 72), (1024, 36), (2048, 18), (4096, 9), (8192, 4), (16384, 2), (32768, 1))
BRAM18 = ((512, 36), (1024, 18), (2048, 9), (4096, 4), (8192, 2), (16384, 1))
URAM = (4096, 72)
# Write ports a primitive has (a store with more needs paging).
WRITE_PORTS = dict(registers=None, distributed=1, block=2, ultra=2)
# Host cycles a register store's block read needs between a state pass's
# write and the next read (foldsim-fold-static `storage`).
BLOCK_STATE_GAP = 4

# The routability fit (fit-route over foldsim_v80_route.csv: 26 OOC builds
# and 3 images; RMS 2.79 points, leave-one-out 3.04) and its limit: midway
# between the fit of the most congested build that routed (23.77 %, f_k2)
# and of the least congested that failed (33.01 %, o8_t8sf), a load of 1.33M;
# 1.5 leave-one-out errors either side.
ROUTE_A, ROUTE_B = 5.893, 11.673
ROUTE_LIMIT = 28.39
# LUT6s of routing a block RAM brings: its signals over a LUT6's 7 pins.
ROUTE_PER = dict(bram=24, uram=26)
# Synthesized LUTs per routed LUT: the median over the OOC builds that routed
# (synth_util.rpt against routed_util.rpt; 1.007-1.090).
SYNTH_PER_ROUTED = 1.038


def route_load(luts, bram=0.0, uram=0.0):
    """A design's routing load L: its synthesized LUTs and its block RAMs'
    LUT6 equivalents."""
    return luts + ROUTE_PER['bram'] * bram + ROUTE_PER['uram'] * uram


def congestion(load):
    """The router's global routing utilization (percent) at a routing load."""
    m = load / 1e6
    return ROUTE_A * m + ROUTE_B * m * m


def fold_load(luts, bram=0.0, uram=0.0):
    """The routing load of a fold priced by the fold-degree model, whose
    LUTs are routed ones; in the model's LUTs."""
    return luts + route_load(0, bram, uram) / SYNTH_PER_ROUTED


def fold_congestion(luts, bram=0.0, uram=0.0):
    """The congestion of a fold priced by the fold-degree model."""
    return congestion(SYNTH_PER_ROUTED * fold_load(luts, bram, uram))


def route_luts():
    """The most routing load, in the model's LUTs, that routes:
    fold_congestion(route_luts()) == ROUTE_LIMIT."""
    a, b, c = ROUTE_B, ROUTE_A, -ROUTE_LIMIT
    return 1e6 * (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a) / SYNTH_PER_ROUTED


def bram36(depth, width):
    """RAMB36 (18K halves at 0.5) holding depth x width in one port pair."""
    best = min(math.ceil(width / w) * math.ceil(depth / d) for d, w in BRAM36)
    half = min(math.ceil(width / w) * math.ceil(depth / d) for d, w in BRAM18)
    return min(best, 0.5 * half)


def uram(depth, width):
    return math.ceil(width / URAM[1]) * math.ceil(depth / URAM[0])


def lanes(width, mask):
    """A masked store's write lanes: one RAM column per lane unless the lanes
    are bytes (the primitives' byte write enables)."""
    if mask <= 1:
        return 1, width
    grain = width // mask
    return (1, width) if grain in (8, 9) else (mask, grain)


def kind_cost(store, kind):
    """What a store costs in a kind: logic LUTs, distributed-RAM LUTs, flops,
    RAMB36, URAM; and the II floor its write ports imply (0: none). None if
    the kind cannot hold it. `store`: depth, width, rports, wports, rlat (0 or
    1), mask, reg (a register store), gap (the copy's state gap)."""
    d, w = store['depth'], store['width']
    r, wr = max(store['rports'], 1), store['wports']
    c = dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0, ii_floor=0)
    ports = WRITE_PORTS[kind]
    if ports is not None and wr > ports:
        # Several host cycles a slot (paging): not built.
        c['ii_floor'] = store.get('threads', 1) * math.ceil(wr / ports)
    if kind == 'registers':
        c['ff'] = d * w
        c['lut'] = r * w * math.ceil((d - 1) / 3) if d > 1 else 0.0
        return c
    if kind == 'distributed':
        c['lutram'] = cost.mem_cost(d, w, r, store['rlat'], 'distributed')[0]
        return c
    if store['reg']:
        if store['gap'] is None or store['gap'] < BLOCK_STATE_GAP:
            return None
    elif store['rlat'] < 1:
        return None
    n, grain = lanes(w, store['mask'])
    # A read port beyond the first is a copy of the RAM; a ROM's two ports read.
    copies = r if wr else math.ceil(r / 2)
    if kind == 'block':
        c['bram'] = copies * n * bram36(d, grain)
    else:
        c['uram'] = copies * n * uram(d, grain)
    return c


def sites(c):
    return c['lut'] + LUTRAM_SITES * c['lutram']


def usage(c, pool):
    if pool == 'sites':
        return sites(c)
    if pool == 'luts':
        return fold_load(c['lut'] + c['lutram'], c['bram'], c['uram'])
    return c[pool]


def assign(units, base, caps, extra=None, paging=False):
    """A kind per store. `units`: dicts with 'costs' {kind: cost (kind_cost,
    summed over the unit's stores)}, 'count' stores of it, and optionally
    'fixed' (a kind it keeps); `base`: the usage of everything else per pool;
    `caps`: per pool, the most it may use (placement); `extra`: more pools
    to meet once placed (the LUTs that route), never at placement's cost.
    Returns (per unit, its stores per kind; the usage per pool).

    Each store first takes its cheapest kind, every pool priced by its share
    of the capacity; then, while a pool is over, the move that frees it at
    the least price elsewhere, never onto a pool without room, so the
    overflow only falls and the repair ends."""
    pools = list(caps) + list(extra or {})
    limit = dict(caps, **(extra or {}))

    def usage_of(i, k, p):
        return usage(units[i]['costs'][k], p)

    def admissible(u, k):
        c = u['costs'].get(k)
        return c is not None and (paging or not c['ii_floor'])

    def price(c):
        return sum(usage(c, p) / caps[p] for p in caps if caps[p])

    choice = []
    for u in units:
        kinds = [u['fixed']] if u.get('fixed') in u['costs'] else \
            [k for k in KINDS if admissible(u, k)]
        k = min(kinds, key=lambda k: (price(u['costs'][k]), KINDS.index(k)))
        choice.append({k: u['count']})
    used = {p: base.get(p, 0.0) + sum(n * usage_of(i, k, p)
                                      for i, ch in enumerate(choice) for k, n in ch.items())
            for p in pools}

    def relieve(p, active):
        """The move that frees pool p at the least price elsewhere per unit
        of p freed: (key, unit, from, to, stores), or None."""
        best = None
        for i, u in enumerate(units):
            if u.get('fixed') in u['costs']:
                continue
            for a, n in choice[i].items():
                for b in KINDS:
                    if b == a or not admissible(u, b):
                        continue
                    relief = usage_of(i, a, p) - usage_of(i, b, p)
                    if relief <= 0:
                        continue
                    m = min(n, math.ceil((used[p] - limit[p]) / relief))
                    for q in active:
                        dq = usage_of(i, b, q) - usage_of(i, a, q)
                        if q != p and dq > 0:
                            room = limit[q] - used[q]
                            m = min(m, int(room // dq) if room > 0 else 0)
                    if m <= 0:
                        continue
                    more = sum((usage_of(i, b, q) - usage_of(i, a, q)) / limit[q]
                               for q in active if q != p and limit[q])
                    key = (more / (relief / limit[p]), i, KINDS.index(b))
                    if best is None or key < best[0]:
                        best = (key, i, a, b, m)
        return best

    def repair(active):
        while True:
            best = None
            for p in active:
                if used[p] > limit[p] + 1e-9:
                    best = relieve(p, active)
                    if best:
                        break
            if best is None:
                return all(used[p] <= limit[p] + 1e-9 for p in active)
            _, i, a, b, m = best
            choice[i][a] -= m
            if not choice[i][a]:
                del choice[i][a]
            choice[i][b] = choice[i].get(b, 0) + m
            for q in pools:
                used[q] += m * (usage_of(i, b, q) - usage_of(i, a, q))

    if repair(list(caps)) and extra:
        repair(pools)
    return choice, used


def read_points(path):
    import csv
    return [r for r in csv.DictReader(open(path))]


def fit_route(points):
    """Least squares of the router's estimate (mean of vertical and
    horizontal) on L and L^2, through the origin; each build's residual and
    leave-one-out error; the routed and failed builds' ranges."""
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from foldsim_degree import lstsq
    load = [route_load(float(p['luts']), float(p['bram']), float(p['uram'])) / 1e6 for p in points]
    rows = [[x, x * x] for x in load]
    ys = [(float(p['gvu']) + float(p['ghu'])) / 2 for p in points]
    coef = lstsq(rows, ys)
    out = []
    for i, p in enumerate(points):
        c = lstsq(rows[:i] + rows[i + 1:], ys[:i] + ys[i + 1:])
        est = sum(a * b for a, b in zip(coef, rows[i]))
        loo = sum(a * b for a, b in zip(c, rows[i]))
        out.append(dict(name=p['name'], luts=int(float(p['luts'])), load=int(load[i] * 1e6),
                        routed=p['routed'], y=ys[i],
                        est=est, res=est - ys[i], loo=loo - ys[i]))
    rms = math.sqrt(sum(o['res'] ** 2 for o in out) / len(out))
    loo = math.sqrt(sum(o['loo'] ** 2 for o in out) / len(out))
    # The limit separates the builds by the fit itself (what a choice is
    # priced with): midway between the most congested routed build's fit
    # and the least congested failed one's.
    routed = max(o['est'] for o in out if o['routed'] == 'yes')
    failed = min(o['est'] for o in out if o['routed'] == 'FAIL')
    return dict(coef=coef, rms=rms, loo=loo, routed=routed, failed=failed,
                limit=(routed + failed) / 2, points=out)


def route_point(d):
    """A points-file row of an out-of-context build directory (vivado.log,
    synth_util.rpt): synthesized LUTs, RAMB36 and URAM, whether routing
    finished legally, and the router's global routing utilization; None if
    the build has not reached the router."""
    log = open(os.path.join(d, 'vivado.log'), errors='replace').read()
    util = open(os.path.join(d, 'synth_util.rpt'), errors='replace').read()

    def row(label):
        m = re.search(r'\|\s*' + re.escape(label) + r'\s*\|\s*([\d.]+)', util)
        return m and float(m.group(1))

    def last(p):
        m = re.findall(p, log)
        return m[-1] if m else None
    gv = last(r'Global Vertical Routing Utilization\s*=\s*([\d.]+)')
    gh = last(r'Global Horizontal Routing Utilization\s*=\s*([\d.]+)')
    if 'not legally routed' in log:
        routed = 'FAIL'
    elif 'Number of Unrouted Nets             = 0' in log and \
            os.path.exists(os.path.join(d, 'routed_timing_summary.rpt')):
        routed = 'yes'
    else:
        return None
    luts = row('CLB LUTs*') or row('CLB LUTs')
    overlaps = last(r'There are (\d+) node overlaps') or ''
    return (f"{os.path.basename(d.rstrip('/'))},ooc,{luts:.0f},{row('LUT as Logic'):.0f},"
            f"{row('LUT as Memory'):.0f},{row('Block RAM Tile'):g},{row('URAM'):.0f},{routed},"
            f"{gv},{gh},{overlaps}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit-route', help='the routability fit and its residuals')
    f.add_argument('--points', default=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                    'foldsim_v80_route.csv'))
    g = sub.add_parser('route-points', help='points-file rows of OOC build directories')
    g.add_argument('dirs', nargs='+')
    a = ap.parse_args()
    if a.cmd == 'route-points':
        for d in a.dirs:
            r = route_point(d) if all(os.path.exists(os.path.join(d, f)) for f in
                                      ('vivado.log', 'synth_util.rpt')) else None
            if r:
                print(r)
        return
    r = fit_route(read_points(a.points))
    for o in sorted(r['points'], key=lambda o: o['luts']):
        print(f"{o['name']:<12} {o['luts']:>9} LUTs, load {o['load']:>9} {o['routed']:>5} measured {o['y']:5.1f} "
              f"fit {o['est']:5.1f} ({o['res']:+5.1f}) leave-one-out {o['loo']:+5.1f}")
    a_, b_ = r['coef']
    print(f"congestion = {a_:.3f} L + {b_:.3f} L^2 (L in millions); RMS {r['rms']:.2f}, "
          f"leave-one-out RMS {r['loo']:.2f} over {len(r['points'])} builds")
    print(f"fit of the routed builds up to {r['routed']:.2f} %, of the failed from "
          f"{r['failed']:.2f} %: limit "
          f"{r['limit']:.2f} % = a load of "
          f"{1e6 * (-a_ + math.sqrt(a_ * a_ + 4 * b_ * r['limit'])) / (2 * b_):.0f}")


if __name__ == '__main__':
    main()
