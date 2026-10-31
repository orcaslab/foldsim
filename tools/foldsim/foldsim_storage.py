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
them; write ports beyond the kind's would need slots of several host cycles,
whose II floor is reported (`ii_floor`).

Paged register state (foldsim-fold-static `page`, `page_cost`). A copy of
`threads` instances whose registers page keeps them in one page RAM of
pow2(threads) x pow2(p) rows of ceil(bits / p) bits (block or ultra RAM), a
fetch buffer and a write-back buffer of `bits` flops each and the RAM's
output register; the write-back's page select is about ceil(p / 3) LUTs a
page bit. Its state passes are p host cycles apart, so the frame is at least
p x threads (and p + 4). A RAM of P 72-bit ports moves 72 P bits a host
cycle, so a copy's II is at least threads x ceil(bits / (72 P)) (`page_bound`):
the bits/ports bound, which p x threads meets up to rounding.

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
design's routing load and how densely the layout packs it, fitted over the
V80 OOC builds and images (`fit-route`, foldsim_v80_route.csv: the OOC
builds of fame5_work/static/ooc2x2 by `route-points`, and the images
img_int, img_fo and img_ii8 from their impl_1 runme.log and
place_report_utilization_0.rpt):

  congestion = A * L + B * L * (L / S)   (percent, L in millions)
  L = LUTs + 24 RAMB36 + 26 URAM288
  S = the SLRs the fold may spread over: the region's (3 out of context,
      1.8 for the image's dynamic region), or its SLR groups if fewer

A block RAM's routing in LUTs is its signals over a LUT6's 7 pins: a RAMB36
at 72 bits has about 165 (72 in, 72 out, two addresses, enables), a URAM288
about 180. The builds cannot fit these weights (their block RAMs hardly
vary), so they are set, not fitted; without them, state moved out of LUTs
into block RAM would look free of routing.

The layout term. At 1.42-1.48M of load the full GPU's II 8 folds split by
layout: out of context, o8_t8n and o8_t8a spread over all three SLRs and
route (30.5-30.7 %); o8_t8sc and o8_t8sf, whose `--slr-groups` kept each
socket's subtree whole in one SLR (2 groups), packed into two SLRs and
failed (35.8-37.6 %), as did img_ii8 in the image, whose dynamic region
spans 1.8 SLRs (36.3 %). Their links do not tell them apart: t8n, t8sc and
t8sf have the same 16 copy pairs and 75.2k link bits, t8a and img_ii8 the
same 28 pairs (interleaved copies) and 75.3k bits. What differs is the
SLRs the placement may use, so the second term is the load times the load
per SLR: wires lengthen as a region fills. The region's SLRs are the slices
of its pblock (the rectangles of the compute shell's pblock_slash: about
195k of the device's 321,776 slices; any value from 1.5 to 2.2 classifies
every build alike); a fold with SLR groups (foldsim --slr-groups)
keeps each whole in one SLR, so G groups occupy at most G SLRs. Fewer
groups than SLRs confine; one per copy, as many as the SLRs or more, do not
(t8a's 7 spread over three).

The fit of the builds that routed reaches 28.8 % (o8_t8n); that of those
that failed starts at 34.5 % (o8_t8sf); a fold routes when its estimate is
below the middle, 31.65 %: a load of 1.32M in the image's region, 1.58M
over three SLRs. Leaving any build out, the fit of the rest still classifies
it. The builds count synthesized LUTs (placed for the images); the
fold-degree model estimates routed ones (its calibration's), which are
3.8 % fewer (the median over the 24 OOC builds that routed, 0.7-9.0 %), so a
fold's LUTs are scaled by 1.038 first: in the region it routes up to a load
of 1.27M in the model's LUTs.

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
# counts them (foldsim_degree.PLACE_SITES), RAMB36, URAM288, and the SLRs its
# slices amount to (about 195k of the device's 321,776 slices over 3 SLRs).
REGION = dict(sites=3269043, bram=2415, uram=1211, slrs=1.8)
# The V80's SLRs: what an out-of-context build may spread over.
DEVICE_SLRS = 3
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

# The routability fit (fit-route over foldsim_v80_route.csv: 29 OOC builds
# and 3 images; RMS 2.27 points, leave-one-out 2.45) and its limit: midway
# between the fit of the most congested build that routed (28.80 %, o8_t8n)
# and of the least congested that failed (34.51 %, o8_t8sf); 1.2
# leave-one-out errors either side.
ROUTE_A, ROUTE_B = 9.924, 19.211
ROUTE_LIMIT = 31.65
# LUT6s of routing a block RAM brings: its signals over a LUT6's 7 pins.
ROUTE_PER = dict(bram=24, uram=26)
# Synthesized LUTs per routed LUT: the median over the OOC builds that routed
# (synth_util.rpt against routed_util.rpt; 1.007-1.090).
SYNTH_PER_ROUTED = 1.038


def route_load(luts, bram=0.0, uram=0.0):
    """A design's routing load L: its synthesized LUTs and its block RAMs'
    LUT6 equivalents."""
    return luts + ROUTE_PER['bram'] * bram + ROUTE_PER['uram'] * uram


def route_slrs(groups=0, region=None):
    """The SLRs a fold may spread over: the region's (REGION's by default),
    or fewer when it has fewer SLR groups, each kept whole in one SLR
    (`groups`: their number, 0 for none)."""
    s = REGION['slrs'] if region is None else region
    return min(s, groups) if groups else s


def congestion(load, slrs=None):
    """The router's global routing utilization (percent) at a routing load
    spread over `slrs` SLRs (route_slrs; the region's by default)."""
    m = load / 1e6
    s = route_slrs() if slrs is None else slrs
    return ROUTE_A * m + ROUTE_B * m * m / s


def fold_load(luts, bram=0.0, uram=0.0):
    """The routing load of a fold priced by the fold-degree model, whose
    LUTs are routed ones; in the model's LUTs."""
    return luts + route_load(0, bram, uram) / SYNTH_PER_ROUTED


def fold_congestion(luts, bram=0.0, uram=0.0, slrs=None):
    """The congestion of a fold priced by the fold-degree model."""
    return congestion(SYNTH_PER_ROUTED * fold_load(luts, bram, uram), slrs)


def route_luts(slrs=None):
    """The most routing load, in the model's LUTs, that routes over `slrs`
    SLRs (the region's by default): fold_congestion(route_luts(s), slrs=s)
    == ROUTE_LIMIT."""
    s = route_slrs() if slrs is None else slrs
    a, b, c = ROUTE_B / s, ROUTE_A, -ROUTE_LIMIT
    return 1e6 * (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a) / SYNTH_PER_ROUTED


def bram36(depth, width):
    """RAMB36 (18K halves at 0.5) holding depth x width in one port pair."""
    best = min(math.ceil(width / w) * math.ceil(depth / d) for d, w in BRAM36)
    half = min(math.ceil(width / w) * math.ceil(depth / d) for d, w in BRAM18)
    return min(best, 0.5 * half)


# The host caches of the virtual memories (foldsim --vmem-cache-rows auto): a
# sixteenth of a memory's rows, at least 256 lines, and all of them within
# this many RAMB36 (5 % of the V80 dynamic region's 2415).
VMEM_CACHE_FRACTION = 16
VMEM_CACHE_FLOOR = 256
VMEM_CACHE_BUDGET = 120


def pow2_floor(n):
    return 1 << (n.bit_length() - 1) if n > 0 else 0


def vmem_cache_rows(vmems, choice, budget):
    """Each virtual memory's host-cache lines (VMEM_INTERFACE.md, v4): a power
    of two (or 0) up to its rows, the largest power of two at most
    rows x instances. `auto`: rows / 16, at least 256; while the caches' data
    take more than `budget` RAMB36, the costliest halves (0 below 2 lines).
    `vmems`: dicts with instances, row_width (address bits) and data_width.
    A cache's lines live in block RAM in one port pair (bram36)."""
    caps = [pow2_floor(v['instances'] << v['row_width']) for v in vmems]
    if choice != 'auto':
        n = int(choice)
        if n < 0 or n == 1 or n & (n - 1):
            sys.exit('--vmem-cache-rows: auto, 0 or a power of two of at least 2')
        return [min(n, c) for c in caps]
    rows = [min(c, max(VMEM_CACHE_FLOOR, pow2_floor(c // VMEM_CACHE_FRACTION))) for c in caps]
    cost = lambda k: bram36(rows[k], vmems[k]['data_width']) if rows[k] else 0
    while sum(cost(k) for k in range(len(rows))) > budget:
        k = max(range(len(rows)), key=cost)
        rows[k] = rows[k] // 2 if rows[k] > 2 else 0
    return rows


def vmem_cache_bram(vmems, choice, budget):
    """RAMB36 of the virtual memories' host caches (0 without a choice: the
    kernel's default, no cache)."""
    if choice is None or not vmems:
        return 0.0
    return sum(bram36(n, v['data_width']) if n else 0.0
               for v, n in zip(vmems, vmem_cache_rows(vmems, choice, budget)))


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


def page_geometry(bits, threads, p):
    """A paged copy's page RAM: rows, page width and 72-bit read ports."""
    rows = (1 << (max(threads, 1) - 1).bit_length()) * (1 << (p - 1).bit_length())
    pw = max(1, math.ceil(bits / p))
    return rows, pw, math.ceil(pw / 72)


def page_cost(bits, threads, p, kind):
    """What a copy's paged register state costs in a kind (block or ultra):
    as kind_cost, with `ii_floor` the frame its state passes need, and
    `ports` the page RAM's 72-bit read ports. None for another kind."""
    if kind not in ('block', 'ultra'):
        return None
    rows, pw, ports = page_geometry(bits, threads, p)
    c = dict(lut=float(pw * math.ceil(p / 3)) if p > 1 else 0.0, lutram=0.0,
             ff=2 * bits + pw, bram=0.0, uram=0, ii_floor=max(p * threads, p + 4),
             ports=ports)
    if kind == 'block':
        c['bram'] = bram36(rows, pw)
    else:
        c['uram'] = uram(rows, pw)
    return c


def page_bound(bits, threads, ports):
    """The frame a copy's register state needs through `ports` 72-bit ports,
    read and written once an instance a frame."""
    return threads * math.ceil(bits / (72 * max(ports, 1)))


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


def point_slrs(p):
    """The SLRs a build of the points file could spread over: the device's
    out of context, the region's in an image, fewer with fewer SLR groups."""
    region = DEVICE_SLRS if p['source'] == 'ooc' else REGION['slrs']
    return route_slrs(int(p.get('groups') or 0), region)


def fit_route(points, layout=True):
    """Least squares of the router's estimate (mean of vertical and
    horizontal) on L and L * L / S (`layout`; else L^2, the fit without the
    layout term), through the origin; each build's residual and leave-one-out
    error; the routed and failed builds' ranges, the limit between them, the
    builds it misclassifies, and those the fit and limit of the others
    misclassify (leave-one-out)."""
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from foldsim_degree import lstsq
    load = [route_load(float(p['luts']), float(p['bram']), float(p['uram'])) / 1e6 for p in points]
    slrs = [point_slrs(p) if layout else 1.0 for p in points]
    rows = [[x, x * x / s] for x, s in zip(load, slrs)]
    ys = [(float(p['gvu']) + float(p['ghu'])) / 2 for p in points]
    routes = [p['routed'] == 'yes' for p in points]

    def limit(est, keep):
        # The limit separates the builds by the fit itself (what a choice is
        # priced with): midway between the most congested routed build's fit
        # and the least congested failed one's.
        routed = max(est[i] for i in keep if routes[i])
        failed = min(est[i] for i in keep if not routes[i])
        return routed, failed, (routed + failed) / 2

    def fit(keep):
        c = lstsq([rows[i] for i in keep], [ys[i] for i in keep])
        return c and [sum(a * b for a, b in zip(c, r)) for r in rows]

    every = range(len(points))
    est = fit(every)
    routed, failed, lim = limit(est, every)
    out = []
    for i, p in enumerate(points):
        keep = [j for j in every if j != i]
        others = fit(keep)
        loo = others and others[i] - ys[i]
        out.append(dict(name=p['name'], luts=int(float(p['luts'])), load=int(load[i] * 1e6),
                        slrs=slrs[i], routed=p['routed'], y=ys[i], est=est[i],
                        res=est[i] - ys[i], loo=loo,
                        wrong=(est[i] <= lim) != routes[i],
                        loo_wrong=others is None or
                        (others[i] <= limit(others, keep)[2]) != routes[i]))
    rms = math.sqrt(sum(o['res'] ** 2 for o in out) / len(out))
    loo = math.sqrt(sum(o['loo'] ** 2 for o in out if o['loo'] is not None) / len(out))
    return dict(coef=lstsq(rows, ys), rms=rms, loo=loo, routed=routed, failed=failed,
                limit=lim, points=out, layout=layout)


def route_point(d):
    """A points-file row of an out-of-context build directory (vivado.log,
    synth_util.rpt): synthesized LUTs, RAMB36 and URAM, whether routing
    finished legally, the router's global routing utilization, and the SLR
    groups the build applied (floorplan.txt: foldsim_floorplan.tcl's
    assignments); None if the build has not reached the router."""
    log = open(os.path.join(d, 'vivado.log'), errors='replace').read()
    util = open(os.path.join(d, 'synth_util.rpt'), errors='replace').read()
    plan = os.path.join(d, 'floorplan.txt')
    groups = len({line.split()[1] for line in open(plan) if len(line.split()) > 1}) \
        if os.path.exists(plan) else 0

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
            f"{gv},{gh},{overlaps},{groups}")


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
    points = read_points(a.points)
    r = fit_route(points)
    for o in sorted(r['points'], key=lambda o: o['load']):
        loo = f"{o['loo']:+5.1f}" if o['loo'] is not None else '    -'
        mark = (' MISCLASSIFIED' if o['wrong'] else '') + \
            (' (leave-one-out: misclassified)' if o['loo_wrong'] else '')
        print(f"{o['name']:<12} {o['luts']:>9} LUTs, load {o['load']:>9}, S {o['slrs']:<3g}"
              f"{o['routed']:>5} measured {o['y']:5.1f} fit {o['est']:5.1f} ({o['res']:+5.1f}) "
              f"leave-one-out {loo}{mark}")
    a_, b_ = r['coef']

    def load_at(slrs):
        return 1e6 * (-a_ + math.sqrt(a_ * a_ + 4 * b_ / slrs * r['limit'])) / (2 * b_ / slrs)

    def wrong(r, key):
        return ', '.join(o['name'] for o in r['points'] if o[key]) or 'none'
    print(f"congestion = {a_:.3f} L + {b_:.3f} L (L / S) (L in millions, S in SLRs); "
          f"RMS {r['rms']:.2f}, leave-one-out RMS {r['loo']:.2f} over {len(r['points'])} builds")
    print(f"fit of the routed builds up to {r['routed']:.2f} %, of the failed from "
          f"{r['failed']:.2f} %: limit {r['limit']:.2f} % = a load of "
          f"{load_at(REGION['slrs']):.0f} over the region's {REGION['slrs']:g} SLRs, "
          f"{load_at(DEVICE_SLRS):.0f} over {DEVICE_SLRS}")
    print(f"misclassified: {wrong(r, 'wrong')}; leave-one-out: {wrong(r, 'loo_wrong')}")
    base = fit_route(points, layout=False)
    print(f"without the layout term (L and L^2): RMS {base['rms']:.2f}, leave-one-out "
          f"{base['loo']:.2f}; routed up to {base['routed']:.2f} %, failed from "
          f"{base['failed']:.2f} %; misclassified: {wrong(base, 'wrong')}")


if __name__ == '__main__':
    main()
