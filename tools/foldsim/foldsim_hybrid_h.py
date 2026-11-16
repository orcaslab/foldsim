#!/usr/bin/env python3
"""foldsim_hybrid_h.py: the S26 hybrid's projected H per SLR region, from what the card measured.

A generation's virtual memories (foldsim_sim.json `vmem`: each memory's class, instances, row bytes and, with
--vmem-slr-groups, its SLR group) are its rows in HBM, fetched on demand. Each region (an SLR group: the
controllers of its classes, on its own HBM ports) moves, per frame, a row read and a row write per instance of each
of its memories that an access touches (`--alpha`: the share of instances whose port is active in a target cycle;
1 = every instance every frame, the policy's bound), each at least one 64-byte AXI beat. At the region's measured
bandwidth (--hbm: a calibration JSON, `busy` = every SLR moving at once, `alone` = one SLR) that takes

    hbm cycles = max(read bytes / read rate, write bytes / write rate)   (rates in bytes per host cycle)

and a read that misses the gap between its request and its use (the schedule's state gap, --gap) waits for the
rest of the loaded round trip (`loaded_latency_ns`, Little's law: the requests the region keeps in flight to reach
its rate) once a frame per serialized class copy (the bubbles). The region's H is

    H_region = max(II, hbm cycles) + bubbles,   H = the largest region's,

and the requests in flight a region needs (Little's law) = rate x loaded latency / bytes per request.

Usage: foldsim_hybrid_h.py <gen dir> [--groups sll.json] [--hbm JSON] [--ii N] [--alpha A] [--gap G] [--mode busy|alone]
Prints one line per region and a summary line `hybrid H <H> (II <ii>, worst <region>)`."""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEAT = 64


def regions_of(vmems, groups, slr_names):
    """Each virtual memory's SLR name: its own `slr_group` (--vmem-slr-groups), else its class's group in the SLL
    check's assignment (`groups`), else the last SLR (where the level and the kernel sit by default)."""
    of_class = {}
    if groups:
        for g, members in enumerate(groups.get('assignment') or []):
            for c in members:
                of_class[c] = g
    out = []
    for v in vmems:
        g = v.get('slr_group')
        if g is None:
            g = of_class.get(v['class'], of_class.get(re.sub(r'_copy\d+$', '', v['class'])))
        out.append(slr_names[g] if g is not None and g < len(slr_names) else slr_names[0])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('gen')
    ap.add_argument('--groups', help="the SLL check's sll.json (its `assignment`: the classes of each SLR group)")
    ap.add_argument('--hbm', default=os.path.join(HERE, 'calibrations', 'hbm-asl9-20261114.json'))
    ap.add_argument('--mode', choices=('busy', 'alone'), default='busy')
    ap.add_argument('--ii', type=int, default=0, help="the frame (default: the generation's schedule's II)")
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--gap', type=int, default=0, help='host cycles between a row request and its use')
    ap.add_argument('--slr-names', default='SLR2,SLR1,SLR0', help="the groups' SLRs in order (--slr-region's)")
    a = ap.parse_args(argv)
    gdir = a.gen if os.path.exists(os.path.join(a.gen, 'foldsim_sim.json')) else os.path.join(a.gen, 'gen')
    sim = json.load(open(os.path.join(gdir, 'foldsim_sim.json')))
    hbm = json.load(open(a.hbm))
    ii = a.ii
    if not ii:
        m = re.search(r'II = (\d+)', open(os.path.join(gdir, 'schedule.txt')).read())
        ii = int(m.group(1)) if m else 0
    groups = json.load(open(a.groups)) if a.groups else None
    names = a.slr_names.split(',')
    vmems = sim.get('vmem') or []
    where = regions_of(vmems, groups, names)
    clk = hbm.get('clock_hz', 200e6)
    per = {}
    for v, slr in zip(vmems, where):
        r = per.setdefault(slr, dict(mems=0, rd=0.0, wr=0.0, classes=set()))
        req = max(BEAT, v.get('row_bytes', BEAT))
        n = v.get('instances', 1) * a.alpha
        r['mems'] += 1
        r['rd'] += n * req
        r['wr'] += n * req
        r['classes'].add(v['class'])
    worst, worst_h = None, ii
    for slr in names:
        if slr not in per:
            print(f'{slr}: no rows in HBM, H {ii} (the schedule)')
            continue
        r = per[slr]
        cal = hbm['slrs'][slr][a.mode]
        rd_rate = cal['read_gbs'] * 1e9 / clk
        wr_rate = cal['write_gbs'] * 1e9 / clk
        cyc = max(r['rd'] / rd_rate, r['wr'] / wr_rate)
        lat = hbm['slrs'][slr]['loaded_latency_ns' if a.mode == 'busy' else 'latency_ns'] * 1e-9 * clk
        bubbles = len(r['classes']) * max(0.0, lat - a.gap)
        h = max(ii, cyc) + bubbles
        flight = math.ceil(rd_rate * lat / BEAT)
        print(f'{slr}: {r["mems"]} memories of {len(r["classes"])} classes, {r["rd"]:,.0f} B read + {r["wr"]:,.0f} B '
              f'written a frame (alpha {a.alpha:g}); {a.mode} {cal["read_gbs"]:g} / {cal["write_gbs"]:g} GB/s = '
              f'{rd_rate:.0f} / {wr_rate:.0f} B a host cycle -> {cyc:,.0f} cycles; bubbles {bubbles:,.0f} '
              f'(loaded round trip {lat:.0f} cycles, gap {a.gap}); H {h:,.0f}; reads in flight for the rate '
              f'(Little) {flight}')
        if h > worst_h:
            worst, worst_h = slr, h
    print(f'hybrid H {worst_h:,.0f} (II {ii}, worst {worst or "the schedule"}, {a.mode}, alpha {a.alpha:g})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
