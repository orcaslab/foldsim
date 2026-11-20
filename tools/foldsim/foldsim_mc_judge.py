#!/usr/bin/env python3
"""Judge a logic screen's end points with foldsim's multicycle paths per
allowed host cycle.

usage: foldsim_mc_judge.py <endpoints.csv> <foldsim_sim.json> [--min-levels N]

<endpoints.csv> is an end point table (beacon-estimate-timing's `output`, or
Beacon's run/endpoints.csv): a row per end point bit with its deepest path's
`levels`, `start` and `end`. foldsim_sim.json's `multicycle` lists each
multicycle copy's register prefix and its host cycles M (foldsim --multicycle):
a path from its capture registers (<prefix>cap) to its end registers
(<prefix>end) has M host cycles, and is judged at ceil(levels / M) levels.

A path that ends at an end register but starts elsewhere, or starts at a
capture register and ends elsewhere, is outside the constraints: it is
reported and the exit status is 1 (it would be timed as one host cycle by the
flow, or worse, its constraint would cover a path that is not the cone's).
Prints the end points by judged levels and the deepest, and with
--min-levels N the count at or above N (the fix list)."""
import argparse
import csv
import json
import re
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('endpoints')
    ap.add_argument('sim_json')
    ap.add_argument('--min-levels', type=int, default=12)
    a = ap.parse_args()
    mc = json.load(open(a.sim_json)).get('multicycle', [])
    bad, hist, deepest = [], Counter(), []
    for r in csv.DictReader(open(a.endpoints)):
        levels, start, end = int(r['levels']), r['start'], r['end']
        judged, cycles = levels, 1
        # A register's own hold (its clock enable, which the screen counts
        # as a choice of its old value) is no path between registers.
        if re.sub(r'\[\d+\]$', '', start) == re.sub(r'\.\w+\[\d+\]$', '', end):
            hist[judged] += 1
            continue
        for e in mc:
            p = e['prefix']
            s_cap, e_end = f'{p}cap' in start, f'{p}end' in end
            if e_end and not s_cap and levels > 0:
                bad.append(f'{end} <- {start}: into end registers from outside the capture registers')
            elif s_cap and not e_end:
                bad.append(f'{end} <- {start}: from capture registers to outside the end registers')
            elif s_cap and e_end:
                cycles = e['cycles']
                judged = -(-levels // cycles)
        hist[judged] += 1
        deepest.append((judged, levels, cycles, re.sub(r'\[\d+\]', '[]', end), re.sub(r'\[\d+\]', '[]', start)))
    print('end points by levels per allowed host cycle: ' +
          ' '.join(f'{k}:{hist[k]}' for k in sorted(hist)))
    seen = set()
    print('deepest (judged, levels, host cycles, end <- start):')
    for d in sorted(deepest, reverse=True):
        if (d[3], d[4]) in seen:
            continue
        seen.add((d[3], d[4]))
        print(f'  {d[0]} {d[1]} {d[2]} {d[3]} <- {d[4]}')
        if len(seen) == 8:
            break
    print(f'at or above {a.min_levels} levels per host cycle: '
          f'{sum(n for k, n in hist.items() if k >= a.min_levels)}')
    for b in bad[:20]:
        print('OUTSIDE THE CONSTRAINTS: ' + b)
    if bad:
        print(f'{len(bad)} end points outside the multicycle constraints')
        sys.exit(1)


if __name__ == '__main__':
    main()
