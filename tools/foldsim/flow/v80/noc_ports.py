#!/usr/bin/env python3
"""The dynamic region's memory NoC ports, read from a static shell's NoC solution.

noc_ports.py <top_wrapper.ncr> [--rp top_i/slash/] [--rows SLR0:0-6,SLR1:7-12,SLR2:13-18] [--json <out>]

A Versal DFX shell fixes the NoC paths that cross the boundary of its
reconfigurable partition (RP): each RP NoC master unit (NMU) keeps the site and
the route the shell's NoC compiler gave it, and an RM that places it anywhere
else fails opt_design (DRC HDPR-122, 2026-09-01). An RM cannot add a path to a
static destination either: the routes run through the static region's NoC
switches. So the shell's NoC solution (the .ncr JSON the RM link reads, or the
one in a linked image's impl_1) says which memory ports the kernel has and in
which SLR each one enters the NoC.

For each NMU in the RP (`--rp`, a prefix of the logical instance name) this
prints its BD name, its site, the site's SLR (by the site's row: `--rows`, the
NOC_NMU512 rows of each SLR; xcv80's by default), the memories its paths reach
(HBM, DDR; else `other`) and the NoC compiler's achieved read latency over its
paths (min / mean / max). With --json it writes them as the region's `noc`
section (foldsim_v80_slr.json): the per-SLR virtual-memory engines
(foldsim_sll_check.py --vmem-engines local) bind each engine's master to a
memory NMU and price the SLR boundaries between them.
"""

import argparse
import json
import re
import sys

# xcv80: the NOC_NMU512 rows of each SLR (Vivado 2025.2, get_sites by SLR,
# slrxing_1108/slr_yranges.tsv).
V80_ROWS = 'SLR0:0-6,SLR1:7-12,SLR2:13-18'


def parse_rows(spec):
    """[(SLR name, first row, last row)] from `SLR0:0-6,...`."""
    out = []
    for part in spec.split(','):
        name, rng = part.split(':')
        lo, hi = rng.split('-')
        out.append((name, int(lo), int(hi)))
    return out


def site_slr(site, rows):
    m = re.search(r'_X\d+Y(\d+)$', site)
    if not m:
        return None
    y = int(m.group(1))
    return next((n for n, lo, hi in rows if lo <= y <= hi), None)


def target(dest):
    d = dest.lower()
    return 'HBM' if 'hbm' in d else 'DDR' if 'ddr' in d else 'other'


def noc_ports(ncr, rp, rows):
    """The RP's NMUs: [{name, site, slr, targets, read_latency}] in name order."""
    site = {c['TrafficLInst']: c['Name'] for c in ncr.get('Components', []) if 'TrafficLInst' in c}
    nmus = {li['Name']: li for li in ncr.get('LogicalInstances', [])
            if li.get('IsMaster') and li.get('CompType') == 'PL_NMU' and li['Name'].startswith(rp)}
    lat, tgts = {}, {}
    for p in ncr.get('Paths', []):
        if p['From'] not in nmus:
            continue
        tgts.setdefault(p['From'], set()).add(target(p['To']))
        lat.setdefault(p['From'], []).extend(n['AchievedLatency'] for n in p.get('Nets', [])
                                             if n.get('CommType') == 'READ')
    out = []
    for name, li in nmus.items():
        s = site.get(name, '')
        ls = lat.get(name) or [0]
        out.append(dict(name=li.get('DesignName', name).strip('/').split('/')[0], site=s, slr=site_slr(s, rows),
                        targets=sorted(tgts.get(name, ())),
                        read_latency=[min(ls), round(sum(ls) / len(ls), 1), max(ls)]))
    return sorted(out, key=lambda x: x['name'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('ncr', help="a NoC solution (.ncr JSON): the shell's, or an image's impl_1/top_wrapper.ncr")
    ap.add_argument('--rp', default='top_i/slash/', help='the RP: a prefix of its NoC instances (default: SLASH\'s)')
    ap.add_argument('--rows', default=V80_ROWS, help=f'each SLR\'s NMU rows (default xcv80: {V80_ROWS})')
    ap.add_argument('--json', help="write the region's `noc` section here")
    a = ap.parse_args()
    rows = parse_rows(a.rows)
    ports = noc_ports(json.load(open(a.ncr)), a.rp, rows)
    if not ports:
        sys.exit(f'noc_ports.py: no NoC master unit under {a.rp} in {a.ncr}')
    for p in ports:
        lo, mean, hi = p['read_latency']
        print(f'{p["name"]}\t{p["site"]}\t{p["slr"]}\t{",".join(p["targets"])}\tread latency {lo}/{mean}/{hi}')
    for slr, _, _ in rows:
        n = {t: sum(1 for p in ports if p['slr'] == slr and t in p['targets']) for t in ('HBM', 'DDR')}
        print(f'{slr}: {n["HBM"]} HBM, {n["DDR"]} DDR')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(dict(rows={n: [lo, hi] for n, lo, hi in rows}, ports=ports), f, indent=1)


if __name__ == '__main__':
    main()
