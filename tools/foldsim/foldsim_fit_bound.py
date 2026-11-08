#!/usr/bin/env python3
"""foldsim_fit_bound.py <gen dir | --count count.json | --schedule schedule.txt> [--fit fit.json]
                       [--slr-region v80|<json>] [--json out] [--sll <sll.txt>]

Whether a fold can fit its region, decided on bounds rather than on a point estimate, in every resource
(LUT, LUTRAM, FF, URAM288, RAMB36), for the whole design and for each class copy against the SLRs.

Levels, cheapest first:

  --schedule  a schedule report (foldsim-fold-static schedule-only): the link storage the compiler has decided
              (link register and staged register bits, link memories' LUTRAM LUTs, RAMB36, URAM). A floor only:
              class state, read buffers, captures and folded-pipeline stores are made when the netlist is
              built and are not in the report.
  <gen dir>   the generated netlist (foldsim_sim.mlir), counted (foldsim_fitdb.count; or --count, its JSON):
              every storage primitive the fold built, by kind, and foldsim_cost's logic LUTs.
  --fit       the calibration database's fit (foldsim_fitdb.py fit): measured per counted over the designs
              synthesized so far, per resource (min / median / max), and the kernel beside the simulator.

Bounds per resource (the design: the simulator and the kernel beside it):
  floor  without --fit, what the netlist's storage is at the least, uncalibrated: distributed RAM that
         cannot be anything else (read combinationally, deeper than 64: `lutram_forced`, RAM64M8 packing,
         no read select) in LUTs and LUTRAM, URAM288 by ram_style times 0.99 (Vivado builds them as asked:
         measured / counted 0.992-1.05 over five designs); flops, logic and block RAM 0.
         With --fit, also the count times the LEAST measured/counted of any design, the kernel at its least.
  ucb    the count times the GREATEST measured/counted of any design, the kernel at its greatest.
  A copy: its own module at the copy fit's ratios, the links read into it (placed with it) at the design's;
         its URAM and RAMB36 as counted (ram_style).
Verdict (exit 1 on REFUSE), the first that applies per resource:
  REFUSE IMPOSSIBLE  a design floor of LUT / LUTRAM / FF above the region's capacity, or of URAM / RAMB36
                     above `memory_fill` of it (g16x-o2, 11-06: 96 % of the region's URAM synthesized, SLR2
                     legalization failed; FSD vD, 11-07: 92 %, placed); a copy's URAM or RAMB36, as counted,
                     above the largest SLR's (the copy must straddle on its widest ports: r16c, 11-07, 517 URAM
                     in one copy against 496 / 511 / 204, failed placement on SLLs). Change the recipe.
  REFUSE LIKELY      the median estimate (the designs' median ratio, the kernel's median) above `fill` of the
                     region (LUT / LUTRAM / FF) or above `memory_fill` (URAM / RAMB36). Change the recipe.
  REFUSE             the SLL check's (--sll) refusal, or its projection times (1 + --sll-error) past the
                     boundary's SLLs.
  WARN UNPROVEN      a design ucb above the region's capacity (LUT / LUTRAM / FF) or above `memory_fill`;
  WARN               a design ucb above `fill`, or a copy's logic ucb above the largest SLR's margin (the
                     placer cuts it: b4e, y8r1di placed so). An image may start, gated after its synthesis:
                     this check again with --synthesis on the synthesis' hierarchical utilization report must
                     pass before placement.
  PASS               otherwise (the image is still gated after synthesis).
  --synthesis        the measured AFU and copies replace the bounds (the image's post-synthesis gate).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOGIC = ('lut', 'lutram', 'ff')
MEMS = ('uram', 'bram')
RES = LOGIC + MEMS
FFS_PER_LUT = 2
DEFAULT_FILL = dict(lut=0.80, lutram=0.80, ff=0.80)
# Between the largest memory fill that placed (FSD vD: 92 % of the region's URAM) and the least that failed
# (g16x-o2: 96 % URAM, 94 % RAMB36): two observations, the line in between.
MEMORY_FILL = 0.94
# URAM288 as built per counted by ram_style, the least seen (g16x-o2: 1,161 of 1,168): an uncalibrated floor.
URAM_FLOOR = 0.99


def region_caps(region):
    slrs = region['slrs']
    tot = dict(lut=sum(s['luts'] for s in slrs), lutram=sum(s.get('lutram_luts', s['luts']) for s in slrs),
               ff=FFS_PER_LUT * sum(s['luts'] for s in slrs), uram=sum(s.get('uram', 0) for s in slrs),
               bram=sum(s.get('ramb36', 0) for s in slrs))
    largest = dict(lut=max(s['luts'] for s in slrs), lutram=max(s.get('lutram_luts', s['luts']) for s in slrs),
                   ff=FFS_PER_LUT * max(s['luts'] for s in slrs), uram=max(s.get('uram', 0) for s in slrs),
                   bram=max(s.get('ramb36', 0) for s in slrs))
    return tot, largest


def schedule_floor(path):
    """The link storage of a schedule report: {lut, lutram, ff, uram, bram} as the compiler counted it."""
    text = open(path).read()
    out = dict(lut=0.0, lutram=0.0, ff=0.0, uram=0.0, bram=0.0)
    m = re.search(r'link kinds: distributed (\d+) (\d+) ([\d.]+); block (\d+) (\d+) ([\d.]+); ultra (\d+) (\d+) '
                  r'([\d.]+); registers (\d+) (\d+)', text)
    if m:
        out['lutram'] = out['lut'] = float(m.group(3))
        out['bram'] = float(m.group(6))
        out['uram'] = float(m.group(9))
        out['ff'] = float(m.group(11))
    s = re.search(r'staged: (\d+) links, (\d+) register bits', text)
    # Staged link registers are link registers too: the larger of the two counts, never their sum.
    out['ff'] = max(out['ff'], float(s.group(2)) if s else 0.0)
    ii = re.search(r'II = (\d+)', text)
    return out, int(ii.group(1)) if ii else None


def ratio(fit, scope, r, which):
    s = ((fit or {}).get(scope) or {}).get(r)
    return s[which] if s else None


def kernel(fit, region, which):
    k = (fit or {}).get('kernel') or {}
    out = {}
    for r in RES:
        s = k.get(r)
        out[r] = max(0.0, s[which]) if s else (region.get('kernel', {}).get(r, 0.0) if which != 'min' else 0.0)
    return out


def storage_floor(cnt):
    """What a count's storage is at the least, uncalibrated."""
    forced = cnt.get('lutram_forced', 0.0)
    return dict(lut=forced, lutram=forced, ff=0.0, uram=URAM_FLOOR * cnt.get('uram', 0.0), bram=0.0)


def design_mid(sim, fit, region):
    """The median estimate: the count at the designs' median ratio, the kernel at its median."""
    if not fit:
        return None
    km = kernel(fit, region, 'median')
    return {r: (ratio(fit, 'design', r, 'median') or 1.0) * sim.get(r, 0.0) + km[r] for r in RES}


def design_bounds(sim, fit, region):
    fl = storage_floor(sim)
    uc = None
    if fit:
        # The kernel beside the simulator is set by the kernel's configuration (ports, virtual-memory
        # masters), not a statistic of the designs: the least and the greatest one measured.
        kmin, kmax = kernel(fit, region, 'min'), kernel(fit, region, 'max')
        uc = {}
        for r in RES:
            lo, hi = ratio(fit, 'design', r, 'lo'), ratio(fit, 'design', r, 'hi')
            fl[r] = max(fl[r], (lo if lo is not None else 1.0) * sim.get(r, 0.0)) + kmin[r]
            uc[r] = (hi if hi is not None else 1.0) * sim.get(r, 0.0) + kmax[r]
    return fl, uc


def copy_bounds(own, links, fit):
    tot = {r: own.get(r, 0.0) + links.get(r, 0.0) for r in RES + ('lutram_forced',)}
    fl = storage_floor(tot)
    # A copy's memories as counted: ram_style is honoured (URAM), block RAM counted at most what is built.
    fl['bram'] = 0.0
    uc = None
    if fit:
        uc = {}
        for r in LOGIC:
            lo, hi = ratio(fit, 'copy', r, 'lo'), ratio(fit, 'copy', r, 'hi')
            dlo, dhi = ratio(fit, 'design', r, 'lo'), ratio(fit, 'design', r, 'hi')
            fl[r] = max(fl[r], max(0.0, lo or 0.0) * own.get(r, 0.0) + max(0.0, dlo or 0.0) * links.get(r, 0.0))
            uc[r] = (hi or 1.0) * own.get(r, 0.0) + (dhi or 1.0) * links.get(r, 0.0)
        for r in MEMS:
            uc[r] = tot[r]
    return fl, uc, tot


def pct(x, c):
    return f'{100 * x / c:.0f} %' if c else '-'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gen', nargs='?')
    ap.add_argument('--count', help='foldsim_fitdb.py count --json of the netlist (instead of counting it here)')
    ap.add_argument('--schedule', help='a schedule report: the link-storage floor only')
    ap.add_argument('--fit', help='foldsim_fitdb.py fit --out of the calibration database')
    ap.add_argument('--slr-region', default='v80')
    ap.add_argument('--sll', help='the SLL check\'s report (foldsim_sll_check.py sll.txt) of this netlist')
    ap.add_argument('--sll-error', type=float, default=0.11,
                    help='the SLL projection\'s worst relative error on placed builds (11-07: -11 %%..+9 %%)')
    ap.add_argument('--memory-fill', type=float, default=MEMORY_FILL)
    ap.add_argument('--synthesis', help='a hierarchical utilization report of a synthesis of this netlist: the '
                                        'measured AFU and copies instead of bounds (the image\'s post-synthesis gate)')
    ap.add_argument('--name')
    ap.add_argument('--json')
    a = ap.parse_args()
    region_path = os.path.join(HERE, 'foldsim_v80_slr.json') if a.slr_region == 'v80' else a.slr_region
    region = json.load(open(region_path))
    cap, largest = region_caps(region)
    fills = dict(DEFAULT_FILL, **(region.get('placement', {}).get('region_fill', {})))
    margin = region.get('margin', 0.85)
    fit = json.load(open(a.fit)) if a.fit else None
    name = a.name or os.path.basename((a.gen or a.count or a.schedule or '').rstrip('/'))
    lines, refuse, warn = [], [], []
    res = dict(name=name, cap=cap)

    if a.schedule:
        fl, ii = schedule_floor(a.schedule)
        lines.append(f'{name}: schedule floor (II {ii}), link storage only: '
                     + ', '.join(f'{r} {fl[r]:,.0f} ({pct(fl[r], cap[r])})' for r in RES))
        refuse += [f'schedule floor {r} {pct(fl[r], cap[r])} of the region' for r in LOGIC if fl[r] > cap[r]]
        refuse += [f'schedule floor {r} {pct(fl[r], cap[r])} of the region' for r in MEMS
                   if fl[r] > a.memory_fill * cap[r]]
        res.update(level='schedule', floor=fl)
    else:
        if a.count:
            cnt = json.load(open(a.count))
        else:
            import foldsim_fitdb
            cnt = foldsim_fitdb.count(a.gen)
        sim = cnt['sim']
        fl, uc = design_bounds(sim, fit, region)
        mid = design_mid(sim, fit, region)
        meas = None
        if a.synthesis:
            import foldsim_fitdb
            meas = foldsim_fitdb.measured(a.synthesis, cnt['copies'])
            if not meas['afu']:
                sys.exit(f'{a.synthesis}: no foldsim_afu_0 row')
            fl = {r: float(meas['afu'][r]) for r in RES}
            uc = dict(fl)
            mid = dict(fl)
            lines.append(f'  measured ({meas["state"]}, {a.synthesis}): the AFU')
        lines.append(f'{name}: {len(cnt["copies"])} copies; counted (simulator): '
                     + ', '.join(f'{r} {sim.get(r, 0):,.0f}' for r in RES + ('lutram_forced', 'logic', 'dsp')))
        lines.append('  region: ' + ', '.join(f'{r} {cap[r]:,.0f}' for r in RES) + '; fill '
                     + ', '.join(f'{r} {100 * fills[r]:.0f} %' for r in LOGIC) + f', memories {100 * a.memory_fill:.0f} %')
        lines.append(f'  design floor{"" if fit else " (uncalibrated storage)"}: '
                     + ', '.join(f'{r} {fl[r]:,.0f} ({pct(fl[r], cap[r])})' for r in RES))
        if uc:
            lines.append('  design median: ' + ', '.join(f'{r} {mid[r]:,.0f} ({pct(mid[r], cap[r])})' for r in RES))
            lines.append('  design ucb: ' + ', '.join(f'{r} {uc[r]:,.0f} ({pct(uc[r], cap[r])})' for r in RES))
        for r in LOGIC:
            if fl[r] > cap[r]:
                refuse.append(f'IMPOSSIBLE: design floor {r} {pct(fl[r], cap[r])} of the region')
            elif mid and mid[r] > fills[r] * cap[r]:
                refuse.append(f'LIKELY: design median {r} {pct(mid[r], cap[r])} of the region, past the fill '
                              f'{100 * fills[r]:.0f} %')
            elif uc and uc[r] > cap[r]:
                warn.append(f'UNPROVEN: design ucb {r} {pct(uc[r], cap[r])} of the region')
            elif uc and uc[r] > fills[r] * cap[r]:
                warn.append(f'design ucb {r} {pct(uc[r], cap[r])} of the region, past the fill {100 * fills[r]:.0f} %')
        for r in MEMS:
            if fl[r] > a.memory_fill * cap[r]:
                refuse.append(f'IMPOSSIBLE: design floor {r} {pct(fl[r], cap[r])} of the region, past '
                              f'{100 * a.memory_fill:.0f} %')
            elif mid and mid[r] > a.memory_fill * cap[r]:
                refuse.append(f'LIKELY: design median {r} {pct(mid[r], cap[r])} of the region, past '
                              f'{100 * a.memory_fill:.0f} %')
            elif uc and uc[r] > a.memory_fill * cap[r]:
                warn.append(f'UNPROVEN: design ucb {r} {pct(uc[r], cap[r])} of the region, past '
                            f'{100 * a.memory_fill:.0f} %')
        rows = []
        links = cnt.get('links', {})
        for c, own in cnt['copies'].items():
            cf, cu, tot = copy_bounds(own, links.get(c, {}), fit)
            if meas and meas['copies'].get(c):
                # The copy as synthesized, the links read into it at the design's ratio (measured / counted).
                m = meas['copies'][c]
                lk = links.get(c, {})
                k = {r: (meas['sim'][r] / sim[r]) if meas['sim'] and sim.get(r) else 1.0 for r in RES}
                cf = {r: float(m[r]) + k[r] * lk.get(r, 0.0) for r in RES}
                cu = dict(cf)
                tot = dict(cf)
            rows.append((c, tot, cf, cu))
        rows.sort(key=lambda x: -max(x[1].get(r, 0.0) / largest[r] for r in RES if largest[r]))
        for c, tot, cf, cu in rows[:4]:
            lines.append(f'  copy {c}: floor ' + ', '.join(f'{r} {pct(cf[r], largest[r])}' for r in RES)
                         + ((' | ucb ' + ', '.join(f'{r} {pct(cu[r], largest[r])}' for r in RES)) if cu else '')
                         + ' of the largest SLR\'s')
        for c, tot, cf, cu in rows:
            for r in MEMS:
                if tot[r] > largest[r]:
                    refuse.append(f'IMPOSSIBLE: copy {c} {r} {tot[r]:,.0f} > the largest SLR\'s {largest[r]:,.0f}')
            for r in LOGIC:
                if cf[r] > largest[r]:
                    warn.append(f'copy {c} {r} floor {pct(cf[r], largest[r])} of the largest SLR: the placer cuts it')
                elif cu and cu[r] > margin * largest[r]:
                    warn.append(f'copy {c} {r} ucb {pct(cu[r], largest[r])} of the largest SLR (margin '
                                f'{100 * margin:.0f} %)')
        res.update(level='netlist' + ('+fit' if fit else ''), floor=fl, ucb=uc,
                   copies={c: dict(floor=cf, ucb=cu, count=tot) for c, tot, cf, cu in rows})
    if a.sll and os.path.exists(a.sll):
        last = [l for l in open(a.sll) if re.match(r'^\S+: (OK|WARN|REFUSE)', l)]
        if last:
            v = last[-1].strip()
            m = re.search(r'([\d,]+) SLLs projected', v)
            lines.append(f'  SLL check: {v}')
            if v.split(': ')[1].startswith('REFUSE'):
                refuse.append('the SLL check refuses')
            elif m and float(m.group(1).replace(',', '')) * (1 + a.sll_error) > region.get('sll_per_boundary', 1e18):
                refuse.append(f'{m.group(1)} SLLs x {1 + a.sll_error:.2f} past the boundary\'s '
                              f'{region.get("sll_per_boundary"):,}')
    verdict = 'REFUSE: ' + '; '.join(refuse) if refuse else ('WARN: ' + '; '.join(warn[:6]) if warn else 'PASS')
    lines.append(f'{name}: {verdict}')
    print('\n'.join(lines))
    if a.json:
        res['verdict'] = verdict
        json.dump(res, open(a.json, 'w'), indent=1, sort_keys=True)
    sys.exit(1 if refuse else 0)


if __name__ == '__main__':
    main()
