#!/usr/bin/env python3
"""foldsim_ctrl_check.py <foldsim_sim.sv> [--slices SLR=N,...] [--warn F] [--refuse F] [--json out.json]

The control sets a simulator's LUT memories cost, per SLR group, before synthesis.

Synthesis builds a LUT memory (distributed RAM: `ram_style = "distributed"`, any depth) of 64-entry LUT RAM banks,
each bank with its own write enable. A write enable is a control set: on Versal it reaches the CLE's control pins
(CTRL) through the few CNODEs of the interconnect tile, which also carry every flip-flop clock enable and reset of
the CLEs around it. A design of many small control sets (a write enable that drives one LUT RAM) runs out of CNODEs
in the dense parts of an SLR and fails routing with node overlaps on `CNODE_OUTS_*`, whatever its SLL or LUT use
(FSD 1,024, 11-14..11-17: 58,355 control sets, 48,230 of fewer than 4 BELs; the LKG that routed: 18,549).

Counted: each instance of a memory module (`<name>_<D>x<W>`, `reg [..] Memory[0:D-1]` with ram_style distributed or
no block/ultra style) as ceil(D/64) write-enable control sets; TINY when the bank holds fewer bits than one LUT-RAM
slice (8 LUTs: W < 4 for a dual-port RAM64X1D bank). Rolled up the instance tree to the `foldsim_sim_slr<k>` modules
(the SLR groups) or the top. Flip-flop clock enables are not counted (a register's enable is not visible in the SV
without elaboration); the region controls' enables are a few thousand per design and did not change between the
LKG and the failing images.

Verdict per SLR group, on predicted control sets / the group's SLICEs (--slices, from the region file): REFUSE past
--refuse (default 0.08), WARN past --warn (default 0.06). Calibration (2 points, Vivado 2025.2 placed):
LKG fsd1kr200 max 5.5 % of an SLR's SLICEs (routed), F28s 10.6 % / 16.4 % (failed). The predicted count is the
synthesized one before opt_design merges equivalent decodes (F28s: predicted 66,883, placed 58,355).
"""
import argparse, collections, json, math, re, sys


def parse(path):
    mods = {}          # module -> {'mems': [(D, W)], 'inst': [(module, count)]}
    memdecl = {}       # memory module -> (D, W, style)
    cur = None
    modre = re.compile(r'^module (\w+)\s*\(')
    regre = re.compile(r'^\s*reg\s*(?:\[(\d+):0\])?\s*Memory\s*\[0:(\d+)\]')
    stylere = re.compile(r'ram_style\s*=\s*"(\w+)"')
    instre = re.compile(r'^\s+(\w+)\s+(?:#\(.*?\)\s*)?(\w+)\s*\(')
    pend_style = None
    known = set()
    lines = []
    with open(path, errors='replace') as f:
        for line in f:
            m = modre.match(line)
            if m:
                cur = m.group(1); known.add(cur); mods[cur] = {'mems': [], 'inst': collections.Counter()}; pend_style = None
                continue
            if cur is None:
                continue
            if line.startswith('endmodule'):
                cur = None; continue
            s = stylere.search(line)
            if s:
                pend_style = s.group(1)
            r = regre.match(line)
            if r:
                w = int(r.group(1)) + 1 if r.group(1) else 1
                memdecl[cur] = (int(r.group(2)) + 1, w, pend_style or '')
                pend_style = None
                continue
            i = instre.match(line)
            if i and not line.lstrip().startswith(('reg', 'wire', 'input', 'output', 'logic', 'assign', 'always', 'if', 'else', 'for', 'end', 'begin', 'case', '`')):
                mods[cur]['inst'][i.group(1)] += 1
    return mods, memdecl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sv')
    ap.add_argument('--slices', default='', help='SLR group slices, e.g. slr0=106000,slr1=102500,slr2=106700 (all=N for an ungrouped design)')
    ap.add_argument('--warn', type=float, default=0.06)
    ap.add_argument('--refuse', type=float, default=0.08)
    ap.add_argument('--json')
    a = ap.parse_args()
    mods, memdecl = parse(a.sv)
    own = {}
    for m, d in memdecl.items():
        D, W, st = d
        if st in ('block', 'ultra'):
            own[m] = (0, 0, 0)
            continue
        banks = max(1, math.ceil(D / 64))
        own[m] = (banks, banks if W < 4 else 0, 1)
    memo = {}

    def roll(m, depth=0):
        if m in memo:
            return memo[m]
        if depth > 200:
            return (0, 0, 0)
        cs, tiny, n = own.get(m, (0, 0, 0))
        for sub, k in mods.get(m, {}).get('inst', {}).items():
            if sub in mods:
                c2, t2, n2 = roll(sub, depth + 1)
                cs += k * c2; tiny += k * t2; n += k * n2
        memo[m] = (cs, tiny, n)
        return memo[m]

    groups = sorted(m for m in mods if re.fullmatch(r'foldsim_sim_slr\d+', m))
    tops = groups or [m for m in mods if m == 'foldsim_sim']
    slices = dict(kv.split('=') for kv in a.slices.split(',') if kv)
    out = {'groups': {}, 'verdict': 'OK'}
    total = roll('foldsim_sim') if 'foldsim_sim' in mods else (0, 0, 0)
    print(f'foldsim_ctrl_check: {a.sv}: predicted LUT-RAM write-enable control sets {total[0]:,} '
          f'({total[1]:,} tiny: banks narrower than a LUT-RAM slice) over {total[2]:,} LUT memories')
    worst = 0.0
    for g in tops:
        cs, tiny, n = roll(g)
        key = g.replace('foldsim_sim_', '') if g != 'foldsim_sim' else 'all'
        sl = float(slices.get(key, 0) or 0)
        frac = cs / sl if sl else None
        v = 'OK'
        if frac is not None:
            worst = max(worst, frac)
            v = 'REFUSE' if frac > a.refuse else 'WARN' if frac > a.warn else 'OK'
        out['groups'][key] = {'control_sets': cs, 'tiny': tiny, 'memories': n, 'slices': sl, 'fraction': frac, 'verdict': v}
        print(f'  {key}: {cs:,} control sets ({tiny:,} tiny) over {n:,} LUT memories' +
              (f'; {100 * frac:.1f} % of {int(sl):,} SLICEs -> {v}' if frac is not None else ''))
    if slices:
        out['verdict'] = 'REFUSE' if worst > a.refuse else 'WARN' if worst > a.warn else 'OK'
        print(f'foldsim_ctrl_check: {out["verdict"]} (worst group {100 * worst:.1f} % of its SLICEs; warn {100 * a.warn:.0f} %, refuse {100 * a.refuse:.0f} %)')
    # the largest contributors
    contrib = sorted(((roll(m)[0], m) for m in memo if m not in groups and m != 'foldsim_sim'), reverse=True)[:8]
    for cs, m in contrib:
        print(f'    {m[:70]}: {cs:,}')
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)
    return 1 if out['verdict'] == 'REFUSE' else 0


if __name__ == '__main__':
    sys.exit(main())
