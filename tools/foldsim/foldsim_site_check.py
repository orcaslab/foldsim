#!/usr/bin/env python3
"""foldsim_site_check.py: site feasibility of a generation's memories in the dynamic region's RAM columns.

The fit gate's per-SLR totals (foldsim_sll_check.py) can pass while a memory cannot be placed: a cascade is vertical in
one RAM column, in consecutive sites the region owns, and on the V80 it does not cross a clock-region boundary. Two
failures: a32-S26X g16i1p1 (11-14 20:33, Place 30-8312: a 64-high URAM cascade in SLR0, whose pblock columns are 24 high)
and g16i1p1d (11-15 01:30, Place 30-8361: four 16-high URAM cascades in SLR0, whose 24 pblock rows are two 12-row clock
region halves, rows 60-71 and 72-83: no 16 consecutive sites inside one clock region). The check packs every cascade of
an SLR into that SLR's runs the way the legalizer must.

    foldsim_site_check.py <gen dir> --sll <sll.txt> [--xdc <impl.xdc>] [--uram-cap N] [--bram-cap N] [--json out]

Memories: the generated simulator's memory modules (<name>_<D>x<W>, `ram_style` "ultra" or "block"), counted per
instance through the hierarchy.
  URAM288: ceil(W/72) columns of ceil(D/4096) URAMs (W <= 36: two words per 72-bit row, ceil(D/8192): Vivado 2025.2
           built g16i1p1d's 262,144 x 32 local memory as 32 URAM288, e8h's 131,072 x 32 as 16), each ONE cascade (Vivado's default without
           -max_uram_cascade_height); --uram-cap N: the synthesis cap (FOLDSIM_URAM_CASCADE_MAX), cascades of at most N;
           the memory's own cascade_height attribute (the generator's) wins over both.
  RAMB36:  the aspect with the fewest primitives (32Kx1 .. 512x72), its depth ceil(D/d) cascaded in one column (Vivado
           cascades deep block RAM by default, -max_bram_cascade_height -1); --bram-cap N likewise (the flow's
           FOLDSIM_BRAM_CASCADE_MAX). A model of synthesis' choice: the depth is an upper bound when Vivado muxes.
Where a memory sits: its parent module -- a class copy (<class>[_copyK]_static) in the SLR its SLR group is pinned to
(the fit gate's sll.txt 'slrG -> SLRx (named): <classes>' lines), a group module (foldsim_sim_slrG) in that group's
SLR, the level (foldsim_sim) in the level's SLR; a memory with no known SLR is checked against the shortest run.
Runs: the pblock's RAMB36 / URAM288 sites in the shell's impl.xdc (resize_pblock pblock_slash -add/-remove {...}) less
the shell's reserved sites (set_property PROHIBIT on RAM sites, and --reserve ranges), cut into runs of consecutive rows
of one column inside one SLR (--slr-rows) and one clock region (--cr-rows: the rows where a clock region begins; the
V80's are 24 high, the SLR0 top one 12; --cr-cross any: cascades may cross them).
Packing: per SLR and kind, the cascades best-fit decreasing into the runs (a cascade takes consecutive free sites of
one run); then the SLR's other primitives (the fit gate's 'SLRx as placed' URAM / RAMB36 less the memories counted
here, in the fit gate's own count: kernel, shell controllers) as single sites into what is left.
Exit 0 OK, 1 REFUSE (a cascade with no run of enough free sites, or the SLR's sites exhausted), 2 error. One verdict
line last: 'site check: OK|REFUSE: ...'."""
import argparse, collections, json, os, re, sys

ASPECTS = [(32768, 1), (16384, 2), (8192, 4), (4096, 9), (2048, 18), (1024, 36), (512, 72)]
KIND = {'ultra': 'URAM288', 'block': 'RAMB36'}
# xcv80 (Vivado 2025.2, a32-S26X-F26 b7x/sites.txt): URAM288 / RAMB36 rows per SLR 0-83, 84-155, 156-227; clock regions
# begin at these rows in every RAM column (SLR0's top clock region, rows 72-83, is half height).
V80_SLR_ROWS = '84,156'
V80_CR_ROWS = '24,48,72,84,108,132,156,180,204'


def ceil(a, b):
    return -(-a // b)


def site_ranges(text, kind):
    """[(x0, y0, x1, y1)] of the `kind` site ranges / single sites in a constraint string."""
    out = []
    for r in re.finditer(kind + r'_X(\d+)Y(\d+)(?::' + kind + r'_X(\d+)Y(\d+))?\b', text):
        x0, y0 = int(r.group(1)), int(r.group(2))
        x1, y1 = (int(r.group(3)), int(r.group(4))) if r.group(3) else (x0, y0)
        out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return out


def region_sites(xdc, kind, reserve=(), device=None):
    """{column x: set of rows} of the pblock's `kind` sites (its site ranges and, with a device, its clock regions,
    kept to the sites the device has), less the shell's PROHIBIT sites and `reserve` ranges. Returns (sites, number of
    reserved sites removed)."""
    text = open(xdc).read()
    dev = (device or {}).get('kinds', {}).get(kind)
    crs = (device or {}).get('cr_rows', {})
    cols = collections.defaultdict(set)
    for m in re.finditer(r'resize_pblock\s+\[get_pblocks\s+pblock_slash\]\s+-(add|remove)\s+\{([^}]*)\}', text):
        add = {}
        for x0, y0, x1, y1 in site_ranges(m.group(2), kind):
            for x in range(x0, x1 + 1):
                add.setdefault(x, set()).update(range(y0, y1 + 1))
        if dev:
            for a0, b0, a1, b1 in site_ranges(m.group(2), 'CLOCKREGION'):
                for x, c in dev.items():
                    if a0 <= c['cr_x'] <= a1:
                        for j in range(b0, b1 + 1):
                            if str(j) in crs:
                                add.setdefault(int(x), set()).update(range(crs[str(j)][0], crs[str(j)][1] + 1))
        for x, ys in add.items():
            cols[x] = cols[x] | ys if m.group(1) == 'add' else cols[x] - ys
    if dev:
        for x in list(cols):
            have = set(y for a, b in dev.get(str(x), {}).get('rows', []) for y in range(a, b + 1))
            cols[x] &= have
    gone = 0
    held = [rng for m in re.finditer(r'set_property\s+PROHIBIT\s+(?:1|true|TRUE)\s+\[get_sites\s+([^\]]*)\]', text)
            for rng in site_ranges(m.group(1), kind)]
    for x0, y0, x1, y1 in held + [r for t in reserve for r in site_ranges(t, kind)]:
        for x in range(x0, x1 + 1):
            ys = set(range(y0, y1 + 1)) & cols.get(x, set())
            gone += len(ys)
            if x in cols:
                cols[x] -= ys
    return cols, gone


def runs(cols, slr_cuts, cr_cuts):
    """{SLR name: [(column, first row, length)]}: consecutive rows of a column, cut at SLR and clock-region starts."""
    names = [f'SLR{i}' for i in range(len(slr_cuts) + 1)]
    out = {n: [] for n in names}
    for x, ys in sorted(cols.items()):
        start = prev = None
        for y in sorted(ys) + [None]:
            if y is not None and prev is not None and y == prev + 1 and y not in slr_cuts and y not in cr_cuts:
                prev = y
                continue
            if start is not None:
                out[names[sum(start >= c for c in slr_cuts)]].append((x, start, prev - start + 1))
            start = prev = y
    return out


def bram_shape(d, w):
    """(cascade height, columns) of a D x W block RAM, the aspect of fewest RAMB36."""
    n, h, c = min((ceil(d, dd) * ceil(w, ww), ceil(d, dd), ceil(w, ww)) for dd, ww in ASPECTS)
    return h, c


def chains(h, cap):
    """The cascade heights of one column of h primitives under a synthesis cap (0: one cascade)."""
    if not cap or h <= cap:
        return [h]
    return [cap] * (h // cap) + ([h % cap] if h % cap else [])


def pack(segs, items):
    """Best-fit decreasing of cascades (height, label) into runs [(column, row, length)]. Returns (free per run,
    [unplaced (height, label)])."""
    free = [s[2] for s in segs]
    bad = []
    for h, lab in sorted(items, key=lambda t: -t[0]):
        fit = [i for i, f in enumerate(free) if f >= h]
        if not fit:
            bad.append((h, lab))
            continue
        i = min(fit, key=lambda i: (free[i], segs[i][0], segs[i][1]))
        free[i] -= h
    return free, bad


def hist(vals):
    c = collections.Counter(vals)
    return ' + '.join(f'{n} x {h}' for h, n in sorted(c.items(), key=lambda t: -t[0])) or 'none'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('gen')
    ap.add_argument('--sll', required=True, help="the fit gate's sll.txt (the SLR groups' SLRs, the SLRs' RAM as placed)")
    ap.add_argument('--xdc', default=os.environ.get('FOLDSIM_SHELL_XDC', os.path.expanduser(
        '~/dev/fame5_work/slash_shell_1113/slash/linker/slashkit/resources/base/compute/constraints/impl.xdc')))
    ap.add_argument('--device', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     'foldsim_v80_ram_sites.json'),
                    help="the part's RAM sites (foldsim_v80_ram_sites.json; 'none': every listed site exists)")
    ap.add_argument('--slr-rows', help="the rows (RAMB36 / URAM288 Y) where SLR1, SLR2 begin (default: the device's)")
    ap.add_argument('--cr-rows', help="the rows where a clock region begins (default: the device's)")
    ap.add_argument('--cr-cross', choices=('none', 'any'), default='none',
                    help='none (the V80: a cascade stays in one clock region) or any (only SLRs cut a column)')
    ap.add_argument('--reserve', action='append', default=[],
                    help='site ranges the shell or the kernel holds inside the region (e.g. URAM288_X0Y72:URAM288_X0Y75)')
    ap.add_argument('--no-other', action='store_true', help="do not pack the SLRs' other RAM primitives (sll.txt)")
    ap.add_argument('--uram-cap', type=int, default=0)
    ap.add_argument('--bram-cap', type=int, default=0)
    ap.add_argument('--json')
    a = ap.parse_args()
    sv = next((p for p in (os.path.join(a.gen, 'gen', 'foldsim_sim.sv'), os.path.join(a.gen, 'foldsim_sim.sv'))
               if os.path.exists(p)), None)
    if not sv or not os.path.exists(a.sll) or not os.path.exists(a.xdc):
        print(f'site check: ERROR: missing input (sv {sv}, sll {a.sll}, xdc {a.xdc})')
        return 2
    device = None if a.device in ('none', '') else json.load(open(a.device))
    slr_rows = a.slr_rows or (','.join(map(str, device['slr_rows'])) if device else V80_SLR_ROWS)
    cr_rows = a.cr_rows or (','.join(str(v[0]) for k, v in sorted(device['cr_rows'].items(), key=lambda t: int(t[0]))
                                     [1:]) if device else V80_CR_ROWS)
    slr_cuts = [int(x) for x in slr_rows.split(',') if x]
    cr_cuts = set() if a.cr_cross == 'any' else {int(x) for x in cr_rows.split(',') if x}
    seg, reserved = {}, {}
    for k, prim in KIND.items():
        cols, reserved[k] = region_sites(a.xdc, prim, a.reserve, device)
        seg[k] = runs(cols, slr_cuts, cr_cuts)
    sll = open(a.sll).read()
    text = open(sv).read()
    mods = {m.group(1): m.group(2) for m in re.finditer(r'^module (\w+)\b(.*?)^endmodule', text, re.S | re.M)}
    grp, gslr = {}, {}
    for m in re.finditer(r'slr(\d+) -> (SLR\d) \(named\): [^:]*: (.*)', sll):
        gslr[int(m.group(1))] = m.group(2)
        for c in m.group(3).split():
            grp[c] = m.group(2)
    m = re.search(r'level \(in (SLR\d)', sll)
    level = m.group(1) if m else None
    placed = {}
    for m in re.finditer(r'(SLR\d) as placed: .*?uram ([\d,.]+) .*?bram ([\d,.]+)', sll):
        placed[m.group(1)] = {'ultra': float(m.group(2).replace(',', '')), 'block': float(m.group(3).replace(',', ''))}
    # Each module's instantiating modules with instance counts (instance lines `<module> [#(...)] <name> (`).
    users = collections.defaultdict(collections.Counter)
    inst = re.compile(r'^\s*(\w+)\s+(?:#\s*\(.*?\)\s*)?\w+\s*\(', re.M)
    for p, body in mods.items():
        for i in inst.finditer(body):
            if i.group(1) in mods and i.group(1) != p:
                users[i.group(1)][p] += 1

    def slr_of(parent):
        g = re.fullmatch(r'foldsim_sim_slr(\d+)', parent)
        if g:
            return gslr.get(int(g.group(1)))
        if parent == 'foldsim_sim':
            return level
        cls = re.sub(r'(_copy\d+)?_static$', '', parent)
        return grp.get(re.sub(r'_static$', '', parent)) or grp.get(cls)

    def home(p):
        return bool(slr_of(p)) or p == 'foldsim_sim' or bool(re.fullmatch(r'foldsim_sim_slr\d+', p)) \
            or p.endswith('_static')

    memo_n, memo_p = {}, {}

    def count(p, seen=()):
        """Instances of module p in the whole design (1 for a top)."""
        if p not in memo_n:
            memo_n[p] = sum(k * count(q, seen + (p,)) for q, k in users.get(p, {}).items() if q not in seen) or 1
        return memo_n[p]

    def placements(m, seen=()):
        """Counter {SLR or None: instances} of a memory module, through its parents up to a class, group or the level."""
        if m in memo_p:
            return memo_p[m]
        out = collections.Counter()
        for p, k in users.get(m, {}).items():
            if home(p):
                out[slr_of(p)] += k * count(p)
            elif p not in seen:
                for s, n in placements(p, seen + (m,)).items():
                    out[s] += k * n
        memo_p[m] = out or collections.Counter({None: 1})
        return memo_p[m]

    rows, items, fitc = [], collections.defaultdict(list), collections.Counter()
    for n, body in mods.items():
        mm = re.search(r'_(\d+)x(\d+)$', n)
        st = re.search(r'ram_style\s*=\s*"(\w+)"', body)
        if not mm or not st or st.group(1) not in KIND:
            continue
        k = st.group(1)
        d, w = int(mm.group(1)), int(mm.group(2))
        if k == 'ultra':
            pk = 2 if w <= 36 else 1  # two narrow words per 72-bit row (Vivado 2025.2: 262,144 x 32 -> 32 URAM288)
            h, c, cap = ceil(d, 4096 * pk), ceil(w * pk, 72), a.uram_cap
        else:
            (h, c), cap = bram_shape(d, w), a.bram_cap
        attr = re.search(r'cascade_height\s*=\s*(\d+)', body)
        if attr and int(attr.group(1)) > 0:  # the generator's per-memory cap (the RTL attribute wins over the option)
            cap = int(attr.group(1))
        ch = chains(h, cap)
        for s, ninst in sorted(placements(n).items(), key=lambda t: str(t[0])):
            rows.append(dict(module=n, kind=k, depth=d, width=w, cascade=h, chains=ch, columns=c, instances=ninst,
                             slr=s or '-', ok=True))
            for _ in range(c * ninst):
                items[(k, s)].extend((x, rows[-1]) for x in ch)
            # the fit gate's count of this memory (its 'as placed' totals price URAM unpacked): what to take off them
            fitc[(k, s)] += ninst * (ceil(d, 4096) * ceil(w, 72) if k == 'ultra' else h * c)
    # Pack per SLR and kind; memories of no known SLR: each cascade against the shortest SLR's tallest run.
    packs, bad = [], 0
    for k in KIND:
        for s in sorted(seg[k]):
            segs = seg[k][s]
            its = items.get((k, s), [])
            if not segs and not its:
                continue
            free, unplaced = pack(segs, its)
            for h, r in unplaced:
                r['ok'] = False
            mem = sum(h for h, _ in its)
            other = 0 if a.no_other else max(0, ceil(int(2 * placed.get(s, {}).get(k, 0)), 2) - fitc[(k, s)])
            left = sum(free) - other
            ok = not unplaced and left >= 0
            bad += len(unplaced) + (left < 0)
            packs.append(dict(kind=k, slr=s, runs=len(segs), run_lengths=hist([g[2] for g in segs]),
                              tallest=max((g[2] for g in segs), default=0), sites=sum(g[2] for g in segs),
                              cascades=hist([h for h, _ in its]), memory_sites=mem, other_sites=other,
                              free_after=left, unplaced=[(h, r['module']) for h, r in unplaced], ok=ok))
        nos = items.get((k, None), [])
        if nos:
            room = min((max((g[2] for g in v), default=0) for v in seg[k].values() if v), default=0)
            for h, r in nos:
                if h > room:
                    r['ok'] = False
                    bad += 1
    rows.sort(key=lambda r: (r['ok'], -max(r['chains'])))
    print(f"runs per SLR (columns cut at {'clock regions' if cr_cuts else 'SLRs only'}), tallest: "
          + ', '.join(f"{KIND[k]} " + str({s: max((g[2] for g in v), default=0) for s, v in seg[k].items()}) for k in KIND)
          + (f'; caps URAM {a.uram_cap}' if a.uram_cap else '') + (f', RAMB36 {a.bram_cap}' if a.bram_cap else '')
          + (f"; reserved URAM {reserved['ultra']}, RAMB36 {reserved['block']}" if any(reserved.values()) else ''))
    for r in rows[:16]:
        print(f"{'REFUSE' if not r['ok'] else 'ok    '} {r['kind']:5s} {r['module']}: {r['depth']}x{r['width']}: cascade "
              f"{r['cascade']}{'' if r['chains'] == [r['cascade']] else ' -> ' + hist(r['chains'])} x {r['columns']} col(s)"
              f" x {r['instances']} in {r['slr']}")
    for p in packs:
        print(f"{'REFUSE' if not p['ok'] else 'ok    '} pack {p['slr']} {KIND[p['kind']]}: {p['runs']} runs "
              f"({p['run_lengths']}; {p['sites']} sites), cascades {p['cascades']} ({p['memory_sites']} sites), other "
              f"{p['other_sites']}, free after {p['free_after']}"
              + (f"; no run of free sites for: " + ', '.join(f'{h} ({m})' for h, m in p['unplaced'][:4])
                 if p['unplaced'] else '') + ('; sites exhausted' if p['free_after'] < 0 else ''))
    if a.json:
        json.dump(dict(runs={k: v for k, v in seg.items()}, rows=rows, packs=packs, refused=bad), open(a.json, 'w'),
                  indent=1, default=str)
    worst = {k: max((max(r['chains']) for r in rows if r['kind'] == k), default=0) for k in KIND}
    print(f"site check: {'REFUSE' if bad else 'OK'}: {bad} memory cascade(s) or SLR(s) without legal sites "
          f"({len(rows)} placements; tallest URAM cascade {worst['ultra']}, RAMB36 {worst['block']}; clock regions "
          f"{'cut' if cr_cuts else 'crossed'})")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
