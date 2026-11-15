#!/usr/bin/env python3
"""foldsim_site_check.py: site feasibility of a generation's memories in the dynamic region's RAM columns.

The fit gate's per-SLR totals (foldsim_sll_check.py) can pass while a single memory cannot be placed: a cascade is
vertical in one RAM column, so a memory deeper than the tallest column its SLR's share of the region offers has no
site (Vivado Place 30-8312: a32-S26X g16i1p1, 11-14 20:33: its 262,144 x 32 URAM local memory, a 64-high cascade, in
SLR0 whose pblock URAM columns are 24 high). G's uram_check.py (S26X/G) found it; this is the generic tool, with block RAM.

    foldsim_site_check.py <gen dir> --sll <sll.txt> [--xdc <impl.xdc>] [--uram-cap N] [--bram-cap N] [--json out]

Memories: the generated simulator's memory modules (<name>_<D>x<W>, `ram_style` "ultra" or "block").
  URAM288: ceil(W/72) columns of ceil(D/4096) URAMs, each ONE cascade (Vivado's default without
           -max_uram_cascade_height); --uram-cap N: the synthesis cap (FOLDSIM_URAM_CASCADE_MAX), a cascade of at most N.
  RAMB36:  the aspect with the fewest primitives (32Kx1 .. 512x72), its depth ceil(D/d) cascaded in one column (Vivado
           cascades deep block RAM by default, -max_bram_cascade_height -1); --bram-cap N likewise (the flow's
           FOLDSIM_BRAM_CASCADE_MAX). A model of synthesis' choice: the depth it reports is an upper bound when Vivado
           muxes instead of cascading.
Where a memory sits: its parent module -- a class copy (<class>[_copyK]_static) in the SLR its SLR group is pinned to
(the fit gate's sll.txt 'slrG -> SLRx (named): <classes>' lines), a group module (foldsim_sim_slrG) in that group's
SLR, the level (foldsim_sim) or a class in no group: the shortest column of the region (conservative).
Columns: the pblock's RAMB36 / URAM288 ranges in the shell's impl.xdc (resize_pblock pblock_slash -add {...}), each
column's longest run of consecutive rows inside one SLR (rows split at --slr-rows, the V80's 84,168: SLR0 below 84).
Exit 0 OK, 1 REFUSE (a cascade taller than every column of its SLR), 2 error. One verdict line last:
'site check: OK|REFUSE: ...'."""
import argparse, json, os, re, sys

ASPECTS = [(32768, 1), (16384, 2), (8192, 4), (4096, 9), (2048, 18), (1024, 36), (512, 72)]


def ceil(a, b):
    return -(-a // b)


def columns(xdc, kind, cuts):
    """{SLR name: tallest run of consecutive sites in one column} of the pblock's `kind` sites."""
    rows = {}
    pat = re.compile(r'resize_pblock\s+\[get_pblocks\s+pblock_slash\]\s+-add\s+\{([^}]*)\}')
    for m in pat.finditer(open(xdc).read()):
        for rng in m.group(1).split():
            r = re.fullmatch(kind + r'_X(\d+)Y(\d+):' + kind + r'_X(\d+)Y(\d+)', rng)
            if not r:
                continue
            x0, y0, x1, y1 = map(int, r.groups())
            for x in range(min(x0, x1), max(x0, x1) + 1):
                rows.setdefault(x, set()).update(range(min(y0, y1), max(y0, y1) + 1))
    names = [f'SLR{i}' for i in range(len(cuts) + 1)]
    best = {n: 0 for n in names}
    for ys in rows.values():
        run, prev = 0, None
        for y in sorted(ys):
            slr = names[sum(y >= c for c in cuts)]
            same = prev is not None and y == prev + 1 and names[sum(prev >= c for c in cuts)] == slr
            run = run + 1 if same else 1
            best[slr] = max(best[slr], run)
            prev = y
    return best


def bram_shape(d, w):
    """(cascade height, columns) of a D x W block RAM, the aspect of fewest RAMB36."""
    n, h, c = min((ceil(d, dd) * ceil(w, ww), ceil(d, dd), ceil(w, ww)) for dd, ww in ASPECTS)
    return h, c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('gen')
    ap.add_argument('--sll', required=True, help="the fit gate's sll.txt (the SLR groups' SLRs)")
    ap.add_argument('--xdc', default=os.environ.get('FOLDSIM_SHELL_XDC', os.path.expanduser(
        '~/dev/fame5_work/slash_shell_1113/slash/linker/slashkit/resources/base/compute/constraints/impl.xdc')))
    ap.add_argument('--slr-rows', default='84,168', help='the rows (RAMB36 / URAM288 Y) where SLR1, SLR2 begin')
    ap.add_argument('--uram-cap', type=int, default=0)
    ap.add_argument('--bram-cap', type=int, default=0)
    ap.add_argument('--json')
    a = ap.parse_args()
    sv = next((p for p in (os.path.join(a.gen, 'gen', 'foldsim_sim.sv'), os.path.join(a.gen, 'foldsim_sim.sv'))
               if os.path.exists(p)), None)
    if not sv or not os.path.exists(a.sll) or not os.path.exists(a.xdc):
        print(f'site check: ERROR: missing input (sv {sv}, sll {a.sll}, xdc {a.xdc})')
        return 2
    cuts = [int(x) for x in a.slr_rows.split(',') if x]
    col = {'ultra': columns(a.xdc, 'URAM288', cuts), 'block': columns(a.xdc, 'RAMB36', cuts)}
    text = open(sv).read()
    mods = {m.group(1): m.group(2) for m in re.finditer(r'^module (\w+)\b(.*?)^endmodule', text, re.S | re.M)}
    grp, gslr, level = {}, {}, None
    for m in re.finditer(r'slr(\d+) -> (SLR\d) \(named\): [^:]*: (.*)', open(a.sll).read()):
        gslr[int(m.group(1))] = m.group(2)
        for c in m.group(3).split():
            grp[c] = m.group(2)
    m = re.search(r'level \(in (SLR\d)', open(a.sll).read())
    level = m.group(1) if m else None
    # Each module's instantiating modules (one pass: instance lines `<module> <name> (`).
    users = {}
    inst = re.compile(r'^\s*(\w+)\s+(?:#\s*\(.*?\)\s*)?\w+\s*\(', re.M)
    for p, body in mods.items():
        for i in inst.finditer(body):
            if i.group(1) in mods:
                users.setdefault(i.group(1), set()).add(p)

    def slr_of(parent):
        g = re.fullmatch(r'foldsim_sim_slr(\d+)', parent)
        if g:
            return gslr.get(int(g.group(1)))
        if parent == 'foldsim_sim':
            return level
        cls = re.sub(r'(_copy\d+)?_static$', '', parent)
        return grp.get(re.sub(r'_static$', '', parent)) or grp.get(cls)

    def homes(m, seen=()):
        """The SLRs of a memory module through its instantiating modules (up to a class, group or the level)."""
        out = set()
        for p in users.get(m, ()):
            s = slr_of(p)
            if s or p == 'foldsim_sim' or re.fullmatch(r'foldsim_sim_slr\d+', p) or p.endswith('_static'):
                out.add(s)
            elif p not in seen:
                out |= homes(p, seen + (m,))
        return out or {None}

    rows, bad = [], 0
    for n, body in mods.items():
        mm = re.search(r'_(\d+)x(\d+)$', n)
        st = re.search(r'ram_style\s*=\s*"(\w+)"', body)
        if not mm or not st or st.group(1) not in ('ultra', 'block'):
            continue
        d, w = int(mm.group(1)), int(mm.group(2))
        if st.group(1) == 'ultra':
            h, c, cap = ceil(d, 4096), ceil(w, 72), a.uram_cap
        else:
            (h, c), cap = bram_shape(d, w), a.bram_cap
        he = min(h, cap) if cap else h
        for s in sorted(homes(n), key=str):
            room = col[st.group(1)].get(s) if s else min(v for v in col[st.group(1)].values() if v)
            ok = he <= (room or 0)
            bad += not ok
            rows.append(dict(module=n, kind=st.group(1), depth=d, width=w, cascade=h, cascade_capped=he, columns=c,
                             slr=s or '-', column_height=room, ok=ok))
    rows.sort(key=lambda r: (r['ok'], -r['cascade_capped']))
    print(f"columns (tallest run per SLR): URAM288 {col['ultra']}, RAMB36 {col['block']}"
          + (f'; caps URAM {a.uram_cap}' if a.uram_cap else '') + (f', RAMB36 {a.bram_cap}' if a.bram_cap else ''))
    for r in rows[:16]:
        print(f"{'REFUSE' if not r['ok'] else 'ok    '} {r['kind']:5s} {r['module']}: {r['depth']}x{r['width']}: cascade "
              f"{r['cascade']}{'' if r['cascade_capped'] == r['cascade'] else ' -> ' + str(r['cascade_capped'])} x "
              f"{r['columns']} col(s) in {r['slr']} (tallest column {r['column_height']})")
    if a.json:
        json.dump(dict(columns=col, rows=rows, refused=bad), open(a.json, 'w'), indent=1)
    worst = {k: max((r['cascade_capped'] for r in rows if r['kind'] == k), default=0) for k in ('ultra', 'block')}
    print(f"site check: {'REFUSE' if bad else 'OK'}: {bad} memory cascade(s) taller than their SLR's tallest column "
          f"({len(rows)} placements; tallest URAM cascade {worst['ultra']}, RAMB36 {worst['block']})")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
