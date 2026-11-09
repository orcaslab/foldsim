#!/usr/bin/env python3
"""foldsim_sll_check.py <gen dir> [--log <driver log>] [--slr-region v80|<json>] [--copy-cap SLR]
                       [--resources netlist|schedule] [--models <models.mlir> [--fold-options <options>]]
                       [--foldsim-opt <bin>] [--out <dir>] [--json <file>]

Whether an existing netlist places over the region's SLRs, and the SLLs each SLR boundary takes, without
regenerating it.

Each class copy's resources are counted on the generated netlist itself (`--resources netlist`, the default; the
check REFUSES without the generation's foldsim_sim.mlir, since the schedule's estimates miss stage registers, read
buffers and the simulator level and said "fits" where placement failed): its LUTs, LUTRAM LUTs, flip-flops,
URAM288 and RAMB36 (foldsim_cost over the copy's module `<copy>_static` and the stores and page RAMs below it; a
memory's RAM by its ram_style, as Vivado builds it: URAM288 by 72-bit columns of 4,096 rows, RAMB36 by
foldsim_cost.mem_cost), LUTs and flops scaled to synthesized ones (the region's netlist_lut_scale /
netlist_ff_scale) and then to placed ones (placed_per_synthesized: the kernel beside the simulator and the placer's
replication), and the simulator level's link memories and registers counted with the copy whose output they hold
(placed at the producer). `--resources schedule` gives an estimate from the class estimates of the generation's
models and its schedule (foldsim_degree.copy_sites), never a verdict.

The generation's fold is run again schedule-only (the `+ ... foldsim-opt <flat.mlir> ... --pass-pipeline=...` line
of the generation log, <gen dir>.gen.log by default; or `--models` and `--fold-options`): its II must be the
netlist's (an older or newer compiler checks another netlist: REFUSE). Then, in order:
  1. every copy fits whole, in each resource, the `--copy-cap` SLR's capacity times the region's margin (SLR a
     name, `smallest`, `largest`, `any` (the default: some SLR) or `none`; r16c, 11-07, a slice copy of 517 URAM
     against SLRs of 496 / 511 / 204 and 713k placed LUTs, failed partitioning at 23,987 SLLs);
  2. the design within the region at the placer's fill (the region's placement model: `fill` of each logic
     resource, LUTs, LUTRAM and flops; memories at their whole counts, their primitives placed apart);
  3. an assignment of whole copies to the SLRs, each within its fill (fold-static sll-estimate over the copies'
     LUT sites and, as further resources, their LUTRAM, flops, URAM and RAMB36); where there is none, the largest
     copies by logic (at most cut_copies) are left out, the others assigned within the SLRs' balanced share (the
     placer balances the SLRs' fill), and each left-out copy fills its SLR's room and levels the next SLRs' fill:
     the placer cuts it;
  4. each boundary's SLLs as placed (the placement model, fit on placed builds): the link wires between the two
     sides (sll_per_link_bit), each cut copy's own cut by Rent's rule over its smaller side (rent_t x placed
     LUTs^rent_p), and the simulator level and kernel (other_slls, sll_per_vmem_bit x the netlist's virtual memory
     port bits); past the boundary's SLLs a refusal, past the budget (sll_per_boundary x sll_margin) or with a copy
     cut a warning.

Exits 0 (OK, or WARN), 1 if a check refuses, 2 on error. Report in <out>/sll.txt (the estimate) and on stdout;
`--json` writes the copies' resources, the wires between them, the assignment, each copy's shares of the SLRs, the
copies that fit no SLR whole (`over`), the cut ones (`cuts`) and each SLR's capacity (`caps`, logic times `margin`).
--slr-policy guide (the default since 11-08 evening): a copy past every SLR (with the default --copy-cap any) is not
a refusal but a cut copy (step 3), its SLLs projected: a straddle is a WARN or OK with the copy in `cuts`, a guide
(the driver and fit_gate.sh print it with the suggested assignment), never a refusal; the region's fill, the
memories and the SLLs past a boundary still refuse. --slr-policy strict: the 11-08 hard rule (a copy past every SLR
refuses; the driver's --slr-policy strict and fit_gate.sh FIT_SLR_POLICY=strict fail on any cut).
"""
import argparse, json, math, os, re, shlex, subprocess, sys
from collections import defaultdict
from itertools import takewhile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foldsim_cost  # noqa: E402
import foldsim_degree  # noqa: E402

RESOURCES = ('lut', 'lutram', 'ff', 'uram', 'bram')
# The logic: what a copy placed whole keeps in one SLR (its memory primitives, on narrow ports, go apart).
LOGIC = ('lut', 'lutram', 'ff')
# A Versal CLB: 8 LUT6 and 16 flip-flops.
FFS_PER_LUT = 2
URAM_DEPTH, URAM_WIDTH = 4096, 72


def split_top(pipeline):
    """The top-level passes of builtin.module(...), braces and parentheses respected."""
    body = pipeline[pipeline.index('(') + 1:pipeline.rindex(')')]
    parts, depth, cur = [], 0, ''
    for ch in body:
        if ch in '({':
            depth += 1
        elif ch in ')}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return parts


def mem_ram(rec):
    """(URAM288, RAMB36) of a foldsim_cost memory record as Vivado builds it: a
    ram_style ultra memory in 72-bit columns of 4,096 rows (Vivado puts a
    small one in a whole URAM too), else foldsim_cost.mem_cost's block RAM."""
    _, d, w, r, wr, rl, st, *_ = rec
    if st == 'ultra':
        # A read port beyond the first is a copy (foldsim_storage.kind_cost).
        copies = max(1, r if wr else math.ceil(r / 2))
        return math.ceil(w / URAM_WIDTH) * math.ceil(d / URAM_DEPTH) * copies, 0.0
    return 0.0, foldsim_cost.mem_cost(d, w, r, rl, st)[1]


def add_memory(acc, rec):
    """A memory's URAM288, RAMB36 and distributed RAM LUTs (in `lut` and in
    `lutram`: a LUTRAM LUT takes a SLICEM's, of which an SLR has a share)."""
    u, b = mem_ram(rec)
    _, d, w, r, wr, rl, st, *_ = rec
    lr = foldsim_cost.mem_cost(d, w, r, rl, st)[0] if st != 'ultra' else 0.0
    acc['uram'] += u
    acc['bram'] += b
    acc['lut'] += lr
    acc['lutram'] += lr
    # The bits its ports carry (a feature of the copy's own wiring).
    acc['port_bits'] = acc.get('port_bits', 0) + w * (r + wr)


def new_acc():
    return dict(lut=0.0, lutram=0.0, ff=0.0, uram=0.0, bram=0.0)


def subtree(mods, name, stop, acc):
    """A module's LUTs (logic, black boxes, distributed RAM), flops and RAM with
    its children, but not the `stop` modules (other copies)."""
    m = mods[name]
    acc['lut'] += sum(m.lut.values())
    acc['ff'] += m.ff
    for rec in m.mems:
        add_memory(acc, rec)
    for child, _ in m.insts:
        if child in stop:
            continue
        base = child.split('__')[0]
        if child in foldsim_cost.BLACKBOX or base in foldsim_cost.BLACKBOX:
            l, f, _ = foldsim_cost.BLACKBOX.get(child, foldsim_cost.BLACKBOX.get(base))
            acc['lut'] += l
            acc['ff'] += f
        elif child in mods:
            subtree(mods, child, stop, acc)
    return acc


def level_registers(sim, modules):
    """(module, name, bits) of each register of the simulator's level modules
    (@foldsim_sim and its SLR group modules)."""
    cur = None
    for line in open(sim):
        t = line.strip()
        if t.startswith('hw.module'):
            m = re.match(r'hw\.module(?:\s+private)?\s+@([\w$.]+)\(', t)
            cur = m.group(1) if m and m.group(1) in modules else None
            continue
        if cur and ' = seq.' in t:
            lhs, rhs = t.split(' = ', 1)
            if rhs.split()[0] in foldsim_cost.REGS:
                yield cur, lhs.strip().lstrip('%'), foldsim_cost.width(foldsim_cost.result_type(rhs))


def synthesized(path):
    """{copy instance: {lut, lutram, ff}} of a Vivado hierarchical utilization
    report (report_utilization -hierarchical) of a synthesis of the netlist:
    each `<copy>_copy` instance's Total LUTs, LUTRAMs (+ SRLs) and FFs."""
    out = {}
    for line in open(path):
        cells = [c.strip() for c in line.split('|')]
        if len(cells) < 12 or cells[1].startswith('('):
            continue
        off = 3 if re.match(r'\d', cells[3]) else 5
        nums = [int(re.match(r'(\d+)', c).group(1)) if re.match(r'\d+', c) else None for c in cells[off:off + 5]]
        if nums[0] is None or not cells[1].endswith('_copy'):
            continue
        out.setdefault(cells[1][:-len('_copy')], dict(lut=nums[0], lutram=nums[2] + (nums[3] or 0), ff=nums[4]))
    return out


def netlist_resources(sim, copies, region, synthesis=None):
    """{copy: {lut, lutram, ff, uram, bram}} of a generated simulator (`sim`,
    its foldsim_sim.mlir) for the schedule's `copies`, scaled to placed ones:
    the netlist count (foldsim_cost) times the region's netlist_*_scale
    (synthesized per counted) and placed_per_synthesized; the links read into
    a copy charged to it; the simulator level's rest and the region's kernel
    in '__top__'. With `synthesis` (a hierarchical utilization report of a
    synthesis of this netlist), each copy's own LUTs, LUTRAM and flops are
    the synthesized ones, and the copies it lacks take the measured ones'
    synthesized / counted ratio (a calibration per design)."""
    mods, _ = foldsim_cost.parse(sim)
    PARSED['mods'] = mods
    statics = {f'{c}_static' for c in copies if f'{c}_static' in mods}
    own, link = {}, {c: new_acc() for c in copies}
    for c in copies:
        acc = new_acc()
        if f'{c}_static' in mods:
            subtree(mods, f'{c}_static', statics - {f'{c}_static'}, acc)
            acc['io_bits'] = mods[f'{c}_static'].ins + mods[f'{c}_static'].outs
        own[c] = acc
    rest = new_acc()
    # The simulator's level: @foldsim_sim and the SLR group modules under it.
    level = [m for m in ['foldsim_sim'] + [ch for ch, _ in mods['foldsim_sim'].insts
                                            if ch in mods and ch not in statics]] if 'foldsim_sim' in mods else []
    names = sorted(copies, key=len, reverse=True)

    def owner(n):
        # A link memory or register `<producer>_<port>_to_<reader>_<port>` is
        # placed with its reader (the compiler puts the links read into an SLR
        # group in that group: b4e's opt checkpoint, 11-07, its group 0 670,951
        # LUTs against 498,646 counted with the links at their producers);
        # without a reader in its name, with its producer.
        return next((c for c in names if f'_to_{c}_' in n or n.endswith(f'_to_{c}')), None) or \
            next((c for c in names if n.startswith(c + '_')), None)
    for lv in level:
        m = mods[lv]
        # The level's array reads are its link registers' reads by instance,
        # which synthesis builds in the registers' read selects (r16c: 2.23M
        # LUTs counted as array_get, the whole simulator level 129k LUTs
        # synthesized besides its copies): its other logic only.
        rest['lut'] += sum(v for k, v in m.lut.items() if k != 'array_get')
        for rec in m.mems:
            o = owner(rec[0])
            add_memory(link[o] if o else rest, rec)
        for child, _ in m.insts:
            if child not in statics and child not in level and child in mods:
                subtree(mods, child, statics, rest)
    for _, reg, w in level_registers(sim, set(level)):
        o = owner(reg)
        (link[o] if o else rest)['ff'] += w
    ls, fs = region.get('netlist_lut_scale', 1.0), region.get('netlist_ff_scale', 1.0)
    lr = region.get('netlist_lutram_scale', ls)
    pl = region.get('placed_per_synthesized', {})
    syn = synthesized(synthesis) if synthesis else {}
    measured = [c for c in copies if c in syn]
    ratio = {r: (sum(syn[c][r] for c in measured) / max(1e-9, sum(own[c][r] for c in measured))) if measured else None
             for r in ('lut', 'lutram', 'ff')}

    def scale(v, sy=None):
        for r, f in (('lut', ls), ('lutram', lr), ('ff', fs)):
            v[r] = (sy[r] if sy else v[r] * (ratio[r] if syn and ratio[r] else f)) * pl.get(r, 1.0)
    out = {}
    for c in copies:
        scale(own[c], syn.get(c))
        scale(link[c])
        out[c] = {r: own[c][r] + link[c][r] for r in RESOURCES}
        out[c]['io_bits'] = own[c].get('io_bits', 0)
        out[c]['port_bits'] = own[c].get('port_bits', 0) + link[c].get('port_bits', 0)
    scale(rest)
    # The kernel beside the simulator (its memory endpoints, virtual memory
    # controllers, host interface): not in the netlist, placed with it.
    k = region.get('kernel', {})
    rest['lut'] += k.get('lut', 0.0)
    rest['ff'] += k.get('ff', 0.0)
    out['__top__'] = rest
    return out


PARSED = {}


def split_copies(sim, gdir, res, names):
    """Copies split over SLRs by foldsim-slr-split (`<copy>_static_p<k>` parts, instances
    `fs_split_<copy>_static_p<k>`, each assigned to an SLR in <gen>/foldsim_floorplan.tcl): {copy: {'slr': [SLR index
    per part], 'frac': {resource: [share per part]}, 'cross': [(part, part, bits)]}}. A part's share of each resource
    is its subtree's share of the copy's netlist count (the copy's resources as counted: its links and scales kept);
    the crossing bits are the part instances' inputs from another part (each crossing a register to a register, one
    SLL a bit per boundary between their SLRs)."""
    mods = PARSED.get('mods') or foldsim_cost.parse(sim)[0]
    fp = os.path.join(gdir, 'foldsim_floorplan.tcl')
    where = {}
    if os.path.exists(fp):
        for m in re.finditer(r'get_cells -quiet -hierarchical (fs_split_\S+)\]\s*\n.*?USER_SLR_ASSIGNMENT (\S+) ',
                             open(fp).read()):
            where[m.group(1)] = m.group(2)
    out = {}
    for c in res:
        parts = sorted((int(m.group(1)), n) for n in mods for m in [re.fullmatch(re.escape(c) + r'_static_p(\d+)', n)]
                       if m)
        if not parts:
            continue
        slr = []
        for k, n in parts:
            s = where.get(f'fs_split_{n}')
            if s not in names:
                sys.exit(f'{c} is split but its part {n} has no SLR in {fp}')
            slr.append(names.index(s))
        accs = [subtree(mods, n, set(), new_acc()) for _, n in parts]
        frac = {r: [x[r] / max(1e-9, sum(y[r] for y in accs)) if sum(y[r] for y in accs) else 1.0 / len(accs)
                    for x in accs] for r in RESOURCES}
        cross = defaultdict(int)
        text = ''.join(line for line in open(sim) if 'hw.instance "fs_split_' + c + '_static_p' in line)
        for line in text.splitlines():
            k = int(re.search(r'hw\.instance "fs_split_' + re.escape(c) + r'_static_p(\d+)"', line).group(1))
            for j, t in re.findall(r'%fs_split_' + re.escape(c) + r'_static_p(\d+)\.[\w.$-]+: ([^,)]+)', line):
                cross[(int(j), k)] += foldsim_cost.width(t)
        # Each part's share of the copy's inputs and outputs (the links it reads and writes).
        io_in, io_out = [0] * len(parts), [0] * len(parts)
        for line in text.splitlines():
            k = int(re.search(r'hw\.instance "fs_split_' + re.escape(c) + r'_static_p(\d+)"', line).group(1))
            for name, t in re.findall(r'(%[\w.$-]+): ([^,)]+)', line.split(') -> ')[0]):
                if not name.startswith('%fs_split_') and not t.startswith('!seq.clock'):
                    io_in[k] += foldsim_cost.width(t)
        body = re.search(r'hw\.module @' + re.escape(c) + r'_static\(.*?\n(.*?)\n\s*hw\.output ([^\n]*)', open(sim).read(), re.S)
        if body:
            for k2 in re.findall(r'%fs_split_' + re.escape(c) + r'_static_p(\d+)\.', body.group(2).split(' : ')[0]):
                io_out[int(k2)] += 1
        norm = lambda v: [x / sum(v) if sum(v) else 1.0 / len(v) for x in v]
        frac['in'], frac['out'] = norm(io_in), norm(io_out)
        out[c] = dict(slr=slr, frac=frac, cross=[(i, j, b) for (i, j), b in cross.items() if i != j],
                      parts=[n for _, n in parts])
    return out


def slr_caps(region):
    """Per SLR (in order), its capacity in each resource: LUTs and flops times
    the margin (logic placed whole must route), URAM and RAMB36 whole (a
    copy's RAM primitives, each on narrow ports, are placed where they fit:
    b4e's opc copy of 1,024 RAMB36, 98 % of SLR1's and past its margin,
    placed and routed, 11-07)."""
    m = region.get('margin', 1.0)
    return [dict(name=s['name'], lut=m * s['luts'], lutram=m * s.get('lutram_luts', s['luts']),
                 ff=m * FFS_PER_LUT * s['luts'], uram=s.get('uram', 0), bram=s.get('ramb36', 0))
            for s in region['slrs']]


def fmt(r, x):
    return f'{x:,.1f}' if r in ('uram', 'bram') else f'{x:,.0f}'


def lut_equivalent(res, caps):
    """A copy's binding resource in LUTs: each resource as the share of the
    region's it takes, times the region's LUTs."""
    tot = {r: sum(c[r] for c in caps) for r in RESOURCES}
    return max(res[r] / tot[r] * tot['lut'] for r in RESOURCES if tot[r])


def schedule_ii(text):
    m = re.search(r'II = (\d+)', text)
    return int(m.group(1)) if m else None


def full_capacities(region):
    """Per SLR (in order, the chain of the region), each resource whole."""
    return [dict(name=s['name'], lut=s['luts'], lutram=s.get('lutram_luts', s['luts']), ff=FFS_PER_LUT * s['luts'],
                 uram=s.get('uram', 0), bram=s.get('ramb36', 0)) for s in region['slrs']]


def spread(top, full):
    """The simulator level's share of each SLR (spread over the region in
    proportion to each resource's capacity)."""
    tot = {r: sum(c[r] for c in full) for r in RESOURCES}
    return [{r: (top or {}).get(r, 0.0) * c[r] / tot[r] if tot[r] else 0.0 for r in RESOURCES} for c in full]


def distribute(res_l, room, home, use, full):
    """A copy's (`res_l`) share of each SLR, per resource: its `home` SLR
    up to its `room`, the rest over the SLRs next out (the nearer first),
    levelling their fill (`use` of `full`: the placer balances the SLRs) up to
    each one's room. {resource: [share per SLR]}, or None if some resource's
    room is short. A copy's memory primitives go apart where there is room,
    its logic as a whole (the `lut` shares, which cut it)."""
    n = len(room)
    out = {}
    for r in RESOURCES:
        need = res_l[r]
        if need <= 0:
            out[r] = [float(s == home) for s in range(n)]
            continue
        amt = [0.0] * n
        amt[home] = min(need, max(0.0, room[home][r]))
        left = need - amt[home]
        for d in range(1, n):
            ring = [s for s in range(n) if abs(s - home) == d and full[s][r] > 0]
            if left <= 1e-9 or not ring:
                continue
            # The fill level at which the ring takes `left` (or all its room).
            def take(level):
                return sum(min(max(0.0, room[s][r]), max(0.0, level * full[s][r] - use[s][r])) for s in ring)
            lo, hi = 0.0, 1.0 + max(use[s][r] / full[s][r] for s in ring) + max(
                max(0.0, room[s][r]) / full[s][r] for s in ring)
            if take(hi) <= left:
                level = hi
            else:
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if take(mid) < left else (lo, mid)
                level = hi
            for s in ring:
                amt[s] = min(max(0.0, room[s][r]), max(0.0, level * full[s][r] - use[s][r]))
            left = need - sum(amt)
        if left > 1e-6 * need:
            return None
        out[r] = [x / need for x in amt]
    return out


def vmem_bits(gdir):
    """The bits of the netlist's virtual memory ports (foldsim_sim.json: each
    port's read and write data, row and instance): its wiring to the kernel's
    memory controllers, which the placer puts apart from the copies."""
    path = os.path.join(gdir, 'foldsim_sim.json')
    if not os.path.exists(path):
        return 0
    return sum(2 * v['data_width'] + v['row_width'] + v['inst_width'] for v in json.load(open(path)).get('vmem', []))


def port_links(stext):
    """{(producer class, reader class): bits} of the schedule's `A -> B: n links, m bits` lines: each link's bits
    once (its producer's output port), what crosses when the two sit in different SLRs."""
    out = {}
    for line in stext.split('\n'):
        m = re.match(r'\s+(\S+) -> (\S+): (\d+) links, (\d+) bits', line)
        if m:
            out[(m.group(1), m.group(2))] = int(m.group(4))
    return out


def link_slls(shares, links, b, splits=None):
    """Port-width link bits over boundary b: each link's bits times the chance its two ends sit on either side
    (a split copy's ends by its parts' shares of the copy's outputs and inputs)."""
    def low(c, end):
        if splits and c in splits:
            sp = splits[c]
            return sum(f for f, g in zip(sp['frac']['out' if end == 'p' else 'in'], sp['slr']) if g <= b)
        return sum(shares[c]['lut'][:b + 1])
    return sum(w * (low(p, 'p') * (1 - low(c, 'c')) + low(c, 'c') * (1 - low(p, 'p')))
               for (p, c), w in links.items() if p in shares and c in shares)


PLAN_OVER = []


def plan(res, splits, links, room, model, other):
    """The global SLR plan (--plan): every whole copy and every split part a node, each in one SLR within `room`
    (per SLR and resource), minimising the worst boundary's SLLs, then their sum: the links' port widths over the
    boundaries (sll_per_port_bit), the split parts' crossings (one SLL a bit) and `other` (the simulator level and
    kernel). Greedy by size, then single moves and swaps while they improve. {node: SLR index}, [SLLs a boundary]."""
    n = len(room)
    nodes = {}
    for c, x in res.items():
        if c in splits:
            for k, part in enumerate(splits[c]['parts']):
                nodes[(c, k)] = {r: x[r] * splits[c]['frac'][r][k] for r in RESOURCES}
        else:
            nodes[(c, None)] = dict(x)
    k_port = model.get('sll_per_port_bit', model.get('sll_per_link_bit', 0.01))

    def ends(c, end):
        """[(node, weight)] a class's link end goes to."""
        if c in splits:
            fr = splits[c]['frac']['out' if end == 'p' else 'in']
            return [((c, k), f) for k, f in enumerate(fr) if f]
        return [((c, None), 1.0)]
    pairs = []
    for (p, c), w in links.items():
        if p in res and c in res:
            for a, fa in ends(p, 'p'):
                for d, fd in ends(c, 'c'):
                    pairs.append((a, d, w * fa * fd * k_port))
    for c, sp in splits.items():
        for i, j, bits in sp['cross']:
            pairs.append(((c, i), (c, j), float(bits)))

    def slls(at):
        out = []
        for b in range(n - 1):
            out.append(other + sum(w for a, d, w in pairs if a in at and d in at and
                                   min(at[a], at[d]) <= b < max(at[a], at[d])))
        return out

    def over(at):
        """How far the assignment is past the caps (each resource's excess as a share of its cap), and whether a
        split copy's parts share an SLR."""
        use = [{r: 0.0 for r in RESOURCES} for _ in range(n)]
        for v, g in at.items():
            for r in RESOURCES:
                use[g][r] += nodes[v][r]
        # Logic within each SLR's cap; memories within the region's (a copy's block RAMs, each on narrow ports, go
        # apart where there is room, as the check's placement model has them).
        x = sum(max(0.0, use[g][r] - room[g][r]) / max(1.0, room[g][r]) for g in range(n) for r in LOGIC)
        x += sum(max(0.0, sum(u[r] for u in use) - sum(rm[r] for rm in room)) / max(1.0, sum(rm[r] for rm in room))
                 for r in ('uram', 'bram'))
        # A split copy's parts each in an SLR of its own (the split is for the copy that fits no SLR with the rest).
        for c, sp in splits.items():
            gs = [at[(c, k)] for k in range(len(sp['parts'])) if (c, k) in at]
            x += len(gs) - len(set(gs))
        return x

    def score(at):
        x = slls(at)
        return (round(over(at), 9), max(x) if x else 0.0, sum(x))
    at = {}
    order = sorted(nodes, key=lambda v: -max(nodes[v][r] / max(1e-9, max(rm[r] for rm in room)) for r in RESOURCES))
    for v in order:
        best = None
        for g in range(n):
            at[v] = g
            sc = score(at)
            if best is None or sc < best[0]:
                best = (sc, g)
        at[v] = best[1]
    improved = True
    while improved:
        improved = False
        cur = score(at)
        for v in list(nodes):
            for g in range(n):
                if g == at[v]:
                    continue
                old = at[v]
                at[v] = g
                sc = score(at)
                if sc < cur:
                    cur = sc
                    improved = True
                else:
                    at[v] = old
        for v in list(nodes):
            for u in list(nodes):
                if at[u] == at[v]:
                    continue
                at[v], at[u] = at[u], at[v]
                sc = score(at)
                if sc < cur:
                    cur = sc
                    improved = True
                else:
                    at[v], at[u] = at[u], at[v]
    if over(at) > 1e-9:
        # Past the caps: the closest assignment, its overflow per SLR and resource reported.
        use = [{r: 0.0 for r in RESOURCES} for _ in range(n)]
        for v, g in at.items():
            for r in RESOURCES:
                use[g][r] += nodes[v][r]
        PLAN_OVER[:] = [(g, r, use[g][r], room[g][r]) for g in range(n) for r in LOGIC
                        if use[g][r] > room[g][r] + 1e-9]
        return None, None
    return at, slls(at)


def boundary_slls(shares, res, wires, top_share, model, vbits=0, splits=None, links=None):
    """Each boundary's SLLs: SLLs per link wire bit times the link wires
    between the copies' shares on either side; Rent's rule (t x LUTs^p) over
    each cut copy's smaller side (and over the simulator level's, with
    top_rent_t); and the simulator level and the kernel beside it (a
    constant, other_slls, and SLLs per virtual memory port bit).
    Returns [(link SLLs, cut SLLs, simulator-level and kernel SLLs)] per
    boundary."""
    n = len(top_share)
    t, p = model['rent_t'], model['rent_p']
    out = []
    for b in range(n - 1):
        low = {c: sum(x['lut'][:b + 1]) for c, x in shares.items()}
        if links is not None and 'sll_per_port_bit' in model:
            # The links' port widths (each link's bits once), calibrated (sll_per_port_bit).
            link = link_slls(shares, links, b, splits) * model['sll_per_port_bit'] / model['sll_per_link_bit']
        else:
            link = sum(w * (low[c] * (1 - low[d]) + low[d] * (1 - low[c]))
                       for (c, d), w in wires.items() if c in low and d in low)
        cut = sum(t * (min(f, 1 - f) * res[c]['lut']) ** p for c, f in low.items()
                  if 1e-9 < f < 1 - 1e-9 and c not in (splits or {}))
        # A split copy's crossings: its parts' bits between SLRs on either side of the boundary.
        for sp in (splits or {}).values():
            cut += sum(bits for i, j, bits in sp['cross'] if min(sp['slr'][i], sp['slr'][j]) <= b < max(sp['slr'][i],
                                                                                                        sp['slr'][j]))
        tl = sum(x['lut'] for x in top_share[:b + 1])
        th = sum(x['lut'] for x in top_share[b + 1:])
        # The simulator level and the kernel beside it, spread over the SLRs.
        sim = model.get('top_rent_t', 0.0) * min(tl, th) ** p + model.get('other_slls', 0.0) + \
            model.get('sll_per_vmem_bit', 0.0) * vbits
        out.append((model['sll_per_link_bit'] * link, cut, sim))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gen')
    ap.add_argument('--log')
    ap.add_argument('--slr-region', default='v80')
    ap.add_argument('--copy-cap', default='any', metavar='SLR',
                    help='where every class copy must fit whole: any (default: some SLR of the region), or the '
                         'SLR named, smallest or largest (by LUTs)')
    ap.add_argument('--resources', choices=('netlist', 'schedule'), default='netlist',
                    help='netlist (default; the check refuses without the generated foldsim_sim.mlir) or '
                         'schedule (an estimate only, never a verdict: stage registers, read buffers and the '
                         'netlist\'s own logic are not in it)')
    ap.add_argument('--foldsim-opt', default=os.path.join(HERE, 'foldsim-opt'))
    ap.add_argument('--out')
    ap.add_argument('--synthesis', help='a hierarchical utilization report (report_utilization -hierarchical) of a '
                                        'synthesis of this netlist: the copies\' synthesized sizes instead of the '
                                        'counted ones (a calibration per design)')
    ap.add_argument('--models', help='the pre-fold models (models.mlir) instead of running the logged pipeline\'s '
                                     'passes before the fold on its input again')
    ap.add_argument('--fold-options', help='the fold\'s options (foldsim-fold-static{...}) instead of the log\'s; '
                                           'needs --models')
    ap.add_argument('--plan', metavar='FLOORPLAN',
                    help='the global SLR plan: every whole copy and split part assigned to an SLR within its caps, '
                         'the worst boundary\'s SLLs least; its floorplan (parts in pblocks, copies soft) written here')
    ap.add_argument('--json', help='write the copies\' resources, wires and assignment here (calibration)')
    ap.add_argument('--register-state-entries', type=int, default=None)
    ap.add_argument('--slr-policy', choices=('guide', 'strict'), default='guide',
                    help='guide (default, 11-08): a copy past every SLR (with --copy-cap any) is cut by the placer like '
                         'a copy left out of the assignment, its SLLs projected: a straddle is a warning with the '
                         'suggested assignment, never a refusal; the design past the placement fill, the memories '
                         'past the region\'s and the SLLs past a boundary still refuse. strict: a copy past every SLR '
                         'refuses (the 11-08 hard SLR rule)')
    a = ap.parse_args()
    gen = os.path.abspath(a.gen.rstrip('/'))
    gdir = os.path.join(gen, 'gen') if os.path.isdir(os.path.join(gen, 'gen')) else gen
    log = a.log or gen + '.gen.log'
    out = os.path.abspath(a.out or gen + '.sll')
    os.makedirs(out, exist_ok=True)
    region_path = os.path.join(HERE, 'foldsim_v80_slr.json') if a.slr_region == 'v80' else a.slr_region
    region = json.load(open(region_path))
    spl = region['sites_per_lut']
    caps = slr_caps(region)
    names = [c['name'] for c in caps]
    budget = int(region.get('sll_margin', 1.0) * region.get('sll_per_boundary', 0))
    name = os.path.basename(gen)
    sim = os.path.join(gdir, 'foldsim_sim.mlir')
    mode = a.resources
    report = os.path.join(out, 'sll.txt')

    def finish(lines, verdict):
        lines.append(f'{name}: {verdict}')
        print('\n'.join(lines))
        with open(report, 'a' if os.path.exists(report) else 'w') as f:
            f.write('\n'.join(lines) + '\n')
        sys.exit(1 if verdict.startswith('REFUSE') else 0)

    if mode == 'netlist' and not os.path.exists(sim):
        if os.path.exists(report):
            os.remove(report)
        finish([], f'REFUSE: no netlist ({sim} missing): the check counts the generated netlist (stage registers, '
                    'read buffers and the simulator level are not in the schedule\'s estimates); keep '
                    'foldsim_sim.mlir in the generation, or run with --resources schedule for an estimate')

    run = lambda cmd: subprocess.run(cmd, check=True, capture_output=True, text=True)
    if a.fold_options is not None:
        # The generation's fold options and pre-fold models given (the driver's --sll-check).
        fold, models = a.fold_options, a.models
        if not models:
            sys.exit('--fold-options needs --models')
    else:
        line = next((l for l in open(log) if 'foldsim-fold-static{' in l and '--pass-pipeline=' in l), None)
        if not line:
            sys.exit(f'no fold pipeline line in {log}')
        # The trace line is unquoted (the pipeline's options hold spaces): the input is the token after
        # foldsim-opt, the pipeline everything from --pass-pipeline= to the final -o.
        flat = line.lstrip('+ ').split()[1]
        m = re.search(r'--pass-pipeline=[\'"]?(builtin\.module\(.*\))[\'"]? -o \S+\s*$', line)
        if not m:
            sys.exit(f'no --pass-pipeline=builtin.module(...) -o in the fold line of {log}')
        passes = split_top(m.group(1))
        fi = next(i for i, p in enumerate(passes) if p.startswith('foldsim-fold-static{'))
        fold = passes[fi][len('foldsim-fold-static{'):-1]
        models = a.models or os.path.join(out, 'models.mlir')
        if not a.models:
            if not os.path.exists(flat):
                for alt in (os.path.join(gdir, 'flat.mlir'), os.path.join(gdir, 'models.mlir')):
                    if os.path.exists(alt):
                        flat = alt
                        break
            if os.path.basename(flat) == 'models.mlir':
                models = flat
            else:
                run([a.foldsim_opt, flat, '--mlir-print-op-on-diagnostic=false',
                     f'--pass-pipeline=builtin.module({",".join(passes[:fi])})', '-o', models])
    # Options this check sets itself, or that would write the generation's outputs.
    fold = ' '.join(o for o in shlex.split(fold) if not re.match(
        r'(report|slr-groups|slr-capacity|slr-capacities|slr-sites|slr-assign|slr-sll-budget|sll-estimate|schedule-only)\b',
        o))
    # The class copies and their instances: the generation's own schedule (same fold options).
    stext = open(os.path.join(gdir, 'schedule.txt')).read()
    ii, classes, _ = foldsim_degree.parse_schedule(stext)
    info = foldsim_degree.parse_classes(stext)

    top_share = None
    if mode == 'netlist':
        res = netlist_resources(sim, list(classes), region, a.synthesis)
        top_share = res.pop('__top__')
    else:
        cal = json.load(open(os.path.join(HERE, 'foldsim_v80.json')))
        entries = a.register_state_entries
        if entries is None:
            m = re.search(r'register-state-entries=(\d+)', fold)
            entries = int(m.group(1)) if m else 2
        of = {c: re.sub(r'_copy\d+$', '', c) for c in classes}
        est = foldsim_degree.class_estimates(models, sorted(set(of.values())))
        res = {c: dict(new_acc(), lut=foldsim_degree.copy_sites(cal, est[of[c]], of[c], n, entries, info.get(c), ii)
                       / spl) for c, n in classes.items()}
    model = region.get('placement', {})
    full = full_capacities(region)
    fills = model.get('fill', {})
    # The simulator level's own logic and unowned memories spread over the region (each SLR keeps its share free);
    # the placer fills an SLR up to its `fill` of each resource (the region's placement model).
    top_spread = spread(top_share, full)
    fill_of = {r: fills.get(r, region.get('margin', 1.0) if r in LOGIC else 1.0) for r in RESOURCES}
    room0 = [{r: fill_of[r] * c[r] - t[r] for r in RESOURCES} for c, t in zip(full, top_spread)]
    # Copies split over SLRs (foldsim-slr-split): each part placed in its SLR (the floorplan's), not assigned; the
    # SLRs' room what the parts leave.
    splits = split_copies(sim, gdir, res, names) if mode == 'netlist' else {}
    part_of = lambda c, k: {r: res[c][r] * splits[c]['frac'][r][k] for r in RESOURCES}
    for c, sp in splits.items():
        for k, g in enumerate(sp['slr']):
            for r in RESOURCES:
                room0[g][r] -= part_of(c, k)[r]
    res_est = dict(res, **{c: new_acc() for c in splits})
    room_whole = [{r: fill_of[r] * c[r] - t[r] for r in RESOURCES} for c, t in zip(full, top_spread)]

    lines = [f'{name}: SLRs {",".join(names)}, resources from the {mode}'
             + (f' (LUT x{region.get("netlist_lut_scale", 1.0)}, FF x{region.get("netlist_ff_scale", 1.0)}, '
                f'placed per synthesized {region.get("placed_per_synthesized", {})})'
                if mode == 'netlist' else ' (AN ESTIMATE ONLY: no stage registers, read buffers or simulator level)')
             + f'; margin {region.get("margin", 1.0)}; SLL budget {budget} a '
             f'boundary ({region.get("sll_per_boundary")} x {region.get("sll_margin")})']
    for c in caps:
        lines.append(f'  {c["name"]} capacity: ' + ', '.join(f'{r} {fmt(r, c[r])}' for r in RESOURCES))
    if top_share:
        lines.append('  simulator level (spread): ' + ', '.join(f'{r} {fmt(r, top_share[r])}' for r in RESOURCES))
    # 1. Each copy whole within an SLR: any of them, or the one named.
    if a.copy_cap == 'none':
        allowed = [dict(c, **{r: math.inf for r in RESOURCES}) for c in caps]
    elif a.copy_cap == 'any':
        allowed = caps
    elif a.copy_cap in ('smallest', 'largest'):
        allowed = [(min if a.copy_cap == 'smallest' else max)(caps, key=lambda c: c['lut'])]
    else:
        allowed = [c for c in caps if c['name'] == a.copy_cap]
        if not allowed:
            sys.exit(f'--copy-cap: no SLR {a.copy_cap} in the region')
    fill = lambda c, s: max(res[c][r] / s[r] if s[r] else (math.inf if res[c][r] else 0.0) for r in RESOURCES)
    big = sorted(res, key=lambda c: -lut_equivalent(res[c], caps))
    over = []
    lines.append(f'  copy cap: {a.copy_cap} (' + ', '.join(s['name'] for s in allowed) + ')')
    for c in big[:6]:
        lines.append(f'    {c}: ' + ', '.join(f'{r} {fmt(r, res[c][r])}' for r in RESOURCES)
                     + '; of each SLR\'s capacity (binding resource): '
                     + ', '.join(f'{s["name"]} {100 * fill(c, s):.0f} %' for s in caps))
    for c, sp in splits.items():
        lines.append(f'    {c} split in {len(sp["slr"])} parts (foldsim-slr-split): ' + '; '.join(
            f'{n} in {names[g]}: ' + ', '.join(f'{r} {fmt(r, part_of(c, k)[r])}' for r in RESOURCES)
            for k, (n, g) in enumerate(zip(sp['parts'], sp['slr'])))
            + '; crossings ' + ', '.join(f'p{i}->p{j} {b:,} bits' for i, j, b in sp['cross']))
    for c in res:
        if c in splits:
            # Each part within its SLR's capacity.
            bad = [(sp_k, r) for sp_k, g in enumerate(splits[c]['slr']) for r in RESOURCES
                   if part_of(c, sp_k)[r] > caps[g][r] + 1e-9]
            if bad:
                over.append((c, names[splits[c]['slr'][bad[0][0]]], sorted({r for _, r in bad})))
            continue
        if all(fill(c, s) > 1 + 1e-9 for s in allowed):
            s = min(allowed, key=lambda s: fill(c, s))
            over.append((c, s['name'], [r for r in RESOURCES if res[c][r] > s[r] + 1e-9]))

    # 2. An assignment of whole copies within each SLR's fill in every resource (fold-static sll-estimate: LUT
    # sites, then LUTRAM LUTs, flops, and URAM and RAMB36 in tenths as further resources); where there is none,
    # the largest copy is left out of the assignment and spread over the room the others leave, from the SLR the
    # search gave it outwards (the placer cuts it).
    # A copy's memories count up to the largest SLR's room: past it their primitives go apart.
    def vec(x, sites, clamp=None):
        return '/'.join([str(max(0, round(sites)))] + [str(max(0, round(x[r]))) for r in LOGIC[1:]] +
                        [str(max(0, round(10 * (min(x[r], clamp[r]) if clamp else x[r])))) for r in ('uram', 'bram')])

    def estimate(sizes, caps_g, memories=True):
        if os.path.exists(report):
            os.remove(report)
        # Without `memories` the copies' memories are left to go apart (the placer balances the logic).
        mem_room = {r: max(0.0, max(x[r] for x in caps_g)) if memories else 0.0 for r in ('uram', 'bram')}
        opts = (f'{fold} schedule-only report={report} sll-estimate slr-sll-budget={budget} slr-capacities='
                + ','.join(vec(c, spl * c['lut']) for c in caps_g) + ' slr-sites='
                + ','.join(f'{c}:{vec(sizes[c], spl * sizes[c]["lut"], mem_room)}' for c in classes))
        run([a.foldsim_opt, models, f'--pass-pipeline=builtin.module(foldsim-fold-static{{{opts}}})', '-o',
             os.devnull])
        text = open(report).read()
        est_lines = text[text.index('  sll estimate:'):].split('\n')
        sec = [est_lines[0]] + list(takewhile(lambda l: l.startswith('    '), est_lines[1:]))
        wires = {}
        for l in sec:
            mw = re.match(r'    wires (\S+) (\S+) (\d+)$', l)
            if mw:
                wires[(mw.group(1), mw.group(2))] = int(mw.group(3))
        sec = [l for l in sec if not l.startswith('    wires ')]
        groups = None
        # A copy larger than every SLR's room is put whole in the largest: not an assignment within the room.
        if re.search(r'\(worst \d+, total \d+\)', '\n'.join(sec)) and \
                not any(l.startswith('    larger than every SLR') for l in sec):
            groups = []
            for g in range(len(room0)):
                mm = re.search(rf'^    slr{g} \(\d+ sites\):(.*)$', '\n'.join(sec), re.M)
                groups.append(mm.group(1).split() if mm else [])
        return text, sec, wires, groups

    text, sec, wires, groups = estimate(res_est, room0)
    # The refold must be the generation's schedule: a compiler that does not take the generation's options the same
    # way (an older one without staging: e8, 11-07, refolded to II 2,748 for 2,931) checks another netlist.
    rii = schedule_ii(text)
    if rii != ii:
        finish(lines, f'REFUSE: the check\'s refold has II {rii}, the netlist\'s schedule II {ii}: this compiler '
                      f'({a.foldsim_opt}) does not reproduce the generation; run the check with the generation\'s')
    # No assignment of whole copies: the largest copies by logic (at most model['cut_copies']) left out one by one
    # and spread over the room the others leave.
    cuts = []
    by_logic = sorted((c for c in res if c not in splits),
                      key=lambda c: -max(res[c][r] / room0[0][r] for r in LOGIC if room0[0][r] > 0))
    # Guide (--slr-policy guide, any SLR): a copy past every SLR is cut by the placer, as a copy left out of the
    # assignment is (its own cut by Rent's rule, its links over the boundary): left out from the start.
    guide = a.slr_policy == 'guide' and a.copy_cap == 'any'
    forced = [c for c, _, _ in over] if guide and mode == 'netlist' else []
    # The other copies within the SLRs' balanced share (the placer balances the SLRs' fill: each SLR at most the
    # region's fill of the whole design plus model['balance_slack']), the cut copies filling the room left.
    tot_full = {r: sum(c[r] for c in full) for r in RESOURCES}
    u = {r: (sum(x[r] for x in res.values()) + (top_share or {}).get(r, 0.0)) / tot_full[r] if tot_full[r] else 0.0
         for r in RESOURCES}
    balanced = [{r: min(room0[g][r], (u[r] + model.get('balance_slack', 0.05)) * full[g][r] - top_spread[g][r])
                 for r in RESOURCES} for g in range(len(full))]
    if forced:
        cuts = list(forced)
        _, sec2, _, groups = estimate(dict(res_est, **{c: new_acc() for c in cuts}), balanced, memories=False)
        sec += ['    (past every SLR: ' + ', '.join(cuts) + ' left out, spread over the room the others leave)'] \
            + sec2[1:]
    while groups is None and mode == 'netlist' and (not over or guide) and \
            len(cuts) < len(forced) + model.get('cut_copies', 2):
        cuts.append(next(c for c in by_logic if c not in cuts))
        _, sec2, _, groups = estimate(dict(res_est, **{c: new_acc() for c in cuts}), balanced, memories=False)
        sec += ['    (no whole assignment: ' + ', '.join(cuts) + ' left out, spread over the room the others leave)'] \
            + sec2[1:]
    cut_copy = cuts[-1] if cuts else None
    lines += sec
    largest = big[0]
    lines.append(f'{name}: largest class copy {largest}: ' + ', '.join(
        f'{s["name"]} {100 * max(res[largest][r] / s[r] if s[r] else 0 for r in RESOURCES):.0f} %' for s in caps)
        + ' of the SLR\'s capacity (its binding resource)')
    shares = slls = failed = None
    # Memories go apart (a copy's primitives where there is room): within the region's, at the fill.
    mem_over = [r for r in ('uram', 'bram') if sum(x[r] for x in res.values()) > sum(x[r] for x in room_whole) + 1e-9]
    # The whole design against the region at the placer's fill (the placed builds: 66-74 % of its LUTs).
    region_fill = {r: (sum(x[r] for x in res.values()) + (top_share or {}).get(r, 0.0)) / sum(c[r] for c in full)
                   for r in LOGIC if sum(c[r] for c in full)}
    # The placer fills an SLR's room to `fill` with whole copies, the region as a whole to `region_fill` (placed
    # builds 66-74 % of its LUTs; r16c failed partitioning at 84 %).
    region_cap = {r: model.get('region_fill', {}).get(r, fill_of[r]) for r in LOGIC}
    lines.append('  the design over the region: ' + ', '.join(f'{r} {100 * f:.0f} %' for r, f in region_fill.items())
                 + ' (region fill ' + ', '.join(f'{r} {100 * region_cap[r]:.0f} %' for r in LOGIC) + ')')
    fill_over = [r for r, f in region_fill.items() if f > region_cap[r] + 1e-9]
    if groups is not None:
        shares = {c: {r: [float(c in g) for g in groups] for r in RESOURCES} for c in classes}
        # A cut copy fills the SLRs up to the region's fill (the placer fills past an SLR's whole-copy room to
        # place it).
        roomr = [{r: (region_cap[r] * full[g][r] - top_spread[g][r]) if r in LOGIC else room_whole[g][r]
                  for r in RESOURCES} for g in range(len(groups))]
        room = [{r: roomr[g][r] - sum(res[c][r] for c in groups[g] if c not in cuts and c not in splits)
                 - sum(part_of(c, k)[r] for c, sp in splits.items() for k, h in enumerate(sp['slr']) if h == g)
                 for r in RESOURCES} for g in range(len(groups))]
        for c, sp in splits.items():
            shares[c] = {r: [sum(sp['frac'][r][k] for k, h in enumerate(sp['slr']) if h == g)
                             for g in range(len(groups))] for r in RESOURCES}
        failed = None
        for c in cuts:
            home = next(g for g, cs in enumerate(groups) if c in cs)
            use = [{r: roomr[g][r] - room[g][r] + top_spread[g][r] for r in RESOURCES} for g in range(len(groups))]
            shares[c] = distribute(res[c], room, home, use, full)
            if shares[c] is None:
                failed = c
                break
            room = [{r: room[g][r] - res[c][r] * shares[c][r][g] for r in RESOURCES} for g in range(len(groups))]
        if failed is None:
            for g, s in enumerate(full):
                use = {r: sum(res[c][r] * shares[c][r][g] for c in classes) + top_spread[g][r] for r in RESOURCES}
                lines.append(f'    {s["name"]} as placed: ' + ', '.join(
                    f'{r} {fmt(r, use[r])} ({100 * use[r] / s[r]:.0f} %)' if s[r] else f'{r} 0' for r in RESOURCES))
            for c in cuts:
                lines.append(f'    {c} cut over the SLRs: ' + ', '.join(
                    f'{s["name"]} {100 * x:.0f} %' for s, x in zip(full, shares[c]['lut'])) + ' of its logic')
            if model:
                slls = boundary_slls(shares, res, wires, top_spread, model, vmem_bits(gdir), splits,
                                     port_links(stext))
                for b, (lk, ct, sm) in enumerate(slls):
                    lines.append(f'    boundary {names[b]}-{names[b + 1]}: {lk + ct + sm:,.0f} SLLs projected '
                                 f'(links {lk:,.0f}, cut copies {ct:,.0f}{" and split crossings" if splits else ""}, '
                                 f'simulator level and kernel {sm:,.0f})')
    if a.plan:
        # The global plan: the region's caps (the margin) less the simulator level's spread share.
        room_p = [{r: caps[g][r] - top_spread[g][r] for r in RESOURCES} for g in range(len(caps))]
        other = model.get('other_slls', 0.0) + model.get('sll_per_vmem_bit', 0.0) * vmem_bits(gdir)
        at, pl = plan(res, splits, port_links(stext), room_p, model, other)
        if at is None:
            lines.append('  plan: no assignment of the copies and split parts within the SLRs\' caps; the closest past them: '
                         + ', '.join(f'{names[g]} {r} {fmt(r, u)} > {fmt(r, c_)}' for g, r, u, c_ in PLAN_OVER))
        else:
            for g, s_ in enumerate(names):
                lines.append(f'  plan {s_}: ' + ' '.join(sorted(c if k is None else f'{c}_p{k}' for (c, k), h in
                                                              at.items() if h == g)))
            lines.append('  plan SLLs: ' + ', '.join(f'{names[b]}-{names[b + 1]} {x:,.0f}' for b, x in enumerate(pl)))
            with open(a.plan, 'w') as f:
                f.write('# Generated by foldsim_sll_check.py --plan: split parts in pblocks of their SLR, every other '
                        'copy a soft USER_SLR_ASSIGNMENT (run with FOLDSIM_SLR_PBLOCKS=0).\n')
                for (c, k), h in sorted(at.items(), key=lambda x: (x[0][0], -1 if x[0][1] is None else x[0][1])):
                    cell = f'fs_split_{c}_static_p{k}' if k is not None else f'{c}_copy'
                    f.write(f'set c [get_cells -quiet -hierarchical {cell}]\n'
                            f'if {{[llength $c] == 1}} {{ set_property USER_SLR_ASSIGNMENT {names[h]} $c'
                            + (f'; fs_slr_pblock $c {names[h]}' if k is not None else '') + ' }\n')
    if a.json:
        json.dump(dict(name=name, ii=ii, mode=mode, copies=res, top=top_share, wires=[[x, y, w] for (x, y), w in
                                                                                     wires.items()],
                       assignment=groups, cuts=cuts, shares=shares,
                       slls=slls, over=[c for c, _, _ in over], caps=caps, margin=region.get('margin', 1.0)),
                  open(a.json, 'w'), indent=1)
    if mode != 'netlist':
        finish(lines, 'ESTIMATE (schedule resources, not a verdict): '
                      + ('an assignment within the SLRs' if groups is not None else 'no assignment'))
    if over and not guide:
        verdict = 'REFUSE: ' + '; '.join(f'{c} fits no SLR whole (the nearest, {s}: over its {"/".join(r)})'
                                         for c, s, r in over) + \
            ' (a copy past its SLR is cut by the placer, its SLLs unknown: r16c failed so)'
    elif fill_over:
        verdict = 'REFUSE: the design\'s ' + ', '.join(f'{r} {100 * region_fill[r]:.0f} %' for r in fill_over) + \
            ' of the region, past the placement fill (' + ', '.join(f'{100 * region_cap[r]:.0f} %' for r in fill_over) + \
            '; r16c failed placement at 84 % of its LUTs)'
    elif mem_over:
        verdict = f'REFUSE: the copies\' {"/".join(mem_over)} past the region\'s at the placement fill'
    elif groups is None:
        verdict = 'REFUSE: no assignment of the copies within the SLRs\' fill, even with the largest cut'
    elif failed is not None:
        verdict = f'REFUSE: {failed} does not fit the room the other copies leave in the SLRs'
    elif slls is None:
        verdict = 'OK: an assignment within the SLRs (no placement model in the region: SLLs not projected)'
    else:
        worst = max(sum(x) for x in slls)
        wb = max(range(len(slls)), key=lambda b: sum(slls[b]))
        where = f'{names[wb]}-{names[wb + 1]}'
        cutn = f'; {", ".join(cuts)} cut over SLRs' if cuts else ''
        if worst > region.get('sll_per_boundary', budget):
            verdict = f'REFUSE: {worst:,.0f} SLLs projected over {where}, past its {region.get("sll_per_boundary")}' \
                      + cutn
        elif worst > budget or cuts:
            verdict = f'WARN: {worst:,.0f} SLLs projected over {where} (budget {budget}){cutn}'
        else:
            verdict = f'OK: {worst:,.0f} SLLs projected over {where}, within the {budget} budget'
    finish(lines, verdict)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr[-3000:] if e.stderr else str(e))
        sys.exit(2)
