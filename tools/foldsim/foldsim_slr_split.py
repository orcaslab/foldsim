#!/usr/bin/env python3
"""foldsim_slr_split.py <gen dir | foldsim_sim.mlir> --copy <copy> [--parts 2,3] [--slr-region v80|<json>]
                        [--slrs SLR2,SLR1[,SLR0]] [--global <regex>] [--broadcast <pins>] [--comb-penalty P]
                        [--serialize 1,2,4] [--sll-json <foldsim_sll_check json>] [--out <dir>] [--json <file>]

Whether one class copy too large for an SLR can be split over SLRs, and at what cost: an analysis of the copy's
own netlist (the generated simulator's `<copy>_static` module, its children flattened; black boxes by
foldsim_cost.BLACKBOX), never a verdict on the rest of the design.

1. Every op is a node with its resources (foldsim_cost's per-op LUT model, its packing and trimming credited per op
   kind; flops; a memory with all its ports one node, LUTRAM / RAMB36 / URAM288 by foldsim_cost.mem_cost and
   foldsim_sll_check.mem_ram), scaled to placed ones as foldsim_sll_check.netlist_resources does (the region's
   netlist_*_scale x placed_per_synthesized). Every SSA value is a net of its bits, its pins its driver and users.
   Each net is classified: registered (driven by a flop, a memory's registered read, or a black box, whose outputs
   are registered), memory-read (a memory's combinational read), port (an input of the copy: a link) or
   combinational (wiring of registered bits stays registered).
2. Replicable control: the greatest set of registers (and memories) whose next state depends only on themselves,
   constants and the global fold-control inputs (`--global`, the fold's state / instance strobes and reset by
   default), and the logic over them alone: every part can hold its own copy instead of the net crossing.
   Broadcasts: registered nets of at least `--broadcast` pins outside that set, whose driving register can be
   duplicated per part (the crossing moves to its next-state input, the fanout splits).
3. A resource-balanced k-way min-cut (Mt-KaHyPar, connectivity objective; for 3 parts mapped onto the SLR chain,
   Steiner-tree objective) of the copy, the parts' targets in proportion to the SLRs' capacities, in three
   variants:
     a  plain: every net cuttable at its bits;
     b  replicable control duplicated per part (its nets never cut; the duplicates' area charged to each part that
        reads them), broadcast registers duplicated (their users join the next-state net);
     c  b, with a net that is neither registered nor a port cuttable only at `--comb-penalty` times its bits: each
        such crossing needs a new stage register (the SLR boundary a fold stage boundary), its latency the
        crossings in series on the longest path (the class's evaluation that many host cycles longer).
4. Per variant: the parts' resources (LUT/LUTRAM/FF/BRAM/URAM) and their share of the SLR, cut nets and bits per
   boundary, the projected SLLs per boundary for each serialization factor k (a k-th of the bits, as
   foldsim_sll_check counts serialized links; plus the rest of the design's crossings from `--sll-json`: links and
   the simulator level), the area added by replication, the crossings needing a new register and the added
   latency (L). The H change itself is measured by a schedule-only refold (foldsim-opt split-latency).

Mt-KaHyPar: `pip install mtkahypar` (the python module). Exits 0, 2 on error.
"""
import argparse, json, math, os, re, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foldsim_cost as fc  # noqa: E402
import foldsim_sll_check as fsc  # noqa: E402

RES = ('lut', 'lutram', 'ff', 'bram', 'uram')
SSA = re.compile(r'%[\w.$-]+')
HDR = re.compile(r'^\s*hw\.module(\.extern)?(?:\s+private)?\s+@([\w$.]+)(?:<[^>]*>)?\((.*)\)\s*'
                 r'(?:->\s*\((.*)\))?\s*(?:attributes.*)?\{?\s*$')
# Net classes a crossing needs no new register for.
REGISTERED = ('reg', 'port')


def module_texts(path, want):
    """{module: [lines]} of `want` and every module it instantiates, from an MLIR file."""
    mods, cur, lines = {}, None, None
    for line in open(path):
        s = line.lstrip()
        if s.startswith('hw.module'):
            m = HDR.match(line)
            name = m.group(2) if m else None
            if m and m.group(1):
                mods[name] = None  # extern
                cur = None
                continue
            cur, lines = name, [line]
            continue
        if cur is not None:
            lines.append(line)
            if s.startswith('}'):
                mods[cur] = lines
                cur = None
    if want not in mods:
        sys.exit(f'no module @{want} in {path}')
    keep, todo = {}, [want]
    while todo:
        n = todo.pop()
        if n in keep or n not in mods:
            continue
        keep[n] = mods[n]
        for l in mods[n] or ():
            if 'hw.instance' in l:
                mm = re.search(r'hw\.instance\s+"[^"]*"\s+(?:sym\s+@\S+\s+)?@([\w$.]+)', l)
                if mm:
                    todo.append(mm.group(1))
    return keep


class Net:
    __slots__ = ('name', 'width', 'driver', 'users', 'cls', 'readaddr')


class Graph:
    """The copy's flat netlist: nodes (ops, registers, memories, black boxes, ports) and nets."""

    def __init__(self):
        self.kind, self.name, self.op, self.res = [], [], [], []
        self.ins = []           # node -> [value names it reads]
        self.nets = {}          # value -> Net
        self.consts = set()
        self.clocks = set()
        self.memlat = {}        # memory node -> read latency
        self.reg_next = {}      # register node -> its next-value operand
        self.rd_ins = defaultdict(list)   # memory node -> its read ports' address / enable values
        self.info = {}          # op / register / instance node -> (op, rhs, operands in order, results)
        self.mports = defaultdict(list)   # memory node -> [('r', result, [address, enable]) | ('w', data, [ctl])]
        self.memw = {}          # memory node -> its word width
        self.cval = {}          # constant value -> int
        self.defnode = {}       # value (and memory) -> the node defining it
        self.body = []          # the module body's op lines, in order (foldsim-slr-split's map order)

    def node(self, kind, name, op=''):
        self.kind.append(kind)
        self.name.append(name)
        self.op.append(op)
        self.res.append(dict.fromkeys(RES, 0.0))
        self.ins.append([])
        return len(self.kind) - 1

    def net(self, v, w, driver):
        n = Net()
        n.name, n.width, n.driver, n.users, n.cls, n.readaddr = v, w, driver, [], None, False
        self.nets[v] = n
        self.defnode[v] = driver
        return n


def field_width(t, field):
    """Width of a struct or union type's field."""
    inner = t[t.index('<') + 1:t.rindex('>')]
    for f in fc.split_top(inner):
        name, _, ft = f.partition(':')
        if name.strip() == field:
            return width(ft)
    return 0


def width(t):
    """foldsim_cost.width, with nested aggregates printed short (!hw.array<3xarray<16xi64>>) read as such."""
    t = re.sub(r'(^|[x\s:])(array|uarray|struct|union)<', r'\1!hw.\2<', t.strip())
    return fc.width(t)


def optype(rhs):
    """The printed type of an op (after its first ' : '; foldsim_cost.result_type miscounts the '>' of '->' as a
    closing bracket, so a struct result after '->' came out as its last field's type)."""
    return rhs.split(' : ', 1)[1].strip() if ' : ' in rhs else ''


def result_width(op, rhs, nops):
    """Width of an op's (single) result from its printed form (the printed type is not always the result's)."""
    head = rhs.split(' : ')[0]
    t = optype(rhs)
    if ' -> ' in t:
        return width(t.rsplit(' -> ', 1)[1].strip())
    if op in ('comb.icmp', 'comb.parity'):
        return 1
    if op == 'comb.concat':
        return sum(width(x) for x in fc.split_top(t))
    if op == 'hw.array_get':
        at = fc.split_top(t)[0].strip()
        n, el = at[at.index('<') + 1:-1].split('x', 1)
        return width(el)
    if op == 'hw.array_create':
        return nops * width(t)
    if op in ('hw.struct_extract', 'hw.union_extract'):
        m = re.search(r'\["([^"]+)"\]', head)
        return field_width(t, m.group(1)) if m else 0
    if op == 'hw.array_inject':
        return width(fc.split_top(t)[0])
    return width(t)


def ports(text):
    out = []
    for p in fc.split_top(text or ''):
        m = re.match(r'\s*(in|out|inout)\s+(%[\w.$-]+|"?[^:"]*"?)\s*:\s*(.*)$', p.strip())
        if m:
            out.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    return out


def build(texts, top, region, model=False):
    """The flat graph of module `top` (children: black boxes or whole-subtree nodes; with `model`, the pre-fold
    class model, its children other classes' instances: no resources of this copy, their outputs links)."""
    g = Graph()
    g.link_insts = set()
    lines = texts[top]
    hm = HDR.match(lines[0])
    mod = fc.Mod(top, hm.group(3) + (',' + hm.group(4) if hm.group(4) else ''))
    for d, n, t in ports(hm.group(3)):
        if d != 'in':
            continue
        if t.startswith('!seq.clock'):
            g.clocks.add(n)
            continue
        g.net(n, width(t), g.node('in', n[1:]))
    mems = {}           # memory value -> (node, rec)
    opnode = {}         # result value -> node (for per-op-kind credits)
    kinds = defaultdict(list)   # op kind -> [nodes]
    child_cost = {}
    for raw in lines[1:]:
        s = raw.strip()
        if s and not s.startswith('hw.output') and not s.startswith('}'):
            g.body.append(s)
        if s.startswith('hw.output'):
            for i, v in enumerate(SSA.findall(s.split(' : ')[0])):
                mod.uses[v] += 1
                n = g.node('out', f'out{i}')
                g.ins[n].append(v)
                g.info[n] = ('out', '', [v], [])
            break
        if s.startswith('seq.firmem.write_port') or s.startswith('seq.firmem.read_write_port'):
            mm = re.search(r'(%[\w.$-]+)\[', s)
            if mm and mm.group(1) in mems:
                mems[mm.group(1)][1][4] += 1
                node = mems[mm.group(1)][0]
                vs = [v for v in SSA.findall(s.split(' : ')[0]) if v != mm.group(1) and v not in g.clocks]
                g.ins[node] += vs
                dm = re.search(r'\]\s*=\s*(%[\w.$-]+)', s)
                data = dm.group(1) if dm else None
                g.mports[node].append(('w', data, [v for v in vs if v != data]))
            continue
        if ' = ' not in s:
            for v in SSA.findall(s.split(' : ')[0]):
                mod.uses[v] += 1
            continue
        if not s.startswith('%'):
            continue
        lhs, rhs = s.split(' = ', 1)
        op = rhs.split()[0]
        res = SSA.findall(lhs)
        operands = SSA.findall(rhs.split(' : ')[0]) if op != 'hw.instance' else \
            SSA.findall(rhs.split(') -> ')[0])
        for v in operands:
            mod.uses[v] += 1
        if op in fc.CONSTS:
            mod.consts.update(res)
            g.consts.update(res)
            if op == 'hw.constant':
                fc.track(mod, op, rhs, res, operands)
                cm = re.match(r'hw\.constant\s+(-?\d+|true|false)', rhs)
                if cm:
                    g.cval[res[0]] = {'true': 1, 'false': 0}.get(cm.group(1), None) if cm.group(1) in ('true', 'false') \
                        else int(cm.group(1))
            continue
        if op in ('seq.from_clock', 'seq.to_clock', 'seq.const_clock'):
            g.clocks.update(res)
            continue
        if op in fc.FREE:
            if op == 'comb.extract' and operands:
                lo = re.search(r'from (\d+)', rhs)
                ew = fc.int_width(fc.result_type(rhs).split('->')[-1])
                if lo and ew:
                    mod.extracts[operands[0]].append((int(lo.group(1)), ew))
            fc.track(mod, op, rhs, res, operands)
            n = g.node('op', res[0][1:], op)
            g.ins[n] = operands
            g.info[n] = (op, rhs, operands, res)
            g.net(res[0], result_width(op, rhs, len(operands)), n)
            continue
        if op in fc.REGS:
            w = result_width(op, rhs, 1)
            mod.ff += w
            n = g.node('reg', res[0][1:], op)
            g.res[n]['ff'] = w
            g.ins[n] = [v for v in operands if v not in g.clocks]
            g.info[n] = (op, rhs, g.ins[n], res)
            if operands:
                g.reg_next[n] = operands[0]
            g.net(res[0], w, n)
            continue
        if op == 'seq.firmem':
            mm = re.search(r'(\d+),\s*(\d+),\s*\w+,\s*\w+.*<\s*(\d+)\s*x\s*(\d+)', rhs)
            nm = re.search(r'name "([^"]*)"', rhs)
            st = re.search(r'ram_style" = "\\22(\w+)', rhs)
            mk = re.search(r'mask\s+(\d+)\s*>', rhs)
            rec = [nm.group(1) if nm else res[0][1:], int(mm.group(3)), int(mm.group(4)), 0, 0, int(mm.group(1)),
                   st.group(1) if st else None, int(mk.group(1)) if mk else 1, bool(nm)]
            n = g.node('mem', rec[0], op)
            mems[res[0]] = (n, rec)
            g.defnode[res[0]] = n
            g.memlat[n] = rec[5]
            g.memw[n] = rec[2]
            continue
        if op == 'seq.firmem.read_port':
            mm = re.search(r'(%[\w.$-]+)\[', rhs)
            node, rec = mems[mm.group(1)]
            rec[3] += 1
            rd = [v for v in operands if v != mm.group(1) and v not in g.clocks]
            g.ins[node] += rd
            g.rd_ins[node] += rd
            g.mports[node].append(('r', res[0], rd))
            g.net(res[0], rec[2], node)
            continue
        if op == 'hw.instance':
            mm = re.search(r'hw\.instance\s+"([^"]*)"\s+(?:sym\s+@\S+\s+)?@([\w$.]+)', rhs)
            child = mm.group(2)
            n = g.node('inst', mm.group(1), child)
            base = child.split('__')[0]
            if child in fc.BLACKBOX or base in fc.BLACKBOX:
                l, f, _ = fc.BLACKBOX.get(child, fc.BLACKBOX.get(base))
                g.res[n]['lut'] += l
                g.res[n]['ff'] += f
            elif model and texts.get(child):
                g.link_insts.add(n)
            elif texts.get(child):
                if child not in child_cost:
                    child_cost[child] = subtree_cost(texts, child)
                for r in RES:
                    g.res[n][r] += child_cost[child][r]
            g.ins[n] = [v for v in operands if v not in g.clocks]
            g.info[n] = ('hw.instance', rhs, g.ins[n], res)
            outs = rhs.split(') -> (', 1)[1].rsplit(')', 1)[0] if ') -> (' in rhs else ''
            types = [p.split(':', 1)[1] for p in fc.split_top(outs) if ':' in p]
            for v, t in zip(res, types):
                g.net(v, width(t), n)
            continue
        c = fc.lut_cost(op, rhs, mod, operands)
        fc.track(mod, op, rhs, res, operands)
        n = g.node('op', res[0][1:] if res else op, op)
        g.ins[n] = operands
        g.info[n] = (op, rhs, operands, res)
        for v in res:
            g.net(v, result_width(op, rhs, len(operands)), n)
        if c is None:
            continue
        k = op.split('.')[1]
        mod.lut[k] += c
        g.res[n]['lut'] += c
        kinds[k].append(n)
        if op in ('comb.shl', 'comb.shru', 'comb.shrs') and c and len(res) == 1:
            mod.shifts[res[0]] = (c, fc.int_width(fc.result_type(rhs)) or 0, op.split('.')[1])
        if op == 'comb.and' and c and len(res) == 1:
            mod.ands[res[0]] = (c, [o for o in operands if o not in mod.consts])
        elif op == 'comb.or' and c and len(res) == 1:
            mod.ors.append((res[0], [o for o in operands if o not in mod.consts], c, fc.int_width(fc.result_type(rhs)) or 0))
        if op == 'comb.mux' and len(operands) == 3 and c:
            mod.muxes[res[0]] = (c, operands[1], operands[2])
    # foldsim_cost's packing / trimming / enable refunds, credited to the op kinds they act on.
    before = dict(mod.lut)
    fc.refund_enables(mod)
    fc.pack_and_or(mod)
    fc.trim_shifts(mod)
    for k, ns in kinds.items():
        f = mod.lut[k] / before[k] if before.get(k) else 1.0
        for n in ns:
            g.res[n]['lut'] *= f
    for _, (n, rec) in mems.items():
        acc = fsc.new_acc()
        fsc.add_memory(acc, rec)
        for r in RES:
            g.res[n][r] += acc[r]
    # Scaled to placed resources (foldsim_sll_check.netlist_resources).
    ls, fs = region.get('netlist_lut_scale', 1.0), region.get('netlist_ff_scale', 1.0)
    lr = region.get('netlist_lutram_scale', ls)
    pl = region.get('placed_per_synthesized', {})
    for x in g.res:
        x['lut'] *= ls * pl.get('lut', 1.0)
        x['lutram'] *= lr * pl.get('lutram', 1.0)
        x['ff'] *= fs * pl.get('ff', 1.0)
    # Nets of no bits (i0) carry nothing.
    for v in [v for v, nt in g.nets.items() if nt.width == 0]:
        del g.nets[v]
    # Users of each net.
    for n, vs in enumerate(g.ins):
        g.ins[n] = [v for v in vs if v in g.nets]
        for v in set(g.ins[n]):
            g.nets[v].users.append(n)
    for n in g.rd_ins:
        g.rd_ins[n] = [v for v in g.rd_ins[n] if v in g.nets]
    return g


def op_map(g, P):
    """Per op of the module's body, in order: its part (P of the node defining it), or -1 (constants, clocks and the
    replicated nodes, which P leaves at -1)."""
    out = []
    for s in g.body:
        if s.startswith('seq.firmem.write_port') or s.startswith('seq.firmem.read_write_port'):
            v = re.search(r'(%[\w.$-]+)\[', s).group(1)
        elif ' = ' in s and s.startswith('%'):
            v = SSA.findall(s.split(' = ', 1)[0])[0]
        else:
            v = None
        n = g.defnode.get(v)
        out.append(P[n] if n is not None else -1)
    return out


def deps(g, n):
    """The values a node's outputs depend on combinationally (a memory: its read addresses only)."""
    return g.rd_ins.get(n, []) if g.kind[n] == 'mem' else g.ins[n]


BITWISE = {'comb.and', 'comb.or', 'comb.xor', 'hw.wire'}


def struct_fields(t):
    """[(name, width, offset)] of a struct type, its first field the most significant."""
    t = t.strip()
    inner = t[t.index('<') + 1:t.rindex('>')]
    fs = []
    for f in fc.split_top(inner):
        name, _, ft = f.partition(':')
        fs.append((name.strip(), width(ft)))
    out, off = [], sum(w for _, w in fs)
    for name, w in fs:
        off -= w
        out.append((name, w, off))
    return out


def array_shape(t):
    """(elements, element width) of an array type."""
    t = width_norm(t)
    inner = t[t.index('<') + 1:t.rindex('>')]
    n, el = inner.split('x', 1)
    return int(n), width(el)


def width_norm(t):
    return re.sub(r'(^|[x\s:])(array|uarray|struct|union)<', r'\1!hw.\2<', t.strip())


def chunk_graph(g, C):
    """The graph with every value cut into C-bit chunks and every op whose result bits follow their operands' bits
    one for one (wiring, bitwise logic, multiplexers, registers, memory columns) cut likewise, so a min-cut can
    separate lanes of a wide datapath; other ops stay whole (a node reading every chunk of its operands)."""
    cg = Graph()
    cg.consts, cg.clocks = g.consts, g.clocks

    def nch(v):
        return max(1, -(-g.nets[v].width // C)) if v in g.nets else 0

    def cn(v, j):
        return v if nch(v) == 1 else f'{v}#{j}'

    def cbits(v, j):
        w = g.nets[v].width
        return min(C, w - j * C)

    def src(v, lo, n):
        """The chunk nets of v covering its bits [lo, lo + n)."""
        if v not in g.nets or n <= 0:
            return []
        return [cn(v, j) for j in range(max(0, lo // C), min(nch(v), -(-(lo + n) // C)))]

    def whole(v):
        return [cn(v, j) for j in range(nch(v))]

    def scale(n, f):
        return {r: g.res[n][r] * f for r in RES}

    def mapped(n, op, res_v, segs, ctl=(), cost_whole=False):
        """Result chunks of `res_v`, each reading the source chunks its bits come from (segs: [(res_lo, len, src,
        src_lo)]) and every chunk of the control values `ctl`."""
        if res_v not in g.nets:
            return
        w = g.nets[res_v].width
        cw = [y for c in ctl for y in whole(c)]
        for j in range(nch(res_v)):
            lo, hi = j * C, j * C + cbits(res_v, j)
            ins = []
            for rl, ln, sv, sl in segs:
                a, b = max(lo, rl), min(hi, rl + ln)
                if a < b:
                    ins += src(sv, sl + a - rl, b - a)
            m = cg.node(g.kind[n], g.name[n] + (f'[{j}]' if nch(res_v) > 1 else ''), op)
            cg.res[m] = scale(n, (hi - lo) / w if w else 0)
            cg.ins[m] = ins + cw
            if g.kind[n] == 'reg':
                cg.reg_next[m] = ins[0] if ins else None
            cg.net(cn(res_v, j), hi - lo, m)

    def concat_segs(ops, ws, w):
        segs, pos = [], w
        for o, ow in zip(ops, ws):
            pos -= ow
            segs.append((pos, ow, o, 0))
        return segs

    for n in range(len(g.kind)):
        k = g.kind[n]
        if k == 'in':
            v = '%' + g.name[n]
            for j in range(nch(v)):
                m = cg.node('in', g.name[n] + (f'[{j}]' if nch(v) > 1 else ''))
                cg.net(cn(v, j), cbits(v, j), m)
            continue
        if k == 'out':
            v = g.info[n][2][0]
            for j in range(max(1, nch(v))):
                m = cg.node('out', g.name[n] + (f'[{j}]' if nch(v) > 1 else ''))
                cg.ins[m] = [cn(v, j)] if v in g.nets else []
            continue
        if k == 'mem':
            W = g.memw[n]
            ncol = max(1, -(-W // C))
            ctl = [c for kind, _, cs in g.mports[n] for c in cs]
            rds = [c for kind, _, cs in g.mports[n] if kind == 'r' for c in cs]
            for j in range(ncol):
                cb = min(C, W - j * C)
                m = cg.node('mem', g.name[n] + (f'[{j}]' if ncol > 1 else ''), 'seq.firmem')
                cg.res[m] = scale(n, cb / W)
                cg.memlat[m] = g.memlat[n]
                cg.memw[m] = cb
                ins = [y for c in ctl for y in whole(c)]
                for kind, d, _ in g.mports[n]:
                    if kind == 'w' and d in g.nets:
                        ins += src(d, j * C, cb)
                    elif kind == 'r' and d in g.nets:
                        cg.net(cn(d, j), cb, m)
                cg.ins[m] = ins
                cg.rd_ins[m] = [y for c in rds for y in whole(c)]
            continue
        op, rhs, ops, res = g.info[n]
        r0 = res[0] if res else None
        t = optype(rhs)
        head = rhs.split(' : ')[0]
        try:
            if op in BITWISE or (op == 'comb.mux' and ops and ops[0] in g.nets and g.nets[ops[0]].width == 1
                                 and r0 in g.nets) or (op == 'comb.mux' and ops and ops[0] in g.consts):
                data = ops[1:] if op == 'comb.mux' else ops
                ctl = ops[:1] if op == 'comb.mux' else []
                mapped(n, op, r0, [(0, g.nets[r0].width, d, 0) for d in data], ctl)
                continue
            if k == 'reg':
                # seq.firreg %next clock %c [reset sync|async %rst, %rv]
                nxt, rest = ops[0], ops[1:]
                ctl = rest[:1]
                rv = rest[1:2]
                mapped(n, op, r0, [(0, g.nets[r0].width, x, 0) for x in [nxt] + rv], ctl)
                continue
            w = g.nets[r0].width if r0 in g.nets else 0
            if op == 'comb.extract':
                lo = int(re.search(r'from (\d+)', rhs).group(1))
                mapped(n, op, r0, [(0, w, ops[0], lo)])
                continue
            if op == 'comb.concat':
                mapped(n, op, r0, concat_segs(ops, [width(x) for x in fc.split_top(t)], w))
                continue
            if op in ('hw.bitcast', 'hw.wire'):
                mapped(n, op, r0, [(0, w, ops[0], 0)])
                continue
            if op == 'hw.struct_extract':
                f = re.search(r'\["([^"]+)"\]', head).group(1)
                fw, off = next((fw, off) for nm, fw, off in struct_fields(width_norm(t)) if nm == f)
                mapped(n, op, r0, [(0, fw, ops[0], off)])
                continue
            if op == 'hw.struct_create':
                fs = struct_fields(width_norm(t))
                mapped(n, op, r0, concat_segs(ops, [fw for _, fw, _ in fs], w))
                continue
            if op == 'hw.struct_inject':
                f = re.search(r'\["([^"]+)"\]', head).group(1)
                fw, off = next((fw, off) for nm, fw, off in struct_fields(width_norm(t)) if nm == f)
                mapped(n, op, r0, [(0, off, ops[0], 0), (off, fw, ops[1], 0), (off + fw, w - off - fw, ops[0],
                                                                                off + fw)])
                continue
            if op in ('hw.union_extract', 'hw.union_create'):
                mapped(n, op, r0, [(0, w, ops[0], 0)])
                continue
            if op == 'hw.array_create':
                ew = width(t)
                nn = len(ops)
                mapped(n, op, r0, [((nn - 1 - i) * ew, ew, o, 0) for i, o in enumerate(ops)])
                continue
            if op == 'hw.array_concat':
                mapped(n, op, r0, concat_segs(ops, [width(x) for x in fc.split_top(t)], w))
                continue
            if op == 'hw.array_get':
                ne, ew = array_shape(fc.split_top(t)[0])
                idx = ops[1]
                if idx in g.cval and g.cval[idx] is not None:
                    mapped(n, op, r0, [(0, ew, ops[0], (g.cval[idx] % ne) * ew)])
                else:
                    mapped(n, op, r0, [(0, ew, ops[0], i * ew) for i in range(ne)], [idx])
                continue
            if op == 'hw.array_slice':
                at = t.split(' -> ')[0].strip().strip('()')
                ne, ew = array_shape(at)
                idx = ops[1]
                if idx in g.cval and g.cval[idx] is not None:
                    mapped(n, op, r0, [(0, w, ops[0], g.cval[idx] * ew)])
                else:
                    mapped(n, op, r0, [(0, w, ops[0], i * ew) for i in range(ne - w // max(1, ew) + 1)], [idx])
                continue
            if op == 'hw.array_inject':
                ne, ew = array_shape(fc.split_top(t)[0])
                idx, val = ops[1], ops[2]
                if idx in g.cval and g.cval[idx] is not None:
                    off = g.cval[idx] * ew
                    mapped(n, op, r0, [(0, off, ops[0], 0), (off, ew, val, 0), (off + ew, w - off - ew, ops[0],
                                                                                  off + ew)])
                else:
                    mapped(n, op, r0, [(0, w, ops[0], 0)] + [(i * ew, ew, val, 0) for i in range(ne)], [idx])
                continue
            if op == 'comb.replicate':
                mapped(n, op, r0, [], [ops[0]])
                continue
        except (StopIteration, ValueError, IndexError, KeyError, AttributeError):
            pass
        # A whole node: reads every chunk of its operands, drives every chunk of its results.
        m = cg.node(k, g.name[n], op)
        cg.res[m] = dict(g.res[n])
        cg.ins[m] = [y for o in ops for y in whole(o)]
        for v in res:
            if v in g.nets:
                for j in range(nch(v)):
                    cg.net(cn(v, j), cbits(v, j), m)
    for n, vs in enumerate(cg.ins):
        cg.ins[n] = [v for v in vs if v in cg.nets]
        for v in set(cg.ins[n]):
            cg.nets[v].users.append(n)
    for n in list(cg.rd_ins):
        cg.rd_ins[n] = [v for v in cg.rd_ins[n] if v in cg.nets]
    return cg


def subtree_cost(texts, name):
    """A child module's resources (raw counts), its own children included."""
    path = os.path.join(OUTDIR, f'.child_{name}.mlir')
    with open(path, 'w') as f:
        for n, ls in texts.items():
            if ls:
                f.writelines(ls)
    mods, _ = fc.parse(path)
    acc = fsc.new_acc()
    fsc.subtree(mods, name, set(), acc)
    os.remove(path)
    return acc


def classify(g):
    """Each net's class: reg, memread, port or comb (wiring of registered bits stays registered)."""
    order = topo(g)
    for n in order:
        k = g.kind[n]
        outs = [v for v in OUTS[n]]
        if not outs:
            continue
        if k == 'inst' and n in LINK_INSTS:
            c = 'port'
        elif k == 'reg' or k == 'inst' or (k == 'mem' and g.memlat[n] > 0):
            c = 'reg'
        elif k == 'mem':
            c = 'memread'
        elif k == 'in':
            c = 'port'
        elif g.op[n] in fc.FREE:
            cs = {g.nets[v].cls for v in g.ins[n]}
            c = 'reg' if cs <= {'reg'} and cs else ('port' if cs <= {'port', 'reg'} and cs else 'comb')
        else:
            c = 'comb'
        for v in outs:
            g.nets[v].cls = c


def outs_of(g):
    o = defaultdict(list)
    for v, nt in g.nets.items():
        o[nt.driver].append(v)
    return o


def is_source(g, n):
    """A node whose outputs start a combinational path."""
    k = g.kind[n]
    return k in ('reg', 'inst', 'in') or (k == 'mem' and g.memlat[n] > 0)


def topo(g):
    """Nodes in an order where every combinational node follows its drivers (sources first)."""
    N = len(g.kind)
    indeg = [0] * N
    succ = defaultdict(list)
    for n in range(N):
        if is_source(g, n):
            continue
        for v in set(deps(g, n)):
            d = g.nets[v].driver
            if not is_source(g, d):
                indeg[n] += 1
                succ[d].append(n)
    out = [n for n in range(N) if is_source(g, n)]
    work = [n for n in range(N) if not is_source(g, n) and indeg[n] == 0]
    while work:
        n = work.pop()
        out.append(n)
        for s in succ[n]:
            indeg[s] -= 1
            if indeg[s] == 0:
                work.append(s)
    seen = set(out)
    rest = [n for n in range(N) if n not in seen]
    if rest:
        # A combinational cycle (through a memory merged with its write port): appended.
        out += rest
    return out


def replicable(g, glob_re):
    """The greatest set of nodes that are functions of themselves, constants and global inputs only."""
    N = len(g.kind)
    rep = [False] * N
    for n in range(N):
        k = g.kind[n]
        if k == 'in':
            rep[n] = bool(glob_re.search(g.name[n]))
        elif k in ('op', 'reg', 'mem'):
            rep[n] = True
    users = defaultdict(set)
    for v, nt in g.nets.items():
        for u in nt.users:
            users[nt.driver].add(u)
    work = [n for n in range(N) if not rep[n]]
    while work:
        n = work.pop()
        for u in users[n]:
            if rep[u]:
                rep[u] = False
                work.append(u)
    return rep


def load_region(arg):
    path = os.path.join(HERE, 'foldsim_v80_slr.json') if arg == 'v80' else arg
    return json.load(open(path))


def partition(nodes, w, edges, ew, k, targets, chain, threads, seed):
    import mtkahypar
    global _MTK
    if _MTK is None:
        _MTK = mtkahypar.initialize(threads, False)
    mtkahypar.set_seed(seed)
    ctx = _MTK.context_from_preset(mtkahypar.PresetType.QUALITY if len(nodes) < 400000 else mtkahypar.PresetType.DEFAULT)
    ctx.set_partitioning_parameters(k, 0.03, mtkahypar.Objective.KM1)
    ctx.set_individual_target_block_weights([int(t) for t in targets])
    ctx.logging = False
    hg = _MTK.create_hypergraph(ctx, len(nodes), len(edges), edges, w, ew)
    if chain and k > 2:
        # The SLRs a chain: mapped onto a path graph (the Steiner-tree objective, a net spanning SLR2 and SLR0
        # crossing both boundaries).
        tg = _MTK.create_target_graph(ctx, k, k - 1, [(i, i + 1) for i in range(k - 1)], [1] * (k - 1))
        ph = hg.map_onto_graph(tg, ctx)
    else:
        ph = hg.partition(ctx)
    return [ph.block_id(i) for i in range(len(nodes))]


_MTK = None
OUTS = None
LOADS, LOADS_U, HIGH = {}, {}, []
LINK_INSTS = set()
CLUSTERS, PARTS = {}, {}
OUTDIR = '.'


def analyse(g, rep, bcast, variant, k, caps, slr_names, a, sll_rest):
    """Partition one variant; returns its report dict."""
    N = len(g.kind)
    use_rep = variant in ('b', 'c', 'r', 'p')
    # Graph nodes: every node, except (b, c) the replicable ones.
    in_graph = [not (use_rep and rep[n]) for n in range(N)]
    idx = {}
    nodes = []
    for n in range(N):
        if in_graph[n]:
            idx[n] = len(nodes)
            nodes.append(n)
    tot = {r: sum(g.res[n][r] for n in nodes) for r in RES}
    # Combined weight: each resource as a share of the parts' total capacity.
    capsum = {r: sum(c[r] for c in caps) for r in RES}
    scale = 1e8 / max(1e-9, sum(tot[r] / capsum[r] for r in RES if capsum[r]))

    def wt(n):
        return max(1, int(scale * sum(g.res[n][r] / capsum[r] for r in RES if capsum[r])))
    w = [wt(n) for n in nodes]
    W = sum(w)
    share = [sum(c[r] / capsum[r] for r in RES if capsum[r]) for c in caps]
    targets = [W * s / sum(share) * (1 + a.imbalance) for s in share]
    edges, ew, enets = [], [], []
    bset = set(bcast) if use_rep and variant != 'r' else set()
    merged = {}   # broadcast register's next-value net -> extra users (the broadcast's)
    for v in bset:
        nx = g.reg_next.get(g.nets[v].driver)
        if nx and nx in g.nets:
            merged.setdefault(nx, set()).update(g.nets[v].users)
    for v, nt in g.nets.items():
        if v in bset:
            continue
        if not in_graph[nt.driver]:
            continue
        pins = {idx[u] for u in nt.users if in_graph[u]} | {idx[nt.driver]}
        if v in merged:
            pins |= {idx[u] for u in merged[v] if in_graph[u]}
        if len(pins) < 2:
            continue
        wgt = max(1, nt.width)
        if variant in ('c', 'p') and nt.cls not in REGISTERED:
            wgt *= a.comb_penalty
        edges.append(sorted(pins))
        ew.append(int(wgt))
        enets.append(v)
    t0 = time.time()
    if variant in ('r', 'p'):
        # r: registered cuts only, every net that is neither registered nor a port uncuttable (its pins one
        # cluster: the cut adds no register). p: what foldsim-slr-split applies, a memory-read net uncuttable (a
        # combinational read stays with its memory), other combinational nets at the penalty (the pass copies their
        # logic into the reading part).
        par = list(range(len(nodes)))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for v, nt in g.nets.items():
            if nt.cls in REGISTERED or v in bset or not in_graph[nt.driver] or (variant == 'p' and nt.cls == 'comb'):
                continue
            pins = [idx[u] for u in nt.users if in_graph[u]] + [idx[nt.driver]]
            r0 = find(pins[0])
            for q in pins[1:]:
                rq = find(q)
                if rq != r0:
                    par[rq] = r0
        roots = sorted({find(i) for i in range(len(nodes))})
        cid = {r: j for j, r in enumerate(roots)}
        cw = [0] * len(roots)
        for i in range(len(nodes)):
            cw[cid[find(i)]] += w[i]
        cedges, cew = [], []
        for e, x in zip(edges, ew):
            pins = sorted({cid[find(i)] for i in e})
            if len(pins) > 1:
                cedges.append(pins)
                cew.append(x)
        # The largest cluster must fit a part: the targets widened to it.
        big = max(cw)
        ctargets = [max(t, big * (1 + a.imbalance)) for t in targets]
        cpart = partition(list(range(len(roots))), cw, cedges, cew, k, ctargets, True, a.threads, a.seed)
        part = [cpart[cid[find(i)]] for i in range(len(nodes))]
        CLUSTERS['r'] = (len(roots), big / max(1, sum(cw)))
    else:
        part = partition(nodes, w, edges, ew, k, targets, True, a.threads, a.seed)
    dt = time.time() - t0
    P = [-1] * N
    for i, n in enumerate(nodes):
        P[n] = part[i]
    PARTS[variant, k] = P
    # Replicable nodes: each part holds those its nodes read, transitively.
    added = dict.fromkeys(RES, 0.0)
    partres = [dict.fromkeys(RES, 0.0) for _ in range(k)]
    for n in nodes:
        for r in RES:
            partres[P[n]][r] += g.res[n][r]
    rep_parts = defaultdict(set)
    if use_rep:
        for p in range(k):
            work = [g.nets[v].driver for n in nodes if P[n] == p for v in g.ins[n] if not in_graph[g.nets[v].driver]]
            seen = set()
            while work:
                d = work.pop()
                if d in seen:
                    continue
                seen.add(d)
                work += [g.nets[v].driver for v in g.ins[d] if not in_graph[g.nets[v].driver]]
            for d in seen:
                rep_parts[d].add(p)
                for r in RES:
                    partres[p][r] += g.res[d][r]
        repN = [n for n in range(N) if not in_graph[n]]
        for n in repN:
            extra = max(0, len(rep_parts[n]) - 1)
            for r in RES:
                added[r] += extra * g.res[n][r]
        # A broadcast register duplicated in every part with its users but not it.
        for v in bset:
            d = g.nets[v].driver
            ps = {P[u] for u in g.nets[v].users if P[u] >= 0} - {P[d]}
            for p in ps:
                for r in RES:
                    partres[p][r] += g.res[d][r]
                    added[r] += g.res[d][r]
    # Cut nets: per boundary of the chain, the bits whose pins span it.
    bnd = [dict(nets=0, bits=0, newreg_nets=0, newreg_bits=0, port_bits=0, by_cls=defaultdict(int))
           for _ in range(k - 1)]
    newreg = set()
    cutnets = []

    def where(n):
        if P[n] >= 0:
            return {P[n]}
        return rep_parts.get(n, set())
    for v, nt in g.nets.items():
        if use_rep and not in_graph[nt.driver]:
            continue   # every part computes it
        ps = {P[u] for u in nt.users if P[u] >= 0}
        if v in bset:
            # The duplicate's next value crosses instead (counted with that net).
            continue
        if v in merged:
            ps |= {P[u] for u in merged[v] if P[u] >= 0}
        dp = P[nt.driver]
        if dp < 0 or not ps or ps <= {dp}:
            continue
        lo, hi = min(ps | {dp}), max(ps | {dp})
        cutnets.append(v)
        for b in range(lo, hi):
            x = bnd[b]
            x['nets'] += 1
            x['bits'] += nt.width
            x['by_cls'][nt.cls] += nt.width
            if nt.cls == 'port':
                x['port_bits'] += nt.width
            if nt.cls not in REGISTERED:
                x['newreg_nets'] += 1
                x['newreg_bits'] += nt.width
        if nt.cls not in REGISTERED:
            newreg.add(v)
    # Latency: the host cycles crossings add on one combinational path. Inserted model: a crossing register on
    # each combinational (or memory-read) cut net, a registered net crossing as it is. Versal model (no Laguna
    # flops: the compiler emits a TX register in the source SLR and an RX register in the destination, USER_SLL_REG):
    # a registered net +1 (its TX a copy of the source register), any other +2. io: on the copy's input-to-output
    # paths (late outputs, chained into other classes within a frame: the only paths on a tight recurrence; the
    # per-instance state loop has the class's state gap of slack).
    ins_cost = lambda v: 1 if v in newreg else 0
    tx_cost = lambda v: 1 if g.nets[v].cls in REGISTERED else 2
    lat = path_latency(g, P, ins_cost)
    lat_io = path_latency(g, P, ins_cost, io=True)
    lat_tx = path_latency(g, P, tx_cost)
    lat_tx_io = path_latency(g, P, tx_cost, io=True)
    # Fanout: the highest-load nets before and after (the most loads one part's copy drives; replicable control
    # per part, its load per part's replica).
    fan = []
    for v in HIGH:
        nt = g.nets[v]
        per = defaultdict(float)
        for u in set(nt.users):
            ps = [P[u]] if P[u] >= 0 else sorted(rep_parts.get(u, ()))
            for p in ps:
                per[p] += LOADS_U[(v, u)]
        fan.append((nt.name, LOADS[v], max(per.values()) if per else 0.0, rep[nt.driver], nt.cls))
    out = dict(variant=variant, parts=k, slrs=slr_names, seconds=round(dt, 1), nodes=len(nodes), nets=len(edges),
               part_res=partres, caps=caps, added=added, boundaries=[], latency=lat,
               latency_io=lat_io, latency_txrx=lat_tx, latency_txrx_io=lat_tx_io, fanout=fan,
               newreg_nets=len(newreg), newreg_bits=sum(g.nets[v].width for v in newreg),
               cut_nets=len(cutnets), cut_bits=sum(g.nets[v].width for v in cutnets))
    for b, x in enumerate(bnd):
        rest = sll_rest[b] if sll_rest and b < len(sll_rest) else 0.0
        out['boundaries'].append(dict(name=f'{slr_names[b]}-{slr_names[b + 1]}', nets=x['nets'], bits=x['bits'],
                                      newreg_nets=x['newreg_nets'], newreg_bits=x['newreg_bits'],
                                      by_cls=dict(x['by_cls']), rest=rest,
                                      slls={kk: math.ceil(x['bits'] / kk) + rest for kk in a.serialize}))
    top = sorted(cutnets, key=lambda v: -g.nets[v].width)[:12]
    out['widest_cuts'] = [(g.nets[v].name, g.nets[v].width, g.nets[v].cls, len(g.nets[v].users)) for v in top]
    return out


def path_latency(g, P, cost, io=False):
    """The most host cycles crossings add on one combinational path: `cost(v)` a crossing of net v (from its
    driver's part to its user's). io: paths from the copy's inputs to its outputs only (its late outputs, which
    chain into other classes' evaluations within a frame); else every path (sources to sinks)."""
    order = topo(g)
    NEG = -10 ** 9
    dist = [0] * len(g.kind)

    def hop(n, v):
        d = g.nets[v].driver
        if is_source(g, d):
            base = 0 if (not io or g.kind[d] == 'in') else NEG
        else:
            base = dist[d]
        return base + (cost(v) if P[d] >= 0 and P[n] >= 0 and P[d] != P[n] else 0)
    for n in order:
        if not is_source(g, n):
            dist[n] = max((hop(n, v) for v in deps(g, n)), default=0 if not io else NEG)
    sinks = [n for n in range(len(g.kind)) if g.kind[n] == 'out'] if io else range(len(g.kind))
    return max(0, max((hop(n, v) for n in sinks for v in g.ins[n]), default=0))


def loads(g):
    """Estimated LUT loads per bit of each net: a narrow (control) bit drives every LUT of a user, a data bit its
    share; a memory's address bits every LUTRAM of it."""
    out, per = {}, {}
    inw = [sum(g.nets[v].width for v in set(vs)) for vs in g.ins]
    for v, nt in g.nets.items():
        t = 0.0
        for u in set(nt.users):
            c = g.res[u]['lut'] + (g.res[u]['lutram'] if g.kind[u] == 'mem' else 0.0)
            if g.kind[u] == 'mem' and v in g.rd_ins.get(u, ()):
                x = max(1.0, g.res[u]['lutram'])
            elif nt.width <= 8:
                x = max(1.0, c)
            else:
                x = max(1.0, c / max(1, inw[u]))
            per[(v, u)] = x
            t += x
        out[v] = t
    return out, per


def fmtres(x):
    return ', '.join(f'{r} {x[r]:,.0f}' if r not in ('bram', 'uram') else f'{r} {x[r]:,.1f}' for r in RES)


def main():
    global OUTS, OUTDIR, LOADS, LOADS_U, HIGH, LINK_INSTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gen')
    ap.add_argument('--copy', required=True, help='the class copy (its module <copy>_static)')
    ap.add_argument('--model', action='store_true',
                    help='the pre-fold class model instead (the input an MLIR of the models, e.g. models.mlir; the '
                         'module <copy>): the split before folding, each part then folded with its own stores')
    ap.add_argument('--parts', default='2,3')
    ap.add_argument('--slr-region', default='v80')
    ap.add_argument('--slrs', help='the SLRs the parts go to, in chain order (default: the largest k, adjacent)')
    ap.add_argument('--global', dest='glob', default=r'^(state|inst)\d*$|^reset$|_reset$',
                    help='regex: the copy inputs that are global fold control (replicable)')
    ap.add_argument('--broadcast', type=int, default=256, help='pins of a registered net counted a broadcast')
    ap.add_argument('--comb-penalty', type=int, default=16)
    ap.add_argument('--imbalance', type=float, default=0.03)
    ap.add_argument('--serialize', default='1,2,4')
    ap.add_argument('--sll-json', help='a foldsim_sll_check --json of the generation: its links and simulator-level '
                                       'SLLs per boundary added to the cut')
    ap.add_argument('--variants', default='a,b,c')
    ap.add_argument('--top-fanout', type=int, default=12)
    ap.add_argument('--apply-map', metavar='FILE',
                    help='write the map foldsim-slr-split applies (the --variants variant at the first --parts count; '
                         'word level, --chunk 0): one part per op of the copy\'s body, -1 replicated')
    ap.add_argument('--chunk', type=int, default=32, help='bits a chunk of a value (0: word level: every value and '
                                                           'op whole)')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='.')
    ap.add_argument('--json')
    a = ap.parse_args()
    a.serialize = [int(x) for x in a.serialize.split(',')]
    OUTDIR = os.path.abspath(a.out)
    os.makedirs(OUTDIR, exist_ok=True)
    sim = a.gen
    if os.path.isdir(sim):
        sim = os.path.join(sim, 'gen', 'foldsim_sim.mlir') if os.path.isdir(os.path.join(sim, 'gen')) else \
            os.path.join(sim, 'foldsim_sim.mlir')
    region = load_region(a.slr_region)
    caps_all = fsc.slr_caps(region)
    t0 = time.time()
    top = a.copy if a.model else f'{a.copy}_static'
    texts = module_texts(sim, top)
    g = build(texts, top, region, a.model)
    links = g.link_insts
    if a.chunk:
        names = {g.name[n] for n in links}
        g = chunk_graph(g, a.chunk)
        links = {n for n in range(len(g.kind)) if g.kind[n] == 'inst' and g.name[n] in names}
    LINK_INSTS = links
    if a.model:
        # Other classes' instances: no resources of this copy.
        for n in links:
            g.res[n] = dict.fromkeys(RES, 0.0)
    OUTS = outs_of(g)
    classify(g)
    glob_re = re.compile(a.glob)
    rep = replicable(g, glob_re)
    lines = [f'{a.copy} in {sim}: {len(g.kind):,} nodes, {len(g.nets):,} nets, parsed in {time.time() - t0:.0f} s']
    tot = {r: sum(x[r] for x in g.res) for r in RES}
    lines.append('  copy (placed): ' + fmtres(tot))
    cls_bits = defaultdict(int)
    cls_nets = defaultdict(int)
    for nt in g.nets.values():
        cls_bits[nt.cls] += nt.width
        cls_nets[nt.cls] += 1
    lines.append('  nets by class: ' + ', '.join(f'{c} {cls_nets[c]:,} nets / {cls_bits[c]:,} bits'
                                                 for c in ('reg', 'memread', 'comb', 'port')))
    kinds = defaultdict(lambda: dict.fromkeys(RES, 0.0))
    for n in range(len(g.kind)):
        for r in RES:
            kinds[g.kind[n]][r] += g.res[n][r]
    for kk in ('op', 'reg', 'mem', 'inst'):
        lines.append(f'    {kk}: ' + fmtres(kinds[kk]))
    repn = [n for n in range(len(g.kind)) if rep[n] and g.kind[n] != 'in']
    rr = dict.fromkeys(RES, 0.0)
    for n in repn:
        for r in RES:
            rr[r] += g.res[n][r]
    regs = [n for n in repn if g.kind[n] == 'reg']
    rep_in = [g.name[n] for n in range(len(g.kind)) if rep[n] and g.kind[n] == 'in']
    lines.append(f'  replicable control: {len(regs):,} registers ({sum(g.res[n]["ff"] for n in regs):,.0f} placed '
                 f'flops), {sum(1 for n in repn if g.kind[n] == "mem")} memories, '
                 f'{sum(1 for n in repn if g.kind[n] == "op")} ops; ' + fmtres(rr)
                 + f'; global inputs {",".join(rep_in)}')
    rep_named = defaultdict(int)
    for n in regs:
        rep_named[re.sub(r'[_\d]+$', '', re.sub(r'\d+', '#', g.name[n]))[:60]] += 1
    lines.append('    largest groups: ' + '; '.join(f'{k} x{v}' for k, v in
                                                   sorted(rep_named.items(), key=lambda x: -x[1])[:8]))
    fan = sorted(((len(nt.users), v) for v, nt in g.nets.items()), reverse=True)
    bcast = [v for f, v in fan if f >= a.broadcast and g.nets[v].cls == 'reg' and not rep[g.nets[v].driver]
             and g.kind[g.nets[v].driver] == 'reg']
    lines.append(f'  broadcasts (registered, >= {a.broadcast} pins, not replicable): {len(bcast)} nets, '
                 f'{sum(g.nets[v].width for v in bcast):,} bits')
    LOADS, LOADS_U = loads(g)
    HIGH = sorted(LOADS, key=lambda v: -LOADS[v])[:a.top_fanout]
    lines.append('  highest-load nets (estimated LUT loads a bit; pins; class; R replicable): ' + '; '.join(
        f'{g.nets[v].name[1:][-70:]} {LOADS[v]:,.0f} ({len(g.nets[v].users)} pins, {g.nets[v].width}b, '
        f'{g.nets[v].cls}{", R" if rep[g.nets[v].driver] else ""})' for v in HIGH[:10]))
    sll_rest = None
    if a.sll_json:
        d = json.load(open(a.sll_json))
        # The design's other crossings: links and the simulator level (the cut copies' Rent term replaced).
        sll_rest = [x[0] + x[2] for x in d['slls']]
    reports = []
    for k in [int(x) for x in a.parts.split(',')]:
        if a.slrs:
            names = a.slrs.split(',')[:k]
        else:
            # The k largest SLRs adjacent in the chain.
            best = max(range(len(caps_all) - k + 1), key=lambda i: sum(c['lut'] for c in caps_all[i:i + k]))
            names = [c['name'] for c in caps_all[best:best + k]]
        caps = [next(c for c in caps_all if c['name'] == nm) for nm in names]
        rest = None
        if sll_rest:
            pos = [next(i for i, c in enumerate(caps_all) if c['name'] == nm) for nm in names]
            rest = [sll_rest[min(pos[b], pos[b + 1])] for b in range(k - 1)]
        for var in a.variants.split(','):
            r = analyse(g, rep, bcast, var, k, caps, names, a, rest)
            reports.append(r)
            lines.append(f'  {k}-way {var} ({",".join(names)}; {r["nodes"]:,} nodes, {r["nets"]:,} nets, '
                         f'{r["seconds"]} s):')
            for p, x in enumerate(r['part_res']):
                c = caps[p]
                lines.append(f'    part {p} -> {names[p]}: ' + fmtres(x) + '; of the SLR\'s cap: ' +
                             ', '.join(f'{rr2} {100 * x[rr2] / c[rr2]:.0f} %' for rr2 in RES if c[rr2]))
            lines.append(f'    cut: {r["cut_nets"]:,} nets, {r["cut_bits"]:,} bits; new stage registers '
                         f'{r["newreg_nets"]:,} nets / {r["newreg_bits"]:,} bits; added latency L = '
                         f'{r["latency"]} host cycles ({r["latency_io"]} on input-to-output paths); Versal TX/RX '
                         f'model L = {r["latency_txrx"]} ({r["latency_txrx_io"]} input-to-output); replication adds '
                         + fmtres(r['added']))
            lines.append('    fanout (loads a bit before -> the most one part drives): ' + '; '.join(
                f'{n[1:][-45:]} {b:,.0f} -> {x:,.0f}{" R" if rr_ else ""}' for n, b, x, rr_, _ in r['fanout'][:8]))
            for b in r['boundaries']:
                lines.append(f'    boundary {b["name"]}: {b["nets"]:,} nets, {b["bits"]:,} bits ('
                             + ', '.join(f'{c2} {v2:,}' for c2, v2 in sorted(b['by_cls'].items()))
                             + f'), new registers {b["newreg_bits"]:,} bits; SLLs '
                             + ', '.join(f'k={kk}: {v2:,.0f}' for kk, v2 in b['slls'].items())
                             + (f' (incl. the design\'s other {b["rest"]:,.0f})' if b['rest'] else ''))
            lines.append('    widest cuts: ' + '; '.join(f'{n[1:][-50:]} {w}b {c2} F{f}'
                                                         for n, w, c2, f in r['widest_cuts'][:6]))
            print('\n'.join(lines[-(len(r['part_res']) + 4 + len(r['boundaries'])):]), flush=True)
    if a.apply_map:
        if a.chunk:
            sys.exit('--apply-map needs --chunk 0 (the pass splits whole ops)')
        var, k = a.variants.split(',')[0], int(a.parts.split(',')[0])
        m = op_map(g, PARTS[var, k])
        with open(a.apply_map, 'w') as f:
            f.write('\n'.join(str(x) for x in m) + '\n')
        lines.append(f'  map: {a.apply_map} ({var}, {k} parts): {len(m):,} ops, '
                     + ', '.join(f'part {p} {sum(1 for x in m if x == p):,}' for p in range(-1, k)))
    if 'r' in CLUSTERS:
        lines.append(f'  registered-only clusters: {CLUSTERS["r"][0]:,}, the largest {100 * CLUSTERS["r"][1]:.1f} % '
                     f'of the copy\'s weight')
    rep_path = os.path.join(OUTDIR, f'{a.copy}.split.txt')
    open(rep_path, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:12]))
    if a.json:
        json.dump(dict(copy=a.copy, sim=sim, total=tot, replicable=rr, broadcasts=len(bcast), reports=reports),
                  open(a.json, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
