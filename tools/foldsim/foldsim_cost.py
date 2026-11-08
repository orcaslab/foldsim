"""Pre-synthesis LUT, flip-flop and memory estimate of the modules of an IR.

A per-operation LUT6 model (an estimate, not synthesis): `parse` reads the
`hw.module`s of an MLIR file; `mod_totals` sums a module and its children,
black boxes by the table below. Memories map to distributed RAM or block RAM by
depth and read latency, as Vivado infers them (`mem_cost`). The fold-degree
search calibrates these per class against Vivado (foldsim_degree.py).
"""
import json, math, re, sys
from collections import defaultdict

# A wide constant's decimal text (hw.constant of a 16,861-bit word: 5,076
# digits in the A100-class models at 64 SMs) is past Python's default limit
# on int() of a string (4,300 digits).
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

SSA = r'%[\w.$-]+'
FREE = {'comb.extract', 'comb.concat', 'hw.bitcast', 'hw.struct_extract', 'hw.struct_create',
        'hw.array_create', 'hw.wire', 'comb.replicate', 'hw.array_concat', 'hw.struct_explode',
        'hw.union_create', 'hw.union_extract', 'hw.constant', 'hw.aggregate_constant',
        'seq.from_clock', 'seq.const_clock', 'seq.to_clock', 'hw.output', 'hw.array_slice',
        'hw.struct_inject', 'sv.verbatim', 'sv.attribute'}
CONSTS = {'hw.constant', 'hw.aggregate_constant'}
REGS = {'seq.firreg', 'seq.compreg', 'seq.compreg.ce'}
# Xilinx floating-point cores (black boxes in the IR): LUTs, FFs, DSPs per instance,
# measured in the I6UE synthesis (fill_blackbox_costs) or these V80 defaults.
BLACKBOX = {'xil_fma': (850, 1400, 2), 'xil_fdiv': (800, 1400, 0), 'xil_fsqrt': (500, 900, 0),
            'xil_fmul': (90, 260, 2), 'xil_fadd': (260, 500, 0)}


def split_top(s, sep=','):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in '<([{':
            depth += 1
        elif ch in '>)]}':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(''.join(cur)); cur = []
        else:
            cur.append(ch)
    if cur:
        out.append(''.join(cur))
    return out


_wcache = {}
def width(t):
    """Bit width of an MLIR hw type string."""
    t = t.strip()
    if t in _wcache:
        return _wcache[t]
    w = 0
    m = re.fullmatch(r'i(\d+)', t)
    if m:
        w = int(m.group(1))
    elif t in ('!seq.clock', 'clock'):
        w = 1
    elif t.startswith('!hw.array<') or t.startswith('!hw.uarray<'):
        inner = t[t.index('<') + 1:-1]
        n, el = inner.split('x', 1)
        w = int(n) * width(el)
    elif t.startswith('!hw.struct<'):
        inner = t[len('!hw.struct<'):-1]
        w = sum(width(f.split(':', 1)[1]) for f in split_top(inner) if ':' in f)
    elif t.startswith('!hw.union<'):
        inner = t[len('!hw.union<'):-1]
        w = max([width(f.split(':', 1)[1]) for f in split_top(inner) if ':' in f] or [0])
    elif t.startswith('!hw.typealias<'):
        inner = t[len('!hw.typealias<'):-1]
        w = width(split_top(inner)[-1])
    _wcache[t] = w
    return w


def result_type(rhs):
    """The trailing ': T' of an op, at depth 0."""
    parts = split_top(rhs, ':')
    return parts[-1].strip() if len(parts) > 1 else ''


class Mod:
    def __init__(self, name, ports):
        self.name = name
        self.ins, self.outs = 0, 0
        for p in split_top(ports):
            p = p.strip()
            m = re.match(r'(in|out|inout)\s+("?[^:"]*"?|' + SSA + r')\s*:\s*(.*)$', p)
            if m:
                w = width(m.group(3))
                if m.group(1) == 'in':
                    self.ins += w
                else:
                    self.outs += w
        self.lut = defaultdict(float)   # by op class
        self.ff = 0
        self.dsp = 0
        self.mems = []                  # (name, depth, width, read ports, write ports, read latency,
                                        #  ram_style, mask bits, named)
        self.regw = defaultdict(int)    # register width -> registers
        self.insts = []                 # (child module, instance name)
        self.consts = set()
        self.Z, self.O, self.W = {}, {}, {}   # known-zero / known-one bits, widths of int values
        self.P = {}                           # bit provenance of wide values: [(length, src, src_lo)]
        self.regs, self.muxes = {}, {}        # register -> next value; mux -> (cost, a, b)
        self.uses = defaultdict(int)          # value -> its uses
        self.ands, self.ors = {}, []          # and -> (cost, operands); or ops (result, operands, cost, width)
        self.shifts, self.extracts = {}, defaultdict(list)   # shift -> (cost, width, op); value -> [(low, width)]
        self.ids = {}

    def sid(self, v):
        return self.ids.setdefault(v, len(self.ids) + 1)


PROV_MIN = 16      # provenance is tracked for values at least this wide


def prov(mod, v, w):
    return mod.P.get(v) or [(w, mod.sid(v), 0)]


def prov_slice(runs, lo, n):
    out, pos = [], 0
    for ln, s, sl in runs:
        a, b = max(lo, pos), min(lo + n, pos + ln)
        if a < b:
            out.append((b - a, s, sl + a - pos))
        pos += ln
        if pos >= lo + n:
            break
    return out


def prov_bits(runs):
    out = []
    for ln, s, sl in runs:
        out.extend((s << 24) | (sl + i) for i in range(ln))
    return out


def prov_runs(bits):
    runs = []
    for b in bits:
        s, sl = b >> 24, b & 0xffffff
        if runs and runs[-1][1] == s and runs[-1][2] + runs[-1][0] == sl:
            runs[-1] = (runs[-1][0] + 1, s, runs[-1][2])
        else:
            runs.append((1, s, sl))
    return runs


def mux_diff_bits(mod, a, b, w):
    """Bits where a mux's two data inputs can differ."""
    full = (1 << w) - 1
    same = (mod.Z.get(a, 0) & mod.Z.get(b, 0)) | (mod.O.get(a, 0) & mod.O.get(b, 0))
    if w < PROV_MIN or (a not in mod.P and b not in mod.P):
        return w - popcount(same & full)
    pa, pb = prov_bits(prov(mod, a, w)), prov_bits(prov(mod, b, w))
    n = 0
    for i in range(w):
        if pa[i] != pb[i] and not (same >> i) & 1:
            n += 1
    return n


def int_width(t):
    m = re.fullmatch(r'\s*i(\d+)\s*', t)
    return int(m.group(1)) if m else None


def track(mod, op, rhs, res, operands):
    """Propagate known-constant bits through wiring and bitwise ops."""
    if len(res) != 1:
        return
    r = res[0]
    t = result_type(rhs)
    if '->' in t:
        t = t.split('->')[1]
    if op == 'comb.concat':
        ws = [int_width(x) for x in split_top(t)]
        if None in ws or len(ws) != len(operands):
            return
        z = o = 0
        for v, w in zip(operands, ws):
            z = (z << w) | (mod.Z.get(v, 0) & ((1 << w) - 1))
            o = (o << w) | (mod.O.get(v, 0) & ((1 << w) - 1))
        w = sum(ws)
    else:
        w = int_width(t)
        if w is None:
            return
        full = (1 << w) - 1
        if op == 'hw.constant':
            tok = rhs.split()[1]
            v = 1 if tok == 'true' else 0 if tok == 'false' else int(tok)
            v &= full
            z, o = full & ~v, v
        elif op == 'comb.extract':
            lo = int(re.search(r'from (\d+)', rhs).group(1))
            a = operands[0]
            z, o = (mod.Z.get(a, 0) >> lo) & full, (mod.O.get(a, 0) >> lo) & full
        elif op == 'comb.replicate':
            a = operands[0]
            wa = int_width(result_type(rhs).split('->')[0].strip('() '))
            if not wa:
                return
            z = o = 0
            for _ in range(w // wa):
                z = (z << wa) | mod.Z.get(a, 0)
                o = (o << wa) | mod.O.get(a, 0)
        elif op == 'comb.and':
            z, o = 0, full
            for a in operands:
                z |= mod.Z.get(a, 0); o &= mod.O.get(a, 0)
        elif op == 'comb.or':
            z, o = full, 0
            for a in operands:
                z &= mod.Z.get(a, 0); o |= mod.O.get(a, 0)
        elif op == 'comb.xor':
            known, val = full, 0
            for a in operands:
                known &= mod.Z.get(a, 0) | mod.O.get(a, 0); val ^= mod.O.get(a, 0)
            z, o = known & ~val & full, known & val
        elif op == 'comb.mux' and len(operands) == 3:
            c, a, b = operands
            if c in mod.O and mod.O[c] & 1:
                z, o = mod.Z.get(a, 0), mod.O.get(a, 0)
            elif c in mod.Z and mod.Z[c] & 1:
                z, o = mod.Z.get(b, 0), mod.O.get(b, 0)
            else:
                z, o = mod.Z.get(a, 0) & mod.Z.get(b, 0), mod.O.get(a, 0) & mod.O.get(b, 0)
        elif op in ('comb.shl', 'comb.shru') and len(operands) == 2 and operands[1] in mod.consts:
            s = mod.O.get(operands[1], 0)
            a = operands[0]
            if s >= w:
                z, o = full, 0
            elif op == 'comb.shl':
                z, o = ((mod.Z.get(a, 0) << s) | ((1 << s) - 1)) & full, (mod.O.get(a, 0) << s) & full
            else:
                z = (mod.Z.get(a, 0) >> s) | (full ^ (full >> s))
                o = mod.O.get(a, 0) >> s
        elif op == 'hw.bitcast' and operands and int_width(result_type(rhs).split('->')[0].strip('() ')) == w:
            z, o = mod.Z.get(operands[0], 0), mod.O.get(operands[0], 0)
        else:
            return
    if z:
        mod.Z[r] = z
    if o:
        mod.O[r] = o
    if w >= PROV_MIN:
        p = track_prov(mod, op, rhs, operands, w, ws if op == 'comb.concat' else None)
        if p is not None and not (len(p) == 1 and p[0][1] == mod.ids.get(r)):
            mod.P[r] = p


def track_prov(mod, op, rhs, operands, w, ws):
    """Bit provenance of a wide result: which source bit each bit passes through from."""
    if op == 'comb.concat':
        runs = []
        for v, wv in reversed(list(zip(operands, ws))):
            runs.extend(prov(mod, v, wv))
        return runs
    if op == 'comb.extract':
        lo = int(re.search(r'from (\d+)', rhs).group(1))
        wa = int_width(result_type(rhs).split('->')[0].strip('() '))
        return prov_slice(prov(mod, operands[0], wa), lo, w) if wa else None
    if op == 'hw.bitcast':
        return prov(mod, operands[0], w)
    if op in ('comb.and', 'comb.or'):
        full = (1 << w) - 1
        ident = mod.O if op == 'comb.and' else mod.Z     # the identity element's bits
        live = [(v, full & ~ident.get(v, 0)) for v in operands]
        live = [(v, l) for v, l in live if l]
        if len(live) == 1:
            return prov(mod, live[0][0], w)
        if len(live) == 0:
            return None
        cover = 0
        for _, l in live:
            cover |= l
        multi = 0
        seen = 0
        for _, l in live:
            multi |= seen & l
            seen |= l
        bits = [None] * w
        for v, l in live:
            single = l & ~multi
            if single:
                pb = prov_bits(prov(mod, v, w))
                for i in range(w):
                    if (single >> i) & 1:
                        bits[i] = pb[i]
        own = mod.sid('%%own%d' % len(mod.ids))
        bits = [b if b is not None else (own << 24) | i for i, b in enumerate(bits)]
        return prov_runs(bits)
    if op == 'comb.mux' and len(operands) == 3:
        c, a, b = operands
        if c in mod.O and mod.O[c] & 1:
            return prov(mod, a, w)
        if c in mod.Z and mod.Z[c] & 1:
            return prov(mod, b, w)
        if a not in mod.P and b not in mod.P:
            return None
        pa, pb = prov_bits(prov(mod, a, w)), prov_bits(prov(mod, b, w))
        own = mod.sid('%%own%d' % len(mod.ids))
        return prov_runs([x if x == y else (own << 24) | i for i, (x, y) in enumerate(zip(pa, pb))])
    return None


def popcount(x):
    return bin(x).count('1')


def tree_cost(counts):
    """LUT6s of a reduction per bit, given each bit's count of live operands."""
    return sum(math.ceil((k - 1) / 5) for k in counts if k >= 2)


def bit_counts(w, live):
    if len(live) == 2:
        return None
    return [sum((l >> b) & 1 for l in live) for b in range(w)]


def lut_cost(op, rhs, mod, operands):
    """LUT6 estimate of one op; known-constant bits (mod.Z/mod.O) cost nothing."""
    t = result_type(rhs)
    nonconst = [o for o in operands if o not in mod.consts]
    if op in ('comb.and', 'comb.or', 'comb.xor'):
        w = width(t)
        full = (1 << w) - 1
        Z = [mod.Z.get(o, 0) for o in operands]
        O = [mod.O.get(o, 0) for o in operands]
        if op == 'comb.or':
            forced = 0
            for o in O:
                forced |= o
            live = [full & ~z & ~forced for z in Z]
        elif op == 'comb.and':
            forced = 0
            for z in Z:
                forced |= z
            live = [full & ~o & ~forced for o in O]
        else:
            live = [full & ~(z | o) for z, o in zip(Z, O)]
        live = [l for l in live if l]
        if len(live) <= 1:
            return 0
        if len(live) == 2:
            return popcount(live[0] & live[1])
        return tree_cost(bit_counts(w, live))
    if op == 'comb.mux':
        if operands and operands[0] in mod.consts:
            return 0
        w = width(t)
        if len(operands) == 3:
            a, b = operands[1], operands[2]
            if a == b:
                return 0
            if int_width(t) is None:
                return w * 0.5
            return mux_diff_bits(mod, a, b, w) * 0.5
        return w * 0.5
    if op == 'comb.icmp':
        w = width(t)
        pred = rhs.split()[1]
        if pred in ('eq', 'ne', 'ceq', 'cne', 'weq', 'wne'):
            per = 6 if len(nonconst) < 2 else 3
            return math.ceil(w / per) + math.ceil(w / per / 6)
        return math.ceil(w / 2)
    if op in ('comb.add', 'comb.sub'):
        return width(t)
    if op == 'comb.mul':
        w = width(t)
        if len(nonconst) < 2:
            return w
        if w <= 8:
            return w * w / 2
        mod.dsp += math.ceil(w / 18) ** 2 if w > 18 else 1
        return 0
    if op in ('comb.divu', 'comb.divs', 'comb.modu', 'comb.mods'):
        w = width(t)
        return w * w if len(nonconst) >= 2 else w * 2
    if op in ('comb.shl', 'comb.shru', 'comb.shrs'):
        w = width(t)
        if len(nonconst) < 2:
            return 0
        amt = operands[1]
        full = (1 << w) - 1
        k = popcount(full & ~(mod.Z.get(amt, 0) | mod.O.get(amt, 0)))
        k = min(k, max(1, math.ceil(math.log2(max(w, 2)))))
        val = operands[0]
        live = w - popcount(full & (mod.Z.get(val, 0) | mod.O.get(val, 0)))
        return max(live, 1) * max(1, math.ceil(k / 2))
    if op == 'comb.parity':
        return math.ceil(width(t) / 5)
    if op == 'hw.array_get':
        m = re.search(r'!hw\.array<(\d+)x(.*)>\s*,', rhs)
        idx = re.findall(r'\[(' + SSA + r')\]', rhs)
        if not m or (idx and idx[0] in mod.consts):
            return 0
        n, w = int(m.group(1)), width(m.group(2))
        return w * (n - 1) / 3
    if op == 'hw.array_inject':
        m = re.search(r'!hw\.array<(\d+)x(.*)>\s*,', rhs)
        idx = re.findall(r'\[(' + SSA + r')\]', rhs)
        if not m:
            return 0
        n, w = int(m.group(1)), width(m.group(2))
        return 0 if (idx and idx[0] in mod.consts) else n * w * 0.5
    return None


def parse(path):
    mods = {}
    cur = None
    hdr = re.compile(r'^\s*hw\.module(?:\.extern)?(?:\s+private)?\s+@([\w$.]+)(?:<[^>]*>)?\((.*)\)\s*(?:->\s*\((.*)\))?\s*(?:attributes.*)?\{?\s*$')
    mems = {}
    unknown = defaultdict(int)
    for line in open(path):
        s = line.strip()
        if s.startswith('hw.module'):
            m = hdr.match(line)
            if m:
                cur = Mod(m.group(1), m.group(2) + (',' + m.group(3) if m.group(3) else ''))
                mods[cur.name] = cur
                mems = {}
                if 'hw.module.extern' in s:
                    cur = None
            continue
        if cur is None:
            continue
        if s.startswith('hw.output'):
            for v in re.findall(SSA, s):
                cur.uses[v] += 1
            refund_enables(cur)
            pack_and_or(cur)
            trim_shifts(cur)
            cur = None
            continue
        if ' = ' not in s:
            for v in re.findall(SSA, s.split(' : ')[0]):
                cur.uses[v] += 1
        if s.startswith('seq.firmem.write_port') or s.startswith('seq.firmem.read_write_port'):
            mm = re.search(r'(' + SSA + r')\[', s)
            if mm and mm.group(1) in mems:
                mems[mm.group(1)][4] += 1
            continue
        if ' = ' not in s or not s.startswith('%'):
            continue
        lhs, rhs = s.split(' = ', 1)
        op = rhs.split()[0]
        res = re.findall(SSA, lhs)
        operands = re.findall(SSA, rhs.split(' : ')[0])
        for v in operands:
            cur.uses[v] += 1
        if op in CONSTS:
            cur.consts.update(res)
            if op == 'hw.constant':
                track(cur, op, rhs, res, operands)
            continue
        if op in FREE:
            if op == 'comb.extract' and operands:
                lo = re.search(r'from (\d+)', rhs)
                ew = int_width(result_type(rhs).split('->')[-1])
                if lo and ew:
                    cur.extracts[operands[0]].append((int(lo.group(1)), ew))
            track(cur, op, rhs, res, operands)
            continue
        if op in REGS:
            cur.ff += width(result_type(rhs))
            cur.regw[width(result_type(rhs))] += 1
            if operands:
                cur.regs[res[0]] = operands[0]
            continue
        if op == 'seq.firmem':
            mm = re.search(r'(\d+),\s*(\d+),\s*\w+,\s*\w+.*<\s*(\d+)\s*x\s*(\d+)', rhs)
            nm = re.search(r'name "([^"]*)"', rhs)
            st = re.search(r'ram_style" = "\\22(\w+)', rhs)
            mk = re.search(r'mask\s+(\d+)\s*>', rhs)
            rec = [nm.group(1) if nm else res[0][1:], int(mm.group(3)), int(mm.group(4)), 0, 0, int(mm.group(1)),
                   st.group(1) if st else None, int(mk.group(1)) if mk else 1, bool(nm)]
            mems[res[0]] = rec
            cur.mems.append(rec)
            continue
        if op == 'seq.firmem.read_port':
            mm = re.search(r'(' + SSA + r')\[', rhs)
            if mm and mm.group(1) in mems:
                mems[mm.group(1)][3] += 1
            continue
        if op == 'hw.instance':
            mm = re.search(r'hw\.instance\s+"([^"]*)"\s+(?:sym\s+@\S+\s+)?@([\w$.]+)', rhs)
            if mm:
                cur.insts.append((mm.group(2), mm.group(1)))
            continue
        c = lut_cost(op, rhs, cur, operands)
        track(cur, op, rhs, res, operands)
        if c is None:
            unknown[op] += 1
            continue
        cur.lut[op.split('.')[1]] += c
        if op in ('comb.shl', 'comb.shru', 'comb.shrs') and c and len(res) == 1:
            cur.shifts[res[0]] = (c, int_width(result_type(rhs)) or 0, op.split('.')[1])
        if op == 'comb.and' and c and len(res) == 1:
            cur.ands[res[0]] = (c, [o for o in operands if o not in cur.consts])
        elif op == 'comb.or' and c and len(res) == 1:
            cur.ors.append((res[0], [o for o in operands if o not in cur.consts], c, int_width(result_type(rhs)) or 0))
        if op == 'comb.mux' and len(operands) == 3 and c:
            cur.muxes[res[0]] = (c, operands[1], operands[2])
    return mods, unknown


def pack_and_or(mod):
    """An AND whose only use is an OR (a sum of products: a one-hot select's
    `data & sel` terms ORed) is not a LUT of its own: synthesis packs the OR's
    terms and their AND inputs into one LUT6 tree per bit, ceil((n - 1) / 5)
    LUTs for n inputs, where foldsim_cost counted a LUT a bit per AND and the
    OR's tree apart (FSD's ToR, 51 % `and` and 15 % `or`, synthesized at about
    half its count)."""
    for r, ops, c_or, w in mod.ors:
        absorbed = [o for o in ops if o in mod.ands and mod.uses[o] == 1]
        if not absorbed or not w:
            continue
        n = sum(len(mod.ands[o][1]) for o in absorbed) + len(ops) - len(absorbed)
        old = c_or + sum(mod.ands[o][0] for o in absorbed)
        new = w * math.ceil((n - 1) / 5)
        if new < old:
            mod.lut['and'] -= sum(mod.ands[o][0] for o in absorbed)
            mod.lut['or'] += new - c_or
    mod.ands, mod.ors = {}, []


def trim_shifts(mod):
    """A shift whose only uses are extracts builds only the bits they take
    (a paged row shifted down and its low word taken: synthesis keeps that
    word's muxes, not the whole row's): its cost in the share of its result
    bits the extracts cover."""
    for r, (c, w, cls) in mod.shifts.items():
        ex = mod.extracts.get(r)
        if not w or not ex or mod.uses[r] != len(ex):
            continue
        bits = set()
        for lo, ew in ex:
            bits.update(range(lo, min(w, lo + ew)))
        if len(bits) < w:
            mod.lut[cls] -= c * (1 - len(bits) / w)
    mod.shifts, mod.extracts, mod.uses = {}, defaultdict(list), defaultdict(int)


def refund_enables(mod):
    """A mux that holds a register (one data input is the register itself) is its clock
    enable: no LUTs. Nested holds (c1 ? (c2 ? x : r) : r) are enables too."""
    done = set()
    for r, nxt in mod.regs.items():
        v = nxt
        for _ in range(6):
            m = mod.muxes.get(v)
            if not m or v in done:
                break
            c, a, b = m
            if r not in (a, b):
                break
            mod.lut['mux'] -= c
            done.add(v)
            v = b if a == r else a


def mem_cost(depth, w, rports, rlat, style=None):
    """(LUTRAM LUTs, BRAM36) of one memory. Synchronous reads from 32 entries up go to
    block RAM, as Vivado infers them, unless the memory asks otherwise."""
    rports = max(rports, 1)
    bram = style == 'block' or (style != 'distributed' and rlat >= 1 and depth >= 32 and depth * w >= 2048)
    if bram:
        if w <= 36 and depth <= 512:
            per = 0.5
        else:
            per = math.ceil(w / 72) * math.ceil(depth / 512)
        return 0, per * max(1, math.ceil(rports / 2))
    if depth <= 32:
        return w * 0.57 * rports, 0
    return w * math.ceil(depth / 64) * 1.14 * rports + (w * (math.ceil(depth / 64) - 1) / 3 if depth > 64 else 0), 0


def mod_totals(mods, name, stop=(), mult=1, acc=None):
    """Aggregate a module and its non-stop children."""
    if acc is None:
        acc = dict(lut=0.0, ff=0, dsp=0, membits=0, lutram=0.0, bram=0.0, bb_lut=0, bb_ff=0, bb_dsp=0)
    m = mods[name]
    acc['lut'] += mult * sum(m.lut.values())
    acc['ff'] += mult * m.ff
    acc['dsp'] += mult * m.dsp
    for (_, d, w, r, wr, rl, st, *_) in m.mems:
        lr, br = mem_cost(d, w, r, rl, st)
        acc['membits'] += mult * d * w
        acc['lutram'] += mult * lr
        acc['bram'] += mult * br
    for child, _ in m.insts:
        if child in stop:
            continue
        base = child.split('__')[0]
        if child in BLACKBOX or base in BLACKBOX:
            l, f, d = BLACKBOX.get(child, BLACKBOX.get(base))
            acc['bb_lut'] += mult * l; acc['bb_ff'] += mult * f; acc['bb_dsp'] += mult * d
            continue
        if child in mods:
            mod_totals(mods, child, stop, mult, acc)
    return acc
