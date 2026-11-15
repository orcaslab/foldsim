"""Pre-synthesis LUT, flip-flop and memory estimate of the modules of an IR.

A per-operation LUT6 model (an estimate, not synthesis): `parse` reads the
`hw.module`s of an MLIR file; `mod_totals` sums a module and its children,
black boxes by the table below. Memories map to distributed RAM or block RAM by
depth and read latency, as Vivado infers them (`mem_cost`). The fold-degree
search calibrates these per class against Vivado (foldsim_degree.py).
"""
import json, math, os, re, sys
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
# Xilinx floating-point cores (black boxes in the IR): LUTs, FFs, DSPs per instance. xil_fma / xil_fdiv / xil_fsqrt
# synthesized (Vivado 2025.2, xcv80, OOC, the cores as hw/scripts/xilinx_ip_gen.tcl configures them: fma latency 16
# Medium_Usage, fdiv / fsqrt latency 28; a32-V1 11-11, uh_VX_{fma,fdiv,fsqrt}_unit_static.rpt: 966 / 1,476 / 897 LUTs,
# 1,084 / 1,389 / 841 FFs, 2 / 0 / 0 DSPs; were 850 / 800 / 500 defaults); xil_fmul / xil_fadd V80 defaults.
BLACKBOX = {'xil_fma': (966, 1084, 2), 'xil_fdiv': (1476, 1389, 0), 'xil_fsqrt': (897, 841, 0),
            'xil_fmul': (90, 260, 2), 'xil_fadd': (260, 500, 0)}


def split_top(s, sep=','):
    # The '>' of a function type's '->' closes nothing ((i6) -> !hw.struct<a: i1, b: i5> is one type, not split
    # at its fields' ':').
    out, depth, cur, prev = [], 0, [], ''
    for ch in s:
        if ch in '<([{':
            depth += 1
        elif ch in '>)]}' and not (ch == '>' and prev == '-'):
            depth -= 1
        prev = ch
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
        self.roms = {}                  # constant array (hw.aggregate_constant, through hw.wire) -> varying bits
        self.block_roms = set()         # those of them a rom_style "block" wire states
        self.rom_reads = {}             # a block ROM's read -> (its LUTs as priced, entries, width)
        self.bram = 0.0                 # block RAM ROMs (the fit count)
        self.count = False              # parse(..., count=True)
        self.cur_res = None             # the op being priced's result (lut_cost)
        self.Z, self.O, self.W = {}, {}, {}   # known-zero / known-one bits, widths of int values
        self.P = {}                           # bit provenance of wide values: [(length, src, src_lo)]
        self.regs, self.muxes = {}, {}        # register -> next value; mux -> (cost, a, b)
        self.uses = defaultdict(int)          # value -> its uses
        self.ands, self.ors = {}, []          # and -> (cost, operands); or ops (result, operands, cost, width)
        self.shifts, self.extracts = {}, defaultdict(list)   # shift -> (cost, width, op); value -> [(low, width)]
        self.cat, self.ext = {}, {}           # SLICED_HOLDS: concat -> [(operand, width)] low first; extract -> (src, low)
        self.ids = {}

    def sid(self, v):
        return self.ids.setdefault(v, len(self.ids) + 1)


PROV_MIN = 16      # provenance is tracked for values at least this wide
# FOLDSIM_COST_SLICED_HOLDS=1 (an option, not the default yet): a register whose next value is
# a concat of slices each holding the register's own bits at their place (mux(c, x, r[lo +: w]),
# as replicate-controls cuts a fetch buffer's load per replicated land strobe) is clock-enabled
# per slice: Vivado maps such a mux to the flops' CE, no LUTs (refund_enables). The fit count
# (parse(..., count=True)) has it on unless FOLDSIM_COST_SLICED_HOLDS=0 (COUNT_SLICED_HOLDS).
SLICED_HOLDS = os.environ.get('FOLDSIM_COST_SLICED_HOLDS', '') == '1'


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


def mux_tree(n):
    """LUT6s a bit of an n:1 select built as a tree of 4:1 LUT6 muxes (Versal's CLB has no MUXF7 / MUXF8: a 2:1
    select of two LUT outputs is a LUT of its own): n = 2..4 one, 8 three, 64 twenty-one ((n - 1) / 3 at powers of
    4, more between them: a32-UR 11-12, l2fs_4's 128-deep distributed RAMs: one LUT3 a bit for the two 64-row
    blocks' select, priced a third)."""
    t = 0
    while n > 1:
        n = math.ceil(n / 4)
        t += n
    return t


# The fit count (parse(..., roms=True): foldsim_fitdb.count, foldsim_sll_check) prices a constant array's read as a
# ROM; FOLDSIM_COST_ROM_ARRAYS=0 there, as a select of its entries (before 11-12). The compiler's own choices keep
# the select (parse's default).
ROM_ARRAYS = os.environ.get('FOLDSIM_COST_ROM_ARRAYS', '1') != '0'
# The fit count's per-slice holds (SLICED_HOLDS): on unless FOLDSIM_COST_SLICED_HOLDS=0 (kit/e1terms counted with
# them, the fit gate without: 21.6k LUTs apart on y64g34, a32-D1 11-11).
COUNT_SLICED_HOLDS = os.environ.get('FOLDSIM_COST_SLICED_HOLDS', '1') != '0'


def const_array_varying(rhs):
    """The bits not the same in every entry of a hw.aggregate_constant array of integers (`rhs` its op's text),
    or None if it is not one."""
    mm = re.search(r'!hw\.array<(\d+)xi(\d+)>\s*$', rhs)
    if not mm:
        return None
    w = int(mm.group(2))
    vals = re.findall(r'(-?\d+|true|false) : i\d+', rhs.split('] :')[0])
    if len(vals) != int(mm.group(1)):
        return w
    a, o = (1 << w) - 1, 0
    for v in vals:
        x = (1 if v == 'true' else 0 if v == 'false' else int(v)) & ((1 << w) - 1)
        a &= x
        o |= x
    return popcount(o & ~a)


def popcount(x):
    return bin(x).count('1')


def tree_cost(counts):
    """LUT6s of a reduction per bit, given each bit's count of live operands."""
    return sum(math.ceil((k - 1) / 5) for k in counts if k >= 2)


def bit_counts(w, live):
    if len(live) == 2:
        return None
    return [sum((l >> b) & 1 for l in live) for b in range(w)]


def reachable(mod, idx, n, rhs):
    """The elements of an n-element array an index can address: an index with known-constant bits (a lane
    index {pid, 4'd6}, say) reaches 2^(free bits) of them, the rest of a get's choice and of an inject's
    element selects constant-fold away."""
    if idx is None:
        return n
    m = re.search(r'\]\s*(?:,\s*' + SSA + r'\s*)?:\s*!hw\.array<\d+x.*>\s*,\s*i(\d+)\s*$', rhs)
    iw = int(m.group(1)) if m else max(1, math.ceil(math.log2(max(n, 2))))
    known = (mod.Z.get(idx, 0) | mod.O.get(idx, 0)) & ((1 << iw) - 1)
    free = iw - popcount(known)
    if not known:
        return n
    # the addresses with the known bits fixed, below n
    o = mod.O.get(idx, 0) & known
    cnt, bits = 0, [b for b in range(iw) if not (known >> b) & 1]
    if (1 << free) > 4 * n:
        return min(n, 1 << free)
    for k in range(1 << free):
        a = o
        for j, b in enumerate(bits):
            if (k >> j) & 1:
                a |= 1 << b
        cnt += a < n
    return max(1, cnt)


def live_width(mod, v, w):
    """The bits of a w-bit value below its highest bit not known zero (mod.Z)."""
    z = mod.Z.get(v, 0)
    for i in range(w - 1, -1, -1):
        if not (z >> i) & 1:
            return i + 1
    return 0


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
        if mod.count and len(operands) == 2:
            # The fit count: a product of its operands' live widths (their high bits known zero dropped: a
            # zero-extended or SIMD-packed operand), DSP58s of 27 x 24 signed (26 x 23 unsigned) each, at the
            # cheaper orientation (a32-UR 11-12, VX_tcu_fedp_tfr: 56 i17 / i22 / i26 products of 13 x 13, 11 x 11 and
            # 9 x 8 live bits, each one DSP58 in Vivado (40 for the 56: some merged), counted 176 by result width).
            a, b = live_width(mod, operands[0], w), live_width(mod, operands[1], w)
            if max(a, b) <= 8:
                return a * b / 2
            mod.dsp += min(math.ceil(a / 26) * math.ceil(b / 23), math.ceil(b / 26) * math.ceil(a / 23))
            return 0
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
        n = reachable(mod, idx[0] if idx else None, n, rhs)
        if operands and operands[0] in mod.roms:
            # A read of a constant array is a ROM (array_read_cost's 'const'), not a select of its entries
            # (a32-UR 11-12: VX_mufu_pe's 512 x 72 table counted 12.3k LUTs, synthesized under 1k); a registered
            # read of a rom_style "block" one a block RAM (block_roms).
            c = array_read_cost(n, w, 'const', varying=min(w, mod.roms[operands[0]]))
            if operands[0] in mod.block_roms and mod.cur_res:
                mod.rom_reads[mod.cur_res] = (c, n, w)
            return c
        return w * (n - 1) / 3
    if op == 'hw.array_inject':
        m = re.search(r'!hw\.array<(\d+)x(.*)>\s*,', rhs)
        idx = re.findall(r'\[(' + SSA + r')\]', rhs)
        if not m:
            return 0
        n, w = int(m.group(1)), width(m.group(2))
        return 0 if (idx and idx[0] in mod.consts) else reachable(mod, idx[0] if idx else None, n, rhs) * w * 0.5
    return None


def parse(path, count=False):
    """{module: Mod}, {unknown op: n} of an MLIR file. `count`: as the fit count prices a netlist (foldsim_fitdb.count,
    foldsim_sll_check): a constant array's read a ROM (ROM_ARRAYS), per-slice holds refunded (COUNT_SLICED_HOLDS)."""
    global SLICED_HOLDS
    saved = SLICED_HOLDS
    if count:
        SLICED_HOLDS = COUNT_SLICED_HOLDS
    try:
        return _parse(path, count and ROM_ARRAYS)
    finally:
        SLICED_HOLDS = saved


def _parse(path, roms):
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
                cur.count = roms
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
            block_roms(cur)
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
            elif res and roms:
                v = const_array_varying(rhs)
                if v is not None:
                    cur.roms[res[0]] = v
            continue
        if op == "hw.wire" and len(res) == 1 and operands and operands[0] in cur.roms:
            cur.roms[res[0]] = cur.roms[operands[0]]
            if re.search(r'rom_style" = "\\22block\\22', rhs):
                cur.block_roms.add(res[0])
        if op in FREE:
            if op == 'comb.extract' and operands:
                lo = re.search(r'from (\d+)', rhs)
                ew = int_width(result_type(rhs).split('->')[-1])
                if lo and ew:
                    cur.extracts[operands[0]].append((int(lo.group(1)), ew))
                    if SLICED_HOLDS and len(res) == 1:
                        cur.ext[res[0]] = (operands[0], int(lo.group(1)))
            elif SLICED_HOLDS and op == 'comb.concat' and len(res) == 1:
                ws = [int_width(x) for x in split_top(result_type(rhs).split('->')[-1])]
                if None not in ws and len(ws) == len(operands):
                    cur.cat[res[0]] = list(reversed(list(zip(operands, ws))))
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
            ch = re.search(r'cascade_height" = "(\d+)"', rhs)
            rec = [nm.group(1) if nm else res[0][1:], int(mm.group(3)), int(mm.group(4)), 0, 0, int(mm.group(1)),
                   st.group(1) if st else None, int(mk.group(1)) if mk else 1, bool(nm), int(ch.group(1)) if ch else None]
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
        cur.cur_res = res[0] if len(res) == 1 else None
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


def block_roms(mod):
    """A rom_style "block" ROM read straight into a register is a block RAM (Vivado absorbs the register as the
    RAM's output register: VX_mufu_pe's 512 x 72 table, one RAMB36, no LUTs): its LUTs refunded, its RAMB36 in
    mod.bram (bram_blocks)."""
    for r, nxt in mod.regs.items():
        if nxt in mod.rom_reads:
            c, n, w = mod.rom_reads.pop(nxt)
            mod.lut['array_get'] -= c
            mod.bram += bram_blocks(n, w)
    mod.rom_reads = {}


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
    if not SLICED_HOLDS:
        return
    # Per-slice holds: each slice of the next value a mux whose held input is the register's bits
    # at that slice's place (nested holds too).
    for r, nxt in mod.regs.items():
        parts = mod.cat.get(nxt)
        if not parts:
            continue
        pos = 0
        for v, w in parts:
            for _ in range(6):
                m = mod.muxes.get(v)
                if not m or v in done:
                    break
                c, a, b = m
                held = mod.ext.get(b) == (r, pos) and a or mod.ext.get(a) == (r, pos) and b
                if not held:
                    break
                mod.lut['mux'] -= c
                done.add(v)
                v = held
            pos += w


# Synthesized per counted, by the kind of array a level read selects from (11-11, a32-G0 OOC of the shell-sized
# profile at 64 SMs, each read in a module of its own, -flatten_hierarchy none): array_create selects 34,275 LUTs
# against 104,272 counted on their live entries (50 reads; 0.21-1.02 each: a link stream's page select of a held
# word whose pages are mostly constant registers costs ~1 LUT a bit, a select of live words its full mux), the
# other arrays 2,625 against 2,688; tables (88) none measurable.
ARRAY_READ_SYNTH = {'const': 1.0, 'create': 0.33, 'other': 1.0}
# A constant table read (a ROM; the folded simulator's slot tables, read at a frame position of up to II entries):
# priced on its contents (FOLDSIM_COST_TABLE_BLOCKS=0: one LUT6 per 64 entries of every varying bit, as before 11-13).
# Each distinct output bit (bit columns of the same contents are one function) is built of 64-entry blocks, one LUT6
# each, of which identical blocks are one and constant blocks none; the blocks are chosen by the index's upper bits:
# a decoder per run of equal blocks and a select of the distinct ones. Measured 11-13 (a32-F4): FSD 1,024 nodes (E3,
# II 5,118), 2,953 table reads in the simulator level and its SLR group modules, counted 397k LUTs as before; placed
# (Vivado 2025.2, F4/pnr/Q2util/hier_afu.rpt) the level they were in holds 640 LUTs (and 54 RAMB36 ROMs).
TABLE_BLOCKS = os.environ.get('FOLDSIM_COST_TABLE_BLOCKS', '1') != '0'


def const_table_cost(n, w, ints):
    """LUT6s of a read of the constant table `ints` (n entries of w bits) at a non-constant index (TABLE_BLOCKS)."""
    blocks = math.ceil(n / 64)
    columns = set()
    for b in range(w):
        columns.add(tuple((x >> b) & 1 for x in ints))
    total = 0.0
    for col in columns:
        pats = [col[k * 64:(k + 1) * 64] for k in range(blocks)]
        const = {p for p in pats if all(v == p[0] for v in p)}
        classes = len(set(pats))
        if classes == 1 and pats[0] in const:
            continue                                   # a constant bit
        live = len(set(pats) - const)                  # one LUT6 a distinct non-constant block
        runs = 1 + sum(1 for k in range(1, blocks) if pats[k] != pats[k - 1])
        total += live + (runs - 1 + mux_tree(classes) if classes > 1 else 0)
    return ARRAY_READ_SYNTH['const'] * total

# The simulator level's other logic (its own ops as `parse` prices them, array reads and memories apart),
# synthesized per counted: 80,549 / 110,618 = 0.73 in the same OOC, 89.3k / 126.6k = 0.71 in P/s11 p1_300 (routed,
# flattened, the same profile's shell-sized fold of 11-09): foldsim_fitdb.count scales the level's logic by it.
LEVEL_LOGIC_SYNTH = 0.73


def array_read_cost(n, w, src, live=None, varying=None):
    """LUT6s of one array read (hw.array_get) with a non-constant index, by
    the array it reads (the simulator level's selects, foldsim_fitdb.count):
    - a constant table (`src` 'const', hw.aggregate_constant): a ROM, one
      LUT6 a varying output bit (`varying`: the bits not the same in every
      entry) for every 64 entries, and a 4:1 select of them beyond 64;
    - an array built of values (`src` 'create', hw.array_create): a select of
      its `live` non-constant entries (a constant entry is folded into the
      select's logic), (live - 1) / 3 LUTs a bit (LUT6 4:1 muxes with F7/F8),
      times the synthesized share measured (ARRAY_READ_SYNTH);
    - any other array: a select of all n entries."""
    if src == 'const':
        v = w if varying is None else varying
        blocks = math.ceil(n / 64)
        return ARRAY_READ_SYNTH['const'] * v * (blocks + mux_tree(blocks))
    k = n if live is None else live
    return ARRAY_READ_SYNTH.get(src, 1.0) * w * max(k - 1, 0) / 3


def array_reads(path, modules, per_read=None):
    """{module: (LUTs, {kind: LUTs})} of the hw.array_get reads with a
    non-constant index in `modules` of an MLIR file, priced by
    array_read_cost on what each array is built of (kinds: const, create,
    other); `per_read`, a dict, takes each read's (kind, LUTs) by (module,
    result). foldsim_fitdb.count adds them to the simulator level (11-11,
    a32-G0: they had been left out, the r16c rule, while the level's link
    stream word selects and tables are such reads)."""
    modules = set(modules)
    out = {}
    cur = None
    for line in open(path):
        st = line.strip()
        if st.startswith('hw.module'):
            m = re.match(r'hw\.module(?:\.extern)?(?:\s+private)?\s+@([\w$.]+)', st)
            cur = m.group(1) if m and m.group(1) in modules and 'extern' not in st else None
            if cur:
                consts, arrays = set(), {}
                out[cur] = [0.0, {'const': 0.0, 'create': 0.0, 'other': 0.0}]
            continue
        if cur is None or ' = ' not in st or not st.startswith('%'):
            continue
        lhs, rhs = st.split(' = ', 1)
        op = rhs.split()[0]
        res = re.findall(SSA, lhs)
        if op == 'hw.constant':
            consts.update(res)
        elif op == 'hw.aggregate_constant' and res:
            mm = re.search(r'!hw\.array<(\d+)xi(\d+)>\s*$', rhs)
            if mm:
                # (An i1 array prints its elements bare, `[false, true, ...]`: no ` : i1` after each. Read as typed only,
                # every 1-bit table was taken as varying in every bit and priced as a full ROM: a32-F4 11-13, FSD 1,024's
                # 2,049 one-run 1-bit tables of 1,024-8,192 entries, ~171 LUTs counted each.)
                vals = re.findall(r'(-?\d+|true|false)(?:\s*:\s*i\d+)?', rhs.split('[', 1)[1].rsplit('] :', 1)[0]) \
                    if '[' in rhs else []
                w = int(mm.group(2))
                if len(vals) == int(mm.group(1)):
                    ints = [(1 if v == 'true' else 0 if v == 'false' else int(v)) & ((1 << w) - 1) for v in vals]
                    a, o = (1 << w) - 1, 0
                    for x in ints:
                        a &= x
                        o |= x
                    varying = popcount(o & ~a)
                else:
                    varying, ints = w, None
                arrays[res[0]] = ('const', varying, ints, {})
            consts.update(res)
        elif op == 'hw.array_create' and res:
            ops = re.findall(SSA, rhs.split(' : ')[0])
            arrays[res[0]] = ('create', sum(1 for o in ops if o not in consts))
        elif op == 'hw.wire' and res and re.search(r'rom_style" = "\\22block\\22', rhs):
            # (foldsim-table-roms: a constant table as a rom_style "block" wire, read into a register: a block RAM
            # ROM, its output register absorbed; no LUTs, its RAMB36 in the kinds' 'rom_bram'.)
            ops = re.findall(SSA, rhs.split(' : ')[0])
            if ops and ops[0] in arrays and arrays[ops[0]][0] == 'const':
                arrays[res[0]] = ('rom',)
        elif op == 'hw.array_get':
            mm = re.search(r'!hw\.array<(\d+)x(.*)>\s*,', rhs)
            idx = re.findall(r'\[(' + SSA + r')\]', rhs)
            ops = re.findall(SSA, rhs.split(' : ')[0])
            if not mm or (idx and idx[0] in consts) or not ops:
                continue
            n, w = int(mm.group(1)), width(mm.group(2))
            src = arrays.get(ops[0])
            if src and src[0] == 'rom':
                out[cur][1]['rom_bram'] = out[cur][1].get('rom_bram', 0.0) + bram_blocks(n, w)
                if per_read is not None and res:
                    per_read[(cur, res[0])] = ('rom', 0.0)
                continue
            if src and src[0] == 'const':
                kind = 'const'
                if TABLE_BLOCKS and src[2] is not None and len(src[2]) == n:
                    if (n, w) not in src[3]:               # (a table read several times is priced once)
                        src[3][(n, w)] = const_table_cost(n, w, src[2])
                    c = src[3][(n, w)]
                else:
                    c = array_read_cost(n, w, 'const', varying=src[1])
            elif src and src[0] == 'create':
                kind, c = 'create', array_read_cost(n, w, 'create', live=src[1])
            else:
                kind, c = 'other', array_read_cost(n, w, 'other')
            out[cur][0] += c
            out[cur][1][kind] += c
            if per_read is not None and res:
                per_read[(cur, res[0])] = (kind, c)
    return {k: (v[0], v[1]) for k, v in out.items()}


# A distributed RAM written with a byte (or wider-lane) mask: Vivado builds it of dual-port single-bit primitives
# (RAM32X1D / RAM64X1D, 2 LUTs a bit and read port), not RAM32M16's 14 bits in 8 LUTs: measured 11-11 (a32-G0 OOC,
# -flatten_hierarchy none, the shell-sized profile at 64 SMs) on 23 masked landings of 8-32 entries, 50,512 bits:
# 100,394 LUTs, 1.99 a bit, against 0.57 counted.
MASKED_LUTRAM_PER_BIT = 2.0
# FOLDSIM_COST_MASKED_LUTRAM=0: priced as unmasked, as before 11-11.
MASKED_LUTRAM = os.environ.get('FOLDSIM_COST_MASKED_LUTRAM', '1') != '0'


# FOLDSIM_COST_VERSAL_LUTRAM=0: the fit count prices a distributed RAM as mem_cost does (before 11-12).
VERSAL_LUTRAM = os.environ.get('FOLDSIM_COST_VERSAL_LUTRAM', '1') != '0'


def lutram_cost(depth, w, rports, mask=1):
    """(RAM LUTs, logic LUTs) of a distributed RAM of `depth` x `w` with `rports` read ports, as Vivado builds it on
    Versal: per read port and 64-row block (32 rows: RAM32M16) the cheaper of RAM64M8 (8 LUTs for 7 bits; RAM32M16 8
    for 14) and RAM64X1D (2 LUTs a bit; RAM32X1D), a byte-masked one of single-bit dual-port primitives
    (MASKED_LUTRAM_PER_BIT); deeper than 64 rows, the blocks' read select (mux_tree, a LUT a bit for two blocks) and a
    write enable a block (a32-UR 11-12, V1's OOC of y64g34: VX_fma_unit's 4,096 x 1 slot RAMs 128 RAM + 89 logic LUTs
    each, l2fs_4's 128 x 24 landing buffers a LUT3 a bit for the select)."""
    rports = max(rports, 1)
    if depth <= 32:
        prim = min(8 * math.ceil(w / 14), 2 * w)
        blocks = 1
    else:
        blocks = math.ceil(depth / 64)
        prim = min(8 * math.ceil(w / 7), 2 * w)
    if mask > 1 and MASKED_LUTRAM:
        prim = w * MASKED_LUTRAM_PER_BIT
    ram = prim * blocks * rports
    logic = (w * mux_tree(blocks) * rports + blocks) if blocks > 1 else 0
    return ram, logic


# RAMB36 aspect ratios (width, depth) on Versal (RAMB36E5: 512 x 72 simple dual port, 1K x 36 ... 32K x 1; a RAMB18E5
# half of it, 512 x 36 ... 16K x 1). FOLDSIM_COST_BRAM_ASPECT=0: priced in 512 x 72 blocks only (RAMB18 at <= 512 x 36),
# as before 11-12.
BRAM36 = ((72, 512), (36, 1024), (18, 2048), (9, 4096), (4, 8192), (2, 16384), (1, 32768))
BRAM18 = ((36, 512), (18, 1024), (9, 2048), (4, 4096), (2, 8192), (1, 16384))
BRAM_ASPECT = os.environ.get('FOLDSIM_COST_BRAM_ASPECT', '1') != '0'


def bram_blocks(depth, w):
    """RAMB36 (a RAMB18 half) of one depth x w block RAM at its cheapest aspect ratio: Vivado builds a 1,024 x 36
    memory in one RAMB36, not two 512 x 72 (a32-UR 11-12: V1's OOC of y64g34, block RAM 1,261 synthesized against
    1,358 counted, the counted VX_mufu_pe's 1,024 x 36 state in two)."""
    if not BRAM_ASPECT:
        return 0.5 if w <= 36 and depth <= 512 else math.ceil(w / 72) * math.ceil(depth / 512)
    c36 = min(math.ceil(w / cw) * math.ceil(depth / cd) for cw, cd in BRAM36)
    c18 = min(math.ceil(w / cw) * math.ceil(depth / cd) for cw, cd in BRAM18) * 0.5
    return min(c36, c18)


# URAM288 Vivado 2025.2 builds a ram_style ultra memory of 72 bits and h x 4,096 rows of without a cascade_height
# (its default "cascade height of 1"), by h (a32-UR 11-12, OOC on the V80, 1R1W, byte-masked / unmasked): 8K 3 / 2,
# 12K 5, 16K 5, 24K 9 / 8; past these h + ceil(h / 2) / h + h // 3 (the measured points' trend).
URAM_NO_CASCADE = {True: {1: 1, 2: 3, 3: 5, 4: 5, 6: 9}, False: {1: 1, 2: 2, 6: 8}}


def uram_count(depth, w, rports, wports, mask=1, cascade=None):
    """URAM288 of one ram_style ultra memory as Vivado builds it: 72-bit columns of 4,096 rows cascaded (the memory
    states its cascade_height, foldsim-fold-static since 11-12), else Vivado's default build of a deep one
    (URAM_NO_CASCADE); a read port beyond the first of a memory without a read-write port a copy."""
    copies = max(1, rports if wports else math.ceil(rports / 2))
    h = math.ceil(depth / 4096)
    if h > 1 and not cascade:
        masked = mask > 1
        h = URAM_NO_CASCADE[masked].get(h, h + (math.ceil(h / 2) if masked else h // 3))
    return math.ceil(w / 72) * h * copies


def mem_parts(depth, w, rports, rlat, style=None, mask=1):
    """(distributed RAM LUTs, logic LUTs, RAMB36) of one memory as Vivado builds it on Versal (the fit count,
    foldsim_fitdb.count and foldsim_sll_check: lutram_cost, bram_blocks; its LUTRAM's read select and write enables
    as logic, as Vivado reports them). Block RAM by mem_cost's rule."""
    rports = max(rports, 1)
    bram = style == 'block' or (style != 'distributed' and rlat >= 1 and depth >= 32 and depth * w >= 2048)
    if bram:
        return 0.0, 0.0, bram_blocks(depth, w) * max(1, math.ceil(rports / 2))
    if not VERSAL_LUTRAM:
        return mem_cost(depth, w, rports, rlat, style, mask)[0], 0.0, 0
    ram, logic = lutram_cost(depth, w, rports, mask)
    return ram, logic, 0


def mem_cost(depth, w, rports, rlat, style=None, mask=1):
    """(LUTRAM LUTs, BRAM36) of one memory. Synchronous reads from 32 entries up go to
    block RAM, as Vivado infers them, unless the memory asks otherwise. `mask`: the
    write mask's lanes (a masked distributed RAM: MASKED_LUTRAM_PER_BIT). The price the
    compiler's choices are made on (foldsim_degree, foldsim_storage); the fit count prices
    a netlist's memories as built (mem_parts)."""
    rports = max(rports, 1)
    bram = style == 'block' or (style != 'distributed' and rlat >= 1 and depth >= 32 and depth * w >= 2048)
    if bram:
        if w <= 36 and depth <= 512:
            per = 0.5
        else:
            per = math.ceil(w / 72) * math.ceil(depth / 512)
        return 0, per * max(1, math.ceil(rports / 2))
    if mask > 1 and MASKED_LUTRAM:
        blocks = math.ceil(depth / 64)
        return w * blocks * MASKED_LUTRAM_PER_BIT * rports + (w * (blocks - 1) / 3 if depth > 64 else 0), 0
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
    for (_, d, w, r, wr, rl, st, mk, *_) in m.mems:
        lr, br = mem_cost(d, w, r, rl, st, mk)
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
