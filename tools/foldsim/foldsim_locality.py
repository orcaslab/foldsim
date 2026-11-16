#!/usr/bin/env python3
"""foldsim_locality.py <gen dir> [--assign <sll.json | CLASS:SLR,...>] [--floorplan <tcl>] [--sll-json <sll.json>]
                       [--slr-region <json>] [--kernel-slr SLRn] [--out <dir>] [--json <file>]

Whether a generated simulator is hierarchical enough to absorb wire delay by spatial locality over the region's SLRs:
not whether it fits (foldsim_sll_check.py), but whether every signal that leaves a region leaves it from a register
and enters the next one into a register. A design that fits and is not local fails routing (y8b, 11-12: a 477k-LUT
cluster of class copies joined by zero-latency links, pinned as one blob; FSD E3: the level's control, the kernel's
token channels and reset fanning out unregistered over the SLRs, WNS -5.3 ns at 200 MHz).

The netlist's simulator level (foldsim_sim and the SLR group modules foldsim_sim_slr<g> under it, and the physical
domains foldsim_sim_dom_<class> under either (--physical-domains: the same netlist, its hierarchy only), inlined) is read as
a graph: registers, memories (a read of latency 0 is logic on its address, the memory a register), logic and wires
(foldsim_cost.FREE), the class copies' pins (`<class>_copy` of `<class>_static`, each copy's ports judged inside its
module: an input registered where every use is a register's or a memory write's operand through wires alone, an
output registered where only wires lie between it and registers), and the kernel (the level's ports).

Regions: a copy's SLR from the floorplan's SLR groups (foldsim_slr<g> pinned whole) or its pinned copies (`# foldsim
pinned copies`), else `--assign` (an sll.json's `assignment`, or CLASS:SLR pairs); a copy left unassigned is judged
by itself only. A level register or memory sits where its loads are (the placer's choice, iterated to a fixpoint
through level registers): in one SLR when they all are (local), else it is GLOBAL. The kernel is a region of its own
(K) unless --kernel-slr places it.

Rules (each finding names the signal group and its bits):
  L1 copies: a copy larger than every SLR (LUT, at the region's margin) or one the fit check cut over SLRs (--sll-json
     `over` / `cuts`): REFUSE (a straddle); a copy larger than a clock-region scale (a quarter of the smallest SLR it
     fits): WARN.
  L2 unregistered crossings: a pin whose combinational cone reaches a leaf in another SLR (a copy's output, a level
     register placed elsewhere) without being registered at both ends -- leaf a register (or a registered copy
     output), only wires between, the pin registered inside its copy: REFUSE, by source and destination group.
     Unregistered clusters: copies joined by such paths, whatever the assignment, must sit in one SLR: a cluster
     past every SLR's margin: REFUSE; past a quarter of the smallest SLR it fits: WARN.
  L3 global fanout: a level register or a kernel port whose loads span SLRs, i.e. no per-region register stage:
     REFUSE when it reaches any load through logic or into an unregistered pin, WARN when register to registered pins
     by wires alone.
Exits 0 (OK or WARN), 1 (REFUSE), 2 (error). Report <out>/locality.txt; --json the findings.
"""
import argparse, json, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foldsim_cost  # noqa: E402

SSA = re.compile(r'%[\w.$-]+')
FREE = set(foldsim_cost.FREE) - {'hw.output'}
REGS = foldsim_cost.REGS
CLOCK = re.compile(r'clock\s+%[\w.$-]+')
OPLINE = re.compile(r'\s*((?:%[\w.$-]+)(?:, %[\w.$-]+)*) = ([\w.]+)(.*)$')
NOLHS = re.compile(r'\s*([\w.]+)\s(.*)$')


def match_paren(s, i):
    """The index of the ')' closing the '(' at s[i]."""
    depth, prev = 0, ''
    for j in range(i, len(s)):
        ch = s[j]
        if ch in '(<[{':
            depth += 1
        elif ch in ')]}' or (ch == '>' and prev != '-'):
            depth -= 1
            if depth == 0:
                return j
        prev = ch
    return len(s) - 1


def modules(path, want):
    """{name: (ports text, [body lines])} of the top-level hw.modules whose name `want(name)` accepts."""
    out, cur, ports, body = {}, None, None, None
    hdr = re.compile(r'^  hw\.module(?:\s+private)?\s+@([\w$.]+)\(')
    with open(path) as f:
        for line in f:
            if cur is None:
                m = hdr.match(line)
                if m and want(m.group(1)):
                    i = line.index('(', m.end() - 1)
                    cur, ports, body = m.group(1), line[i + 1:match_paren(line, i)], []
                continue
            if line.startswith('  }'):
                out[cur] = (ports, body)
                cur = None
                continue
            body.append(line)
    return out


def port_list(ports):
    """[(dir, name, width)] of a module's port text."""
    out = []
    for p in foldsim_cost.split_top(ports):
        m = re.match(r'\s*(in|out|inout)\s+%?([\w$.]+)\s*:\s*(.*)$', p)
        if m and 'clock' not in m.group(3):
            out.append((m.group(1), m.group(2), foldsim_cost.width(m.group(3))))
    return out


def instance(rest):
    """(instance name, module, [(port, operand, width)], [(out port, width)]) of an hw.instance's text."""
    # (A parameterized module's instance: its parameters `<NAME: type = value, ...>` before the operands.)
    m = re.match(r'\s*"([^"]*)"(?:\s+sym\s+@\S+)?\s+@([\w$.]+)(?:<[^(]*>)?\(', rest)
    i = m.end() - 1
    j = match_paren(rest, i)
    ins = []
    for p in foldsim_cost.split_top(rest[i + 1:j]):
        mm = re.match(r'\s*"?([\w$.]+)"?\s*:\s*(%[\w.$-]+)\s*:\s*(.*)$', p)
        if mm and 'clock' not in mm.group(3):
            ins.append((mm.group(1), mm.group(2), foldsim_cost.width(mm.group(3))))
    outs = []
    k = rest.find('->', j)
    if k >= 0:
        a = rest.index('(', k)
        for p in foldsim_cost.split_top(rest[a + 1:match_paren(rest, a)]):
            mm = re.match(r'\s*"?([\w$.]+)"?\s*:\s*(.*)$', p)
            if mm:
                outs.append((mm.group(1), foldsim_cost.width(mm.group(2))))
    return m.group(1), m.group(2), ins, outs


def operands(rest):
    return SSA.findall(CLOCK.sub('', rest.split(' : ')[0] if ' : ' in rest else rest))


# ----------------------------------------------------------------------------------------------- a copy's own ports
def copy_ports(ports, body):
    """({input: registered}, {output: registered}) of a class copy's module: an input is registered where every use
    of it, through wires alone, is a register's operand (data, enable, reset) or a memory write's; an output where only
    wires lie between it and registers, registered memory reads or constants."""
    plist = port_list(ports)
    defs, uses, memlat = {}, defaultdict(list), {}
    outs = []
    for line in body:
        m = OPLINE.match(line)
        if m:
            res = m.group(1).split(', ')
            op, rest = m.group(2), m.group(3)
            if op == 'seq.firmem':
                mm = re.match(r'\s*(\d+)', rest)
                memlat[res[0]] = int(mm.group(1)) if mm else 1
                continue
            ops = operands(rest)
            if op == 'seq.firmem.read_port' or op == 'seq.firmem.read_write_port':
                mem = ops[0]
                kind = 'reg' if memlat.get(mem, 1) > 0 else 'logic'
                for v in ops[1:]:
                    uses[v].append(kind if op == 'seq.firmem.read_port' else 'reg')
                defs[res[0]] = ('reg' if kind == 'reg' else 'logic', [])
                continue
            if op in REGS:
                kind = 'reg'
            elif op in FREE:
                kind = 'const' if op in foldsim_cost.CONSTS else 'wire'
            elif op == 'hw.instance':
                kind = 'inst'
            else:
                kind = 'logic'
            for r in res:
                defs[r] = (kind, ops if kind == 'wire' else [])
            for v in ops:
                uses[v].append(('wire', res) if kind == 'wire' else kind)
            continue
        m = NOLHS.match(line)
        if not m:
            continue
        op, rest = m.group(1), m.group(2)
        if op == 'hw.output':
            outs = SSA.findall(rest.split(' : ')[0])
            for v in outs:
                uses[v].append('out')
        elif op in ('seq.firmem.write_port',):
            for v in operands(rest)[1:]:
                uses[v].append('reg')

    def in_reg(v, seen):
        if v in seen:
            return True
        seen.add(v)
        us = uses.get(v, [])
        if not us:
            return True
        for u in us:
            if isinstance(u, tuple):
                if not all(in_reg(r, seen) for r in u[1]):
                    return False
            elif u != 'reg':
                return False
        return True

    def out_reg(v, seen):
        if v in seen:
            return True
        seen.add(v)
        d = defs.get(v)
        if d is None:
            return False          # an input fed through
        if d[0] in ('reg', 'const'):
            return True
        if d[0] == 'wire':
            return all(out_reg(x, seen) for x in d[1])
        return False
    ins_r = {n: in_reg('%' + n, set()) for d, n, _ in plist if d == 'in'}
    onames = [n for d, n, _ in plist if d == 'out']
    outs_r = {n: out_reg(v, set()) for n, v in zip(onames, outs)}
    return ins_r, outs_r


# ----------------------------------------------------------------------------------------------- the level graph
class Graph:
    def __init__(self):
        self.kind = []        # wire | logic | reg | mem | in | cout | const
        self.preds = []       # comb predecessors (wire, logic)
        self.leaf = []        # a latency-0 memory read: its memory node (the read is logic on the address)
        self.name = []
        self.width = []
        self.group = []       # SLR group index of the module the node lives in (None: the top level)
        self.copy = []        # cout: the copy instance
        self.creg = []        # cout: registered inside its copy
        self.ids = {}
        self.sinks = []       # (kind, node, bits, label, owner): pin (owner copy), store (owner node), kout

    def node(self, key, kind='wire', name=None, width=0, group=None):
        i = self.ids.get(key)
        if i is None:
            i = len(self.kind)
            self.ids[key] = i
            self.kind.append(kind)
            self.preds.append([])
            self.leaf.append(None)
            self.name.append(name or key)
            self.width.append(width)
            self.group.append(group)
            self.copy.append(None)
            self.creg.append(True)
        elif kind != 'wire' and self.kind[i] == 'wire' and not self.preds[i]:
            self.kind[i] = kind
            self.width[i] = width or self.width[i]
            self.group[i] = group
            if name:
                self.name[i] = name
        return i


def unit_outputs(path):
    """{external module: {output: 'reg' | 'in:<input>'}} of the externals whose foldsim.unit_outputs states how their
    RTL drives each output (fold-static vmem-s28's units): a register, wires from an input; others are logic."""
    out = {}
    rx = re.compile(r'^  hw\.module\.extern(?:\s+private)?\s+@([\w$.]+).*foldsim\.unit_outputs = \[([^\]]*)\]')
    with open(path) as f:
        for line in f:
            m = rx.match(line)
            if m:
                out[m.group(1)] = dict(x.split('=', 1) for x in re.findall(r'"([^"]*)"', m.group(2)))
    return out


def build_level(path, sim='foldsim_sim'):
    """The simulator level as a Graph, the SLR group modules inlined; {copy: (class, group)}; copy port maps."""
    level_mods = modules(path, lambda n: n == sim or re.fullmatch(re.escape(sim) + r'_slr\d+', n) is not None
                         or n.startswith(sim + '_dom_'))
    if sim not in level_mods:
        raise SystemExit(f'foldsim_locality.py: {path} has no module {sim}')
    g = Graph()
    units = unit_outputs(path)
    copies = {}
    pins = []   # (copy, port, node, width)
    couts = []  # (copy, port, node)

    def walk(mname, prefix, group, argmap):
        ports, body = level_mods[mname]
        key = lambda v: prefix + v
        plist = port_list(ports)
        for d, n, w in plist:
            if d == 'in':
                k = key('%' + n)
                if argmap is None:
                    g.node(k, 'in', n, w, group)
                else:
                    i = g.node(k, 'wire', n, w, group)
                    g.preds[i] = [argmap[n]] if n in argmap else []
        memlat = {}
        outs = []
        for line in body:
            m = OPLINE.match(line)
            if m:
                res = m.group(1).split(', ')
                op, rest = m.group(2), m.group(3)
                if op == 'hw.instance':
                    iname, mod, ins, outl = instance(rest)
                    if mod in level_mods:
                        # (A physical domain stays in the group of the module that instantiates it.)
                        mg = re.search(r'_slr(\d+)$', mod)
                        gi = int(mg.group(1)) if mg else group
                        amap = {p: g.node(key(v)) for p, v, _ in ins}
                        sub = walk(mod, prefix + iname + '/', gi, amap)
                        for r, (on, w) in zip(res, outl):
                            i = g.node(key(r), 'wire', r, w, group)
                            g.preds[i] = [sub[on]]
                        continue
                    if mod.endswith('_static'):
                        copies[iname] = (mod[:-len('_static')], group)
                        for p, v, w in ins:
                            pins.append((iname, p, g.node(key(v)), w))
                        for r, (on, w) in zip(res, outl):
                            i = g.node(key(r), 'cout', f'{iname}.{on}', w, group)
                            g.copy[i] = iname
                            couts.append((iname, on, i))
                        continue
                    # another module (a black box, a kernel helper): its outputs are logic on its inputs, unless its
                    # contract (foldsim.unit_outputs) says an output is a register (its inputs then end in it) or
                    # wires from one input
                    pre = [g.node(key(v)) for _, v, _ in ins]
                    kinds = units.get(mod, {})
                    byname = {p: g.node(key(v)) for p, v, _ in ins}
                    for r, (on, w) in zip(res, outl):
                        k = kinds.get(on, '')
                        if k == 'reg':
                            i = g.node(key(r), 'reg', r, w, group)
                            for v in pre:
                                g.sinks.append(('store', v, w, r, i))
                        elif k.startswith('in:') and k[3:] in byname:
                            i = g.node(key(r), 'wire', r, w, group)
                            g.preds[i] = [byname[k[3:]]]
                        else:
                            i = g.node(key(r), 'logic', r, w, group)
                            g.preds[i] = pre
                    continue
                if op == 'seq.firmem':
                    mm = re.match(r'\s*(\d+)', rest)
                    memlat[res[0]] = int(mm.group(1)) if mm else 1
                    mm = re.search(r'<(\d+) x (\d+)', rest)
                    g.node(key(res[0]), 'mem', res[0], int(mm.group(2)) if mm else 0, group)
                    continue
                ops = operands(rest)
                w = foldsim_cost.width(foldsim_cost.result_type(rest)) if ' : ' in rest else 0
                if op in ('seq.firmem.read_port', 'seq.firmem.read_write_port'):
                    mem = g.node(key(ops[0]))
                    if memlat.get(ops[0], 1) == 0:
                        i = g.node(key(res[0]), 'logic', res[0], g.width[mem], group)
                        g.preds[i] = [g.node(key(v)) for v in ops[1:]]
                        g.leaf[i] = mem
                    else:
                        i = g.node(key(res[0]), 'reg', res[0], g.width[mem], group)
                        for v in ops[1:]:
                            g.sinks.append(('store', g.node(key(v)), 0, res[0], i))
                    if op == 'seq.firmem.read_write_port':
                        for v in ops[1:]:
                            g.sinks.append(('store', g.node(key(v)), 0, ops[0], mem))
                    continue
                if op in REGS:
                    i = g.node(key(res[0]), 'reg', res[0], w, group)
                    for v in ops:
                        g.sinks.append(('store', g.node(key(v)), w, res[0], i))
                    continue
                kind = 'const' if op in foldsim_cost.CONSTS else 'wire' if op in FREE else 'logic'
                if op == 'hw.array_get' and len(ops) == 2 and re.match(r'%c\d|%c-|%false|%true', ops[1]):
                    kind = 'wire'
                pre = [g.node(key(v)) for v in ops]
                for r in res:
                    i = g.node(key(r), kind, r, w, group)
                    g.preds[i] = pre
                continue
            m = NOLHS.match(line)
            if not m:
                continue
            op, rest = m.group(1), m.group(2)
            if op == 'hw.output':
                outs = SSA.findall(rest.split(' : ')[0])
            elif op == 'seq.firmem.write_port':
                ops = operands(rest)
                mem = g.node(key(ops[0]))
                for v in ops[1:]:
                    g.sinks.append(('store', g.node(key(v)), 0, ops[0], mem))
        onames = [(n, w) for d, n, w in plist if d == 'out']
        res = {}
        for (n, w), v in zip(onames, outs):
            res[n] = g.node(key(v))
            if argmap is None:
                g.sinks.append(('kout', res[n], w, n, None))
        return res
    walk(sim, '', None, None)
    return g, copies, pins, couts


# ----------------------------------------------------------------------------------------------- categories
PORTS = set()


def category(name, kind):
    n = name.lower().lstrip('%')
    base = re.sub(r'_(valid|ready|bits|count|head|entry|due|full|room)\w*$', '', n)
    if 'reset' in n:
        return 'reset'
    if kind == 'cout':
        return 'link'
    if re.search(r'(^|[_.])vm\d+_|vmem|vport', n):
        return 'vmem'
    if re.search(r'_to_|_links|_staged|_held|_written|_land', n):
        return 'link'
    if kind == 'in' or base in PORTS or re.search(r'_(valid|ready|bits)$|^dcr_|^start|^busy|token|uart', n):
        return 'env/token'
    if re.search(r'(^|_)(seq|pos\d|bubble\d*|state\d|inst\d|go|advance|ahead\d*|vready\w*|frame\w*|slot\w*|'
                 r'ready\d|pending\w*|parity|fold\d*|lookahead|region\d*|decision|sequencer|run|stall)(_|$|\d)', n):
        return 'control'
    return 'other'


def assignment(spec):
    if not spec:
        return {}
    if os.path.exists(spec):
        j = json.load(open(spec))
        names = [c['name'] for c in j['caps']]
        return {c: names[i] for i, cs in enumerate(j['assignment']) for c in cs}
    return dict(x.split(':', 1) for x in spec.split(','))


def floorplan(path):
    """({group: SLR}, {class: SLR}) pinned by the floorplan (SLR groups whole, pinned copies)."""
    if not path or not os.path.exists(path):
        return {}, {}
    text = open(path).read()
    groups = {int(gr): w for gr, w in re.findall(
        r'-hierarchical foldsim_slr(\d+)\]\s*\n[^\n]*USER_SLR_ASSIGNMENT (SLR\d+) \$c', text)}
    pinned = {}
    if '# foldsim pinned copies' in text:
        sec = text[text.index('# foldsim pinned copies'):]
        pinned = dict(re.findall(r'-hierarchical (\w+)_copy\]\s*\n[^\n]*USER_SLR_ASSIGNMENT (SLR\d+) \$c', sec))
    return groups, pinned


# ----------------------------------------------------------------------------------------------- the analysis
def analyze(gdir, assign=None, fplan=None, sll_json=None, region=None, kernel_slr=None, log=print):
    sim = os.path.join(gdir, 'foldsim_sim.mlir')
    if not os.path.exists(sim):
        sim = os.path.join(gdir, 'gen', 'foldsim_sim.mlir')
    if not os.path.exists(sim):
        raise SystemExit(f'foldsim_locality.py: no foldsim_sim.mlir in {gdir}')
    if fplan is None:
        for c in (os.path.join(os.path.dirname(sim), 'foldsim_floorplan.tcl'),):
            fplan = c if os.path.exists(c) else None
    region = region or json.load(open(os.path.join(HERE, 'foldsim_v80_slr.json')))
    slrs = [s['name'] for s in region['slrs']]
    caps = {s['name']: s['luts'] * region.get('margin', 0.85) for s in region['slrs']}
    g, copies, pins, couts = build_level(sim)
    PORTS.clear()
    PORTS.update(re.sub(r'_(valid|ready|bits)$', '', g.name[i].lower()) for i in range(len(g.kind)) if g.kind[i] == 'in')
    log(f'level: {len(g.kind):,} nodes, {len(copies)} copies, {len(pins):,} copy pins, {len(g.sinks):,} sinks')
    # Each copy's own ports.
    cmods = modules(sim, lambda n: n.endswith('_static') and n[:-len('_static')] in {c for c, _ in copies.values()})
    creg = {}
    for cls, (ports, body) in cmods.items():
        creg[cls[:-len('_static')]] = copy_ports(ports, body)
        cmods[cls] = None
    del cmods
    groups, pinned = floorplan(fplan)
    asg = assignment(assign)
    where, how = {}, {}
    for inst, (cls, grp) in copies.items():
        if grp is not None and grp in groups:
            where[inst], how[inst] = groups[grp], 'group'
        elif cls in pinned:
            where[inst], how[inst] = pinned[cls], 'pinned'
        elif cls in asg:
            where[inst], how[inst] = asg[cls], 'assigned'
    for i, (inst, port, node) in enumerate(couts):
        cls = copies[inst][0]
        g.creg[node] = creg.get(cls, ({}, {}))[1].get(port, False)
    gslr = {i: groups.get(gr) for i, gr in enumerate(g.group)}   # a node in a pinned SLR group module

    N = len(g.kind)
    # Topological order of the combinational graph (registers, memories, inputs, copy outputs are leaves).
    order, state = [], bytearray(N)
    for s in range(N):
        if state[s]:
            continue
        stack = [(s, 0)]
        while stack:
            v, k = stack.pop()
            if k == 0:
                if state[v]:
                    continue
                state[v] = 1
                stack.append((v, 1))
                if g.kind[v] in ('wire', 'logic'):
                    for p in g.preds[v]:
                        if not state[p]:
                            stack.append((p, 0))
            else:
                state[v] = 2
                order.append(v)
    pin_sinks = [('pin', node, w, f'{inst}.{port}', inst) for inst, port, node, w in pins]
    sinks = pin_sinks + g.sinks
    K = kernel_slr or 'K'

    def sink_region(s, store_reg):
        kind, _, _, _, owner = s
        if kind == 'pin':
            r = where.get(owner)
            return {r} if r else set()
        if kind == 'kout':
            return {K}
        if gslr.get(owner):
            return {gslr[owner]}
        # A store feeding a global store follows it (the global one is the finding, not its feeders).
        return store_reg[owner] if len(store_reg[owner]) == 1 else set()
    # Where each level store sits: its loads' regions, to a fixpoint through stores.
    store_reg = defaultdict(set)
    is_store = [g.kind[i] in ('reg', 'mem') for i in range(N)]
    is_store_in = [k in ('reg', 'mem', 'in') for k in g.kind]
    for it in range(8):
        load = [set() for _ in range(N)]
        for s in sinks:
            r = sink_region(s, store_reg)
            if r:
                load[s[1]] |= r
        for v in reversed(order):
            if load[v] and g.kind[v] in ('wire', 'logic'):
                for p in g.preds[v]:
                    load[p] |= load[v]
                if g.leaf[v] is not None:
                    load[g.leaf[v]] |= load[v]
        changed = False
        for v in range(N):
            if is_store_in[v] and load[v] - store_reg[v]:
                store_reg[v] |= load[v]
                changed = True
        if not changed:
            break
    leafreg = {}
    for v in range(N):
        if gslr.get(v) and g.kind[v] in ('reg', 'mem'):
            leafreg[v] = {gslr[v]}
        elif g.kind[v] == 'cout':
            r = where.get(g.copy[v])
            leafreg[v] = {r} if r else set()
        elif g.kind[v] == 'in' and kernel_slr:
            leafreg[v] = {kernel_slr}
        elif is_store_in[v]:
            leafreg[v] = set(store_reg[v])
    glob = {v for v, r in leafreg.items() if len(r) >= 2 and g.kind[v] != 'cout'}
    # Cone summaries: logic depth, leaf regions, copies reached, all-registered, global leaves.
    depth = [0] * N
    tags = [None] * N
    cps = [None] * N
    allreg = [True] * N
    gl = [None] * N
    E = frozenset()
    for v in order:
        k = g.kind[v]
        if k in ('reg', 'mem', 'in', 'cout', 'const'):
            r = leafreg.get(v, set())
            allreg[v] = k != 'in' and (k != 'cout' or g.creg[v])
            # region -> (logic levels from that region's leaves, all of them registers)
            tags[v] = {next(iter(r)): (0, allreg[v])} if len(r) == 1 else {}
            cps[v] = frozenset([g.copy[v]]) if k == 'cout' else E
            gl[v] = frozenset([v]) if v in glob else E
            continue
        ps = list(g.preds[v]) + ([g.leaf[v]] if g.leaf[v] is not None else [])
        d = max((depth[p] for p in ps), default=0) + (1 if k == 'logic' else 0)
        depth[v] = d
        t, c, a, gg = {}, E, True, E
        inc = 1 if k == 'logic' else 0
        for p in ps:
            if tags[p] is None:   # a leaf the walk has not met (an operand without a def: a block argument)
                continue
            for rr, (dd, aa) in tags[p].items():
                o = t.get(rr)
                t[rr] = (dd + inc, aa) if o is None else (max(o[0], dd + inc), o[1] and aa)
            c = c | cps[p] if cps[p] else c
            a = a and allreg[p]
            gg = gg | gl[p] if gl[p] else gg
        tags[v], cps[v], allreg[v], gl[v] = t, c, a, gg
    # Findings.
    cross = defaultdict(lambda: [0, 0])       # (src group, dst group, kind) -> [bits, pins]
    gfan = {}                                   # global leaf -> info
    clusters = {c: c for c in copies}

    def find(x):
        while clusters[x] != x:
            clusters[x] = clusters[clusters[x]]
            x = clusters[x]
        return x

    def src_label(v):
        if g.kind[v] == 'cout':
            return copies[g.copy[v]][0]
        return category(g.name[v], g.kind[v])
    leaves_of_cache = {}

    def leaves_of(v, cap=64):
        """A few named leaves of v's cone (for labels)."""
        if v in leaves_of_cache:
            return leaves_of_cache[v]
        out, stack, seen = [], [v], set()
        while stack and len(out) < cap:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if g.kind[x] in ('reg', 'mem', 'in', 'cout'):
                out.append(x)
                continue
            stack.extend(g.preds[x])
            if g.leaf[x] is not None:
                stack.append(g.leaf[x])
        leaves_of_cache[v] = out
        return out
    nsink = 0
    for s in sinks:
        kind, v, w, label, owner = s
        if tags[v] is None:
            continue
        reg_dst = True
        if kind == 'pin':
            cls = copies[owner][0]
            reg_dst = creg.get(cls, ({}, {}))[0].get(label.split('.', 1)[1], False)
            dst = cls
        elif kind == 'kout':
            dst = 'kernel'
        else:
            dst = 'level ' + category(g.name[owner], g.kind[owner])
        bits = w or g.width[v]
        unreg = depth[v] > 0 or not allreg[v] or not reg_dst
        if kind == 'pin' and unreg:
            for c in cps[v]:
                if c != owner:
                    a, b = find(c), find(owner)
                    if a != b:
                        clusters[a] = b
        rs = sink_region(s, store_reg)
        if gl[v]:
            for x in gl[v]:
                f = gfan.setdefault(x, dict(sinks=0, bits=0, logic=False, unreg_dst=False, dsts=set()))
                f['sinks'] += 1
                f['bits'] += bits
                f['logic'] |= depth[v] > 0
                f['unreg_dst'] |= not reg_dst
                f['dsts'].add(dst)
        if len(rs) != 1:
            # A global store's own cone: gathering leaves of several SLRs through logic is a convergence over SLRs.
            if kind == 'store' and owner in glob and depth[v] > 0 and len([t for t in tags[v] if t != K]) >= 2:
                key = ('/'.join(sorted(tags[v])), 'level global ' + category(g.name[owner], g.kind[owner]),
                       'unregistered')
                cross[key][0] += bits
                cross[key][1] += 1
            continue
        r = next(iter(rs))
        other = {t: x for t, x in tags[v].items() if t != r}
        if not other:
            continue
        # Registered at both ends: the foreign leaves registers, wires alone to a registered sink. Registered at the
        # source only: registers, then logic before the sink's register (one end). Else unregistered.
        if all(a for _, a in other.values()):
            unreg = 'registered' if (max(d for d, _ in other.values()) == 0 and reg_dst) else 'source-registered'
        else:
            unreg = 'unregistered'
        nsink += 1
        srcs = sorted({src_label(x) for x in leaves_of(v) if (leafreg.get(x) and leafreg[x] != {r}
                                                                and len(leafreg[x]) == 1)}) or ['?']
        if r == K or set(other) == {K}:
            unreg += ' kernel'      # the kernel's side: judged as a region only with --kernel-slr
        key = (' + '.join(srcs[:3]) + (' + ...' if len(srcs) > 3 else ''), dst, unreg)
        cross[key][0] += bits
        cross[key][1] += 1
    res = {}
    # L1: copies.
    sizes = {}
    over, cuts = [], []
    if sll_json and os.path.exists(sll_json):
        sj = json.load(open(sll_json))
        sizes = {c: v['lut'] for c, v in sj.get('copies', {}).items()}
        over, cuts = sj.get('over') or [], sj.get('cuts') or []
    l1 = []
    for inst, (cls, _) in sorted(copies.items()):
        lut = sizes.get(cls)
        if lut is None:
            continue
        fits = [s for s in slrs if lut <= caps[s]]
        smallest = min(fits, key=lambda s: caps[s]) if fits else None
        verdict = None
        if not fits or cls in over or cls in cuts:
            verdict = 'REFUSE'
        elif lut > caps[smallest] / 4 / region.get('margin', 0.85):
            verdict = 'WARN'
        l1.append(dict(copy=cls, lut=lut, fits=fits, where=where.get(inst), how=how.get(inst), verdict=verdict,
                       cut=cls in cuts, quarter=lut / (caps[smallest] / region.get('margin', 0.85)) * 4 if smallest
                       else None))
    # L2 clusters.
    cl = defaultdict(list)
    for c in copies:
        cl[find(c)].append(c)
    l2c = []
    for root, members in cl.items():
        if len(members) < 2:
            continue
        lut = sum(sizes.get(copies[m][0], 0) for m in members)
        slrset = sorted({where[m] for m in members if m in where})
        fits = [s for s in slrs if lut <= caps[s]]
        smallest = min(fits, key=lambda s: caps[s]) if fits else None
        verdict = 'REFUSE' if (len(slrset) > 1 or (sizes and not fits)) else \
            'WARN' if smallest and lut > caps[smallest] / 4 / region.get('margin', 0.85) else None
        l2c.append(dict(members=sorted(copies[m][0] for m in members), lut=lut, slrs=slrset, fits=fits,
                        verdict=verdict))
    l2c.sort(key=lambda x: -x['lut'])
    l2 = [dict(src=k[0], dst=k[1], kind=k[2], bits=v[0], pins=v[1],
               verdict={'unregistered': 'REFUSE', 'source-registered': 'WARN', 'unregistered kernel': 'WARN',
                        'source-registered kernel': 'WARN'}.get(k[2])) for k, v in cross.items()]
    l2.sort(key=lambda x: -x['bits'])
    l3 = []
    for x, f in gfan.items():
        slr_only = [r for r in leafreg[x] if r != K]
        if len(slr_only) < 2:
            verdict = 'WARN'          # one SLR and the kernel: a kernel boundary, not a fanout over SLRs
        else:
            verdict = 'REFUSE' if (f['logic'] or f['unreg_dst'] or g.kind[x] == 'in') else 'WARN'
        l3.append(dict(signal=g.name[x], kind=g.kind[x], category=category(g.name[x], g.kind[x]),
                       width=g.width[x], regions=sorted(leafreg[x]), sinks=f['sinks'], bits=f['bits'],
                       logic=f['logic'], unreg_dst=f['unreg_dst'], dsts=sorted(f['dsts'])[:8], verdict=verdict))
    l3.sort(key=lambda x: (x['verdict'] != 'REFUSE', -x['sinks']))
    res = dict(gen=gdir, copies={i: dict(cls=c, slr=where.get(i), how=how.get(i)) for i, (c, _) in copies.items()},
               l1=l1, clusters=l2c, crossings=l2, global_fanout=l3, groups=groups,
               unassigned=sorted(copies[i][0] for i in copies if i not in where))
    worst = 'OK'
    for item in l1 + l2c + l2 + l3:
        if item.get('verdict') == 'REFUSE':
            worst = 'REFUSE'
            break
        if item.get('verdict') == 'WARN':
            worst = 'WARN'
    res['verdict'] = worst
    return res


def summarize(res):
    """Totals by category: (L2 unregistered crossing bits by destination kind, L3 global signals by category)."""
    by = defaultdict(lambda: [0, 0, 0])
    for x in res['global_fanout']:
        b = by[x['category']]
        b[0] += 1
        b[1] += x['sinks']
        b[2] += x['bits']
    return by


def report(res, out):
    L = []
    p = L.append
    p(f"locality of {res['gen']}: {res['verdict']}")
    if res['unassigned']:
        p(f"  copies without a region (judged alone): {', '.join(res['unassigned'][:12])}"
          f"{' ...' if len(res['unassigned']) > 12 else ''}")
    p('  L1 class copies (LUT placed, from the fit check; quarter = x a quarter of the smallest SLR it fits):')
    for x in sorted(res['l1'], key=lambda x: -x['lut']):
        if x['verdict'] or x['lut'] > 50000:
            p(f"    {x['verdict'] or 'ok':6} {x['copy']}: lut {x['lut']:,.0f}, fits {'/'.join(x['fits']) or 'no SLR'}"
              f"{', CUT over SLRs' if x['cut'] else ''}, at {x['where'] or '-'} ({x['how'] or 'unassigned'})"
              f"{', %.1f x quarter' % x['quarter'] if x['quarter'] else ''}")
    p('  L2 unregistered clusters (copies joined by paths not registered at both ends; one region each):')
    for x in res['clusters'][:12]:
        p(f"    {x['verdict'] or 'ok':6} lut {x['lut']:,.0f} over {'/'.join(x['slrs']) or '-'} (fits "
          f"{'/'.join(x['fits']) or 'no SLR'}): {', '.join(x['members'][:10])}{' ...' if len(x['members']) > 10 else ''}")
    p('  L2 crossings into another SLR (source group -> destination, bits, pins):')
    tot = defaultdict(int)
    for x in res['crossings']:
        tot[x['kind'].replace(' kernel', '')] += x['bits']
        if x['kind'].endswith('kernel'):
            tot['kernel'] += x['bits']
    p(f"    totals: unregistered {tot['unregistered']:,} bits, registered at the source only {tot['source-registered']:,} "
      f"bits, registered at both ends {tot['registered']:,} bits (of them at the kernel's boundary {tot['kernel']:,})")
    for x in res['crossings'][:30]:
        p(f"    {x['verdict'] or 'ok':6} {x['kind']:24} {x['src']} -> {x['dst']}: {x['bits']:,} bits, {x['pins']:,} sinks")
    p('  L3 global fanout (a level register / kernel port whose loads span SLRs, no per-region register stage):')
    by = summarize(res)
    for c, (n, s, b) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        p(f"    {c}: {n:,} signals, {s:,} sinks, {b:,} sink bits")
    for x in res['global_fanout'][:40]:
        p(f"    {x['verdict']:6} {x['category']:9} {x['signal']} ({x['kind']}, {x['width']} b) over "
          f"{'/'.join(x['regions'])}: {x['sinks']:,} sinks{', through logic' if x['logic'] else ''}"
          f"{', into unregistered pins' if x['unreg_dst'] else ''} -> {', '.join(x['dsts'][:4])}")
    p(f"{os.path.basename(res['gen'].rstrip('/'))}: locality {res['verdict']}")
    text = '\n'.join(L) + '\n'
    if out:
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, 'locality.txt'), 'w').write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('gen')
    ap.add_argument('--assign', help='an sll.json (its assignment) or CLASS:SLR,... for copies the floorplan leaves')
    ap.add_argument('--floorplan', help='the implementation\'s floorplan (default the generation\'s)')
    ap.add_argument('--sll-json', help='the fit check\'s sll.json: copy sizes, over, cuts (L1, clusters)')
    ap.add_argument('--slr-region', default=os.path.join(HERE, 'foldsim_v80_slr.json'))
    ap.add_argument('--kernel-slr', help='the SLR of the kernel (its env, controllers); default a region of its own')
    ap.add_argument('--out')
    ap.add_argument('--json')
    ap.add_argument('--warn-only', action='store_true', help='exit 0 on REFUSE (report only)')
    a = ap.parse_args()
    try:
        res = analyze(a.gen, a.assign or a.sll_json, a.floorplan, a.sll_json, json.load(open(a.slr_region)),
                      a.kernel_slr, log=lambda s: print(s, file=sys.stderr))
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f'foldsim_locality.py: error: {e!r}', file=sys.stderr)
        sys.exit(2)
    sys.stdout.write(report(res, a.out))
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1, default=list)
    sys.exit(1 if res['verdict'] == 'REFUSE' and not a.warn_only else 0)


if __name__ == '__main__':
    main()
