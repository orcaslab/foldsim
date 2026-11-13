#!/usr/bin/env python3
"""foldsim_xcheck.py <foldsim_sim.mlir> [--top foldsim_sim] [--list N] (a32-F4, 11-12): the cross-group edges of a folded
simulator with SLR groups (foldsim --slr-groups: the level module `foldsim_sim` and its group instances foldsim_slr<g>),
each judged REGISTERED or not. An edge is a value that leaves a group's module or the level and is read in another
part: a group's input (from the level, or from another group through the level) or a group's output read by the level.
It is registered when its value, through wiring alone (extract, concat, bitcast, replicate, wire, array create/get of
constant index), comes from registers (seq.firreg / seq.compreg), memory read data, constants or the simulator's own
ports (the kernel side, judged there). Anything else (a comb op on the way) is LOGIC: a cross-SLR path with logic
before the crossing. Prints a summary (edges and bits, registered / logic) and the logic edges by source part and name
pattern; exit 0 when every edge is registered, 1 otherwise. Shared with a32-HX (fit_gate's locality check)."""
import re, sys, argparse, collections as C

WIRING = ('comb.extract', 'comb.concat', 'hw.bitcast', 'comb.replicate', 'hw.wire', 'hw.array_create',
          'hw.struct_create', 'hw.struct_extract', 'hw.aggregate_constant')
REG = ('seq.firreg', 'seq.compreg', 'seq.compreg.ce', 'seq.firmem.read_port', 'seq.read_port', 'seq.firmem')
CONST = ('hw.constant',)

def parse(path):
    """{module: {'args': [names], 'defs': {value: (op, operands)}, 'out': [values], 'insts': [...]}}"""
    mods, cur = {}, None
    for line in open(path):
        m = re.match(r'^  hw\.module (?:private )?@(\w+)\((.*)', line)
        if m:
            cur = mods[m.group(1)] = {'args': re.findall(r'in %([\w.$]+) :', m.group(2)),
                                      'defs': {}, 'out': [], 'insts': []}
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith('hw.output'):
            cur['out'] = re.findall(r'%([\w.$]+)', s.split(':')[0])
            continue
        m = re.match(r'^(%[\w.$]+(?::\d+)?(?:\s*,\s*%[\w.$]+)*)\s*=\s*([\w.]+)\s*(.*)$', s)
        if not m:
            continue
        lhs, op, rest = m.groups()
        names = re.findall(r'%([\w.$]+)', lhs)
        operands = re.findall(r'%([\w.$]+)', rest.split(' : ')[0] if op != 'hw.instance' else rest)
        if op == 'hw.instance':
            im = re.match(r'"(\w+)"\s+@(\w+)\((.*)\)\s*->', rest)
            if im:
                ins = re.findall(r'(\w+):\s*%([\w.$]+)', im.group(3))
                cur['insts'].append((im.group(1), im.group(2), ins, names))
                for k, n in enumerate(names):
                    cur['defs'][n] = ('hw.instance', [im.group(1), str(k)])
            continue
        for n in names:
            cur['defs'][n] = (op, operands)
    return mods

def kind(mod, v, seen=None):
    """'reg' | 'logic' | 'port' | ('inst', inst, k): what drives value v within module mod, through wiring."""
    seen = seen if seen is not None else set()
    if v in seen:
        return 'reg'
    seen.add(v)
    if v in mod['args']:
        return 'port'
    d = mod['defs'].get(v)
    if d is None:
        return 'port'
    op, ops = d
    if op in REG or op.startswith('seq.firmem') or op.startswith('seq.firreg'):
        return 'reg'
    if op in CONST:
        return 'const'
    if op == 'hw.instance':
        return ('inst', ops[0], int(ops[1]))
    if op in WIRING or (op == 'hw.array_get' and len(ops) >= 2 and mod['defs'].get(ops[1], ('',))[0] in CONST):
        ks = [kind(mod, o, seen) for o in ops]
        if any(k == 'logic' for k in ks):
            return 'logic'
        insts = [k for k in ks if isinstance(k, tuple)]
        return insts[0] if insts else ('reg' if 'reg' in ks else ks[0] if ks else 'const')
    return 'logic'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mlir'); ap.add_argument('--top', default='foldsim_sim'); ap.add_argument('--list', type=int, default=12)
    a = ap.parse_args()
    mods = parse(a.mlir)
    top = mods[a.top]
    groups = {i: (mname, ins, res) for i, mname, ins, res in top['insts'] if i.startswith('foldsim_slr')}
    width = {}
    for line in open(a.mlir):
        for n, w in re.findall(r'in %([\w.$]+) : i(\d+)', line):
            width.setdefault(n, int(w))
    stats = C.Counter(); logic = C.defaultdict(lambda: [0, 0])
    def judge(src_part, src_mod, v):
        k = kind(src_mod, v)
        if isinstance(k, tuple):                       # an instance output read through the level
            _, inst, idx = k
            if inst in groups:
                gm = mods[groups[inst][0]]
                return inst, kind(gm, gm['out'][idx]) if idx < len(gm['out']) else 'port', gm['out'][idx] if idx < len(gm['out']) else v
            return inst, 'reg', v                       # a class copy's output: registered inside (judged per class)
        return src_part, k, v
    # Level -> group and group -> group (a group's inputs).
    for inst, (mname, ins, res) in groups.items():
        for port, v in ins:
            if port in ('hostClock',):
                continue
            src, k, name = judge('level', top, v)
            w = width.get(port, 1)
            stats[(k == 'logic', 'in')] += 1; stats[(k == 'logic', 'in_bits')] += w
            if k == 'logic':
                key = (src, inst, re.sub(r'\d+', '#', name)); logic[key][0] += 1; logic[key][1] += w
    # Group -> level (a group's outputs read by level logic).
    users = C.Counter()
    for v, (op, ops) in top['defs'].items():
        if op != 'hw.instance':
            for o in ops:
                users[o] += 1
    for inst, (mname, ins, res) in groups.items():
        gm = mods[mname]
        for idx, r in enumerate(res):
            if not users.get(r):
                continue                                # read only by other groups (judged as their inputs) or the ports
            k = kind(gm, gm['out'][idx]) if idx < len(gm['out']) else 'port'
            stats[(k == 'logic', 'out')] += 1
            if k == 'logic':
                key = (inst, 'level', re.sub(r'\d+', '#', gm['out'][idx])); logic[key][0] += 1
    reg_in, log_in = stats[(False, 'in')], stats[(True, 'in')]
    print(f'cross-group edges: group inputs {reg_in + log_in} ({stats[(False, "in_bits")] + stats[(True, "in_bits")]} bits): '
          f'registered {reg_in}, LOGIC {log_in} ({stats[(True, "in_bits")]} bits); group outputs read by the level: '
          f'{stats[(False, "out")] + stats[(True, "out")]}, LOGIC {stats[(True, "out")]}')
    for (s, d, n), (c, w) in sorted(logic.items(), key=lambda x: -x[1][0])[:a.list]:
        print(f'  LOGIC {s} -> {d}: {c} x {n}' + (f' ({w} bits)' if w else ''))
    sys.exit(1 if logic else 0)

if __name__ == '__main__':
    main()
