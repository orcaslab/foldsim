#!/usr/bin/env python3
"""Pass splitting, priced: the lanes of a model's logic and what folding them saves.

A copy's logic evaluated over P passes per instance costs less only where the
passes run the SAME logic: a static FPGA keeps every stage's configuration in
its LUTs, so P different stages are P times the LUTs of one. Pass splitting
therefore folds *lanes*: fanout-free cones of the model's logic (MFFCs) of one
structure (operations, attributes and types alike; their leaves -- state,
inputs, constants, logic outside the cone -- abstracted by position), R of
them evaluated on ceil(R/P) physical lanes, one replica a pass. A leaf position
all replicas share costs nothing; one whose values differ is a P:1 choice by
the pass (a constant one a small table). A replica's result is held in a
register (the cut) from its pass to the end of the evaluation, so nothing
downstream of a lane may feed a lane (checked).

`census(path, module, passes)` returns the groups and the LUTs saved at each P;
`main` prints them. The same structure test and cost rule as
foldsim-fold-static's --pass-split (the compiler estimates cost by op width;
this module prices with foldsim_cost's per-op LUT6 model).
"""
import argparse, json, math, re, sys
from collections import defaultdict
import foldsim_cost as cost

SSA = cost.SSA
NONLOGIC = cost.REGS | cost.CONSTS | {'seq.firmem', 'seq.firmem.read_port', 'seq.firmem.write_port',
                                       'seq.firmem.read_write_port', 'hw.instance', 'hw.output',
                                       'sv.verbatim', 'sv.attribute', 'seq.from_clock', 'seq.to_clock',
                                       'seq.const_clock'}
STR = re.compile(r'"(?:[^"\\]|\\.)*"')
NAMEATTR = re.compile(r'\{[^{}]*\}')


def norm(rhs):
    """An op's text without its value names and string attributes (names, hints)."""
    body = rhs.split(' : ')
    head = re.sub(SSA, '%', STR.sub('""', body[0]))
    head = NAMEATTR.sub('{}', head)
    return head + ' : ' + ' : '.join(body[1:])


def read_module(path, module):
    """(ops, users, costs, types) of one hw.module: ops[res] = (norm, operands);
    costs[res] = LUT6 estimate (foldsim_cost's, enables refunded)."""
    hdr = re.compile(r'^\s*hw\.module(?:\.extern)?(?:\s+private)?\s+@([\w$.]+)(?:<[^>]*>)?\((.*)\)')
    ops, costs, types = {}, {}, {}
    sinks = []                       # operands of non-logic users (registers, memories, outputs, instances)
    cur = None
    for line in open(path):
        s = line.strip()
        if cur is None:
            if s.startswith('hw.module'):
                m = hdr.match(line)
                if m and m.group(1) == module:
                    cur = cost.Mod(module, m.group(2))
            continue
        if s.startswith('hw.output') or s.startswith('seq.firmem.write_port') or \
                s.startswith('seq.firmem.read_write_port'):
            sinks += re.findall(SSA, s.split(' : ')[0])
            if s.startswith('hw.output'):
                break
            continue
        if ' = ' not in s or not s.startswith('%'):
            continue
        lhs, rhs = s.split(' = ', 1)
        op = rhs.split()[0]
        res = re.findall(SSA, lhs)
        operands = re.findall(SSA, rhs.split(' : ')[0])
        for r in res:
            types[r] = cost.result_type(rhs)
        if op in cost.CONSTS:
            cur.consts.update(res)
            if op == 'hw.constant':
                cost.track(cur, op, rhs, res, operands)
            continue
        if op in NONLOGIC:
            sinks += operands
            if op in cost.REGS and operands:
                cur.regs[res[0]] = operands[0]
            continue
        c = 0.0
        if op not in cost.FREE:
            c = cost.lut_cost(op, rhs, cur, operands) or 0.0
        cost.track(cur, op, rhs, res, operands)
        if op == 'comb.mux' and len(operands) == 3 and c:
            cur.muxes[res[0]] = (c, operands[1], operands[2])
        if len(res) != 1:
            sinks += operands        # multi-result logic (struct_explode): a boundary
            continue
        ops[res[0]] = (norm(rhs), operands)
        costs[res[0]] = c
    if cur is None:
        sys.exit(f'foldsim_split: no module {module}')
    # Enables (a mux holding its register) cost nothing (foldsim_cost.refund_enables).
    for r, nxt in cur.regs.items():
        v = nxt
        for _ in range(6):
            m = cur.muxes.get(v)
            if not m:
                break
            _, a, b = m
            if r not in (a, b):
                break
            costs[v] = 0.0
            v = b if a == r else a
    users = defaultdict(int)
    for r, (_, operands) in ops.items():
        for o in operands:
            users[o] += 1
    for o in sinks:
        users[o] += 1
    return ops, users, costs, types, set(sinks), cur.consts, set(cur.regs)


GATHER = ('hw.array_create', 'comb.concat', 'hw.array_inject', 'hw.struct_create')


def mffc(root, ops, users):
    """The fanout-free cone of `root`: root and every logic op all of whose uses
    are inside it (reference counting)."""
    cone, ref, stack = {root}, {}, [root]
    while stack:
        v = stack.pop()
        for o in ops[v][1]:
            if o not in ops or o in cone:
                continue
            ref[o] = ref.get(o, users[o]) - 1
            if ref[o] == 0:
                cone.add(o)
                stack.append(o)
    return cone


def signature(root, cone, ops, types):
    """(structure, leaves): the cone in post-order from root, operands as cone
    indices or leaf positions; leaves = [(value, type)] by position."""
    index, sig, leaves = {}, [], []
    stack = [(root, False)]
    while stack:
        v, done = stack.pop()
        if v in index:
            continue
        if not done:
            stack.append((v, True))
            for o in reversed(ops[v][1]):
                if o in cone and o not in index:
                    stack.append((o, False))
            continue
        refs = []
        for o in ops[v][1]:
            if o in cone:
                refs.append(index[o])
            else:
                refs.append(-1 - len(leaves))
                leaves.append((o, types.get(o, '?')))
        index[v] = len(index)
        sig.append((ops[v][0], tuple(refs)))
    return (tuple(sig), tuple(t for _, t in leaves)), [l for l, _ in leaves], list(index)


def census(path, module, passes=(2, 4, 8, 16, 32), min_ops=2, max_cone=20000):
    ops, users, costs, types, sinks, consts, regs = read_module(path, module)
    total = sum(costs.values())
    # Candidate roots: a fanout point, a value leaving the logic, or a part of a gather.
    cand = []
    for v, (n, operands) in ops.items():
        multi = users[v] != 1 or v in sinks
        cand.append(v) if multi else None
    gathered = set()
    for v, (n, operands) in ops.items():
        if n.startswith(GATHER):
            gathered.update(o for o in operands if o in ops and users[o] == 1)
    cand = list(dict.fromkeys(cand + sorted(gathered)))
    groups = defaultdict(list)
    for v in cand:
        cone = mffc(v, ops, users)
        if len(cone) < min_ops or len(cone) > max_cone:
            continue
        key, leaves, order = signature(v, cone, ops, types)
        c = sum(costs[u] for u in cone)
        if c <= 0:
            continue
        groups[key].append((v, cone, leaves, c))
    # Greedy, largest group saving first; members disjoint from what is taken.
    taken, chosen = set(), []
    for key, mem in sorted(groups.items(), key=lambda kv: -sum(m[3] for m in kv[1])):
        mem = [m for m in mem if not (m[1] & taken)]
        if len(mem) < 2:
            continue
        # Disjoint among themselves (a cone inside another of the same shape: keep the first).
        sel, own = [], set()
        for m in mem:
            if not (m[1] & own):
                sel.append(m); own |= m[1]
        if len(sel) < 2:
            continue
        chosen.append((key, sel))
        taken |= own
    # Exactness: no lane's cone or leaves downstream of a held result (iterate).
    while True:
        roots = {m[0] for _, sel in chosen for m in sel}
        down, stack = set(), list(roots)
        fwd = defaultdict(list)
        for v, (_, operands) in ops.items():
            for o in operands:
                fwd[o].append(v)
        while stack:
            v = stack.pop()
            for u in fwd[v]:
                if u not in down:
                    down.add(u); stack.append(u)
        keep = []
        for key, sel in chosen:
            bad = any((m[1] & down) or any(l in down or l in roots for l in m[2]) for m in sel)
            keep.append((key, sel)) if not bad else None
        if len(keep) == len(chosen):
            break
        chosen = keep
    out = []
    for key, sel in chosen:
        R, C = len(sel), sel[0][3]
        width = [cost.width(t) for t in key[1]]
        npos = len(width)
        # Leaf positions by their values across replicas: shared, constant, or chosen.
        cols = defaultdict(int)
        shared = constant = chosen_bits = state_bits = 0
        seen = set()
        for i in range(npos):
            col = tuple(m[2][i] for m in sel)
            if col in seen:
                continue
            seen.add(col)
            if len(set(col)) == 1:
                shared += width[i]
            elif all(v in consts for v in col):
                constant += width[i]
            else:
                chosen_bits += width[i]
                if all(v in regs for v in col):
                    state_bits += width[i]
        g = dict(root=sel[0][0], replicas=R, ops=len(sel[0][1]), lut=C, shared_bits=shared,
                 const_bits=constant, select_bits=chosen_bits, state_bits=state_bits, hold_bits=cost.width(types.get(sel[0][0], 'i1')),
                 sig=key[0][-1][0][:80])
        out.append(g)
    return total, out


def saving(g, P, rows=False):
    """LUTs saved by folding group g over P passes (ceil(R/P) lanes); <= 0: not folded.
    rows: a replica's own state read as a row of its store at its pass (the address
    chooses it), so only the other chosen leaves cost a P:1 choice."""
    R, P = g['replicas'], min(P, g['replicas'])
    L = math.ceil(R / P)
    bits = g['select_bits'] - (g.get('state_bits', 0) if rows else 0)
    sel = L * (bits * (P - 1) / 3 + g['const_bits'] * 0.25 * math.ceil(P / 32))
    return (R - L) * g['lut'] - sel


def summary(total, groups, passes):
    rows = {}
    for P in passes:
        s = sum(max(0.0, saving(g, P)) for g in groups)
        sr = sum(max(0.0, saving(g, P, True)) for g in groups)
        hold = sum(g['hold_bits'] * g['replicas'] for g in groups if saving(g, P) > 0)
        rows[P] = dict(saved=round(s, 1), frac=round(s / total, 4) if total else 0, hold_bits=hold,
                       groups=sum(1 for g in groups if saving(g, P) > 0), saved_rows=round(sr, 1))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('models')
    ap.add_argument('module', nargs='+')
    ap.add_argument('--passes', default='2,4,8,16,32')
    ap.add_argument('--top', type=int, default=15, help='groups listed per module')
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    passes = [int(x) for x in a.passes.split(',')]
    res = {}
    for m in a.module:
        total, groups = census(a.models, m, passes)
        rows = summary(total, groups, passes)
        res[m] = dict(lut=round(total, 1), groups=len(groups),
                      lane_lut=round(sum(g['lut'] * g['replicas'] for g in groups), 1),
                      by_p={str(k): v for k, v in rows.items()},
                      top=sorted(groups, key=lambda g: -g['lut'] * g['replicas'])[:a.top], groups_all=groups)
        print(f'{m}: logic {total:,.0f} LUT (cost model), {len(groups)} lane groups covering '
              f'{res[m]["lane_lut"]:,.0f}')
        for P, r in rows.items():
            print(f'  P={P:3d}: saves {r["saved"]:>10,.0f} ({100 * r["frac"]:.1f} %), {r["groups"]} groups, '
                  f'hold {r["hold_bits"]:,} bits; state as rows {r["saved_rows"]:,.0f}')
        for g in res[m]['top'][:a.top]:
            print(f'    x{g["replicas"]:<4d} {g["lut"]:8.1f} LUT {g["ops"]:5d} ops  sel {g["select_bits"]:5d} '
                  f'state {g["state_bits"]:5d} const {g["const_bits"]:4d} shared {g["shared_bits"]:6d}  {g["root"]}  {g["sig"]}')
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
