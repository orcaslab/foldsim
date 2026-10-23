#!/usr/bin/env python3
"""Compare a timing screen with Vivado's routed end points.

usage: validate.py <screen dir> <ep.csv> [--native MODULE=PREFIX[:SUBPATH]]
                   [--metric levels|retimed] [--json FILE]

Matches per RTL register or memory: the instance path inside its fold class
(or below the gate's harness prefix) and the register name, with Vivado's
cell suffixes removed. A key's screen value is the deepest end point bit of
any instance of it; its Vivado values are the deepest path and the worst
slack among its failing end points.
"""
import argparse
import collections
import csv
import json
import re
import sys

import numpy as np

# Fold classes of the static fold: module in the design -> cell prefix in
# the folded netlist.
FOLD_CLASSES = {
    'VX_socket__lift': 'VX_socket__lift_model_copy',
    'VX_core__lift': 'VX_core__lift_model_copy',
    'VX_compute_slice__lift': 'VX_compute_slice__lift_copy',
    'foldsim_target': 'foldsim_target_model_copy',
}

VIVADO_CELL = re.compile(
    r'_reg((\[\d+\])*(_replica(_\d+)?)?(_[fb]ret(_\d+)?)*(_rep(_\d+)?)?'
    r'|_\d+_\d+_\d+_\d+|_bram_\d+)$')
FOLD_SUFFIX = re.compile(
    r'(_ext(_\d+)?|_fold|_read(_\d+)?|_[fb]ret(_\d+)?|_replica(_\d+)?)+$')
FOLD_SHELL = re.compile(r'^fold\d+_')


def norm(s):
    return s.replace('.', '_').replace('[', '_').replace(']', '')


def ep_key(cell_path, native):
    """(class, key) of a Vivado end point pin, or None for fold-shell cells.

    Folded: sim/<class>/<path with 2F for />[/<primitive>]/<pin>.
    Native: <harness>/<path>/<cell>[/<primitive>]/<pin>, <class> being the
    harness prefix.
    """
    parts = cell_path.split('/')
    if native:
        cls = parts[0]
        hier = parts[1:-1]
        # Distributed RAM primitives below the memory cell.
        for i, p in enumerate(hier):
            if p.startswith('Memory_reg') or re.match(r'RAM[A-H]', p):
                hier = hier[:i]
                break
    else:
        if parts[0] != 'sim' or len(parts) < 3:
            return None
        cls = parts[1]
        hier = parts[2].split('2F')
    if not hier or FOLD_SHELL.match(hier[-1]):
        return None
    leaf = VIVADO_CELL.sub('', hier[-1])
    leaf = FOLD_SUFFIX.sub('', leaf)
    return cls, norm('/'.join(hier[:-1] + [leaf]))


def spearman(a, b):
    def rank(x):
        x = np.asarray(x, dtype=float)
        order = np.argsort(x, kind='mergesort')
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x))
        # Average the ranks of ties.
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=ranks)
        return (sums / counts)[inv]
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def endpoints(screen):
    """The rows of a screen's end point table, as dicts."""
    with open(f'{screen}/endpoints.csv') as f:
        yield from csv.DictReader(f)


def load_instances(path):
    inst = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            inst[int(row['id'])] = (int(row['parent']), row['instance'],
                                    row['module'])
    return inst


def class_paths(inst, classes):
    """ctx -> (class cell prefix, path below the class root)."""
    memo = {}

    def get(ctx):
        if ctx in memo:
            return memo[ctx]
        parent, name, module = inst[ctx]
        if module in classes:
            res = (classes[module], '')
        elif ctx == 0:
            res = (None, '')
        else:
            cls, p = get(parent)
            res = (cls, f'{p}/{name}' if p else name)
        memo[ctx] = res
        return res

    sys.setrecursionlimit(10000)
    return {ctx: get(ctx) for ctx in inst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('screen')
    ap.add_argument('ep')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--native', metavar='MODULE=PREFIX[:SUBPATH]',
                    help='native gate: the screened top and the harness '
                    'prefix of its cells (e.g. VX_socket__lift=dut); with '
                    'SUBPATH, only the screened end points below it, which '
                    'the harness instantiates as PREFIX')
    ap.add_argument('--json', help='also write the agreement to this file')
    ap.add_argument('--metric', default='levels', choices=['levels', 'retimed'],
                    help="the screen's column to compare")
    args = ap.parse_args()

    inst = load_instances(f'{args.screen}/instances.csv')
    classes = FOLD_CLASSES
    sub = None
    if args.native:
        module, prefix = args.native.split('=')
        prefix, _, sub = prefix.partition(':')
        classes = {module: prefix}
    paths = class_paths(inst, classes)

    ours = collections.defaultdict(lambda: [0, 0])  # key -> [max levels, n]
    # (class, instance path) -> {register base name: key}, for Vivado names
    # that place a register in the generate scope that assigns it rather
    # than the one that declares it.
    by_dir = collections.defaultdict(dict)
    for row in endpoints(args.screen):
        cls, p = paths[int(row['end_inst'])]
        if cls is None:
            continue
        # Registers of inlined modules are named <instance path>/<name>.
        base = row['end'].rsplit('.', 1)[0]
        full = f'{p}/{base}' if p else base
        if sub:
            if not full.startswith(sub + '/'):
                continue
            full = full[len(sub) + 1:]
        key = (cls, norm(full))
        v = ours[key]
        if not v[1]:
            d, _, base = full.rpartition('/')
            by_dir[(cls, norm(d))][norm(base.rsplit('.', 1)[-1])] = key
        v[0] = max(v[0], int(row[args.metric]))
        v[1] += 1

    def fallback(key):
        cls, k = key
        d, _, leaf = k.rpartition('/')
        best = None
        for name, cand in by_dir.get((cls, d), {}).items():
            if (leaf == name or leaf.endswith('_' + name)) and (
                    best is None or len(name) > len(best[0])):
                best = (name, cand)
        return best[1] if best else key

    viv = collections.defaultdict(lambda: [0, 0.0, 0])  # max lv, min slack, n
    rows = unmatched_rows = shell_rows = 0
    unmatched = collections.Counter()
    with open(args.ep) as f:
        r = csv.reader(f)
        next(r)
        for slack, levels, logic, net, start, end in r:
            rows += 1
            key = ep_key(end, args.native)
            if key is None or key[0] not in classes.values():
                shell_rows += 1
                continue
            if key not in ours and key[1].endswith('_rdata'):
                # The read register of a synchronous RAM is the memory's
                # read port in the design.
                key = (key[0], key[1][:-len('_rdata')])
            if key not in ours:
                key = fallback(key)
            if key not in ours:
                unmatched_rows += 1
                unmatched[key] += 1
                continue
            v = viv[key]
            v[0] = max(v[0], int(levels))
            v[1] = min(v[1], float(slack))
            v[2] += 1

    keys = sorted(viv)
    a = np.array([ours[k][0] for k in keys])
    lv = np.array([viv[k][0] for k in keys])
    sl = np.array([viv[k][1] for k in keys])
    print(f'ep rows {rows}: fold shell / channels {shell_rows}, '
          f'unmatched {unmatched_rows}, matched {rows - shell_rows - unmatched_rows} '
          f'in {len(keys)} registers/memories')
    print(f'screen keys in the classes: {len(ours)}')
    print(f'Spearman(screen levels, Vivado levels) = {spearman(a, lv):.3f}')
    print(f'Spearman(screen levels, -slack)        = {spearman(a, -sl):.3f}')
    ratio = a[lv > 0] / lv[lv > 0]
    print(f'screen/Vivado levels: median {np.median(ratio):.2f}, '
          f'p10 {np.percentile(ratio, 10):.2f}, p90 {np.percentile(ratio, 90):.2f}')

    print('\nVivado levels -> screen levels (registers/memories):')
    for v in range(0, int(lv.max()) + 1):
        sel = a[lv == v]
        if len(sel):
            print(f'  {v:3d}: n={len(sel):6d} screen median {np.median(sel):5.1f} '
                  f'p10 {np.percentile(sel, 10):5.1f} p90 {np.percentile(sel, 90):5.1f}')

    if args.native:
        # The ep.csv of a gate lists every failing end point, so a screened
        # register without one meets timing.
        scr = np.array([v[0] for v in ours.values()])
        fails = np.array([k in viv for k in ours])
        print('\nflag registers at >= T levels (gate lists all failing end '
              'points):')
        print('     T  flagged  precision  recall')
        for t in range(4, 21, 2):
            flagged = scr >= t
            if not flagged.any():
                break
            prec = fails[flagged].mean()
            rec = flagged[fails].mean()
            print(f'  {t:4d} {flagged.sum():8d} {prec:10.2f} {rec:7.2f}')

    if args.json:
        summary = {
            'rows': rows, 'shell': shell_rows, 'unmatched': unmatched_rows,
            'registers': len(keys), 'screened': len(ours),
            'rho_levels': round(spearman(a, lv), 3),
            'rho_slack': round(spearman(a, -sl), 3),
            'ratio': round(float(np.median(ratio)), 2),
            'pairs': [[int(x), int(y), n] for (x, y), n in
                      sorted(collections.Counter(zip(a, lv)).items())],
        }
        if args.native:
            summary['pr'] = [
                [t, int((scr >= t).sum()), round(float(fails[scr >= t].mean()), 3),
                 round(float((scr >= t)[fails].mean()), 3)]
                for t in range(2, 23) if (scr >= t).any()]
        with open(args.json, 'w') as f:
            json.dump(summary, f)

    print(f'\nunmatched Vivado cells (top {args.top} by rows):')
    for (cls, k), n in unmatched.most_common(args.top):
        print(f'  {n:6d} {cls}: {k}')

    order = np.argsort(-a, kind='mergesort')
    print(f'\ndeepest screened registers with failing Vivado end points:')
    for i in order[:args.top]:
        k = keys[i]
        print(f'  screen {a[i]:3d} vivado {lv[i]:3d} slack {sl[i]:7.3f} '
              f'n {viv[k][2]:5d} {k[0]}: {k[1]}')


if __name__ == '__main__':
    main()
