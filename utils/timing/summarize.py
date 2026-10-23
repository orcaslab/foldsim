#!/usr/bin/env python3
"""Group a timing screen's end points by RTL unit.

usage: summarize.py <screen dir> [--min-levels N] [--ep ep.csv [--native MODULE=PREFIX]]

A unit is the instance path of an end point with generate indices replaced
by '*' (g_banks_3_bank -> g_banks_*_bank), relative to its fold class when
the design is folded. For each unit: end point bits at or above the
threshold, the deepest and most common levels, and the deepest path's start
and end. With --ep, Vivado's failing end points at or above the threshold
are grouped the same way beside them.
"""
import argparse
import collections
import csv
import re

import validate

INDEX = re.compile(r'_\d+(?=_|/|$)')


def unit(key):
    d = key.rpartition('/')[0]
    return INDEX.sub('_*', d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('screen')
    ap.add_argument('--min-levels', type=int, default=10)
    ap.add_argument('--ep')
    ap.add_argument('--native', metavar='MODULE=PREFIX')
    ap.add_argument('--top', type=int, default=40)
    args = ap.parse_args()

    inst = validate.load_instances(f'{args.screen}/instances.csv')
    classes = validate.FOLD_CLASSES
    if args.native:
        module, prefix = args.native.split('=')
        classes = {module: prefix}
    paths = validate.class_paths(inst, classes)

    groups = collections.defaultdict(
        lambda: {'n': 0, 'max': -1, 'hist': collections.Counter(),
                 'worst': None})
    with open(f'{args.screen}/endpoints.csv') as f:
        r = csv.reader(f)
        next(r)
        for levels, kind, skind, fo, cross, sctx, start, ectx, end in r:
            levels = int(levels)
            if levels < args.min_levels:
                continue
            cls, p = paths[int(ectx)]
            full = f'{p}/{end}' if p else end
            g = groups[(cls, unit(validate.norm(full)))]
            g['n'] += 1
            g['hist'][levels] += 1
            if levels > g['max']:
                sp = paths[int(sctx)][1]
                g['max'] = levels
                g['worst'] = (f'{sp}/{start}' if sp else start, full, fo)

    viv = collections.defaultdict(lambda: [0, 0])  # n, max levels
    if args.ep:
        with open(args.ep) as f:
            r = csv.reader(f)
            next(r)
            for slack, levels, logic, net, start, end in r:
                if int(levels) < args.min_levels:
                    continue
                key = validate.ep_key(end, args.native)
                if key is None:
                    continue
                v = viv[(key[0], unit(key[1]))]
                v[0] += 1
                v[1] = max(v[1], int(levels))

    order = sorted(groups, key=lambda k: (-groups[k]['max'], -groups[k]['n']))
    print(f'end point bits of at least {args.min_levels} levels, by unit '
          f'({sum(g["n"] for g in groups.values())} in {len(groups)} units)')
    hdr = f'{"max":>4} {"mode":>4} {"bits":>8}'
    if args.ep:
        hdr += f' {"viv n":>7} {"viv max":>7}'
    print(hdr + '  unit')
    for k in order[:args.top]:
        g = groups[k]
        mode = g['hist'].most_common(1)[0][0]
        line = f'{g["max"]:4d} {mode:4d} {g["n"]:8d}'
        if args.ep:
            v = viv.get(k, [0, 0])
            line += f' {v[0]:7d} {v[1]:7d}'
        print(f'{line}  {k[0] or ""}: {k[1]}')
        s, e, fo = g['worst']
        print(f'{"":>20}from {s}\n{"":>20}to   {e}')
    if args.ep:
        missed = sorted((k for k in viv if k not in groups),
                        key=lambda k: -viv[k][0])
        print(f'\nVivado units at or above {args.min_levels} levels the '
              f'screen ranks below it ({len(missed)}):')
        for k in missed[:args.top]:
            print(f'{viv[k][0]:8d} {viv[k][1]:4d}  {k[0]}: {k[1]}')


if __name__ == '__main__':
    main()
