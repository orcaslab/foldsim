#!/usr/bin/env python3
"""Screen vs Vivado levels, split by whether Vivado retimed the paths.

usage: retime_split.py <screen dir> <ep.csv> MODULE=PREFIX

A register's Vivado paths count as retimed when the start or end cell of
any of its failing paths is a retiming register (_fret/_bret).
"""
import collections
import csv
import re
import sys

import numpy as np

import validate

RETIMED = re.compile(r'_[fb]ret')


def main():
    screen, ep, native = sys.argv[1:4]
    module, prefix = native.split('=')
    inst = validate.load_instances(f'{screen}/instances.csv')
    paths = validate.class_paths(inst, {module: prefix})
    ours = collections.defaultdict(int)
    with open(f'{screen}/endpoints.csv') as f:
        r = csv.reader(f)
        next(r)
        for levels, kind, skind, fo, cross, sctx, start, ectx, end in r:
            cls, p = paths[int(ectx)]
            base = end.rsplit('.', 1)[0]
            key = (cls, validate.norm(f'{p}/{base}' if p else base))
            ours[key] = max(ours[key], int(levels))
    viv = collections.defaultdict(lambda: [0, False])
    with open(ep) as f:
        r = csv.reader(f)
        next(r)
        for slack, levels, logic, net, start, end in r:
            key = validate.ep_key(end, True)
            if key not in ours:
                continue
            v = viv[key]
            v[0] = max(v[0], int(levels))
            v[1] |= bool(RETIMED.search(start) or RETIMED.search(end))
    for retimed in (False, True):
        keys = [k for k, v in viv.items() if v[1] == retimed]
        a = np.array([ours[k] for k in keys])
        b = np.array([viv[k][0] for k in keys])
        if len(keys) < 3:
            continue
        ratio = np.median(a[b > 0] / b[b > 0]) if (b > 0).any() else float('nan')
        print(f'retimed={retimed!s:5}: {len(keys):5d} registers, Spearman '
              f'{validate.spearman(a, b):.3f}, median screen/Vivado {ratio:.2f}, '
              f'screen-Vivado median {np.median(a - b):+.1f}')


if __name__ == '__main__':
    main()
