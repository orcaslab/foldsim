#!/usr/bin/env python3
# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Column locality of a state-in-ram-pp class (measurement only).

foldsim_pp_columns.py <ppdump file> <ppcols file> [--depth D ...] [--cluster-columns K ...]

Inputs, from a fold run with FOLDSIM_PP_DUMP=<p> and FOLDSIM_PP_COLUMNS=<c>:
<p>.<class> (the row planner's candidates: index, name, width, enabled, group, level, row, hazard, next stage :
the candidates its next value reads) and <c>.<class> (the column build's layout: name, width, row, first bit in the
row's word, enable run).

Per 72-bit column (a RAM, a row each pass), under a register assignment:
  writers   the registers stored in the column (any row): its write port's P:1 select gathers their next values;
  wclust    their distinct hierarchy clusters (the path's first D components);
  readers   the candidates whose next value reads a register of the column (their cones read its output);
  rclust    the readers' distinct clusters.
(Readers through the outputs, ready cones and stores are not in the planner's graph: a floor.)

Assignments: `today` (the column build's layout) and, for each K, `cluster K` (the design note's A + B dry run):
registers clustered by hierarchy (a path-prefix tree, a subtree a cluster while its widest row fits K columns,
else its children; siblings then merged first-fit while the merged widest row fits), and each cluster given a
column range of its widest row's columns, used by every row; within a row, a cluster's registers laid out by
enable run, each run padded to the lane (the class's grain), as the column build does."""

import argparse
import collections
import math
import re
import statistics


def load(pp_dump, pp_cols):
    cand = {}
    deps = {}
    for line in open(pp_dump):
        head, _, tail = line.partition(':')
        f = head.split()
        k, name = int(f[0]), f[1]
        cand[k] = name
        deps[k] = [int(x) for x in tail.split()]
    regs = []
    grain = 8
    for line in open(pp_cols):
        if line.startswith('#'):
            m = re.search(r'grain (\d+)', line)
            if m:
                grain = int(m.group(1))
            continue
        name, width, row, bit, en = line.split()
        regs.append(dict(name=name, width=int(width), row=int(row), bit=int(bit), en=int(en)))
    # Readers: candidate names whose next value reads each name.
    readers = collections.defaultdict(set)
    for k, ds in deps.items():
        for d in ds:
            if d in cand:
                readers[cand[d]].add(cand[k])
    return regs, readers, grain


def cluster_of(name, depth):
    return '/'.join(name.split('/')[:depth])


def columns_of(bit, width):
    return range(bit // 72, (bit + width - 1) // 72 + 1)


def metrics(regs, place, readers, depth):
    """Per column: writers, writer clusters, readers, reader clusters (place: reg index -> (row, bit))."""
    cols = collections.defaultdict(lambda: dict(w=set(), r=set()))
    for i, r in enumerate(regs):
        row, bit = place[i]
        for c in columns_of(bit, r['width']):
            cols[c]['w'].add(r['name'])
            cols[c]['r'] |= readers.get(r['name'], set())
    out = []
    for c in sorted(cols):
        w, rd = cols[c]['w'], cols[c]['r']
        out.append(dict(col=c, writers=len(w), wclust=len({cluster_of(n, depth) for n in w}),
                        readers=len(rd), rclust=len({cluster_of(n, depth) for n in rd})))
    return out


def lay_rows(members, regs, grain):
    """A cluster's per-row layout: {row: [(reg index, offset)]}, each enable run from a lane boundary and padded
    to whole lanes (the always-written run first), and the widest row's bits."""
    rows = collections.defaultdict(list)
    for i in members:
        rows[regs[i]['row']].append(i)
    layout, widest = {}, 0
    for row, idx in rows.items():
        runs = collections.OrderedDict()
        for i in sorted(idx, key=lambda i: (regs[i]['en'] >= 0, regs[i]['en'], regs[i]['name'])):
            runs.setdefault(regs[i]['en'], []).append(i)
        at = 0
        place = []
        for en, run in runs.items():
            at = math.ceil(at / grain) * grain
            for i in run:
                place.append((i, at))
                at += regs[i]['width']
            at = math.ceil(at / grain) * grain
        layout[row] = place
        widest = max(widest, at)
    return layout, widest


def clusters_by_hierarchy(regs, grain, cap_bits):
    """Path-prefix tree; a subtree is a cluster while its widest row fits cap_bits, else its children; siblings
    merged first-fit while the merged widest row fits."""
    def split(members, depth):
        _, widest = lay_rows(members, regs, grain)
        names = [regs[i]['name'].split('/') for i in members]
        if widest <= cap_bits or all(len(n) <= depth for n in names):
            return [members]
        kids = collections.OrderedDict()
        for i, n in zip(members, names):
            kids.setdefault(n[depth] if len(n) > depth else '', []).append(i)
        if len(kids) == 1:
            return split(members, depth + 1)
        parts = []
        for kid in kids.values():
            parts += split(kid, depth + 1)
        # Merge small siblings first-fit (largest first) while the merged widest row fits.
        parts.sort(key=lambda p: -lay_rows(p, regs, grain)[1])
        merged = []
        for p in parts:
            for m in merged:
                if lay_rows(m + p, regs, grain)[1] <= cap_bits:
                    m += p
                    break
            else:
                merged.append(list(p))
        return merged
    return split(list(range(len(regs))), 0)


def assign_by_clusters(regs, grain, cap_bits):
    clusters = clusters_by_hierarchy(regs, grain, cap_bits)
    place, base = {}, 0
    for members in clusters:
        layout, widest = lay_rows(members, regs, grain)
        for row, items in layout.items():
            for i, off in items:
                place[i] = (row, base * 72 + off)
        base += max(1, math.ceil(widest / 72))
    return place, base, len(clusters)


def assign_by_prefix(regs, grain, depth):
    """Coarse ranges: a cluster per path prefix of `depth` components, each a column range of its widest row,
    every row's registers of the cluster laid out in it."""
    groups = collections.OrderedDict()
    for i, r in sorted(enumerate(regs), key=lambda x: x[1]['name']):
        groups.setdefault(cluster_of(r['name'], depth), []).append(i)
    place, base = {}, 0
    for members in groups.values():
        layout, widest = lay_rows(members, regs, grain)
        for row, items in layout.items():
            for i, off in items:
                place[i] = (row, base * 72 + off)
        base += max(1, math.ceil(widest / 72))
    return place, base, len(groups)


def assign_rowwise(regs, grain):
    """Locality within a row only: each row's registers sorted by path, then laid out by enable run within
    each 3-component subtree (the columns of a row then hold neighbours; a column still meets other subtrees in
    other rows). Same column count as a packed layout."""
    place, cols = {}, 0
    rows = collections.defaultdict(list)
    for i, r in enumerate(regs):
        rows[r['row']].append(i)
    for row, idx in rows.items():
        at = 0
        groups = collections.OrderedDict()
        for i in sorted(idx, key=lambda i: regs[i]['name']):
            groups.setdefault(cluster_of(regs[i]['name'], 4), []).append(i)
        for members in groups.values():
            layout, _ = lay_rows(members, regs, grain)
            for i, off in layout.get(row, []):
                place[i] = (row, at + off)
            at += max(off + regs[i]['width'] for i, off in layout.get(row, [(members[0], 0)]))
            at = math.ceil(at / grain) * grain
        cols = max(cols, math.ceil(at / 72))
    return place, cols, None


def replica_metrics(regs, place, readers, depth, rdepth):
    """state-in-ram-pp-replicas=rdepth: per output register (a column's replica per subtree of the registers it
    holds), its registers, their readers (cones) and the readers' clusters at `depth`."""
    outs = collections.defaultdict(lambda: dict(w=set(), r=set()))
    for i, r in enumerate(regs):
        row, bit = place[i]
        for c in columns_of(bit, r['width']):
            key = (c, cluster_of(r['name'], rdepth))
            outs[key]['w'].add(r['name'])
            outs[key]['r'] |= readers.get(r['name'], set())
    return [dict(col=k, writers=len(v['w']), wclust=len({cluster_of(n, depth) for n in v['w']}),
                 readers=len(v['r']), rclust=len({cluster_of(n, depth) for n in v['r']}))
            for k, v in outs.items()]


def summary(ms, key):
    xs = [m[key] for m in ms]
    xs.sort()
    p90 = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
    return f'{statistics.mean(xs):7.1f} {statistics.median(xs):6.0f} {p90:5d} {max(xs):5d}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('pp_dump')
    ap.add_argument('pp_cols')
    ap.add_argument('--depth', type=int, action='append')
    ap.add_argument('--cluster-columns', type=int, action='append')
    ap.add_argument('--replicas', type=int, action='append',
                    help='also today\'s layout with state-in-ram-pp-replicas=D: per output register (its column '
                         'count is the replicas\')')
    a = ap.parse_args()
    regs, readers, grain = load(a.pp_dump, a.pp_cols)
    depths = a.depth or [3, 4, 5]
    ks = a.cluster_columns or [4, 8]
    today = {i: (r['row'], r['bit']) for i, r in enumerate(regs)}
    today_cols = max(c for i, r in enumerate(regs) for c in columns_of(r['bit'], r['width'])) + 1
    variants = [('today', today, today_cols, None)]
    for k in ks:
        place, cols, nclust = assign_by_clusters(regs, grain, k * 72)
        variants.append((f'cluster K={k}', place, cols, nclust))
    variants.append(('row-sorted',) + assign_rowwise(regs, grain))
    for d in (3, 4):
        place, cols, n = assign_by_prefix(regs, grain, d)
        variants.append((f'ranges d={d}', place, cols, n))
    print(f'{len(regs)} registers in rows, grain {grain}; per column: mean median p90 max')
    for depth in depths:
        print(f'\nclusters = path prefix of {depth} components')
        print(f'  {"assignment":14s} {"columns":>8s} {"clusters":>8s} | {"writers":^24s} | {"writer clusters":^24s} '
              f'| {"readers (cones)":^24s} | {"reader clusters":^24s}')
        for name, place, cols, nclust in variants:
            ms = metrics(regs, place, readers, depth)
            print(f'  {name:14s} {cols:8d} {str(nclust or "-"):>8s} | {summary(ms, "writers")} | '
                  f'{summary(ms, "wclust")} | {summary(ms, "readers")} | {summary(ms, "rclust")}')
        for rd in a.replicas or []:
            ms = replica_metrics(regs, today, readers, depth, rd)
            print(f'  {"replicas d=" + str(rd):14s} {len(ms):8d} {"(regs)":>8s} | {summary(ms, "writers")} | '
                  f'{summary(ms, "wclust")} | {summary(ms, "readers")} | {summary(ms, "rclust")}')


if __name__ == '__main__':
    main()
