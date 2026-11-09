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

"""The state scratch as built: a generated simulator's block and ultra RAM, by
kind, priced as the fit check prices it (foldsim_fitdb.memory: URAM288 in
72-bit columns of 4,096 rows, RAMB36 by foldsim_cost.mem_cost), split into
the memories the state scratch places (the scratch's: a state-in-ram-pp
class's columns, resident rows and landing window, packed and state-in-RAM
rows, folded pipelines' stores, on-chip pages) and every other memory (the
fold's others: links, read buffers and rings, slot stores, kept stores,
target memories...), and set against the planner's own figures (its
`state-scratch: planned` line).

foldsim_scratch_built.py <gen dir or foldsim_sim.mlir> [--log fold.log]
[--json out.json]

As a module: built(path) -> {'scratch': {uram, bram}, 'other': {...}}, in
primitives; planned(log text) -> the planner's {device, planned, fill} per
kind, in bits."""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foldsim_cost  # noqa: E402
import foldsim_fitdb  # noqa: E402

URAM_BITS = 288 * 1024
BRAM_BITS = 36 * 1024
# The memories the state scratch places, by the names the fold gives them.
SCRATCH = re.compile(r'pp_col\d+|pp_res\d+|region\d+_sir|fold\d+_store|region\d+_pages')


def scratch_memory(name):
    return bool(SCRATCH.fullmatch(name))


def built(path):
    """{'scratch': {'uram', 'bram'}, 'other': {...}, 'memories': {name: kind
    counts...}} of a generated simulator, every instance of every module from
    foldsim_sim counted."""
    sim = path if path.endswith('.mlir') else foldsim_fitdb.sim_path(path)
    mods, _ = foldsim_cost.parse(sim)
    out = {'scratch': {'uram': 0.0, 'bram': 0.0}, 'other': {'uram': 0.0, 'bram': 0.0}}
    names = {}

    def walk(name):
        m = mods[name]
        for rec in m.mems:
            mm = foldsim_fitdb.memory(rec)
            kind = 'scratch' if scratch_memory(rec[0]) else 'other'
            for r in ('uram', 'bram'):
                out[kind][r] += mm[r]
            if mm['uram'] or mm['bram']:
                key = re.sub(r'\d+', '#', rec[0])
                n = names.setdefault((kind, key), [0, 0.0, 0.0])
                n[0] += 1
                n[1] += mm['uram']
                n[2] += mm['bram']
        for child, _ in m.insts:
            if child in mods:
                walk(child)
    if 'foldsim_sim' not in mods:
        sys.exit(f'{sim}: no foldsim_sim module')
    walk('foldsim_sim')
    out['memories'] = {f'{k} {key}': v for (k, key), v in sorted(names.items(),
                                                               key=lambda x: -(x[1][1] * 8 + x[1][2]))}
    return out


def planned(log):
    """The planner's figures from the fold's log: per kind the device's bits
    at the fill (`device`) and the bits it planned (`planned`: the models'
    memories, the reserve, the fold stores and the rows it kept), from its
    `state-scratch: planned` line; None without one."""
    m = re.search(r'state-scratch: planned ultra (\d+) of (\d+), block (\d+) of (\d+)', log)
    if not m:
        return None
    pu, du, pb, db = (int(x) for x in m.groups())
    return {'uram': {'planned': pu, 'device': du}, 'bram': {'planned': pb, 'device': db}}


def table(b, p):
    """Lines: per kind, the planner's planned bits against the built (the
    scratch's, the others', the total), in bits and as a share."""
    lines = ['state scratch as built (bits; primitives in parentheses):',
             '  kind   planned       built: scratch   others        total         built/planned  of device']
    for k, bits in (('uram', URAM_BITS), ('bram', BRAM_BITS)):
        s, o = b['scratch'][k], b['other'][k]
        tot = (s + o) * bits
        pl = p[k]['planned'] if p else 0
        dev = p[k]['device'] if p else 0
        ratio = f'{tot / pl:.3f}' if pl else '-'
        share = f'{tot / dev:.3f}' if dev else '-'
        lines.append(f'  {k:5s}  {pl:12,d}  {s * bits:14,.0f} ({s:.0f})  {o * bits:12,.0f} ({o:.0f})  '
                     f'{tot:12,.0f}  {ratio:>13s}  {share}')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('path')
    ap.add_argument('--log', default='')
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    b = built(a.path)
    p = planned(open(a.log).read()) if a.log else None
    print('\n'.join(table(b, p)))
    for key, (n, u, br) in list(b['memories'].items())[:12]:
        print(f'  {key}: {n} memories, {u:.0f} URAM288, {br:.0f} RAMB36')
    if a.json:
        json.dump({'built': b, 'planned': p}, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
