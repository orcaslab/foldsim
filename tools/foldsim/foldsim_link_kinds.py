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

"""The link memories of a generated simulator against the rest, per kind, as
the fit check prices them (foldsim_fitdb.count's totals; each memory by
foldsim_fitdb.memory: URAM288, RAMB36, LUTRAM LUTs): the rest is what
foldsim-fold-static's link-kind-others takes ("<uram>,<ramb36>,<luts>": the
rest's URAM288 and RAMB36, and its logic and LUTRAM LUTs).

foldsim_link_kinds.py <gen dir or foldsim_sim.mlir> [--json out.json]

Prints the table, then the `--link-kind-others` value on its last line."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foldsim_cost  # noqa: E402
import foldsim_fitdb  # noqa: E402


def is_link(name):
    """A link memory: `<producer port>_to_<reader input>_<at>` (and its
    paged and staged forms), as StaticFold names them."""
    return '_to_' in name


def links(path):
    """{'uram', 'bram', 'lutram', 'memories', 'bits'} of the link memories,
    every instance of every module from foldsim_sim counted."""
    sim = path if path.endswith('.mlir') else foldsim_fitdb.sim_path(path)
    mods, _ = foldsim_cost.parse(sim)
    out = {'uram': 0.0, 'bram': 0.0, 'lutram': 0.0, 'memories': 0, 'bits': 0}

    def walk(name):
        m = mods[name]
        for rec in m.mems:
            if not is_link(rec[0]):
                continue
            mm = foldsim_fitdb.memory(rec)
            for r in ('uram', 'bram', 'lutram'):
                out[r] += mm[r]
            out['memories'] += 1
            out['bits'] += rec[1] * rec[2]
        for child, _ in m.insts:
            if child in mods:
                walk(child)
    if 'foldsim_sim' not in mods:
        sys.exit(f'{sim}: no foldsim_sim module')
    walk('foldsim_sim')
    return out


def others(path):
    """(uram, ramb36, luts) the rest of a built simulator takes (its total less
    its link memories): link-kind-others."""
    gen = os.path.dirname(path) if path.endswith('.mlir') else path
    total = foldsim_fitdb.count(gen)['sim']
    lk = links(path)
    return (max(0.0, total.get('uram', 0) - lk['uram']),
            max(0.0, total.get('bram', 0) - lk['bram']),
            max(0.0, total.get('logic', 0) + total.get('lutram', 0) - lk['lutram'])), lk, total


def groups(path):
    """With SLR groups: per group (foldsim_sim_slr<g>, every copy and link in
    it) its resources and its links' as links() counts them, and the rest's
    (uram, ramb36, luts): link-kind-group-others. [] without groups."""
    sim = path if path.endswith('.mlir') else foldsim_fitdb.sim_path(path)
    mods, _ = foldsim_cost.parse(sim)
    out = []
    present = [int(m[len('foldsim_sim_slr'):]) for m in mods
               if m.startswith('foldsim_sim_slr') and m[len('foldsim_sim_slr'):].isdigit()]
    for g in range(max(present) + 1 if present else 0):
        if f'foldsim_sim_slr{g}' not in mods:
            # An empty group: nothing in it.
            out.append({'total': {'uram': 0, 'bram': 0, 'logic': 0, 'lutram': 0},
                        'links': {'uram': 0.0, 'bram': 0.0, 'lutram': 0.0}, 'others': (0.0, 0.0, 0.0)})
            continue
        acc = foldsim_fitdb.subtree(mods, f'foldsim_sim_slr{g}', set(), foldsim_fitdb.new())
        lk = {'uram': 0.0, 'bram': 0.0, 'lutram': 0.0}

        def walk(name):
            for rec in mods[name].mems:
                if is_link(rec[0]):
                    mm = foldsim_fitdb.memory(rec)
                    for r in lk:
                        lk[r] += mm[r]
            for child, _ in mods[name].insts:
                if child in mods:
                    walk(child)
        walk(f'foldsim_sim_slr{g}')
        rest = (max(0.0, acc['uram'] - lk['uram']), max(0.0, acc['bram'] - lk['bram']),
                max(0.0, acc['logic'] + acc['lutram'] - lk['lutram']))
        out.append({'total': {r: acc[r] for r in ('uram', 'bram', 'logic', 'lutram')}, 'links': lk,
                    'others': rest})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('path')
    ap.add_argument('--json', default='')
    ap.add_argument('--groups', action='store_true',
                    help='per SLR group (foldsim_sim_slr<g>): the link-kind-group-others value')
    a = ap.parse_args()
    if a.groups:
        gs = groups(a.path)
        for g, x in enumerate(gs):
            t, lk = x['total'], x['links']
            print(f"group {g}: {t['uram']:.0f} URAM288, {t['bram']:.1f} RAMB36, {t['logic']:,.0f} logic + "
                  f"{t['lutram']:,.0f} LUTRAM LUTs; links {lk['uram']:.0f} URAM288, {lk['bram']:.1f} RAMB36, "
                  f"{lk['lutram']:,.0f} LUTRAM LUTs")
        if a.json:
            json.dump(gs, open(a.json, 'w'), indent=1)
        print(';'.join(f'{x["others"][0]:.0f},{x["others"][1]:.0f},{x["others"][2]:.0f}' for x in gs))
        return
    others_, lk, total = others(a.path)
    others = others_
    print(f"links: {lk['memories']} memories, {lk['bits']:,} bits: {lk['uram']:.0f} URAM288, "
          f"{lk['bram']:.1f} RAMB36, {lk['lutram']:,.0f} LUTRAM LUTs")
    print(f"total: {total.get('uram', 0):.0f} URAM288, {total.get('bram', 0):.1f} RAMB36, "
          f"{total.get('logic', 0):,.0f} logic + {total.get('lutram', 0):,.0f} LUTRAM LUTs")
    if a.json:
        json.dump({'links': lk, 'total': total, 'others': others}, open(a.json, 'w'), indent=1)
    print(f'{others[0]:.0f},{others[1]:.0f},{others[2]:.0f}')


if __name__ == '__main__':
    main()
