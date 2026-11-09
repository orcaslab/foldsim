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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('path')
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    gen = os.path.dirname(os.path.dirname(a.path)) if a.path.endswith('.mlir') else a.path
    total = foldsim_fitdb.count(gen)['sim']
    lk = links(a.path)
    others = (max(0.0, total.get('uram', 0) - lk['uram']),
              max(0.0, total.get('bram', 0) - lk['bram']),
              max(0.0, total.get('logic', 0) + total.get('lutram', 0) - lk['lutram']))
    print(f"links: {lk['memories']} memories, {lk['bits']:,} bits: {lk['uram']:.0f} URAM288, "
          f"{lk['bram']:.1f} RAMB36, {lk['lutram']:,.0f} LUTRAM LUTs")
    print(f"total: {total.get('uram', 0):.0f} URAM288, {total.get('bram', 0):.1f} RAMB36, "
          f"{total.get('logic', 0):,.0f} logic + {total.get('lutram', 0):,.0f} LUTRAM LUTs")
    if a.json:
        json.dump({'links': lk, 'total': total, 'others': others}, open(a.json, 'w'), indent=1)
    print(f'{others[0]:.0f},{others[1]:.0f},{others[2]:.0f}')


if __name__ == '__main__':
    main()
