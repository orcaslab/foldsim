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

"""What foldsim_kernel.vh says of the simulator's virtual memories.

foldsim --virtual-memories gives foldsim_sim.vh FOLDSIM_VMEM_COUNT; each
memory's foldsim_vmem controller gets its own device-memory master, numbered
after the memory ports. This prints FOLDSIM_VMEM_REPEAT(f,s): `f(port, k) for
each memory k joined by `s, spelled out as the Makefile spells out
FOLDSIM_MEM_REPEAT, and FOLDSIM_VMEM_CACHE_ROWS when given. Each memory k's
host cache takes FOLDSIM_VMEM_<k>_CACHE_ROWS where an earlier header (or the
command line) defines it, else FOLDSIM_VMEM_CACHE_ROWS (0: none); the
foldsim_afu.sv of a kernel.vh without FOLDSIM_VMEM_PER_MEMORY_ROWS takes the
global one for every memory. Nothing for a simulator without them.
With --masters, it prints the kernel's device-memory masters instead, and
with --port-masters the memory ports' own.

The memory ports' masters: one per port, or one in all where the simulator's
memory ports are folded (foldsim --fold-ports: FOLDSIM_PORTS in foldsim_sim.vh),
whose endpoint (foldsim_mem_fold) serves every port on a single master. The
controllers' masters are numbered after these.

Grouped memories (foldsim --vmem-group: FOLDSIM_VMEM_GROUPS in
foldsim_sim.vh, each memory's FOLDSIM_VMEM_<k>_GROUP): the controllers are the
groups, so FOLDSIM_VMEM_REPEAT and the masters below count groups, `f(port, g)`;
FOLDSIM_VMEM_MEMBERS(f,s) lists every memory k of group g as member i,
`f(g, i, k)`, and FOLDSIM_VMEM_G<g>_REPEAT(f,s) the members of group g. Group
g's host cache: FOLDSIM_VMEM_G<g>_CACHE_ROWS where the compiler gave it, else
--cache-rows.

Layout v7 (FOLDSIM_VMEM_LAYOUT_V7): each controller's row bytes in device
memory, known-zero bits on or off and their block (FOLDSIM_VMEM_<k>_ROW_BYTES,
_KNOWN_ZERO, _KZ_SHIFT; groups G<g>) are the compiler's where foldsim_sim.vh
gives them, else the v6 layout's (DW/8 rounded up to a power of two of at
least 64 bytes; a bit per row).

Masters for the controllers (--vmem-masters M, Makefile FOLDSIM_VMEM_MASTERS):
by default one per controller, as many as fit beside the memory ports (at
most MAX_MASTERS in all). With fewer masters M than controllers N, controller k
shares master k mod M with the others there (foldsim_vmem_share): the header
then says FOLDSIM_VMEM_MASTERS M and FOLDSIM_VMEM_MASTER_REPEAT(f,s), `f(port,
j) for each master j, and FOLDSIM_VMEM_REPEAT gives each controller its
master's port. With M = N the header is as without the option. Where no
master is left beside the memory ports (or with M = 0), controller k shares
the memory ports' master k mod P with its endpoint: FOLDSIM_VMEM_ON_PORTS.

Static ports (foldsim --vport-engines: FOLDSIM_VPORTS in foldsim_sim.vh) not
bound straight to HBM channels (--direct-vports: the kernel flow's [platform]
hbm_direct) are controllers too, numbered after the virtual memories' (port p
is controller N + p, N the memories' or groups' controllers): they count in
the masters above and always share them (the header then says
FOLDSIM_VMEM_MASTERS, or FOLDSIM_VMEM_ON_PORTS), so that the kernel puts each
port on its master's foldsim_vmem_share by its controller number alone.
FOLDSIM_VMEM_REPEAT and its parts list the virtual memories' controllers only
(nothing of them without any).

Masters by SLR (foldsim --vmem-slr-groups: foldsim_sim.vh gives each
controller's SLR group, FOLDSIM_VMEM_<k>_SLR or FOLDSIM_VMEM_G<g>_SLR, -1 for
none named): with fewer masters than controllers, the M masters are split over
the groups by their controllers' port bits (largest remainder, at least one a
group with controllers: foldsim_sll_check.slr_masters, the split the SLL check
prices), numbered group by group, and the controllers of a group take its
masters in turn. The header then also says FOLDSIM_VMEM_MASTER_OF, controller
k's master for each k (the kernel's sharing follows it instead of k mod M), and
FOLDSIM_VMEM_MASTER_SLR, each master's group. With --floorplan it prints, in
place of the header, the Tcl that assigns each controller (g_vg<g>, or vmem_<k>)
and each master's foldsim_vmem_share (g_vmem_master[j].share) to its group's
SLR (FOLDSIM_VMEM_SLR_NAMES), for the implementation's floorplan
(foldsim_floorplan.tcl, flow floorplan.tcl). A static port on these masters
takes its group from FOLDSIM_VPORT<p>_SLR (its page engine's members' group),
its port bits 2 DW + OW, its cell vport<p>.
"""

import argparse
import re
import sys

# The kernel's device-memory masters, as the emulator binds them.
MAX_MASTERS = 32
# FOLDSIM_VMEM_REPEAT_P<q>: the controllers in parts of PART_SIZE, PARTS of them.
PART_SIZE = 8
PARTS = 16
# FOLDSIM_VPORT_REPEAT<q>: the static ports on these masters, 8 a part, at most VPORT_PARTS parts.
VPORT_PARTS = 32


def vmem_count(sim_vh):
    with open(sim_vh) as f:
        m = re.search(r'^`define FOLDSIM_VMEM_COUNT (\d+)\s*$', f.read(), re.M)
    return int(m.group(1)) if m else 0


def port_masters(sim_vh, mem_ports):
    """The memory ports' device-memory masters: one where the simulator's
    ports are folded (FOLDSIM_PORTS), else one per port."""
    with open(sim_vh) as f:
        folded = re.search(r'^`define FOLDSIM_PORTS \d+\s*$', f.read(), re.M)
    return 1 if folded else mem_ports


def vport_count(sim_vh):
    with open(sim_vh) as f:
        m = re.search(r'^`define FOLDSIM_VPORTS (\d+)\s*$', f.read(), re.M)
    return int(m.group(1)) if m else 0


def vmem_groups(sim_vh):
    """Each group's memories, in member order, from foldsim_sim.vh; None
    without groups."""
    with open(sim_vh) as f:
        text = f.read()
    m = re.search(r'^`define FOLDSIM_VMEM_GROUPS (\d+)\s*$', text, re.M)
    if not m:
        return None
    groups = [[] for _ in range(int(m.group(1)))]
    for k, g in sorted((int(k), int(g)) for k, g in
                       re.findall(r'^`define FOLDSIM_VMEM_(\d+)_GROUP (\d+)\s*$', text, re.M)):
        groups[g].append(k)
    return groups


def vmem_widths(sim_vh):
    """Each memory's and each group's DW from foldsim_sim.vh: {'<k>' or 'G<g>': DW}."""
    with open(sim_vh) as f:
        return {k: int(w) for k, w in re.findall(r'^`define FOLDSIM_VMEM_(G?\d+)_DW (\d+)\s*$', f.read(), re.M)}


def vmem_slr(sim_vh, groups, nm=0, nv=0):
    """Each controller's SLR group from foldsim_sim.vh (FOLDSIM_VMEM_<k>_SLR, or
    FOLDSIM_VMEM_G<g>_SLR with groups; -1: none named) and the groups' SLR
    names (FOLDSIM_VMEM_SLR_NAMES); (None, None) without them."""
    with open(sim_vh) as f:
        text = f.read()
    key = 'G' if groups is not None else ''
    at = {int(k): int(s) for k, s in re.findall(rf'^`define FOLDSIM_VMEM_{key}(\d+)_SLR (-?\d+)\s*$', text, re.M)}
    # (Static ports on the masters: controllers nm + p.)
    at.update({nm + int(p): int(s) for p, s in re.findall(r'^`define FOLDSIM_VPORT(\d+)_SLR (-?\d+)\s*$', text, re.M)
               if int(p) < nv})
    if not at:
        return None, None
    m = re.search(r'^`define FOLDSIM_VMEM_SLR_NAMES "([^"]*)"\s*$', text, re.M)
    return at, (m.group(1).split() if m else None)


def port_bits(sim_vh, groups, n, nm=None):
    """Each controller's port bits (its members' read and write data, row and
    instance: foldsim_sll_check's weight)."""
    with open(sim_vh) as f:
        text = f.read()
    d = dict(re.findall(r'^`define FOLDSIM_VMEM_(\w+) (\d+)\s*$', text, re.M))
    vp = dict(re.findall(r'^`define FOLDSIM_VPORT(\w+) (\d+)\s*$', text, re.M))
    nm = n if nm is None else nm
    bits = lambda k: 2 * int(d.get(f'{k}_DW', 0)) + int(d.get(f'{k}_AW', 0)) + int(d.get(f'{k}_IW', 0))
    pbits = lambda p: 2 * int(vp.get(f'{p}_DW', 0)) + int(vp.get(f'{p}_OW', 0))
    return [(sum(bits(str(k)) for k in groups[c]) if groups is not None else bits(str(c))) if c < nm else pbits(c - nm)
            for c in range(n)]


def split_masters(weights, total):
    """Masters per group: `total` split by `weights` (largest remainder), at
    least one where a weight is positive (foldsim_sll_check.slr_masters)."""
    live = [i for i, w in enumerate(weights) if w > 0]
    if not live:
        return [0] * len(weights)
    total = max(total, len(live))
    tw = sum(weights[i] for i in live)
    out = [0] * len(weights)
    for i in live:
        out[i] = max(1, int(total * weights[i] / tw))
    while sum(out) > total:
        i = max((i for i in live if out[i] > 1), key=lambda i: out[i] - total * weights[i] / tw, default=None)
        if i is None:
            break
        out[i] -= 1
    while sum(out) < total:
        i = max(live, key=lambda i: total * weights[i] / tw - out[i])
        out[i] += 1
    return out


def masters_by_slr(slr, bits, n, m):
    """([master of controller k], [group of master j]): the m masters split over
    the controllers' SLR groups by port bits, numbered group by group (the
    named groups in order, then the unnamed), each group's controllers on its
    masters in turn. A group never takes more masters than controllers."""
    keys = sorted({slr.get(k, -1) for k in range(n)}, key=lambda g: (g < 0, g))
    weights = [sum(max(1, bits[k]) for k in range(n) if slr.get(k, -1) == g) for g in keys]
    count = [sum(1 for k in range(n) if slr.get(k, -1) == g) for g in keys]
    per = split_masters(weights, m)
    # (No group past its controllers: the rest to the next by weight.)
    spare = sum(max(0, p - c) for p, c in zip(per, count))
    per = [min(p, c) for p, c in zip(per, count)]
    while spare:
        i = max((i for i in range(len(keys)) if per[i] < count[i]), key=lambda i: weights[i] / per[i])
        per[i] += 1
        spare -= 1
    first, at = {}, 0
    for g, p in zip(keys, per):
        first[g] = at
        at += p
    seen = {g: 0 for g in keys}
    of = []
    for k in range(n):
        g = slr.get(k, -1)
        of.append(first[g] + seen[g] % per[keys.index(g)])
        seen[g] += 1
    return of, [g for g, p in zip(keys, per) for _ in range(p)]


def floorplan(slr, names, of, mslr, groups, nm=None):
    """The Tcl assigning each controller and each master's share to its
    group's SLR (names: the groups' SLRs, else fs_slr<g>, the placer's
    choice); an unnamed group's are left to the placer."""
    where = lambda g: (names[g] if names and g < len(names) else f'fs_slr{g}')
    # (Group g's controller: g_vg<g>'s foldsim_vmem_group, foldsim_vmem or, with
    # vport controllers, g_vp's foldsim_vport_ctl; memory k's: vmem_<k> or vport_<k>.)
    mem = ((lambda k: f'.*/g_vg{k}[./](g_vp[./])?(vmem|group|vport)$') if groups is not None
           else (lambda k: f'.*/(vmem|vport)_{k}$'))
    # (Static port p on the masters, controller nm + p: its foldsim_vport vport<p>.)
    nm = len(of) if nm is None else nm
    cell = lambda k: mem(k) if k < nm else f'.*/vport{k - nm}$'
    out = ['# Virtual-memory controllers and their masters\' shares in their SLR group (vmem_ports.py --floorplan).']
    for k in range(len(of)):
        if slr.get(k, -1) >= 0:
            out += [f'set c [get_cells -quiet -hierarchical -regexp {{{cell(k)}}}]',
                    f'if {{[llength $c] == 1}} {{ set_property USER_SLR_ASSIGNMENT {where(slr[k])} $c }}']
    for j, g in enumerate(mslr):
        if g >= 0:
            out += [f'set c [get_cells -quiet -hierarchical -regexp {{.*/g_vmem_master\\[{j}\\][./]share$}}]',
                    f'if {{[llength $c] == 1}} {{ set_property USER_SLR_ASSIGNMENT {where(g)} $c }}']
    return '\n'.join(out)


def layout_defaults(name, dw):
    """The v6 layout's row bytes and a known-zero bit per row, where the
    compiler gave none."""
    nb = (dw + 7) // 8
    rb = max(64, 1 << (nb - 1).bit_length())
    return ''.join(f'`ifndef FOLDSIM_VMEM_{name}_{key}\n`define FOLDSIM_VMEM_{name}_{key} {v}\n`endif\n'
                   for key, v in (('ROW_BYTES', rb), ('KNOWN_ZERO', 1), ('KZ_SHIFT', 0))).rstrip('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--sim-vh', required=True, help="the simulator's foldsim_sim.vh")
    ap.add_argument('--mem-ports', type=int, required=True, help='the memory ports (FOLDSIM_MEM_PORTS)')
    ap.add_argument('--cache-rows', type=int, default=0, help='host-cache rows per controller (0: none)')
    ap.add_argument('--masters', action='store_true', help='print the device-memory masters')
    ap.add_argument('--port-masters', action='store_true', help="print the memory ports' masters")
    ap.add_argument('--direct-vports', action='store_true',
                    help='the static ports are bound straight to HBM channels: not on these masters')
    ap.add_argument('--vmem-masters', type=int, default=None,
                    help="the controllers' own masters (default: one each, or as many as fit; "
                         "0: none, on the memory ports' masters)")
    ap.add_argument('--floorplan', action='store_true',
                    help="print the floorplan Tcl of the controllers and their masters' shares (SLR groups)")
    ap.add_argument('--sp', metavar='FOLDSIM_SP',
                    help="print each device-memory master's HBM port (linker tag) from foldsim's foldsim_sp.txt: "
                         "the memory ports' masters the level's ports, each SLR group's controller masters its own "
                         "SLR's ports in turn (an unnamed group's the level's); with --sp-kernel, as `sp=` lines")
    ap.add_argument('--sp-kernel', default='', metavar='NAME',
                    help='with --sp: the kernel instance the `sp=<NAME>.m_axi_mem_<i>:<tag>` lines name')
    a = ap.parse_args()
    groups = vmem_groups(a.sim_vh)
    nm = len(groups) if groups is not None else vmem_count(a.sim_vh)
    nv = 0 if a.direct_vports else vport_count(a.sim_vh)
    n = nm + nv
    p = port_masters(a.sim_vh, a.mem_ports)
    room = MAX_MASTERS - p
    m = min(n, room if a.vmem_masters is None else a.vmem_masters)
    if m < 0:
        sys.exit(f'vmem_ports.py: {p} memory-port masters, beyond the kernel\'s {MAX_MASTERS}')
    if m > room:
        sys.exit(f'vmem_ports.py: {p} memory-port masters and {m} for the virtual memories '
                 f'need {p + m} device-memory masters; the kernel has at most {MAX_MASTERS}')
    if a.cache_rows and (a.cache_rows < 2 or a.cache_rows & (a.cache_rows - 1)):
        sys.exit('vmem_ports.py: --cache-rows is 0 or a power of two of at least 2')
    slr, names = vmem_slr(a.sim_vh, groups, nm, nv) if n else (None, None)
    of = mslr = None
    if slr is not None and 0 < m < n:
        # (A master at least for each group with controllers: m grows to the groups.)
        of, mslr = masters_by_slr(slr, port_bits(a.sim_vh, groups, n, nm), n, m)
        m = len(mslr)
        if m > room:
            sys.exit(f'vmem_ports.py: {m} SLR groups of controllers need a master each beside {p} memory-port '
                     f'masters; the kernel has at most {MAX_MASTERS}')
        if m == n and not nv:
            of = mslr = None
    if a.sp:
        tags, level = {}, []
        for line in open(a.sp):
            f = line.split()
            if len(f) >= 4 and f[0] == 'group':
                tags[int(f[1])] = f[3:]
            elif len(f) >= 3 and f[0] == 'level':
                level = f[2:]
        if not level:
            sys.exit(f'vmem_ports.py: {a.sp} has no level line')
        if of is not None:
            mgroup = mslr
        elif slr is not None and m == n:
            mgroup = [slr.get(k, -1) for k in range(n)]
        else:
            mgroup = [-1] * m
        out, used = [], {}
        for i in range(p):
            out.append(level[i % len(level)])
        for j in range(m):
            g = mgroup[j]
            t = tags.get(g, level) if g >= 0 else level
            out.append(t[used.get(g, 0) % len(t)])
            used[g] = used.get(g, 0) + 1
        for i, t in enumerate(out):
            print(f'sp={a.sp_kernel}.m_axi_mem_{i}:{t}' if a.sp_kernel else f'm_axi_mem_{i} {t}')
        return
    if a.port_masters:
        print(p)
        return
    if a.masters:
        print(p + m)
        return
    if n == 0:
        return
    if a.floorplan:
        if slr is None:
            sys.exit('vmem_ports.py: --floorplan: foldsim_sim.vh gives no SLR group (foldsim --vmem-slr-groups)')
        print(floorplan(slr, names, of or list(range(n)), mslr or [], groups, nm))
        return
    if of is not None:
        port = lambda k: p + of[k]
    else:
        port = (lambda k: k % p) if m == 0 else (lambda k: p + k % m)
    def masters():
        if m == 0:
            print('`define FOLDSIM_VMEM_ON_PORTS')
        # (Static ports on the masters: always shared, even one controller a master.)
        if 0 < m < n or (m and nv):
            print(f'`define FOLDSIM_VMEM_MASTERS {m}')
            print('`define FOLDSIM_VMEM_MASTER_REPEAT(f,s) '
                  + ' `s '.join(f'`f({p + j}, {j})' for j in range(m)))
            if of is not None:
                print('`define FOLDSIM_VMEM_MASTER_OF ' + ', '.join(str(j) for j in of))
                print('`define FOLDSIM_VMEM_MASTER_SLR ' + ', '.join(str(g) for g in mslr))
        # The static ports, 8 a line (as the direct binding's): `f(p); parts 0-7 always, parts 8 and
        # up only for ports past 64 (the kernel instantiates those parts where they are defined).
        for q in range(max(8, -(-nv // 8))) if nv else ():
            xs = [f'`f({v})' for v in range(8 * q, min(8 * q + 8, nv))]
            print(f'`define FOLDSIM_VPORT_REPEAT{q}(f,s) ' + ('`s ' if q and xs else '') + ' `s '.join(xs))
        if nv > 8 * VPORT_PARTS:
            sys.exit(f'vmem_ports.py: {nv} static ports; the kernel lists at most {8 * VPORT_PARTS}')
    if nm == 0:
        masters()
        return
    print('`define FOLDSIM_VMEM_REPEAT(f,s) ' + ' `s '.join(f'`f({port(k)}, {k})' for k in range(nm)))
    # The same in parts of PART_SIZE, each a line of its own where the kernel
    # instantiates the controllers (a part after the first led by `s unless
    # it is empty): Verilator takes at most 40000 tokens on a line.
    print(f'`define FOLDSIM_VMEM_PARTS {PARTS}')
    for q in range(PARTS):
        ks = range(q * PART_SIZE, min(nm, (q + 1) * PART_SIZE))
        print(f'`define FOLDSIM_VMEM_REPEAT_P{q}(f,s) ' + ('`s ' if q and ks else '')
              + ' `s '.join(f'`f({port(k)}, {k})' for k in ks))
    if nm > PARTS * PART_SIZE:
        sys.exit(f'vmem_ports.py: {nm} controllers; the kernel lists at most {PARTS * PART_SIZE}')
    masters()
    if a.cache_rows:
        print(f'`define FOLDSIM_VMEM_CACHE_ROWS {a.cache_rows}')
    print('`define FOLDSIM_VMEM_PER_MEMORY_ROWS')
    print('`define FOLDSIM_VMEM_LAYOUT_V7')
    widths = vmem_widths(a.sim_vh)
    if groups is not None:
        print('`define FOLDSIM_VMEM_MEMBERS(f,s) '
              + ' `s '.join(f'`f({g}, {i}, {k})' for g, ks in enumerate(groups) for i, k in enumerate(ks)))
        for g, ks in enumerate(groups):
            print(f'`define FOLDSIM_VMEM_G{g}_REPEAT(f,s) '
                  + ' `s '.join(f'`f({g}, {i}, {k})' for i, k in enumerate(ks)))
            print(f'`ifndef FOLDSIM_VMEM_G{g}_CACHE_ROWS\n`define FOLDSIM_VMEM_G{g}_CACHE_ROWS {a.cache_rows}\n`endif')
            print(layout_defaults(f'G{g}', widths[f'G{g}']))
        return
    for k in range(nm):
        print(f'`ifndef FOLDSIM_VMEM_{k}_CACHE_ROWS\n`define FOLDSIM_VMEM_{k}_CACHE_ROWS {a.cache_rows}\n`endif')
        print(layout_defaults(str(k), widths[str(k)]))


if __name__ == '__main__':
    main()
