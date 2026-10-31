#!/usr/bin/env python3
"""Compile a target to a foldsim simulator, with CIRCT and foldsim-opt.

usage: foldsim --srcs <srcs.f> --top <top> --out <dir> [--circt-bin <dir>]
               [--foldsim-bin <dir>] <class> [<class> ...]

CIRCT (circt-verilog, circt-opt, firtool) comes from --circt-bin or PATH, and
foldsim-opt from --foldsim-bin, the directory of this driver, or PATH.

The target is imported with circt-verilog and flattened, but for the classes:
the modules simulated as models of their own, each class's instances threaded
through one copy (FAME-5). A class may contain others. Each module with class
instances, innermost first, is cut into a netlist of its own logic and the
models (foldsim-extract-models); the target's netlist is then composed into the
simulator (foldsim-compose): one threaded module per class, the rest of the
target as a model of one thread, and a fabric of token channels between them.
A class `A+B` is two modules of the same parent whose instances are paired and
simulated as one model (foldsim-fuse-models): what they exchange combinationally
both ways then takes a thread one host cycle, not a round trip through channels.

Writes <out>/foldsim_sim.sv, the module `foldsim_sim` with a channel per target
port (`<port>_valid`, `_ready`, `_bits`) and a `profile_finish` bit per class;
<out>/foldsim_sim.json, its port list and the classes' thread counts, in the
order of those bits; and <out>/foldsim_sim.vh, their number for the kernel.
"""
import argparse
import json
import os
import re
import subprocess
import sys

# Passes each model gets before it is threaded: register constants, shared
# adders and narrowed low bits matter for area once every register of a model
# is replicated per thread. A conditional write of one array element becomes a
# select per element, and sorting lets CSE merge duplicates whose users come
# first in a module body.
# Zero-width values first (foldsim-fold-zero-width): an import keeps them where
# a parameter sizes a field to nothing (a one-entry table indexed by a one-bit
# number), and firtool emits neither their compares nor their constants.
OPTIMIZE = ['hw.module(hw-expand-array-writes)', 'hw.module(foldsim-fold-zero-width)',
            'hw.module(seq-reg-const-prop)',
            'hw.module(comb-share-adders)', 'canonicalize', 'hw.module(comb-low-bits-narrowing)',
            'canonicalize', 'foldsim-sort', 'cse']

# The simulator's byte-masked memories as 64-bit columns, one RAM byte lane
# per byte (foldsim-split-masked-memories).
SPLIT = 'hw.module(foldsim-split-masked-memories)'

# Logic of up to this many ops that drives a model's input from other models
# is computed in the model, so the input comes from them directly rather than
# through the parent's model.
ABSORB_LOGIC = 64

# A diagnostic of foldsim-fold-static is on the netlist: without this, MLIR
# prints the whole netlist after it.
NO_OP_DUMP = '--mlir-print-op-on-diagnostic=false'

# The host caches of the virtual memories (--vmem-cache-rows): sized in
# foldsim_storage.py, beside the block-RAM table they are priced by.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from foldsim_storage import VMEM_CACHE_BUDGET, bram36, vmem_cache_rows  # noqa: E402


def run(cmd, check=True, **kw):
    print('+', ' '.join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=check, text=True, **kw)


def publish(src, public):
    """The design with only the `public` modules public."""
    s = re.sub(r'hw\.module @', 'hw.module private @', src)
    for m in public:
        s, n = re.subn(r'hw\.module private @' + re.escape(m) + r'\(', f'hw.module @{m}(', s, count=1)
        if not n:
            sys.exit(f'no module {m} in the design')
    return s


def instantiated(text, module, names):
    """The modules of `names` the module `module` instantiates."""
    m = re.search(r'hw\.module (?:private )?@' + re.escape(module) + r'\(.*?\n  \}\n', text, re.S)
    return {n for n in re.findall(r'hw\.instance "[^"]*" @(\w+)\(', m.group(0)) if n in names}


def signature(text, module):
    """A module's ports: [(direction, name, width)]."""
    m = re.search(r'hw\.module (?:private )?@' + re.escape(module) + r'\((.*?)\) (?:attributes .*?)?\{',
                  text, re.S)
    ports = []
    for d, n, t in re.findall(r'(in|out) %?(\w+) : ([^,)]+)', m.group(1)):
        w = re.fullmatch(r'i(\d+)', t.strip())
        ports.append((d, n, int(w.group(1)) if w else 0))
    return ports


def read_srcs(path, defines, incdirs, files, base=None, seen=()):
    """Adds the `-D`s, `-I`s and source files of the source list `path` to the
    lists. A list holds, separated by whitespace, `+define+<macro>[=<value>]`
    and `+incdir+<dir>`, each repeatable within one argument, source files, and
    other lists: `-f <list>`, whose paths are as written, and `-F <list>`,
    whose paths are relative to its directory. `//` starts a comment. The
    paths of this list are relative to `base`, if any."""
    resolve = lambda p: os.path.join(base, p) if base else p
    if os.path.realpath(path) in seen:
        sys.exit(f'{path}: source list includes itself')
    seen += (os.path.realpath(path),)
    try:
        text = open(path).read()
    except OSError as e:
        sys.exit(f'source list: {e}')
    tokens = []
    for line in text.split('\n'):
        tokens += line.split('//', 1)[0].split()
    tokens = iter(tokens)
    for token in tokens:
        if token.startswith('+define+'):
            defines += ['-D' + d for d in token[len('+define+'):].split('+') if d]
        elif token.startswith('+incdir+'):
            incdirs += ['-I' + resolve(d) for d in token[len('+incdir+'):].split('+') if d]
        elif token in ('-f', '-F'):
            nested = next(tokens, None)
            if nested is None:
                sys.exit(f'{path}: {token} without a source list')
            nested = resolve(nested)
            read_srcs(nested, defines, incdirs, files,
                      (os.path.dirname(nested) or '.') if token == '-F' else None, seen)
        elif token.startswith(('+', '-')):
            sys.exit(f'{path}: unsupported option {token} in a source list')
        else:
            files.append(resolve(token))


def frame_options(a):
    """The fold-static options of the frame: the earlier one restored, or
    consecutive frames overlapped, if asked."""
    fold = ''
    if a.two_cycle_links:
        fold += ' one-cycle-links=false'
    if a.serial_frames:
        fold += ' overlap-frames=false'
    if a.modulo:
        fold += ' modulo'
    if a.interleave_copies:
        fold += ' interleave-copies'
    if getattr(a, 'spec_groups', ''):
        fold += f' spec-groups={a.spec_groups}'
        if a.axi_spec == 2:
            fold += ' spec-depth=2'
    if a.env_sinks:
        fold += f' env-sinks={a.env_sinks}'
    if a.zero_latency_links:
        fold += ' zero-latency-links'
    if a.zero_latency_chains:
        fold += ' zero-latency-chains'
    if a.short_lead:
        fold += ' short-lead'
    if a.bubbles:
        fold += ' bubbles'
    if getattr(a, 'virtual_memories', None):
        if re.search(r'[\s{}]', a.virtual_memories):
            sys.exit('--virtual-memories: a regex without spaces or braces')
        fold += f' virtual-memories={a.virtual_memories}'
        if getattr(a, 'vmem_merge_lanes', False):
            fold += ' vmem-merge-lanes'
    if a.unstaged_env:
        fold += ' staged-env=false'
    if getattr(a, 'register_state_entries', None) is not None:
        fold += f' register-state-entries={a.register_state_entries}'
    if getattr(a, 'ready_levels', 'auto') != 'auto':
        fold += f' ready-levels={a.ready_levels}'
    if getattr(a, 'forward_after_levels', None) is not None:
        fold += f' forward-after-levels={a.forward_after_levels}'
    if getattr(a, 'no_share_link_writes', False):
        fold += ' share-link-writes=false'
    return fold


# An AXI4 port's inputs the endpoint offers for either fire of R in the
# cycle before (the kernel's memory endpoint with FOLDSIM_MEM_SPEC), and at
# depth 2, by the fires of the two cycles before, also AW's and W's ready.
AXI_SPEC_INPUTS = ('rvalid', 'rdata', 'rid', 'rresp', 'rlast', 'arready')
AXI_SPEC2_INPUTS = AXI_SPEC_INPUTS + ('awready', 'wready')


def axi_spec_groups(top, design_text, depth):
    """The fold-static spec groups of every AXI4 port of the top: each port's
    R ready selects its offered values, R valid first."""
    m = re.search(r'hw\.module @' + re.escape(top) + r'\((.*?)\)\s*(attributes|\{)', design_text, re.S)
    if not m:
        sys.exit(f'--axi-spec: no module {top}')
    ports = re.findall(r'\b(in|out)\s+%?"?([A-Za-z0-9_$]+)"?\s*:', m.group(1))
    ins = {n for d, n in ports if d == 'in'}
    outs = {n for d, n in ports if d == 'out'}
    groups = []
    for o in sorted(outs):
        if not o.endswith('rready'):
            continue
        prefix = o[:-len('rready')]
        members = [prefix + n for n in (AXI_SPEC2_INPUTS if depth == 2 else AXI_SPEC_INPUTS)]
        if all(n in ins for n in members):
            groups.append(o + '=' + ':'.join(members))
    if not groups:
        sys.exit('--axi-spec: no AXI4 port on the top')
    return ','.join(groups)


# The V80 builds that calibrate --copies auto by default (foldsim_degree.py
# calibrate): area, placement capacity and the period fit.
DEGREE_CALIBRATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'foldsim_v80.json')


def choose_copies(a, foldsim_opt, flat, passes, out):
    """The fold degree of each class: every choice (or, past
    --degree-max-points, a search) scheduled, priced and the fastest within
    the LUT budget kept (foldsim_degree.py)."""
    import foldsim_degree
    models = out('models.mlir')
    run([foldsim_opt, flat, f'--pass-pipeline=builtin.module({",".join(passes + OPTIMIZE)})',
         '-o', models])
    fold = f'module={a.top} sim=foldsim_sim'
    if a.nest:
        fold += ' nest'
    fold += frame_options(a)
    work = out('degree')
    os.makedirs(work, exist_ok=True)
    base = os.path.join(work, 'schedule_base.txt')
    # The schedules compared go to <out>/degree, not to the log.
    r = subprocess.run([foldsim_opt, models, NO_OP_DUMP, f'--pass-pipeline=builtin.module('
                        f'foldsim-fold-static{{{fold} report={base} schedule-only}})',
                        '-o', os.devnull], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    path = a.degree_calibration or DEGREE_CALIBRATION
    cal = json.load(open(path)) if path != 'none' and os.path.exists(path) else None
    result = foldsim_degree.evaluate(models, open(base).read(), fold, foldsim_opt, work, cal,
                                     a.jobs, a.lut_budget, max_points=a.degree_max_points,
                                     store='auto' if a.storage == 'auto' else 'fixed',
                                     route=a.routability, slr_groups=a.slr_groups,
                                     vmem=(a.vmem_cache_rows, a.vmem_cache_budget)
                                     if a.vmem_cache_rows is not None else None)
    with open(out('degree.txt'), 'w') as f:
        f.write(foldsim_degree.table(result) + '\n')
    with open(out('degree.json'), 'w') as f:
        json.dump(result, f, indent=1, default=str)
    if not result['pick']:
        routes = ' and routes' if a.routability else ''
        sys.exit(f'--copies auto: no choice that places{routes} within {a.lut_budget or "any"} '
                 f'LUTs; see {out("degree.txt")}')
    pick = result['pick']
    timing = (f", {pick['mhz']} MHz, {pick['ns']} ns per cycle" if 'ns' in pick else '')
    rules = pick.get('storage', '')
    print(f"foldsim: copies {pick['copies']}: II {pick['ii']}, about {pick['luts']} LUTs"
          f"{timing} ({os.path.basename(path) if cal else 'uncalibrated'})"
          + (f"; storage: {len(rules.split(','))} rules" if rules else ''), file=sys.stderr)
    return ('' if pick['copies'] == '-' else pick['copies']), rules


def slr_options(a, foldsim_opt, models, fold, out):
    """The fold-static options of --slr-groups: each copy's LUT sites, from a
    schedule without groups (the copies and their instances) and the V80
    calibration (foldsim_degree.copy_sites), and a group's capacity."""
    import foldsim_degree
    base = out('schedule_slr_base.txt')
    plain = fold.replace(f' report={out("schedule.txt")}', '')
    r = subprocess.run([foldsim_opt, models, f'--pass-pipeline=builtin.module('
                        f'foldsim-fold-static{{{plain} report={base} schedule-only}})',
                        '-o', os.devnull], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    _, classes, _ = foldsim_degree.parse_schedule(open(base).read())
    path = a.degree_calibration or DEGREE_CALIBRATION
    cal = json.load(open(path)) if path != 'none' and os.path.exists(path) else None
    entries = a.register_state_entries if a.register_state_entries is not None else 2
    of = {c: re.sub(r'_copy\d+$', '', c) for c in classes}
    est = foldsim_degree.class_estimates(models, sorted(set(of.values())))
    sites = {c: round(foldsim_degree.copy_sites(cal, est[of[c]], of[c], n, entries))
             for c, n in classes.items()}
    capacity = a.slr_capacity or round(foldsim_degree.PLACE_MARGIN *
                                       foldsim_degree.PLACE_SITES / a.slr_count)
    print(f'foldsim: {a.slr_count} SLR groups of at most {capacity} LUT sites; copies: '
          + ', '.join(f'{c} {s}' for c, s in sites.items()), file=sys.stderr)
    opts = (f' slr-groups={a.slr_count} slr-capacity={capacity} slr-sites='
            + ','.join(f'{c}:{s}' for c, s in sites.items()))
    if a.slr_assign:
        opts += f' slr-assign={a.slr_assign}'
    return opts


def write_slr_groups(sv, path):
    # A group split over an SLR boundary pays a crossing on paths wholly inside
    # it, and Vivado otherwise places by area alone. A group name, not an SLR,
    # leaves the choice of SLR to the placer; each group (its copies and the
    # links read into it) is one instance of the simulator.
    body = sv[sv.index('module foldsim_sim('):]
    body = body[:body.index('endmodule')]
    groups = re.findall(r'^\s+\w+ (foldsim_slr\d+) \(', body, re.M)
    with open(path, 'w') as f:
        f.write('# Generated by foldsim. Do not edit.\n')
        for g in groups:
            f.write(f'set c [get_cells -quiet -hierarchical {g}]\n'
                    f'if {{[llength $c] == 1}} {{ set_property USER_SLR_ASSIGNMENT fs_{g[len("foldsim_"):]} $c }}\n')


def vmem_groups(vmems, choice):
    """The kernel's controllers for --vmem-group (VMEM_INTERFACE.md, v6): a
    class copy's virtual memories with the same instances, row size (bytes in
    device memory) and read-during-write order share one, in port order, at
    most N of them (or all, `copy`). Within a controller's image, member m's
    rows follow the larger members': row r of instance i is the controller's
    row GROW_m + r, so each member keeps its own image (the layout of a
    controller of its own) at a byte offset of the controller's region. The
    members are placed largest first, so GROW_m is a multiple of 2^AW_m and
    GROW_m + r is GROW_m | r. Regions 4 KiB aligned; each memory's `base` is
    its image's offset in the vmem space. Updates `vmems` in place."""
    keys = {}
    for k, v in enumerate(vmems):
        keys.setdefault((v['class'], v['instances'], v['inst_width'], v['row_bytes'],
                         v['read_first']), []).append(k)
    cap = None if choice == 'copy' else int(choice)
    groups, offset = [], 0
    for members in keys.values():
        for i in range(0, len(members), cap or len(members)):
            ks = members[i:i + (cap or len(members))]
            first = vmems[ks[0]]
            grow = 0
            for k in sorted(ks, key=lambda k: -vmems[k]['row_width']):
                vmems[k]['group_row'] = grow
                grow += 1 << vmems[k]['row_width']
            # The controller's rows are whole bytes (a member's data is
            # zero-extended to them).
            dw = -(-max(vmems[k]['data_width'] for k in ks) // 8) * 8
            g = {'members': ks, 'size': len(ks), 'class': first['class'],
                 'inst_width': first['inst_width'], 'instances': first['instances'],
                 'rows': grow, 'row_width': max(1, (grow - 1).bit_length()), 'data_width': dw,
                 'mask_width': dw // 8 if any(vmems[k]['mask_width'] > 1 for k in ks) else 1,
                 'read_first': first['read_first'], 'row_bytes': first['row_bytes'],
                 'base': offset, 'bytes': grow * first['instances'] * first['row_bytes']}
            for j, k in enumerate(ks):
                vmems[k].update(group=len(groups), group_member=j,
                                base=offset + vmems[k]['group_row'] * first['instances'] * first['row_bytes'])
            groups.append(g)
            offset += -(-g['bytes'] // 4096) * 4096
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--srcs', required=True,
                    help='source list: +define+, +incdir+, -f/-F lists and files')
    ap.add_argument('--top', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--circt-bin', default='')
    ap.add_argument('--foldsim-bin', default='')
    ap.add_argument('--region-loads', type=int, default=0,
                    help="split each threaded model's state into regions of at most this "
                         'many loads, each with its own thread index and edge (foldsim-fame5)')
    ap.add_argument('--registered-state', action='store_true',
                    help='read the thread state a host cycle ahead into registers and commit '
                         "a single thread's state by select (foldsim-fame5)")
    ap.add_argument('--commit-select', action='store_true',
                    help="commit a single thread's state by select (foldsim-fame5)")
    ap.add_argument('--state-stages', type=int, default=1,
                    help='with --registered-state: host cycles from a thread\'s edge to its '
                         'state\'s write in models of three threads or more (foldsim-fame5)')
    ap.add_argument('--staged-ram-writes', action='store_true',
                    help='with regions: write registered-read memories a host cycle late '
                         'too, forwarding to their reads (foldsim-fame5)')
    ap.add_argument('--pipeline-multipliers', action='store_true',
                    help='with --state-stages 2: state computed by a multiplier takes its '
                         'first stage at its inputs (foldsim-fame5)')
    ap.add_argument('--compact-staging', action='store_true',
                    help='with regions: staging located at its state, staged write '
                         'enables without preset (foldsim-fame5)')
    ap.add_argument('--fold-pipelines', action='store_true',
                    help='simulate fixed-latency pipelines by an unthreaded copy and a '
                         'per-thread store of their results (foldsim-fame5)')
    ap.add_argument('--idle-slots', action='store_true',
                    help='with --registered-state, picked in turn: idle host cycles in the '
                         'turns of models with too few threads, rather than forwarding '
                         'their writes to the read ahead (foldsim-fame5)')
    ap.add_argument('--pipeline-depth', type=int, default=0,
                    help='with --registered-state and --state-stages 2: split state cones '
                         'deeper than this many logic levels over two host cycles '
                         '(foldsim-fame5)')
    ap.add_argument('--memory-loads', type=int, default=0,
                    help='split token memories of more than this many loads by bits, '
                         'each piece with write registers of its own (foldsim-compose)')
    ap.add_argument('--channel-depth', type=int, default=2,
                    help='slots per producer thread of each channel (foldsim-compose)')
    ap.add_argument('--staged-tokens', action='store_true',
                    help='write each token into its slot a host cycle after it is taken, '
                         'forwarded to its readers until then (foldsim-compose)')
    ap.add_argument('--schedule', choices=('ready', 'in-turn'),
                    help='how each threaded model picks its next thread: the next whose '
                         'inputs are all there, or the next in turn (foldsim-compose). '
                         'Default: in turn with --registered-state, whose look-ahead pick '
                         'is a host cycle staler, ready otherwise')
    ap.add_argument('--domains', action='store_true',
                    help='emit the simulator as a tree of domains, one module per class '
                         'nested as the class tree (foldsim-compose)')
    ap.add_argument('--nest', action='store_true',
                    help="fold nested classes recursively: each class's threads numbered so "
                         "that a thread and its parent's are current together (foldsim-compose); with "
                         "--static, instances numbered by their parents' and the slot tables "
                         "driven by a sequencer tree along the class tree (foldsim-fold-static)")
    ap.add_argument('--nest-skew', type=int, default=-1,
                    help='with --nest: host cycles a thread is current after its parent\'s '
                         '(default: a hop of the channels between them)')
    ap.add_argument('--commit-enable', action='store_true',
                    help='with --registered-state and --commit-select: one-thread models take '
                         'every other host cycle and commit by clock enable, without a select '
                         'per bit (foldsim-compose)')
    ap.add_argument('--block-state-depth', type=int, default=0,
                    help='with --registered-state: thread state of models of at least this '
                         'many threads asks for block RAM (foldsim-compose)')
    ap.add_argument('--banked-tokens', action='store_true',
                    help="keep slots read per thread from several producers in memories "
                         "(foldsim-compose)")
    ap.add_argument('--nest-control', choices=('local', 'wired', 'counters'), default='local',
                    help="with --nest: the control tree's turns counted by each class from the "
                         "position the tree plans (local), brought by the parents' controllers "
                         "(wired), or each class's own counter on a rotated numbering "
                         "(counters) (foldsim-compose)")
    ap.add_argument('--nest-offsets', default='',
                    help='with --nest: <model>:<cycles>,...: models current this many host '
                         'cycles after the nested order (foldsim-compose)')
    ap.add_argument('--two-cycle-links', action='store_true',
                    help='with --static: no forwarding into link reads (a reader two host cycles after its producer)')
    ap.add_argument('--serial-frames', action='store_true',
                    help='with --static: decide the next frame only after every output of this one is pushed')
    ap.add_argument('--modulo', action='store_true',
                    help='with --static: overlap consecutive frames, a slot past the frame running in '
                         'the next frame\'s, when that frame is shorter')
    ap.add_argument('--axi-spec', nargs='?', type=int, const=1, default=0, choices=(1, 2),
                    metavar='DEPTH',
                    help="with --static: each AXI4 port's R and AR ready are offered by the "
                         "memory endpoint for either fire of R in the cycle before, and taken "
                         "with the simulator's own R fire (the kernel's FOLDSIM_MEM_SPEC); "
                         "--axi-spec=2: for the fires of the two cycles before, AW's and W's "
                         "ready too")
    ap.add_argument('--env-sinks', default='',
                    help='with --static: outputs of the top no endpoint token depends on, comma '
                         'separated: they close no round trip through the environment')
    ap.add_argument('--interleave-copies', action='store_true',
                    help='with --static: a copy takes every k-th instance, not a contiguous range')
    ap.add_argument('--short-lead', action='store_true',
                    help='with --static: decide a frame three host cycles before its first slot, '
                         'not four (the slot tables read the decided position combinationally)')
    ap.add_argument('--bubbles', action='store_true',
                    help='with --static: an input `advance`; a host cycle it is low puts an idle '
                         'slot in the frame (for slots that wait on data, e.g. memories off-chip)')
    ap.add_argument('--virtual-memories', metavar='REGEX',
                    help='with --static: keep the registered-read memories whose names (or '
                         '<class model>:<name>) match outside the simulator, behind the kernel\'s '
                         'foldsim_vmem controllers (HBM); a slot whose read is not back waits in a bubble')
    ap.add_argument('--vmem-cache-rows', metavar='N|auto',
                    help="with --virtual-memories: each memory's host-cache lines in the kernel "
                         "(foldsim_sim.vh FOLDSIM_VMEM_<k>_CACHE_ROWS): N (0: none, else a power "
                         "of two), or `auto`: a sixteenth of the memory's rows, at least 256, "
                         "within --vmem-cache-budget; never more than the memory's rows. "
                         "Absent: the kernel's default (FOLDSIM_VMEM_CACHE_ROWS)")
    ap.add_argument('--vmem-merge-lanes', action='store_true',
                    help='with --virtual-memories: the arrays a front end split from one target RAM '
                         '(byte lanes, ways: the same read address and enable, the same write '
                         'address) are one virtual memory, each lane its own bytes of the row')
    ap.add_argument('--vmem-group', metavar='copy|N',
                    help="with --virtual-memories: one kernel controller (foldsim_vmem_group) for a "
                         "class copy's virtual memories of the same row size and read-during-write "
                         "order (copy), or for at most N of them; each memory keeps its own region "
                         "within the controller's, and the simulator's ports are unchanged. "
                         "Absent: a controller per memory")
    ap.add_argument('--vmem-cache-budget', type=float, default=VMEM_CACHE_BUDGET, metavar='RAMB36',
                    help='with --vmem-cache-rows auto: the block RAM all the host caches\' lines may '
                         'take, in RAMB36 (default %(default)g, 5%% of the V80 dynamic region\'s)')
    ap.add_argument('--forward-after-levels', type=int, metavar='N',
                    help='with --static: a link read forwarded past the read register where the '
                         "reader's logic behind it stays below N levels (default 8; 0: into the "
                         'read register) (foldsim-fold-static)')
    ap.add_argument('--no-share-link-writes', action='store_true',
                    help="with --static: write each reader's link from the producer's bits in the "
                         'pass, not a host cycle late from one register of them (foldsim-fold-static)')
    ap.add_argument('--register-state-entries', type=int, metavar='N',
                    help='with --static: per-instance state of copies with at most N instances in flops, '
                         'deeper in distributed RAM (default 2; 0: always distributed RAM)')
    ap.add_argument('--ready-levels', choices=('off', 'auto', 'always', 'rank'), default='auto',
                    help='with --static: split a model\'s ready outputs into ready passes by '
                         'level where one ready pass per model leaves the frame with a cycle '
                         '(auto), never (off), for every choice of ready passes (always), or '
                         'for every choice a pass per rank, never merged (rank: a test)')
    ap.add_argument('--unstaged-env', action='store_true',
                    help="with --static: the endpoints take a token in the host cycle it is offered "
                         "(the kernel's memory endpoints unstaged)")
    ap.add_argument('--zero-latency-chains', action='store_true',
                    help='with --zero-latency-links: every late link, chaining across models within '
                         'a host cycle as the native design does within a cycle')
    ap.add_argument('--zero-latency-links', action='store_true',
                    help='with --static: a reader may take a shallow late link in its '
                         'producer\'s own host cycle')
    ap.add_argument('--copies', default='',
                    help='<model>:<count>,...: models simulated by several threaded copies '
                         '(foldsim-compose, foldsim-fold-static); with --static, `auto`: '
                         'the fastest choice within --lut-budget (foldsim_degree.py), '
                         'reported in <out>/degree.txt')
    ap.add_argument('--lut-budget', type=int, default=0,
                    help='with --copies auto: the most LUTs the simulator may take (0: any)')
    ap.add_argument('--degree-calibration', default='',
                    help='with --copies auto: builds that calibrate the estimate '
                         '(foldsim_degree.py calibrate); the V80\'s, foldsim_v80.json, '
                         'by default, `none` for none')
    ap.add_argument('--degree-max-points', type=int, default=256,
                    help='with --copies auto: schedule every choice up to this many, '
                         'else search them (foldsim_degree.py evaluate)')
    ap.add_argument('--routability', action=argparse.BooleanOptionalAction, default=True,
                    help='with --copies auto: the pick must also route (the V80 routability '
                         'term, foldsim_storage.py: load and the SLRs it spreads over, fewer '
                         'with --slr-groups when a choice has fewer copies than SLRs)')
    ap.add_argument('--storage', default='',
                    help='with --static: the storage of folded state stores, as '
                         '<model>:<store>=<kind>,... (foldsim-fold-static storage; kind a Vivado '
                         'ram_style: registers, distributed, block, ultra); with --copies auto, '
                         '`auto`: a kind per store chosen with the fold degree against the V80 '
                         'region (foldsim_storage.py), else the default rules')
    ap.add_argument('--slr-groups', action='store_true',
                    help='with --static: the copies in --slr-count groups, each in an SLR '
                         '(Vivado picks which): within --slr-capacity, the shortest frame, '
                         'then the fewest link bits between groups; a link between groups '
                         'registered on both sides and a host cycle later. '
                         '<out>/foldsim_floorplan.tcl assigns each group, for an '
                         'implementation flow to source after synthesis')
    ap.add_argument('--slr-count', type=int, default=3,
                    help='with --slr-groups: the most SLR groups (the V80: 3 SLRs)')
    ap.add_argument('--slr-capacity', type=int, default=0,
                    help="with --slr-groups: a group's most LUT sites (foldsim_degree.py's "
                         "estimate; default: the V80 dynamic region's, with margin, over "
                         "--slr-count)")
    ap.add_argument('--slr-assign', default='',
                    help='with --slr-groups: <class>:<group>,...: classes pinned to a group '
                         '(copy j > 0 of a class is <class>_copy<j>)')
    ap.add_argument('--jobs', type=int, default=4,
                    help='with --copies auto: schedules compared at once')
    ap.add_argument('--relay', default='',
                    help='models, or <model>:<model> pairs, whose channels get a relay stage, '
                         'comma-separated (foldsim-compose)')
    ap.add_argument('--static', action='store_true',
                    help='a statically scheduled simulator: every target cycle a frame of '
                         'host cycles fixed at compile time (foldsim-fold-static); the '
                         'options of the token fabric do not apply')
    ap.add_argument('--schedule-only', action='store_true',
                    help='with --static: stop once the schedule is chosen, written to '
                         '<out>/schedule.txt (the --copies given, or one copy per class '
                         'for none or auto); no simulator')
    ap.add_argument('--region-bits', type=int, default=0,
                    help="with --static: the most state bits a region's copy of the slot "
                         'controls drives (foldsim-fold-static)')
    ap.add_argument('--control-fanout', type=int, default=96,
                    help='with --static: the most loads (LUT pins per bit) a copy of the slot '
                         'controls drives; one over it is replicated, the replicas loaded in '
                         'lockstep (foldsim-replicate-controls); 0: none replicated')
    ap.add_argument('classes', nargs='*',
                    help='none: the whole target one model of one thread, unthreaded')
    a = ap.parse_args()
    tool = lambda t: os.path.join(a.circt_bin, t) if a.circt_bin else t
    here = os.path.dirname(os.path.realpath(__file__))
    foldsim_bin = a.foldsim_bin or (here if os.path.exists(os.path.join(here, 'foldsim-opt')) else '')
    foldsim_opt = os.path.join(foldsim_bin, 'foldsim-opt') if foldsim_bin else 'foldsim-opt'
    os.makedirs(a.out, exist_ok=True)
    out = lambda f: os.path.join(a.out, f)

    defines, incdirs, files = [], [], []
    read_srcs(a.srcs, defines, incdirs, files)
    design = out('design.mlir')
    run([tool('circt-verilog'), '--ir-hw', f'--top={a.top}', '--allow-use-before-declare',
         '--allow-hierarchical-const', '--ignore-uninstantiated-modules', '--detect-memories',
         '--preserve-unpacked-ports', *defines, *incdirs, *files, '-o', design])

    # A module declared a black box (a core the synthesis flow provides) stays
    # an instance through flattening; foldsim-fame5 folds it by its contract.
    if 'black_box' in open(design).read():
        run([foldsim_opt, design, '--foldsim-black-boxes', '-o', design + '.tmp'])
        os.replace(design + '.tmp', design)

    groups = [c.split('+') for c in a.classes if '+' in c]
    if any(len(g) != 2 for g in groups):
        sys.exit('a fused class is two modules, A+B')
    modules = [m for c in a.classes for m in c.split('+')]

    # Registers reset asynchronously are stepped a target cycle at a time.
    if 'reset async' in open(design).read():
        run([foldsim_opt, design, '--pass-pipeline=builtin.module(hw.module(foldsim-sync-async-resets))',
             '-o', design + '.tmp'])
        os.replace(design + '.tmp', design)

    # Flat but for the classes.
    flat = out('flat.mlir')
    with open(flat + '.in', 'w') as f:
        f.write(publish(open(design).read(), [a.top] + modules))
    # A RAM that asks for block RAM reads synchronously once its wrapper is
    # flattened into the module that registers its address.
    run([tool('circt-opt'), flat + '.in', '--symbol-dce', '--hw-flatten-modules', '--symbol-dce',
         '--canonicalize', '--cse', f'--seq-retime-firmem-reads=report={out("memories.rpt")}',
         '--canonicalize', '--cse', '-o', flat])
    os.remove(flat + '.in')
    text = open(flat).read()

    # Each module with class instances, innermost first, becomes a netlist.
    classes = set(modules)
    children = {m: instantiated(text, m, classes) for m in [a.top] + modules}
    order, seen = [], set()

    def visit(m):
        if m in seen:
            return
        seen.add(m)
        for c in sorted(children[m]):
            visit(c)
        if children[m]:
            order.append(m)
    visit(a.top)
    unused = classes - seen
    if unused:
        sys.exit(f'classes not in {a.top}: {" ".join(sorted(unused))}')
    # Without classes, the target is a netlist of its own logic alone.
    if not order:
        order = [a.top]
    passes = []
    for m in order:
        models = f' models={",".join(sorted(children[m]))}' if children[m] else ''
        passes.append(f'foldsim-extract-models{{module={m}{models} netlist absorb-logic={ABSORB_LOGIC}}}')
        fused = [g for g in groups if set(g) <= children[m]]
        if fused:
            passes.append(f'foldsim-fuse-models{{module={m} groups={",".join("+".join(g) for g in fused)}}}')
    for g in groups:
        if not any(set(g) <= children[m] for m in order):
            sys.exit(f'{"+".join(g)}: not instantiated by the same module')
    compose = f'module={a.top} sim=foldsim_sim'
    # With registered state the next thread is picked a host cycle earlier, from
    # inputs a cycle older: on classes of 16 threads that costs 37.5 % more host
    # cycles than picking in turn, and on s2w4i2 picking in turn costs nothing.
    schedule = a.schedule or ('in-turn' if a.registered_state else 'ready')
    if schedule == 'in-turn':
        compose += ' ready-schedule=false'
    if a.region_loads:
        compose += f' region-loads={a.region_loads}'
    if a.registered_state:
        compose += ' registered-state'
    if a.commit_select:
        compose += ' commit-select'
    if a.state_stages > 1:
        compose += f' state-stages={a.state_stages}'
    if a.staged_ram_writes:
        compose += ' staged-ram-writes'
    if a.pipeline_multipliers:
        compose += ' pipeline-multipliers'
    if a.compact_staging:
        compose += ' compact-staging'
    if a.fold_pipelines:
        compose += ' fold-pipelines'
    if a.idle_slots:
        compose += ' idle-slots'
    if a.pipeline_depth:
        compose += f' pipeline-depth={a.pipeline_depth}'
    if a.memory_loads:
        compose += f' memory-loads={a.memory_loads}'
    if a.channel_depth != 2:
        compose += f' channel-depth={a.channel_depth}'
    if a.domains:
        compose += ' domains'
    if a.copies:
        compose += f' copies={a.copies}'
    if a.commit_enable:
        compose += ' commit-enable'
    if a.block_state_depth:
        compose += f' block-state-depth={a.block_state_depth}'
    if a.banked_tokens:
        compose += ' banked-tokens'
    if a.nest:
        compose += ' nest'
        if a.nest_skew >= 0:
            compose += f' nest-skew={a.nest_skew}'
        if a.nest_control != 'local':
            compose += f' nest-control={a.nest_control}'
        if a.nest_offsets:
            compose += f' nest-offsets={a.nest_offsets}'
    if a.relay:
        compose += f' relay={a.relay}'
    if a.staged_tokens:
        compose += ' staged-tokens'
    sim_input = flat
    if a.axi_spec:
        if not a.static:
            sys.exit('--axi-spec needs --static')
        a.spec_groups = axi_spec_groups(a.top, open(flat).read(), a.axi_spec)
    if a.copies == 'auto' and not a.static:
        sys.exit('--copies auto needs --static')
    if a.vmem_cache_rows is not None:
        if not a.virtual_memories:
            sys.exit('--vmem-cache-rows needs --virtual-memories')
        if a.vmem_cache_rows != 'auto' and not a.vmem_cache_rows.isdigit():
            sys.exit('--vmem-cache-rows: auto, 0 or a power of two of at least 2')
        vmem_cache_rows([], a.vmem_cache_rows, a.vmem_cache_budget)
    if a.vmem_group is not None:
        if not a.virtual_memories:
            sys.exit('--vmem-group needs --virtual-memories')
        if a.vmem_group != 'copy' and not (a.vmem_group.isdigit() and int(a.vmem_group) >= 1):
            sys.exit('--vmem-group: copy, or the memories per controller (at least 1)')
    if a.storage and not a.static:
        sys.exit('--storage needs --static')
    if a.schedule_only and not a.static:
        sys.exit('--schedule-only needs --static')
    if a.static:
        fold = f'module={a.top} sim=foldsim_sim report={out("schedule.txt")}'
        if a.region_bits:
            fold += f' region-bits={a.region_bits}'
        if a.schedule_only:
            models = out('models.mlir')
            run([foldsim_opt, flat, f'--pass-pipeline=builtin.module({",".join(passes + OPTIMIZE)})',
                 '-o', models])
            if a.copies and a.copies != 'auto':
                fold += f' copies={a.copies}'
            if a.nest:
                fold += ' nest'
            fold += frame_options(a) + ' schedule-only'
            r = run([foldsim_opt, models, NO_OP_DUMP,
                     f'--pass-pipeline=builtin.module(foldsim-fold-static{{{fold}}})', '-o', os.devnull],
                    check=False)
            sys.exit(r.returncode and 'foldsim: no schedule')
        copies, rules = a.copies, a.storage
        if a.storage == 'auto' and copies != 'auto':
            sys.exit('--storage auto needs --copies auto')
        if copies == 'auto':
            copies, chosen = choose_copies(a, foldsim_opt, flat, passes, out)
            if a.storage == 'auto':
                rules = chosen
        if copies:
            fold += f' copies={copies}'
        if rules:
            fold += f' storage={rules}'
        if a.nest:
            fold += ' nest'
        fold += frame_options(a)
        if a.slr_groups:
            if a.copies != 'auto':
                run([foldsim_opt, flat, f'--pass-pipeline=builtin.module({",".join(passes + OPTIMIZE)})',
                     '-o', out('models.mlir')])
            fold += slr_options(a, foldsim_opt, out('models.mlir'), fold, out)
        replicate = f'hw.module(foldsim-replicate-controls{{loads={a.control_fanout}}})'
        pipeline = passes + OPTIMIZE + [f'foldsim-fold-static{{{fold}}}', SPLIT, replicate,
                                        'canonicalize', 'foldsim-sort', 'cse']
        if a.copies == 'auto' or a.slr_groups:
            sim_input = out('models.mlir')
            pipeline = pipeline[len(passes) + len(OPTIMIZE):]
    else:
        pipeline = passes + OPTIMIZE + [f'foldsim-compose{{{compose}}}', SPLIT,
                                        'canonicalize', 'foldsim-sort', 'cse']
    sim = out('foldsim_sim.mlir')
    run([foldsim_opt, sim_input, NO_OP_DUMP, f'--pass-pipeline=builtin.module({",".join(pipeline)})',
         '-o', sim])
    sim_text = open(sim).read()
    # A RAM with a registered read states its read-under-write behavior to
    # Vivado, which otherwise leaves a read colliding with a write undefined.
    run([tool('firtool'), '--format=mlir', sim,
         '--add-vivado-ram-address-conflict-synthesis-bug-workaround', '-o', out('foldsim_sim.sv')])

    if a.slr_groups:
        if not a.static:
            sys.exit('--slr-groups needs --static')
        write_slr_groups(open(out('foldsim_sim.sv')).read(), out('foldsim_floorplan.tcl'))

    threads = re.search(r'foldsim\.threads = \[(.*?)\]', sim_text).group(1)
    info = {
        'top': a.top,
        'threads': dict(t.strip('"').split('=') for t in re.findall(r'"[^"]*"', threads)),
        'ports': [{'dir': d, 'name': n, 'width': w} for d, n, w in signature(sim_text, 'foldsim_sim')],
    }
    for name, n in info['threads'].items():
        info['threads'][name] = int(n)
    # Virtual memories, in the simulator's vm<k>_ port order.
    vmems = re.search(r'foldsim\.vmems = \[(.*?)\]', sim_text)
    info['vmem'] = []
    for k, entry in enumerate(re.findall(r'"([^"]*)"', vmems.group(1)) if vmems else []):
        name, cls, iw, aw, dw, mw, rf, insts = entry.split(':')
        info['vmem'].append({'port': f'vm{k}', 'memory': name, 'class': cls, 'inst_width': int(iw),
                             'row_width': int(aw), 'data_width': int(dw), 'mask_width': int(mw),
                             'read_first': rf == '1', 'instances': int(insts)})
    # Each memory's region in the kernel's vmem space (VMEM_INTERFACE.md):
    # row r of instance i at ((r * INSTANCES + i) << log2(ROW_BYTES)), rows a
    # power of two of at least 64 bytes; regions 4 KiB aligned.
    offset = 0
    for v in info['vmem']:
        row_bytes = max(64, 1 << ((v['data_width'] + 7) // 8 - 1).bit_length())
        size = (v['instances'] << v['row_width']) * row_bytes
        v.update(row_bytes=row_bytes, base=offset, bytes=size)
        offset += -(-size // 4096) * 4096
    info['vmem_bytes'] = offset
    if a.vmem_group is not None and info['vmem']:
        info['vmem_groups'] = vmem_groups(info['vmem'], a.vmem_group)
        info['vmem_bytes'] = sum(-(-g['bytes'] // 4096) * 4096 for g in info['vmem_groups'])
        if a.vmem_cache_rows is not None:
            pseudo = [{'instances': g['instances'] * g['rows'], 'row_width': 0,
                       'data_width': g['data_width']} for g in info['vmem_groups']]
            for g, n in zip(info['vmem_groups'], vmem_cache_rows(pseudo, a.vmem_cache_rows,
                                                                 a.vmem_cache_budget)):
                g['cache_rows'] = n
        print('foldsim: vmem controllers: ' + ', '.join(
            f'g{i} {"+".join(info["vmem"][k]["port"] for k in g["members"])}'
            + (f' {g["cache_rows"]} lines' if 'cache_rows' in g else '')
            for i, g in enumerate(info['vmem_groups'])), file=sys.stderr)
    elif a.vmem_cache_rows is not None and info['vmem']:
        for v, n in zip(info['vmem'], vmem_cache_rows(info['vmem'], a.vmem_cache_rows,
                                                      a.vmem_cache_budget)):
            v['cache_rows'] = n
        print('foldsim: vmem host caches: ' + ', '.join(
            f'{v["port"]} {v["cache_rows"]} lines ({bram36(v["cache_rows"], v["data_width"]) if v["cache_rows"] else 0:g} RAMB36)'
            for v in info['vmem']), file=sys.stderr)
    with open(out('foldsim_sim.json'), 'w') as f:
        json.dump(info, f, indent=1)
    # The kernel's view of it: how many classes report their finishes, bit c
    # of `profile_finish` for the c-th class of the JSON's `threads`; and
    # whether it takes idle slots (an `advance` input).
    with open(out('foldsim_sim.vh'), 'w') as f:
        f.write('// Generated by foldsim. Do not edit.\n'
                f'`define FOLDSIM_NUM_CLASSES {len(info["threads"])}\n')
        if any(p['name'] == 'advance' for p in info['ports']):
            f.write('`define FOLDSIM_ADVANCE\n')
        if info['vmem']:
            f.write(f'`define FOLDSIM_VMEM_COUNT {len(info["vmem"])}\n')
            for k, v in enumerate(info['vmem']):
                for key, d in (('IW', 'inst_width'), ('AW', 'row_width'), ('DW', 'data_width'),
                               ('MW', 'mask_width'), ('INSTANCES', 'instances')):
                    f.write(f'`define FOLDSIM_VMEM_{k}_{key} {v[d]}\n')
                f.write(f'`define FOLDSIM_VMEM_{k}_RDWRF {int(v["read_first"])}\n')
                f.write(f'`define FOLDSIM_VMEM_{k}_BASE 64\'h{v["base"]:x}\n')
                if 'cache_rows' in v:
                    f.write(f'`define FOLDSIM_VMEM_{k}_CACHE_ROWS {v["cache_rows"]}\n')
                if 'group' in v:
                    f.write(f'`define FOLDSIM_VMEM_{k}_GROUP {v["group"]}\n')
                    f.write(f'`define FOLDSIM_VMEM_{k}_GROW {v["group_row"]}\n')
            if 'vmem_groups' in info:
                f.write(f'`define FOLDSIM_VMEM_GROUPS {len(info["vmem_groups"])}\n')
                for g, v in enumerate(info['vmem_groups']):
                    for key, d in (('N', 'size'), ('IW', 'inst_width'), ('AW', 'row_width'),
                                   ('DW', 'data_width'), ('MW', 'mask_width'),
                                   ('INSTANCES', 'instances')):
                        f.write(f'`define FOLDSIM_VMEM_G{g}_{key} {v[d]}\n')
                    f.write(f'`define FOLDSIM_VMEM_G{g}_RDWRF {int(v["read_first"])}\n')
                    f.write(f'`define FOLDSIM_VMEM_G{g}_BASE 64\'h{v["base"]:x}\n')
                    if 'cache_rows' in v:
                        f.write(f'`define FOLDSIM_VMEM_G{g}_CACHE_ROWS {v["cache_rows"]}\n')
            f.write(f'`define FOLDSIM_VMEM_BYTES 64\'h{info["vmem_bytes"]:x}\n')
    print(f'foldsim: {", ".join(f"{k} x{v}" for k, v in info["threads"].items())}', file=sys.stderr)


if __name__ == '__main__':
    main()
