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
OPTIMIZE = ['hw.module(hw-expand-array-writes)', 'hw.module(seq-reg-const-prop)',
            'hw.module(comb-share-adders)', 'canonicalize', 'hw.module(comb-low-bits-narrowing)',
            'canonicalize', 'foldsim-sort', 'cse']

# Logic of up to this many ops that drives a model's input from other models
# is computed in the model, so the input comes from them directly rather than
# through the parent's model.
ABSORB_LOGIC = 64


def run(cmd, **kw):
    print('+', ' '.join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, text=True, **kw)


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
                         "that a thread and its parent's are current together (foldsim-compose)")
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
    ap.add_argument('--copies', default='',
                    help='<model>:<count>,...: models simulated by several threaded copies '
                         '(foldsim-compose)')
    ap.add_argument('--relay', default='',
                    help='models, or <model>:<model> pairs, whose channels get a relay stage, '
                         'comma-separated (foldsim-compose)')
    ap.add_argument('--static', action='store_true',
                    help='a statically scheduled simulator: every target cycle a frame of '
                         'host cycles fixed at compile time (foldsim-fold-static); the '
                         'options of the token fabric do not apply')
    ap.add_argument('--region-bits', type=int, default=0,
                    help="with --static: the most state bits a region's copy of the slot "
                         'controls drives (foldsim-fold-static)')
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
    if a.static:
        fold = f'module={a.top} sim=foldsim_sim report={out("schedule.txt")}'
        if a.region_bits:
            fold += f' region-bits={a.region_bits}'
        pipeline = passes + OPTIMIZE + [f'foldsim-fold-static{{{fold}}}',
                                        'canonicalize', 'foldsim-sort', 'cse']
    else:
        pipeline = passes + OPTIMIZE + [f'foldsim-compose{{{compose}}}',
                                        'canonicalize', 'foldsim-sort', 'cse']
    sim = out('foldsim_sim.mlir')
    run([foldsim_opt, flat, f'--pass-pipeline=builtin.module({",".join(pipeline)})', '-o', sim])
    sim_text = open(sim).read()
    # A RAM with a registered read states its read-under-write behavior to
    # Vivado, which otherwise leaves a read colliding with a write undefined.
    run([tool('firtool'), '--format=mlir', sim,
         '--add-vivado-ram-address-conflict-synthesis-bug-workaround', '-o', out('foldsim_sim.sv')])

    threads = re.search(r'foldsim\.threads = \[(.*?)\]', sim_text).group(1)
    info = {
        'top': a.top,
        'threads': dict(t.strip('"').split('=') for t in re.findall(r'"[^"]*"', threads)),
        'ports': [{'dir': d, 'name': n, 'width': w} for d, n, w in signature(sim_text, 'foldsim_sim')],
    }
    for name, n in info['threads'].items():
        info['threads'][name] = int(n)
    with open(out('foldsim_sim.json'), 'w') as f:
        json.dump(info, f, indent=1)
    # The kernel's view of it: how many classes report their finishes, bit c
    # of `profile_finish` for the c-th class of the JSON's `threads`.
    with open(out('foldsim_sim.vh'), 'w') as f:
        f.write('// Generated by foldsim. Do not edit.\n'
                f'`define FOLDSIM_NUM_CLASSES {len(info["threads"])}\n')
    print(f'foldsim: {", ".join(f"{k} x{v}" for k, v in info["threads"].items())}', file=sys.stderr)


if __name__ == '__main__':
    main()
