#!/usr/bin/env python3
"""Compile a target to a foldsim simulator, with CIRCT and foldsim-opt.

usage: foldsim --srcs <srcs.f> --top <top> --out <dir> [--circt-bin <dir>]
               [--foldsim-bin <dir>] <class> [<class> ...]
       foldsim --manifest <design.toml> --out <dir> [<option> ...]

A target manifest (foldsim_manifest.py) holds a design's whole recipe: its
RTL, classes, environment sinks, memory policy and platform binding; the
command line overrides any of it.

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
import fnmatch
import json
import os
import re
import subprocess
import sys

# Passes each model gets before it is threaded: state no output observes
# (foldsim-remove-dead-state: an unconnected counter and its cone), register constants, shared
# adders and narrowed low bits matter for area once every register of a model
# is replicated per thread. A conditional write of one array element becomes a
# select per element, and sorting lets CSE merge duplicates whose users come
# first in a module body.
# Zero-width values first (foldsim-fold-zero-width): an import keeps them where
# a parameter sizes a field to nothing (a one-entry table indexed by a one-bit
# number), and firtool emits neither their compares nor their constants.
OPTIMIZE = ['hw.module(hw-expand-array-writes)', 'hw.module(foldsim-fold-zero-width)',
            'hw.module(foldsim-remove-dead-state)',
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
import foldsim_storage  # noqa: E402
import foldsim_manifest  # noqa: E402


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


INSTANCE = re.compile(r'hw\.instance "([^"]+)"( sym @\S+)? @([\w$.]+)\(')


# `--class-at TAG@inner:GLOB`: innermost matches only (a word every shell
# passes through, unlike a sign).
INNER = 'inner:'


def class_at(text, top, specs):
    """`--class-at TAG@GLOB`: the instances whose path from `top` (instance
    names joined by '.') matches GLOB become a class of their own. The modules
    along their paths that other instances share are copied, so that the
    matched instances alone change module; each parameter set of them (each
    module they were instances of) becomes a module named TAG, or TAG_<k> for
    several, in the order first met. The first GLOB that matches takes an
    instance; a GLOB may match none. A GLOB written `inner:GLOB` takes only an
    innermost match: an instance with a match of any GLOB below it is not
    taken, the walk goes on below it (a tree of arbiters whose size depends on
    the configuration: its slices one class and its root another, or the
    whole tree where it is one arbiter). Several GLOBs may share a TAG.
    Returns the design and the new modules' names."""
    lines = text.split('\n')
    span, order = {}, []
    for i, line in enumerate(lines):
        m = re.match(r'  hw\.module (?:private )?@([\w$.]+)\(', line)
        if m:
            span[m.group(1)] = [i, None]
            order.append(m.group(1))
        elif line == '  }' and order and span[order[-1]][1] is None:
            span[order[-1]][1] = i
    for tag, _ in specs:
        if tag in span:
            sys.exit(f'--class-at {tag}: a module of the design has that name')
    if top not in span:
        sys.exit(f'--class-at: no module {top}')

    def instances(mod):
        s, e = span[mod]
        return [(i - s, m.group(1), m.group(3)) for i in range(s, e)
                for m in [INSTANCE.search(lines[i])] if m]
    body = {m: instances(m) for m in span}
    # Every instance walked from the top, the search descending only below
    # unmatched instances. What a module's instance needs changed below it is
    # its variant: the (relative line, new callee) pairs of its instance ops,
    # a matched one becoming its tag's class and an unmatched one a variant of
    # its own module. Instances of a module that need the same variant share
    # one module (the instances of a class with several instances, say a
    # crossbar inside each core), so that a module is copied only for another
    # variant or where an instance of it keeps it unchanged.
    variant_id = {}         # (module, changes) -> variant number
    variants = {}           # module -> [changes], in the order first met
    kept = set()            # modules some instance keeps unchanged
    leaves = []             # (tag, leaf module), every matched instance
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    innermost = [g.startswith(INNER) for _, g in specs]
    globs = [(t, g[len(INNER):] if inner else g) for (t, g), inner in zip(specs, innermost)]

    def match(p):
        return next((k for k, (_, g) in enumerate(globs) if fnmatch.fnmatchcase(p, g)), None)

    def below(mod, path):
        """Whether an instance below `mod` (at `path`) matches a GLOB."""
        for _, name, callee in body.get(mod, ()):
            p = f'{path}.{name}'
            if match(p) is not None or below(callee, p):
                return True
        return False

    def walk(mod, path):
        changes = []
        for rel, name, callee in body[mod]:
            p = f'{path}.{name}' if path else name
            k = match(p)
            if k is not None and innermost[k] and below(callee, p):
                k = None
            tag = globs[k][0] if k is not None else None
            if tag:
                leaves.append((tag, callee))
                changes.append((rel, ('class', tag, callee)))
            elif callee in body:
                v = walk(callee, p)
                if v is None:
                    kept.add(callee)
                else:
                    changes.append((rel, ('mod', callee, v)))
        if not changes:
            return None
        key = (mod, tuple(changes))
        if key not in variant_id:
            variant_id[key] = len(variants.setdefault(mod, []))
            variants[mod].append(key[1])
        return variant_id[key]
    walk(top, '')
    # A glob a configuration has no instance for is not an error: one class
    # list serves configurations whose crossbars take different forms.
    for t in dict.fromkeys(t for t, _ in specs):
        if all(tag != t for tag, _ in leaves):
            print(f'class-at {t}: no instance matches {" or ".join(g for u, g in specs if u == t)}',
                  file=sys.stderr)
    # Each parameter set a module of its own, named after the tag.
    groups = {}
    for tag, leaf in leaves:
        groups.setdefault(tag, [])
        if leaf not in groups[tag]:
            groups[tag].append(leaf)
    name = {(t, l): t if len(ls) == 1 else f'{t}_{k}'
            for t, ls in groups.items() for k, l in enumerate(ls)}
    for (tag, leaf), new in name.items():
        if new in span:
            sys.exit(f'--class-at {tag}: a module of the design is named {new}')

    # A variant's module: the module itself when no instance keeps it and it
    # is its first variant, a copy `<module>__at<k>` otherwise, numbered in the
    # order the walk finished them.
    vname, copies = {}, 0
    for (mod, _), v in variant_id.items():
        if v == 0 and mod not in kept:
            vname[(mod, v)] = mod
        else:
            vname[(mod, v)] = f'{mod}__at{copies}'
            copies += 1
    for (mod, _), new in vname.items():
        if new != mod and new in span:
            sys.exit(f'--class-at: a module of the design is named {new}')

    end = max(i for i, line in enumerate(lines) if line == '}')

    def copy(mod, new):
        nonlocal end
        s, e = span[mod]
        text = lines[s:e + 1]
        text[0] = text[0].replace(f'@{mod}(', f'@{new}(', 1)
        span[new] = [end, end + e - s]
        lines[end:end] = text
        end += len(text)

    # Copies first, from the modules as they were; then each variant's
    # instance ops pointed at their new callees.
    for (mod, v), new in vname.items():
        if new != mod:
            copy(mod, new)
    for (tag, leaf), new in name.items():
        copy(leaf, new)

    def retarget(mod, rel, new):
        i = span[mod][0] + rel
        m = INSTANCE.search(lines[i])
        lines[i] = lines[i][:m.start(3)] + new + lines[i][m.end(3):]

    for (mod, v), new in vname.items():
        for rel, (kind, a1, a2) in variants[mod][v]:
            retarget(new, rel, name[(a1, a2)] if kind == 'class' else vname[(a1, a2)])
    for (tag, leaf), new in name.items():
        n = sum(1 for t, l in leaves if (t, l) == (tag, leaf))
        print(f'class-at {tag}: {new} ({leaf}) x{n}', file=sys.stderr)
    return '\n'.join(lines), list(name.values())


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


def rewrite_sources(files, rewrites, outdir):
    """The source list with each file a --rewrite names read from a rewritten
    copy in `outdir`."""
    if not rewrites:
        return files
    by = {}
    for r in rewrites:
        if not isinstance(r, dict) or set(r) != {'file', 'from', 'to'}:
            sys.exit(f'--rewrite {r}: not {{"file": F, "from": A, "to": B}}')
        by.setdefault(os.path.realpath(r['file']), []).append(r)
    known = {os.path.realpath(f) for f in files}
    for f in by:
        if f not in known:
            sys.exit(f'--rewrite: {f} is not a source of the design')
    os.makedirs(outdir, exist_ok=True)
    out = []
    for f in files:
        rs = by.get(os.path.realpath(f))
        if rs:
            text = open(f).read()
            for r in rs:
                if r['from'] not in text:
                    sys.exit(f'--rewrite {f}: no {r["from"]!r} in it')
                text = text.replace(r['from'], r['to'])
            f = os.path.join(outdir, os.path.basename(f))
            with open(f, 'w') as g:
                g.write(text)
        out.append(f)
    return out


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
    if getattr(a, 'min_ii', 0):
        fold += f' min-ii={a.min_ii}'
    if getattr(a, 'modulo_linear', None) is not None:
        fold += f' modulo-linear={a.modulo_linear}'
    if getattr(a, 'ready_choice', ''):
        fold += f' ready-choice={a.ready_choice}'
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
        if getattr(a, 'vmem_skip_held_reads', False):
            fold += ' vmem-skip-held-reads'
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
    if getattr(a, 'zero_chain_hops', None):
        fold += f' zero-chain-hops={a.zero_chain_hops}'
    if getattr(a, 'page', '') and a.page != 'auto':
        fold += f' page={page_option(a.page)}'
    if getattr(a, 'no_page_masks', False):
        fold += ' page-masks=false'
    if getattr(a, 'page_memory_bits', None) is not None:
        fold += f' page-memory-bits={a.page_memory_bits}'
    if getattr(a, 'page_memories', None):
        if re.search(r'[\s{}]', a.page_memories):
            sys.exit('--page-memories: a regex without spaces or braces')
        fold += f' page-memories={a.page_memories}'
    if getattr(a, 'page_rows', None):
        if re.search(r'[\s{}]', a.page_rows):
            sys.exit('--page-rows: a regex without spaces or braces')
        fold += f' page-rows={a.page_rows}'
    if getattr(a, 'page_hbm_window', None):
        fold += f' page-hbm-window={a.page_hbm_window}'
    if getattr(a, 'page_hbm_column_bits', None):
        fold += f' page-hbm-column-bits={a.page_hbm_column_bits}'
    if getattr(a, 'page_engines', None):
        fold += f' page-engines={4096 if a.page_engines == "auto" else int(a.page_engines)}'
    if getattr(a, 'compact_links', False):
        fold += ' compact-links'
    if getattr(a, 'share_links', False):
        fold += ' share-links'
    fold += f" drop-dead-inputs={'true' if getattr(a, 'drop_dead_inputs', True) else 'false'}"
    if getattr(a, 'staged_links', False):
        fold += ' staged-links'
        if getattr(a, 'staged_link_memories', False):
            fold += ' staged-link-memories'
        fold += f" staged-link-direct={'true' if getattr(a, 'staged_link_direct', True) else 'false'}"
    if getattr(a, 'link_block_depth', 0):
        fold += f' link-block-depth={a.link_block_depth}'
    if getattr(a, 'link_ultra_depth', 0):
        fold += f' link-ultra-depth={a.link_ultra_depth}'
    if getattr(a, 'link_hbm_lead', 0):
        fold += f' link-hbm-lead={a.link_hbm_lead}'
    if getattr(a, 'link_hbm_row_bits', 0):
        fold += f' link-hbm-row-bits={a.link_hbm_row_bits}'
    if getattr(a, 'link_hbm_min_lead', 0):
        fold += f' link-hbm-min-lead={a.link_hbm_min_lead}'
    if getattr(a, 'link_hbm_shift', 0):
        fold += f' link-hbm-shift={a.link_hbm_shift}'
    if getattr(a, 'page_write_gate', False):
        fold += ' page-write-gate'
        if getattr(a, 'page_write_gate_pages', ''):
            fold += f' page-write-gate-pages={a.page_write_gate_pages}'
    if getattr(a, 'read_buffer_bits', 0):
        fold += f' read-buffer-bits={a.read_buffer_bits}'
    fold += f' fold-store-ram={getattr(a, "fold_store_ram", 32768)}'
    if getattr(a, 'fold_store_style', 'auto') != 'auto':
        fold += f' fold-store-style={a.fold_store_style}'
    if getattr(a, 'link_hbm_fit_slots', False):
        fold += ' link-hbm-fit-slots'
    if getattr(a, 'fold_ports', ''):
        fold += f' fold-ports={a.fold_ports}'
        if a.fold_ports_reset != 'reset':
            fold += f' fold-ports-reset={a.fold_ports_reset}'
    if getattr(a, 'link_kinds', ''):
        fold += f' link-kinds={a.link_kinds}'
    if getattr(a, 'registered_links', ''):
        fold += f' registered-links={a.registered_links}'
    if getattr(a, 'stage_levels', 0):
        fold += f' stage-levels={a.stage_levels}'
    if getattr(a, 'multicycle', ''):
        fold += f' multicycle={a.multicycle}'
    if getattr(a, 'pass_split', ''):
        fold += f' pass-split={a.pass_split}'
    if getattr(a, 'state_in_ram', ''):
        fold += f' state-in-ram={a.state_in_ram}'
        if getattr(a, 'state_in_ram_read_bits', 0):
            fold += f' state-in-ram-read-bits={a.state_in_ram_read_bits}'
        if getattr(a, 'state_in_ram_h', 0):
            fold += f' state-in-ram-h={a.state_in_ram_h}'
    return fold


def page_option(spec):
    """fold-static's `page` of --page: a count pages every model's copies,
    else <model>:<p>,... as given."""
    if spec == 'auto':
        return spec
    if re.fullmatch(r'\d+', spec):
        return f'*:{spec}'
    if not re.fullmatch(r'[\w$*]+:\d+(,[\w$*]+:\d+)*', spec):
        sys.exit(f"--page: '{spec}' is not <p> or <model>:<p>,...")
    return spec


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
# calibrate): area, placement capacity and the period fit. Area factors fitted
# on another estimator than the one in use (foldsim_degree.ESTIMATOR: the IR cost model by default,
# Beacon's area with --degree-estimator beacon-area) are not used.
DEGREE_CALIBRATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'foldsim_v80.json')


def memory_registers(a, flat, out):
    """The foldsim-memories-to-registers pass of --memory-registers, or None
    when no memory is kept as registers: `auto` (the default with --storage
    auto), the memories cheaper as registers for every copy of their module's
    instances (foldsim_storage.memory_as_registers); `off` (the default
    otherwise), none; else a regex, the memories that can be registers whose
    name or <module>:<name> it finds. The choice goes to
    <out>/memory_registers.txt."""
    mode = a.memory_registers or ('auto' if a.storage == 'auto' else 'off')
    if mode == 'off':
        return None
    rx = None if mode == 'auto' else re.compile(mode)
    mems, instances = foldsim_storage.memory_ports(flat, a.top)
    chosen, lines = [], []
    for m in mems:
        key = f"{m['module']}:{m['name']}"
        shape = (f"{m['depth']}x{m['width']} reads {m['reads']} ({m['const_reads']} constant) "
                 f"writes {m['wports']}, {instances[m['module']]} instances")
        why = foldsim_storage.register_ineligible(m)
        if why:
            if rx is not None and (rx.search(m['name']) or rx.search(key)):
                lines.append(f'  memory {key}: {shape}; cannot be registers: {why}')
                print(f'foldsim: --memory-registers {mode}: {key} cannot be registers: {why}',
                      file=sys.stderr)
            continue
        if rx is None:
            take, rows = foldsim_storage.memory_as_registers(m, instances[m['module']])
            price = '; LUT sites as memory / registers: ' + ', '.join(
                f'{t} instances {mem:.0f} / {reg:.0f}' for t, mem, reg in rows)
        else:
            take, price = bool(rx.search(m['name']) or rx.search(key)), ''
        if take:
            chosen.append(key)
            lines.append(f'  registers {key}: {shape}{price}')
    if not chosen:
        print(f'foldsim: --memory-registers {mode}: no memory kept as registers', file=sys.stderr)
        return None
    with open(out('memory_registers.txt'), 'w') as f:
        f.write(f'--memory-registers {mode}: {len(chosen)} of {len(mems)} memories kept as '
                'registers (foldsim-memories-to-registers)\n' + '\n'.join(lines) + '\n')
    print(f'foldsim: --memory-registers {mode}: {len(chosen)} of {len(mems)} memories kept as '
          f'registers; see {out("memory_registers.txt")}', file=sys.stderr)
    return f'foldsim-memories-to-registers{{memories={",".join(chosen)}}}'


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
    calibration = os.path.basename(path) if cal else 'uncalibrated'
    if cal and not result.get('area_calibrated', True):
        calibration += ': LUTs at the estimator scale'
    timing = (f", {pick['mhz']} MHz, {pick['ns']} ns per cycle" if 'ns' in pick else '')
    rules = pick.get('storage', '')
    print(f"foldsim: copies {pick['copies']}: II {pick['ii']}, about {pick['luts']} LUTs"
          f"{timing} ({calibration})"
          + (f"; storage: {len(rules.split(','))} rules" if rules else ''), file=sys.stderr)
    return ('' if pick['copies'] == '-' else pick['copies']), rules


def choose_pages(a, foldsim_opt, flat, passes, out, fold):
    """--page auto: the paging that fits the V80 region at the shortest
    frame (foldsim_degree.choose_pages), priced on a schedule of the fold
    without paging; the `page` option, '' for none."""
    import foldsim_degree
    models = out('models.mlir')
    if not os.path.exists(models):
        run([foldsim_opt, flat, f'--pass-pipeline=builtin.module({",".join(passes + OPTIMIZE)})',
             '-o', models])
    base = out('schedule_page_base.txt')
    plain = fold.replace(f' report={out("schedule.txt")}', '')
    r = subprocess.run([foldsim_opt, models, NO_OP_DUMP, f'--pass-pipeline=builtin.module('
                        f'foldsim-fold-static{{{plain} report={base} schedule-only}})',
                        '-o', os.devnull], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    text = open(base).read()
    ii, classes, _ = foldsim_degree.parse_schedule(text)
    path = a.degree_calibration or DEGREE_CALIBRATION
    cal = json.load(open(path)) if path != 'none' and os.path.exists(path) else None
    names = sorted({re.sub(r'_copy\d+$', '', c) for c in classes})
    est = foldsim_degree.class_estimates(models, names)
    m = re.search(r'virtual-memories=(\S+)', fold)
    est = foldsim_degree.split_virtual(est, m.group(1) if m else '')
    entries = a.register_state_entries if a.register_state_entries is not None else 2
    spec, frame, res = foldsim_degree.choose_pages(
        cal, est, classes, foldsim_degree.parse_classes(text), ii,
        store='auto' if a.storage == 'auto' else 'fixed', route=a.routability, entries=entries)
    if spec is None:
        sys.exit(f'--page auto: no paging fits the V80 region (unpaged II {ii}); '
                 f'see {base}')
    off = res.get('offchip', [])
    print(f"foldsim: page {spec or 'none'}: II at least {frame} (unpaged {ii}), about "
          f"{res['luts']} LUTs, {res['bram']} RAMB36"
          + (f", {res['uram']} URAM" if 'uram' in res else '')
          + (f", {res['page_ports']} page ports" if 'page_ports' in res else '')
          + (f"; off chip: {', '.join(off)}, {res['hbm_bytes']} bytes a frame" if off else ''),
          file=sys.stderr)
    return spec, [f'{m}:pages=hbm' for m in off]


def choose_state(a, foldsim_opt, flat, passes, out, fold):
    """--state auto: where each class's state lives (foldsim_degree.choose_state:
    on chip, paged on chip, or in HBM, priced against the V80 region, its
    routing and H), on a schedule of the fold as given. Returns the fold
    with the choice: its `page`, `page-rows`, the virtual memories it moved
    and the storage rules (pages=hbm among them); the choice is written to
    <out>/state.txt."""
    import foldsim_degree
    models = out('models.mlir')
    if not os.path.exists(models):
        run([foldsim_opt, flat, f'--pass-pipeline=builtin.module({",".join(passes + OPTIMIZE)})',
             '-o', models])
    base = out('schedule_state_base.txt')
    plain = re.sub(r' report=\S+', '', fold).replace(' schedule-only', '')
    r = subprocess.run([foldsim_opt, models, NO_OP_DUMP, f'--pass-pipeline=builtin.module('
                        f'foldsim-fold-static{{{plain} report={base} schedule-only}})',
                        '-o', os.devnull], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    text = open(base).read()
    ii, classes, _ = foldsim_degree.parse_schedule(text)
    path = a.degree_calibration or DEGREE_CALIBRATION
    cal = json.load(open(path)) if path != 'none' and os.path.exists(path) else None
    names = sorted({re.sub(r'_copy\d+$', '', c) for c in classes})
    m = re.search(r' virtual-memories=(\S+)', fold)
    pattern = m.group(1) if m else ''
    est = foldsim_degree.split_virtual(foldsim_degree.class_estimates(models, names), pattern)
    entries = a.register_state_entries if a.register_state_entries is not None else 2
    weights = json.load(open(a.vmem_weights)) if a.vmem_weights else None
    vmem = (a.vmem_cache_rows, a.vmem_cache_budget) if a.vmem_cache_rows is not None else None
    links = foldsim_degree.parse_links(text)
    res = foldsim_degree.choose_state(cal, est, classes, foldsim_degree.parse_classes(text), ii,
                                      route=a.routability, entries=entries, vmem=vmem,
                                      pattern=pattern, weights=weights,
                                      row_bytes_min=a.vmem_row_bytes_min)
    if res is None:
        sys.exit(f'--state auto: nothing fits the V80 region (II {ii})'
                 + (f" with the links' {links['lutram']:.0f} LUTRAM LUTs, {links['bram']} RAMB36, "
                    f"{links['uram']} URAM288" if links else '') + f'; see {base}')
    with open(out('state.txt'), 'w') as f:
        f.write(f'II {ii} (schedule)\n' + foldsim_degree.state_table(res, classes, est) + '\n')
    h = res['h']
    print(f"foldsim: state auto: frame {res['frame']} (II {ii}), H at most {h['h']}; page "
          f"{len(res['page'].split(',')) if res['page'] else 0} models, {len(res['rows_new'])} "
          f"larger memories as rows, {len(res['moved'])} memories and "
          f"{len(res['pages_offchip'])} models' pages to HBM ({h['hbm_bytes']} bytes a frame); "
          f"{res['res']['luts']} LUTs, {res['res']['bram']} RAMB36, "
          f"{round(res['res'].get('uram', 0))} URAM288; see {out('state.txt')}", file=sys.stderr)
    if res['page']:
        fold = re.sub(r' page=\S+', '', fold) + f" page={res['page']}"
    if res['rows']:
        fold += f" page-rows={res['rows']}"
    if res.get('link_kinds'):
        fold = re.sub(r' link-kinds=\S+', '', fold) + f" link-kinds={res['link_kinds']}"
    if res['pattern'] and res['pattern'] != pattern:
        if m:
            fold = fold.replace(m.group(0), f" virtual-memories={res['pattern']}")
        else:
            fold += f" virtual-memories={res['pattern']}"
            if a.vmem_merge_lanes:
                fold += ' vmem-merge-lanes'
            if a.vmem_skip_held_reads:
                fold += ' vmem-skip-held-reads'
    if res['storage']:
        m = re.search(r' storage=(\S+)', fold)
        fold = (fold.replace(m.group(0), f" storage={m.group(1)},{res['storage']}") if m
                else fold + f" storage={res['storage']}")
    return fold


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
    text = open(base).read()
    ii, classes, _ = foldsim_degree.parse_schedule(text)
    info = foldsim_degree.parse_classes(text)
    path = a.degree_calibration or DEGREE_CALIBRATION
    cal = json.load(open(path)) if path != 'none' and os.path.exists(path) else None
    entries = a.register_state_entries if a.register_state_entries is not None else 2
    of = {c: re.sub(r'_copy\d+$', '', c) for c in classes}
    est = foldsim_degree.class_estimates(models, sorted(set(of.values())))
    # Each copy's state as that schedule keeps it (paged copies in page RAMs).
    sites = {c: round(foldsim_degree.copy_sites(cal, est[of[c]], of[c], n, entries, info.get(c), ii))
             for c, n in classes.items()}
    if a.slr_sites:
        # Measured: each copy's LUTs in a generated netlist of the same recipe
        # (foldsim_sll_check.py's netlist resources, JSON {copy: {lut: ..}}),
        # in LUT sites (the region's sites_per_lut).
        rj = a.slr_region if a.slr_region not in ('', 'v80') else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')
        spl = json.load(open(rj))['sites_per_lut']
        measured = json.load(open(a.slr_sites))
        for c in sites:
            if c in measured:
                sites[c] = round(measured[c]['lut'] * spl)
        print(f'foldsim: copy sites measured from {a.slr_sites} ({sum(c in measured for c in sites)} of {len(sites)})',
              file=sys.stderr)
    if a.slr_region:
        # Unequal SLRs: group g is the region's g-th SLR, each with its own
        # capacity (its LUTs, with margin, in LUT sites); the floorplan names it.
        # No boundary may carry more link bits than its SLLs, with margin.
        caps = slr_region_capacities(a.slr_region)
        print(f'foldsim: {len(caps)} SLR groups of at most '
              + ', '.join(f'{n} {c}' for n, c in caps) + ' LUT sites; copies: '
              + ', '.join(f'{c} {s}' for c, s in sites.items()), file=sys.stderr)
        opts = (f' slr-groups={len(caps)} slr-capacities={",".join(str(c) for _, c in caps)} slr-sites='
                + ','.join(f'{c}:{s}' for c, s in sites.items())
                + f' slr-sll-budget={slr_region_sll_budget(a.slr_region)}')
    else:
        capacity = a.slr_capacity or round(foldsim_degree.PLACE_MARGIN *
                                           foldsim_degree.PLACE_SITES / a.slr_count)
        print(f'foldsim: {a.slr_count} SLR groups of at most {capacity} LUT sites; copies: '
              + ', '.join(f'{c} {s}' for c, s in sites.items()), file=sys.stderr)
        opts = (f' slr-groups={a.slr_count} slr-capacity={capacity} slr-sites='
                + ','.join(f'{c}:{s}' for c, s in sites.items()))
    if a.slr_assign:
        opts += f' slr-assign={a.slr_assign}'
    if a.slr_serialize:
        opts += f' slr-serialize={a.slr_serialize}'
    return opts


def slr_region_capacities(path):
    """[(SLR name, LUT sites)] of a dynamic region's SLRs (--slr-region): each
    SLR's LUTs times the margin a group may fill, in foldsim_degree's LUT
    sites (sites_per_lut, calibrated on a placed build)."""
    if path == 'v80':
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')
    region = json.load(open(path))
    return [(slr['name'], round(region['margin'] * slr['luts'] * region['sites_per_lut']))
            for slr in region['slrs']]


def slr_region_sll_budget(path):
    """The most link bits an SLR boundary of the region may carry: its SLLs
    (super long lines; the V80: 18,870 a boundary) times the region's
    sll_margin (Vivado refuses a partition past the SLLs; leave room for the
    shell's own crossings and the placer)."""
    if path == 'v80':
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')
    region = json.load(open(path))
    return int(region.get('sll_margin', 1.0) * region.get('sll_per_boundary', 0))


def write_slr_groups(sv, path, names=None):
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
            # With --slr-region, group g was sized for the region's g-th SLR:
            # assigned to it by name, not left to the placer.
            where = names[int(g[len('foldsim_slr'):])] if names else f'fs_{g[len("foldsim_"):]}'
            f.write(f'set c [get_cells -quiet -hierarchical {g}]\n'
                    f'if {{[llength $c] == 1}} {{ set_property USER_SLR_ASSIGNMENT {where} $c }}\n')


def write_multicycle_xdc(entries, path, splits=()):
    """Each multicycle copy's paths from its capture registers to its end
    registers, M host cycles (setup M, hold M - 1). The compiler checks that
    these are all the paths out of the one and into the other; the flow
    checks it again on the implemented netlist (multicycle_check.tcl)."""
    with open(path, 'w') as f:
        f.write('# Generated by foldsim. Do not edit.\n')
        for e in entries:
            m, p = e['cycles'], e['prefix']
            if m < 2:
                continue
            f.write(f'# {p}: {e["capture_bits"]} capture bits, {e["end_bits"]} end bits, {m} host cycles\n'
                    f'set_multicycle_path {m} -setup '
                    f'-from [get_cells -hierarchical -filter {{IS_SEQUENTIAL && NAME =~ "*{p}cap*"}}] '
                    f'-to [get_cells -hierarchical -filter {{IS_SEQUENTIAL && NAME =~ "*{p}end*"}}]\n'
                    f'set_multicycle_path {m - 1} -hold '
                    f'-from [get_cells -hierarchical -filter {{IS_SEQUENTIAL && NAME =~ "*{p}cap*"}}] '
                    f'-to [get_cells -hierarchical -filter {{IS_SEQUENTIAL && NAME =~ "*{p}end*"}}]\n')
        # Pass splitting: a pass of `span` host cycles from the captures and the pass counters to
        # the holds (the holds' paths to the deferred writes stay single-cycle).
        for e in splits:
            m, p = e['span'], e['prefix']
            if m < 2:
                continue
            pairs = [(f'*{p}cap*', f'*{p}hold*'), (f'*{p}pass*', f'*{p}hold*'), (f'*{p}sub*', f'*{p}hold*')]
            f.write(f'# {p}: {e["passes"]} passes of {m} host cycles, {e["hold_bits"]} hold bits\n')
            for a_, b_ in pairs:
                for kind, n in (('setup', m), ('hold', m - 1)):
                    f.write(f'set_multicycle_path {n} -{kind} '
                            f'-from [get_cells -hierarchical -filter {{IS_SEQUENTIAL && NAME =~ "{a_}"}}] '
                            f'-to [get_cells -hierarchical -filter {{IS_SEQUENTIAL && NAME =~ "{b_}"}}]\n')


def vmem_partition(vmems, sets, choice, weights):
    """The memories of each set (a class copy's memories that may share a
    controller) split into controllers. `copy`: a set each; N: at most N
    memories each, in port order; `auto[:M]` (v7): M controllers in all
    (default 32, at least one a set) given out one at a time to the set whose
    controllers carry the most expected accesses each, and each set's
    memories spread over its controllers largest load first, each to the
    least loaded (LPT). A memory's load: its instances (a state pass issues at
    most one entry a frame each) times its weight: its accesses per instance
    as measured (`weights`, by port or memory name, so a profile of one fold
    carries to another of the same classes), else 1."""
    if not choice.startswith('auto'):
        cap = None if choice == 'copy' else int(choice)
        return [ms[i:i + (cap or len(ms))] for ms in sets for i in range(0, len(ms), cap or len(ms))]
    budget = int(choice.split(':')[1]) if ':' in choice else 32
    weights = weights or {}
    load = lambda k: vmems[k]['instances'] * float(weights.get(vmems[k]['port'], weights.get(vmems[k]['memory'], 1.0)))
    count = [1] * len(sets)
    total = [sum(load(k) for k in ms) for ms in sets]
    for _ in range(budget - len(sets)):
        open_ = [i for i, ms in enumerate(sets) if count[i] < len(ms) and total[i] > 0]
        if not open_:
            break
        i = max(open_, key=lambda i: total[i] / count[i])
        count[i] += 1
    out = []
    for ms, n in zip(sets, count):
        bins = [[0.0, []] for _ in range(n)]
        for k in sorted(ms, key=lambda k: (-load(k), k)):
            b = min(bins, key=lambda b: b[0])
            b[0] += load(k)
            b[1].append(k)
        out += [sorted(b[1]) for b in bins if b[1]]
    return out


def vmem_known_zero(rows, budget):
    """A controller's known-zero bits (v7): on, with a bit per block of
    2^shift image rows, the smallest block that keeps the bits within
    `budget`; off with a budget of 0. `rows`: the image's rows (2^AW x
    instances)."""
    if budget <= 0:
        return 0, 0
    shift = 0
    while -(-rows // (1 << shift)) > budget:
        shift += 1
    return 1, shift


def vmem_groups(vmems, choice, weights=None, kz_bits=0):
    """The kernel's controllers for --vmem-group (VMEM_INTERFACE.md, v6, v7): a
    class copy's virtual memories with the same instances, row size (bytes in
    device memory) and read-during-write order may share one: all of them
    (`copy`), at most N in port order, or as `auto[:M]` spreads them
    (vmem_partition). Within a controller's image, member m's
    rows follow the larger members': row r of instance i is the controller's
    row GROW_m + r, so each member keeps its own image (the layout of a
    controller of its own) at a byte offset of the controller's region. The
    members are placed largest first, so GROW_m is a multiple of 2^AW_m and
    GROW_m + r is GROW_m | r. Regions 4 KiB aligned; each memory's `base` is
    its image's offset in the vmem space. Updates `vmems` in place."""
    keys = {}
    for k, v in enumerate(vmems):
        # foldsim_vmem_group keeps one instance for a member's read and write
        # of a host cycle: an off-chip page column (a read of one instance's
        # page and a write of another's in one host cycle) has a controller
        # of its own.
        keys.setdefault((v['class'], v['instances'], v['inst_width'], v['row_bytes'],
                         v['read_first'], k if v.get('pages') else -1), []).append(k)
    groups, offset = [], 0
    for ks in vmem_partition(vmems, list(keys.values()), choice, weights):
        first = vmems[ks[0]]
        grow = 0
        for k in sorted(ks, key=lambda k: -vmems[k]['row_width']):
            vmems[k]['group_row'] = grow
            grow += 1 << vmems[k]['row_width']
        # The controller's rows are whole bytes (a member's data is
        # zero-extended to them).
        dw = -(-max(vmems[k]['data_width'] for k in ks) // 8) * 8
        if len(ks) > 1 and any(vmems[k].get('tag_width') for k in ks):
            raise SystemExit('foldsim: a virtual memory with tagged requests shares a controller')
        g = {'members': ks, 'size': len(ks), 'class': first['class'],
             'inst_width': first['inst_width'], 'instances': first['instances'],
             'rows': grow, 'row_width': max(1, (grow - 1).bit_length()), 'data_width': dw,
             'mask_width': dw // 8 if any(vmems[k]['mask_width'] > 1 for k in ks) else 1,
             'read_first': first['read_first'], 'row_bytes': first['row_bytes'],
             'base': offset, 'bytes': grow * first['instances'] * first['row_bytes']}
        g['known_zero'], g['kz_shift'] = vmem_known_zero((1 << g['row_width']) * g['instances'], kz_bits)
        if first.get('tag_width'):
            g['tag_width'] = first['tag_width']
        # An HBM link stream's controller: no known-zero bits (as ungrouped).
        if first['class'].startswith(('__link', '__pages')):
            g['known_zero'], g['kz_shift'] = 0, 0
        for j, k in enumerate(ks):
            vmems[k].update(group=len(groups), group_member=j,
                            base=offset + vmems[k]['group_row'] * first['instances'] * first['row_bytes'])
        groups.append(g)
        offset += -(-g['bytes'] // 4096) * 4096
    return groups


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--manifest', default='', metavar='FILE',
                    help="the design's target manifest (foldsim_manifest.py): its options are "
                         "the defaults, each overridden by the command line's")
    ap.add_argument('--manifest-var', action='append', default=[], metavar='VAR=VALUE',
                    help="a ${VAR} of the manifest's strings (else the environment's)")
    ap.add_argument('--manifest-dump', action='store_true',
                    help='print the options, the manifest applied, as JSON and exit')
    ap.add_argument('--srcs', default='',
                    help='source list: +define+, +incdir+, -f/-F lists and files')
    ap.add_argument('--src', action='append', default=[], metavar='FILE',
                    help='a source file after the list\'s')
    ap.add_argument('--incdir', action='append', default=[], metavar='DIR',
                    help='an include directory after the list\'s')
    ap.add_argument('--define', action='append', default=[], metavar='MACRO[=VALUE]',
                    help='a define after the list\'s')
    ap.add_argument('--rewrite', action='append', default=[], type=json.loads,
                    metavar='{"file": F, "from": A, "to": B}',
                    help='before the import, the source F read from a copy (<out>/rewritten) with '
                         'every A replaced by B (A must occur): a construct the importer does not '
                         'lower, written as one it does, the design itself unchanged')
    ap.add_argument('--top', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--circt-bin', default='')
    ap.add_argument('--foldsim-bin', default='')
    ap.add_argument('--max-unroll-trip-count', type=int, default=0, metavar='N',
                    help="circt-verilog's limit on the iterations of a process's loop it unrolls "
                         '(--max-unroll-trip-count); 0: its default (1024)')
    ap.add_argument('--max-generate-steps', type=int, default=0, metavar='N',
                    help="circt-verilog's (slang's) generate-loop step limit, which counts the "
                         'blocks of a loop times those of the loops around it; 0: its default')
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
    ap.add_argument('--class-at', action='append', default=[], metavar='TAG@GLOB',
                    help='the instances whose path (instance names joined by ".") matches GLOB '
                         'are a class: one module per parameter set, TAG or TAG_<k>; repeatable, '
                         'the first match takes an instance; `inner:GLOB` only an instance with no '
                         'match below it. A class `A+B` may name them')
    ap.add_argument('--absorb-exclusive', default='', metavar='CLASS,...',
                    help="for these classes (globs), a model input whose logic in the parent is "
                         "over the absorb limit takes the part of it that reaches that input "
                         "alone, the rest as inputs: a per-port select in front of a per-port "
                         "class is then one copy in the class (foldsim-extract-models "
                         "absorb-exclusive)")
    ap.add_argument('--absorb-registers', action='store_true',
                    help="with --absorb-exclusive: the logic a model input alone reads may pass "
                         "through registers (a per-port pipeline register and the select in front "
                         "of it), which move into the model as its state (foldsim-extract-models "
                         "absorb-registers)")
    ap.add_argument('--absorb-register-logic', type=int, default=0, metavar='N',
                    help='with --absorb-registers: the most ops a model input\'s logic through '
                         'registers may take (default: the absorb limit, %d)' % ABSORB_LOGIC)
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
    ap.add_argument('--min-ii', type=int, default=0, metavar='N',
                    help='with --static --modulo: the shortest frame in host cycles (foldsim-fold-static '
                         'min-ii); positions past the schedule\'s own stay idle, a longer prefetch distance')
    ap.add_argument('--ready-choice', default='', metavar='[split:]CLASS,...',
                    help='with --static: the classes that get a ready pass, given instead of searched '
                         '(foldsim-fold-static ready-choice; "split:" first when ready outputs are split by '
                         'level): a schedule found before, rebuilt without the search. '
                         'ready_choice.py <schedule.txt> prints it from a report')
    ap.add_argument('--modulo', action='store_true',
                    help='with --static: overlap consecutive frames, a slot past the frame running in '
                         'the next frame\'s, when that frame is shorter')
    ap.add_argument('--modulo-linear', type=int, metavar='N',
                    help='with --modulo: the periods tried one at a time from the lower bound before '
                         'the search skips ahead (foldsim-fold-static modulo-linear; default 16)')
    ap.add_argument('--spec-groups', default='', metavar='OUT=VALID:IN:...,...',
                    help='with --static: inputs the environment offers for either value of a one-bit select '
                         'output (foldsim-fold-static spec-groups; --axi-spec derives them for AXI4 ports)')
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
    ap.add_argument('--vmem-skip-held-reads', action='store_true',
                    help='with --virtual-memories: a read whose address register holds (a registered '
                         'read address with its enable low) is not issued unless its row was written '
                         'since the instance last read it; the instance keeps its read data (exact). '
                         "Its requests leave a host cycle after the pass, from registers (room 12)")
    ap.add_argument('--balance-mux', metavar='MIN_CHAIN', type=int, default=0,
                    help='every model\'s linear priority mux chains of at least MIN_CHAIN conditions '
                         '(an if/else-if ladder, a loop that keeps the last match; 16 is comb-balance-mux\'s '
                         'own default) rebalanced into trees of logarithmic depth before folding (circt '
                         'comb-balance-mux). Exact; for a module whose ladders grow with the design (a '
                         'select over every port of a shared cache). 0: none')
    ap.add_argument('--compress-adds', metavar='MIN_ADDENDS', type=int, default=0,
                    help='every model\'s sums of at least MIN_ADDENDS addends (a tree of additions, '
                         'zero-extended ones that cannot wrap included: a population count over many '
                         'inputs) as a compressor tree of full adders and one final addition, before '
                         'folding (foldsim-compress-add-trees, CIRCT\'s datapath compressor). Exact; 0: none')
    ap.add_argument('--prefix-negate', metavar='MIN_BITS', type=int, default=0,
                    help='negations of at least MIN_BITS bits (x & -x, the lowest set bit of a '
                         'priority select) as a logarithmic prefix OR instead of a carry chain, before '
                         'folding (foldsim-lower-wide-negation). Exact; 0: none')
    ap.add_argument('--retime-memory-reads', metavar='MIN_BITS', type=int,
                    help="memories of at least MIN_BITS bits without a ram_style hint, read "
                         'asynchronously at an address that is a function of registers alone (a '
                         "queue's head, a buffer's tail), read through a register instead: a target "
                         'cycle early, at the address\'s next value, reading new data (circt-opt '
                         'seq-retime-firmem-reads mode=auto). Exact; such a memory may then be '
                         'virtual (--virtual-memories) or in block RAM')
    ap.add_argument('--vmem-group', metavar='copy|N|auto[:M]',
                    help="with --virtual-memories: one kernel controller (foldsim_vmem_group) for a "
                         "class copy's virtual memories of the same row size and read-during-write "
                         "order (copy), or for at most N of them, or (auto) M controllers in all "
                         "(default 32) spread by expected accesses (--vmem-weights, else instances); "
                         "each memory keeps its own region within the controller's, and the "
                         "simulator's ports are unchanged. Absent: a controller per memory")
    ap.add_argument('--vmem-weights', metavar='JSON',
                    help='with --vmem-group auto: each memory\'s expected accesses per instance (a JSON '
                         'object by port vm<k> or memory name, e.g. from a profiling run; default 1)')
    ap.add_argument('--vmem-row-bytes-min', type=int, default=1, metavar='BYTES',
                    help='with --virtual-memories: a row takes its bytes rounded up to a power of two '
                         'in device memory, at least this many (default %(default)s: rows below 64 '
                         'bytes share lines; 64: the v6 layout)')
    ap.add_argument('--vmem-known-zero-bits', type=int, default=1 << 16, metavar='N',
                    help="with --virtual-memories: each controller's known-zero bits at most (a bit per "
                         "block of rows, the smallest block within N; default %(default)s); 0: none")
    ap.add_argument('--vmem-forward', type=int, choices=(0, 1), default=1,
                    help="with virtual memories (or page engines, link streams): whether a read is "
                         "answered from its controller's write buffer (1, default) or waits for the "
                         "write's B and reads memory (0: the kernel's foldsim_vmem FORWARD 0, the "
                         'buffer with one read port; for folds whose static order puts a row\'s read '
                         'long after its write)')
    ap.add_argument('--vmem-cache-budget', type=float, default=VMEM_CACHE_BUDGET, metavar='RAMB36',
                    help='with --vmem-cache-rows auto: the block RAM all the host caches\' lines may '
                         'take, in RAMB36 (default %(default)g, 5%% of the V80 dynamic region\'s)')
    ap.add_argument('--forward-after-levels', type=int, metavar='N',
                    help='with --static: a link read forwarded past the read register where the '
                         "reader's logic behind it stays below N levels (default 8; 0: into the "
                         'read register) (foldsim-fold-static)')
    ap.add_argument('--zero-chain-hops', type=int, metavar='N',
                    help="with --zero-latency-chains: at most N links taken in their producers' host "
                         'cycle on a combinational path; a link past them is taken a host cycle later '
                         '(default: no bound) (foldsim-fold-static)')
    ap.add_argument('--compact-links', action='store_true',
                    help="with --static: a link keeps only the producer instances its reader takes (a "
                         "per-lane port read by its own lane's instances, say), not an entry for every "
                         'instance (foldsim-fold-static)')
    ap.add_argument('--link-block-depth', type=int, default=0, metavar='N',
                    help='with --static: a link memory at least N deep that is its reader input\'s only '
                         'source and never forwarded is a block RAM, its output register the read '
                         'register (foldsim-fold-static; 0: none)')
    ap.add_argument('--link-ultra-depth', type=int, default=0, metavar='N',
                    help='as --link-block-depth, in ultra RAM (foldsim-fold-static; 0: none)')
    ap.add_argument('--link-hbm-lead', type=int, default=0, metavar='N',
                    help='with --static and hbm link kinds: the most positions a stream row\'s request '
                         'precedes its landing, the memory\'s round trip in host cycles and a margin '
                         '(foldsim-fold-static; 0: its default, 48)')
    ap.add_argument('--link-hbm-row-bits', type=int, default=0, metavar='N',
                    help='with --static and hbm link kinds: the most bits of a stream row, its controller\'s '
                         'data width (8 to 4096; foldsim-fold-static; 0: its default, 512). A wider word is cut '
                         'into rows a host cycle each, which spaces the producer\'s and reader\'s passes')
    ap.add_argument('--link-hbm-min-lead', type=int, default=0, metavar='N',
                    help='with --static and hbm link kinds: the least positions from a row\'s write to its '
                         'landing for a link to be streamed; less stays an on-chip link (foldsim-fold-static; '
                         '0: the decision lead + 6). The memory\'s round trip: no landing waits for it')
    ap.add_argument('--link-hbm-shift', type=int, default=0, metavar='N',
                    help='with --static, hbm link kinds and --page-engines: the most positions a stream\'s '
                         'request moves earlier, or its write later, so that streams whose requests or writes '
                         'fall at the same positions share an engine (within the room from the row\'s write to '
                         'its request; foldsim-fold-static; 0: none)')
    ap.add_argument('--read-buffer-bits', type=int, default=0, metavar='N',
                    help="with --static: a threaded memory's per-instance read-register buffer of at least N bits "
                         'read synchronously (a block RAM, read first) instead of combinationally (LUTs); 0: none '
                         '(foldsim-fold-static read-buffer-bits)')
    ap.add_argument('--fold-store-ram', type=int, default=32768, metavar='N',
                    help="with --static: folded pipelines' result stores of at least N bits read through a "
                         'register, a host cycle or two ahead, in block or ultra RAM (their slot counters '
                         'the copy\'s own; a write landing in between forwarded unless the schedule shows '
                         'none can). Exact; default 32768; 0: read combinationally, LUTs (foldsim-fold-static fold-store-ram)')
    ap.add_argument('--fold-store-style', choices=('auto', 'ultra', 'block'), default='auto',
                    help='with --fold-store-ram: the stores\' RAM style (auto: ultra where it takes fewer than half as '
                         'many primitives as block RAM, else block); a --storage rule <model>:fold<k>_store=<kind> takes precedence')
    ap.add_argument('--page-write-gate-pages', metavar='MODEL:N,...', default='',
                    help='with --page-write-gate: per model, gate only the first N pages of each column; the rest '
                         'through the page select (foldsim-fold-static page-write-gate-pages)')
    ap.add_argument('--page-write-gate', action='store_true',
                    help='with --static and --page-engines: off-chip pages written back through a gated '
                         'register a page (loaded the host cycle before its write, else held at zero) and an OR '
                         'of the pages, instead of a p:1 select by the page index: about (p - 1) / 5 LUTs a '
                         'page bit instead of (p - 1) / 3, at a register a paged bit (foldsim-fold-static '
                         'page-write-gate)')
    ap.add_argument('--link-hbm-fit-slots', action='store_true',
                    help='with --static, hbm link kinds and --page-engines: each stream takes the fewest landing '
                         'slots its requests need (a power of two, at least 8; foldsim-fold-static)')
    ap.add_argument('--fold-ports', default='', metavar='REGEX',
                    help='with --static: netlist port bundles folded into two record streams (req_* / rsp_*), '
                         'the regex\'s first capture the bundle index, its second the field, e.g. '
                         '\'m_axi_mem_([0-9]+)_(.*)\' (POSIX extended), or without a group a prefix: '
                         'm_axi_mem_ for m_axi_mem_<bundle>_<field> (foldsim-fold-static; foldsim_sim.vh FOLDSIM_PORTS_*)')
    ap.add_argument('--fold-ports-reset', default='reset', metavar='INPUT',
                    help='with --fold-ports: the netlist input whose frame token rides on each request record')
    ap.add_argument('--link-kinds', default='', metavar='P:R:KIND,...',
                    help='with --static: the RAM (distributed, block or ultra) of the links from model P '
                         'to model R that --link-block-depth\'s rules let block or ultra RAM hold, '
                         'whatever their depth (--state auto chooses them; foldsim-fold-static)')
    ap.add_argument('--stage-levels', type=int, default=0, metavar='N',
                    help='with --static: a class whose state cone is deeper than N estimated logic '
                         'levels (no black box or folded pipeline) evaluates over several host '
                         'cycles, its logic cut into segments of at most N levels by stage '
                         'registers that step with the frame; each pass reads its state and '
                         'inputs the stages before its slot, so late outputs take the stages as '
                         'latency (foldsim-fold-static; 0: none)')
    ap.add_argument('--multicycle', default='', metavar='CLASS[:M],...',
                    help='with --stage-levels: these classes (globs) staged without stage registers: '
                         'one evaluation at a time over M host cycles (the cone over the levels a '
                         'host cycle, or M as given), from capture registers taken at the read to end registers '
                         'taken before the state pass; the schedule spaces their state passes M '
                         'apart. <out>/foldsim_multicycle.xdc gives the flow the M-cycle paths, '
                         'foldsim_sim.json lists them (foldsim-fold-static)')
    ap.add_argument('--state-in-ram', default='', metavar='CLASS[:MIN_BITS[:P|auto]],...',
                    help='with --static: state in RAM (SiR) for these classes (globs on the model name): the '
                         'written register bits of an instance (at least MIN_BITS, default 1) kept in a RAM of a '
                         'row an instance times P plus a pass, in 72-bit columns with a mask lane a write enable. '
                         'P = 1 (default): read in place two host cycles ahead, written a host cycle after the '
                         'state pass (state gap 4); P > 1: built as on-chip pages with masks (capture registers, '
                         'a P:1 write select). Style by --storage MODEL:sir=block|ultra, else block while the '
                         'rows fit 512 x 72, ultra otherwise (foldsim-fold-static state-in-ram)')
    ap.add_argument('--state-in-ram-read-bits', type=int, default=0, metavar='N',
                    help='with --state-in-ram: P = ceil(bits / N) for a class whose spec gives no P '
                         '(foldsim-fold-static state-in-ram-read-bits; 0: P = 1)')
    ap.add_argument('--state-in-ram-h', type=int, default=0, metavar='N',
                    help='with --state-in-ram and P auto: the most host cycles a copy\'s passes take '
                         '(instances x P; foldsim-fold-static state-in-ram-h; 0: no bound)')
    ap.add_argument('--pass-split', default='', metavar='CLASS:P,...',
                    help='these classes (globs; CLASS:P[:S]): the isomorphic fanout-free cones (lanes) of '
                         'the logic only registers\' next values use run on ceil(R/P) shared lanes, one '
                         'replica a pass of S host cycles after the state pass, from its leaves captured '
                         'there; those registers are written (P + 1) x S host cycles after it. '
                         'foldsim_multicycle.xdc gives the flow the passes\' paths (foldsim-fold-static)')
    ap.add_argument('--staged-links', action='store_true',
                    help="with --static: an early link of one producer instance in two registers, "
                         'copied at a fixed position of the frame, where the schedule allows '
                         '(foldsim-fold-static)')
    ap.add_argument('--staged-link-memories', action='store_true',
                    help="with --staged-links: the staged links of one producer port to one reader class "
                         "keep their first registers as one distributed-RAM memory, each link's second "
                         "register loaded from it at its own copy position (foldsim-fold-static)")
    ap.add_argument('--staged-link-direct', action=argparse.BooleanOptionalAction, default=True,
                    help="with --staged-links: a reader of one instance reads a staged link's second register "
                         'in place, with no read register of its own (the copy a host cycle later in the frame): '
                         'two flops a link bit instead of three (foldsim-fold-static staged-link-direct)')
    ap.add_argument('--drop-dead-inputs', action=argparse.BooleanOptionalAction, default=True,
                    help='with --static: a model input its body never reads takes no link (no storage, no read '
                         'register) (foldsim-fold-static drop-dead-inputs)')
    ap.add_argument('--share-links', action='store_true',
                    help='with --static: outputs of a model driven by the same value (a bus broadcast '
                         'to several children) share one link (foldsim-fold-static)')
    ap.add_argument('--registered-links', default='', metavar='CLASS,...',
                    help='with --zero-latency-links: classes (globs) whose links a reader always takes '
                         "from its read register, never in their producer's host cycle: a class whose "
                         "late outputs end a deep path (foldsim-fold-static)")
    ap.add_argument('--no-share-link-writes', action='store_true',
                    help="with --static: write each reader's link from the producer's bits in the "
                         'pass, not a host cycle late from one register of them (foldsim-fold-static)')
    ap.add_argument('--page', default='', metavar='P|MODEL:P,...',
                    help='with --static: paged register state (foldsim-fold-static page), or auto: the '
                         'paging that fits the V80 region at the shortest frame (foldsim_degree.py '
                         'choose_pages). A '
                         "copy of several instances keeps its registers in a page RAM per "
                         'region and moves an instance\'s state in and out P host cycles at a '
                         'time, double-buffered; a count pages every model, else per model')
    ap.add_argument('--page-memory-bits', type=int, metavar='N',
                    help='with --page: memories read combinationally of at most N bits page as '
                         'rows with the registers (foldsim-fold-static page-memory-bits; default '
                         '4096, 0: registers only)')
    ap.add_argument('--state', choices=('auto',),
                    help="with --static: where each class's state lives, priced "
                         '(foldsim_degree.py choose_state): on chip (flops, distributed, block or '
                         'ultra RAM), paged on chip (--page, larger memories as rows: --page-rows), '
                         'or in HBM (virtual memories, pages off chip), the least state off chip '
                         'that lets the fold place and route on the V80 region at the shortest '
                         'frame, H projected; the choice in <out>/state.txt')
    ap.add_argument('--page-rows', metavar='REGEX',
                    help='with --page: the memories (<model>:<name>) that page as rows whatever '
                         'their size, as --state auto chooses them (foldsim-fold-static page-rows)')
    ap.add_argument('--page-memories', metavar='REGEX',
                    help='with --page: only the memories this regex finds page as rows '
                         '(foldsim-fold-static page-memories)')
    ap.add_argument('--page-hbm-window', type=int, metavar='K',
                    help='with pages off chip: an on-chip window of at most K instances\' pages per '
                         'copy, prefetched K state passes ahead (foldsim-fold-static page-hbm-window)')
    ap.add_argument('--page-hbm-column-bits', type=int, metavar='N',
                    help='with --page and --storage <model>:pages=hbm: the most bits of a page word '
                         'in one virtual memory row (foldsim-fold-static page-hbm-column-bits; '
                         'default 4096)')
    ap.add_argument('--page-engines', metavar='N|auto',
                    help='with pages off chip: the page columns carried by at most N shared engines '
                         '(HBM page streams: one virtual memory of tagged requests each, columns of '
                         'copies whose page traffic falls in other positions sharing one), instead of '
                         'a controller per column; auto: as few as the positions allow '
                         '(foldsim-fold-static page-engines)')
    ap.add_argument('--no-page-masks', action='store_true',
                    help='with --page: a dense page word, a register under an enable writing its old '
                         'value where the enable is low, instead of byte write masks by enable '
                         '(foldsim-fold-static page-masks=false)')
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
    ap.add_argument('--degree-estimator', default='foldsim_cost.py', choices=('foldsim_cost.py', 'beacon-area'),
                    help='the area model of --copies / --page / --state auto: the IR cost model (default: '
                         'the calibrations fitted on it) or Beacon\'s area (foldsim-estimate-area)')
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
    ap.add_argument('--memory-registers', default='', metavar='auto|off|REGEX',
                    help='memories kept as registers, a register an entry (foldsim-memories-to-'
                         'registers): `auto`, those cheaper so for every copy of their instances '
                         '(foldsim_storage.py: an associative table read at every entry); `off`, '
                         'none; else those whose name or <module>:<name> the regex finds. '
                         'Default: auto with --storage auto, else off. The choice in '
                         '<out>/memory_registers.txt')
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
    ap.add_argument('--slr-region', default='', metavar='JSON',
                    help='with --slr-groups: the dynamic region\'s SLRs, unequal (JSON: per SLR its name '
                         'and LUTs, the margin a group may fill and LUT sites per placed LUT; the V80 '
                         'compute shell\'s: foldsim_v80_slr.json, or `v80`): one group per SLR, each '
                         'within its own SLR\'s capacity and assigned to that SLR by name in '
                         '<out>/foldsim_floorplan.tcl; replaces --slr-count and --slr-capacity')
    ap.add_argument('--sll-check', action='store_true',
                    help='with --static and --slr-region: check the generated netlist against the region\'s SLRs '
                         '(foldsim_sll_check.py: each copy\'s LUT, LUTRAM, FF, URAM and RAMB36 counted on the '
                         'netlist, an assignment within the SLRs, the SLLs each boundary takes as placed), report '
                         'in <out>/sll/sll.txt, and fail on a refusal; with --slr-groups the groups also keep the '
                         'boundary budget (sll_per_boundary x sll_margin) in link bits')
    ap.add_argument('--slr-sites', default='', metavar='JSON',
                    help="with --slr-groups: each copy's LUTs measured on a generated netlist of the same recipe "
                         '(JSON {copy: {"lut": n}}, foldsim_sll_check.py netlist resources) instead of the '
                         "estimate, in LUT sites (the region's sites_per_lut)")
    ap.add_argument('--slr-serialize', default='', metavar='[GLOB:]K,...',
                    help='with --slr-groups: the links between groups of the given classes (globs) carried '
                         'over K host cycles on a K-th of the wires (SLLs), K alone for every class; the '
                         "reader takes them K host cycles later and the producer's passes of a kind are kept "
                         'K apart (foldsim-fold-static slr-serialize)')
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
    ap.add_argument('--fanout', type=int, default=0,
                    help='with --static: every other register (copies\' state, read selects) of more loads '
                         'than this is replicated the same way, each replica at most this many '
                         '(foldsim-replicate-controls fanout); 0: none')
    ap.add_argument('--fanout-logic', action='store_true',
                    help='with --fanout: one-LUT logic between such a register and its loads (a reset or '
                         'enable decoded from a few flops) copied with each replica')
    ap.add_argument('classes', nargs='*',
                    help='none: the whole target one model of one thread, unthreaded')
    return ap


def main():
    ap = build_parser()
    a = foldsim_manifest.parse_args(ap)
    if a.degree_estimator != 'foldsim_cost.py':
        sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
        import foldsim_degree
        foldsim_degree.set_estimator(a.degree_estimator)
    for opt, v in (('--srcs or --src', a.srcs or a.src), ('--top', a.top), ('--out', a.out)):
        if not v:
            ap.error(f'{opt} is required (on the command line or in the manifest)')
    tool = lambda t: os.path.join(a.circt_bin, t) if a.circt_bin else t
    here = os.path.dirname(os.path.realpath(__file__))
    foldsim_bin = a.foldsim_bin or (here if os.path.exists(os.path.join(here, 'foldsim-opt')) else '')
    foldsim_opt = os.path.join(foldsim_bin, 'foldsim-opt') if foldsim_bin else 'foldsim-opt'
    os.makedirs(a.out, exist_ok=True)
    out = lambda f: os.path.join(a.out, f)

    defines, incdirs, files = [], [], []
    if a.srcs:
        read_srcs(a.srcs, defines, incdirs, files)
    defines += ['-D' + d for d in a.define]
    incdirs += ['-I' + d for d in a.incdir]
    files += a.src
    files = rewrite_sources(files, a.rewrite, out('rewritten'))
    design = out('design.mlir')
    run([tool('circt-verilog'), '--ir-hw', f'--top={a.top}', '--allow-use-before-declare',
         '--allow-hierarchical-const', '--ignore-uninstantiated-modules', '--detect-memories',
         '--preserve-unpacked-ports',
         *([f'-Xslang=--max-generate-steps={a.max_generate_steps}'] if a.max_generate_steps else []),
         *([f'--max-unroll-trip-count={a.max_unroll_trip_count}'] if a.max_unroll_trip_count else []),
         *defines, *incdirs, *files, '-o', design])

    # A module declared a black box (a core the synthesis flow provides) stays
    # an instance through flattening; foldsim-fame5 folds it by its contract.
    if 'black_box' in open(design).read():
        run([foldsim_opt, design, '--foldsim-black-boxes', '-o', design + '.tmp'])
        os.replace(design + '.tmp', design)

    if a.class_at:
        specs = []
        for spec in a.class_at:
            tag, sep, glob = spec.partition('@')
            if not sep or not re.fullmatch(r'[A-Za-z_]\w*', tag) or not glob:
                sys.exit(f'--class-at {spec}: not TAG@GLOB')
            specs.append((tag, glob))
        text, new = class_at(open(design).read(), a.top, specs)
        with open(design, 'w') as f:
            f.write(text)
        named = {m for c in a.classes for m in c.split('+')}
        a.classes = a.classes + [n for n in new if n not in named]
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
         '--canonicalize', '--cse', f'--seq-retime-firmem-reads=report={out("memories.rpt")}'
         + (f' mode=auto min-bits={a.retime_memory_reads}' if a.retime_memory_reads is not None else ''),
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
    # State no output observes goes before the models are cut (their ports then
    # carry only what is read) and again in each model (OPTIMIZE).
    passes = ['hw.module(foldsim-remove-dead-state)']
    registers = memory_registers(a, flat, out)
    if registers:
        passes.append(registers)
    for m in order:
        models = f' models={",".join(sorted(children[m]))}' if children[m] else ''
        exclusive = f' absorb-exclusive={a.absorb_exclusive}' if a.absorb_exclusive else ''
        if a.absorb_exclusive and a.absorb_registers:
            exclusive += ' absorb-registers'
            if a.absorb_register_logic:
                exclusive += f' absorb-register-logic={a.absorb_register_logic}'
        passes.append(f'foldsim-extract-models{{module={m}{models} netlist absorb-logic={ABSORB_LOGIC}{exclusive}}}')
        fused = [g for g in groups if set(g) <= children[m]]
        if fused:
            passes.append(f'foldsim-fuse-models{{module={m} groups={",".join("+".join(g) for g in fused)}}}')
    for g in groups:
        if not any(set(g) <= children[m] for m in order):
            sys.exit(f'{"+".join(g)}: not instantiated by the same module')
    # Depth the design's own structure gives a select over many inputs, in every
    # model before it is folded (exact rewrites; the RTL is unchanged).
    if a.prefix_negate:
        passes.append(f'hw.module(foldsim-lower-wide-negation{{min-width={a.prefix_negate}}})')
    if a.balance_mux:
        passes.append(f'hw.module(comb-balance-mux{{mux-chain-threshold={a.balance_mux}}})')
    if a.compress_adds:
        passes.append(f'hw.module(foldsim-compress-add-trees{{min-operands={a.compress_adds}}})')
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
        if not a.virtual_memories and a.state != 'auto':
            sys.exit('--vmem-cache-rows needs --virtual-memories (or --state auto)')
        if a.vmem_cache_rows != 'auto' and not a.vmem_cache_rows.isdigit():
            sys.exit('--vmem-cache-rows: auto, 0 or a power of two of at least 2')
        vmem_cache_rows([], a.vmem_cache_rows, a.vmem_cache_budget)
    if a.state and not a.static:
        sys.exit('--state needs --static')
    if a.state and a.page == 'auto':
        sys.exit('--state auto chooses the pages: no --page auto with it')
    if a.page and not a.static:
        sys.exit('--page needs --static')
    if a.vmem_group is not None:
        if not a.virtual_memories and a.state != 'auto':
            sys.exit('--vmem-group needs --virtual-memories (or --state auto)')
        if not (a.vmem_group == 'copy' or (a.vmem_group.isdigit() and int(a.vmem_group) >= 1)
                or re.fullmatch(r'auto(:[1-9][0-9]*)?', a.vmem_group)):
            sys.exit('--vmem-group: copy, the memories per controller (at least 1), or auto[:M]')
    if a.vmem_weights and not (a.vmem_group or '').startswith('auto'):
        sys.exit('--vmem-weights needs --vmem-group auto')
    m = a.vmem_row_bytes_min
    if m < 1 or m & (m - 1):
        sys.exit('--vmem-row-bytes-min: a power of two')
    if a.vmem_known_zero_bits != 0 and a.vmem_known_zero_bits < 16:
        sys.exit('--vmem-known-zero-bits: 0 (none) or at least 16')
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
            # Storage rules that change the schedule (pages kept off chip) apply here too.
            if a.storage and a.storage != 'auto':
                fold += f' storage={a.storage}'
            if a.nest:
                fold += ' nest'
            fold += frame_options(a)
            if a.state == 'auto':
                fold = choose_state(a, foldsim_opt, flat, passes, out, fold)
            fold += ' schedule-only'
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
        if a.page == 'auto':
            spec, hbm = choose_pages(a, foldsim_opt, flat, passes, out, fold)
            if spec:
                fold += f' page={spec}'
            if hbm:
                m = re.search(r' storage=(\S+)', fold)
                fold = (fold.replace(m.group(0), f' storage={m.group(1)},{",".join(hbm)}') if m
                        else fold + f' storage={",".join(hbm)}')
        if a.state == 'auto':
            fold = choose_state(a, foldsim_opt, flat, passes, out, fold)
        if a.slr_groups or a.sll_check:
            if a.copies != 'auto':
                run([foldsim_opt, flat, f'--pass-pipeline=builtin.module({",".join(passes + OPTIMIZE)})',
                     '-o', out('models.mlir')])
            if a.slr_groups:
                fold += slr_options(a, foldsim_opt, out('models.mlir'), fold, out)
        sll_fold = fold
        rc = f'loads={a.control_fanout}'
        if a.fanout:
            rc += f' fanout={a.fanout}' + (' fanout-logic' if a.fanout_logic else '')
        replicate = f'hw.module(foldsim-replicate-controls{{{rc}}})'
        pipeline = passes + OPTIMIZE + [f'foldsim-fold-static{{{fold}}}', SPLIT, replicate,
                                        'canonicalize', 'foldsim-sort', 'cse']
        if a.copies == 'auto' or a.slr_groups or a.sll_check:
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

    if a.sll_check and not (a.slr_region and a.static):
        sys.exit('--sll-check needs --static and --slr-region')
    if a.slr_groups:
        if not a.static:
            sys.exit('--slr-groups needs --static')
        names = [n for n, _ in slr_region_capacities(a.slr_region)] if a.slr_region else None
        write_slr_groups(open(out('foldsim_sim.sv')).read(), out('foldsim_floorplan.tcl'), names)

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
        # An optional ninth field: the tag bits at the top of the instance
        # port (an HBM link stream's landing tags; foldsim_vmem TAGW): the
        # controller's instance index is the port's other bits.
        # A tenth, `pages`: an off-chip page column, whose read and write of a
        # host cycle may be of two instances (never in a group).
        name, cls, iw, aw, dw, mw, rf, insts, *tag = entry.split(':')
        info['vmem'].append({'port': f'vm{k}', 'memory': name, 'class': cls, 'inst_width': int(iw),
                             'row_width': int(aw), 'data_width': int(dw), 'mask_width': int(mw),
                             'read_first': rf == '1', 'instances': int(insts)})
        if tag and int(tag[0]):
            info['vmem'][-1]['tag_width'] = int(tag[0])
        if len(tag) > 1 and tag[1] == 'pages':
            info['vmem'][-1]['pages'] = True
    # Each memory's region in the kernel's vmem space (VMEM_INTERFACE.md):
    # row r of instance i at ((r * INSTANCES + i) << log2(ROW_BYTES)), a row
    # its bytes rounded up to a power of two, at least --vmem-row-bytes-min
    # (v7; 64 in v6); regions 4 KiB aligned. A controller of its own has
    # known-zero bits within --vmem-known-zero-bits.
    offset = 0
    for v in info['vmem']:
        row_bytes = max(a.vmem_row_bytes_min, 1 << ((v['data_width'] + 7) // 8 - 1).bit_length())
        size = (v['instances'] << v['row_width']) * row_bytes
        v.update(row_bytes=row_bytes, base=offset, bytes=size)
        v['known_zero'], v['kz_shift'] = vmem_known_zero((1 << v['row_width']) * v['instances'],
                                                         a.vmem_known_zero_bits)
        # An HBM link stream's rows: each written before it is read but in
        # the first frames, whose reads the runtime's cleared image answers;
        # no known-zero bits (a bit a row would be block RAM).
        if v['class'].startswith(('__link', '__pages')):
            v['known_zero'], v['kz_shift'] = 0, 0
        offset += -(-size // 4096) * 4096
    info['vmem_bytes'] = offset
    if a.vmem_group is not None and info['vmem']:
        weights = json.load(open(a.vmem_weights)) if a.vmem_weights else None
        info['vmem_groups'] = vmem_groups(info['vmem'], a.vmem_group, weights, a.vmem_known_zero_bits)
        info['vmem_bytes'] = sum(-(-g['bytes'] // 4096) * 4096 for g in info['vmem_groups'])
        if a.vmem_cache_rows is not None:
            pseudo = [{'instances': g['instances'] * g['rows'], 'row_width': 0, 'class': g['class'],
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
    # Multicycle copies: their capture-to-end paths, M host cycles, as the
    # flow's timing constraints (foldsim_multicycle.xdc, empty without any).
    mc = re.search(r'foldsim\.multicycle = \[(.*?)\]', sim_text)
    info['multicycle'] = []
    for entry in re.findall(r'"([^"]*)"', mc.group(1)) if mc else []:
        prefix, cycles, caps, ends = entry.split(':')
        info['multicycle'].append({'prefix': prefix, 'cycles': int(cycles),
                                   'capture_bits': int(caps), 'end_bits': int(ends)})
    ps = re.search(r'foldsim\.pass_split = \[(.*?)\]', sim_text)
    splits = []
    for e in re.findall(r'"([^"]*)"', ps.group(1)) if ps else []:
        prefix, span, passes, hold = e.rsplit(':', 3)
        splits.append({'prefix': prefix, 'span': int(span), 'passes': int(passes), 'hold_bits': int(hold)})
    if splits:
        info['pass_split'] = splits
    write_multicycle_xdc(info['multicycle'], out('foldsim_multicycle.xdc'), splits)
    if info.get('vmem'):
        info['vmem_forward'] = a.vmem_forward
    # Folded port bundles: the records' layouts (foldsim.ports).
    pf = re.search(r'foldsim\.ports = \[(.*?)\]', sim_text)
    if pf:
        entries = re.findall(r'"([^"]*)"', pf.group(1))
        _, n, cw = entries[0].split(':')
        info['port_bundles'] = {'count': int(n), 'chan_width': int(cw), 'req': [], 'rsp': []}
        for e in entries[1:]:
            kind, name, lo, w = e.split(':')
            info['port_bundles'][kind.lower()].append({'name': name, 'lo': int(lo), 'width': int(w)})
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
        if 'port_bundles' in info:
            pf = info['port_bundles']
            f.write(f'`define FOLDSIM_PORTS {pf["count"]}\n')
            f.write(f'`define FOLDSIM_PORTS_CW {pf["chan_width"]}\n')
            for kind in ('req', 'rsp'):
                total = sum(x['width'] for x in pf[kind])
                f.write(f'`define FOLDSIM_PORTS_{kind.upper()}_W {total}\n')
                for x in pf[kind]:
                    f.write(f'`define FOLDSIM_PORTS_{kind.upper()}_{x["name"].upper()}_LO {x["lo"]}\n')
                    f.write(f'`define FOLDSIM_PORTS_{kind.upper()}_{x["name"].upper()}_W {x["width"]}\n')
        if info['vmem']:
            f.write(f'`define FOLDSIM_VMEM_COUNT {len(info["vmem"])}\n')
            if not a.vmem_forward:
                f.write('`define FOLDSIM_VMEM_FORWARD 0\n')
            # Tagged requests (TAGW, the port's top bits): every memory's
            # TAGW then (0 but the tagged), and every group's.
            tags = any(v.get('tag_width') for v in info['vmem'])
            if tags:
                f.write('`define FOLDSIM_VMEM_TAGS\n')
            for k, v in enumerate(info['vmem']):
                for key, d in (('IW', 'inst_width'), ('AW', 'row_width'), ('DW', 'data_width'),
                               ('MW', 'mask_width'), ('INSTANCES', 'instances')):
                    f.write(f'`define FOLDSIM_VMEM_{k}_{key} {v[d]}\n')
                f.write(f'`define FOLDSIM_VMEM_{k}_RDWRF {int(v["read_first"])}\n')
                if tags:
                    f.write(f'`define FOLDSIM_VMEM_{k}_TAGW {v.get("tag_width", 0)}\n')
                f.write(f'`define FOLDSIM_VMEM_{k}_BASE 64\'h{v["base"]:x}\n')
                f.write(f'`define FOLDSIM_VMEM_{k}_ROW_BYTES {v["row_bytes"]}\n')
                if 'vmem_groups' not in info:
                    f.write(f'`define FOLDSIM_VMEM_{k}_KNOWN_ZERO {v["known_zero"]}\n')
                    f.write(f'`define FOLDSIM_VMEM_{k}_KZ_SHIFT {v["kz_shift"]}\n')
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
                    if tags:
                        f.write(f'`define FOLDSIM_VMEM_G{g}_TAGW {v.get("tag_width", 0)}\n')
                    f.write(f'`define FOLDSIM_VMEM_G{g}_BASE 64\'h{v["base"]:x}\n')
                    f.write(f'`define FOLDSIM_VMEM_G{g}_ROW_BYTES {v["row_bytes"]}\n')
                    f.write(f'`define FOLDSIM_VMEM_G{g}_KNOWN_ZERO {v["known_zero"]}\n')
                    f.write(f'`define FOLDSIM_VMEM_G{g}_KZ_SHIFT {v["kz_shift"]}\n')
                    if 'cache_rows' in v:
                        f.write(f'`define FOLDSIM_VMEM_G{g}_CACHE_ROWS {v["cache_rows"]}\n')
            f.write(f'`define FOLDSIM_VMEM_BYTES 64\'h{info["vmem_bytes"]:x}\n')
    print(f'foldsim: {", ".join(f"{k} x{v}" for k, v in info["threads"].items())}', file=sys.stderr)
    if a.sll_check:
        # The per-copy SLR / SLL check on the netlist just generated (its
        # stage registers, read buffers and simulator level counted; schedule
        # estimates said "fits" where placement failed): <out>/sll/sll.txt;
        # a refusal fails the generation, its outputs kept.
        r = run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_sll_check.py'),
                 a.out, '--models', out('models.mlir'), '--fold-options', sll_fold, '--foldsim-opt', foldsim_opt,
                 '--slr-region', a.slr_region, '--out', out('sll')], check=False, capture_output=True)
        verdict = next((l for l in reversed(r.stdout.splitlines()) if re.search(r': (OK|WARN|REFUSE):', l)), None)
        print(f'foldsim: SLL check: {verdict or "failed: " + (r.stderr or r.stdout)[-2000:]}', file=sys.stderr)
        if r.returncode:
            sys.exit(f'foldsim: --sll-check refuses this netlist (report {out("sll")}/sll.txt)')


if __name__ == '__main__':
    main()
