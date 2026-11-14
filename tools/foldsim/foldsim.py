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
import collections
import fnmatch
import json
import os
import re
import shlex
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
# `--class-at TAG@nest:GLOB`: the class's own instances are searched too, so a
# class may hold classes (an arbiter whose per-output buffers are a class).
NEST = 'nest:'


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
    whole tree where it is one arbiter). A GLOB written `nest:GLOB` takes its
    match as a class and goes on searching below it, so that the class holds
    the classes the other GLOBs make of its own instances. Several GLOBs may
    share a TAG. Returns the design and the new modules' names."""
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

    nested = [g.startswith(NEST) for _, g in specs]
    specs = [(t, g[len(NEST):] if n else g) for (t, g), n in zip(specs, nested)]
    innermost = [g.startswith(INNER) for _, g in specs]
    globs = [(t, g[len(INNER):] if inner else g) for (t, g), inner in zip(specs, innermost)]

    def mod_of(leaf):
        """A leaf is its module, or (module, variant) for a `nest:` class
        whose own instances changed."""
        return leaf[0] if isinstance(leaf, tuple) else leaf

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
                leaf = callee
                if nested[k] and callee in body:
                    v = walk(callee, p)
                    if v is not None:
                        leaf = (callee, v)
                leaves.append((tag, leaf))
                changes.append((rel, ('class', tag, leaf)))
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
        copy(mod_of(leaf), new)

    def retarget(mod, rel, new):
        i = span[mod][0] + rel
        m = INSTANCE.search(lines[i])
        lines[i] = lines[i][:m.start(3)] + new + lines[i][m.end(3):]

    for (mod, v), new in vname.items():
        for rel, (kind, a1, a2) in variants[mod][v]:
            retarget(new, rel, name[(a1, a2)] if kind == 'class' else vname[(a1, a2)])
    # A `nest:` class's copy takes its own instances' changes.
    for (tag, leaf), new in name.items():
        if isinstance(leaf, tuple):
            for rel, (kind, a1, a2) in variants[leaf[0]][leaf[1]]:
                retarget(new, rel, name[(a1, a2)] if kind == 'class' else vname[(a1, a2)])
    for (tag, leaf), new in name.items():
        n = sum(1 for t, l in leaves if (t, l) == (tag, leaf))
        print(f'class-at {tag}: {new} ({mod_of(leaf)}) x{n}', file=sys.stderr)
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
    if getattr(a, 'affine_slots', False):
        fold += ' affine-slots'
    if getattr(a, 'affine_auto', False):
        fold += ' affine-auto'
    if getattr(a, 'affine_auto_segments', None):
        fold += f' affine-auto-segments={a.affine_auto_segments}'
    if getattr(a, 'affine_stride', None):
        fold += f' affine-stride={a.affine_stride}'
    if getattr(a, 'affine_auto_bits', None) is not None:
        fold += f' affine-auto-bits={a.affine_auto_bits}'
    if getattr(a, 'slot_segments', 0):
        fold += f' slot-segments={a.slot_segments}'
    if getattr(a, 'slot_column_affine', False):
        fold += ' slot-column-affine'
    if getattr(a, 'slot_table_compose', False):
        fold += ' slot-table-compose'
    if getattr(a, 'pp_sync_banks', 0):
        fold += f' pp-sync-banks={a.pp_sync_banks}'
    if getattr(a, 'slot_ahead', 0):
        fold += f' slot-ahead={a.slot_ahead}'
    if getattr(a, 'modulo_linear', None) is not None:
        fold += f' modulo-linear={a.modulo_linear}'
    if getattr(a, 'ready_choice', ''):
        fold += f' ready-choice={a.ready_choice}'
    if a.interleave_copies:
        fold += ' interleave-copies'
    if getattr(a, 'csf_ready_copies', ''):
        fold += f' csf-ready-copies={a.csf_ready_copies}'
    if getattr(a, 'csf_ready_cones', False):
        fold += ' csf-ready-cones'
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
    if getattr(a, 'vmem_auto_bits', 0):
        fold += f' vmem-auto-bits={a.vmem_auto_bits}'
    if getattr(a, 'vmem_auto_instance_bits', 0):
        fold += f' vmem-auto-instance-bits={a.vmem_auto_instance_bits}'
    if getattr(a, 'vmem_cost', False):
        fold += ' vmem-cost'
    if getattr(a, 'vmem_rings', 0):
        fold += f' vmem-rings={a.vmem_rings} vmem-ring-entries={a.vmem_ring_entries}'
    if getattr(a, 'fold_store_rings', 0):
        fold += f' fold-store-rings={a.fold_store_rings}'
    if getattr(a, 'vmem_rings', 0) or getattr(a, 'fold_store_rings', 0):
        if getattr(a, 'vmem_ring_flight', 32) != 32:
            fold += f' vmem-ring-flight={a.vmem_ring_flight}'
        if a.vmem_ring_image_each:
            fold += ' vmem-ring-image-each=true'
        if getattr(a, 'vmem_ring_class_control', False):
            fold += ' vmem-ring-class-control=true'
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
    if getattr(a, 'zero_chain_levels', None):
        fold += f' zero-chain-levels={a.zero_chain_levels}'
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
    if getattr(a, 'page_engines_pp', False):
        fold += ' page-engines-pp'
    if getattr(a, 'vmem_on_engines', False):
        fold += ' vmem-on-engines'
    if getattr(a, 'vmem_stagger', 0):
        fold += f' vmem-stagger={a.vmem_stagger}'
    if getattr(a, 'vport_engines', False):
        fold += ' vport-engines'
        if a.vport_fence_shift != 4:
            fold += f' vport-fence-shift={a.vport_fence_shift}'
        if a.vport_hz:
            fold += f' vport-hz={a.vport_hz}'
        if a.vport_share:
            fold += f' vport-share={a.vport_share}'
        if a.vport_copy:
            fold += ' vport-copy'
        if a.vport_write_beats:
            fold += ' vport-write-beats'
        if a.vmem_fold_on_engines:
            fold += ' vmem-fold-on-engines'
    if getattr(a, 'vmem_slr_groups', ''):
        fold += ' vmem-slr-groups=' + ','.join(f'{c}:{g}' for c, g in vmem_slr_map(a.vmem_slr_groups))
    if getattr(a, 'port_plan_out', None):
        fold += f' port-plan-out={os.path.abspath(a.port_plan_out)}'
    if getattr(a, 'pp_beat_bits', 0):
        fold += f' pp-beat-bits={a.pp_beat_bits}'
    if getattr(a, 'link_hbm_beat_bits', 0):
        fold += f' link-hbm-beat-bits={a.link_hbm_beat_bits}'
    if getattr(a, 'link_hbm_beat_rows', False):
        fold += ' link-hbm-beat-rows'
    if getattr(a, 'link_landing_block', 0):
        fold += f' link-landing-block={a.link_landing_block}'
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
    if getattr(a, 'fold_store_hbm', 0):
        fold += f' fold-store-hbm={a.fold_store_hbm}'
    if getattr(a, 'fold_store_stream', False):
        fold += ' fold-store-stream'
    if getattr(a, 'fold_stage_levels', 0):
        fold += f' fold-stage-levels={a.fold_stage_levels}'
    if getattr(a, 'fold_stage_levels_class', ''):
        fold += f' fold-stage-levels-class={a.fold_stage_levels_class}'
    if getattr(a, 'fold_store_style', 'auto') != 'auto':
        fold += f' fold-store-style={a.fold_store_style}'
    if getattr(a, 'link_hbm_fit_slots', False):
        fold += ' link-hbm-fit-slots'
    if getattr(a, 'fold_ports', ''):
        fold += f' fold-ports={a.fold_ports}'
        if a.fold_ports_reset != 'reset':
            fold += f' fold-ports-reset={a.fold_ports_reset}'
        if getattr(a, 'fold_ports_rows', False):
            fold += ' fold-ports-rows'
    if getattr(a, 'link_kinds', ''):
        fold += f' link-kinds={a.link_kinds}'
    if getattr(a, 'link_kind_cost', False):
        # The device's URAM288, RAMB36 and LUTs at the fill target (the region's,
        # at --state-scratch-fill), unless given.
        dev = getattr(a, 'link_kind_device', '')
        if not dev:
            dev = ','.join(str(x) for x in region_kind_capacity(check_region(a),
                                                                getattr(a, 'state_scratch_fill', 0.85)))
        fold += f' link-kind-cost link-kind-device={dev}'
        if getattr(a, 'link_kind_others', ''):
            fold += f' link-kind-others={a.link_kind_others}'
        # With SLR groups (a region): each group's budget, the region's g-th
        # SLR's at the fill target; and what the rest takes of each, if given.
        if getattr(a, 'slr_groups', False) and getattr(a, 'slr_region', ''):
            fold += ' link-kind-groups=' + ';'.join(
                ','.join(str(x) for x in caps) for caps in region_group_capacity(
                    a.slr_region, getattr(a, 'state_scratch_fill', 0.85)))
            if getattr(a, 'link_kind_group_others', ''):
                fold += f' link-kind-group-others={a.link_kind_group_others}'
    if getattr(a, 'slr_kept', None):
        fold += f' slr-kept={a.slr_kept}'
    if getattr(a, 'slr_link_rams', False):
        fold += ' slr-link-rams'
    if getattr(a, 'slr_group_control', False):
        fold += ' slr-group-control'
    if getattr(a, 'slr_group_reset', False):
        fold += ' slr-group-reset'
    if getattr(a, 'slr_group_local', 0):
        fold += f' slr-group-local={a.slr_group_local}'
    if getattr(a, 'slr_group_own', 0):
        fold += f' slr-group-own={a.slr_group_own}'
    if getattr(a, 'registered_links', ''):
        fold += f' registered-links={a.registered_links}'
    if getattr(a, 'split_latency', ''):
        fold += f' split-latency={a.split_latency}'
    if getattr(a, 'split_latency_file', ''):
        fold += f' split-latency-file={os.path.abspath(a.split_latency_file)}'
    if getattr(a, 'stage_levels', 0):
        fold += f' stage-levels={a.stage_levels}'
    if getattr(a, 'stage_levels_class', ''):
        fold += f' stage-levels-class={a.stage_levels_class}'
    if getattr(a, 'stage_levels_mapped', ''):
        fold += f' stage-levels-mapped={a.stage_levels_mapped}'
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
        if getattr(a, 'state_in_ram_zero_links', False):
            fold += ' state-in-ram-zero-links'
        if getattr(a, 'state_in_ram_pp', False):
            fold += ' state-in-ram-pp'
        if getattr(a, 'state_in_ram_pp_replicas', 0):
            fold += f' state-in-ram-pp-replicas={a.state_in_ram_pp_replicas}'
        if getattr(a, 'state_in_ram_pp_no_pack', False):
            fold += ' state-in-ram-pp-pack=false'
        if getattr(a, 'state_in_ram_pp_fanout', 0):
            fold += f' state-in-ram-pp-fanout={a.state_in_ram_pp_fanout}'
    scratch = getattr(a, 'state_scratch_bits', 'auto')
    if scratch == 'auto':
        # The device's on-chip RAM at the placer's fill: the fold pass takes
        # the other memories it places off it (state-scratch-device-bits).
        uram, bram = (region_ram_bits(check_region(a), getattr(a, 'state_scratch_fill', 0.85))
                      if getattr(a, 'static', False) else (0, 0))
        if getattr(a, 'state_scratch_device', ''):
            uram, bram = (int(x) for x in a.state_scratch_device.split(','))
        if uram or bram:
            fold += f' state-scratch-device-bits={uram} state-scratch-device-block-bits={bram}'
    elif scratch is not None:
        fold += f' state-scratch-bits={int(scratch)}'
    if getattr(a, 'state_stream_window', 0):
        fold += f' state-stream-window={a.state_stream_window}'
    if getattr(a, 'state_scratch_reserve', ''):
        block, _, ultra = str(a.state_scratch_reserve).partition(',')
        fold += f' state-scratch-reserve={int(block)}'
        if ultra:
            fold += f' state-scratch-reserve-ultra={int(ultra)}'
    if getattr(a, 'state_scratch_whole', False):
        fold += ' state-scratch-partial=false'
    if getattr(a, 'state_scratch_fixed', False):
        fold += ' state-scratch-fixed'
    if getattr(a, 'state_scratch_fixed_pages', False):
        fold += ' state-scratch-fixed-pp=false'
    elif getattr(a, 'state_scratch_fixed', False) and getattr(a, 'state_in_ram_pp', False) \
            and not getattr(a, 'state_in_ram', ''):
        fold += ' state-in-ram-pp'
    if getattr(a, 'csf_single_bank', False):
        fold += ' csf-single-bank'
    if getattr(a, 'csf_mailbox_hbm_chains', False):
        fold += ' csf-mailbox-hbm-chains'
    if getattr(a, 'csf_mailbox_hbm_split', False):
        fold += ' csf-mailbox-hbm-split'
    if getattr(a, 'csf_mailbox_hbm_share', False):
        fold += ' csf-mailbox-hbm-share'
    if getattr(a, 'link_rings', 0):
        fold += f' link-rings={a.link_rings}'
    if getattr(a, 'csf_mailbox_builder', False):
        fold += ' csf-mailbox-builder'
    if getattr(a, 'csf_mailbox_hbm', False):
        fold += ' csf-mailbox-hbm'
    if getattr(a, 'csf_mailbox_hbm_row_bits', 0):
        fold += f' csf-mailbox-hbm-row-bits={a.csf_mailbox_hbm_row_bits}'
    if getattr(a, 'csf_mailbox_hbm_lead', 0):
        fold += f' csf-mailbox-hbm-lead={a.csf_mailbox_hbm_lead}'
    if getattr(a, 'csf_mailboxes', False):
        fold += ' csf-mailboxes'
        if getattr(a, 'csf_mailbox_latency', 1) != 1:
            fold += f' csf-mailbox-latency={a.csf_mailbox_latency}'
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
            # (Static, no copies asked for: each module's instances in one
            # copy, the only shape to price.)
            take, rows = foldsim_storage.memory_as_registers(
                m, instances[m['module']], one_copy=bool(a.static) and not a.copies)
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


# The state-home policy's calibration to synthesized LUTs by default
# (foldsim_degree.STATE_CAL: fitted on the 8-SM y8b image's post-synthesis
# hierarchy, foldsim_degree.py state-fit).
STATE_CALIBRATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibrations',
                                 'state-v80-y8b-20261113.json')


def state_measured(path, cal):
    """Per class (model) logic and LUTRAM factors of a generation of the same
    recipe (--state-measured <its out dir>): its copies as counted in its
    foldsim_sim.mlir (foldsim_degree.count_copies, times the calibration's
    count_to_synth) against what its --state auto priced them at (its
    state.json, with the factors it used)."""
    import foldsim_degree
    prev = json.load(open(os.path.join(path, 'state.json')))
    counted = foldsim_degree.count_copies(os.path.join(path, 'foldsim_sim.mlir'))
    k = (cal or {}).get('count_to_synth', 1.0)
    sums = {}
    for key, v in prev['copies'].items():
        c = key.rpartition('#')[0]
        if c not in counted:
            continue
        model = re.sub(r'_copy\d+$', '', c)
        x = sums.setdefault(model, [0.0, 0.0, 0.0, 0.0])
        x[0] += (counted[c]['lut'] - counted[c]['lutram']) * k
        x[1] += counted[c]['lutram'] * k
        x[2] += v['logic'] + v['store_lut']
        x[3] += v['store_lutram']
    out = {}
    for model, (ml, mr, pl, pr) in sums.items():
        fl, fr = prev['factors'].get(model, (1.0, 1.0))
        out[model] = dict(logic=round(fl * ml / pl, 3) if pl > 0 else fl,
                          lutram=round(fr * mr / pr, 3) if pr > 0 and mr > 0 else fr)
    return out


def state_policy(a, out):
    """The state-home policy of --state auto (foldsim_degree.state_policy):
    the calibration to synthesized LUTs, each copy's budget within the
    smallest SLR it may sit in, on chip before HBM; the --storage rules given
    kept as they are (GIVEN); a generation's measure (--state-measured).
    Returns a line for the log."""
    import foldsim_degree
    calp = '' if a.state_calibration == 'none' else (a.state_calibration or STATE_CALIBRATION)
    budget = None if a.state_copy_budget in ('', '0', 'off', 'none') else a.state_copy_budget
    region = a.slr_region if a.slr_region not in ('',) else 'v80'
    rf = None if a.state_region_budget in ('', '0', 'off', 'none') else float(a.state_region_budget)
    ram = None if a.state_ram_budget in ('', '0', 'off', 'none') else float(a.state_ram_budget)
    cap = foldsim_degree.state_policy(calp, budget, region, rf, ram)
    if a.state_home == 'onchip':
        foldsim_degree.storage.ONCHIP_ONLY = True
    if a.state_measured:
        if not foldsim_degree.STATE_CAL:
            sys.exit('--state-measured needs a --state-calibration')
        foldsim_degree.STATE_CAL['classes'] = state_measured(a.state_measured, foldsim_degree.STATE_CAL)
    if a.storage and a.storage != 'auto':
        foldsim_degree.GIVEN = foldsim_degree.given_rules(a.storage)
    return ('calibration ' + (os.path.basename(calp) if calp else 'none')
            + (f", copy budget {cap['lut']:.0f} LUTs (x{cap['frac']:g} of {cap['name']})" if cap else '')
            + (f", region {cap['region_lut']:.0f} LUTs (x{cap['region_frac']:g})" if cap and cap.get('region_lut') else '')
            + (f", region RAM x{cap['region_ram']:g}" if cap and cap.get('region_ram') else '')
            + (f", measured {a.state_measured}" if a.state_measured else '')
            + (', state on chip only' if a.state_home == 'onchip' else ''))


def state_budget_check(a, sim, out):
    """After the fold, each copy as counted in the netlist (synthesized:
    count_to_synth) against the policy's budget: the copies past it, and the
    measure to choose again with (--state-measured)."""
    import foldsim_degree
    cap, cal = foldsim_degree.COPY_CAP, foldsim_degree.STATE_CAL or {}
    if not cap:
        return
    k = cal.get('count_to_synth', 1.0)
    counted = foldsim_degree.count_copies(sim)
    rows = sorted(((c, v['lut'] * k, v['lutram'] * k) for c, v in counted.items()), key=lambda r: -r[1])
    over = [r for r in rows if r[1] > cap['lut'] or r[2] > cap['lutram']]
    with open(out('state_budget.txt'), 'w') as f:
        f.write(f"copy budget {cap['lut']:.0f} LUTs, {cap['lutram']:.0f} LUTRAM (x{cap['frac']:g} of "
                f"{cap['name']}); counted x {k}\n")
        for c, lut, lr in rows:
            f.write(f"{c:48s} {lut:10.0f} {lr:10.0f} {100 * lut / cap['lut']:6.1f} %"
                    + ('  OVER' if (c, lut, lr) in over else '') + '\n')
    largest = rows[0] if rows else ('none', 0, 0)
    print(f"foldsim: state budget: largest copy {largest[0]} {largest[1]:.0f} LUTs "
          f"({100 * largest[1] / cap['lut']:.0f} % of the budget, counted x {k})"
          + (f"; {len(over)} past it ({', '.join(r[0] for r in over)}): choose again with "
             f"--state-measured {os.path.dirname(sim)}" if over else '')
          + f"; see {out('state_budget.txt')}", file=sys.stderr)


def choose_state(a, foldsim_opt, flat, passes, out, fold):
    """--state auto: where each class's state lives (foldsim_degree.choose_state:
    on chip, paged on chip, or in HBM, priced against the V80 region, its
    routing and H), on a schedule of the fold as given. Returns the fold
    with the choice: its `page`, `page-rows`, the virtual memories it moved
    and the storage rules (pages=hbm among them); the choice is written to
    <out>/state.txt."""
    import foldsim_degree
    policy = state_policy(a, out)
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
    if foldsim_degree.STATE_CAL is not None:
        # The kernel's memory endpoints: one a m_axi_mem port, or one folded
        # (--fold-ports) priced as the calibration's folded share.
        foldsim_degree.STATE_CAL['ports'] = (foldsim_degree.STATE_CAL.get('kernel_port_folded', 1.0)
                                             if getattr(a, 'fold_ports', '') else
                                             foldsim_degree.kernel_mem_ports(text))
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
        last = foldsim_degree.LAST_REFUSAL
        over = sorted({p.partition('@')[2] for p in last.get('over', []) if '#' in p})
        if 'lut@region' in last.get('over', []):
            over.append('the region')
        if over:
            # The copies whose own logic leaves no room in their budget, with
            # every store moved.
            over = [f"{c} ({last['copies'].get(c, {}).get('total', 0):.0f} LUTs at the schedule's frame)"
                    if c in last['copies'] else c for c in over]
        sys.exit(f'--state auto: nothing fits the V80 region (II {ii})'
                 + (f" with the links' {links['lutram']:.0f} LUTRAM LUTs, {links['bram']} RAMB36, "
                    f"{links['uram']} URAM288" if links else '') + f' ({policy})'
                 + (f"; past their copy budget: {', '.join(over)}" if over else '') + f'; see {base}')
    cap = foldsim_degree.COPY_CAP
    over = (res['res'].get('copy_cap') or {}).get('over', [])
    with open(out('state.txt'), 'w') as f:
        f.write(f'II {ii} (schedule); policy: {policy}\n' + foldsim_degree.state_table(res, classes, est) + '\n')
        if cap:
            f.write(f"copies (synthesized LUTs, the budget {cap['lut']:.0f}):\n")
            for key, v in sorted(res['res']['copies'].items(), key=lambda kv: -kv[1]['total']):
                f.write(f"  {key:48s} {v['total']:9.0f} ({v['logic']:.0f} logic, {v['store_lut']:.0f} store LUTs, "
                        f"{v['store_lutram']:.0f} LUTRAM, {v['bram']} RAMB36, {v['uram']} URAM) "
                        f"{100 * v['total'] / cap['lut']:5.1f} %\n")
            if over:
                f.write('past their caps: ' + ', '.join(over) + '\n')
    # The outcome per copy and the factors it was priced with (for --state-measured).
    json.dump(dict(copies=res['res']['copies'], copy_cap=cap,
                   factors={re.sub(r'_copy\d+$', '', c): foldsim_degree.state_factors(c) for c in classes}),
              open(out('state.json'), 'w'), indent=1)
    if over:
        sys.exit(f'--state auto: copies past the per-copy budget ({policy}): {", ".join(over)}; '
                 f'see {out("state.txt")}')
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
        rules = res['storage']
        if m:
            # The given rules stay the only ones for their stores (the
            # compiler takes one rule a store).
            given = {r.rpartition('=')[0] for r in m.group(1).split(',')}
            rules = ','.join(r for r in rules.split(',') if r.rpartition('=')[0] not in given)
        if rules:
            fold = (fold.replace(m.group(0), f" storage={m.group(1)},{rules}") if m
                    else fold + f" storage={rules}")
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
        # An SLL check's sll.json of the same recipe: its copies, and its
        # simulator level with the kernel (`top`), which the floorplan puts
        # beside the level's group (--slr-level).
        level_sites = 0
        rams = {}
        if isinstance(measured.get('copies'), dict):
            top = measured.get('top', {})
            level_sites = round(top.get('lut', 0) * spl)
            # Its block and ultra RAM too (tenths of a URAM288 / RAMB36):
            # each group within its SLR's RAM at the region's margin.
            level_rams = (round(10 * top.get('uram', 0)), round(10 * top.get('bram', 0)))
            measured = measured['copies']
            rams = {c: (round(10 * m.get('uram', 0)), round(10 * m.get('bram', 0))) for c, m in measured.items()}
        for c in sites:
            if c in measured:
                sites[c] = round(measured[c]['lut'] * spl)
        print(f'foldsim: copy sites measured from {a.slr_sites} ({sum(c in measured for c in sites)} of {len(sites)})',
              file=sys.stderr)
    if a.slr_region:
        # Unequal SLRs: group g is the region's g-th SLR, each with its own
        # capacity (its LUTs, with margin, in LUT sites); the floorplan names it.
        # No boundary may carry more link bits than its SLLs, with margin.
        caps = slr_region_capacities(a.slr_region, a.slr_fill)
        # The level and the kernel beside group G (--slr-level) take their
        # room from G's SLR: measured (an sll.json as --slr-sites), else none.
        level = slr_level_group(getattr(a, 'slr_level', None), a.slr_region, len(caps))
        if a.slr_sites and level is not None and level_sites:
            n, c = caps[level]
            caps[level] = (n, max(0, c - level_sites))
            print(f'foldsim: SLR group {level} ({n}) keeps {level_sites} LUT sites for the simulator level and the '
                  f'kernel (measured, {a.slr_sites})', file=sys.stderr)
        caps_more = None
        if a.slr_sites and rams:
            rj2 = a.slr_region if a.slr_region != 'v80' else rj
            r2 = json.load(open(rj2))
            rf = a.slr_ram_fill or r2['margin']
            caps_more = [[round(10 * rf * slr['uram']), round(10 * rf * slr['ramb36'])]
                         for slr in r2['slrs']]
            if level is not None:
                caps_more[level] = [max(0, x - y) for x, y in zip(caps_more[level], level_rams)]
            print('foldsim: SLR groups\' URAM288 / RAMB36 (tenths) at most '
                  + ', '.join(f'{n} {u}/{b}' for (n, _), (u, b) in zip(caps, caps_more)), file=sys.stderr)
        print(f'foldsim: {len(caps)} SLR groups of at most '
              + ', '.join(f'{n} {c}' for n, c in caps) + ' LUT sites; copies: '
              + ', '.join(f'{c} {s}' for c, s in sites.items()), file=sys.stderr)
        if caps_more:
            opts = (f' slr-groups={len(caps)} slr-capacities='
                    + ','.join(f'{c}/{u}/{b}' for (_, c), (u, b) in zip(caps, caps_more)) + ' slr-sites='
                    + ','.join(f'{c}:{s}/{rams.get(c, (0, 0))[0]}/{rams.get(c, (0, 0))[1]}' for c, s in sites.items()))
        else:
            opts = (f' slr-groups={len(caps)} slr-capacities={",".join(str(c) for _, c in caps)} slr-sites='
                    + ','.join(f'{c}:{s}' for c, s in sites.items()))
        opts += (''
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


def slr_region_capacities(path, fill=None):
    """[(SLR name, LUT sites)] of a dynamic region's SLRs (--slr-region): each
    SLR's LUTs times the margin a group may fill (the SLR's own `margin` if it
    has one, e.g. an SLR whose copies route only at a lower fill, else the
    region's), in foldsim_degree's LUT sites (sites_per_lut, calibrated on a
    placed build)."""
    if path == 'v80':
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')
    region = json.load(open(path))
    return [(slr['name'], round((fill if fill else slr.get('margin', region['margin'])) * slr['luts']
                                * region['sites_per_lut']))
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


def scratch_as_built(a, foldsim_opt, sim_input, pipeline, sim, out):
    """The state scratch as built (foldsim_scratch_built.py): the netlist's
    block and ultra RAM by kind, the scratch's and the others', against the
    planner's (its state_scratch.json). Where the netlist is past a kind's
    fill, or the planner missed a kind's built RAM by more than 5 % of its
    device, the fold runs once more with the others as built
    (state-scratch-built-others), so that its planner subtracts them and not
    an estimate (no reserve). The table, both times,
    in <out>/state_scratch.txt. Returns the pipeline built last."""
    import foldsim_scratch_built as sb
    lines = []

    def look(tag):
        b = sb.built(sim)
        p = json.load(open(out('state_scratch.json')))
        rows = sb.table(b, p)
        lines.append(f'{tag}:')
        lines.extend(rows)
        for row in rows:
            print(f'foldsim: {row}', file=sys.stderr)
        return b, p
    b, p = look('first fold')
    # Refolded where the netlist is past a kind's fill, or the planner missed
    # a kind's built RAM by more than 5 % of its device (either way), so that
    # the planner's figures are the netlist's.
    built_bits = {k: (b['scratch'][k] + b['other'][k]) * bits
                  for k, bits in (('uram', sb.URAM_BITS), ('bram', sb.BRAM_BITS))}
    over = [k for k in built_bits if built_bits[k] > p[k]['device']]
    room = [k for k in built_bits if abs(p[k]['planned'] - built_bits[k]) > 0.05 * p[k]['device']]
    miss = over + [k for k in room if k not in over]
    if miss and not getattr(a, 'state_scratch_no_rebuild', False) and not p.get('as_built'):
        # The others as built, and what the planner's own pricing missed of
        # what it placed (its rows built larger than planned: more columns as
        # packed), so that the refold leaves room for it.
        eff = {}
        for k, bits in (('uram', sb.URAM_BITS), ('bram', sb.BRAM_BITS)):
            short = max(0, b['scratch'][k] * bits - p.get('placed', {}).get(k, b['scratch'][k] * bits))
            eff[k] = int(b['other'][k] * bits + short)
        others = f"{eff['uram']},{eff['bram']}"
        why = ('past the fill in ' + ', '.join(over)) if over else \
            ('the planner missed the built ' + ', '.join(room) + ' by more than 5 % of the device')
        print(f'foldsim: state scratch: {why}: refolding with the others as built ({others} bits)',
              file=sys.stderr)
        pipeline = [re.sub(r'(foldsim-fold-static\{[^}]*)\}',
                           lambda m: m.group(1) + f' state-scratch-built-others={others}}}', step, count=1)
                    if step.startswith('foldsim-fold-static{') else step for step in pipeline]
        run([foldsim_opt, sim_input, NO_OP_DUMP, f'--pass-pipeline=builtin.module({",".join(pipeline)})',
             '-o', sim])
        look('as built (refolded)')
    with open(out('state_scratch.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return pipeline


def link_kinds_as_built(a, foldsim_opt, sim_input, pipeline, sim):
    """--link-kind-cost without --link-kind-others: the fold runs once more
    with what the rest of the simulator took as built (foldsim_link_kinds.py:
    its URAM288, RAMB36 and logic and LUTRAM LUTs less its links'), so that
    links are priced against the room left, not the whole device. Returns the
    pipeline built last."""
    import foldsim_link_kinds as lk
    (u, b, l), links, total = lk.others(sim)
    others = f'{u:.0f},{b:.0f},{l:.0f}'
    print(f'foldsim: link-kind-cost: the rest as built {others} (links {links["uram"]:.0f} URAM288, '
          f'{links["bram"]:.1f} RAMB36, {links["lutram"]:,.0f} LUTRAM LUTs): refolding with link-kind-others',
          file=sys.stderr)
    add = f' link-kind-others={others}'
    # With SLR groups: each group's rest too (its own module, foldsim_sim_slr<g>),
    # the budgets the plan places each group's links against.
    if getattr(a, 'slr_groups', False) and not getattr(a, 'link_kind_group_others', ''):
        gs = lk.groups(sim)
        if gs:
            group_others = ';'.join(f'{x["others"][0]:.0f},{x["others"][1]:.0f},{x["others"][2]:.0f}'
                                    for x in gs)
            print(f'foldsim: link-kind-cost: the rest as built per group {group_others}', file=sys.stderr)
            add += f' link-kind-group-others={group_others}'
    pipeline = [re.sub(r'(foldsim-fold-static\{[^}]*)\}',
                       lambda m: m.group(1) + add + '}', step, count=1)
                if step.startswith('foldsim-fold-static{') else step for step in pipeline]
    run([foldsim_opt, sim_input, NO_OP_DUMP, f'--pass-pipeline=builtin.module({",".join(pipeline)})',
         '-o', sim])
    return pipeline


def region_ram_bits(path, target=None):
    """A region's ultra and block RAM bits at a fill target (`target`, a
    share of each kind: --state-scratch-fill), never past the fit check's own
    limits (foldsim_sll_check.py: the region JSON's placement `fill` of uram
    and bram); (0, 0) without a region."""
    if not path:
        return 0, 0
    region = json.load(open(path))
    fill = region.get('placement', {}).get('fill', {})
    fu, fb = fill.get('uram', 1.0), fill.get('bram', 1.0)
    if target is not None:
        fu, fb = min(fu, target), min(fb, target)
    uram = sum(s.get('uram', 0) for s in region.get('slrs', [])) * 288 * 1024
    bram = sum(s.get('ramb36', 0) for s in region.get('slrs', [])) * 36 * 1024
    return int(uram * fu), int(bram * fb)


def region_kind_capacity(path, target=0.85):
    """A region's URAM288, RAMB36 and LUTs at a fill target (--link-kind-cost's
    budget), never past the fit check's own limits; the V80's without a region."""
    region = json.load(open(path or v80_region()))
    fill = region.get('placement', {}).get('fill', {})
    total = lambda k: sum(s.get(k, 0) for s in region.get('slrs', []))
    return (int(total('uram') * min(fill.get('uram', 1.0), target)),
            int(total('ramb36') * min(fill.get('bram', 1.0), target)),
            int(total('luts') * min(region.get('placement', {}).get('region_fill', {}).get('lut', 1.0), target)))


def region_group_capacity(path, target=0.85):
    """Each SLR's URAM288, RAMB36 and LUTs at a fill target, in the region's
    order (SLR group g is its g-th SLR)."""
    region = json.load(open(v80_region() if path in ('', 'v80', 'auto') else path))
    fill = region.get('placement', {}).get('fill', {})
    return [(int(s.get('uram', 0) * min(fill.get('uram', 1.0), target)),
             int(s.get('ramb36', 0) * min(fill.get('bram', 1.0), target)),
             int(s.get('luts', 0) * min(fill.get('lut', 1.0), target)))
            for s in region.get('slrs', [])]


def v80_region():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')


def check_region(a):
    """The region of the SLR rule (--region): a JSON path, or '' when none
    applies (no --static, or a manifest device other than the V80's)."""
    if not a.static:
        return ''
    r = a.region
    if r in ('', 'auto'):
        r = a.slr_region
    if r in ('', 'auto'):
        device = foldsim_manifest.get(a.manifest, 'platform.device', a.manifest_var) if a.manifest else None
        if device and not fnmatch.fnmatch(str(device), 'xcv80-*'):
            return ''
        r = 'v80'
    return v80_region() if r == 'v80' else r


# The SLR check's physical refusals (guide policy: they fail): the design past
# the placement fill, the memories past the region's, SLLs past a boundary, no
# assignment even with copies cut.
PHYSICAL = re.compile(r'past the placement fill|past the region\'s at the placement fill|SLLs projected over .* past '
                      r'its|no assignment of the copies within|does not fit the room the other copies leave')
STRADDLE = re.compile(r'fits no SLR whole|no assignment of the copies|does not fit the room the other copies leave')


def slr_guide(a, out, verdict):
    """--slr-policy guide (default): a class copy crossing an SLR boundary is a
    guide, not a failure: a warning naming each straddling copy, its resources,
    the SLRs it spreads over, the projected SLLs per boundary and the suggested
    assignment of the others (<out>/sll/sll.json). foldsim_sim.sv is kept."""
    j = json.load(open(out('sll/sll.json'))) if os.path.exists(out('sll/sll.json')) else {}
    copies, caps, cuts = j.get('copies', {}), j.get('caps', []), list(j.get('cuts') or [])
    if a.allow_slr_straddle:
        print('foldsim: warning: --allow-slr-straddle is no longer needed (--slr-policy guide: a straddle is a '
              'guide, not a failure); ignored', file=sys.stderr)
    if not cuts:
        return
    res = ('lut', 'lutram', 'ff', 'uram', 'bram')
    fmt = lambda r, x: f'{x:,.1f}' if r in ('uram', 'bram') else f'{x:,.0f}'
    names = [s['name'] for s in caps]
    lines = [f'foldsim: GUIDE: SLR straddle (a guide, --slr-policy guide): {verdict}']
    for c in cuts:
        sh = (j.get('shares') or {}).get(c) or {}
        spread = ''
        if sh.get('lut'):
            spread = ' over ' + '/'.join(f'{n} {100 * x:.0f} %' for n, x in zip(names, sh['lut']) if x > 0.005)
        lines.append(f'  GUIDE: {c} straddles{spread}: ' + ', '.join(f'{r} {fmt(r, copies[c][r])}' for r in res
                                                                    if c in copies))
    for b, sl in enumerate(j.get('slls') or []):
        if b + 1 < len(names):
            lines.append(f'  GUIDE: boundary {names[b]}-{names[b + 1]}: {sum(sl):,.0f} SLLs projected '
                         f'(links {sl[0]:,.0f}, cut copies {sl[1]:,.0f}, simulator level and kernel {sl[2]:,.0f})')
    for g, cs in enumerate(j.get('assignment') or []):
        if g < len(names):
            lines.append(f'  GUIDE: suggested {names[g]}: ' + ' '.join(x for x in cs if x not in cuts))
    lines.append(f'  report: {out("sll")}/sll.txt')
    print('\n'.join(lines), file=sys.stderr)


def slr_rule(a, out, verdict, rc):
    """--slr-policy strict, the hard SLR rule on the check's verdict and JSON
    (<out>/sll/sll.json): no class copy may cross an SLR boundary. Exits with an
    error naming each straddling copy, its resources and each SLR's capacity,
    unless --allow-slr-straddle gives the reason."""
    if rc == 2 or verdict is None:
        sys.exit('foldsim: error: SLR rule: the SLR check failed, so no copy is proven to fit an SLR whole '
                 f'(report {out("sll")}); --allow-slr-straddle does not cover a failed check')
    j = json.load(open(out('sll/sll.json'))) if os.path.exists(out('sll/sll.json')) else {}
    copies, caps = j.get('copies', {}), j.get('caps', [])
    over = list(j.get('over', []))
    cuts = [c for c in j.get('cuts', []) if c not in over]
    if not (over or cuts or STRADDLE.search(verdict)):
        return
    res = ('lut', 'lutram', 'ff', 'uram', 'bram')
    fmt = lambda r, x: f'{x:,.1f}' if r in ('uram', 'bram') else f'{x:,.0f}'
    lines = [f'SLR rule: no class copy may cross an SLR boundary; this netlist: {verdict}']
    for c in over + cuts:
        why = 'fits no SLR whole' if c in over else 'is left out of every assignment of whole copies to the SLRs'
        sh = (j.get('shares') or {}).get(c)
        if sh and c in cuts:
            # The resource it is cut most in.
            r = max((r for r in res if r in sh), key=lambda r: 1 - max(sh[r]))
            why += (' (the check spreads it: ' + ', '.join(f'{s["name"]} {100 * x:.0f} %' for s, x in zip(caps, sh[r]))
                    + f' of its {r})')
        lines.append(f'  {c} {why}: ' + ', '.join(f'{r} {fmt(r, copies[c][r])}' for r in res if c in copies))
    if caps:
        lines.append(f'  SLR capacities (logic x margin {j.get("margin", 1.0)}, memories whole):')
        lines += [f'    {s["name"]}: ' + ', '.join(f'{r} {fmt(r, s[r])}' for r in res) for s in caps]
    lines.append(f'  report: {out("sll")}/sll.txt')
    if a.allow_slr_straddle:
        import getpass
        import time
        note = dict(reason=a.allow_slr_straddle, time=time.strftime('%F %T'), user=getpass.getuser(),
                    straddle=over + cuts, verdict=verdict)
        json.dump(note, open(out('SLR_STRADDLE_APPROVED'), 'w'), indent=1)
        info = json.load(open(out('foldsim_sim.json')))
        info['slr_straddle_approved'] = note
        with open(out('foldsim_sim.json'), 'w') as f:
            json.dump(info, f, indent=1)
        bar = '!' * 78
        print('\n'.join([bar, f'foldsim: WARNING: SLR STRADDLE APPROVED: {a.allow_slr_straddle}'] + lines
                        + [f'  recorded in {out("SLR_STRADDLE_APPROVED")} and foldsim_sim.json', bar]),
              file=sys.stderr)
        return
    if os.path.exists(out('foldsim_sim.sv')):
        os.replace(out('foldsim_sim.sv'), out('foldsim_sim.sv.slr-refused'))
    sys.exit('\n'.join(['foldsim: error: ' + lines[0]] + lines[1:]
                       + ['  foldsim_sim.sv renamed foldsim_sim.sv.slr-refused; change the recipe (copies, paging, '
                          'storage) so every copy fits an SLR whole, or override with '
                          '--allow-slr-straddle=<who, when, why>']))


def slr_split(a, foldsim_opt, sim, out):
    """--slr-split: each matching copy's <copy>_static module split into parts at its registers (the partition by
    foldsim_slr_split.py --apply-map, the split by the foldsim-slr-split pass, the simulator rewritten in place).
    [(instance name, SLR)] of the parts."""
    import fnmatch
    text = open(sim).read()
    statics = re.findall(r'hw\.module(?: private)? @(\S+?)_static\(', text)
    plan = []
    for spec in a.slr_split:
        f = spec.split(':')
        if len(f) < 2 or not f[1].isdigit() or int(f[1]) < 2:
            sys.exit(f'--slr-split: {spec!r} is not GLOB:PARTS[:SLR,...] with PARTS >= 2')
        glob, k = f[0], int(f[1])
        slrs = f[2].split(',') if len(f) > 2 and f[2] else ['SLR2', 'SLR1', 'SLR0']
        if len(slrs) < k:
            sys.exit(f'--slr-split: {spec!r} names fewer SLRs than parts')
        copies = [c for c in statics if fnmatch.fnmatchcase(c, glob)]
        if not copies:
            sys.exit(f'--slr-split: no class copy matches {glob!r}')
        for c in copies:
            mp = out(f'slr_split_{c}.map')
            run([a.slr_split_python, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'foldsim_slr_split.py'), sim, '--copy', c, '--parts', str(k),
                 '--slrs', ','.join(slrs[:k]), '--chunk', '0', '--apply-map', mp, '--out', out('slr_split')]
                + shlex.split(a.slr_split_options.replace('+', ' ')))
            run([foldsim_opt, sim, NO_OP_DUMP, '--pass-pipeline=builtin.module(foldsim-slr-split{'
                 f'module={c}_static map={mp} parts={k} {a.slr_split_pass.replace(",", " ")}}})', '-o', sim + '.split'])
            os.replace(sim + '.split', sim)
            plan += [(f'fs_split_{c}_static_p{p}', slrs[p]) for p in range(k)]
    return plan


def write_slr_split(plan, path):
    """--slr-split: each part of a split copy assigned to its SLR (the flow holds it there with a pblock with
    FOLDSIM_SLR_PBLOCKS=1)."""
    with open(path, 'a' if os.path.exists(path) else 'w') as f:
        if f.tell() == 0:
            f.write('# Generated by foldsim. Do not edit.\n')
        for inst, slr in plan:
            f.write(f'set c [get_cells -quiet -hierarchical {inst}]\n'
                    f'if {{[llength $c] == 1}} {{ set_property USER_SLR_ASSIGNMENT {slr} $c }}\n')


def slr_level_group(spec, region, groups):
    """--slr-level: the SLR group (index) the simulator level goes with. auto: the
    group of the region's SLR with the most HBM NoC ports (its noc section: the
    virtual-memory controllers' masters and the kernel meet the memory there), else
    group 0; or a group index; none: the level left to the placer."""
    if spec in (None, '', 'none'):
        return None
    if spec != 'auto':
        g = int(spec)
        if not 0 <= g < groups:
            sys.exit(f'--slr-level: group {g} of {groups}')
        return g
    if region:
        path = region if region != 'v80' else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')
        r = json.load(open(path))
        order = [slr['name'] for slr in r['slrs']]
        hbm = collections.Counter(p['slr'] for p in r.get('noc', {}).get('ports', [])
                                  if 'HBM' in p.get('targets', []))
        if hbm:
            best = max(order, key=lambda n: (hbm.get(n, 0), -order.index(n)))
            if order.index(best) < groups:
                return order.index(best)
    return 0


def write_slr_level(path, group, names=None):
    """--slr-level: appends to the floorplan the simulator level -- the simulator's
    cells outside its SLR groups: the frame decision, the token channels'
    queues, the readiness registers -- and the environment beside it (the
    simulator's sibling `env`, the shell's token engine its channels meet) in
    group `group`'s SLR. Left to the placer, the level lands where room is
    (FSD 1,024 nodes, 11-12: SLR0, its groups in SLR2/SLR1, the env split over
    both): every control and token path then crosses the device."""
    where = names[group] if names else f'fs_slr{group}'
    with open(path, 'a') as f:
        f.write(f'# The simulator level and its environment with group {group} (--slr-level).\n'
                f'set g [get_cells -quiet -hierarchical foldsim_slr{group}]\n'
                f'if {{[llength $g] == 1}} {{\n'
                f'  set sim [get_property PARENT $g]\n'
                f'  set c [get_cells -quiet -filter {{NAME !~ "*/foldsim_slr*"}} $sim/*]\n'
                f'  if {{[llength $c] > 0}} {{ set_property USER_SLR_ASSIGNMENT {where} $c }}\n'
                f'  set e [get_cells -quiet [get_property PARENT [get_cells $sim]]/env]\n'
                f'  if {{[llength $e] == 1}} {{ set_property USER_SLR_ASSIGNMENT {where} $e }}\n'
                f'}}\n')


def write_slr_memory_ports(path, region, names, groups, level):
    """<out>/foldsim_sp.txt: each SLR group's HBM ports -- the region's NoC ports
    to HBM in the group's SLR, by their linker tag (the region JSON's noc ports
    with a `tag`, e.g. the asl9 shell's MEM_SLR0_0) -- and the level's (--slr-level:
    the simulator level and its environment, with the kernel's memory endpoints
    beside them). A region's device-memory masters bind to its own SLR's ports
    (`sp=<kernel>.<master>:<tag>`): no memory traffic crosses an SLR boundary on
    SLL wires. Not written when the region names no tags."""
    rpath = region if region != 'v80' else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')
    ports = [p for p in json.load(open(rpath)).get('noc', {}).get('ports', [])
             if 'HBM' in p.get('targets', []) and p.get('tag')]
    if not ports:
        return
    tags = {}
    for p in ports:
        tags.setdefault(p['slr'], [])
        if p['tag'] not in tags[p['slr']]:
            tags[p['slr']].append(p['tag'])
    with open(path, 'w') as f:
        f.write('# Generated by foldsim: the HBM ports (linker tags) of each SLR group\'s SLR; '
                'a group\'s memory masters bind to its own SLR\'s ports.\n')
        for g in range(groups):
            f.write(f'group {g} {names[g]} {" ".join(tags.get(names[g], [])) or "-"}\n')
        if level is not None:
            f.write(f'level {names[level]} {" ".join(tags.get(names[level], [])) or "-"}\n')


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


# An AXI4 device-memory master's wires (512-bit data, 64-bit address, 8-bit
# IDs) where the region gives none (its noc section's master_bits).
VMEM_MASTER_BITS = 1290


def vmem_engine_groups(sv):
    """{virtual memory k: its SLR group's instance (foldsim_slr<n>)}: the group
    whose ports carry memory k's (foldsim_sim's vm<k>_* wired to it)."""
    body = sv[sv.index('module foldsim_sim('):]
    body = body[:body.index('endmodule')]
    out = {}
    for m in re.finditer(r'^\s+\w+ (foldsim_slr\d+) \((.*?)\);', body, re.M | re.S):
        for k in re.findall(r'\(vm(\d+)_rd_valid\)', m.group(2)):
            out[int(k)] = m.group(1)
    return out


def write_vmem_engines(sv, info, path, mode, master_bits, names=None):
    """Appends to the floorplan (write_slr_groups) the virtual-memory
    controllers placed with their class copy (--vmem-engines local, or auto
    where the ports are wider than the controller's device-memory master):
    each in its copy's SLR group, so that only its master crosses to the
    memory NoC ports rather than its ports (foldsim_sll_check.py
    vmem_slls). The kernel's instances: foldsim_afu's vmem_<k> (a memory's
    own controller) or g_vg<g>.vmem / g_vg<g>.group (group g's). Returns
    [(controller, group)]."""
    if mode not in ('local', 'auto') or not info.get('vmem'):
        return []
    at = vmem_engine_groups(sv)
    bits = lambda v: 2 * v['data_width'] + v['row_width'] + v['inst_width']
    if info.get('vmem_groups'):
        ctrls = [(f'.*/g_vg{g}[./](g_vp[./])?(vmem|group|vport)$', x['members'],
                  sum(bits(info['vmem'][k]) for k in x['members']))
                 for g, x in enumerate(info['vmem_groups'])]
    else:
        ctrls = [(f'.*/(vmem|vport)_{k}$', [k], bits(v)) for k, v in enumerate(info['vmem'])]
    placed = []
    with open(path, 'a') as f:
        f.write(f'# Virtual-memory controllers with their class copy (--vmem-engines {mode}).\n')
        for cell, members, port_bits in ctrls:
            port = info['vmem'][members[0]].get('port', f'vm{members[0]}')
            group = at.get(int(port[len('vm'):]))
            if group is None or (mode == 'auto' and port_bits <= master_bits):
                continue
            where = names[int(group[len('foldsim_slr'):])] if names else f'fs_{group[len("foldsim_"):]}'
            f.write(f'set c [get_cells -quiet -hierarchical -regexp {{{cell}}}]\n'
                    f'if {{[llength $c] == 1}} {{ set_property USER_SLR_ASSIGNMENT {where} $c }}\n')
            placed.append((cell, group))
    return placed


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


def vmem_slr_map(spec):
    """[(class, group)] of --vmem-slr-groups: a JSON file whose `assignment`
    lists the classes of each SLR group in order (foldsim_sll_check.py
    --json), or `<class>:<group>,...`."""
    if os.path.isfile(spec):
        groups = json.load(open(spec)).get('assignment') or []
        return [(c, g) for g, names in enumerate(groups) for c in names]
    out = []
    for x in spec.split(','):
        c, _, g = x.rpartition(':')
        if not c or not g.isdigit():
            sys.exit(f'--vmem-slr-groups: {x!r} is not <class>:<group>')
        out.append((c, int(g)))
    return out


def page_engine_slr(sched, pairs):
    """{`__pages<k>`: SLR group} of the schedule's page engines under --vmem-slr-groups (`pairs`): the group most of
    its members name (see vmem_slr_of); a static port vp<k> of vport-engines is page engine k."""
    of = dict(pairs)
    engines = {}
    for m in re.finditer(r'^\s+page engine (\d+) \d+ bits, tag \d+, row \d+: (.*)$', sched, re.M):
        votes = {}
        for x in m.group(2).split():
            cls = x[5:].split('>', 1)[-1].split('(')[0] if x.startswith('link:') else x.split(':', 1)[0]
            if cls in of:
                votes[of[cls]] = votes.get(of[cls], 0) + 1
        if len(votes) > 1:
            print(f'foldsim: warning: --vmem-slr-groups: page engine {m.group(1)} carries the members of groups '
                  f'{sorted(votes)}', file=sys.stderr)
        engines[f'__pages{m.group(1)}'] = max(votes, key=lambda g: (votes[g], -g)) if votes else -1
    return engines


def vmem_slr_of(info, sched, pairs):
    """Each controller's SLR group under --vmem-slr-groups (`pairs`, [(class,
    group)]), in the kernel's controller order (info['vmem_groups'], else
    info['vmem']): a class's own controller, its class's group; a page engine
    (class `__pages<k>`), its members' (the schedule's `page engine <k> ...:`
    line: a column or PP row chunk `<class>:<name>`, a link stream
    `link:<producer>><reader>` its reader's, as fold-static vmem-slr-groups
    places them), the one most of them name; -1 where none is named."""
    of = dict(pairs)
    engines = page_engine_slr(sched, pairs)
    ctrls = info.get('vmem_groups') or info.get('vmem') or []
    return [engines.get(c['class'], of.get(c['class'], -1)) for c in ctrls]


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
        # of its own, as has a memory with tagged requests (its tags are its own).
        keys.setdefault((v['class'], v['instances'], v['inst_width'], v['row_bytes'],
                         v['read_first'], k if v.get('pages') or v.get('tag_width') else -1), []).append(k)
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
                         'match below it; `nest:GLOB` a class whose own instances are searched too '
                         '(a class holding classes). A class `A+B` may name them')
    ap.add_argument('--absorb-exclusive', default='', metavar='CLASS,...',
                    help="for these classes (globs), a model input whose logic in the parent is "
                         "over the absorb limit takes the part of it that reaches that input "
                         "alone, the rest as inputs: a per-port select in front of a per-port "
                         "class is then one copy in the class (foldsim-extract-models "
                         "absorb-exclusive)")
    ap.add_argument('--direct-outputs', action='store_true',
                    help='an output of a module that only wires out its classes\' outputs, its inputs and constants (a '
                         'port bundle of a child passed straight out of the top) is wired so in the netlist, not passed '
                         'through the module\'s own model (foldsim-extract-models direct-outputs)')
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
    ap.add_argument('--affine-auto-bits', type=int, default=None, metavar='BITS',
                    help='with --affine-auto: only classes whose undecoded slot rows hold at least BITS bits '
                         '(foldsim-fold-static affine-auto-bits, default 262144)')
    ap.add_argument('--affine-auto-segments', type=int, default=None, metavar='K',
                    help='with --affine-auto: classes whose own slot rows need more than K segments are made '
                         'affine (foldsim-fold-static affine-auto-segments; default: --slot-segments)')
    ap.add_argument('--affine-stride', choices=['pow2', 'exact'], default=None,
                    help='with --affine-slots / --affine-auto: each group\'s stride rounded up to a power of two '
                         '(pow2, the default) or exact (its resource\'s group count or its spacing; runs of '
                         'other strides decode with the position counters) (foldsim-fold-static affine-stride)')
    ap.add_argument('--affine-auto', action='store_true',
                    help='with --static --modulo --slot-segments: only the classes whose slot rows do not decode '
                         'placed affine, the frame scheduled again (foldsim-fold-static affine-auto)')
    ap.add_argument('--affine-slots', action='store_true',
                    help='with --static --modulo: each copy\'s passes of a kind placed as one affine run over its '
                         'instances (foldsim-fold-static affine-slots): slot tables of a segment a pass')
    ap.add_argument('--slot-table-compose', action='store_true',
                    help='with --slot-segments: a table read at a position table\'s value that does not decode is '
                         'read through the inner table\'s position (composed rows) (foldsim-fold-static '
                         'slot-table-compose)')
    ap.add_argument('--pp-sync-banks', type=int, default=0, metavar='ROWS',
                    help='with --state-in-ram-pp: a combinational read of a memory of more than ROWS threaded rows '
                         'is banked and registered from its registers\' next values (same data; Beacon) '
                         '(foldsim-fold-static pp-sync-banks)')
    ap.add_argument('--slot-ahead', type=int, default=0, metavar='K',
                    help='with --slot-segments: position tables read through a class\'s pass sequence, and '
                         'position tables decoded in K segments or more, from registers beside the position '
                         'registers loaded a host cycle ahead (same values; Beacon) (foldsim-fold-static '
                         'slot-ahead)')
    ap.add_argument('--slot-column-affine', action='store_true',
                    help='with --slot-segments: instance tables the segments do not decode cut into column-affine '
                         'pieces (a period\'s rows plus a few slopes times the quotient, by a constant division) '
                         '(foldsim-fold-static slot-column-affine)')
    ap.add_argument('--slot-segments', type=int, default=0, metavar='K',
                    help='with --static: slot tables of at most K affine segments as decoders of the position or '
                         'instance (a size independent of II and N), not constant arrays (foldsim-fold-static '
                         'slot-segments)')
    ap.add_argument('--csf-ready-copies', default='', metavar='MODEL:K,...',
                    help='with --static and --csf-ready-cones: each model\'s ready cone (its late outputs of '
                         'logic alone) as a model of its own, MODEL_rc, of K copies (the ready lanes) '
                         '(foldsim-fold-static csf-ready-copies)')
    ap.add_argument('--csf-ready-cones', action='store_true',
                    help='with --csf-ready-copies: build the lanes (foldsim-fold-static csf-ready-cones); '
                         'without it csf-ready-copies is schedule only')
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
    ap.add_argument('--vmem-auto-instance-bits', type=int, default=0, metavar='N',
                    help='as --vmem-auto-bits, by one instance\'s image (rows x width) at least N bits: the same '
                         'memories are virtual at every instance count (foldsim-fold-static vmem-auto-instance-bits)')
    ap.add_argument('--vmem-cost', action='store_true',
                    help='a memory the virtual-memory rules pick (pattern or size) stays on chip where its '
                         'threaded image there costs fewer LUTs than as a virtual memory (foldsim-fold-static '
                         'vmem-cost; each choice reported)')
    ap.add_argument('--vmem-auto-bits', type=int, default=0, metavar='N',
                    help='with --static: every registered-read target memory (one read port, at most one write '
                         'port) whose threaded image is at least N bits is a virtual memory too, whatever storage '
                         'rule names it (foldsim-fold-static vmem-auto-bits; 0: none by size)')
    ap.add_argument('--fold-store-rings', type=int, default=0, metavar='LEAD',
                    help="with --fold-store-stream and without --vmem-rings: the streamed folded stores' read "
                         "buffers alone as rings of this lead (a fixed ring and table of reads in flight, not a "
                         "buffer row and pending bit per instance; foldsim-fold-static fold-store-rings; 0: none)")
    ap.add_argument('--vmem-rings', type=int, default=0, metavar='LEAD',
                    help="with virtual memories: each one's per-instance read buffer as a fixed ring, refilled "
                         "from an image of the answers in HBM LEAD positions before each instance's first use "
                         "(foldsim-fold-static vmem-rings; 0: none)")
    ap.add_argument('--vmem-ring-entries', type=int, default=1024, metavar='N',
                    help='with --vmem-rings: the most entries a ring may take (default 1024)')
    ap.add_argument('--vmem-ring-flight', type=int, default=32, metavar='N',
                    help='with --vmem-rings: entries of each ring\'s reads-in-flight table and answer queue (a power of two '
                         '>= 8, default 32); 0: per ring by its read rate, 8 to 32 (foldsim-fold-static vmem-ring-flight)')
    ap.add_argument('--vmem-ring-class-control', action='store_true',
                    help='with --vmem-rings: a class\'s rings whose reads issue together share one reads-in-flight '
                         'table (a bit per ring an entry) and one prefetch queue, in place of one of each per ring '
                         '(foldsim-fold-static vmem-ring-class-control)')
    ap.add_argument('--vmem-ring-image-each', action='store_true',
                    help='with --vmem-rings: each ring its own image and controller (default: a class\'s rings '
                         'share one image, the ring in the tag: one controller a class; foldsim-fold-static '
                         'vmem-ring-image-each)')
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
    ap.add_argument('--localize-read-addresses', choices=('hinted', 'auto'), default='hinted',
                    help='copy a memory read\'s address registers out of a kept class instance (foldsim-localize-'
                         'read-addresses) for memories that ask for block RAM (hinted, the default), or also for '
                         'the unhinted ones --retime-memory-reads converts (auto)')
    ap.add_argument('--retime-memory-reads', metavar='MIN_BITS', type=int,
                    help="memories of at least MIN_BITS bits without a ram_style hint, read "
                         'asynchronously at an address that is a function of registers alone (a '
                         "queue's head, a buffer's tail), read through a register instead: a target "
                         'cycle early, at the address\'s next value, reading new data (circt-opt '
                         'seq-retime-firmem-reads mode=auto). Exact; such a memory may then be '
                         'virtual (--virtual-memories) or in block RAM')
    ap.add_argument('--narrow-shifts', type=int, default=17, metavar='MIN_WIDTH',
                    help='before folding: a shift of at least MIN_WIDTH bits by a non-constant amount takes the '
                         "amount's low ceil(log2 width) bits and an OR of the rest (out of range), so a pass "
                         'split carries those bits, not the whole amount (foldsim-narrow-shift-amounts; exact; '
                         '0: none)')
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
    ap.add_argument('--vmem-controller', choices=('vmem', 'vport'), default='vmem',
                    help="the kernel's controller of each virtual memory (group, page engine): foldsim_vmem "
                         "(default), or vport: a static port in its place (shell foldsim_vport_ctl: one AXI id, "
                         "in-order, counter fences instead of hazard tables and write buffers, a zero fill of "
                         "the image after a reset instead of known-zero bits; FOLDSIM_VMEM_VPORT and each "
                         "image's bytes in foldsim_sim.vh)")
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
    ap.add_argument('--zero-chain-levels', type=int, default=0, metavar='L',
                    help='with --zero-latency-chains: the most estimated levels a chain of zero-latency links may '
                         'reach into a reader\'s state; the first link past it is registered (foldsim-fold-static '
                         'zero-chain-levels)')
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
    ap.add_argument('--pp-beat-bits', type=int, default=0, metavar='N',
                    help='with --page-engines-pp: streamed PP rows as beats of at most N bits, '
                         'drained from a row buffer (foldsim-fold-static pp-beat-bits)')
    ap.add_argument('--vmem-stagger', type=int, default=0, metavar='D',
                    help='with --vmem-on-engines: a target RAM\'s requests may leave up to D positions after its '
                         'pass (held in registers) so a copy\'s memories share engines at successive positions '
                         '(foldsim-fold-static vmem-stagger; 0: at the pass)')
    ap.add_argument('--vport-engines', action='store_true',
                    help='with --page-engines: each page engine a static port (foldsim_vport, vp<p>_* simulator '
                         'ports, FOLDSIM_VPORT<p>_* in foldsim_sim.vh; foldsim-fold-static vport-engines)')
    ap.add_argument('--vport-copy', action='store_true',
                    help='with --vport-engines and --vmem-rings: ring answers copied into their images by the port '
                         '(foldsim-fold-static vport-copy)')
    ap.add_argument('--vmem-fold-on-engines', action='store_true',
                    help='with --vmem-on-engines and --fold-store-stream: streamed folded stores on page engines '
                         '(their issues\' positions) instead of controllers of their own (foldsim-fold-static '
                         'vmem-fold-on-engines)')
    ap.add_argument('--vmem-slr-groups', default='', metavar='JSON|SPEC',
                    help='with --page-engines: the SLR group of each class, so that a page engine carries one '
                         'group\'s members only (foldsim-fold-static vmem-slr-groups): a JSON whose `assignment` '
                         'lists the classes of each SLR in the region\'s order (foldsim_sll_check.py --json), or '
                         '<class>:<group>,...')
    ap.add_argument('--vport-write-beats', action='store_true',
                    help='with --vport-engines, --vmem-on-engines, --vmem-stagger and --vport-hz: target RAM rows '
                         'wider than a 512-bit beat written beat by beat at successive positions (one-beat port '
                         'write rows; foldsim-fold-static vport-write-beats)')
    ap.add_argument('--vport-share', type=int, default=0, metavar='PCT',
                    help='with --vport-engines and --vmem-on-engines: a target RAM on the port (positions free, its '
                         'beats within PCT %% of a frame) whose row grows least, not the first of wide-enough rows '
                         '(foldsim-fold-static vport-share; 0: first fit)')
    ap.add_argument('--vport-hz', type=int, default=0, metavar='N',
                    help='with --vport-engines: read after write by a 2^N-entry table of write numbers a port '
                         '(by a hash of the row) instead of fence classes (0)')
    ap.add_argument('--vport-fence-shift', type=int, default=4, metavar='FS',
                    help='with --vport-engines: the ports\' fence sample interval\'s log2 (0: every read waits for '
                         'every earlier write)')
    ap.add_argument('--port-plan-out', metavar='JSON',
                    help='report only: every HBM member\'s request positions (page engine members as placed, class '
                         'virtual memories\' state passes and issues) in this JSON file, a static-port placement '
                         'model\'s input (foldsim-fold-static port-plan-out)')
    ap.add_argument('--vmem-on-engines', action='store_true',
                    help='with --page-engines and --virtual-memories: the target RAMs ride the engines (static '
                         'ports) at their own positions, no controller or member queue per memory '
                         '(foldsim-fold-static vmem-on-engines)')
    ap.add_argument('--page-engines-pp', action='store_true',
                    help='with --page-engines and --state-in-ram-pp streamed rows: PP row chunks ride the engines '
                         '(foldsim-fold-static page-engines-pp)')
    ap.add_argument('--link-hbm-beat-bits', type=int, default=0, metavar='N',
                    help='with --static and hbm link kinds: a stream row wider than N bits moves as beats of '
                         'a controller N bits wide where the positions allow (foldsim-fold-static '
                         'link-hbm-beat-bits; 0: whole rows)')
    ap.add_argument('--link-landing-block', type=int, default=0, metavar='R',
                    help='HBM link streams\' landing memories in block RAM (a registered read a host cycle '
                         'ahead) where their distributed RAM costs at least R LUTs a RAMB36 (foldsim-fold-static '
                         'link-landing-block)')
    ap.add_argument('--link-hbm-beat-rows', action='store_true',
                    help='with --link-hbm-beat-bits: a stream of beats lands in rows of a beat (a landing '
                         'memory slots x beats deep and a beat wide), read beat by beat into its members\' '
                         'registers (foldsim-fold-static link-hbm-beat-rows)')
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
    ap.add_argument('--fold-store-hbm', type=int, default=0, metavar='N',
                    help='with --static: folded pipelines\' stores of at least N bits (depth 2 or more) kept in HBM '
                         'as virtual memories, a row read a frame ahead and written from a landing buffer at each '
                         'state pass (foldsim-fold-static fold-store-hbm; a storage rule <model>:fold<k>_store=hbm '
                         'names one, another kind keeps one on chip); 0: none by size')
    ap.add_argument('--fold-store-stream', action='store_true',
                    help="with --static: every folded pipeline's store of depth 2 or more streams from HBM "
                         '(not the state scratch), its writes and reads from a landing ring of a fixed number of '
                         'entries, the reads answered into the vmem ring with --vmem-rings (foldsim-fold-static '
                         'fold-store-stream)')
    ap.add_argument('--fold-store-ram', type=int, default=32768, metavar='N',
                    help="with --static: folded pipelines' result stores of at least N bits read through a "
                         'register, a host cycle or two ahead, in block or ultra RAM (their slot counters '
                         'the copy\'s own; a write landing in between forwarded unless the schedule shows '
                         'none can). Exact; default 32768; 0: read combinationally, LUTs (foldsim-fold-static fold-store-ram)')
    ap.add_argument('--fold-stage-levels-class', default='', metavar='CLASS:N|CLASS:off,...',
                    help='with --static: per class over --fold-stage-levels (globs on the model name; the last '
                         'matching spec wins; a spec of no class is an error; foldsim-fold-static '
                         'fold-stage-levels-class)')
    ap.add_argument('--fold-stage-levels', type=int, default=0, metavar='N',
                    help="with --static: folded pipelines' layer logic cut by host-clock stage registers at N "
                         'estimated levels (as --stage-levels counts them); results land later in the store, '
                         'checked against the schedule (foldsim-fold-static fold-stage-levels; 0: none)')
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
    ap.add_argument('--fold-ports-rows', action='store_true',
                    help='with --fold-ports: the bundles\' response tokens as rows of one memory (bundle, frame), read '
                         'at the reading instance\'s bundle; a held bit per bundle and one counter instead of token, '
                         'frame and previous-frame registers per bundle and field (foldsim-fold-static fold-ports-rows)')
    ap.add_argument('--fold-ports-reset', default='reset', metavar='INPUT',
                    help='with --fold-ports: the netlist input whose frame token rides on each request record')
    ap.add_argument('--link-kind-cost', action='store_true',
                    help='with --static: each link memory that may be block or ultra RAM takes the kind (block, '
                         'ultra or distributed RAM) by primitive cost against the remaining device budget per kind '
                         '(foldsim-fold-static link-kind-cost), not by --link-block-depth / --link-ultra-depth')
    ap.add_argument('--link-kind-device', default='', metavar='URAM,RAMB36,LUTS',
                    help='with --link-kind-cost: the budget per kind (default: the region\'s at --state-scratch-fill)')
    ap.add_argument('--link-kind-others', default='', metavar='URAM,RAMB36,LUTS',
                    help='with --link-kind-cost: what the rest of the simulator takes of each (foldsim_link_kinds.py '
                         '<gen dir>); default: as built by a first fold, refolded with it')
    ap.add_argument('--link-kind-group-others', default='', metavar='U,B,L;...',
                    help='with --link-kind-cost and --slr-groups: what the rest takes of each group\'s '
                         '(foldsim_link_kinds.py <gen dir> --groups, from a first build)')
    ap.add_argument('--slr-kept', type=int, default=None, metavar='K',
                    help='with --slr-groups: the assignments (fewest link bits between groups) scheduled to choose '
                         'the shortest frame, stopping at one at the modulo lower bound (foldsim-fold-static '
                         'slr-kept; default 8)')
    ap.add_argument('--slr-link-rams', action='store_true',
                    help='with --slr-groups: links may be block or ultra RAM too, in their reader\'s group '
                         '(foldsim-fold-static slr-link-rams); else every link of a grouped fold is distributed '
                         'RAM or registers')
    ap.add_argument('--slr-level', default=None, metavar='auto|G|none',
                    help='with --slr-groups: the floorplan puts the simulator level (its cells outside the groups) '
                         'and the shell environment beside it with group G (auto: the group of the region\'s SLR '
                         'with the most HBM NoC ports, else 0; default none: left to the placer)')
    ap.add_argument('--slr-group-control', action='store_true',
                    help='with --slr-groups: each class\'s sequencer, bubble register, slot tables and their register '
                         'chains built in its class\'s group (foldsim-fold-static slr-group-control); the same netlist '
                         'but for the hierarchy: only the frame decision and a parent\'s sequencer cross into a group')
    ap.add_argument('--slr-group-local', type=int, default=0, metavar='OPS',
                    help='with --slr-groups: each group computes the level\'s combinational values it reads (the frame '
                         'decision, a frame input\'s select) from their registers, cloning cones of up to OPS ops '
                         '(foldsim-fold-static slr-group-local): no logic before a crossing into a group; same values')
    ap.add_argument('--slr-group-own', type=int, default=0, metavar='OPS',
                    help='with --slr-groups: each group owns the level\'s control it reads -- a level register a group '
                         'reads, directly or through logic, replicated into the group with its next-value cone, back '
                         'through the registers it reads, up to OPS ops a closure (foldsim-fold-static slr-group-own): '
                         'no level register fans out over the groups; the same values in the same host cycles')
    ap.add_argument('--slr-group-reset', action='store_true',
                    help='with --slr-groups: the simulator\'s reset registered in each group and in the level '
                         '(foldsim-fold-static slr-group-reset); one host cycle more after reset, no value moves')
    ap.add_argument('--link-kinds', default='', metavar='P:R:KIND,...',
                    help='with --static: the RAM (distributed, block or ultra) of the links from model P '
                         'to model R that --link-block-depth\'s rules let block or ultra RAM hold, '
                         'whatever their depth (--state auto chooses them; foldsim-fold-static)')
    ap.add_argument('--stage-levels-mapped', default='', metavar='TARGET',
                    help='with --stage-levels: size each class\'s stages from the depth of its logic mapped to '
                         'the Beacon TARGET\'s cells (versal for the V80), not from the estimate alone: a class '
                         'whose mapped depth fits --stage-levels is not staged (foldsim-fold-static '
                         'stage-levels-mapped)')
    ap.add_argument('--stage-levels-class', default='', metavar='CLASS:N|CLASS:off,...',
                    help='with --static: per class over --stage-levels (globs on the model name; the last '
                         'matching spec wins): stage that class at N levels, or not at all (off); a spec that '
                         'matches no class is an error (foldsim-fold-static stage-levels-class)')
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
    ap.add_argument('--state-in-ram-zero-links', action='store_true',
                    help='with --state-in-ram and --zero-latency-links: links into a P > 1 class may be taken '
                         'in their producer\'s host cycle (default: from read registers, so that no producer '
                         'logic chains into its last-pass evaluation; foldsim-fold-static state-in-ram-zero-links)')
    ap.add_argument('--state-in-ram-pp', action='store_true',
                    help='with --state-in-ram: classes of P > 1 in the CSF pass-partitioned form: rows read '
                         'in place at their pass and written from the logic, staged with P stages, no capture '
                         'or write-back buffer (foldsim-fold-static state-in-ram-pp)')
    ap.add_argument('--state-scratch-bits', default='auto', metavar='N|auto',
                    help='with --static: the constant shell\'s on-chip state scratch in bits (folded pipelines\' '
                         'HBM-eligible stores first, then state rows: whole classes, largest first, while they '
                         'fit; then a class\'s instances); the rest streams from HBM (foldsim-fold-static '
                         'state-scratch-bits). auto (default): the region\'s block and ultra RAM at the placer\'s '
                         'fill, less the other memories the fold places on chip and state-scratch-reserve '
                         '(foldsim-fold-static state-scratch-device-bits); 250,000,000 without a region')
    ap.add_argument('--state-scratch-fill', type=float, default=0.85, metavar='F',
                    help='with --state-scratch-bits auto: the share of the region\'s ultra and block RAM the '
                         'simulator may fill (default 0.85, for routing; never past the fit check\'s own limits)')
    ap.add_argument('--state-scratch-no-rebuild', action='store_true',
                    help='with --state-scratch-bits auto: no as-built rerun (the other memories measured on the '
                         'first netlist; by default the fold reruns once when the planner missed them by more '
                         'than 5 %% of a kind\'s device)')
    ap.add_argument('--state-scratch-device', default='', metavar='URAM,BRAM',
                    help='with --state-scratch-bits auto: the ultra and block RAM bits the scratch is taken from, '
                         'in place of the region\'s at the placer\'s fill (a smaller device, or a test)')
    ap.add_argument('--state-scratch-reserve', default='', metavar='BLOCK[,ULTRA]',
                    help='with --state-scratch-bits auto: block (and ultra) RAM bits kept from the scratch for what '
                         'the fold places after it (links, rings, read buffers, the shell; foldsim-fold-static '
                         'state-scratch-reserve / state-scratch-reserve-ultra; default 45,000,000 and 10,000,000)')
    ap.add_argument('--state-scratch-whole', action='store_true',
                    help='with --state-scratch-bits: a PP class past the scratch streams whole (default: the '
                         'instances the rest of the scratch holds stay resident; foldsim-fold-static '
                         'state-scratch-partial=false)')
    ap.add_argument('--state-scratch-fixed', action='store_true',
                    help='with --state-scratch-bits: the scratch as a fixed allocation, the same RAM at every '
                         'instance count: each PP or paged class takes its rows\' share of it in instances (its '
                         'resident RAMs that deep, under-filled at small counts); a PP class streams the rest '
                         '(at least one instance in 16), a paged class streams whole past its share; a storage '
                         'rule\'s on-chip page kind names the RAM, no longer keeps the pages on chip past the '
                         'share (foldsim-fold-static state-scratch-fixed)')
    ap.add_argument('--state-scratch-fixed-pages', action='store_true',
                    help='with --state-scratch-fixed and --state-in-ram-pp: paged classes stay paged (default: every '
                         'paged class of several instances is state in RAM, PP with P its pages, so that the fixed '
                         'scratch streams part of it; foldsim-fold-static state-scratch-fixed-pp=false)')
    ap.add_argument('--state-stream-window', type=int, default=0, metavar='N',
                    help='with --state-scratch-bits: a streamed PP class\'s landing window, in instance slots '
                         '(foldsim-fold-static state-stream-window; 0: a RAMB36 deep, 512 / P)')
    ap.add_argument('--state-in-ram-pp-replicas', type=int, default=0, metavar='N',
                    help='with --state-in-ram-pp: each column\'s output register replicated per subtree (the first '
                         'N path components) of the registers it holds (foldsim-fold-static state-in-ram-pp-replicas; '
                         '0: off)')
    ap.add_argument('--state-in-ram-pp-no-pack', action='store_true',
                    help='with --state-in-ram-pp: the registers a class\'s rows leave out keep a store each '
                         '(default: packed into 72-bit columns of an instance\'s row each; foldsim-fold-static '
                         'state-in-ram-pp-pack)')
    ap.add_argument('--state-in-ram-pp-fanout', type=int, default=0, metavar='N',
                    help='with --state-in-ram-pp: the most RAM primitives a combinationally read memory\'s '
                         'registered address copy drives (foldsim-fold-static state-in-ram-pp-fanout; 0: 96)')
    ap.add_argument('--csf-mailbox-hbm-row-bits', type=int, default=0, metavar='N',
                    help='with --csf-mailbox-hbm: the most bits of a mailbox stream\'s row (foldsim-fold-static '
                         'csf-mailbox-hbm-row-bits; 0: its default, 4096)')
    ap.add_argument('--csf-mailbox-hbm-lead', type=int, default=0, metavar='N',
                    help='with --csf-mailbox-hbm: the most positions a mailbox stream\'s request precedes its landing '
                         '(foldsim-fold-static csf-mailbox-hbm-lead; 0: its default, 112)')
    ap.add_argument('--csf-mailbox-hbm', action='store_true',
                    help='constant-shell fold: reader-indexed link mailboxes in HBM, as HBM link streams of reader '
                         'rows (foldsim-fold-static csf-mailbox-hbm)')
    ap.add_argument('--csf-mailbox-builder', action='store_true',
                    help='constant-shell fold: the links of a producer class to a reader class as lanes of one '
                         'reader-indexed mailbox, a row per reader instance (foldsim-fold-static csf-mailbox-builder)')
    ap.add_argument('--csf-single-bank', action='store_true',
                    help='constant-shell fold: an early link read before its producer\'s next write keeps one '
                         'bank, the others two, as the schedule decides (foldsim-fold-static csf-single-bank)')
    ap.add_argument('--csf-mailbox-hbm-chains', action='store_true',
                    help='with --csf-mailbox-hbm: early chain links (a ring of siblings\' registered outputs) stream '
                         'through HBM too; zero-latency chains stay on chip (foldsim-fold-static csf-mailbox-hbm-chains)')
    ap.add_argument('--csf-mailbox-hbm-split', action='store_true',
                    help='with --csf-mailbox-hbm: a stream sliced by reader whose reader page is wider than a row is '
                         'split into streams of a row each (foldsim-fold-static csf-mailbox-hbm-split)')
    ap.add_argument('--csf-mailbox-hbm-share', action='store_true',
                    help='with --csf-mailbox-hbm: the sources of a reader input that are members of one stream at one '
                         'offset share one set of read registers (foldsim-fold-static csf-mailbox-hbm-share)')
    ap.add_argument('--link-rings', type=int, default=0, metavar='K',
                    help='constant-shell fold: a link memory whose entries are each live for a short window of the '
                         'frame keeps at most K entries (a power of two), the entry index mod K, where the schedule '
                         'shows no other entry of that address written in any read\'s window; early links drop their '
                         'banks (foldsim-fold-static link-rings)')
    ap.add_argument('--csf-merge-links', type=int, default=0, metavar='BITS',
                    help='constant-shell fold: link memories of equal ports (a producer and reader class '
                         'pair) merged side by side into mailboxes of at most BITS bits '
                         '(foldsim-merge-memories; 0: off)')
    ap.add_argument('--csf-mailboxes', action='store_true',
                    help='constant-shell fold, schedule only: links as reader-owned RAM mailboxes '
                         '(foldsim-fold-static csf-mailboxes)')
    ap.add_argument('--csf-mailbox-latency', type=int, default=1, metavar='N',
                    help='with --csf-mailboxes: the host cycles a mailbox read takes past a register link\'s')
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
    ap.add_argument('--state-copy-budget', default='0.75', metavar='FRAC[:SLR]',
                    help="with --state auto: the state-home policy's budget per class copy: its LUTs "
                         '(its logic, its stores\' LUTs and LUTRAM, synthesized: --state-calibration) at most '
                         'FRAC of the smallest SLR it may sit in (or the SLR named) of --slr-region (the V80\'s '
                         'by default), its block and ultra RAM within the region\'s margin of it; a copy over '
                         'moves stores to block or ultra RAM first (registered reads), then to HBM; default '
                         '0.75, 0: none')
    ap.add_argument('--state-home', choices=('any', 'onchip'), default='any',
                    help="with --state auto: onchip: the policy never moves state off chip (no virtual memories or "
                         "pages in HBM chosen; the HBM tier's relief is not admissible), e.g. the rule that up to 16 "
                         "SMs the state lives in block / ultra RAM; any (default): on chip first, then HBM")
    ap.add_argument('--state-region-budget', default='0.65', metavar='FRAC',
                    help="with --state auto and a copy budget: the whole region's LUTs (copies, level, kernel; "
                         'LUTRAM a LUT each, synthesized) at most FRAC of the region\'s, the state moved into block '
                         'and ultra RAM (then HBM) until it is: headroom under the per-SLR share for whole copies on '
                         'unequal SLRs (8 SMs split at 76 % left a copy without an SLR); default 0.65, 0: none')
    ap.add_argument('--state-ram-budget', default='', metavar='FRAC',
                    help="with --state auto and a copy budget: the whole region's block and ultra RAM (stores, "
                         'links, kernel) at most FRAC of the region\'s (default: the region\'s margin), the '
                         'stores and links that use RAM least well moved to LUTRAM (or HBM) until it is: RAM '
                         'headroom for whole copies on unequal SLRs, as --state-region-budget is for LUTs')
    ap.add_argument('--state-calibration', default='', metavar='JSON|none',
                    help="with --state auto: the area model's calibration to synthesized LUTs (copies' logic "
                         'and LUTRAM, the level, the kernel beside the simulator: foldsim_degree.py state-fit; '
                         'default calibrations/state-v80-y8b-20261113.json)')
    ap.add_argument('--state-measured', default='', metavar='DIR',
                    help='with --state auto: per class factors from a generation of the same recipe (its out '
                         'dir: state.json and foldsim_sim.mlir), its copies as counted against as priced')
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
    ap.add_argument('--lane-scan', action='store_true',
                    help='fold the top\'s per-child glue: the instances of each class with at least '
                         '--lane-scan-min instances in the top (or of --lane-scan-modules) are lanes; '
                         'their per-lane logic becomes a lane cell class <top>_lane_<module>, the reductions '
                         'across them zero-latency scans through its instances (foldsim-lane-scan; needs '
                         '--zero-latency-links --zero-latency-chains)')
    ap.add_argument('--lane-scan-modules', default='', metavar='M,...',
                    help='with --lane-scan: the lane modules (default: every module with enough instances)')
    ap.add_argument('--lane-scan-min', type=int, default=4, metavar='N',
                    help='with --lane-scan: the fewest instances that make lanes (default %(default)s)')
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
    ap.add_argument('--slr-split', action='append', default=[], metavar='GLOB:PARTS[:SLR,...]',
                    help='with --static: each class copy matching GLOB (its <copy>_static module) split into PARTS '
                         'modules at its registers (foldsim_slr_split.py --apply-map: a min-cut of the copy\'s '
                         'netlist that a combinational memory read never crosses; then the foldsim-slr-split pass, '
                         'which copies the little logic a part reads from another into it): no register added, no '
                         'cycle changed. Each part <prefix><copy>_static_p<k> is assigned to the SLR named in order '
                         '(default SLR2,SLR1,SLR0) in <out>/foldsim_floorplan.tcl. Repeatable')
    ap.add_argument('--slr-split-pass', default='',
                    help='foldsim-slr-split options for --slr-split, comma-separated, e.g. crossing=txrx,latency=2,slack=8 (a send and '
                         'a receive register on every crossing, the copy retimed around them, its outputs latency late: '
                         'give the fold the same --split-latency)')
    ap.add_argument('--split-latency', default='', metavar='GLOB:N,...',
                    help='with --static: the classes\' results N host cycles later (fold-static split-latency): the '
                         'schedule and the links\' writes; for a copy split with --slr-split-pass crossing=txrx latency=N')
    ap.add_argument('--split-latency-file', default='', metavar='FILE',
                    help='with --static: per class output its own split-latency (fold-static split-latency-file: '
                         'lines <class> <output> <n>, as foldsim-slr-split crossing=txrx lags= writes them)')
    ap.add_argument('--slr-split-python', default=sys.executable,
                    help='the Python that runs foldsim_slr_split.py (it needs the mtkahypar module)')
    ap.add_argument('--slr-split-options', default='--variants p --imbalance 0.15',
                    help='foldsim_slr_split.py options for --slr-split (the partition; "+" separates them too)')
    ap.add_argument('--slr-groups', action='store_true',
                    help='with --static: the copies in --slr-count groups, each in an SLR '
                         '(Vivado picks which): within --slr-capacity, the shortest frame, '
                         'then the fewest link bits between groups; a link between groups '
                         'registered on both sides and a host cycle later. '
                         '<out>/foldsim_floorplan.tcl assigns each group, for an '
                         'implementation flow to source after synthesis')
    ap.add_argument('--slr-count', type=int, default=3,
                    help='with --slr-groups: the most SLR groups (the V80: 3 SLRs)')
    ap.add_argument('--slr-fill', type=float, default=0, metavar='FRAC',
                    help="with --slr-groups and --slr-region: the fraction of each SLR's LUTs a group may fill "
                         "(default the region's margin), e.g. a fit rule's per-SLR LUT limit")
    ap.add_argument('--slr-ram-fill', type=float, default=0, metavar='FRAC',
                    help="with --slr-groups and an sll.json as --slr-sites: the fraction of each SLR's URAM288 and "
                         "RAMB36 a group may fill (default the region's margin)")
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
                    help='with --static and a region (--region or --slr-region): also fail on the SLR check\'s '
                         'refusals other than a straddle (the design past the region\'s placement fill, the SLLs '
                         'projected past a boundary), which are otherwise warnings; with --slr-groups the groups '
                         'also keep the boundary budget (sll_per_boundary x sll_margin) in link bits')
    ap.add_argument('--vmem-engines', choices=('placer', 'auto', 'nmu', 'local', 'flat'), default='placer',
                    help='where the kernel\'s virtual-memory controllers go, for the SLR check '
                         '(foldsim_sll_check.py --vmem-engines) and the floorplan: placer (default), left to the '
                         'placer, the check\'s fitted term; nmu, with the memory NoC ports (the region\'s noc '
                         'section); local, each with its class copy (with --slr-groups, in its SLR group in '
                         '<out>/foldsim_floorplan.tcl), its device-memory master crossing to the NoC ports; auto, '
                         'local where the ports are wider than the master')
    ap.add_argument('--region', default='auto', metavar='JSON',
                    help='with --static: the dynamic region the simulator is placed in (JSON, per SLR its LUTs, '
                         'LUTRAM, URAM and RAMB36 and the margin: foldsim_v80_slr.json, or `v80`). After the fold the '
                         'driver checks the netlist (foldsim_sll_check.py: each copy\'s LUT, LUTRAM, FF, URAM and '
                         'RAMB36 counted on foldsim_sim.mlir, an assignment of whole copies to the SLRs, the SLLs '
                         'each boundary takes; report in <out>/sll/sll.txt): a copy crossing an SLR boundary is a '
                         'GUIDE warning (--slr-policy guide, the default), the design past the placement fill, the '
                         'memories past the region\'s or SLLs past a boundary fail; --slr-policy strict keeps the '
                         '11-08 hard rule (a straddle fails, foldsim_sim.sv renamed foldsim_sim.sv.slr-refused). '
                         'Default `auto`: --slr-region if given, else the V80 compute shell\'s region (the static '
                         'fold\'s target) unless the manifest\'s [platform] device is not an xcv80 part')
    ap.add_argument('--allow-slr-straddle', default='', metavar='REASON',
                    help='with --slr-policy strict, the only override of the SLR rule: the compilation finishes with '
                         'a straddling copy, the reason recorded in <out>/SLR_STRADDLE_APPROVED and foldsim_sim.json '
                         '(slr_straddle_approved); with guide (the default) accepted and ignored (no longer needed)')
    ap.add_argument('--slr-policy', choices=('guide', 'strict'), default='guide',
                    help='with a region (--region): guide (default, 11-08): a class copy crossing an SLR boundary is a '
                         'warning (GUIDE lines: each straddling copy, its resources and SLRs, the projected SLLs per '
                         'boundary, the suggested assignment), foldsim_sim.sv kept; the SLR check\'s physical refusals '
                         '(the design past the placement fill, memories past the region\'s, SLLs past a boundary) '
                         'fail. strict: the 11-08 hard rule (a straddle fails, foldsim_sim.sv renamed '
                         'foldsim_sim.sv.slr-refused, only --allow-slr-straddle overrides it; the other refusals '
                         'fail with --sll-check, else warn)')
    ap.add_argument('--slr-sites', default='', metavar='JSON',
                    help="with --slr-groups: each copy's LUTs measured on a generated netlist of the same recipe "
                         '(JSON {copy: {"lut": n}}, foldsim_sll_check.py netlist resources) instead of the '
                         "estimate, in LUT sites (the region's sites_per_lut)")
    ap.add_argument('--slr-serialize', default='', metavar='[GLOB:]K,...',
                    help='with --slr-groups: the links between groups of the given classes (globs) carried '
                         'over K host cycles on a K-th of the wires (SLLs), K alone for every class; the '
                         "reader takes them K host cycles later and the producer's passes of a kind are kept "
                         'K apart (foldsim-fold-static slr-serialize); auto[:KMAX] (default 8): each class the '
                         'largest K <= KMAX with K x instances <= the ungrouped modulo lower bound')
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
    # A memory whose address registers sit in a class kept as an instance (a
    # deeper fold) gets a copy of them beside it first (foldsim-localize-read-
    # addresses): the class boundary stays. Memories that ask for block RAM
    # always (else the retime fails); unhinted ones (mode auto) only with
    # --localize-read-addresses auto, which may convert more of them.
    run([tool('circt-opt'), flat + '.in', '--symbol-dce', '--hw-flatten-modules', '--symbol-dce',
         '--canonicalize', '--cse', '-o', flat + '.pre'])
    os.remove(flat + '.in')
    loc_mode = ' mode=auto' if a.localize_read_addresses == 'auto' and a.retime_memory_reads is not None else ''
    loc_bits = f' min-bits={a.retime_memory_reads}' if a.retime_memory_reads is not None else ''
    run([foldsim_opt, flat + '.pre',
         f'--pass-pipeline=builtin.module(foldsim-localize-read-addresses{{{loc_mode}{loc_bits} verbose}})',
         '-o', flat + '.loc'])
    os.remove(flat + '.pre')
    run([tool('circt-opt'), flat + '.loc', f'--seq-retime-firmem-reads=report={out("memories.rpt")}'
         + (f' mode=auto min-bits={a.retime_memory_reads}' if a.retime_memory_reads is not None else ''),
         '--canonicalize', '--cse', '-o', flat])
    os.remove(flat + '.loc')
    # The top's per-child glue folded: lanes, a lane cell class, scans.
    if a.lane_scan:
        lanes = f' lanes={a.lane_scan_modules}' if a.lane_scan_modules else ''
        r = run([foldsim_opt, flat, f'--pass-pipeline=builtin.module(foldsim-lane-scan{{module={a.top}{lanes}'
                 f' min-lanes={a.lane_scan_min}}},hw.module(foldsim-remove-dead-state),canonicalize,cse)',
                 '-o', flat + '.ls'], check=False, capture_output=True)
        sys.stderr.write(r.stderr)
        if r.returncode:
            sys.exit('foldsim: lane-scan failed')
        os.replace(flat + '.ls', flat)
        with open(out('lane_scan.txt'), 'w') as f:
            f.write(r.stderr)
        for cell in re.findall(r'^lane-scan: cell (\S+)$', r.stderr, re.M):
            modules.append(cell)
            a.classes.append(cell)
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
        direct = ' direct-outputs' if getattr(a, 'direct_outputs', False) else ''
        passes.append(f'foldsim-extract-models{{module={m}{models} netlist absorb-logic={ABSORB_LOGIC}{exclusive}{direct}}}')
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
    if a.narrow_shifts:
        passes.append(f'hw.module(foldsim-narrow-shift-amounts{{min-width={a.narrow_shifts}}})')
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
        if not a.virtual_memories and a.state != 'auto' and not getattr(a, 'vmem_auto_bits', 0) and not getattr(a, 'vmem_auto_instance_bits', 0):
            sys.exit('--vmem-cache-rows needs --virtual-memories (or --state auto, --vmem-auto-bits)')
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
        if not a.virtual_memories and a.state != 'auto' and not getattr(a, 'vmem_auto_bits', 0) and not getattr(a, 'vmem_auto_instance_bits', 0):
            sys.exit('--vmem-group needs --virtual-memories (or --state auto, --vmem-auto-bits)')
        if not (a.vmem_group == 'copy' or (a.vmem_group.isdigit() and int(a.vmem_group) >= 1)
                or re.fullmatch(r'auto(:[1-9][0-9]*)?', a.vmem_group)):
            sys.exit('--vmem-group: copy, the memories per controller (at least 1), or auto[:M]')
    if a.vmem_weights and not (a.vmem_group or '').startswith('auto'):
        sys.exit('--vmem-weights needs --vmem-group auto')
    m = a.vmem_row_bytes_min
    if m < 1 or m & (m - 1):
        sys.exit('--vmem-row-bytes-min: a power of two')
    if a.vmem_controller == 'vport' and a.vmem_row_bytes_min < 64:
        # A static port moves whole 64-byte beats: rows of at least a beat.
        a.vmem_row_bytes_min = 64
    if a.vmem_known_zero_bits != 0 and a.vmem_known_zero_bits < 16:
        sys.exit('--vmem-known-zero-bits: 0 (none) or at least 16')
    if a.storage and not a.static:
        sys.exit('--storage needs --static')
    if a.schedule_only and not a.static:
        sys.exit('--schedule-only needs --static')
    # The SLR check's region (--region): the check runs on every static fold's netlist (--slr-policy).
    region = check_region(a)
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
        if a.slr_groups or region:
            if a.copies != 'auto':
                run([foldsim_opt, flat, f'--pass-pipeline=builtin.module({",".join(passes + OPTIMIZE)})',
                     '-o', out('models.mlir')])
            if a.slr_groups:
                fold += slr_options(a, foldsim_opt, out('models.mlir'), fold, out)
        sll_fold = fold
        # (The fold's own planner report, not the SLR check's refold's.)
        if 'state-scratch-device-bits=' in fold:
            fold += f' state-scratch-report={out("state_scratch.json")}'
            if os.path.exists(out('state_scratch.json')):
                os.remove(out('state_scratch.json'))
        rc = f'loads={a.control_fanout}'
        if a.fanout:
            rc += f' fanout={a.fanout}' + (' fanout-logic' if a.fanout_logic else '')
        replicate = f'hw.module(foldsim-replicate-controls{{{rc}}})'
        merge = ([f'hw.module(foldsim-merge-memories{{max-bits={a.csf_merge_links}}})']
                 if getattr(a, 'csf_merge_links', 0) else [])
        pipeline = passes + OPTIMIZE + [f'foldsim-fold-static{{{fold}}}'] + merge + [SPLIT, replicate,
                                        'canonicalize', 'foldsim-sort', 'cse']
        if a.copies == 'auto' or a.slr_groups or region:
            sim_input = out('models.mlir')
            pipeline = pipeline[len(passes) + len(OPTIMIZE):]
    else:
        pipeline = passes + OPTIMIZE + [f'foldsim-compose{{{compose}}}', SPLIT,
                                        'canonicalize', 'foldsim-sort', 'cse']
    sim = out('foldsim_sim.mlir')
    run([foldsim_opt, sim_input, NO_OP_DUMP, f'--pass-pipeline=builtin.module({",".join(pipeline)})',
         '-o', sim])
    if a.static and os.path.exists(out('state_scratch.json')):
        pipeline = scratch_as_built(a, foldsim_opt, sim_input, pipeline, sim, out)
    if a.static and getattr(a, 'link_kind_cost', False) and not getattr(a, 'link_kind_others', ''):
        pipeline = link_kinds_as_built(a, foldsim_opt, sim_input, pipeline, sim)
    split_plan = []
    if a.slr_split:
        if not a.static:
            sys.exit('--slr-split needs --static')
        split_plan = slr_split(a, foldsim_opt, sim, out)
    sim_text = open(sim).read()
    if a.static and a.state == 'auto':
        state_budget_check(a, sim, out)
    # A RAM with a registered read states its read-under-write behavior to
    # Vivado, which otherwise leaves a read colliding with a write undefined.
    run([tool('firtool'), '--format=mlir', sim,
         '--add-vivado-ram-address-conflict-synthesis-bug-workaround', '-o', out('foldsim_sim.sv')])

    if a.sll_check and not region:
        sys.exit('--sll-check needs --static and a region (--region or --slr-region)')
    if a.slr_groups:
        if not a.static:
            sys.exit('--slr-groups needs --static')
        names = [n for n, _ in slr_region_capacities(a.slr_region)] if a.slr_region else None
        write_slr_groups(open(out('foldsim_sim.sv')).read(), out('foldsim_floorplan.tcl'), names)
        sv_text = open(out('foldsim_sim.sv')).read()
        ngroups = len(set(re.findall(r'^\s+\w+ (foldsim_slr\d+) \(', sv_text, re.M)))
        level = slr_level_group(getattr(a, 'slr_level', None), a.slr_region, ngroups)
        if level is not None and ngroups:
            write_slr_level(out('foldsim_floorplan.tcl'), level, names)
            print(f'foldsim: the simulator level and its environment with SLR group {level}'
                  + (f' ({names[level]})' if names else ''), file=sys.stderr)
        if names and ngroups:
            write_slr_memory_ports(out('foldsim_sp.txt'), a.slr_region, names, ngroups, level)
    if split_plan:
        write_slr_split(split_plan, out('foldsim_floorplan.tcl'))

    threads = re.search(r'foldsim\.threads = \[(.*?)\]', sim_text).group(1)
    info = {
        'top': a.top,
        'threads': dict(t.strip('"').split('=') for t in re.findall(r'"[^"]*"', threads)),
        'ports': [{'dir': d, 'name': n, 'width': w} for d, n, w in signature(sim_text, 'foldsim_sim')],
    }
    for name, n in info['threads'].items():
        info['threads'][name] = int(n)
    # Static ports (fold-static vport-engines), in the simulator's vp<p>_ order:
    # each its FOLDSIM_VPORT<p>_* parameters (VBa/STATUS 5 item 9).
    vps = re.search(r'foldsim\.vports = \[(.*?)\]', sim_text)
    info['vports'] = []
    for entry in re.findall(r'"([^"]*)"', vps.group(1)) if vps else []:
        info['vports'].append(dict(zip(('DW', 'MW', 'TW', 'OW', 'LMAX', 'COPY', 'NEED_IN', 'HZ', 'BYTES', 'ZERO', 'FS', 'KS'),
                                       (int(x) for x in entry.split(':')))))
    # Each port's region where the kernel binds the ports to its device-memory
    # masters (no direct HBM channels): after the virtual memories' regions in
    # the kernel's vmem space, at FOLDSIM_VPORT<p>_BASE of the ports' part,
    # 4 KiB aligned; FOLDSIM_VPORT_REGION_BYTES in all.
    info['vport_bases'], off = [], 0
    for q in info['vports']:
        info['vport_bases'].append(off)
        off += -(-q['BYTES'] // 4096) * 4096
    info['vport_region_bytes'] = off
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
    if a.slr_groups and a.vmem_engines in ('local', 'auto'):
        noc = json.load(open(region)).get('noc', {}) if region else {}
        placed = write_vmem_engines(open(out('foldsim_sim.sv')).read(), info, out('foldsim_floorplan.tcl'),
                                    a.vmem_engines, noc.get('master_bits', VMEM_MASTER_BITS), names)
        info['vmem_engines'] = {'mode': a.vmem_engines, 'local': [c for c, _ in placed]}
        print(f'foldsim: --vmem-engines {a.vmem_engines}: {len(placed)} controllers with their copies '
              f'in foldsim_floorplan.tcl', file=sys.stderr)
    if getattr(a, 'vmem_slr_groups', '') and (info.get('vmem') or info.get('vports')):
        # Each controller's SLR group (foldsim_sim.vh FOLDSIM_VMEM_[G]<k>_SLR), the
        # kernel's masters by SLR and its floorplan (flow/v80/vmem_ports.py); a
        # static port's (FOLDSIM_VPORT<p>_SLR) its page engine's.
        sched = open(out('schedule.txt')).read() if os.path.exists(out('schedule.txt')) else ''
        for c, g in zip(info.get('vmem_groups') or info.get('vmem') or [], vmem_slr_of(info, sched, vmem_slr_map(a.vmem_slr_groups))):
            c['slr_group'] = g
        engines = page_engine_slr(sched, vmem_slr_map(a.vmem_slr_groups))
        info['vport_slr'] = [engines.get(f'__pages{p_}', -1) for p_ in range(len(info.get('vports') or []))]
        if region:
            info['vmem_slr_names'] = [s_['name'] for s_ in json.load(open(region)).get('slrs', [])]
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
        if info.get('vports'):
            f.write(f'`define FOLDSIM_VPORTS {len(info["vports"])}\n')
            for p, q in enumerate(info['vports']):
                for k, v in q.items():
                    f.write(f'`define FOLDSIM_VPORT{p}_{k} {v}\n')
                f.write(f'`define FOLDSIM_VPORT{p}_BASE 64\'h{info["vport_bases"][p]:x}\n')
            f.write(f'`define FOLDSIM_VPORT_REGION_BYTES 64\'h{info["vport_region_bytes"]:x}\n')
            for p_, g in enumerate(info.get('vport_slr') or []):
                f.write(f'`define FOLDSIM_VPORT{p_}_SLR {g}\n')
            if info.get('vport_slr') and info.get('vmem_slr_names') and not info['vmem']:
                f.write(f'`define FOLDSIM_VMEM_SLR_NAMES "{" ".join(info["vmem_slr_names"])}"\n')
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
            ctrls = info.get('vmem_groups') or info['vmem']
            if 'slr_group' in ctrls[0]:
                key = 'G' if 'vmem_groups' in info else ''
                for k, c in enumerate(ctrls):
                    f.write(f'`define FOLDSIM_VMEM_{key}{k}_SLR {c["slr_group"]}\n')
                if info.get('vmem_slr_names'):
                    f.write(f'`define FOLDSIM_VMEM_SLR_NAMES "{" ".join(info["vmem_slr_names"])}"\n')
            if getattr(a, 'vmem_controller', 'vmem') == 'vport':
                f.write('`define FOLDSIM_VMEM_VPORT\n')
                for k, v in enumerate(info['vmem']):
                    f.write(f'`define FOLDSIM_VMEM_{k}_BYTES 64\'h{v["bytes"]:x}\n')
                for g, v in enumerate(info.get('vmem_groups', [])):
                    f.write(f'`define FOLDSIM_VMEM_G{g}_BYTES 64\'h{v["bytes"]:x}\n')
            f.write(f'`define FOLDSIM_VMEM_BYTES 64\'h{info["vmem_bytes"]:x}\n')
    print(f'foldsim: {", ".join(f"{k} x{v}" for k, v in info["threads"].items())}', file=sys.stderr)
    if region:
        # The per-copy SLR / SLL check on the netlist just generated (its
        # stage registers, read buffers and simulator level counted; schedule
        # estimates said "fits" where placement failed): <out>/sll/sll.txt.
        # --slr-policy guide (default): a copy crossing an SLR boundary is a
        # GUIDE warning (slr_guide); the check's physical refusals fail.
        # strict: the 11-08 hard rule (slr_rule; only --allow-slr-straddle
        # overrides it); the check's other refusals fail with --sll-check,
        # else warn.
        os.makedirs(out('sll'), exist_ok=True)
        r = run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_sll_check.py'),
                 a.out, '--models', out('models.mlir'), '--fold-options', sll_fold, '--foldsim-opt', foldsim_opt,
                 '--slr-region', region, '--out', out('sll'), '--json', out('sll/sll.json'),
                 '--slr-policy', a.slr_policy, '--vmem-engines', a.vmem_engines],
                check=False, capture_output=True)
        line = next((l for l in reversed(r.stdout.splitlines()) if re.search(r': (OK|WARN|REFUSE):', l)), None)
        print(f'foldsim: SLL check: {line or "failed: " + (r.stderr or r.stdout)[-2000:]}', file=sys.stderr)
        verdict = line.split(': ', 1)[1] if line else None
        if a.slr_policy == 'guide':
            # A straddle is a guide; a failed check, the design past the placement fill, the memories past the
            # region's and SLLs past a boundary are physical failures (r16c: Place 30-5425): they fail.
            if r.returncode == 2 or verdict is None:
                sys.exit(f'foldsim: error: the SLR check failed (report {out("sll")})')
            slr_guide(a, out, verdict)
            if r.returncode and PHYSICAL.search(verdict):
                sys.exit(f'foldsim: error: the SLR check refuses this netlist: {verdict} (report {out("sll")}/sll.txt; '
                         'a placement past the fill or SLLs past a boundary fails as placed)')
            if r.returncode:
                # Not a physical refusal (a refold that does not reproduce the generation's II, ...): as before.
                if a.sll_check:
                    sys.exit(f'foldsim: --sll-check refuses this netlist (report {out("sll")}/sll.txt)')
                print(f'foldsim: WARNING: the SLL check refuses this netlist (report {out("sll")}/sll.txt; '
                      '--sll-check fails on it)', file=sys.stderr)
            return
        slr_rule(a, out, verdict, r.returncode)
        if r.returncode and not STRADDLE.search(verdict):
            if a.sll_check:
                sys.exit(f'foldsim: --sll-check refuses this netlist (report {out("sll")}/sll.txt)')
            print(f'foldsim: WARNING: the SLL check refuses this netlist (report {out("sll")}/sll.txt; '
                  '--sll-check fails on it)', file=sys.stderr)


if __name__ == '__main__':
    main()
