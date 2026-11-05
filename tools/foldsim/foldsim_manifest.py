#!/usr/bin/env python3
"""Target manifests: one file per design, the whole recipe of its simulator.

A manifest (TOML, or JSON by its extension) says what a design is and how it is
folded: its RTL (top, source list or files, include directories, defines, the
importer's limits), its classes (modules, `class-at` globs, fused pairs,
absorbed selects, copies), the environment's sinks, its memory policy
(virtual memories, storage kinds, link kinds, pages), its platform binding
(the memory endpoint's AXI4 speculation, the kernel clock, the memory ports)
and any other option of the driver. The compiler holds nothing of a design:
everything a design needs is in its manifest.

    [design]
    name = "tree-arb"
    top = "Top"                      # --top

    [rtl]
    srcs = "top.f"                   # --srcs (relative to the manifest)
    files = ["extra.sv"]             # --src, appended to the list's files
    incdirs = ["include"]            # --incdir
    defines = ["DEPTH=4"]            # --define
    max_generate_steps = 4194304

    [classes]
    modules = ["root+Bank"]          # the positional classes
    at = ["sl@*.slice", "root@inner:arb"]   # --class-at, in order
    copies = { sl = 2 }              # --copies sl:2

    [env]
    sinks = ["busy"]                 # --env-sinks busy

    [memory]
    virtual = "regex"                # --virtual-memories
    storage = ["M:regs>=0=registers"]   # --storage, joined by ','

    [platform]
    axi_spec = 2                     # --axi-spec=2
    clock_mhz = 250                  # the flow's (FOLDSIM_FREQ)

    [options]
    static = true                    # --static
    zero_latency_links = true

Every key of a section other than [flow], [lift] and the platform keys the
flow reads (FLOW_KEYS) is a driver option, by its long name with '_' or '-'
(the renames above aside); which section holds it is for the reader. A list
is the option repeated where the option repeats (--class-at), else its items
joined by ','; a table {k = v} is 'k:v' joined by ','; true sets a switch,
false leaves it (or --no-X for a --X/--no-X pair). Strings may use ${VAR},
from --manifest-var VAR=VALUE, then the environment; ${MANIFEST_DIR} is the
manifest's directory. The source paths (srcs, files, incdirs) are relative to
the manifest.

The manifest gives the driver's defaults: an option on the command line
overrides it, a repeated option or the positional classes replacing the
manifest's whole list.

[flow] and [lift] (lift_boundaries.py's --unit list) and the platform's flow
keys are the design flow's (sources generation, boundary lifting, the kernel
shell), kept here so a design is one file; the driver ignores them and
`foldsim_manifest.py get <section.key>` prints them for a Makefile.

usage: foldsim_manifest.py get <manifest> <section>.<key> [--var VAR=VALUE ...]
       foldsim_manifest.py from-argv [--name N] -- <driver argv ...>   (a manifest of a command line)
"""
import argparse
import json
import os
import re
import sys

FORMAT = 1

# Manifest keys whose option has another name.
RENAMES = {
    ('design', 'top'): 'top',
    ('rtl', 'srcs'): 'srcs',
    ('rtl', 'files'): 'src',
    ('rtl', 'incdirs'): 'incdir',
    ('rtl', 'defines'): 'define',
    ('classes', 'modules'): 'classes',
    ('classes', 'at'): 'class_at',
    ('env', 'sinks'): 'env_sinks',
    ('memory', 'virtual'): 'virtual_memories',
}
# Sections and keys of the flow, not of the driver.
FLOW_SECTIONS = ('flow', 'lift')
FLOW_KEYS = {('design', 'name'), ('design', 'description'), ('design', 'format'),
             ('platform', 'clock_mhz'), ('platform', 'device'), ('platform', 'kernel'),
             ('platform', 'adapter'), ('platform', 'mem_ports'), ('platform', 'mem_id_width'),
             ('platform', 'mem_channel_shift'), ('platform', 'vmem_masters'),
             ('platform', 'vmem_cache_rows_kernel'), ('platform', 'control')}
# Options that are paths, resolved against the manifest's directory.
PATH_DESTS = ('srcs', 'src', 'incdir')
# Where from-argv puts an option (the first rule whose pattern matches its dest).
SECTION_OF = [
    ('design', r'top$'),
    ('rtl', r'(srcs|src|incdir|define|max_generate_steps|max_unroll_trip_count|retime_memory_reads)$'),
    ('classes', r'(classes|class_at|absorb_exclusive|copies|nest.*|multicycle|registered_links)$'),
    ('env', r'(env_sinks|unstaged_env)$'),
    ('memory', r'(virtual_memories|vmem_.*|storage|memory_.*|page.*|state|register_state_entries|'
               r'no_page_masks|link_.*|compact_links|share_links|staged_links|no_share_link_writes)$'),
    ('platform', r'(axi_spec|spec_groups|fold_ports.*|slr_.*|relay|domains)$'),
    ('options', r'.*'),
]
# Options from-argv writes as lists (or copies as a table): ','-joined lists
# whose items hold no ','.
LIST_DESTS = ('copies', 'env_sinks', 'storage', 'link_kinds', 'absorb_exclusive', 'multicycle',
              'registered_links', 'spec_groups', 'slr_assign')
# Driver options a manifest never holds: where the tools are and what is
# written, which belong to the run.
RUN_DESTS = ('out', 'circt_bin', 'foldsim_bin', 'manifest', 'manifest_var', 'manifest_dump', 'jobs', 'help')

VAR = re.compile(r'\$\{([A-Za-z_]\w*)\}')


class ManifestList(list):
    """A list option's value from the manifest (a command line's replaces it)."""


def load(path):
    try:
        if path.endswith('.json'):
            return json.load(open(path))
        import tomllib
        with open(path, 'rb') as f:
            return tomllib.load(f)
    except (OSError, ValueError) as e:
        sys.exit(f'manifest {path}: {e}')


def expand(v, variables, where):
    if isinstance(v, str):
        def sub(m):
            if m.group(1) in variables:
                return variables[m.group(1)]
            if m.group(1) in os.environ:
                return os.environ[m.group(1)]
            sys.exit(f'manifest {where}: ${{{m.group(1)}}} is not set (--manifest-var {m.group(1)}=...)')
        return VAR.sub(sub, v)
    if isinstance(v, list):
        return [expand(x, variables, where) for x in v]
    if isinstance(v, dict):
        return {k: expand(x, variables, where) for k, x in v.items()}
    return v


def variables_of(path, pairs):
    v = {'MANIFEST_DIR': os.path.dirname(os.path.abspath(path))}
    for p in pairs or ():
        k, sep, val = p.partition('=')
        if not sep:
            sys.exit(f'--manifest-var {p}: not VAR=VALUE')
        v[k] = val
    return v


def entries(m, path):
    """(section, key, value) of every key of the manifest, flow keys included."""
    if not isinstance(m, dict):
        sys.exit(f'manifest {path}: not a table')
    fmt = m.get('design', {}).get('format', FORMAT)
    if fmt != FORMAT:
        sys.exit(f'manifest {path}: format {fmt}, this driver reads {FORMAT}')
    for sec, body in m.items():
        if not isinstance(body, dict):
            sys.exit(f'manifest {path}: [{sec}] is not a table')
        for k, v in body.items():
            yield sec, k, v


def get(path, dotted, pairs=()):
    """The value at <section>.<key> (None when absent), variables expanded."""
    m = load(path)
    sec, _, key = dotted.partition('.')
    v = m.get(sec, {})
    for k in key.split('.') if key else ():
        v = v.get(k) if isinstance(v, dict) else None
    if v is None:
        return None
    return expand(v, variables_of(path, pairs), f'{path} {dotted}')


def option_value(action, v, where):
    """The value an argparse action takes for a manifest value."""
    name = '/'.join(action.option_strings) or action.dest
    def scalar(x):
        if isinstance(x, bool):
            sys.exit(f'manifest {where}: {name} takes a value, not {str(x).lower()}')
        if action.type is not None and isinstance(x, str):
            try:
                x = action.type(x)
            except (TypeError, ValueError):
                sys.exit(f'manifest {where}: {name}: invalid value {x!r}')
        elif action.type is None and not isinstance(x, str):
            x = str(x)
        if action.choices is not None and x not in action.choices:
            sys.exit(f'manifest {where}: {name}: {x!r} is not one of {list(action.choices)}')
        return x
    kind = type(action).__name__
    if kind in ('_StoreTrueAction', '_StoreFalseAction', 'BooleanOptionalAction'):
        if not isinstance(v, bool):
            sys.exit(f'manifest {where}: {name} is a switch (true or false)')
        return v if kind != '_StoreFalseAction' else not v
    if kind == '_AppendAction' or action.nargs in ('*', '+'):
        return ManifestList(scalar(x) for x in (v if isinstance(v, list) else [v]))
    if isinstance(v, dict):
        v = ','.join(f'{k}:{x}' for k, x in v.items())
    elif isinstance(v, list):
        v = ','.join(str(x) for x in v)
    return scalar(v)


def defaults(path, parser, pairs=()):
    """The driver's defaults a manifest gives: {dest: value}."""
    m = load(path)
    variables = variables_of(path, pairs)
    base = variables['MANIFEST_DIR']
    actions = {a.dest: a for a in parser._actions}
    out, where_of = {}, {}
    for sec, k, v in entries(m, path):
        where = f'{path} [{sec}] {k}'
        if sec in FLOW_SECTIONS or (sec, k) in FLOW_KEYS:
            continue
        dest = RENAMES.get((sec, k), k.replace('-', '_'))
        if dest in RUN_DESTS or dest not in actions:
            sys.exit(f'manifest {where}: no driver option {dest}')
        if dest in out:
            sys.exit(f'manifest {where}: {dest} given twice (also {where_of[dest]})')
        v = option_value(actions[dest], expand(v, variables, where), where)
        if dest == 'rewrite':
            for r in v:
                if isinstance(r, dict) and isinstance(r.get('file'), str) and not os.path.isabs(r['file']):
                    r['file'] = os.path.normpath(os.path.join(base, r['file']))
        if dest in PATH_DESTS:
            res = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
            v = ManifestList(res(p) for p in v) if isinstance(v, list) else res(v)
        out[dest], where_of[dest] = v, where
    return out


def parse_args(parser, argv=None):
    """parser.parse_args with the defaults of --manifest, if any: the command
    line overrides each, and a list option given on it replaces the
    manifest's list. --manifest-dump prints the options as JSON and exits."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--manifest', default='')
    pre.add_argument('--manifest-var', action='append', default=[])
    p, _ = pre.parse_known_args(argv)
    given = {}
    if p.manifest:
        given = defaults(p.manifest, parser, p.manifest_var)
        parser.set_defaults(**given)
    a = parser.parse_args(argv)
    for dest, v in given.items():
        got = getattr(a, dest)
        # argparse appends a command line's items to a copy of the default.
        if isinstance(v, ManifestList) and got is not v and isinstance(got, list) \
                and len(got) > len(v) and list(got[:len(v)]) == list(v):
            setattr(a, dest, list(got[len(v):]))
        elif isinstance(got, ManifestList):
            setattr(a, dest, list(got))
    if getattr(a, 'manifest_dump', False):
        json.dump({k: v for k, v in sorted(vars(a).items()) if k not in RUN_DESTS},
                  sys.stdout, indent=1)
        print()
        sys.exit(0)
    return a


# --- from-argv: a manifest of a driver command line ------------------------

def toml_value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)   # a TOML basic string is a JSON string
    if isinstance(v, list):
        if len(v) > 1 or (v and len(toml_value(v[0])) > 60):
            return '[\n' + ''.join(f'  {toml_value(x)},\n' for x in v) + ']'
        return '[' + ', '.join(toml_value(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{ ' + ', '.join(f'{k} = {toml_value(x)}' for k, x in v.items()) + ' }'
    raise TypeError(v)


def toml_key(k):
    return k if re.fullmatch(r'[A-Za-z0-9_-]+', k) else json.dumps(k)


def dump(m):
    out = []
    for sec, body in m.items():
        if not body:
            continue
        out.append(f'[{sec}]')
        for k, v in body.items():
            out.append(f'{toml_key(k)} = {toml_value(v)}')
        out.append('')
    return '\n'.join(out)


def from_argv(parser, argv, name='', extra=None):
    """The manifest of a driver command line: every option it gives that is
    not its default (run options aside), lists split where the option joins
    them by ',' and copies as a table. `extra`: flow sections to add."""
    a, d = parser.parse_args(argv), parser.parse_args([])
    back = {v: k for k, v in RENAMES.items()}
    m = {'design': {'name': name} if name else {}}
    for act in parser._actions:
        dest = act.dest
        if dest in RUN_DESTS or not hasattr(a, dest):
            continue
        v = getattr(a, dest)
        if dest not in ('srcs', 'top') and v == getattr(d, dest):
            continue
        if v is None or v == [] or (dest in ('srcs',) and not v):
            continue
        if isinstance(v, str) and dest in LIST_DESTS and v:
            parts = v.split(',')
            if dest == 'copies' and all(re.fullmatch(r'[\w$*]+:\d+', x) for x in parts):
                v = {x.split(':')[0]: int(x.split(':')[1]) for x in parts}
            else:
                v = parts
        sec, key = back.get(dest, (None, None))
        if sec is None:
            sec = next(s for s, pat in SECTION_OF if re.fullmatch(pat, dest))
            key = dest
        m.setdefault(sec, {})[key] = v
    for sec, body in (extra or {}).items():
        m.setdefault(sec, {}).update(body)
    order = ['design', 'rtl', 'classes', 'env', 'memory', 'platform', 'options', 'lift', 'flow']
    return {s: m[s] for s in order + [s for s in m if s not in order] if s in m}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'get':
        ap = argparse.ArgumentParser(prog='foldsim_manifest.py get')
        ap.add_argument('manifest')
        ap.add_argument('key')
        ap.add_argument('--var', action='append', default=[])
        a = ap.parse_args(sys.argv[2:])
        v = get(a.manifest, a.key, a.var)
        if v is None:
            return
        if isinstance(v, dict):
            v = [f'{k}:{x}' for k, x in v.items()]
        print(' '.join(str(x) for x in v) if isinstance(v, list) else
              (str(v).lower() if isinstance(v, bool) else v))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == 'from-argv':
        ap = argparse.ArgumentParser(prog='foldsim_manifest.py from-argv')
        ap.add_argument('--name', default='')
        ap.add_argument('--section', action='append', default=[], metavar='SEC.KEY=JSON',
                        help='a flow key to add (its value JSON, else a string)')
        ap.add_argument('argv', nargs=argparse.REMAINDER)
        a = ap.parse_args(sys.argv[2:])
        argv = a.argv[1:] if a.argv[:1] == ['--'] else a.argv
        import importlib.machinery
        import importlib.util
        here = os.path.dirname(os.path.realpath(__file__))
        drv = next(p for p in (os.path.join(here, 'foldsim.py'), os.path.join(here, 'foldsim'))
                   if os.path.exists(p))
        spec = importlib.util.spec_from_loader('foldsim_driver',
                                               importlib.machinery.SourceFileLoader('foldsim_driver', drv))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        extra = {}
        for s in a.section:
            k, _, v = s.partition('=')
            sec, _, key = k.partition('.')
            try:
                v = json.loads(v)
            except ValueError:
                pass
            extra.setdefault(sec, {})[key] = v
        print(dump(from_argv(mod.build_parser(), argv, a.name, extra)), end='')
        return
    sys.exit('usage: ' + __doc__.split('usage: ', 1)[1].rstrip())


if __name__ == '__main__':
    main()
