#!/usr/bin/env python3
"""Choosing the fold degree: how many copies each class is folded into.

A class of n instances folded into k copies costs k copies of its logic and
k stores of its state, each about n/k instances deep (the first n mod k one
deeper), and the frame it allows: the deepest copy's. For every choice of k per class:

  II      exact: the scheduler's (foldsim-fold-static schedule-only);
  LUTs    Beacon's area of each class (foldsim-opt: foldsim-beacon-map area,
          foldsim-estimate-area, on the V80's target), calibrated per class
          against Vivado's hierarchical utilization of a build of the design;
  sites   the LUT sites the placer needs, distributed RAM weighted, against
          the V80 dynamic region (fits);
  period  a fit of the routed period against the largest copy's LUTs and,
          with a logic screen of every build, the netlist's depth;
  storage with --storage auto, a kind per state store (flip-flops,
          distributed, block or ultra RAM) against the region's sites,
          RAMB36 and URAM288 (foldsim_storage.py); fixed: the compiler's
          default rules against the device's RAMB36;
  routes  the routability term (foldsim_storage.py): the router's congestion
          fitted on the V80 builds against LUTs and block RAMs and the SLRs
          they spread over (the region's, fewer with fewer SLR groups); a pick
          must route as well as place (--no-routability: place only).

A simulated cycle takes II x period ns: since the memory endpoint is in the
frame, H = II. Reports the Pareto front of ns per simulated cycle against LUTs,
the fastest point that places within a LUT budget, and the unfolded target's
area (what an unfolded simulator needs at the least). The model ranks and
prunes; it does not promise a clock: retiming and routing dominate the period.

Every choice is scheduled while there are few; lanes folded as classes (256
instances each) make tens of thousands, and those are searched (evaluate).

usage:
  foldsim_degree.py calibrate --models <models.mlir> --schedule <schedule.txt>
      --util <routed_util_hier.rpt> --timing <routed_timing_summary.rpt>
      --period <ns> --name <build> --out <calibration.json> [--screen <dir>]
  foldsim_degree.py fit --calibration <calibration.json>
  foldsim_degree.py explore --models <models.mlir> --module <top> --workdir <dir>
      [--calibration F] [--screen <dir>] [--storage fixed|auto] [--no-routability]
      [--no-uram] [--slr-groups] [--schedules <dir> [--cached-only]]
  foldsim.py ... --static --copies auto [--storage auto] [--lut-budget N]
      [--degree-calibration F]
"""
import argparse
from collections import defaultdict
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import foldsim_cost as cost  # noqa: E402
import foldsim_storage as storage  # noqa: E402


# The V80 dynamic region's LUT sites as the placer counts them, and what a LUT
# of distributed RAM costs there in sites: a RAM64M8 claims its whole SLICEM
# however few bits it holds. Fitted to the one full image that failed to
# place (1,094,665 logic + 547,062 LUTRAM LUTs needed 3,326,213 sites of
# 3,269,043); images at 1.9M sites placed.
PLACE_SITES = 3269043
LUTRAM_SITES = 4.08
# The V80's block RAMs (RAMB36E5).
BRAM_SITES = 3741
# Placement is not routing: keep a margin for a design that also routes.
PLACE_MARGIN = 0.9


def placement(logic, lutram, bram=0.0):
    """(LUT sites the placer needs, whether they, as placed (times
    storage.PLACED_PER_PRICED), and the block RAMs fit the region with
    margin)."""
    sites = logic + LUTRAM_SITES * lutram
    return sites, (sites * storage.PLACED_PER_PRICED <= PLACE_MARGIN * PLACE_SITES
                   and bram <= PLACE_MARGIN * BRAM_SITES)


def parse_schedule(text):
    """II, each class's instance count, and the last output's slot."""
    ii = int(re.search(r'II = (\d+)', text).group(1))
    classes = {m.group(1): int(m.group(2))
               for m in re.finditer(r'^  (\S+) x(\d+)', text, re.M)}
    slots = [int(s) for s in re.findall(r'^  output \S+ @(\d+)', text, re.M)]
    return ii, classes, max(slots) if slots else 0


def _ir_estimates(models, names):
    """Per class, one instance: logic LUTs (black boxes included), register
    bits and the memories, from the models' IR."""
    mods, _ = cost.parse(models)
    out = {}
    for name in names:
        acc = dict(lut=0.0, ff=0, mems=[], regw={})

        def visit(n):
            m = mods[n]
            acc['lut'] += sum(m.lut.values())
            acc['ff'] += m.ff
            acc['mems'] += [list(r) for r in m.mems]
            for w, k in m.regw.items():
                acc['regw'][w] = acc['regw'].get(w, 0) + k
            for child, _ in m.insts:
                base = child.split('__')[0]
                if child in cost.BLACKBOX or base in cost.BLACKBOX:
                    acc['lut'] += cost.BLACKBOX.get(child, cost.BLACKBOX.get(base))[0]
                elif child in mods:
                    visit(child)
        if name in mods:
            visit(name)
        out[name] = acc
    for e in out.values():
        e.update(lut_ip=0.0, scale=1.0)
    return out


# The Beacon target whose area prices a class: the calibration is the V80's.
AREA_TARGET = 'versal'
# The area model a build's estimates (est_logic, est_lutram, and the kappa and
# lam fitted on them) come from. The area factors of a build fitted on another
# model, or with none recorded, are not used; its measured period, LUTs and
# largest copy still serve the period fit.
BEACON = 'beacon-area'
# The IR cost model (foldsim_cost.py: our calibrations, our pricer's storage / paging / link terms) or Beacon's
# area (upstream 11-01). A build without a recorded estimator was fitted on the IR cost model. Default: the IR
# cost model (the fold-a100 pricer's behaviour); --estimator beacon-area selects Beacon's (set_estimator).
IR_COST = 'foldsim_cost.py'
ESTIMATORS = (IR_COST, BEACON)
ESTIMATOR = IR_COST


def set_estimator(name):
    global ESTIMATOR
    if name not in ESTIMATORS:
        sys.exit(f'unknown estimator {name} (one of {", ".join(ESTIMATORS)})')
    ESTIMATOR = name


def build_estimator(b):
    """The estimator a calibration build was fitted on (none recorded: the IR cost model)."""
    return b.get('estimator', IR_COST)


def default_foldsim_opt():
    """foldsim-opt beside this script (the build or install tree), else PATH."""
    here = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'foldsim-opt')
    return here if os.path.exists(here) else 'foldsim-opt'


def pow2(threads):
    """The instance index's range: a power of two."""
    return 1 << (threads - 1).bit_length() if threads > 1 else 1


def _beacon_estimates(models, classes, foldsim_opt, out=None):
    """Per class, one instance, from Beacon's area of the models' IR
    (foldsim-estimate-area): the area mapping's logic LUTs before Beacon's
    device scale (`lut`, what kappa calibrates, so a new scale leaves the
    fitted factors valid), the Vivado cores' LUTs (`lut_ip`), the device
    scale (`scale`, the factor without a calibration), flip-flops, and its
    state at each power-of-two instance count up to the largest class
    (state_lutram). `out` keeps the pass's module table."""
    threads = [1 << i for i in range(pow2(max(classes.values(), default=1)).bit_length())]
    with tempfile.TemporaryDirectory() as tmp:
        table = out or os.path.join(tmp, 'area.json')
        pipeline = (f'builtin.module(foldsim-beacon-map{{target={AREA_TARGET} area}},'
                    f'foldsim-estimate-area{{target={AREA_TARGET} modules={table} report= '
                    f'state-threads={",".join(map(str, threads))}}})')
        r = subprocess.run([foldsim_opt, models, f'--pass-pipeline={pipeline}', '-o', os.devnull],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'foldsim-estimate-area failed on {models}:\n{r.stderr}')
        doc = json.load(open(table))
        mods, scale = doc['modules'], doc.get('lut_scale', 1.0)
    est = {}
    for name in classes:
        m = mods.get(name)
        if not m:
            est[name] = dict(lut=0.0, lut_ip=0.0, scale=scale, ff=0,
                             state={t: (0.0, 0.0, 0.0) for t in threads})
            continue
        state = {int(t): (s['reg_lutram'], s['mem_lutram'], s['mem_bram'])
                 for t, s in m['state'].items()}
        # The memories' read multiplexers are priced with the memories, at the
        # depth a copy gives them (state_lutram), not as the class's logic.
        mux = state[1][1] - m['lutram']
        # A Vivado core's flip-flops are its pipeline, not the class's state.
        est[name] = dict(lut=m['lut_mapped'], lut_ip=max(m['lut_fixed'] - mux, 0.0),
                         scale=scale, ff=m['ff'] - m.get('ip_ff', 0), state=state)
    return est


def class_estimates(models, classes, foldsim_opt=None, out=None):
    """Per class, one instance: the estimator's (ESTIMATOR) area -- the IR cost model's logic LUTs, register
    bits, register widths and memories (lut_ip 0, scale 1), or Beacon's (_beacon_estimates)."""
    if ESTIMATOR == BEACON:
        return _beacon_estimates(models, classes if isinstance(classes, dict) else {c: 1 for c in classes},
                                 foldsim_opt or default_foldsim_opt(), out)
    return _ir_estimates(models, classes)


# A class's generic cost features (class_features): what its logic is made
# of, from the models' IR, for any design. A calibration's `generic` model
# prices a class that none of its families names by these.
GENERIC_FEATURES = ('log_lut', 'mux_share', 'arith_share', 'ff_per_lut', 'mem_bits_per_lut')


def class_features(models, names):
    """Per class: GENERIC_FEATURES of one instance (logic LUTs as the model
    prices them, by op kind; register bits; memory bits)."""
    mods, _ = cost.parse(models)
    out = {}
    for name in names:
        acc = dict(lut=defaultdict(float), ff=0, mem_bits=0)

        def visit(n):
            m = mods[n]
            for k, v in m.lut.items():
                acc['lut'][k] += v
            acc['ff'] += m.ff
            acc['mem_bits'] += sum(r[1] * r[2] for r in m.mems)
            for child, _ in m.insts:
                if child in mods and child not in cost.BLACKBOX and child.split('__')[0] not in cost.BLACKBOX:
                    visit(child)
        if name in mods:
            visit(name)
        lut = sum(acc['lut'].values())
        if lut <= 0:
            continue
        arith = sum(v for k, v in acc['lut'].items() if k in ('add', 'sub', 'mul', 'icmp', 'shl', 'shru', 'shrs'))
        out[name] = dict(log_lut=math.log10(lut), mux_share=acc['lut'].get('mux', 0.0) / lut,
                         arith_share=arith / lut, ff_per_lut=acc['ff'] / lut,
                         mem_bits_per_lut=acc['mem_bits'] / lut / 64.0)
    return out


def generic_factor(model, feats):
    """A generic model's factor ({features, coef, lo, hi}: exp of a linear
    form of the features, clamped to the fitted families' range)."""
    x = model['coef'][0] + sum(c * feats[f] for c, f in zip(model['coef'][1:], model['features']))
    return min(max(math.exp(x), model['lo']), model['hi'])


def state_lutram(est, threads, form='lutram'):
    """Distributed-RAM LUTs of a copy's state at `threads` instances: its
    registers as memories of one entry per instance (flip-flops for one, a
    ring or a store in flops), and its memories `threads` times deeper, an
    entry for each instance index (a power of two)."""
    if 'state' in est:   # Beacon's area: the state at each power-of-two instance count
        reg, mem_lutram, mem_bram = est['state'][pow2(threads)]
        return (reg if form == 'lutram' else 0.0) + mem_lutram, mem_bram
    threads = 1 << (threads - 1).bit_length() if threads > 1 else 1
    luts = 0.0 if form != 'lutram' else cost.mem_cost(threads, est['ff'], 1, 0)[0]
    bram = 0.0
    for (_, d, w, r, _, rl, st, *_) in est['mems']:
        lr, br = cost.mem_cost(d * threads, w, r, rl, st)
        luts += lr
        bram += br
    return luts, bram


def split_virtual(est, pattern):
    """The classes' estimates with the memories foldsim-fold-static keeps
    outside the simulator (`virtual-memories`: the named memories the regex
    finds in a memory's name or in `<class>:<name>`, in HBM behind the
    kernel's foldsim_vmem controllers) taken off chip: listed under 'vmems',
    no longer under 'mems'."""
    if not pattern:
        return est
    rx = re.compile(pattern)
    base = lambda c: re.sub(r'_copy\d+$', '', c)
    virtual = lambda c, m: m[8] and (rx.search(m[0]) or rx.search(f'{base(c)}:{m[0]}'))
    return {c: dict(e, mems=[m for m in e['mems'] if not virtual(c, m)],
                    vmems=e.get('vmems', []) + [m for m in e['mems'] if virtual(c, m)])
            for c, e in est.items()}


def vmem_controllers(mems, threads, vmem_group=None, row_bytes_min=1):
    """The kernel's controllers of a copy's virtual memories (`mems`:
    class_estimates' memories, `threads` instances): one a memory, or with
    foldsim --vmem-group copy one a row size (members placed largest first,
    rows in a power of two of the group's image). Each: dict(data_width,
    row_width, instances, members)."""
    if vmem_group != 'copy':
        return [dict(data_width=m[2], row_width=max(1, (m[1] - 1).bit_length()), instances=threads,
                     members=1) for m in mems]
    groups = {}
    for m in mems:
        groups.setdefault(storage.vmem_row_bytes(m[2], row_bytes_min), []).append(m)
    return [dict(data_width=max(m[2] for m in g),
                 row_width=max(1, (sum(1 << max(0, (m[1] - 1).bit_length()) for m in g) - 1).bit_length()),
                 instances=threads, members=len(g)) for g in groups.values()]


def vmem_bram(est, classes, degree, vmem, vmem_group=None, row_bytes_min=1):
    """RAMB36 of the host caches of a fold degree's virtual memories: a
    controller per memory of each copy (vmem_controllers: a row size of a
    copy with --vmem-group copy), of the copy's instances times its rows,
    each cache's lines and tags as the kernel builds them (storage.ctl_cost);
    `vmem`: (foldsim --vmem-cache-rows, its budget), or None for none (the
    kernel's default)."""
    if not vmem:
        return 0.0
    vms = []
    for c, n in classes.items():
        k = degree.get(c, 1)
        for j in range(k):
            t = copy_threads(n, k, j)
            vms += vmem_controllers(est[c].get('vmems', []), t, vmem_group, row_bytes_min)
    return storage.vmem_cache_bram(vms, *vmem)


# A generated simulator's controllers (state --built-json: its foldsim_sim.json
# and the foldsim_kernel.vh beside it), priced as built: every foldsim_vmem
# the kernel instantiates (a virtual memory or group, an HBM link stream's
# own, a page engine) by storage.ctl_cost with its row bits, host-cache rows,
# members and known-zero bits; FORWARD from foldsim_sim.vh; the masters, the
# shares and the SLASH sc_mem tree (storage.platform_cost). Set by state_main;
# when set, storage_area prices no controller from the estimates.
BUILT = None
# The page engines' count (storage_area): 'max' every off-chip copy's columns
# and every stream on engines of their own (an upper bound); 'first-fit' as
# foldsim-fold-static assigns them (storage.engines_first_fit), `shift` the
# fold's link-hbm-shift (state --engine-rule, --link-hbm-shift).
ENGINES = dict(rule='max', shift=0)
# The platform's masters (storage.platform_cost 'best'): by the whole HBM share
# (default) or by the point's own HBM traffic (state --masters-by-traffic).
PLATFORM = dict(by_traffic=False)
# A class copy's own block and ultra RAM, each within one SLR's (state
# --copy-cap): {'uram': most URAM288, 'bram': most RAMB36} a copy may hold, or
# None (no cap). A copy is placed whole in an SLR (Vivado keeps a module's
# cells together where it can; a copy past an SLR's memory must be cut, its
# wide page and store ports then crossing SLLs: r16c 11-07, a 16-SM slice copy
# of 517 URAM against SLRs of 496 / 511 / 204, failed partitioning at 23,987
# SLLs). The stores' kinds are chosen under these caps (pools per copy in
# storage.assign).
COPY_CAP = None


def slr_copy_cap(region, slr='smallest', luts=False):
    """COPY_CAP of a dynamic region's SLR (foldsim_v80_slr.json: per SLR its
    ramb36 and uram, and the margin a copy may fill): `slr` its name, or
    smallest / largest (by LUTs); with `luts` (state --copy-cap-luts) its
    LUTs and LUTRAM LUTs too."""
    slrs = region['slrs']
    if slr in ('smallest', 'largest'):
        pick = (min if slr == 'smallest' else max)(slrs, key=lambda s: s['luts'])
    else:
        pick = next((s for s in slrs if s['name'] == slr), None)
        if pick is None:
            raise ValueError(f'no SLR {slr} in the region (' + ', '.join(s['name'] for s in slrs) + ')')
    m = region.get('margin', 1.0)
    cap = dict(name=pick['name'], uram=m * pick['uram'], bram=m * pick['ramb36'])
    if luts:
        # LUTs: the copy's own logic plus its stores' LUTs and LUTRAM (g32cdj,
        # 11-07: a slice copy of 604.9k LUTs refused against SLR2's 601.6k);
        # LUTRAM within the SLR's SLICEMs.
        # Priced LUTs place as storage.PLACED_PER_PRICED as many.
        cap.update(lut=m * pick['luts'] / storage.PLACED_PER_PRICED,
                   lutram=m * pick.get('lutram_luts', pick['luts']) / storage.PLACED_PER_PRICED,
                   # A Versal CLB: two flip-flops a LUT.
                   ff=m * 2 * pick['luts'])
    return cap


# The state-home policy's area in synthesized LUTs (foldsim --state auto with
# --state-calibration): factors on the cost model's copy logic (`copy_logic`:
# a copy's own logic and its stores' selects) and its stores' distributed RAM
# (`copy_lutram`), a class's own where a generation measured it (`classes`:
# {model: {logic, lutram}}, foldsim --state-measured), the simulator level's
# (`level`: the links and the rest outside the copies), the kernel and the
# shell glue beside the simulator (`kernel_logic`, `kernel_lutram`, fixed),
# and the placed LUTs per synthesized one (`placed_per_priced`, the region
# pool's margin, storage.PLACED_PER_PRICED). None: the cost model as it is.
STATE_CAL = None
# A store's kind the flow fixes (foldsim --storage rules given with --state
# auto): {'mem': {(model, store): kind}, 'regs': {model: [(width, kind)]}}.
# Such a store is priced in its kind and never moved; the choice emits no
# rule for it (the given one stays the only one).
GIVEN = None


def state_factors(c):
    """STATE_CAL's (logic, LUTRAM) factors of a class (its copies' model)."""
    if not STATE_CAL:
        return 1.0, 1.0
    m = (STATE_CAL.get('classes') or {}).get(re.sub(r'_copy\d+$', '', c))
    if m:
        return m['logic'], m['lutram']
    return STATE_CAL.get('copy_logic', 1.0), STATE_CAL.get('copy_lutram', 1.0)


def copy_budget(region, frac, slr='smallest'):
    """COPY_CAP of the state-home policy: each copy's LUTs (its logic plus its
    stores' LUTs and LUTRAM, synthesized: STATE_CAL) at most `frac` of the
    smallest SLR it may sit in (`slr`: smallest / largest or a name), placed
    over synthesized (the region's placed_per_synthesized), its LUTRAM within
    the share of the SLR's SLICEM LUTs, and its block and ultra RAM within the
    region's margin of the SLR's."""
    slrs = region['slrs']
    if slr in ('smallest', 'largest'):
        pick = (min if slr == 'smallest' else max)(slrs, key=lambda x: x['luts'])
    else:
        pick = next((x for x in slrs if x['name'] == slr), None)
        if pick is None:
            raise ValueError(f'no SLR {slr} in the region (' + ', '.join(x['name'] for x in slrs) + ')')
    m = region.get('margin', 1.0)
    pl = region.get('placed_per_synthesized', {}).get('lut', 1.0)
    return dict(name=pick['name'], frac=frac, uram=m * pick['uram'], bram=m * pick['ramb36'],
                lut=frac * pick['luts'] / pl,
                lutram=frac * pick.get('lutram_luts', pick['luts']) / pl)


def given_rules(text):
    """GIVEN of a --storage rule list (<model>:<store>=<kind>,...)."""
    out = dict(mem={}, regs={})
    for rule in filter(None, (text or '').split(',')):
        key, _, kind = rule.rpartition('=')
        model, _, store = key.partition(':')
        if store.startswith('regs>='):
            out['regs'].setdefault(model, []).append((int(store[6:]), kind))
        elif store == 'regs':
            out['regs'].setdefault(model, []).append((0, kind))
        elif store:
            out['mem'][(model, store)] = kind
    return out


def fix_given(units):
    """The units GIVEN names: fixed at the given kind (a register store by
    width: the largest bound at most its width, as the compiler takes it)."""
    if not GIVEN:
        return
    for u in units:
        if u.get('link') or u.get('fixed') or u['store'] == 'pages':
            continue
        model = re.sub(r'_copy\d+$', '', u['class'])
        if u['store'].startswith('regs>='):
            w, kind = int(u['store'][6:]), None
            for bound, k in sorted(GIVEN['regs'].get(model, [])):
                if bound <= w:
                    kind = k
        else:
            kind = GIVEN['mem'].get((model, u['store']))
        if kind is not None:
            u['fixed'] = kind
            u['given'] = True


def state_units(units, lev=1.0):
    """The units' costs with STATE_CAL's factors: a store's LUTs by its
    class's logic factor, its LUTRAM by its LUTRAM factor; a link's by the
    level's (copies of the cost dicts)."""
    if not STATE_CAL:
        return
    for u in units:
        fl, fr = (lev, lev) if u.get('link') else state_factors(u['class'])

        def scale(c, k):
            if c is None or k == 'hbm' and not u.get('link'):
                # Off chip: the controller's own price (ctl_cost, routed).
                return c
            return dict(c, lut=c.get('lut', 0.0) * fl, lutram=c.get('lutram', 0.0) * fr)
        u['costs'] = {k: scale(c, k) for k, c in u['costs'].items()}
        if 'per' in u:
            u['per'] = {k: [(j, scale(c, k)) for j, c in v] for k, v in u['per'].items()}


def copy_cap_pools(units, caps, base, copy_logic=None):
    """COPY_CAP as pools of storage.assign: for each copy j of a class c, its
    stores' URAM (`uram@c#j`) and RAMB36 (`bram@c#j`) at each kind, from the
    units' per-copy costs, each pool capped. With the cap's `lut` (and
    `lutram`), the copy's LUTs too (`lut@c#j`: its stores' LUTs and LUTRAM at
    each kind, on a base of the copy's own logic, `copy_logic`; `lutram@c#j`:
    the LUTRAM alone, within the SLR's SLICEMs), so that a copy whose LUTRAM
    stores outgrow an SLR moves them to block or ultra RAM or off chip. Links
    (placed at the producer, the simulator's level) are not a copy's."""
    if not COPY_CAP:
        return
    for u in units:
        if u.get('link'):
            continue
        for kind, parts in u.get('per', {}).items():
            if kind not in u['costs'] or kind in storage.RELIEF_KINDS:
                # Off chip: the store's controller is the kernel's, beside
                # the simulator, not the copy's.
                continue
            for j, pc in parts:
                amounts = {res: pc.get(res, 0.0) for res in ('uram', 'bram')}
                if COPY_CAP.get('lut') is not None:
                    amounts['lut'] = pc.get('lut', 0.0) + pc.get('lutram', 0.0)
                    amounts['lutram'] = pc.get('lutram', 0.0)
                if COPY_CAP.get('ff') is not None:
                    # The stores' flops (register stores, read buffers' read
                    # registers); a staged copy's stage registers are not
                    # priced here (foldsim_sll_check.py counts them on the
                    # netlist).
                    amounts['ff'] = pc.get('ff', 0.0)
                for res, x in amounts.items():
                    key = f"{res}@{u['class']}#{j}"
                    if res == 'lut':
                        # Every copy with a store gets its LUT pool, its own
                        # logic the base: a copy over on logic alone is named.
                        caps[key] = COPY_CAP[res]
                        base.setdefault(key, (copy_logic or {}).get((u['class'], j), 0.0))
                    if x:
                        u['costs'][kind][key] = u['costs'][kind].get(key, 0.0) + x
                        caps[key] = COPY_CAP[res]
                        base.setdefault(key, 0.0)


def read_built(path):
    """BUILT from a foldsim_sim.json (and its foldsim_sim.vh /
    foldsim_kernel.vh): dict(controllers=[...], logic, lutram, bram, forward,
    platform, counts)."""
    j = json.load(open(path))
    d = os.path.dirname(os.path.abspath(path))
    vh = ''.join(open(os.path.join(d, f)).read() for f in ('foldsim_sim.vh', 'foldsim_kernel.vh')
                 if os.path.exists(os.path.join(d, f)))
    m = re.search(r'FOLDSIM_VMEM_FORWARD (\d)', vh)
    forward = int(m.group(1)) if m else int(j.get('vmem_forward', 1))
    m = re.search(r'FOLDSIM_VMEM_MASTERS (\d+)', vh)
    vmasters = int(m.group(1)) if m else 0 if 'FOLDSIM_VMEM_ON_PORTS' in vh else None
    # The memory ports' masters (FOLDSIM_MEM_MASTERS: one where the ports are
    # folded); a kernel.vh before it bound one per port.
    m = re.search(r'FOLDSIM_MEM_MASTERS (\d+)', vh) or re.search(r'FOLDSIM_MEM_PORTS (\d+)', vh)
    ports = int(m.group(1)) if m else None
    ctls = j.get('vmem_groups') or j.get('vmem') or []
    save = storage.CTL_FORWARD
    storage.CTL_FORWARD = forward
    out = dict(controllers=[], logic=0.0, lutram=0.0, bram=0.0, forward=forward, mem_ports=ports,
               vmem_masters=vmasters, counts={})
    for c in ctls:
        cls = str(c.get('class', ''))
        kind = 'link' if cls.startswith('__link') else 'engine' if cls.startswith('__pages') else 'vmem'
        kz = 0
        if c.get('known_zero'):
            kz = -(-((1 << c['row_width']) * c['instances']) // (1 << c.get('kz_shift', 0)))
        x = storage.ctl_cost(kind, c['data_width'], c.get('cache_rows', 0) or 0, c.get('size', 1) or 1,
                             kz=kz)
        out['controllers'].append(dict(kind=kind, dw=c['data_width'], cache_rows=c.get('cache_rows', 0), **x))
        out['counts'][kind] = out['counts'].get(kind, 0) + 1
        for key in ('logic', 'lutram', 'bram'):
            out[key] += x[key]
    # Static ports (vport-engines: foldsim_sim.json `vports`, after the controllers above in this order): each a
    # foldsim_vport in the kernel, priced as a page engine's controller of its row width (kind 'vport'; no fit of
    # its own yet, and an engine's carries the drop-in controller too: an upper bound).
    for v in j.get('vports') or []:
        x = storage.ctl_cost('engine', v['DW'])
        out['controllers'].append(dict(kind='vport', dw=v['DW'], cache_rows=0, **x))
        out['counts']['vport'] = out['counts'].get('vport', 0) + 1
        for key in ('logic', 'lutram', 'bram'):
            out[key] += x[key]
    storage.CTL_FORWARD = save
    return out


def read_util(path):
    """Measured LUTs of each class's copies (logic, distributed RAM) and of the
    rest, from a hierarchical utilization report of a foldsim_afu build."""
    copies, rest = {}, 0
    for line in open(path):
        f = line.split('|')
        if len(f) < 7 or not line.startswith('|'):
            continue
        name = f[1].rstrip()
        depth = len(name) - len(name.lstrip())
        try:
            lut, logic, lutram = int(f[3]), int(f[4]), int(f[5])
        except ValueError:
            continue
        name = name.strip()
        if depth == 1:
            total = lut
            totals = dict(logic=logic, lutram=lutram)
        if depth != 5:
            continue
        m = re.match(r'(.+?)(_copy\d+)?_copy$', name)
        if m:
            c = copies.setdefault(m.group(1), dict(copies=0, logic=0, lutram=0, lut=0))
            c['copies'] += 1
            c['logic'] += logic
            c['lutram'] += lutram
            c['lut'] += lut
    rest = total - sum(c['lut'] for c in copies.values())
    return copies, rest, total, totals


def read_wns(path):
    text = open(path).read()
    m = re.search(r'Design Timing Summary.*?\n(?:.*\n){5}\s*(-?\d+\.\d+)', text)
    return float(m.group(1))


def calibrate(a):
    text = open(a.schedule).read()
    ii, classes, _ = parse_schedule(text)
    sched = parse_classes(text)
    entries = a.register_state_entries
    est = class_estimates(a.models, classes, a.foldsim_opt)
    copies, rest, total, totals = read_util(a.util)
    period = a.period - read_wns(a.timing)
    cal = json.load(open(a.out)) if os.path.exists(a.out) else {}
    build = dict(name=a.name, estimator=ESTIMATOR, luts=total, rest=rest, period=period,
                 classes={}, **totals)
    if a.screen:
        build['depth'] = screen_depth(a.screen)
    for name, n in classes.items():
        base = re.sub(r'_copy\d+$', '', name)
        m = copies.get(base)
        if not m or name != base:
            continue
        k = m['copies']
        t = n
        e = est[name]
        form, bypass = state_form(t, sched.get(name), ii, entries)
        lr, _ = state_lutram(e, t, form)
        sl = state_logic(cal, form, bypass, e['ff'])
        build['classes'][name] = dict(
            copies=k, threads=t, logic=m['logic'] / k, lutram=m['lutram'] / k,
            est_logic=e['lut'], est_lutram=lr, form=form, state_logic=sl,
            kappa=(max(m['logic'] / k - sl - e.get('lut_ip', 0.0), 0) / e['lut']
                   if e['lut'] else e.get('scale', 1.0)),
            lam=(m['lutram'] / k) / lr if lr else 1.0)
    build['max_copy'] = max(m['lut'] / m['copies'] for m in copies.values())
    cal.setdefault('builds', [b for b in cal.get('builds', []) if b['name'] != a.name])
    cal['builds'] = [b for b in cal['builds'] if b['name'] != a.name] + [build]
    json.dump(cal, open(a.out, 'w'), indent=1)
    for name, c in build['classes'].items():
        print(f"{a.name} {name}: logic {c['logic']:.0f} (est {c['est_logic']:.0f}, "
              f"x{c['kappa']:.2f}), lutram {c['lutram']:.0f} (est {c['est_lutram']:.0f}, "
              f"x{c['lam']:.2f})")
    print(f"{a.name}: {total} LUTs, rest {rest}, period {period:.3f} ns, "
          f"largest copy {build['max_copy']:.0f}")


def area_builds(cal):
    """The builds whose area factors were fitted on the estimator in use (ESTIMATOR)."""
    return [b for b in (cal or {}).get('builds', []) if build_estimator(b) == ESTIMATOR]


def retired_builds(cal):
    """The builds whose area factors were fitted on another estimator."""
    return [b['name'] for b in (cal or {}).get('builds', []) if build_estimator(b) != ESTIMATOR]


def factors(cal, name, est_lut, scale=1.0):
    """A class's calibration: from the build whose estimate of it is nearest
    (the same design at another degree, else the nearest size), the latest
    such build of a design built more than once (the compiler it reflects),
    else the median over every class and build, else Beacon's device scale
    for the LUTs. Only builds fitted on Beacon's area count (area_builds)."""
    builds = area_builds(cal)
    near = [(round(abs(math.log(c['est_logic'] / est_lut)), 6), -i, c)
            for i, b in enumerate(builds) for n, c in b['classes'].items()
            if n == name and c['est_logic'] and est_lut]
    if near:
        c = min(near, key=lambda x: x[:2])[2]
        return c['kappa'], c['lam']
    ks = sorted(c['kappa'] for b in builds for c in b['classes'].values())
    ls = sorted(c['lam'] for b in builds for c in b['classes'].values())
    if ks:
        return ks[len(ks) // 2], ls[len(ls) // 2]
    return scale, 1.0


def rest_of(cal, est_total):
    """The LUTs outside the copies (memory endpoint, links, control) of the
    build nearest in estimated size (of those fitted on Beacon's area), else
    the median over every build (measured, whatever the estimator)."""
    builds = area_builds(cal)
    if not builds:
        rests = sorted(b['rest'] for b in (cal or {}).get('builds', []))
        return rests[len(rests) // 2] if rests else 0.0
    def size(b):
        return sum(c['est_logic'] * c['copies'] for c in b['classes'].values())
    return min(builds, key=lambda b: abs(math.log(max(size(b), 1) / max(est_total, 1))))['rest']


def area(cal, est, classes, degree, rest, sched=None, ii=0, entries=2, store='fixed',
         route=False, uram=True, slr_groups=False, vmem=None, tiers=None):
    """LUTs of one fold degree ({class: copies}, 1 when absent): logic and
    distributed RAM apart, the sites they take, and the largest copy. With the
    degree's schedule report (parse_classes) each copy's state takes the form
    the compiler gives it (state_form). `store`: 'fixed', the compiler's
    default storage (block RAM by Vivado's inference, the rest distributed),
    against the whole device's RAMB36; 'auto', a kind per store
    (storage_area) against the V80 dynamic region. `route`: whether a point
    must also route (the routability term, foldsim_storage.congestion).
    `uram`: whether the storage may use UltraRAM. `slr_groups`: whether each
    copy is kept whole in one SLR (foldsim --slr-groups), which confines the
    fold to as many SLRs as it has copies. Virtual memories (split_virtual)
    are off chip; their host caches (`vmem`, vmem_bram) take block RAM.
    The kernel's controllers' logic is not priced. `tiers` (store 'auto'
    only): the off-chip tier (storage_area), as choose_state gives it."""
    vg = (tiers or {}).get('vmem_group')
    rbm = (tiers or {}).get('row_bytes_min', 1)
    cache = vmem_bram(est, classes, degree, vmem, vg, rbm) if BUILT is None else 0.0
    slrs = storage.route_slrs(sum(degree.get(c, 1) for c in classes) if slr_groups else 0)
    if store == 'auto':
        res = storage_area(cal, est, classes, degree, rest, sched, ii, entries, route, uram,
                           slrs, cache, tiers)
        if tiers is not None and res.get('vmem_new') and vmem:
            # The host caches of the memories it moved off chip: priced again
            # with them in the base (their lines are block RAM).
            more = vmem_bram(split_virtual(est, tiers_pattern(res['vmem_new'])), classes,
                             degree, vmem, vg, rbm)
            if more > cache:
                cache = more
                res = storage_area(cal, est, classes, degree, rest, sched, ii, entries, route,
                                   uram, slrs, cache, dict(tiers, force=res['vmem_new'],
                                                           force_pages=res['pages_offchip'],
                                                           force_rows=res['rows_new']))
    else:
        res = fixed_area(cal, est, classes, degree, rest, sched, ii, entries, cache)
    if cache:
        res['vmem_cache_bram'] = round(cache, 1)
    if route:
        c = storage.fold_congestion(res['luts'], res['bram'], res.get('uram', 0), slrs)
        res['congestion'] = round(c, 2)
        res['routes'] = c <= storage.ROUTE_LIMIT
    return res


# A paged class's memories of at most this many bits page as rows with its
# registers (foldsim-fold-static page-memory-bits); --state auto may page
# larger ones, up to ROW_BITS_MAX, where that is cheaper (storage_area).
PAGE_MEMORY_BITS = 4096
ROW_BITS_MAX = 1 << 16


def paged_state(est):
    """A paged copy's state as foldsim-fold-static pages it: its registers'
    bits and its small memories' rows (and a registered read's output
    register); the read ports' choice of rows in LUTs (a flop store's read,
    foldsim_storage.kind_cost 'registers'); and its other memories, which
    keep their stores. Registers a ready pass reads are not known here: all
    page."""
    bits, luts, rest = est['ff'], 0.0, []
    for m in est['mems']:
        (_, d, w, r, _, rl, *_) = m
        if rl <= 1 and d * w <= PAGE_MEMORY_BITS:
            bits += d * w + (w * r if rl == 1 else 0)
            luts += max(r, 1) * w * math.ceil((d - 1) / 3) if d > 1 else 0
        else:
            rest.append(m)
    return bits, luts, dict(est, mems=rest)


def paged_fields(paged):
    """The paged register state of a point, for its row: bits (every
    instance's), the flops of the fetch and write-back buffers, and the
    page RAMs' 72-bit read ports."""
    if not paged['bits']:
        return {}
    out = dict(paged_bits=paged['bits'], paged_ff=paged['ff'], page_ports=paged['ports'])
    if paged.get('hbm'):
        out.update(hbm_bytes=paged['hbm'],
                   hbm_ii=math.ceil(paged['hbm'] / storage.HBM_BYTES_PER_CYCLE))
    return out


def fixed_area(cal, est, classes, degree, rest, sched=None, ii=0, entries=2, cache=0.0):
    logic = lutram = largest = 0.0
    bram = cache
    paged = dict(bits=0, ff=0, ports=0, hbm=0)
    for c, n in classes.items():
        k = degree.get(c, 1)
        kappa, lam = factors(cal, c, est[c]['lut'], est[c].get('scale', 1.0))
        for j in range(k):
            t = copy_threads(n, k, j)
            info = (sched or {}).get(c if j == 0 else f'{c}_copy{j}')
            form, bypass = state_form(t, info, ii, entries) if sched is not None else \
                ('one' if t == 1 else 'lutram', False)
            lr, br = state_lutram(est[c], t, form)
            copy_logic = (kappa * est[c]['lut'] + est[c].get('lut_ip', 0.0)
                          + state_logic(cal, form, bypass, est[c]['ff']))
            if form == 'paged':
                bits, mux, others = paged_state(est[c])
                lr, br = state_lutram(others, t, form)
                pc = storage.page_cost(bits, t, info['page'],
                                       'hbm' if info.get('offchip') else 'block')
                copy_logic += pc['lut'] + mux
                br += pc['bram']
                paged['ff'] += pc['ff']
                paged['ports'] += pc['ports']
                paged['bits'] += bits * t
                paged['hbm'] += pc.get('hbm_bytes', 0)
            logic += copy_logic
            lutram += lam * lr
            bram += br
            largest = max(largest, copy_logic + lam * lr)
    logic += rest
    lk = link_cost(cal, (sched or {}).get('__links__'))
    lutram += lk['lutram']
    bram += lk['bram']
    sites, fits = placement(logic, lutram, bram)
    return dict(luts=round(logic + lutram), logic=round(logic), lutram=round(lutram),
                sites=round(sites), fits=fits, bram=round(bram, 1), largest_copy=round(largest),
                **paged_fields(paged),
                uram=lk['uram'], link_lutram=round(lk['lutram']), link_bram=round(lk['bram'], 1),
                link_uram=lk['uram'], link_ff=lk['ff'])


# The kernel and the simulator's own level, which the class estimates do not
# see (st32_1031/lk2/img4dbg/FINDINGS.md, cost_terms.md: the 4-SM A100-class
# image, Vivado 2025.2 synthesis on the V80): each memory endpoint of the
# kernel (V7B foldsim_mem, its DRAM bank model; default synthesis), the
# simulator top level's logic per link register bit (read registers and
# selects, staged copies, slot tables), a threaded copy's thread controls,
# and whether memories the model gives ultra RAM get it (Vivado put them in
# block RAM); the simulator's channel queues of each memory port (census of
# the top level: 3.5k LUTs a port, 0.27 a link register bit besides). On with --kernel-terms (KERNEL['on']); `mem_ports` from the
# report (the netlist's m_axi_mem ports).
KERNEL = dict(on=False, mem_ports=0, endpoint_logic=24565.0, endpoint_lutram=4718.0,
              port_channel_logic=3500.0, top_logic_per_link_bit=0.27, threaded_copy_logic=1500.0,
              uram_memories=False, fold_ports=False, folded_logic=40000.0, folded_lutram=5000.0,
              folded_uram_bits_per_port=330000.0, folded_measured=None, copy_logic=None,
              calibration=None,
              sim_ports_folded=False, sim_port_folded_logic=1100.0,
              sim_ports_shared_logic=8200.0, sim_ports_shared_lutram=1700.0)
# sim_ports_folded (foldsim --fold-ports, built): the simulator's side of the
# memory ports as request / response record streams. Per port what stays:
# the inputs' tokens and frame registers (rdata's 4 copies and its frame:
# ~1.0k LUTs at 4 SMs, census_cells.txt) and the outputs' counters and
# pointers (~0.1k); shared: the record memories' write selects (a 32-way
# AND-OR a request bit, ~11 LUTs a bit of 744) and their LUTRAM (744 bits x
# 2 blocks of 64 rows). Estimates from the 4-SM census and the generated
# netlist (foldsim_cost: -42.7k raw at 4 SMs), not yet a Vivado point.
# fold_ports (a what-if, not built): the memory ports as one replicated unit
# folded like a class: one bank-model engine, one endpoint shell and one
# token-queue engine (`folded_logic`, `folded_lutram`) serving the channels in
# turn, each channel's state (calendars, bank table, queues, the shell's
# buffers: ~4.7k LUTRAM + 7.7k flops + 2.9k LUTRAM measured a port) in ultra
# RAM. copy_logic: {class model: synthesized logic LUTs of its copy} (measured,
# st32_1031/lk2/img4dbg/class_table.md) in place of kappa x estimate + state
# logic and the threaded-copy term. folded_measured (--fold-ports-measured): the
# folded memory endpoint as built and synthesized out of context (foldsim_mem_fold,
# lk2/ooc): {logic, lutram, ramb36, uram} for all the ports, in place of the
# what-if's folded_* estimates.


def top_link_logic(links):
    """The simulator top level's logic the links carry. A calibration with
    `top_logic_terms` prices it per driver: staged link register bits
    (`staged`: a staged register loads at its producer's pass, an enable and
    no select), the other link register bits (`reg`) and the read-select bits
    of reader inputs with several sources (`select`: width x (sources - 1)
    over the link uses); else one ratio on all link register bits
    (`top_logic_per_link_bit`)."""
    links = links or {}
    cal = KERNEL['calibration']
    terms = cal.get('top_logic_terms') if cal else None
    if terms:
        staged = links.get('staged_ff', 0)
        return (terms['staged'] * staged + terms['reg'] * (links.get('ff', 0) - staged) +
                terms['select'] * links.get('select_bits', 0))
    tlb = cal['top_logic_per_link_bit'] if cal else KERNEL['top_logic_per_link_bit']
    return tlb * links.get('ff', 0)


# A whole-design calibration (--calibrate <file>, calibrations/*.json): per class
# family (a regex on the class model's name, first match) a factor on its copy's
# logic and store select logic and one on its stores' LUTRAM; the top level's
# logic per link register bit; a factor on the links' LUTRAM; and fixed kernel
# terms (vmem controllers, the control endpoint) the model has no estimate of.
# Factors are synthesized / priced x SYNTH_PER_ROUTED, fitted on one whole-design
# synthesis (the file says which). The families are a design's own fit, data
# of its calibration file (calibrations/vortex_families.json names Vortex's);
# a class no family names takes the calibration's `generic` model of its cost
# features (class_features; calibrations/fit_generic.py), if it has one, else 1.
def cal_factors(c):
    cal = KERNEL['calibration']
    if not cal:
        return 1.0, 1.0
    model = re.sub(r'_copy\d+$', '', c)
    for fam in cal.get('families', []):
        if re.search(fam['match'], model):
            return fam['logic'], fam['lutram']
    gen = cal.get('generic')
    if gen and KERNEL.get('models'):
        if KERNEL.get('features') is None:
            KERNEL['features'] = {}
        if model not in KERNEL['features']:
            KERNEL['features'].update(class_features(KERNEL['models'], [model]))
        f = KERNEL['features'].get(model)
        if f:
            return generic_factor(gen['logic'], f), generic_factor(gen['lutram'], f)
    return 1.0, 1.0


def cal_lutram_add(c):
    """LUTRAM a copy of the class has that no store prices (calibrated)."""
    cal = KERNEL['calibration']
    if not cal:
        return 0.0
    model = re.sub(r'_copy\d+$', '', c)
    for fam in cal.get('families', []):
        if re.search(fam['match'], model):
            return fam.get('lutram_add', 0.0)
    return 0.0


def calibrate_units(units):
    """The units' costs with the calibration's factors (copies of the cost
    dicts, so a unit shared between calls is not scaled twice)."""
    cal = KERNEL['calibration']
    if not cal:
        return
    for u in units:
        if u.get('link'):
            fl, fr = 1.0, cal['link_lutram']
        else:
            fl, fr = cal_factors(u['class'])
        def scale(c, k):
            if c is None:
                return c
            # A store in HBM costs its virtual-memory controller, priced
            # routed (storage.ctl_cost): the calibration's controller factors
            # (fitted against the cache-less OOC line) no longer apply.
            gl, gr = (1.0, 1.0) if k == 'hbm' else (fl, fr)
            c = dict(c)
            c['lut'] = c.get('lut', 0.0) * gl
            c['lutram'] = c.get('lutram', 0.0) * gr
            return c
        u['costs'] = {k: scale(c, k) for k, c in u['costs'].items()}
        if 'per' in u:
            u['per'] = {k: [(j, scale(c, k)) for j, c in v] for k, v in u['per'].items()}


def kernel_mem_ports(text):
    """The netlist's memory ports (m_axi_mem_<k>) a schedule report names."""
    ks = {int(k) for k in re.findall(r'\bm_axi_mem_(\d+)_', text)}
    return max(ks) + 1 if ks else 0


def storage_area(cal, est, classes, degree, rest, sched, ii, entries, route, uram=True,
                 slrs=None, cache=0.0, tiers=None):
    """area() with a kind chosen per store (foldsim_storage.assign) against
    the V80 dynamic region: its sites, RAMB36 and URAM288 and, with `route`,
    the LUTs that route over `slrs` SLRs (the region's by default). Every
    copy of a class takes its stores' kinds (the compiler's `storage` rules
    are per class); a class's register stores choose by width, and only
    where every copy of several instances keeps them in distributed RAM (no
    flops or ring) with a state gap of 4.

    `tiers` (choose_state): the off-chip tier as relief kinds (foldsim_storage
    RELIEF_KINDS): a memory a virtual memory may hold (vmem_eligible in every
    copy) may go to HBM (vmem_cost), a paged copy's pages too (pages=hbm),
    against an `hbm` pool of HBM_STATE_SHARE of the frame's HBM bytes
    (`frame` host cycles at HBM_BYTES_PER_CYCLE) less the traffic of the
    memories already virtual (`base_hbm`). Keys: frame, base_hbm, weights
    ({memory name or <class>:<name>: accesses per instance and target cycle,
    default 1}), row_bytes_min, pages_hbm (whether pages may leave), force /
    force_pages (moves already chosen, kept). The result names the memories
    moved (`vmem_new`: (class, memory) pairs) and the classes whose pages
    left (`pages_offchip`), with their traffic (`hbm_bytes`)."""
    # A calibration's class families were fitted on whole copies: the copies'
    # thread controls and the estimate of the rest are inside them (`rest`).
    logic = rest * (KERNEL['calibration'].get('rest', 1.0) if KERNEL['on'] and KERNEL['calibration'] else 1.0)
    units, copy_logic, base_lutram = [], {}, 0.0
    # The state-home policy's calibration (STATE_CAL): the level's factor on
    # the rest and the links, the kernel and glue beside the simulator.
    lev = STATE_CAL.get('level', 1.0) if STATE_CAL else 1.0
    logic *= lev
    if STATE_CAL:
        logic += STATE_CAL.get('kernel_logic', 0.0)
        base_lutram += STATE_CAL.get('kernel_lutram', 0.0)
    kernel_uram = kernel_bram = 0.0
    if KERNEL['on']:
        if KERNEL['fold_ports'] and KERNEL['folded_measured']:
            fm = KERNEL['folded_measured']
            logic += fm['logic']
            base_lutram += fm['lutram']
            kernel_bram, kernel_uram = fm['ramb36'], fm['uram']
        elif KERNEL['fold_ports']:
            logic += KERNEL['folded_logic']
            base_lutram += KERNEL['folded_lutram']
            kernel_uram = math.ceil(KERNEL['mem_ports'] * KERNEL['folded_uram_bits_per_port'] /
                                    (storage.URAM[0] * storage.URAM[1]))
        else:
            channel = KERNEL['sim_port_folded_logic'] if KERNEL['sim_ports_folded'] \
                else KERNEL['port_channel_logic']
            logic += KERNEL['mem_ports'] * (KERNEL['endpoint_logic'] + channel)
            base_lutram += KERNEL['mem_ports'] * KERNEL['endpoint_lutram']
            if KERNEL['sim_ports_folded']:
                logic += KERNEL['sim_ports_shared_logic']
                base_lutram += KERNEL['sim_ports_shared_lutram']
        logic += top_link_logic((sched or {}).get('__links__'))
        if KERNEL['calibration']:
            fx = KERNEL['calibration']['fixed']
            logic += fx['logic']
            base_lutram += fx['lutram']
            kernel_bram += fx['ramb36']
        if not KERNEL['copy_logic'] and not KERNEL['calibration']:
            logic += KERNEL['threaded_copy_logic'] * sum(1 for c, n in classes.items() if n > 1)
    page_of = {}
    n_ctl = 0
    paged = dict(bits=0, ff=0, ports=0, hbm=0)
    # With page engines: each off-chip paged copy's (columns, engine share).
    engine_cols = []
    for c, n in classes.items():
        k = degree.get(c, 1)
        kappa, lam = factors(cal, c, est[c]['lut'])
        forms = []
        for j in range(k):
            t = copy_threads(n, k, j)
            info = (sched or {}).get(c if j == 0 else f'{c}_copy{j}')
            form, bypass = state_form(t, info, ii, entries) if sched is not None else \
                ('one' if t == 1 else 'lutram', False)
            cl = kappa * est[c]['lut'] + state_logic(cal, form, bypass, est[c]['ff'])
            if KERNEL['on'] and KERNEL['copy_logic'] and \
                    re.sub(r'_copy\d+$', '', c) in KERNEL['copy_logic']:
                cl = float(KERNEL['copy_logic'][re.sub(r'_copy\d+$', '', c)])
            elif KERNEL['on']:
                cl *= cal_factors(c)[0]
                base_lutram += cal_lutram_add(c)
            cl *= state_factors(c)[0]
            logic += cl
            copy_logic[(c, j)] = cl
            forms.append((j, t, form, info['gap'] if info else None))
            if form == 'paged':
                page_of[(c, j)] = info['page']
            if form == 'paged' and info.get('offchip'):
                bits, mux, _ = paged_state(est[c])
                pc = storage.page_cost(bits, t, info['page'], 'hbm')
                # Its controllers (a column each) too; with page engines, its
                # share of them (counted again for the fold below).
                if tiers is not None and tiers.get('page_engines'):
                    eng = storage.page_hbm_cost(bits, t, info['page'], tiers['frame'])
                    cl, cr = eng['lut'] - pc['lut'], eng['lutram'] - pc['lutram']
                    engine_cols.append((eng['engine_columns'], eng['ctl']))
                else:
                    cl, cr = storage.page_ctl(bits, info['page'], pc['controllers'])
                    n_ctl += pc['controllers']
                if BUILT is not None:
                    cl = cr = 0.0
                logic += pc['lut'] + mux + cl
                base_lutram += cr
                copy_logic[(c, j)] += pc['lut'] + mux + cl
                paged['ff'] += pc['ff']
                paged['bits'] += bits * t
                paged['hbm'] += pc['hbm_bytes']
            elif form == 'paged':
                # A page RAM per paged copy: block or ultra, as the rest choose.
                bits, mux, _ = paged_state(est[c])
                logic += mux
                copy_logic[(c, j)] += mux
                pcost = {kind: storage.page_cost(bits, t, info['page'], kind)
                         for kind in ('block', 'ultra')}
                # The frame its state passes need is the schedule's (`ii`).
                pcost = {k: dict(v, lutram=0.0, ii_floor=0) for k, v in pcost.items()}
                if tiers is not None and (tiers.get('pages_hbm') or
                                          c in tiers.get('force_pages', ())):
                    # On engines, off chip at its own depth (page_hbm_depth:
                    # few columns, so few engines).
                    ph = storage.page_hbm_depth(bits, t, info['page'], tiers['frame'],
                                                PAGE_PORT_SHARE) \
                        if tiers.get('page_engines') else info['page']
                    pcost['hbm'] = dict(storage.page_hbm_cost(
                                            bits, t, ph,
                                            tiers['frame'] if tiers.get('page_engines') else None),
                                        ii_floor=0, page=ph, bubbles=t * max(
                                            0, storage.HBM_ROUND_TRIP + 7 - tiers['frame'] // t))
                units.append(dict(costs=pcost,
                                  per={k: [(j, v)] for k, v in pcost.items()}, count=1,
                                  fixed=info.get('page_kind') if BUILT is not None else None,
                                  store='pages', default='block',
                                  **{'class': c}))
                paged['ff'] += pcost['block']['ff']
                paged['ports'] += pcost['block']['ports']
                paged['bits'] += bits * t
        # The controllers of its memories already virtual (the pattern's), one
        # a memory of each copy (a row size of a copy with --vmem-group
        # copy), in the kernel; built (BUILT) the generation's own instead.
        if BUILT is None:
            for j in range(k):
                t = copy_threads(n, k, j)
                for v in vmem_controllers(est[c].get('vmems', []), t,
                                          (tiers or {}).get('vmem_group'),
                                          (tiers or {}).get('row_bytes_min', 1)):
                    x = storage.ctl_cost('vmem', v['data_width'], 0, v['members'],
                                         (1 << v['row_width']) * v['instances'])
                    logic += x['logic']
                    base_lutram += x['lutram']
                    kernel_bram += x['bram']
                    n_ctl += 1
        deep = [f for f in forms if f[2] == 'lutram']
        regs_free = all(f[2] in ('lutram', 'one') for f in forms)
        for w, cnt in sorted(est[c]['regw'].items()):
            per = []
            for j, t, _, gap in deep:
                per.append((j, dict(depth=1 << (t - 1).bit_length(), width=w, rports=1, wports=1,
                                    rlat=0, mask=1, reg=True, gap=gap, threads=t), cnt))
            if per:
                units.append(unit(c, f'regs>={w}', per, lam, None if regs_free else 'distributed'))
        mems = {}
        for (name, d, w, r, wr, rl, st, mask, named, *_) in est[c]["mems"]:
            mems.setdefault((name, named, d, w, r, wr, rl, st, mask), 0)
            mems[(name, named, d, w, r, wr, rl, st, mask)] += 1
        for (name, named, d, w, r, wr, rl, st, mask), cnt in mems.items():
            per = []
            for j, t, f, gap in forms:
                # A paged copy's small memories are rows of its pages.
                if f == 'paged' and rl <= 1 and d * w <= PAGE_MEMORY_BITS:
                    continue
                th = 1 << (t - 1).bit_length() if t > 1 else 1
                per.append((j, dict(depth=d * th, width=w, rports=r, wports=wr, rlat=rl,
                                    mask=mask, reg=False, gap=gap, threads=t), cnt))
            if not per:
                continue
            # Vivado's inference at the deepest copy's depth, or the hint.
            th = max(1 << (t - 1).bit_length() if t > 1 else 1 for _, t, _, _ in forms)
            default = 'block' if cost.mem_cost(d * th, w, r, rl, st)[1] else 'distributed'
            # A memory the compiler's rules can name: named, and its name
            # free of the pass options' separators.
            nameable = named and not re.search(r'[,=:{}\s]', name)
            u = unit(c, name, per, lam, None if nameable else default)
            u['default'] = default
            if tiers is not None and not tiers.get('as_built') and nameable and \
                    len(per) == len(forms) and all(
                    not storage.vmem_eligible((name, d, w, r, wr, rl, st, mask, named), gap)
                    for _, _, _, gap in forms):
                weights = tiers.get('weights') or {}
                wt = float(weights.get(f'{c}:{name}', weights.get(name, 1.0)))
                hc = dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0, ii_floor=0, hbm=0.0, ctl=0)
                parts = []
                for j, t, _, fgap in forms:
                    v = storage.vmem_cost(d, w, t, wt, tiers.get('row_bytes_min', 1))
                    cr = v['lutram'] - v.get('ctl_lutram', 0.0)
                    v = dict(v, lutram=lam * cr + v.get('ctl_lutram', 0.0))
                    if tiers.get('vmem_group') == 'copy':
                        # foldsim --vmem-group copy: one controller for a
                        # copy's memories of a row size, its LUTs and block
                        # RAM shared by those the class may move.
                        rb = storage.vmem_row_bytes(w, tiers.get('row_bytes_min', 1))
                        share = max(1, sum(1 for m in est[c]['mems']
                                           if storage.vmem_row_bytes(m[2], tiers.get('row_bytes_min', 1))
                                           == rb and not storage.vmem_eligible(m, fgap)))
                        v['lut'] = v['lut'] / share
                        v['lutram'] -= v.get('ctl_lutram', 0.0) * (1 - 1 / share)
                        v['bram'] -= v.get('ctl_bram', 0.0) * (1 - 1 / share)
                        v['ctl'] = 1 / share
                    v = {k: (x if k == 'ii_floor' else x * cnt) for k, x in v.items()}
                    parts.append((j, v))
                    for key in hc:
                        if key != 'ii_floor':
                            hc[key] += v[key]
                rtt = storage.HBM_ROUND_TRIP
                hc['bubbles'] = cnt * sum(t * max(0, rtt - gap) for _, t, _, gap in forms
                                          if gap is not None)
                u['costs']['hbm'] = hc
                u['per']['hbm'] = parts
                if (c, name) in tiers.get('force', ()):
                    u['fixed'] = 'hbm'
            # A larger memory of a paged class read combinationally or through
            # a register may page as rows too (foldsim-fold-static
            # page-memories): a kind priced as paged_state prices the small
            # ones, its page RAM a share of the class's (URAM).
            as_row = BUILT is not None and BUILT.get('rows') and \
                re.search(BUILT['rows'], re.sub(r'_copy\d+$', '', c) + ':' + name)
            if tiers is not None and (not tiers.get('as_built') or as_row) and nameable and \
                    len(per) == len(forms) and rl <= 1 and \
                    all(f == 'paged' for _, _, f, _ in forms) and \
                    d * w <= tiers.get('row_bits_max', ROW_BITS_MAX):
                rc = dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0.0, ii_floor=0)
                parts = []
                for j, t, _, _ in forms:
                    p = page_of[(c, j)]
                    bits = d * w + (w * r if rl == 1 else 0)
                    rows = (1 << (t - 1).bit_length()) * (1 << (p - 1).bit_length())
                    v = dict(lut=cnt * (max(r, 1) * w * math.ceil((d - 1) / 3) +
                                        bits / p * storage.page_select_luts(p)),
                             lutram=0.0, ff=cnt * 2 * bits, bram=0.0,
                             uram=cnt * bits / p / storage.URAM[1] * math.ceil(rows / storage.URAM[0]),
                             ii_floor=0)
                    parts.append((j, v))
                    for key in rc:
                        if key != 'ii_floor':
                            rc[key] += v[key]
                u['costs']['rows'] = rc
                u['per']['rows'] = parts
                if (c, name) in tiers.get('force_rows', ()) or as_row:
                    u['fixed'] = 'rows'
            units.append(u)
    if not uram:
        for u in units:
            u['costs'].pop('ultra', None)
    if KERNEL['on'] and not KERNEL['uram_memories']:
        # Memories (not pages, not links) in ultra RAM: Vivado inferred block
        # RAM for most of them; priced as it builds them.
        for u in units:
            if (not u.get('link') and u['store'] != 'pages' and len(u['costs']) > 1 and
                    u.get('fixed') != 'ultra'):
                u['costs'].pop('ultra', None)
                u['per'].pop('ultra', None)
    caps = storage.capacities()
    # The links' storage, as the compiler emits it (the schedule report's
    # link kinds), in the base usage the stores' kinds are chosen against.
    lk = link_cost(cal, (sched or {}).get('__links__'))
    # HBM link streams built (a schedule made with them): their logic, and
    # their traffic in the HBM pool and H.
    logic += lk['lut'] * lev
    built_hbm = lk['hbm']
    # With the report's link uses, a pair of models' links that block or
    # ultra RAM may hold is a unit of its own (the link tier); the rest stays
    # in the base.
    lu = link_units(cal, (sched or {}).get('__links__'), ii, tiers.get('link_pages', True),
                    tiers.get('link_hbm', False), tiers['frame'],
                    tiers.get('page_engines', False)) if tiers is not None else None
    if lu:
        units.extend(lu[0])
        lk = dict(lu[1], lutram=lu[1]['lutram'] + lk['lutram'] -
                  link_cost(cal, dict((sched or {}).get('__links__'), streams=[]))['lutram'])
    lk = dict(lk, uram=lk['uram'] + kernel_uram, bram=lk['bram'] + kernel_bram,
              lutram=lk['lutram'] * lev)
    base = dict(sites=logic + storage.LUTRAM_SITES * (lk['lutram'] + base_lutram),
                luts=storage.fold_load(logic + lk['lutram'] + base_lutram, cache + lk['bram'],
                                       lk['uram']),
                bram=cache + lk['bram'], uram=lk['uram'])
    if tiers is not None:
        caps['hbm'] = max(0.0, storage.HBM_STATE_SHARE * tiers['frame'] *
                          storage.HBM_BYTES_PER_CYCLE - tiers.get('base_hbm', 0.0) - built_hbm)
        caps['bubbles'] = max(0.0, storage.BUBBLE_SHARE * tiers['frame'] -
                              tiers.get('base_bubbles', 0.0))
        for u in units:
            if u['store'] == 'pages' and u['class'] in tiers.get('force_pages', ()):
                u['fixed'] = 'hbm'
    if BUILT is not None and BUILT.get('storage') is not None:
        # As built: each store at the kind the generation's --storage rules
        # gave it (a memory its rule's or Vivado's default; register stores
        # by width, the largest bound at most theirs, distributed below).
        mem_rules, reg_rules = {}, {}
        for rule in filter(None, BUILT['storage'].split(',')):
            key, _, kind = rule.rpartition('=')
            model, _, store = key.partition(':')
            if store.startswith('regs>='):
                reg_rules.setdefault(model, []).append((int(store[6:]), kind))
            else:
                mem_rules[(model, store)] = kind
        for u in units:
            if u.get('link') or u.get('fixed'):
                continue
            model = re.sub(r'_copy\d+$', '', u['class'])
            if u['store'].startswith('regs>='):
                w = int(u['store'][6:])
                kind = 'distributed'
                for bound, k in sorted(reg_rules.get(model, [])):
                    if bound <= w:
                        kind = k
            else:
                kind = mem_rules.get((model, u['store']), u.get('default'))
            if kind in u['costs']:
                u['fixed'] = kind
    calibrate_units(units)
    state_units(units, lev)
    fix_given(units)
    if KERNEL['calibration']:
        # The links' LUTRAM in the base (those no tier chooses) too.
        lk = dict(lk, lutram=lk['lutram'] * KERNEL['calibration']['link_lutram'])
        base = dict(base, sites=logic + storage.LUTRAM_SITES * (lk['lutram'] + base_lutram),
                    luts=storage.fold_load(logic + lk['lutram'] + base_lutram, cache + lk['bram'], lk['uram']))
    copy_cap_pools(units, caps, base, copy_logic)
    choice, used = storage.assign(units, base, caps,
                                  dict(luts=storage.route_luts(slrs)) if route else None,
                                  price_extra=tiers is not None)
    fits = all(used[p] <= caps[p] + 1e-9 for p in caps)
    lutram, uram = lk['lutram'] + base_lutram, lk['uram']
    bram = cache + lk['bram']
    per_copy = dict(copy_logic)
    store_lut, store_lutram = {}, {}
    vmem_ctl = dict(lut=0.0, lutram=0.0, bram=0.0)
    rules, floor = [], 0
    off = dict(vmem_new=[], pages_offchip=[], rows_new=[], hbm_bytes=0.0, controllers=0,
               bubbles=0.0, **({'link_hbm_bytes': built_hbm} if built_hbm else {}))
    link_rules = []
    for u, ch in zip(units, choice):
        (kind, _), = ch.items()     # one store a unit: never split
        if u.get('link'):
            c = u['costs'][kind]
            if kind == 'hbm':
                off['hbm_bytes'] += c['hbm']
                off['controllers'] += c['ctl']
                off.setdefault('links_hbm', []).append(u['store'])
                if 'engine_columns' in c:
                    engine_cols.append((1, c['ctl'], 'stream'))
                off['link_hbm_bytes'] = off.get('link_hbm_bytes', 0.0) + c['hbm']
            lk['ff'] += c['ff']
            lk['lutram'] += c['lutram']
            lk['bram'] += c['bram']
            lk['uram'] += c['uram']
            lutram += c['lutram']
            bram += c['bram']
            uram += c['uram']
            logic += c['lut']
            u['kind'] = kind
            link_rules.append(f"{u['store']}:{kind}")
            continue
        for j, c in u['per'][kind]:
            lutram += c['lutram']
            bram += c['bram']
            uram += c['uram']
            logic += c['lut']
            if u['store'] != 'pages':
                floor = max(floor, c['ii_floor'])
            if kind != 'hbm':
                # (Off chip, the controller is the kernel's: not the copy's.)
                per_copy[(u['class'], j)] += c['lut'] + c['lutram']
            if kind == 'hbm':
                for x in ('lut', 'lutram', 'bram'):
                    vmem_ctl[x] += c[x]
            else:
                store_lut[(u['class'], j)] = store_lut.get((u['class'], j), 0.0) + c['lut']
                store_lutram[(u['class'], j)] = store_lutram.get((u['class'], j), 0.0) + c['lutram']
            if kind == 'hbm':
                off['hbm_bytes'] += c['hbm']
                off['controllers'] += c['ctl']
        if kind == 'hbm':
            off['bubbles'] += u['costs']['hbm'].get('bubbles', 0.0)
        u['kind'] = kind
        if kind == 'hbm' and u['store'] == 'pages':
            off['pages_offchip'].append(u['class'])
            # The model's page off chip (a model's copies take one page: the
            # shallowest of theirs).
            model = re.sub(r'_copy\d+$', '', u['class'])
            ph = u['costs']['hbm'].get('page')
            if ph:
                off.setdefault('page_hbm', {})
                off['page_hbm'][model] = min(off['page_hbm'].get(model, ph), ph)
            if 'engine_columns' in u['costs']['hbm']:
                engine_cols.append((u['costs']['hbm']['engine_columns'], u['costs']['hbm']['ctl']))
            paged['ports'] -= u['costs']['block']['ports']
            paged['hbm'] += u['costs']['hbm']['hbm']
        elif kind == 'hbm':
            off['vmem_new'].append((u['class'], u['store']))
        elif kind == 'rows':
            off['rows_new'].append((u['class'], u['store']))
    # The compiler's rules name a model: a copy (<class>_copy<j>, its own class
    # here when the schedule lists it apart) takes its model's, the first
    # copy's choice for a store where they differ.
    taken = set()
    for c in classes:
        model = re.sub(r'_copy\d+$', '', c)
        for rule in storage_rules(model, [u for u in units if u['class'] == c]):
            key = rule.rsplit('=', 1)[0]
            if key not in taken:
                taken.add(key)
                rules.append(rule)
    # Page engines: the fold's, at least the shares' sum, whole (the shares are
    # priced already; the rest added), and as foldsim-fold-static assigns them
    # (assignPageEngines: streams first, each its own engine unless another
    # stream's requests leave its positions free; then each off-chip copy's
    # columns on a run of engines where its page traffic finds free positions,
    # around the streams): priced at most, every off-chip copy's columns
    # and every stream on an engine of its own (sharing is left to the
    # compiler, which takes fewer). Copies' page traffic falls at their state
    # passes, so the shares' sum (all this priced before) is far too few: the
    # 16-SM folds g16 / g16b (virtA_1105) built 23 engines for 27 page columns
    # and 14 streams, 24 for 16 and 14, where the shares' sum gave 4; the
    # columns alone (or the streams alone) 27 / 16.
    page_engines = 0
    member_luts = 0.0
    # A schedule built with page engines: at least as many as it has. A
    # schedule-only report lists the engines of its link streams alone (the
    # page columns are laid out with the copies): the pages' engines on top.
    built = (sched or {}).get('__links__') or {}
    built_engines = built.get('engines', 0)
    if (engine_cols or built_engines) and BUILT is None:
        share = sum(e[1] for e in engine_cols)
        cols = sum(e[0] for e in engine_cols if len(e) < 3)
        streams = sum(1 for e in engine_cols if len(e) == 3)
        member_luts = 0.0
        if ENGINES['rule'] == 'first-fit':
            # As the compiler assigns them (storage.engines_first_fit): the
            # page columns on the streams' engines where their traffic finds
            # room, the streams sharing under link-hbm-shift; each engine's
            # request select over its members (engine_member_luts).
            ff, members = storage.engines_first_fit(
                [(e[0], e[1] / max(e[0], 1)) for e in engine_cols if len(e) < 3],
                [e[1] for e in engine_cols if len(e) == 3], ENGINES['shift'], members=True)
            own = max([e[0] for e in engine_cols] + [math.ceil(share - 1e-9), ff])
            member_luts = sum(storage.engine_member_luts(m) for m in members)
            logic += member_luts
        else:
            own = max([e[0] for e in engine_cols] + [math.ceil(share - 1e-9), cols + streams])
        page_engines = max(own, built_engines) if built.get('engine_columns') else \
            own + built_engines
        el, er = storage.page_engine_ctl()
        logic += (page_engines - share) * el
        lutram += (page_engines - share) * er
        bram += (page_engines - share) * storage.PAGE_ENGINE_BRAM
    # The kernel's controllers and what they add to the platform's way into
    # device memory (storage.platform_cost: the masters' shares, SLASH's
    # sc_mem tree), with --kernel-terms: built, the generation's own (BUILT);
    # else those this point has (memories virtual already and moved, streams
    # with a controller of their own, page engines, off-chip page columns).
    platform = None
    if BUILT is not None:
        logic += BUILT['logic']
        lutram += BUILT['lutram']
        bram += BUILT['bram']
        n_ctl = len(BUILT['controllers'])
        page_engines = BUILT['counts'].get('engine', 0)
    else:
        chosen = [(u, next(iter(ch))) for u, ch in zip(units, choice)]
        n_ctl += sum(u['costs']['hbm'].get('ctl', 0) for u, kind in chosen
                     if kind == 'hbm' and not u.get('link') and
                     'engine_columns' not in u['costs']['hbm']) + \
            sum(1 for u, kind in chosen
                if kind == 'hbm' and u.get('link') and 'engine_columns' not in u['costs']['hbm']) + \
            link_cost(cal, (sched or {}).get('__links__'))['ctl'] + page_engines
    if KERNEL['on'] and (KERNEL['mem_ports'] or n_ctl):
        # The memory ports' masters: one where they are folded (the folded
        # endpoint's, the only one the kernel flow binds).
        ports = KERNEL['mem_ports']
        if KERNEL['fold_ports'] or KERNEL['sim_ports_folded']:
            ports = min(ports, 1)
        if BUILT is not None and BUILT.get('mem_ports') is not None:
            ports = BUILT['mem_ports']
        # Built: the kernel's masters as generated; planned: the count of least
        # LUTs (FOLDSIM_VMEM_MASTERS, a flow setting, recorded in the outcome).
        # With --masters-by-traffic the count of least LUTs that carries this
        # point's HBM bytes a host cycle (storage.MASTER_HEADROOM), not the whole
        # HBM share.
        bpc = None
        if PLATFORM['by_traffic'] and tiers is not None and BUILT is None:
            bpc = (off['hbm_bytes'] + tiers.get('base_hbm', 0.0) + built_hbm) / max(tiers['frame'], 1)
        platform = storage.platform_cost(ports, math.ceil(n_ctl - 1e-9),
                                         BUILT.get('vmem_masters') if BUILT else 'best', bpc)
        logic += platform['logic']
        lutram += platform['lutram']
    sites = logic + storage.LUTRAM_SITES * lutram
    res = dict(luts=round(logic + lutram), logic=round(logic), lutram=round(lutram),
               sites=round(sites), fits=fits, bram=round(bram, 1), uram=uram,
               link_lutram=round(lk['lutram']), link_bram=round(lk['bram'], 1),
               link_uram=lk['uram'], link_ff=lk['ff'],
               largest_copy=round(max(per_copy.values())),
               copies={f'{c}#{j}': dict(logic=round(copy_logic[(c, j)]), total=round(per_copy[(c, j)]),
                                        store_lut=round(store_lut.get((c, j), 0.0)),
                                        store_lutram=round(store_lutram.get((c, j), 0.0)),
                                        uram=round(used.get(f'uram@{c}#{j}', 0.0), 1),
                                        bram=round(used.get(f'bram@{c}#{j}', 0.0), 1))
                       for (c, j) in per_copy},
               **({'copy_cap': dict(COPY_CAP, over=sorted(
                   p for p in caps if '@' in p and used[p] > caps[p] + 1e-9))} if COPY_CAP else {}),
               vmem_ctl={x: round(v, 1) for x, v in vmem_ctl.items()},
               top_logic=round(top_link_logic((sched or {}).get('__links__'))) if KERNEL['on'] else 0,
               storage=','.join(rules),
               **({'link_kinds': ','.join(link_rules)} if lu else {}),
               # The frame the chosen link kinds let (link_ii): the
               # projection's H where longer than the frame priced.
               **({'link_ii': lii} if (lii := link_ii((sched or {}).get('__links__'),
                                                      link_rules)) else {}),
               **({'ii_floor': floor} if floor > ii else {}), **paged_fields(paged),
               **({'page_engines': page_engines} if page_engines else {}),
               **({'engine_member_luts': round(member_luts)} if page_engines and member_luts else {}),
               kernel_controllers=round(n_ctl, 1),
               **({'platform': {k: round(v) for k, v in platform.items()}} if platform else {}))
    if platform or BUILT is not None:
        # The controllers and the platform terms are priced after the kinds:
        # whether the point still fits its pools.
        res['fits'] = fits and res['sites'] <= caps['sites'] + 1e-9 and \
            res['bram'] <= caps['bram'] + 1e-9 and res['uram'] <= caps['uram'] + 1e-9
    if tiers is not None:
        res.update(off, hbm_bytes=round(off['hbm_bytes']), hbm_cap=round(caps['hbm']))
    return res


def unit(cls, store, per, lam, fixed):
    """A store of a class, every copy's: its cost per kind (None where some
    copy cannot take it), each copy's apart for the largest copy."""
    costs, by_copy = {}, {}
    for kind in storage.KINDS:
        total = dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0, ii_floor=0)
        parts = []
        for j, st, cnt in per:
            c = storage.kind_cost(st, kind)
            if c is None:
                total = None
                break
            c = dict(c, lutram=lam * c['lutram'])
            c = {k: (v if k == 'ii_floor' else v * cnt) for k, v in c.items()}
            parts.append((j, c))
            for key in total:
                total[key] = max(total[key], c[key]) if key == 'ii_floor' else total[key] + c[key]
        if total is not None:
            costs[kind] = total
            by_copy[kind] = parts
    return dict(costs=costs, per=by_copy, count=1, fixed=fixed, store=store, **{'class': cls})


def storage_rules(cls, units):
    """The compiler's `storage` rules of a class's choice: its memories whose
    kind is not the default, and its register stores by width (each width
    whose kind differs from the narrower's: the largest bound applies)."""
    rules = []
    regs = sorted((int(u['store'].split('>=')[1]), u['kind']) for u in units
                  if u['store'].startswith('regs>=') and not u['fixed'])
    current = 'distributed'
    for w, kind in regs:
        if kind != current:
            rules.append(f'{cls}:regs>={w}={kind}')
            current = kind
    for u in units:
        # A memory kept off chip is a virtual memory (choose_state's pattern),
        # not a storage rule; pages off chip are (`<class>:pages=hbm`).
        if u['kind'] in ('hbm', 'rows') and u['store'] != 'pages':
            continue
        if not u['store'].startswith('regs>=') and u['kind'] != u['default'] and \
                (not u['fixed'] or u['fixed'] == 'hbm'):
            rule = f"{cls}:{u['store']}={u['kind']}"
            if rule not in rules:
                rules.append(rule)
    return rules


# The screen's measure of a netlist's depth for the period fit: log2 of the end
# points at least this many levels deep. A count, not the deepest path: the
# few deepest are what Vivado retimes or optimizes away (a modulo by a
# constant), a unit deep throughout is not.
DEPTH_LEVELS = 14


def screen_depth(screen):
    """Depth of a screened netlist (ci/logic_screen.py screen <dir>)."""
    import csv
    deep = sum(1 for r in csv.DictReader(open(f'{screen}/endpoints.csv'))
               if int(r['levels']) >= DEPTH_LEVELS)
    return round(math.log2(max(deep, 1)), 3)


def lstsq(rows, ys):
    """Least squares with the normal equations (few points, few terms)."""
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[p], b[i], b[p] = a[p], a[i], b[p], b[i]
        if abs(a[i][i]) < 1e-12:
            return None
        for r in range(n):
            if r != i:
                f = a[r][i] / a[i][i]
                a[r] = [x - f * y for x, y in zip(a[r], a[i])]
                b[r] -= f * b[i]
    return [b[i] / a[i][i] for i in range(n)]


def firesim_factors(cal):
    """FireSim's LUTs against our estimate of the unfolded target, logic and
    distributed RAM apart: the least over the FireSim builds calibrated, so a
    projection is what FireSim needs at the least. Their estimate is the
    calibrated area, so only builds against Beacon's count."""
    fs = [b for b in (cal or {}).get('firesim', []) if build_estimator(b) == ESTIMATOR]
    if not fs:
        return None
    return dict(logic=min(b['logic'] / b['est_logic'] for b in fs),
                lutram=min(b['lutram'] / max(b['est_lutram'], 1) for b in fs),
                builds=[b['name'] for b in fs])


def firesim_area(cal, unfolded):
    """FireSim's projected LUTs and sites for a design: the unfolded target
    scaled by the FireSim factors (None without FireSim builds)."""
    f = firesim_factors(cal)
    if not f:
        return None
    logic, lutram = f['logic'] * unfolded['logic'], f['lutram'] * unfolded['lutram']
    sites, fits = placement(logic, lutram, unfolded['bram'])
    return dict(luts=round(logic + lutram), lutram=round(lutram), sites=round(sites), fits=fits)


def calibrate_firesim(a):
    """Add a FireSim build of a design: its LUTs from Vivado's utilization
    (the kernel's row) and its period, against the unfolded target's estimate
    from the design's models IR (calibrated as the Foldsim builds are)."""
    row = [a.luts, a.logic, a.lutram] if a.luts else None
    cols = None
    for line in ([] if row else open(a.util)):
        f = [x.strip() for x in line.split('|')]
        if 'Total LUTs' in f:
            cols = [f.index(c) for c in ('Total LUTs', 'Logic LUTs', 'LUTRAMs')]
        elif cols and len(f) > max(cols) and f[1] == a.instance:
            row = [int(f[c].split('(')[0]) for c in cols]
            break
    if row is None:
        sys.exit(f'no row {a.instance} in {a.util}')
    cal = json.load(open(a.out)) if os.path.exists(a.out) else {}
    _, classes, _ = parse_schedule(open(a.schedule).read())
    est = class_estimates(a.models, classes, a.foldsim_opt)
    unf = area(cal, est, classes, dict(classes), rest_of(cal, sum(e['lut'] for e in est.values())))
    b = dict(name=a.name, estimator=ESTIMATOR, luts=row[0], logic=row[1], lutram=row[2],
             period=a.period,
             est_logic=unf['logic'], est_lutram=unf['lutram'])
    cal['firesim'] = [x for x in cal.get('firesim', []) if x['name'] != a.name] + [b]
    json.dump(cal, open(a.out, 'w'), indent=1)
    print(f"{a.name}: {row[0]} LUTs (logic {row[1]}, lutram {row[2]}) against the unfolded "
          f"target's {unf['logic']} + {unf['lutram']}: x{row[1] / unf['logic']:.2f} logic, "
          f"x{row[2] / max(unf['lutram'], 1):.2f} lutram, period {a.period} ns")


TERMS = {'const': (), 'size': ('size',), 'size+depth': ('size', 'depth'),
         'copies': ('copies',), 'span': ('span',), 'size+copies': ('size', 'copies'),
         'size+span': ('size', 'span')}


def features(largest, depth, copies, luts):
    """A design's terms for the period fit: log2 of its largest copy's LUTs,
    the screen's depth, its copies over every class, and the share of the
    V80's placement capacity its LUTs take (how far it spreads over the dies)."""
    return dict(size=math.log2(max(largest, 1.0)), depth=depth or 0.0,
                copies=float(copies), span=luts / PLACE_SITES)


def build_features(b):
    return features(b['max_copy'], b.get('depth', 0.0),
                    sum(c['copies'] for c in b['classes'].values()), b['luts'])


def fit_row(terms, feats):
    return [1.0] + [feats[t] for t in terms]


def fit_terms(builds, terms):
    """Least squares of the period over the builds, and each build's
    leave-one-out error (empty when the builds are too few)."""
    rows = [fit_row(terms, build_features(b)) for b in builds]
    ys = [b['period'] for b in builds]
    coef = lstsq(rows, ys)
    if coef is None:
        return None
    loo = {}
    if len(builds) > len(coef) + 1:
        for i, b in enumerate(builds):
            c = lstsq(rows[:i] + rows[i + 1:], ys[:i] + ys[i + 1:])
            if c is None:
                return dict(coef=coef, terms=terms, loo={}, rms=None)
            loo[b['name']] = round(sum(x * y for x, y in zip(c, rows[i])) - ys[i], 3)
    rms = math.sqrt(sum(e * e for e in loo.values()) / len(loo)) if loo else None
    # A design is priced within the range the builds cover: no term is
    # extrapolated past them.
    span = {t: (min(build_features(b)[t] for b in builds),
                max(build_features(b)[t] for b in builds)) for t in terms}
    return dict(coef=coef, terms=terms, loo=loo, rms=rms and round(rms, 3), range=span)


def period_fit(cal, use_depth=True):
    """The period fit over the calibrated builds: of a constant, the largest
    copy's size, and size plus the screen's depth (with `use_depth` and a
    screen of every build), the one with the least leave-one-out error; size
    alone when the builds are too few to tell. A build marked
    `"period_fit": false` calibrates area only."""
    builds = [b for b in (cal or {}).get('builds', []) if b.get('period_fit', True)]
    if not builds:
        return None
    names = [n for n, t in TERMS.items()
             if 'depth' not in t or (use_depth and all('depth' in b for b in builds))]
    fits = [f for f in (fit_terms(builds, TERMS[n]) for n in names) if f]
    scored = [f for f in fits if f['rms'] is not None]
    if scored:
        return min(scored, key=lambda f: f['rms'])
    return next((f for f in fits if f['terms'] == TERMS['size']), fits[0])


def fit_period(fit, feats):
    """A candidate's period from the fit, each term held to the builds' range."""
    held = {t: min(max(feats[t], lo), hi) for t, (lo, hi) in fit.get('range', {}).items()}
    return sum(x * y for x, y in zip(fit['coef'], fit_row(fit['terms'], {**feats, **held})))


def parse_links(text):
    """The links' storage of a schedule report (`link kinds`: memories, bits
    and cost per kind, the cost in LUTRAM LUTs, RAMB36 and URAM288; register
    bits), or None for a report from before links were priced."""
    m = re.search(r'^  link kinds: distributed (\d+) (\d+) (\d+); block (\d+) (\d+) ([\d.e+]+); '
                  r'ultra (\d+) (\d+) ([\d.e+]+); registers (\d+) (\d+)', text, re.M)
    if not m:
        return None
    g = m.groups()
    uses = [dict(p=u[0], r=u[1], width=int(u[2]), depth=int(u[3]), entries=int(u[4]), kind=u[5],
                 sync=u[6] == '1', write_gap=int(u[7]), read_gap=int(u[8]), lead=int(u[9]),
                 reads=int(u[10]), banked=u[11] == '1', sources=int(u[12]),
                 producer_passes=int(u[13]), reader_passes=int(u[14]), page=int(u[15]))
            for u in re.findall(r'^    link use (\S+) (\S+) (\d+) (\d+) (\d+) (\S+) ([01]) (-?\d+) '
                                r'(-?\d+) (-?\d+) (\d+) ([01]) (\d+) (\d+) (\d+) (\d+)$', text, re.M)]
    # HBM link streams (foldsim-fold-static `link-kinds` kind hbm): every pair
    # a stream could carry (`link pair`), and those built (`link stream`).
    pairs = {(x[0], x[1]): dict(word=int(x[2]), dw=int(x[3]), pages=int(x[4]),
                                max_need=int(x[5]), read_rows=int(x[6]), write_rows=int(x[7]),
                                producer_passes=int(x[8]), reader_passes=int(x[9]))
             for x in re.findall(r'^    link pair (\S+) (\S+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) '
                                 r'(\d+) (\d+)$', text, re.M)}
    streams = [dict(p=x[0], r=x[1], kept=int(x[2]), dropped=int(x[3]), word=int(x[4]),
                    dw=int(x[5]), pages=int(x[6]), write_rows=int(x[7]), read_rows=int(x[8]),
                    max_need=int(x[9]))
               for x in re.findall(r'^    link stream (\S+) (\S+) (\d+) (\d+) (\d+) (\d+) (\d+) '
                                   r'(\d+) (\d+) (\d+)(?: \d+)?$', text, re.M)]
    # Their landing slots where the schedule fitted them (link-hbm-fit-slots:
    # a line of each built stream's, in their order), else LINK_STREAM_SLOTS.
    m = re.search(r'^  link stream slots:((?: \d+)*)$', text, re.M)
    if m and len(m.group(1).split()) == len(streams):
        for st, k in zip(streams, m.group(1).split()):
            st['slots'] = int(k)
    # HBM page streams built (page-engines): the engines, and the link
    # streams they carry (no controller of their own).
    m = re.search(r'^  page engines: (\d+) engines, (\d+) page columns', text, re.M)
    on_engines = set(re.findall(r' link:(\S+)>(\S+)', '\n'.join(
        re.findall(r'^    page engine \d+ .*$', text, re.M))))
    # The staged links' register bits (of `ff`, link storage line) and the
    # read-select bits of reader inputs with several sources (top_link_logic).
    ms = re.search(r'^  link storage: .*; staged: \d+ links, (\d+) register bits', text, re.M)
    staged_ff = int(ms.group(1)) if ms else 0
    select_bits = sum(u['width'] * (u['sources'] - 1) for u in uses if u['sources'] > 1)
    return dict(lutram=float(g[2]), lutram_bits=int(g[1]), bram=float(g[5]), bram_bits=int(g[4]),
                uram=float(g[8]), uram_bits=int(g[7]), ff=int(g[10]), uses=uses, pairs=pairs,
                staged_ff=staged_ff, select_bits=select_bits,
                streams=streams, engines=int(m.group(1)) if m else 0,
                engine_columns=int(m.group(2)) if m else 0, engine_streams=on_engines)


def link_lutram(depth, width):
    """A link memory's LUTRAM LUTs as the schedule report counts them
    (foldsim-fold-static reportLinks: foldsim_cost.mem_cost's distributed
    RAM)."""
    if depth <= 32:
        return width * 0.57
    blocks = (depth + 63) // 64
    return width * blocks * 1.14 + (width * (blocks - 1) / 3.0 if depth > 64 else 0.0)


# The link tier's kinds: the RAM a group of links (every link from one model to
# another) is held in, where block or ultra RAM may hold it (`sync`: the
# reader's only source or read a host cycle ahead, never forwarded or taken in
# the producer's host cycle), and paged (`<kind>:<p>`, foldsim-fold-static
# `link-kinds` with a page): an early link, its input's only source, as p rows
# an entry of a p-th of its bits, its write held (W flops) and paged out over
# the p host cycles after the pass, its read gathered page by page into W
# flops; the select of the write's page as a paged copy's (page_select_luts LUTs a
# page bit). Paging spaces both models' passes p apart (their link ports): a
# page is offered only where that takes at most LINK_PORT_SHARE of the frame
# priced (a longer frame lets more links page, the schedule then stretching
# to it); and where each entry is written 2p + 3 host cycles before its read
# in the schedule.
LINK_KINDS = ('distributed', 'block', 'ultra')
LINK_PAGES = (2, 4, 8)
LINK_PORT_SHARE = 0.5
# Links in HBM (the `hbm` relief kind, priced only with link_hbm): each entry
# written and each new entry read through a virtual memory a frame (a row of
# at least an AXI beat per 4096-bit column), the read prefetched a round trip
# ahead at its fixed slot into a landing buffer of the reads in flight, a
# controller per column. The compiler builds no HBM links yet.
LINK_HBM_LEAD = 8


def link_page_ok(u, p, ii):
    """Whether a link may page at p (see LINK_KINDS) at a frame of `ii`."""
    return (u['sync'] and u['sources'] == 1 and u['banked'] and u['entries'] > 1 and
            u['lead'] >= 2 * p + 3 and
            max(u['producer_passes'], u['reader_passes']) * p <= LINK_PORT_SHARE * ii)


def link_kind_cost(u, kind, f=1.0, b=1.0):
    """A link's cost in a link kind (`kind` or `kind:p`), or None where it
    cannot take it unpaged; a paged kind it cannot take costs its kind's."""
    zero = dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0.0, ii_floor=0, hbm=0.0, ctl=0)
    base, _, page = kind.partition(':')
    if base == 'distributed' or not u['sync']:
        return dict(zero, lutram=f * link_lutram(u['depth'], u['width']))
    p = int(page) if page else 0
    if not p:
        rows, cols = u['depth'], u['width']
        c = dict(zero)
    else:
        rows, cols = u['depth'] * (1 << (p - 1).bit_length()), math.ceil(u['width'] / p)
        c = dict(zero, ff=2 * u['width'], lut=float(cols * storage.page_select_luts(p)))
    if base == 'block':
        c['bram'] = b * storage.bram36(rows, cols)
    else:
        c['uram'] = float(storage.uram(rows, cols))
    return c


def link_hbm_cost(u, frame, f=1.0):
    """A link kept in HBM (LINK_HBM_LEAD), or None where its reads come too
    soon after the writes for a prefetch at its slot."""
    if not u['banked'] or u['lead'] < storage.HBM_ROUND_TRIP + LINK_HBM_LEAD:
        return None
    cols = math.ceil(u['width'] / storage.VMEM_ROW_BITS)
    row = cols * max(storage.vmem_row_bytes(min(u['width'], storage.VMEM_ROW_BITS)),
                     storage.HBM_BEAT_BYTES)
    flight = min(u['reads'], math.ceil(storage.HBM_ROUND_TRIP / max(1, u['read_gap'])) + 1)
    return dict(lut=0.0, lutram=f * link_lutram(max(2, flight), u['width']), ff=u['width'],
                bram=0.0, uram=0.0, ii_floor=0, hbm=float((u['entries'] + u['reads']) * row),
                ctl=cols)


# HBM link streams (foldsim-fold-static `link-kinds` <P>:<R>:hbm, as built):
# a pair's early links whose read comes late enough after the write (lead >=
# LINK_STREAM_LEAD and the reader's landings and the write's pages) in device
# memory behind one virtual memory of rows of `dw` bits; on chip its
# controller (link_stream_ctl), a landing buffer of LINK_STREAM_SLOTS rows
# in distributed RAM, the write's select of a row of the held word (as a
# paged copy's: page_select_luts a row bit) and its word in flops, the read
# registers' enables; HBM a row (at least a beat) per page written (every
# producer instance's) and per landing a frame. Ports: the producer's passes
# a write's pages apart, the reader's its landings + 2 apart, within
# LINK_PORT_SHARE of the frame.
LINK_STREAM_LEAD = 48
# The fewest positions a request may precede its landing (the compiler's
# linkHbmMinLead: the decisions in flight and a margin); a pair whose reads
# come sooner after the writes than that and its landings stays on chip.
LINK_STREAM_MIN_LEAD = 10
LINK_STREAM_SLOTS = 64
LINK_STREAM_ENABLE_LUTS = 2


# A stream's controller: the V7B foldsim_vmem as the kernel instantiates it
# for a stream (INSTANCES 1, a 6-bit tag, AW 16, a whole-row enable, no host
# cache, no known-zero bits, QUEUE 32, OUTSTANDING 64), priced routed
# (storage.ctl_cost 'link': vtB-vtD, vtG streams of 64 and 256 bits routed
# 1,172-1,270 logic + 363-774 LUTRAM). The OOC line of before (st32_1031/lk2/
# cal, synthesis + opt_design: DW 512 1,435 + 753; 256 1,302 + 457) is kept for
# comparison only.
LINK_STREAM_CTL_LOGIC = (1169.0, 0.52)
LINK_STREAM_CTL_LUTRAM = (161.0, 1.156)


def link_stream_ctl(dw):
    """A stream's controller (logic, LUTRAM LUTs) for rows of `dw` bits."""
    c = storage.ctl_cost('link', dw)
    return c['logic'], c['lutram']


def link_stream_cost(pair, uses, frame, f=1.0, engines=False):
    """A pair's links as an HBM stream (`pair`: its report line), the links it
    cannot carry distributed; None where its ports do not fit `frame` or it
    carries none. With `engines` (foldsim --page-engines: streams ride the
    shared page engines), its controller is its share of an engine, the
    positions its requests take of the frame (`ctl`, `engine_columns` 1)."""
    ports = max(pair['producer_passes'] * pair['pages'],
                pair['reader_passes'] * (pair['max_need'] + 2))
    if ports > LINK_PORT_SHARE * frame:
        return None
    need = LINK_STREAM_MIN_LEAD + pair['max_need'] + pair['pages'] + 6
    carried = [u for u in uses if u['banked'] and u['entries'] > 1 and u['lead'] >= need]
    if not carried:
        return None
    rest = sum(f * link_lutram(u['depth'], u['width']) for u in uses if u not in carried)
    cl, cr = link_stream_ctl(pair['dw'])
    share = 1.0
    if engines:
        share = min(1.0, ports / max(frame, 1))
        el, er = storage.page_engine_ctl()
        cl, cr = share * el, share * er
    row = max(storage.HBM_BEAT_BYTES, storage.vmem_row_bytes(pair['dw']))
    slices = sum(math.ceil(u['width'] / pair['dw']) for u in carried)
    return dict(lut=cl + pair['dw'] * storage.page_select_luts(pair['pages']) +
                LINK_STREAM_ENABLE_LUTS * slices,
                lutram=cr + link_lutram(LINK_STREAM_SLOTS, pair['dw']) + rest,
                ff=pair['word'] + 32, bram=0.0, uram=0.0, ii_floor=0,
                hbm=float((pair['read_rows'] + pair['write_rows']) * row), ctl=share,
                **({'engine_columns': 1} if engines else {}))


def link_streams_built(links):
    """The streams a report's schedule built (`link stream` lines): their
    on-chip cost and HBM bytes a frame, as link_stream_cost prices them."""
    out = dict(lut=0.0, lutram=0.0, ff=0, hbm=0.0, ctl=0)
    on_engines = (links or {}).get('engine_streams', set())
    for s in (links or {}).get('streams', []):
        # A stream carried by the page engines: its controller is theirs
        # (counted with the engines, area).
        cl, cr = (0.0, 0.0) if (s['p'], s['r']) in on_engines or BUILT is not None \
            else link_stream_ctl(s['dw'])
        row = max(storage.HBM_BEAT_BYTES, storage.vmem_row_bytes(s['dw']))
        out['lut'] += cl + s['dw'] * storage.page_select_luts(s['pages']) + \
            LINK_STREAM_ENABLE_LUTS * s['kept']
        out['lutram'] += cr + link_lutram(s.get('slots', LINK_STREAM_SLOTS), s['dw'])
        out['ff'] += s['word'] + 32
        out['hbm'] += (s['read_rows'] + s['write_rows']) * row
        out['ctl'] += 0 if (s['p'], s['r']) in on_engines else 1
    return out


def model_passes(links):
    """Each model's most passes a frame (instances x (1 + ready passes)) over
    its classes, from a report's `link use` and `link pair` lines."""
    out = {}
    for u in (links or {}).get('uses', []):
        out[u['p']] = max(out.get(u['p'], 0), u['producer_passes'])
        out[u['r']] = max(out.get(u['r'], 0), u['reader_passes'])
    for (p, r), pair in (links or {}).get('pairs', {}).items():
        out[p] = max(out.get(p, 0), pair['producer_passes'])
        out[r] = max(out.get(r, 0), pair['reader_passes'])
    return out


def link_spacing(links, kinds):
    """The positions foldsim-fold-static keeps each model's passes apart for
    the link kinds `kinds` (its `link-kinds` rules, a comma separated string
    or a list, `<producer model>:<reader model>:<kind>[:<page>]`): a paged
    rule spaces every class of both models its page apart (parseLinkKinds);
    an HBM stream (a pair the report lists as `link pair`) the producer's by
    its pages and the reader's by its landings + 2 (streamPages). The most
    over the rules naming a model; {model: spacing}, spacings of 2 or more."""
    out = {}
    pairs = (links or {}).get('pairs', {})
    rules = kinds.split(',') if isinstance(kinds, str) else list(kinds or ())
    for rule in (r.strip() for r in rules):
        parts = rule.split(':')
        if len(parts) < 3:
            continue
        p, r, kind = parts[:3]
        if kind == 'hbm':
            pair = pairs.get((p, r))
            if pair:
                out[p] = max(out.get(p, 0), pair['pages'])
                out[r] = max(out.get(r, 0), pair['max_need'] + 2)
        elif len(parts) > 3 and int(parts[3]) > 1:
            for m in (p, r):
                out[m] = max(out.get(m, 0), int(parts[3]))
    return {m: s for m, s in out.items() if s > 1}


def link_ii(links, kinds):
    """The shortest frame the link kinds `kinds` let the schedule take: each
    class's passes their model's link spacing apart, the next frame's first
    a spacing after the last (foldsim-fold-static placeModulo: spacing x
    passes). A frame shorter than this stretches to it: the projected H
    takes the longer (link_spacing; 0 without spaced models)."""
    passes = model_passes(links)
    return max([s * passes.get(m, 0) for m, s in link_spacing(links, kinds).items()],
               default=0)


def link_units(cal, links, ii=0, pages=True, hbm=False, frame=0, engines=False):
    """The links as storage_area units, a unit per pair of models with a
    link that block or ultra RAM may hold (foldsim-fold-static `link-kinds`):
    its cost in each of LINK_KINDS, paged (`<kind>:<p>`, at the schedule's
    frame `ii`) and with `hbm` in HBM (relief, at `frame`), its other links
    distributed; and the cost of the rest (registers and links only
    distributed RAM may hold), for the base usage. None without the report's
    link uses."""
    if not links or not links.get('uses'):
        return None
    k = (cal or {}).get('links', {})
    f, b = k.get('lutram', 1.0), k.get('bits', 1.0)
    zero = dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0.0, ii_floor=0)
    rest = dict(zero, ff=round(b * links['ff']))
    groups = {}
    for u in links['uses']:
        if not u['sync']:
            rest['lutram'] += f * link_lutram(u['depth'], u['width'])
            continue
        groups.setdefault((u['p'], u['r']), []).append(u)
    units = []
    # A page spaces every class of both models (not only the uses that page):
    # offered where their most passes fit the port share (link_spacing).
    passes = model_passes(links)
    for (p, r), uses in sorted(groups.items()):
        kinds = list(LINK_KINDS)
        if pages and ii:
            kinds += [f'{kind}:{q}' for kind in ('block', 'ultra') for q in LINK_PAGES
                      if any(link_page_ok(u, q, ii) for u in uses) and
                      max(passes.get(p, 0), passes.get(r, 0)) * q <= LINK_PORT_SHARE * ii]
        costs = {}
        for kind in kinds:
            q = int(kind.partition(':')[2] or 0)
            c = dict(zero, hbm=0.0, ctl=0)
            for u in uses:
                one = link_kind_cost(u, kind if not q or link_page_ok(u, q, ii)
                                     else kind.partition(':')[0], f, b)
                for key in c:
                    c[key] += one.get(key, 0)
            costs[kind] = c
        pair = links.get('pairs', {}).get((p, r))
        if hbm and frame and pair:
            # The compiler's HBM stream (all the pair's links, not only those
            # block or ultra RAM may hold).
            every = [u for u in links['uses'] if u['p'] == p and u['r'] == r]
            c = link_stream_cost(pair, every, frame, f, engines)
            if c is not None:
                c['lutram'] -= sum(f * link_lutram(u['depth'], u['width'])
                                   for u in every if not u['sync'])
                costs['hbm'] = c
        elif hbm and frame:
            parts = [link_hbm_cost(u, frame, f) for u in uses]
            if all(x is not None for x in parts):
                c = dict(zero, hbm=0.0, ctl=0)
                for x in parts:
                    for key in c:
                        c[key] += x.get(key, 0)
                cl, cr = storage.vmem_ctl(storage.HBM_COLUMN_BITS)
                c['lut'] += cl * max(x['ctl'] for x in parts)
                c['lutram'] += cr * max(x['ctl'] for x in parts)
                c['ctl'] = max(x['ctl'] for x in parts)
                costs['hbm'] = c
        units.append(dict(costs=costs, per={kind: [(0, c)] for kind, c in costs.items()},
                          count=1, fixed=None, store=f'{p}:{r}', default='distributed',
                          link=True, **{'class': '__links__'}))
    return units, rest


# The links as the fold-degree model prices them: what the schedule report
# gives, scaled by a calibration's `links` factors (`lutram`, `bits`: the
# built simulator's link LUTRAM and bits over the report's, measured on the
# 1-SM a1 fold's netlist).
def link_cost(cal, links):
    if not links:
        return dict(lutram=0.0, bram=0.0, uram=0.0, ff=0, lut=0.0, hbm=0.0, ctl=0)
    k = (cal or {}).get('links', {})
    f, b = k.get('lutram', 1.0), k.get('bits', 1.0)
    st = link_streams_built(links)
    return dict(lutram=f * links['lutram'] + st['lutram'], bram=b * links['bram'],
                uram=links['uram'], ff=round(b * links['ff']) + st['ff'], lut=st['lut'],
                hbm=st['hbm'], ctl=st['ctl'])


def parse_classes(text):
    """Each class of a schedule report (a copy of a class its own): its
    instances, the state gap, and each instance's ready and state slots;
    under '__links__' the links' storage (parse_links)."""
    out, cur = {}, None
    links = parse_links(text)
    if links:
        out['__links__'] = links
    for line in text.splitlines():
        m = re.match(r'^  (\S+) x(\d+)( \(ready pass\))?(?: state gap (\d+))?', line)
        if m:
            page = re.search(r' page (\d+)( off chip)?', line)
            cur = out[m.group(1)] = dict(n=int(m.group(2)), ready=[], state=[],
                                         gap=int(m.group(4)) if m.group(4) else None,
                                         page=int(page.group(1)) if page else 0,
                                         offchip=bool(page and page.group(2)))
            continue
        m = re.match(r'^  pages (\S+): p \d+, .* page RAMs .*, (block|ultra);', line)
        if m and m.group(1) in out:
            out[m.group(1)]['page_kind'] = m.group(2)
            continue
        m = re.match(r'^    (ready|state): ([\d .]+)$', line)
        if m and cur is not None:
            slots = []
            for x in m.group(2).split():
                a, _, b = x.partition('..')
                slots += list(range(int(a), int(b) + 1)) if b else [int(a)]
            cur[m.group(1)] = slots
    return out


def ring_order(info, ii):
    """Whether a copy keeps its registers in a ring (foldsim-fold-static's
    ringOrder): every pass inside the frame, and each instance's passes after
    the state pass of the instance before it, ending with its own."""
    if info['n'] < 2 or len(info['state']) != info['n']:
        return False
    passes = sorted([(s, 1, i) for i, s in enumerate(info['state'])] +
                    [(s, 0, i) for i, s in enumerate(info['ready'])])
    if any(not 0 <= s < ii for s, _, _ in passes) or \
            len({s for s, _, _ in passes}) != len(passes):
        return False
    start = max(n + 1 for n, (_, pass_, _) in enumerate(passes) if pass_ == 1)
    current = None
    for n in range(len(passes)):
        _, pass_, inst = passes[(start + n) % len(passes)]
        current = inst if current is None else current
        if inst != current:
            return False
        if pass_ == 1:
            current = None
    return current is None


def state_form(threads, info, ii, entries):
    """How a copy of `threads` instances keeps its registers: 'one' (a flop),
    'ring' (a flop per instance, rotated), 'flops' (a store in flops: read
    select) or 'lutram'; and whether its state bypass is built (gap below 3)."""
    bypass = info is not None and info['gap'] is not None and info['gap'] < 3
    if threads == 1:
        return 'one', bypass
    # Paged (foldsim-fold-static `page`): a page RAM, priced apart
    # (foldsim_storage.page_cost).
    if info is not None and info.get('page'):
        return 'paged', False
    if threads <= entries:
        if info is not None and ring_order(info, ii):
            return 'ring', bypass
        return 'flops', bypass
    return 'lutram', bypass


# Logic LUTs per register bit of a copy's state beyond its one-instance logic,
# by form, without and with the state bypass: the read select and the bypass
# muxes. Routed, the mid slice copy of two instances in distributed RAM with
# the bypass (m_rt4c8, 57.5k logic LUTs) against the small one of one instance
# (s_rt4ro, 34.3k): 23.2k over 25,764 bits (0.90). In flops, synthesized
# alone, 1.7k fewer (0.83). A ring has no select of its registers, but its
# read buffers keep theirs: synthesized alone 49.0k, x0.92 routed (the
# m_rt4c8 copy: 57.5k routed, 62.2k synthesized) 45.1k, 10.8k over the one
# instance's (0.42). Without the bypass a flop store's select is a 2:1 mux
# (0.5, two to a LUT); distributed RAM's read port is the select (0).
# Overridden by a calibration's 'state_logic'.
STATE_LOGIC = dict(one=(0.0, 0.0), ring=(0.0, 0.42), flops=(0.5, 0.83), lutram=(0.0, 0.90),
                   paged=(0.0, 0.0))


def state_logic(cal, form, bypass, bits):
    table = dict(STATE_LOGIC, **{k: tuple(v) for k, v in (cal or {}).get('state_logic', {}).items()})
    return table[form][1 if bypass else 0] * bits


def copy_sites(cal, est, name, threads, entries=2, info=None, ii=0):
    """LUT sites the placer needs for one copy of class `name` (`est`, its
    class_estimates entry) at `threads` instances, calibrated: its state as
    the compiler keeps it, given the copy's line of a schedule report
    (`info`, parse_classes; `ii` its II) or, without one, as before a schedule
    is known (state_form). A paged copy keeps its registers and small
    memories in page RAMs, not a distributed RAM of an entry per instance:
    its page selects and its other memories' stores, as fixed_area prices it
    (r16c, 11-07: the 64-instance slice copy, paged, was counted 1,470,914
    sites without its schedule, 168 % of the largest SLR; as built 559,939
    LUTs)."""
    kappa, lam = factors(cal, name, est['lut'])
    form, bypass = state_form(threads, info, ii, entries)
    lr, _ = state_lutram(est, threads, form)
    logic = kappa * est['lut'] + state_logic(cal, form, bypass, est['ff'])
    if form == 'paged':
        bits, mux, others = paged_state(est)
        lr, _ = state_lutram(others, threads, form)
        pc = storage.page_cost(bits, threads, info['page'],
                               'hbm' if info.get('offchip') else 'block')
        logic += pc['lut'] + mux
    return placement(logic, lam * lr)[0]


def copy_threads(n, k, j):
    """Instances of copy j of a class of n folded into k copies."""
    return n // k + (1 if j < n % k else 0)


def degrees(n):
    """The copy counts worth trying for a class of n instances: the fewest
    copies for each depth of the deepest copy, which bounds the frame."""
    ks, seen = [], set()
    for k in range(1, n + 1):
        deepest = -(-n // k)
        if deepest not in seen:
            seen.add(deepest)
            ks.append(k)
    return ks


def deepest(n, k):
    return -(-n // k)


def by_depth(sizes, choices):
    """One choice per frame depth: every class on the fewest copies whose
    deepest copy is no deeper. The frame is bounded by the deepest copy of
    any class, so a class folded finer than the others only adds copies."""
    depths = sorted({deepest(n, k) for n, ks in zip(sizes, choices) for k in ks})
    return list(dict.fromkeys(
        tuple(next(k for k in ks if deepest(n, k) <= d) for n, ks in zip(sizes, choices))
        for d in depths))


def neighbours(point, choices):
    """The choices one degree finer or coarser in a single class."""
    for i, k in enumerate(point):
        j = choices[i].index(k)
        for m in (j - 1, j + 1):
            if 0 <= m < len(choices[i]):
                yield point[:i] + (choices[i][m],) + point[i + 1:]


def pareto(points, key):
    front = []
    for p in sorted(points, key=lambda p: (p['luts'], p[key])):
        if not front or p[key] < front[-1][key]:
            front.append(p)
    return front


def evaluate(models, base_schedule, fold_options, foldsim_opt, workdir, cal,
             jobs=4, budget=0, depth=None, max_points=256, store='fixed', route=False,
             cache=None, cached_only=False, uram=True, slr_groups=False, vmem=None):
    """Every choice of copies per class: II, LUTs, whether it places, period
    and ns per cycle; the Pareto front, and the fastest point that places and
    stays within `budget` LUTs (0: none). `depth` is the design's screen depth
    (screen_depth), for a fit that has the depth term. `store` and `route`:
    the storage choice and the routability term (area); with `route` the
    pick must also route; `slr_groups`: each copy whole in one SLR (area).
    `cache`: a directory of schedule reports of the same models and options
    (an earlier explore's workdir), read instead of scheduling; with
    `cached_only`, its choices are the only ones priced. The fold options'
    `virtual-memories` are priced off chip, their host caches by `vmem`
    ((--vmem-cache-rows, budget), area).

    Past `max_points` choices (lanes folded as classes: 256 instances each),
    every choice is too many to schedule; the search then schedules one choice
    per frame depth (by_depth), then the neighbours of the front until the
    front stops changing or `max_points` are scheduled."""
    _, classes, _ = parse_schedule(base_schedule)
    m = re.search(r'register-state-entries=(\d+)', fold_options)
    entries = int(m.group(1)) if m else 2
    os.makedirs(workdir, exist_ok=True)
    target = class_estimates(models, classes, foldsim_opt, os.path.join(workdir, 'area.json'))
    m = re.search(r'virtual-memories=(\S+)', fold_options)
    est = split_virtual(target, m.group(1) if m else '')
    rest = rest_of(cal, sum(est[c]['lut'] for c in classes))
    names = [c for c, n in classes.items() if n > 1]
    choices = [degrees(classes[c]) for c in names]
    fit = period_fit(cal, use_depth=depth is not None)
    retired = retired_builds(cal)
    if retired:
        print(f"foldsim_degree: the area factors of {', '.join(retired)} were fitted on "
              f"another estimator than the one in use ({ESTIMATOR}) and are not used"
              f"{'' if area_builds(cal) else ': LUTs at the estimator scale'}; re-fit them with "
              "`foldsim_degree.py calibrate` on their models", file=sys.stderr)

    def one(ks):
        copies = ','.join(f'{c}:{k}' for c, k in zip(names, ks) if k > 1)
        tag = '_'.join(str(k) for k in ks) or 'base'
        report = os.path.join(workdir, f'schedule_{tag}.txt')
        cached = cache and os.path.join(cache, f'schedule_{tag}.txt')
        if cached and os.path.exists(cached):
            report = cached
        else:
            opts = fold_options + f' report={report} schedule-only'
            if copies:
                opts += f' copies={copies}'
            r = subprocess.run([foldsim_opt, models,
                                f'--pass-pipeline=builtin.module(foldsim-fold-static{{{opts}}})',
                                '-o', os.devnull], capture_output=True, text=True)
            if r.returncode:
                return dict(copies=copies or '-', error=r.stderr.strip().splitlines()[-1:])
        text = open(report).read()
        ii, _, _ = parse_schedule(text)
        ar = area(cal, est, classes, dict(zip(names, ks)), rest, parse_classes(text), ii,
                  entries, store, route, uram, slr_groups, vmem)
        # `copies` here is the copies spec tried; the area's per-copy table
        # (its own `copies`, as the state outcome names it) is `per_copy`.
        res = dict(ar, copies=copies or '-', ii=ii, per_copy=ar.get('copies'))
        if fit:
            period = fit_period(fit, features(res['largest_copy'], depth, sum(ks),
                                              res['luts']))
            res.update(period=round(period, 3), mhz=round(1000 / period, 1),
                       ns=round(ii * period, 1))
        return res

    key = 'ns' if fit else 'ii'
    space = math.prod(len(ks) for ks in choices)
    done = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        def run(batch):
            done.update(zip(batch, pool.map(one, batch)))
        if cached_only:
            tags = [f[len('schedule_'):-len('.txt')] for f in sorted(os.listdir(cache))
                    if f.startswith('schedule_') and f != 'schedule_base.txt']
            run([tuple(int(k) for k in t.split('_')) for t in tags
                 if len(t.split('_')) == len(names)])
        elif space <= max_points:
            run(list(itertools.product(*choices)))
        else:
            sizes = [classes[c] for c in names]
            run(by_depth(sizes, choices)[:max_points])
            while len(done) < max_points:
                point = {id(p): ks for ks, p in done.items()}
                front = pareto([p for p in done.values() if 'ii' in p], key)
                batch = list(dict.fromkeys(
                    q for p in front for q in neighbours(point[id(p)], choices)
                    if q not in done))
                if not batch:
                    break
                run(batch[:max_points - len(done)])
    points = list(done.values())
    good = [p for p in points if 'ii' in p]
    front = pareto(good, key)
    within = [p for p in good if p['fits'] and (not budget or p['luts'] <= budget)
              and p.get('routes', True)]
    pick = min(within, key=lambda p: (p[key], p['luts'])) if within else None
    # Every instance its own copy and its state in flip-flops: the target's
    # own area, which an unfolded simulator (FireSim) needs at the least.
    unfolded = area(cal, target, classes, dict(classes), rest)
    return dict(classes=classes, points=points, front=front, pick=pick, budget=budget,
                unfolded=unfolded, firesim=firesim_area(cal, unfolded), depth=depth, fit=fit,
                space=space, store=store, route=route, uram=uram, slr_groups=slr_groups,
                vmems=sum(len(e.get('vmems', [])) for e in est.values()),
                area_calibrated=bool(area_builds(cal)), retired=retired)


def copy_estimates(est, classes):
    """`est` (class_estimates of the models' class names) keyed by every
    class of a schedule, a copy (`<class>_copy<j>`) taking its class's."""
    return {c: est[re.sub(r'_copy\d+$', '', c)] for c in classes}


def page_choice(classes, sched, ii):
    """--page auto at a frame of `ii`: each copy of several instances paged
    as deep as the frame lets its state passes (p x instances <= ii, and
    p + 4 <= ii), per model the shallowest of its copies'; {model: p}."""
    pages = {}
    for c, n in classes.items():
        if n < 2:
            continue
        p = min(ii // n, ii - 4)
        if p < 1:
            return None
        m = re.sub(r'_copy\d+$', '', c)
        pages[m] = min(pages.get(m, p), p)
    return pages


def with_pages(sched, pages, offchip=()):
    """A schedule's classes (parse_classes) with each copy's page set, and
    the models in `offchip` with their pages off chip."""
    def model(c):
        return re.sub(r'_copy\d+$', '', c)
    # '__links__' (parse_links) is the links' storage, not a class.
    return {c: info if c == '__links__' else
            dict(info, page=pages.get(model(c), 0) if info['n'] > 1 else 0,
                 offchip=model(c) in offchip and info['n'] > 1)
            for c, info in sched.items()}


def choose_pages(cal, est, classes, sched, ii, store='fixed', route=False, entries=2,
                 max_ii=1 << 16):
    """--page auto: the paging of a schedule's copies that fits the V80
    region at the shortest frame. Unpaged first (no page RAM if the state
    fits as it is); else each frame from `ii` up at which some copy's page
    may deepen (a multiple of its instances), every copy of several
    instances paged as deep as that frame lets it (fewest ports: its page
    word ceil(bits / p) over p host cycles; no distributed RAM for its
    registers), until one fits (and routes, with `route`). Returns (the
    `page` option, or '' unpaged, or None if no frame fits; the frame it
    is priced at, which the paged schedule may still lengthen; the area)."""
    est = copy_estimates(est, classes)
    rest = rest_of(cal, sum(e['lut'] for e in {id(e): e for e in est.values()}.values()))

    def ok(res):
        return res['fits'] and (not route or res.get('routes', True))
    res = area(cal, est, classes, {}, rest, sched, ii, entries, store, route)
    if ok(res):
        return '', ii, res
    deep = sorted({n for n in classes.values() if n > 1})
    frames = sorted({f for n in deep for f in range(n * ((ii + n - 1) // n), max_ii + 1, n)} | {ii})
    last = None
    # Off chip, as little state as lets the rest fit: off-chip pages cross
    # HBM twice a frame, so the frame grows with the bits moved. Nothing,
    # then each model alone, fewest bits first, then the largest models
    # together (two, three, ...).
    def bits_of(m):
        return sum(est[c]['ff'] * n for c, n in classes.items()
                   if re.sub(r'_copy\d+$', '', c) == m)
    models = sorted({re.sub(r'_copy\d+$', '', c) for c, n in classes.items() if n > 1},
                    key=bits_of)
    trials = [set()] + [{m} for m in models] + \
        [set(models[-k:]) for k in range(2, len(models) + 1)]
    for offchip in trials:
        last = None
        for f in frames:
            if f < ii:
                continue
            pages = page_choice(classes, sched, f)
            if pages is None or pages == last:
                continue
            last = pages
            res = area(cal, est, classes, {}, rest, with_pages(sched, pages, offchip), f, entries,
                       store, route)
            if ok(res) and res.get('hbm_ii', 0) <= f:
                spec = ','.join(f'{m}:{p}' for m, p in sorted(pages.items()))
                return spec, f, dict(res, offchip=sorted(offchip))
            if ok(res) and res.get('hbm_ii', 0) > f:
                # The bandwidth sets the frame: the least frame it allows.
                return (','.join(f'{m}:{p}' for m, p in sorted(pages.items())), res['hbm_ii'],
                        dict(res, offchip=sorted(offchip)))
    return None, None, None


def ere_escape(text):
    """`text` as a literal in a POSIX extended regex (llvm::Regex)."""
    return re.sub(r'([.\[\]()*+?^$|\\])', r'\\\1', text)


def tiers_pattern(pairs):
    """The virtual-memories regex that names exactly these (class, memory)
    pairs as foldsim-fold-static matches them: `<class model>:<name>`, the
    model of a copy its class's."""
    names = sorted({re.sub(r'_copy\d+$', '', c) + ':' + m for c, m in pairs})
    return '^(' + '|'.join(ere_escape(n) for n in names) + ')$' if names else ''


def join_patterns(*patterns):
    """Virtual-memories regexes as one: any of them."""
    ps = [p for p in patterns if p]
    return ps[0] if len(ps) == 1 else '|'.join(f'({p})' for p in ps)


# The share of the frame a copy's page port may be busy (page_pick).
PAGE_PORT_SHARE = 0.5


def state_span(slots, frame):
    """The shortest arc of the frame (slots modulo `frame`) that holds every
    one of a copy's state passes, in slots."""
    if not slots or frame <= 0:
        return frame
    rs = sorted({s % frame for s in slots})
    gaps = [rs[k + 1] - rs[k] for k in range(len(rs) - 1)] + [rs[0] + frame - rs[-1]]
    return frame - max(gaps) + 1


def page_pick(est, classes, frame, kinds=('block', 'ultra'), sched=None):
    """Each multi-instance copy's page at a frame: the p from 1 to as many as
    PAGE_PORT_SHARE of the frame lets its state passes (p x instances <=
    share x frame, p + 4 <= frame: a page port busy every slot leaves the
    scheduler no room, and a full port fails placement after placement)
    whose page RAM (block or ultra, foldsim_storage.page_cost) and
    write-back select take the least share of the V80 region, fewer pages
    first on a tie; per model the shallowest of its copies'. None when some
    copy cannot page at that frame.

    With the unpaged schedule (`sched`, parse_classes), a copy's page also
    keeps its state passes within the arc the schedule gave them: p x
    instances <= that arc. Passes packed in a short arc sit on a chain of
    links taken in one frame (the L2's per-requester request stage behind
    the SMs' requests); spaced p apart they would stretch it and the frame
    with it (4 SMs: II 576 -> 768 with the request stage at p 4, 640 at
    p 2, 576 at p 1)."""
    def share(c):
        return (c['bram'] / (storage.MARGIN * storage.REGION['bram']) +
                c['uram'] / (storage.MARGIN * storage.REGION['uram']) +
                c['lut'] / (storage.MARGIN * storage.REGION['sites']))
    pages = {}
    for c, n in classes.items():
        if n < 2:
            continue
        top = min(max(1, int(PAGE_PORT_SHARE * frame) // n), frame - 4)
        info = (sched or {}).get(c)
        if info and info.get('state'):
            top = min(top, max(1, state_span(info['state'], frame) // n))
        if top < 1:
            return None
        bits = paged_state(est[c])[0]
        best = min(range(1, top + 1), key=lambda p: (min(share(storage.page_cost(bits, n, p, k))
                                                         for k in kinds), p))
        m = re.sub(r'_copy\d+$', '', c)
        pages[m] = min(pages.get(m, best), best)
    return pages


def project_h(ii, classes, sched, est, pattern, pages=None, offchip=(), weights=None,
              row_bytes_min=1, rtt=None):
    """The host cycles a target cycle takes (H) with state off chip, an upper
    bound: the frame, or the HBM traffic's frame if longer (every virtual
    memory read and written once an instance a frame, `weights` times;
    pages off chip moved whole twice a frame; at HBM_BYTES_PER_CYCLE), plus
    the bubbles of reads younger than the round trip: a state pass of a copy
    with virtual memories whose state gap is below HBM_ROUND_TRIP waits the
    difference (every read a miss), and a copy with pages off chip waits the
    round trip and the fetch's 7 host cycles beyond the slots between its
    state passes (no prefetch window). Returns a dict: ii, hbm_bytes, hbm_ii,
    bubbles, h, and per class its share."""
    rtt = storage.HBM_ROUND_TRIP if rtt is None else rtt
    weights = weights or {}
    vest = split_virtual(est, pattern)
    traffic, bubbles, per = 0.0, 0, {}
    for c, n in classes.items():
        model = re.sub(r'_copy\d+$', '', c)
        info = (sched or {}).get(c) or {}
        gap = info.get('gap')
        t = b = 0.0
        vms = vest[c].get('vmems', [])
        for m in vms:
            wt = float(weights.get(f'{model}:{m[0]}', weights.get(m[0], 1.0)))
            t += storage.vmem_cost(m[1], m[2], n, wt, row_bytes_min)['hbm']
        if vms and gap is not None and gap < rtt:
            b += n * (rtt - gap)
        if model in offchip and n > 1:
            bits = paged_state(est[c])[0]
            t += 2 * bits * n / 8
            b += n * max(0, rtt + 7 - ii // n)
        if t or b:
            per[c] = dict(hbm_bytes=round(t), bubbles=round(b))
        traffic += t
        bubbles += b
    hbm_ii = math.ceil(traffic / storage.HBM_BYTES_PER_CYCLE)
    return dict(ii=ii, hbm_bytes=round(traffic), hbm_ii=hbm_ii, bubbles=round(bubbles),
                h=max(ii, hbm_ii) + round(bubbles), per=per)


# choose_state looks at frames up to this many times the schedule's, and
# where paging every model does not fit, at paging each first few of them up
# to this many.
FRAME_GROWTH = 4
PAGE_FEW = 8
FRAME_STEPS = (1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8)


# choose_state's unpaged point at the schedule's frame (what it leaves over).
LAST_REFUSAL = {}


def choose_state(cal, est, classes, sched, ii, route=True, entries=2, uram=True, vmem=None,
                 pattern='', weights=None, pages_hbm=True, row_bytes_min=1, max_ii=1 << 16,
                 page=True, link_pages=True, link_hbm=False, frame_growth=None,
                 vmem_group=None, page_engines=False, only_frames=None):
    """--state auto: where each class's state lives, three tiers, priced.

      on chip    register stores and memories in flops, distributed, block or
                 ultra RAM (storage_area's kinds), or a class's registers and
                 small memories paged on chip (a page RAM, p host cycles a
                 state; the read-port bound p x instances <= frame);
      HBM        a memory as a virtual memory (foldsim_vmem: a row a state
                 pass), or a class's pages off chip (pages=hbm: all its state
                 twice a frame).

    The frame is the schedule's (`ii`) unless nothing fits there. At a frame,
    the fold as it is (no paging, nothing new off chip) is priced by
    storage_area, with the off-chip tier as relief kinds against the HBM
    bytes the frame carries; if it does not place (and route), the models
    page (page_pick), those of most register state first, as few as fit
    (a binary search over how many). Frames grow by the instances of the
    copies (deeper pages) until one fits. `est`: class_estimates, with
    the memories already virtual split off (`pattern`); `sched`: the
    schedule's classes (parse_classes). Returns None if nothing fits by
    max_ii, else a dict: page (the `page` option, '' for none), pattern (the
    virtual memories now: `pattern` and those moved), storage (rules, with
    pages=hbm), frame, the area (res), the projection (h) and what moved.

    The links' storage (the schedule report's link kinds, `sched['__links__']`,
    priced by link_cost as storage_area does) is in the base usage; none of
    the tiers moves links yet (paging them, written and read at fixed slots,
    is the next tier). When the links alone (with the logic outside the
    copies) leave no room, nothing fits at any frame: None at once."""
    est = copy_estimates(est, classes)
    rest = rest_of(cal, sum(e['lut'] for e in {id(e): e for e in est.values()}.values()))
    # Traffic of the memories already virtual (est has them under 'vmems').
    base_hbm = 0.0
    for c, n in classes.items():
        for m in est[c].get('vmems', []):
            model = re.sub(r'_copy\d+$', '', c)
            wt = float((weights or {}).get(f'{model}:{m[0]}', (weights or {}).get(m[0], 1.0)))
            base_hbm += storage.vmem_cost(m[1], m[2], n, wt, row_bytes_min)['hbm']

    base_bubbles = 0
    for c, n in classes.items():
        gap = (sched.get(c) or {}).get('gap')
        if est[c].get('vmems') and gap is not None:
            base_bubbles += n * max(0, storage.HBM_ROUND_TRIP - gap)

    def ok(res):
        return res['fits'] and (not route or res.get('routes', True))

    def outcome(res, pages, frame):
        moved = res.get('vmem_new', [])
        pat = join_patterns(pattern, tiers_pattern(moved))
        # Pages off chip: those moved, and those the schedule has already.
        # (A model paged anew at this point keeps its pages where the area put
        # them, not where the schedule had them.)
        # Pages off chip at their own depth (page_hbm_depth), where deeper.
        pages = dict(pages or {}, **{m: p for m, p in res.get('page_hbm', {}).items()
                                     if pages and m in pages and p > pages[m]})
        off = sorted({re.sub(r'_copy\d+$', '', c) for c in res.get('pages_offchip', [])} |
                     {re.sub(r'_copy\d+$', '', c) for c, info in (sched or {}).items()
                      if c != '__links__' and info.get('offchip') and
                      re.sub(r'_copy\d+$', '', c) not in (pages or {})})
        full = {c: dict(e, vmems=e.get('vmems', [])) for c, e in est.items()}
        # project_h splits by the whole pattern from the raw memories.
        raw = {c: dict(e, mems=e['mems'] + e.get('vmems', []), vmems=[]) for c, e in full.items()}
        h = project_h(frame, classes, with_pages(sched, pages or {}, off), raw, pat,
                      pages, off, weights, row_bytes_min)
        if res.get('link_hbm_bytes'):
            # Links in HBM: their traffic too (prefetched at their slots: no
            # bubbles).
            h['hbm_bytes'] += round(res['link_hbm_bytes'])
            h['hbm_ii'] = math.ceil(h['hbm_bytes'] / storage.HBM_BYTES_PER_CYCLE)
            h['h'] = max(h['ii'], h['hbm_ii']) + h['bubbles']
        if res.get('link_ii', 0) > h['ii']:
            # The link kinds' spacing stretches the frame (link_ii): H is
            # the stretched frame's.
            h['link_ii'] = res['link_ii']
            h['ii'] = res['link_ii']
            h['h'] = max(h['ii'], h['hbm_ii']) + h['bubbles']
        spec = ','.join(f'{m}:{p}' for m, p in sorted((pages or {}).items()))
        return dict(page=spec, pattern=pat, storage=res.get('storage', ''), frame=frame,
                    res=res, h=h, moved=moved, pages_offchip=off,
                    rows=tiers_pattern(res.get('rows_new', [])), rows_new=res.get('rows_new', []),
                    link_kinds=res.get('link_kinds', ''), links_hbm=res.get('links_hbm', []),
                    # The engines' rule and the shift a flow builds the choice with (foldsim
                    # --link-hbm-shift), where priced first-fit.
                    **({'engine_rule': ENGINES['rule'], 'link_hbm_shift': ENGINES['shift']}
                       if ENGINES['rule'] != 'max' else {}))

    if BUILT is not None:
        # As built (--built-json): the schedule's frame, its pages, virtual
        # memories and link kinds as generated; nothing moves.
        tiers = dict(frame=ii, base_hbm=base_hbm, base_bubbles=base_bubbles, weights=weights,
                     pages_hbm=False, row_bytes_min=row_bytes_min, link_pages=link_pages,
                     link_hbm=False, vmem_group=vmem_group, page_engines=page_engines, as_built=True)
        res = area(cal, est, classes, {}, rest, sched, ii, entries, 'auto', route, uram,
                   vmem=None, tiers=tiers)
        return outcome(res, {}, ii)
    lk = link_cost(cal, (sched or {}).get('__links__'))
    top = min(max_ii, (frame_growth or FRAME_GROWTH) * ii)
    # With HBM link streams, at the longest frame tried (their ports fit the
    # longest best): a pair a stream may carry is at its least there too.
    lu = link_units(cal, (sched or {}).get('__links__'), 0, True, link_hbm, top, page_engines)
    if lu:
        # The links that may move are the tier's: the rest alone, each group
        # at its least in every pool.
        lk = dict(lu[1])
        for u in lu[0]:
            for p in ('lutram', 'bram', 'uram'):
                lk[p] += min(c[p] for c in u['costs'].values())
    caps = storage.capacities()
    if (rest + storage.LUTRAM_SITES * lk['lutram'] > caps['sites'] or lk['bram'] > caps['bram']
            or lk['uram'] > caps['uram'] or
            (route and storage.fold_load(rest + lk['lutram'], lk['bram'], lk['uram']) >
             storage.route_luts())):
        return None
    deep = sorted({n for n in classes.values() if n > 1})
    steps = {round(ii * x) for x in FRAME_STEPS if ii * x <= top}
    frames = sorted({f for n in deep for f in range(n * ((ii + n - 1) // n), top + 1, n)} |
                    {ii} | steps)
    if only_frames:
        # Given frames only, each paging every number of models (the
        # least load that fits and routes): a large fold's frame is set by
        # its HBM bytes far past the schedule's, where the first-fit search
        # over every frame takes hours, and the load is not monotonic in
        # the models paged (selects against freed RAM).
        frames = sorted(set(only_frames))
        steps = set(frames)
    # Each frame's point is projected with the H its link kinds give
    # (link_ii): a point whose kinds stretch the frame is kept while the
    # shorter frames are tried, and the least H wins.
    last, best = None, None

    def take(o):
        nonlocal best
        if best is None or o['h']['h'] < best['h']['h']:
            best = o
        return 'link_ii' not in o['h']
    for f in frames:
        if f < ii:
            continue
        if best is not None and f >= best['h']['h']:
            break
        tiers = dict(frame=f, base_hbm=base_hbm, base_bubbles=base_bubbles, weights=weights,
                     pages_hbm=pages_hbm, row_bytes_min=row_bytes_min, link_pages=link_pages,
                     link_hbm=link_hbm, vmem_group=vmem_group, page_engines=page_engines)
        if f == ii or (f in steps and not only_frames):
            # Unpaged: at the schedule's frame, and at the frame's steps too,
            # where a longer frame carries the HBM bytes of more stores off
            # chip (the state-home policy's last tier, once a copy's block and
            # ultra RAM are spent).
            res = area(cal, est, classes, {}, rest, sched, f, entries, 'auto', route, uram,
                       vmem=vmem, tiers=tiers)
            if f == ii:
                # What the schedule's frame leaves over (the copies past their
                # caps), for the refusal's message.
                LAST_REFUSAL.clear()
                LAST_REFUSAL.update(over=(res.get('copy_cap') or {}).get('over', []),
                                    copies=res.get('copies', {}), fits=res['fits'],
                                    routes=res.get('routes', True))
            if ok(res) and take(outcome(res, {}, f)):
                return best
        pages = page_pick(est, classes, f, sched=sched) if page else None
        # A longer frame helps through deeper pages, and through the HBM
        # bytes and link pages it carries: frames where the pages deepen,
        # and a few steps of the frame (FRAME_STEPS of the schedule's).
        if not pages or (pages == last and f not in steps):
            continue
        last = pages
        # The least paging that fits: the models of most register state
        # first (they free the most distributed RAM a page port), as few as
        # fit and route; every page constrains the schedule (its copy's state
        # passes p apart).
        order = sorted(pages, key=lambda m: -sum(paged_state(est[c])[0] * n
                                                 for c, n in classes.items()
                                                 if re.sub(r'_copy\d+$', '', c) == m))
        tried = {}

        def at(k):
            if k not in tried:
                some = {m: pages[m] for m in order[:k]}
                tried[k] = (some, area(cal, est, classes, {}, rest, with_pages(sched, some), f,
                                       entries, 'auto', route, uram, vmem=vmem, tiers=tiers))
            return tried[k]
        if only_frames:
            fit = [k for k in range(0, len(order) + 1) if ok(at(k)[1])]
            if not fit:
                continue
            k = min(fit, key=lambda k: (at(k)[1].get('congestion', at(k)[1]['luts']), at(k)[1]['luts'], k))
            if take(outcome(at(k)[1], at(k)[0], f)):
                return best
            continue
        if not ok(at(len(order))[1]):
            # Paging costs logic (the write-back's selects): where the
            # routing binds, paging every model may not route while paging
            # the few of most register state does. The first few alone.
            some = next((k for k in range(1, min(len(order), PAGE_FEW) + 1)
                         if ok(at(k)[1])), None)
            if some is None:
                continue
            if take(outcome(at(some)[1], at(some)[0], f)):
                return best
            continue
        lo, hi = 1, len(order)
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(at(mid)[1]):
                hi = mid
            else:
                lo = mid + 1
        some, res = at(lo)
        if take(outcome(res, some, f)):
            return best
    return best


def state_table(out, classes, est):
    """choose_state's outcome as text: the tiers per class, the area, H."""
    res, h = out['res'], out['h']
    lines = [f"frame {out['frame']}: projected H {h['h']} (II {h['ii']}, HBM {h['hbm_bytes']} "
             f"bytes a frame = {h['hbm_ii']} host cycles at {storage.HBM_BYTES_PER_CYCLE} B, "
             f"bubbles at most {h['bubbles']})" +
             (f"; the link kinds stretch the frame to {h['link_ii']}" if h.get('link_ii') else ''),
             f"area: {res['luts']} LUTs ({res['logic']} logic + {res['lutram']} LUTRAM), "
             f"{res['bram']} RAMB36, {res.get('uram', 0)} URAM288, fits {res['fits']}"
             + (f", congestion {res['congestion']} % routes {res['routes']}"
                if 'congestion' in res else ''),
             f"page: {out['page'] or 'none'}",
             f"page rows (beyond {PAGE_MEMORY_BITS} bits): {out['rows'] or 'none'}",
             f"virtual memories: {out['pattern'] or 'none'}",
             f"pages off chip: {', '.join(out['pages_offchip']) or 'none'}",
             f"storage: {out['storage'] or 'default'}",
             f"links: {res.get('link_lutram', 0)} LUTRAM LUTs, {res.get('link_bram', 0)} RAMB36, "
             f"{round(res.get('link_uram', 0))} URAM288 counted",
             f"link kinds: {out.get('link_kinds') or 'by depth'}"
             + (f"; in HBM: {', '.join(out['links_hbm'])}" if out.get('links_hbm') else '')]
    est = copy_estimates(est, classes)
    vest = split_virtual({c: dict(e, mems=e['mems'] + e.get('vmems', []), vmems=[])
                          for c, e in est.items()}, out['pattern'])
    pages = dict(p.split(':') for p in out['page'].split(',')) if out['page'] else {}
    on = paged_on = hbm = 0.0
    lines.append(f"{'class':36s} {'inst':>5s} {'tier of registers':>22s} {'on-chip Mbit':>12s} "
                 f"{'HBM Mbit':>9s}")
    for c, n in sorted(classes.items(), key=lambda kv: -kv[1] * est[kv[0]]['ff']):
        model = re.sub(r'_copy\d+$', '', c)
        e = vest[c]
        mem_on = sum(m[1] * m[2] for m in e['mems']) * n
        mem_off = sum(m[1] * m[2] for m in e.get('vmems', [])) * n
        regs = e['ff'] * n
        if model in out['pages_offchip'] and n > 1:
            tier, off_regs = f"paged p {pages.get(model)} off chip", regs
        elif model in pages and n > 1:
            tier, off_regs = f"paged p {pages[model]}", 0
        else:
            tier, off_regs = ('flops' if n == 1 else 'stores'), 0
        if model in pages and n > 1:
            paged_on += regs - off_regs
        else:
            on += regs
        on += mem_on
        hbm += mem_off + off_regs
        lines.append(f"{c:36s} {n:5d} {tier:>22s} {(regs - off_regs + mem_on) / 1e6:12.2f} "
                     f"{(mem_off + off_regs) / 1e6:9.2f}")
    lines.append(f"state: on chip {(on + paged_on) / 1e6:.1f} Mbit (paged {paged_on / 1e6:.1f}), "
                 f"HBM {hbm / 1e6:.1f} Mbit")
    out['state_bits'] = dict(on_chip=on + paged_on, paged=paged_on, hbm=hbm)
    return '\n'.join(lines)


def table(result):
    lines = []
    fit = result['fit']
    store, route = result.get('store', 'fixed'), result.get('route', False)
    cols = ['copies', 'ii', 'luts', 'lutram', 'sites', 'fits', 'bram'] + \
        (['uram'] if store == 'auto' else []) + ['largest_copy'] + \
        (['congestion', 'routes'] if route else []) + (['mhz', 'ns'] if fit else [])
    lines.append(' '.join(f'{c:>12}' if c != 'copies' else f'{c:<60}' for c in cols))
    front = {id(p) for p in result['front']}
    for p in sorted(result['points'], key=lambda p: p.get('luts', 0)):
        if 'error' in p:
            lines.append(f"{p['copies']:<60} error: {' '.join(p['error'])}")
            continue
        mark = (' *' if id(p) in front else '') + (' <- pick' if p is result['pick'] else '')
        if p.get('ii_floor'):
            mark += f" (II >= {p['ii_floor']} with paging)"
        lines.append(' '.join(f'{str(p[c]):>12}' if c != 'copies' else f'{p[c]:<60}'
                              for c in cols) + mark)
    u = result['unfolded']
    lines.append(f"{'unfolded (every instance its own copy)':<60} {'-':>12} {u['luts']:>12} "
                 f"{u['lutram']:>12} {u['sites']:>12} {str(u['fits']):>12} {u['bram']:>12}")
    fs = result.get('firesim')
    if fs:
        lines.append(f"{'FireSim, projected (at least: unfolded x FireSim builds)':<60} {'-':>12} "
                     f"{fs['luts']:>12} {fs['lutram']:>12} {fs['sites']:>12} {str(fs['fits']):>12}")
    if result.get('vmems'):
        lines.append(f"virtual memories off chip (in HBM): {result['vmems']}; their host "
                     "caches' RAMB36 counted in bram, the controllers' logic not priced")
    space = result.get('space', len(result['points']))
    if space > len(result['points']):
        lines.append(f"searched {len(result['points'])} of {space} choices: one per frame depth, "
                     'then the neighbours of the front')
    if store == 'auto':
        r = storage.REGION
        lines.append('* Pareto front (ns per simulated cycle against LUTs); '
                     f"budget {result['budget'] or 'none'}; storage chosen per store; fits: "
                     f"{storage.MARGIN:.0%} of the V80 region's {r['sites']} LUT sites over the placed / priced "
                     f"{storage.PLACED_PER_PRICED} (a LUTRAM "
                     f"LUT {storage.LUTRAM_SITES} sites), {r['bram']} RAMB36 and "
                     + (f"{r['uram']} URAM288" if result.get('uram', True) else 'no URAM288'))
        pick = result['pick']
        if pick and pick.get('storage'):
            lines.append(f"storage of the pick: {pick['storage']}")
    else:
        lines.append('* Pareto front (ns per simulated cycle against LUTs); '
                     f"budget {result['budget'] or 'none'}; fits: {PLACE_MARGIN:.0%} of "
                     f'{PLACE_SITES} V80 LUT sites over the placed / priced {storage.PLACED_PER_PRICED} '
                     f'(a LUTRAM LUT {LUTRAM_SITES} sites) and of '
                     f'{BRAM_SITES} RAMB36')
    if route:
        per = storage.ROUTE_PER
        slrs = storage.route_slrs()
        groups = ', fewer if a choice has fewer copies (SLR groups)' \
            if result.get('slr_groups') else ''
        lines.append(f'routes: congestion = {storage.ROUTE_A} L + {storage.ROUTE_B} L (L / S) % '
                     f"(L = {storage.SYNTH_PER_ROUTED} LUTs + {per['bram']} RAMB36 + "
                     f"{per['uram']} URAM288, in millions; S = the region's {slrs:g} SLRs"
                     f'{groups}; foldsim_storage.py fit-route) <= {storage.ROUTE_LIMIT} %: a load '
                     f'of at most {storage.route_luts():.0f} LUTs')
    if result.get('retired'):
        lines.append(f"area factors not used (fitted on another estimator than {ESTIMATOR}): "
                     f"{', '.join(result['retired'])}"
                     f"{'' if result.get('area_calibrated') else '; LUTs at the Beacon device scale'}")
    if fit:
        c = fit['coef']
        label = dict(size='log2(largest copy LUTs)',
                     depth=f'log2(end points >= {DEPTH_LEVELS} levels) [{result["depth"]}]',
                     copies='copies', span='share of the V80\'s LUT sites')
        held = fit.get('range', {})
        terms = ' + '.join([f'{c[0]:.3f}'] + [
            f'{x:.3f} {label[t]}' + (f' (held to {held[t][0]:g}..{held[t][1]:g})' if t in held else '')
            for x, t in zip(c[1:], fit['terms'])])
        lines.append(f'period = {terms} ns; leave-one-out RMS {fit["rms"]} ns {fit["loo"] or ""}')
    else:
        lines.append('no calibration: LUTs at the Beacon device scale, no period term (ranked by II)')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('calibrate-firesim', help='add a FireSim build of a design')
    g.add_argument('--models', required=True)
    g.add_argument('--schedule', required=True,
                   help='a schedule report of the design (any fold degree): its classes')
    g.add_argument('--util', default='', help="FireSim's report_utilization_firesim.txt")
    g.add_argument('--luts', type=int, default=0, help='instead of --util: total LUTs')
    g.add_argument('--logic', type=int, default=0)
    g.add_argument('--lutram', type=int, default=0)
    g.add_argument('--instance', default='top_i')
    g.add_argument('--period', type=float, required=True, help='the routed period, ns')
    g.add_argument('--name', required=True)
    g.add_argument('--out', required=True)
    c = sub.add_parser('calibrate', help='add a build to a calibration file')
    c.add_argument('--models', required=True)
    c.add_argument('--schedule', required=True)
    c.add_argument('--util', required=True)
    c.add_argument('--timing', required=True)
    c.add_argument('--period', type=float, required=True)
    c.add_argument('--name', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--screen', default='',
                   help='the logic screen of the build\'s netlist (ci/logic_screen.py screen)')
    c.add_argument('--register-state-entries', type=int, default=2,
                   help="the build's foldsim --register-state-entries (0 for builds before it)")
    e = sub.add_parser('explore', help='the front of a models IR')
    e.add_argument('--models', required=True)
    e.add_argument('--module', required=True)
    e.add_argument('--workdir', required=True)
    e.add_argument('--calibration', default='')
    e.add_argument('--lut-budget', type=int, default=0)
    e.add_argument('--jobs', type=int, default=4)
    e.add_argument('--max-points', type=int, default=256,
                   help='schedule every choice up to this many, else search (evaluate)')
    e.add_argument('--screen', default='',
                   help='the logic screen of the design\'s netlist, for the depth term')
    e.add_argument('--storage', choices=('fixed', 'auto'), default='fixed',
                   help="fixed: the compiler's default storage; auto: a kind per store "
                        '(foldsim_storage.py), as foldsim --storage auto emits it')
    e.add_argument('--routability', action=argparse.BooleanOptionalAction, default=True,
                   help='a pick must also route (foldsim_storage.congestion)')
    e.add_argument('--uram', action=argparse.BooleanOptionalAction, default=True,
                   help='with --storage auto: whether stores may take UltraRAM')
    e.add_argument('--slr-groups', action='store_true',
                   help='each copy kept whole in one SLR, as foldsim --slr-groups '
                        '(the routability term)')
    e.add_argument('--schedules', default='',
                   help='a directory of schedule reports of the same models and options '
                        '(an earlier explore\'s workdir), read instead of scheduling')
    e.add_argument('--cached-only', action='store_true',
                   help='with --schedules: price its choices only, schedule none')
    e.add_argument('--fold-options', default='',
                   help='more foldsim-fold-static options, as the driver gives them '
                        '(nest, modulo, ...)')
    e.add_argument('--vmem-cache-rows', default=None, metavar='N|auto',
                   help="the fold options' virtual memories' host-cache lines, as foldsim "
                        '--vmem-cache-rows (their block RAM; none by default)')
    e.add_argument('--vmem-cache-budget', type=float, default=storage.VMEM_CACHE_BUDGET,
                   metavar='RAMB36', help='as foldsim --vmem-cache-budget')
    st = sub.add_parser('state', help="--state auto on a schedule: each class's state on chip, "
                                      'paged or in HBM, and the projected H')
    st.add_argument('--models', required=True)
    st.add_argument('--schedule', required=True, help='a schedule report of the models')
    st.add_argument('--calibration', default='')
    st.add_argument('--fold-options', default='',
                    help="the schedule's foldsim-fold-static options (its virtual-memories)")
    st.add_argument('--vmem-cache-rows', default=None, metavar='N|auto')
    st.add_argument('--vmem-cache-budget', type=float, default=storage.VMEM_CACHE_BUDGET)
    st.add_argument('--vmem-weights', default='', metavar='JSON',
                    help='accesses per instance and target cycle by memory (default 1)')
    st.add_argument('--vmem-row-bytes-min', type=int, default=1)
    st.add_argument('--no-pages-hbm', action='store_true', help='keep every page on chip')
    st.add_argument('--no-page', action='store_true', help='no paging')
    st.add_argument('--no-routability', action='store_true')
    st.add_argument('--register-state-entries', type=int, default=2)
    st.add_argument('--out', default='', help='the outcome as JSON')
    st.add_argument('--links', choices=('report', 'none'), default='report',
                    help="count the links' memories from the schedule report (or --links-from's)")
    st.add_argument('--no-link-pages', action='store_true', help='no paged links')
    st.add_argument('--vmem-group', choices=('copy',), default=None,
                    help='price the controllers as foldsim --vmem-group copy builds them (one a '
                         'copy and row size)')
    st.add_argument('--route-limit', type=float, default=None, metavar='PCT',
                    help='route only below this congestion estimate (default foldsim_storage.'
                         'ROUTE_LIMIT, the classifier\'s midpoint, 31.65): a lower one keeps '
                         'routing headroom for a build')
    st.add_argument('--frame-growth', type=int, default=FRAME_GROWTH,
                    help='frames up to this many times the schedule\'s')
    st.add_argument('--max-frame', type=int, default=1 << 16,
                    help='frames up to this many host cycles')
    st.add_argument('--vmem-forward', type=int, choices=(0, 1), default=1,
                    help='0: controllers priced without forwarding from their write buffers '
                         '(foldsim --vmem-forward 0, the kernel\'s FORWARD 0)')
    st.add_argument('--frames', default='', metavar='F,...',
                    help='only these frames, each with every number of models paged '
                         '(the least load that fits)')
    st.add_argument('--kernel-terms', action='store_true',
                    help='price the kernel memory endpoints (x the netlist memory ports), the simulator '
                         "top level's logic per link register bit, a threaded copy's controls, and "
                         'memories in block rather than ultra RAM (KERNEL; st32_1031 4-SM calibration)')
    st.add_argument('--fold-ports', action='store_true',
                    help='with --kernel-terms: the memory ports folded like a class (a what-if: shared '
                         'engines, each channel\'s state in ultra RAM)')
    st.add_argument('--fold-ports-measured', default='', metavar='LOGIC,LUTRAM,RAMB36,URAM',
                    help='with --fold-ports: the folded memory endpoint as synthesized (all ports), in '
                         'place of the what-if estimates')
    st.add_argument('--calibrate', default='', metavar='JSON',
                    help='with --kernel-terms: a whole-design calibration (calibrations/*.json): per class '
                         'family factors, the top level per link bit, the links\' LUTRAM, fixed kernel terms')
    st.add_argument('--sim-ports-folded', action='store_true',
                    help='with --kernel-terms and the endpoints as built: the simulator\'s side of the memory '
                         'ports folded into record streams (foldsim --fold-ports): its per-port channel queues '
                         'priced as what stays a port plus the shared record memories')
    st.add_argument('--uram-memories', action='store_true',
                    help='with --kernel-terms: memories may take ultra RAM (as the storage rules ask)')
    st.add_argument('--copy-logic', default='', metavar='JSON',
                    help='with --kernel-terms: measured logic LUTs of each class model\'s copy')
    st.add_argument('--links-as-built', action='store_true',
                    help="the links' storage as the report gives it (their kinds fixed: a "
                         'schedule made with a link-kinds choice), not a tier to choose')
    st.add_argument('--exact', action='store_true',
                    help='choose the kinds exactly (a 0-1 program, HiGHS through scipy) instead of '
                         'the repair and polish: the least routing load with every pool in cap')
    st.add_argument('--page-engines', action='store_true',
                    help='price off-chip pages carried by shared page engines (foldsim '
                         '--page-engines: HBM page streams) instead of a controller per column')
    st.add_argument('--engine-rule', choices=('max', 'first-fit'), default='max',
                    help="with --page-engines: the engines priced: 'max' (every off-chip copy's columns "
                         "and every stream on its own engines, an upper bound) or 'first-fit' (as "
                         'foldsim-fold-static assigns them: columns on the engines where their traffic '
                         'finds room, streams sharing under --link-hbm-shift)')
    st.add_argument('--link-hbm-shift', type=int, default=0, metavar='N',
                    help="the fold's foldsim --link-hbm-shift (streams on the page engines move their "
                         "requests up to N positions to share an engine), for --engine-rule first-fit")
    st.add_argument('--page-write-gate', action='store_true',
                    help='with --page-engines: off-chip pages written back gated (foldsim --page-write-gate): '
                         "an OR of the pages' gated registers instead of the p:1 select")
    st.add_argument('--masters-by-traffic', action='store_true',
                    help="the controllers' masters (FOLDSIM_VMEM_MASTERS) of least LUTs that carry the "
                         "point's own HBM bytes a host cycle (x storage.MASTER_HEADROOM), not the whole HBM "
                         'share: fewer masters, shares and sc_mem nodes where the traffic is light')
    st.add_argument('--link-hbm', action='store_true',
                    help='price links in HBM too (a what-if: the compiler builds none)')
    st.add_argument('--built-json', default='', metavar='JSON',
                    help="price as built: the generation's foldsim_sim.json (and the foldsim_sim.vh / "
                         'foldsim_kernel.vh beside it) gives the kernel\'s controllers (virtual memories '
                         'and groups, streams\' own, page engines; their host caches, known-zero bits, '
                         'FORWARD and masters); the schedule\'s frame, pages and link kinds, nothing moved')
    st.add_argument('--copy-cap', default='', metavar='REGION[:SLR]',
                    help="each class copy's own block and ultra RAM within one SLR of the region "
                         "(foldsim_v80_slr.json, or `v80`), with its margin: SLR the name of the "
                         'one every copy must fit, or smallest (default) / largest; the stores\' '
                         'kinds are chosen under it')
    st.add_argument('--read-buffers', action='store_true',
                    help="price a threaded memory's per-instance read registers (a buffer of an entry per "
                         'instance, LUTRAM of the read width) in every on-chip kind of the memory, as the fold '
                         "builds them (rows page them with the instance's registers)")
    st.add_argument('--copy-cap-luts', action='store_true',
                    help="with --copy-cap: each copy's LUTs (its logic plus its stores' LUTs and LUTRAM, the "
                         'read buffers priced: implies --read-buffers) and LUTRAM within the SLR\'s too (times '
                         'the margin, over the placed / priced LUTs), and its stores\' flops within the SLR\'s; '
                         'a copy whose LUTRAM stores outgrow it moves them to block or ultra RAM or off chip')
    st.add_argument('--links-from', default='', metavar='SCHEDULE',
                    help='another schedule report of the same fold to read the link storage from '
                         '(one made by a compiler that reports it)')
    st.add_argument('--state-calibration', default='', metavar='JSON',
                    help="the state-home policy's calibration to synthesized LUTs (STATE_CAL; "
                         'calibrations/state-*.json, state-fit)')
    st.add_argument('--copy-budget', default='', metavar='FRAC[:SLR]',
                    help="the state-home policy's per-copy budget (copy_budget): each copy's LUTs at most "
                         'FRAC of the SLR (smallest by default) of the V80 region; with --state-calibration')
    sf = sub.add_parser('state-fit', help="fit the state-home policy's calibration (STATE_CAL) on a "
                                          'synthesized build: its as-built state outcome against Vivado')
    sf.add_argument('--state', required=True, metavar='JSON',
                    help="`state --built-json ... --out` of the build's generation, without a calibration")
    sf.add_argument('--hier', required=True, metavar='RPT',
                    help="the build's report_utilization -hierarchical (post-synthesis)")
    sf.add_argument('--sim', default='sim', help="the simulator's instance in the report")
    sf.add_argument('--afu', default='foldsim_afu_0', help="the kernel's (the simulator's parent)")
    sf.add_argument('--top', default='top_wrapper', help="the region's top (the kernel's parent)")
    sf.add_argument('--netlist', default='', metavar='MLIR',
                    help="the build's foldsim_sim.mlir: the synthesized LUTs per counted (foldsim_cost) of its "
                         'copies (count_to_synth, the per-class measure of foldsim --state-measured)')
    sf.add_argument('--name', default='')
    sf.add_argument('--out', required=True)
    f = sub.add_parser('fit', help='the period fit of a calibration and its leave-one-out error')
    f.add_argument('--calibration', required=True)
    for p in (g, c, e):
        p.add_argument('--foldsim-opt', default=default_foldsim_opt(),
                       help='the foldsim-opt that prices the models (default: beside this '
                            'script, else PATH)')
    for p in sub.choices.values():
        if not any('--estimator' in x.option_strings for x in p._actions):
            p.add_argument('--estimator', choices=ESTIMATORS, default=IR_COST,
                           help='the area model: the IR cost model (foldsim_cost.py, default: the calibrations '
                                'fitted on it) or Beacon\'s area (beacon-area: foldsim-estimate-area)')
    a = ap.parse_args()
    set_estimator(getattr(a, 'estimator', IR_COST))
    if a.cmd == 'fit':
        cal = json.load(open(a.calibration))
        builds = [b for b in cal['builds'] if b.get('period_fit', True)]
        for name, terms in TERMS.items():
            if 'depth' in terms and not all('depth' in b for b in builds):
                continue
            print(name, fit_terms(builds, terms))
        print('chosen', period_fit(cal)['terms'] or 'const')
        return 0
    if a.cmd == 'calibrate-firesim':
        return calibrate_firesim(a)
    if a.cmd == 'calibrate':
        return calibrate(a)
    if a.cmd == 'state':
        return state_main(a)
    if a.cmd == 'state-fit':
        return state_fit_main(a)
    cal = json.load(open(a.calibration)) if a.calibration else None
    fold = f'module={a.module} sim=foldsim_sim'
    if a.fold_options:
        fold += ' ' + a.fold_options
    base = os.path.join(a.workdir, 'schedule_base.txt')
    os.makedirs(a.workdir, exist_ok=True)
    if a.schedules and os.path.exists(os.path.join(a.schedules, 'schedule_base.txt')):
        base = os.path.join(a.schedules, 'schedule_base.txt')
    else:
        r = subprocess.run([a.foldsim_opt, a.models,
                            f'--pass-pipeline=builtin.module(foldsim-fold-static{{{fold} '
                            f'report={base} schedule-only}})', '-o', os.devnull],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr)
    result = evaluate(a.models, open(base).read(), fold, a.foldsim_opt, a.workdir, cal,
                      a.jobs, a.lut_budget, screen_depth(a.screen) if a.screen else None,
                      a.max_points, a.storage, a.routability, a.schedules or None,
                      a.cached_only, a.uram, a.slr_groups,
                      (a.vmem_cache_rows, a.vmem_cache_budget) if a.vmem_cache_rows else None)
    print(table(result))
    json.dump(result, open(os.path.join(a.workdir, 'degree.json'), 'w'), indent=1,
              default=str)


def hier_rows(path):
    """{instance: {lut, lutram (with SRLs), ff}} of a Vivado hierarchical
    utilization report (the first row of each name)."""
    hdr, out = None, {}
    for line in open(path):
        if not line.startswith('|'):
            continue
        c = [x.strip() for x in line.split('|')[1:-1]]
        if hdr is None and c and c[0] == 'Instance':
            hdr = c
            continue
        if hdr and len(c) == len(hdr) and c[0] not in out:
            d = dict(zip(hdr, c))

            def num(k):
                m = re.match(r'[\d.]+', d.get(k, '') or '')
                return float(m.group(0)) if m else 0.0
            out[c[0]] = dict(lut=num('Total LUTs'), lutram=num('LUTRAMs') + num('SRLs'), ff=num('FFs'))
    return out


def state_fit(state, hier, sim='sim', afu='foldsim_afu_0', top='top_wrapper'):
    """STATE_CAL fitted on one synthesized build: `state` the as-built
    outcome of its generation (state --built-json, uncalibrated), `hier` its
    hierarchical utilization (hier_rows). The copies' logic and LUTRAM
    factors (least squares through zero over every copy the report has, as
    `<copy>_copy`); the level's (the simulator less its copies, against the
    outcome's LUTs outside the copies); the kernel and glue (the region's top
    less the simulator), fixed. Returns (STATE_CAL, per-copy rows)."""
    copies = state['res']['copies']
    rows = []
    for key, v in copies.items():
        c, _, j = key.rpartition('#')
        meas = hier.get(f'{c}_copy') if j == '0' else None
        if meas:
            rows.append(dict(copy=c, logic=v['logic'] + v['store_lut'], lutram=v['store_lutram'],
                             m_logic=meas['lut'] - meas['lutram'], m_lutram=meas['lutram'],
                             m_lut=meas['lut'], model=v['total']))
    fl = sum(r['logic'] * r['m_logic'] for r in rows) / max(1e-9, sum(r['logic'] ** 2 for r in rows))
    fr = sum(r['lutram'] * r['m_lutram'] for r in rows) / max(1e-9, sum(r['lutram'] ** 2 for r in rows))
    model_copies = sum(v['total'] for v in copies.values())
    m_copies = sum(r['m_lut'] for r in rows)
    out_level = state['res']['luts'] - model_copies
    m_level = hier[sim]['lut'] - m_copies
    k_lut = hier[top]['lut'] - hier[sim]['lut']
    k_lutram = hier[top]['lutram'] - hier[sim]['lutram']
    cal = dict(copy_logic=round(fl, 3), copy_lutram=round(fr, 3),
               level=round(m_level / max(out_level, 1), 3),
               kernel_logic=round(k_lut - k_lutram), kernel_lutram=round(k_lutram),
               placed_per_priced=1.02)
    for r in rows:
        r['fitted'] = fl * r['logic'] + fr * r['lutram']
    return cal, rows


def count_copies(sim):
    """{copy: {lut, lutram, ff}} of a generated simulator (foldsim_sim.mlir):
    each class copy's own module subtree (`<copy>_static`, the other copies
    left out, no links), counted by foldsim_cost."""
    import foldsim_cost
    import foldsim_sll_check as sll
    mods, _ = foldsim_cost.parse(sim, count=True)
    statics = {m for m in mods if m.endswith('_static')}
    out = {}
    for st in statics:
        acc = sll.new_acc()
        sll.subtree(mods, st, statics - {st}, acc)
        out[st[:-len('_static')]] = dict(lut=acc['lut'], lutram=acc['lutram'], ff=acc['ff'])
    return out


def state_fit_main(a):
    state = json.load(open(a.state))
    hier = hier_rows(a.hier)
    cal, rows = state_fit(state, hier, a.sim, a.afu, a.top)
    if a.netlist:
        counted = count_copies(a.netlist)
        # The LUT-weighted median of the copies' synthesized / counted (robust
        # to the few small copies whose link rows the count charges them).
        both = sorted((hier[f'{c}_copy']['lut'] / counted[c]['lut'], hier[f'{c}_copy']['lut'])
                      for c in counted if f'{c}_copy' in hier and counted[c]['lut'])
        half, acc = sum(w for _, w in both) / 2, 0.0
        for r, w in both:
            acc += w
            if acc >= half:
                cal['count_to_synth'] = round(r, 3)
                break
    cal = dict(name=a.name or os.path.basename(a.out).rsplit('.', 1)[0],
               source=f'{os.path.abspath(a.state)} against {os.path.abspath(a.hier)}', **cal)
    json.dump(cal, open(a.out, 'w'), indent=1)
    print(f"copies: logic x{cal['copy_logic']}, LUTRAM x{cal['copy_lutram']}; level x{cal['level']}; "
          f"kernel {cal['kernel_logic']} + {cal['kernel_lutram']} LUTRAM")
    print(f"{'copy':40s} {'model':>9s} {'measured':>9s} {'fitted':>9s} {'error':>6s}")
    for r in sorted(rows, key=lambda r: -r['m_lut']):
        print(f"{r['copy'][:40]:40s} {r['model']:9.0f} {r['m_lut']:9.0f} {r['fitted']:9.0f} "
              f"{100 * (r['fitted'] / max(r['m_lut'], 1) - 1):+5.0f}%")
    return 0


def state_policy(cal_path, budget=None, region_path='v80'):
    """Set the state-home policy (foldsim --state auto): STATE_CAL from
    `cal_path` (or none), the per-copy budget `budget` (FRAC[:SLR]: COPY_CAP)
    over the region's SLRs, the on-chip-first repair. Returns COPY_CAP."""
    global STATE_CAL, COPY_CAP
    STATE_CAL = json.load(open(cal_path)) if cal_path else None
    if STATE_CAL and STATE_CAL.get('placed_per_priced'):
        storage.PLACED_PER_PRICED = STATE_CAL['placed_per_priced']
    if budget:
        frac, _, slr = str(budget).partition(':')
        if region_path in ('', 'v80'):
            region_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')
        COPY_CAP = copy_budget(json.load(open(region_path)), float(frac), slr or 'smallest')
        storage.READ_BUFFERS = True
        storage.ONCHIP_FIRST = True
    return COPY_CAP


def state_main(a):
    global BUILT
    if getattr(a, 'state_calibration', '') or getattr(a, 'copy_budget', ''):
        state_policy(a.state_calibration, a.copy_budget)
    ENGINES.update(rule=getattr(a, 'engine_rule', 'max'), shift=getattr(a, 'link_hbm_shift', 0))
    PLATFORM.update(by_traffic=getattr(a, 'masters_by_traffic', False))
    storage.PAGE_WRITE_GATE = getattr(a, 'page_write_gate', False)
    if a.exact:
        storage.ASSIGN_EXACT = True
    if a.vmem_forward == 0:
        # Controllers without forwarding from their write buffers (the
        # kernel's FORWARD 0, every controller; the json carries it to the
        # build).
        storage.no_forward()
    global COPY_CAP
    if getattr(a, 'read_buffers', False) or getattr(a, 'copy_cap_luts', False):
        # Per-copy LUT caps count the read buffers too (g64cj, 11-07: 175k
        # LUTs of a slice copy's 401k LUTRAM, unpriced, past SLR2's cap).
        storage.READ_BUFFERS = True
    if getattr(a, 'copy_cap', ''):
        path, _, slr = a.copy_cap.partition(':')
        if path == 'v80':
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldsim_v80_slr.json')
        COPY_CAP = slr_copy_cap(json.load(open(path)), slr or 'smallest',
                                getattr(a, 'copy_cap_luts', False))
    BUILT = read_built(a.built_json) if getattr(a, 'built_json', '') else None
    if BUILT is not None:
        m = re.search(r'(?:^| )storage=(\S+)', a.fold_options)
        BUILT['storage'] = m.group(1) if m else ''
        # The memories the generation paged as rows (--page-rows).
        m = re.search(r'(?:^| )page-rows=(\S+)', a.fold_options)
        BUILT['rows'] = m.group(1) if m else ''
    if a.route_limit is not None:
        storage.ROUTE_LIMIT = a.route_limit
    cal = json.load(open(a.calibration)) if a.calibration else None
    text = open(a.schedule).read()
    ii, classes, _ = parse_schedule(text)
    sched = parse_classes(text)
    if getattr(a, 'kernel_terms', False):
        KERNEL.update(on=True, mem_ports=kernel_mem_ports(text),
                      fold_ports=getattr(a, 'fold_ports', False),
                      folded_measured=dict(zip(('logic', 'lutram', 'ramb36', 'uram'),
                                               map(float, a.fold_ports_measured.split(','))))
                      if getattr(a, 'fold_ports_measured', '') else None,
                      calibration=json.load(open(a.calibrate)) if getattr(a, 'calibrate', '') else None,
                      models=a.models, features=None,
                      sim_ports_folded=getattr(a, 'sim_ports_folded', False),
                      uram_memories=getattr(a, 'uram_memories', False),
                      copy_logic=json.load(open(a.copy_logic)) if getattr(a, 'copy_logic', '') else None)
    m = re.search(r'virtual-memories=(\S+)', a.fold_options)
    pattern = m.group(1) if m else ''
    names = sorted({re.sub(r'_copy\d+$', '', c) for c in classes})
    est = split_virtual(class_estimates(a.models, names), pattern)
    weights = json.load(open(a.vmem_weights)) if a.vmem_weights else None
    vmem = (a.vmem_cache_rows, a.vmem_cache_budget) if a.vmem_cache_rows else None
    sched.pop('__links__', None)
    links = None
    if a.links == 'report':
        links = parse_links(open(a.links_from).read() if a.links_from else text)
    if links and a.links_as_built:
        links = dict(links, uses=[])
    if links:
        sched['__links__'] = links
    out = choose_state(cal, est, classes, sched, ii, route=not a.no_routability,
                       entries=a.register_state_entries, vmem=vmem, pattern=pattern,
                       weights=weights, pages_hbm=not a.no_pages_hbm,
                       row_bytes_min=a.vmem_row_bytes_min, page=not a.no_page,
                       link_pages=not a.no_link_pages, link_hbm=a.link_hbm,
                       frame_growth=a.frame_growth, vmem_group=a.vmem_group, max_ii=a.max_frame,
                       page_engines=a.page_engines,
                       only_frames=[int(x) for x in a.frames.split(',') if x])
    if links:
        lk = link_cost(cal, links)
        print(f"links: {(links['lutram_bits'] + links['bram_bits'] + links['uram_bits']) / 1e6:.1f} "
              f"Mbit in memories ({lk['lutram']:.0f} LUTRAM LUTs, {lk['bram']:.1f} RAMB36, "
              f"{lk['uram']:.0f} URAM288), {links['ff']} register bits; region caps "
              f"{storage.capacities()['sites'] / 1e6:.2f}M sites (placed / priced {storage.PLACED_PER_PRICED}), "
              f"{storage.MARGIN * storage.REGION['bram']:.0f} RAMB36, "
              f"{storage.MARGIN * storage.REGION['uram']:.0f} URAM288")
    if out is None:
        print(f'--state auto: nothing fits the V80 region (II {ii})'
              + (' with the links\' memories' if links else ''))
        return 1
    print(f'II {ii} (schedule)')
    print(state_table(out, classes, est))
    if BUILT is not None:
        print(f"as built: {len(BUILT['controllers'])} controllers "
              f"({', '.join(f'{v} {k}' for k, v in sorted(BUILT['counts'].items()))}), FORWARD "
              f"{BUILT['forward']}: {BUILT['logic']:,.0f} logic + {BUILT['lutram']:,.0f} LUTRAM LUTs, "
              f"{BUILT['bram']:.1f} RAMB36; platform {out['res'].get('platform')}")
    if a.out:
        out['vmem_forward'] = a.vmem_forward
        json.dump(out, open(a.out, 'w'), indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
