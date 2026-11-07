#!/usr/bin/env python3
"""foldsim_sll_check.py <gen dir> [--log <driver log>] [--slr-region v80|<json>] [--foldsim-opt <bin>] [--out <dir>]

The SLLs an existing netlist needs at the least, checked against the region's budget, without regenerating it: the
driver's logged fold pipeline (the `+ ... foldsim-opt <flat.mlir> ... --pass-pipeline=...foldsim-fold-static{...}`
line of the generation log, <gen dir>.gen.log by default) is cut before foldsim-fold-static to rebuild models.mlir,
each class copy's LUT sites are estimated as --slr-groups does (foldsim_degree.copy_sites over the copies and
instances of the generation's own schedule.txt), and the fold is run again schedule-only with sll-estimate (fold-static) over the region's SLRs.

Prints the estimate (link bits over each boundary of the best assignment, the classes larger than every SLR) and
exits 0 if the worst boundary is within the budget (sll_per_boundary x sll_margin) and every class copy fits an SLR,
1 if not (a plan to refuse), 2 on error. Report in <out>/sll.txt.
"""
import argparse, json, os, re, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foldsim_degree  # noqa: E402


def split_top(pipeline):
    """The top-level passes of builtin.module(...), braces and parentheses respected."""
    body = pipeline[pipeline.index('(') + 1:pipeline.rindex(')')]
    parts, depth, cur = [], 0, ''
    for ch in body:
        if ch in '({':
            depth += 1
        elif ch in ')}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gen')
    ap.add_argument('--log')
    ap.add_argument('--slr-region', default='v80')
    ap.add_argument('--foldsim-opt', default=os.path.join(HERE, 'foldsim-opt'))
    ap.add_argument('--out')
    ap.add_argument('--register-state-entries', type=int, default=None)
    a = ap.parse_args()
    gen = os.path.abspath(a.gen.rstrip('/'))
    log = a.log or gen + '.gen.log'
    out = os.path.abspath(a.out or gen + '.sll')
    os.makedirs(out, exist_ok=True)
    region_path = os.path.join(HERE, 'foldsim_v80_slr.json') if a.slr_region == 'v80' else a.slr_region
    region = json.load(open(region_path))
    caps = [round(region['margin'] * s['luts'] * region['sites_per_lut']) for s in region['slrs']]
    names = [s['name'] for s in region['slrs']]
    budget = int(region.get('sll_margin', 1.0) * region.get('sll_per_boundary', 0))

    line = next((l for l in open(log) if 'foldsim-fold-static{' in l and '--pass-pipeline=' in l), None)
    if not line:
        sys.exit(f'no fold pipeline line in {log}')
    # The trace line is unquoted (the pipeline's options hold spaces): the input is the token after foldsim-opt,
    # the pipeline everything from --pass-pipeline= to the final -o.
    flat = line.lstrip('+ ').split()[1]
    m = re.search(r'--pass-pipeline=[\'"]?(builtin\.module\(.*\))[\'"]? -o \S+\s*$', line)
    if not m:
        sys.exit(f'no --pass-pipeline=builtin.module(...) -o in the fold line of {log}')
    passes = split_top(m.group(1))
    fi = next(i for i, p in enumerate(passes) if p.startswith('foldsim-fold-static{'))
    fold = passes[fi][len('foldsim-fold-static{'):-1]
    # Options this check sets itself, or that would write the generation's outputs.
    fold = ' '.join(o for o in shlex.split(fold) if not re.match(
        r'(report|slr-groups|slr-capacity|slr-capacities|slr-sites|slr-assign|slr-sll-budget|sll-estimate|schedule-only)\b', o))
    models = os.path.join(out, 'models.mlir')
    run = lambda cmd: subprocess.run(cmd, check=True, capture_output=True, text=True)
    run([a.foldsim_opt, flat, '--mlir-print-op-on-diagnostic=false',
         f'--pass-pipeline=builtin.module({",".join(passes[:fi])})', '-o', models])
    # The class copies and their instances: the generation's own schedule (same fold options).
    _, classes, _ = foldsim_degree.parse_schedule(open(os.path.join(gen, 'gen', 'schedule.txt')).read())
    cal = json.load(open(foldsim_degree_calibration()))
    entries = a.register_state_entries
    if entries is None:
        m = re.search(r'register-state-entries=(\d+)', fold)
        entries = int(m.group(1)) if m else 2
    of = {c: re.sub(r'_copy\d+$', '', c) for c in classes}
    est = foldsim_degree.class_estimates(models, sorted(set(of.values())))
    sites = {c: round(foldsim_degree.copy_sites(cal, est[of[c]], of[c], n, entries)) for c, n in classes.items()}
    report = os.path.join(out, 'sll.txt')
    opts = (f'{fold} schedule-only report={report} sll-estimate slr-sll-budget={budget} slr-capacities='
            + ','.join(map(str, caps)) + ' slr-sites=' + ','.join(f'{c}:{s}' for c, s in sites.items()))
    run([a.foldsim_opt, models, f'--pass-pipeline=builtin.module(foldsim-fold-static{{{opts}}})', '-o', os.devnull])
    text = open(report).read()
    lines = text[text.index('  sll estimate:'):].split('\n')
    sec = '\n'.join([lines[0]] + [l for l in __import__('itertools').takewhile(lambda l: l.startswith('    '), lines[1:])])
    m = re.search(r'\(worst (\d+), total (\d+)\)', sec)
    worst = int(m.group(1)) if m else None
    top = max(sites, key=sites.get)
    ratio = sites[top] / max(caps)
    # Calibrated on two V80 builds (11-07): b4e (largest copy 89 % of the largest SLR, whole-class cut 42,407 link
    # bits) placed with 3,818 / 10,134 SLLs: Vivado splits a large class where its own cut is small, so a whole-class
    # cut over the budget is not by itself a failure (WARN). r16c (largest copy 168 %) failed partitioning (23,987 SLLs
    # needed between SLR1 and SLR2): a class copy larger than every SLR must be cut, at a cost not known before
    # placement (REFUSE). A plan that pins whole groups to SLRs (--slr-groups --slr-region) is exact: over the budget
    # it is refused by the compiler itself.
    large = ratio > 1.0
    ok = not large
    warn = worst is None or worst > budget
    print(f'{os.path.basename(gen)}: SLRs {",".join(names)} capacities {",".join(map(str, caps))} sites, '
          f'SLL budget {budget} a boundary ({region.get("sll_per_boundary")} x {region.get("sll_margin")})')
    print(sec.rstrip())
    print(f'{os.path.basename(gen)}: largest class copy {top} {sites[top]} sites = {100 * ratio:.0f} % of the '
          f'largest SLR\'s capacity')
    print(f'{os.path.basename(gen)}: {"REFUSE" if large else "WARN" if warn else "OK"}: '
          + ('a class copy larger than every SLR (Vivado must cut it, its SLLs unknown: r16c failed so)' if large else
             f'whole-class cut {worst} link bits over the {budget} budget (Vivado may still split classes: b4e placed '
             'so)' if warn else f'worst boundary {worst} link bits within {budget}'))
    sys.exit(0 if ok else 1)


def foldsim_degree_calibration():
    return os.path.join(HERE, 'foldsim_v80.json')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr[-3000:] if e.stderr else str(e))
        sys.exit(2)
