#!/usr/bin/env python3
"""fit_generic.py: a design-independent calibration from a design's family fit.

A family calibration (calm4b4s4-*.json: per class family a logic and a LUTRAM
factor, the family a regex on the class model's name) prices only the design
it was fitted on: another design's classes match no family and keep 1. This
fits the factors instead on the classes' generic cost features
(foldsim_degree.class_features: log10 of the logic LUTs, the shares of
selects and of arithmetic, flops and memory bits per LUT), so that any
design's classes are priced by what they are made of:

    factor = clamp(exp(c0 + sum(ci * feature_i)), lo, hi)

one model for logic, one for LUTRAM, by ridge least squares over the
families (each family's features the LUT-weighted mean of its classes in
--models, weighted by the square root of its LUTs), lo..hi the families'
range. Leave-one-family-out errors say how a family's classes would be
priced had the fit not seen them.

Writes --out: the base calibration's design-independent terms (rest, the top
level's logic per link bit, links' LUTRAM, the virtual-memory controllers,
the control endpoint, SYNTH_PER_ROUTED) with no families and the `generic`
model. The family fit stays the design's own file (data).

usage: fit_generic.py --calibration <family fit.json> --models <models.mlir> --out <generic.json>
                      [--ridge L] [--name N]
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import foldsim_degree as fd  # noqa: E402


def fit(xs, ys, ws, ridge):
    """Ridge least squares on standardized features; coefficients on the raw
    features, the intercept first."""
    n = len(xs[0])
    mu = [sum(w * x[i] for x, w in zip(xs, ws)) / sum(ws) for i in range(n)]
    sd = [math.sqrt(sum(w * (x[i] - mu[i]) ** 2 for x, w in zip(xs, ws)) / sum(ws)) or 1.0 for i in range(n)]
    rows, t = [], []
    for x, y, w in zip(xs, ys, ws):
        s = math.sqrt(w)
        rows.append([s] + [s * (x[i] - mu[i]) / sd[i] for i in range(n)])
        t.append(s * y)
    for i in range(n):
        rows.append([0.0] * (i + 1) + [math.sqrt(ridge * sum(ws))] + [0.0] * (n - i - 1))
        t.append(0.0)
    c = fd.lstsq(rows, t)
    if c is None:
        sys.exit('fit_generic: singular fit')
    coef = [c[0] - sum(c[i + 1] * mu[i] / sd[i] for i in range(n))] + [c[i + 1] / sd[i] for i in range(n)]
    return coef


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ridge', type=float, default=0.3)
    ap.add_argument('--name', default='')
    a = ap.parse_args()
    cal = json.load(open(a.calibration))
    names = sorted(set(re.findall(r'hw\.module (?:private )?@([\w$.]+)\(', open(a.models).read())))
    feats = fd.class_features(a.models, names)
    fams = []
    for fam in cal['families']:
        cls = [n for n in names if n in feats and re.search(fam['match'], n)
               and not any(re.search(g['match'], n) for g in cal['families'][:cal['families'].index(fam)])]
        if not cls:
            continue
        w = [10 ** feats[n]['log_lut'] for n in cls]
        f = {k: sum(feats[n][k] * x for n, x in zip(cls, w)) / sum(w) for k in fd.GENERIC_FEATURES}
        fams.append(dict(family=fam['family'], classes=cls, lut=sum(w), feats=f,
                         logic=fam['logic'], lutram=fam['lutram']))
    if len(fams) < len(fd.GENERIC_FEATURES) + 2:
        sys.exit(f'fit_generic: {len(fams)} families with classes in --models, too few')
    xs = [[f['feats'][k] for k in fd.GENERIC_FEATURES] for f in fams]
    ws = [math.sqrt(f['lut']) for f in fams]
    out = dict(cal)
    out.pop('families', None)
    out['families'] = []
    out['name'] = a.name or os.path.splitext(os.path.basename(a.out))[0]
    out['source'] = (f"design-independent: {os.path.basename(a.calibration)}'s terms, its families replaced by a "
                     f"generic model of the class cost features (fit_generic.py, ridge {a.ridge}) fitted on "
                     f"{len(fams)} families' classes in {a.models}")
    gen = {}
    report = []
    for key in ('logic', 'lutram'):
        ys = [math.log(max(f[key], 1e-3)) for f in fams]
        coef = fit(xs, ys, ws, a.ridge)
        lo, hi = min(f[key] for f in fams), max(f[key] for f in fams)
        model = dict(features=list(fd.GENERIC_FEATURES), coef=[round(c, 6) for c in coef], lo=lo, hi=hi)
        loo = {}
        for i, f in enumerate(fams):
            c = fit(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:], ws[:i] + ws[i + 1:], a.ridge)
            p = fd.generic_factor(dict(model, coef=c), f['feats'])
            loo[f['family']] = dict(fitted=round(fd.generic_factor(model, f['feats']), 3), loo=round(p, 3),
                                    family=f[key])
        # The LUT-weighted error of the families' factors, fitted and left out.
        tot = sum(f['lut'] * f[key] for f in fams)
        err_fit = sum(f['lut'] * loo[f['family']]['fitted'] for f in fams) / tot - 1
        err_loo = sum(f['lut'] * loo[f['family']]['loo'] for f in fams) / tot - 1
        model['fit'] = dict(families=loo, weighted_error=round(err_fit, 4), weighted_loo_error=round(err_loo, 4))
        gen[key] = model
        report.append(f'{key}: weighted error {err_fit * 100:+.1f} %, left out {err_loo * 100:+.1f} %')
    out['generic'] = gen
    json.dump(out, open(a.out, 'w'), indent=1)
    for f in fams:
        print(f"{f['family']:14s} lut {f['lut']:9.0f} " +
              ' '.join(f"{k} {f['feats'][k]:.3f}" for k in fd.GENERIC_FEATURES) +
              f" | logic {f['logic']:.3f} fit {gen['logic']['fit']['families'][f['family']]['fitted']:.3f}"
              f" loo {gen['logic']['fit']['families'][f['family']]['loo']:.3f}"
              f" | lutram {f['lutram']:.3f} fit {gen['lutram']['fit']['families'][f['family']]['fitted']:.3f}"
              f" loo {gen['lutram']['fit']['families'][f['family']]['loo']:.3f}")
    print('\n'.join(report))


if __name__ == '__main__':
    main()
