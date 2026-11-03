#!/usr/bin/env python3
"""check.py <calibration.json> <state args json> [--tol PCT]: does a whole-design calibration reproduce the synthesis it
was fitted on? Prices the same netlist twice with foldsim_degree's `state` (args from the JSON list, e.g. a run's
argv with --kernel-terms --links-as-built --no-page): uncalibrated, recording each store's kind, then calibrated with
every store held at that kind (the kinds the netlist was built with; the calibrated costs would otherwise move the
choice). Compares the calibrated logic, LUTRAM and total (routed x SYNTH_PER_ROUTED) with the calibration's
measured_total; exit 1 if any is off by more than --tol (3 %)."""
import io, json, os, sys, contextlib
here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(here))
import foldsim_degree as fd  # noqa: E402
import foldsim_storage as st  # noqa: E402

cal_path, args_path = sys.argv[1], sys.argv[2]
tol = float(sys.argv[sys.argv.index('--tol') + 1]) if '--tol' in sys.argv else 3.0
cal = json.load(open(cal_path)); argv = json.load(open(args_path))
out = argv[argv.index('--out') + 1]
key = lambda u: (u.get('class'), u.get('store'), bool(u.get('link')))
kinds, orig = {}, st.assign


def record(units, base, caps, extra=None, paging=False, price_extra=False):
    r = orig(units, base, caps, extra, paging, price_extra)
    kinds.clear()
    for u, ch in zip(units, r[0]):
        (k, _), = ch.items()
        kinds[key(u)] = k
    return r


def replay(units, base, caps, extra=None, paging=False, price_extra=False):
    for u in units:
        if key(u) in kinds and kinds[key(u)] in u['costs']:
            u['fixed'] = kinds[key(u)]
    return orig(units, base, caps, extra, paging, price_extra)


def run(extra, assign):
    st.assign = assign
    fd.KERNEL.update(calibration=None)
    with contextlib.redirect_stdout(io.StringIO()):
        sys.argv = ["foldsim_degree.py"] + argv + extra
        rc = fd.main()
    return rc, json.load(open(out))['res']


rc0, r0 = run([], record)
rc1, r1 = run(['--calibrate', cal_path, '--route-limit', '99'], replay)
S = cal['synth_per_routed']; m = cal['measured_total']
ok = True
for name, model, meas in (('logic', r1['logic'], m['logic']), ('LUTRAM', r1['lutram'], m['lutram']),
                          ('total', r1['logic'] + r1['lutram'], m['logic'] + m['lutram'])):
    err = 100.0 * (model * S / meas - 1.0)
    ok &= abs(err) <= tol
    print(f"{name:7s} calibrated {model * S:12,.0f} measured {meas:12,} {err:+6.2f} %  (uncalibrated "
          f"{(r0['logic'] if name == 'logic' else r0['lutram'] if name == 'LUTRAM' else r0['logic'] + r0['lutram']) * S:,.0f})")
print(f"RAMB36 calibrated {r1['bram']} measured {m['ramb36']} + {m['ramb18']} RAMB18; URAM {r1['uram']} measured {m['uram']}")
print('PASS' if ok else 'FAIL', f'(tolerance {tol} %)')
sys.exit(0 if ok else 1)
