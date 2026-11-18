#!/usr/bin/env python3
"""foldsim_h_check.py <gen dir> <gate H> [--tol 0.01] (a32-R4 11-17): the gate's measured H against what the generation promised.

The fold's II (schedule.txt) and the --state auto projection ("H at most N" in the gen log) bound the host cycles a target
cycle when no memory is in HBM; a gate H above max(II, projection) * (1 + tol) is a per-frame cost nobody priced (F-C1: the
policy's block-RAM link, +8 a frame at 4 SMs for two days). Exit 0 OK, 1 OVER (prints the excess per frame), 2 no data.
With virtual memories (FOLDSIM_VMEM_COUNT in foldsim_sim.vh) the bound is reported, not enforced (HBM stalls are workload- and
latency-dependent: compare with the latency sweep instead)."""
import re, sys, os, glob
g = sys.argv[1]; h = float(sys.argv[2]); tol = 0.01
if '--tol' in sys.argv: tol = float(sys.argv[sys.argv.index('--tol') + 1])
sch = os.path.join(g, 'gen', 'schedule.txt')
ii = None
if os.path.exists(sch):
    m = re.search(r'II = (\d+)', open(sch).read(2000)); ii = int(m[1]) if m else None
proj = None
for lg in [g + '.log', *glob.glob(os.path.join(g, '*.log')), *glob.glob(os.path.join(g, 'gen', '*.log'))]:
    if os.path.exists(lg):
        m = re.search(r'state auto: frame \d+ \(II \d+\), H at most (\d+)', open(lg, errors='replace').read())
        if m: proj = int(m[1]); break
vh = os.path.join(g, 'gen', 'foldsim_sim.vh')
vmem = os.path.exists(vh) and re.search(r'^`define FOLDSIM_VMEM_COUNT', open(vh).read(), re.M)
bound = max(x for x in (ii, proj) if x is not None) if (ii or proj) else None
if bound is None:
    print(f'h_check: {g}: no II / projection'); sys.exit(2)
over = h - bound
v = 'OVER' if h > bound * (1 + tol) else 'OK'
print(f'h_check: {os.path.basename(g)}: gate H {h:.2f}, II {ii}, projected {proj}, bound {bound}: {v}'
      f' ({over:+.2f} host cycles a frame){" [virtual memories: reported only]" if vmem else ""}')
sys.exit(1 if v == 'OVER' and not vmem else 0)
