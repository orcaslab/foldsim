"""--state auto in the cost model (foldsim_degree.choose_state), on made-up
classes: argv[1] is the foldsim driver, beside the modules."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_degree as d  # noqa: E402
import foldsim_storage as s  # noqa: E402

# What a virtual memory may be: a registered read, one read and at most one
# write port, byte enables, a row the controller takes, a state gap of 5.
gpr = ('gpr/ram', 64, 2048, 1, 1, 1, 'block', 256, True)
for m, gap in ((gpr, 100), (gpr, 4), (('q', 16, 64, 1, 1, 0, None, 1, True), 100),
               (('w', 16, 8192, 1, 1, 1, None, 1, True), 100),
               (('m', 16, 64, 1, 1, 1, None, 4, True), 100)):
    print(f'eligible {m[0]} gap {gap}: {s.vmem_eligible(m, gap) or "yes"}')
c = s.vmem_cost(64, 2048, 512)
print(f'vmem 512 x 64 x 2048: hbm {c["hbm"]:.0f} lutram {c["lutram"]} bram {c["bram"]} '
      f'lut {c["lut"]:.0f}')
# The calibrated prices (st32_1031 §10.2; the controllers routed since virtA_1105: ctl_cost): a page bit's write-back select and
# a virtual memory's controller (logic, LUTRAM) by row width.
print('select', [s.page_select_luts(p) for p in (1, 2, 3, 4, 8, 16, 31)])
print('controller', [tuple(round(x) for x in s.vmem_ctl(w)) for w in (64, 512, 1024, 4096)])
print('pattern', d.tiers_pattern([('Opc_copy1', 'gpr/ram'), ('Opc', 'a.b[0]')]))


def sched(classes, gap=None, frame=4096):
    # Each copy's state passes spread over the frame.
    return {c: dict(n=n, ready=[], state=[i * (frame // n) for i in range(n)],
                    gap=gap or 1000, page=0)
            for c, n in classes.items()}


# An SM-like fold: 512 operand units, each six 64 x 2048 register-file banks
# (403 Mbit) and 20,000 register bits; a frame of 4,096. On chip the banks
# would take 1,392 URAM288 (232 each), more than the region's 1,211; as many
# go to HBM as the rest needs to fit and route, their traffic (256 KiB a
# frame each) within the frame's share, so H stays the frame.
classes = dict(Opc=512, Top=1)
gprs = [(f'g{k}/ram', 64, 2048, 1, 1, 1, 'block', 256, True) for k in range(6)]
est = dict(Opc=dict(lut=20000.0, ff=20000, mems=gprs, regw={20000: 1}),
           Top=dict(lut=50000.0, ff=1000, mems=[], regw={}))
out = d.choose_state(None, est, classes, sched(classes), 4096, vmem=('auto', 120))
print(f"opc: page {out['page']!r} frame {out['frame']} H {out['h']['h']} "
      f"hbm {out['h']['hbm_bytes']} fits {out['res']['fits']} routes {out['res']['routes']}")
print('opc moved', sorted(m for _, m in out['moved']), 'pages off', out['pages_offchip'])
print('opc pattern', out['pattern'])
# A small fold fits as it is: nothing moves.
est['Opc'] = dict(lut=20000.0, ff=2000, mems=[('g/ram', 64, 32, 1, 1, 1, 'block', 4, True)],
                  regw={2000: 1})
out = d.choose_state(None, est, dict(Opc=64, Top=1), sched(dict(Opc=64, Top=1), frame=256), 256)
print(f"small: moved {out['moved']} H {out['h']['h']} fits {out['res']['fits']}")
# With routing headroom asked for (foldsim_degree.py state --route-limit): the
# same fold, its congestion estimate against the lower limit.
limit = s.ROUTE_LIMIT
s.ROUTE_LIMIT = out['res']['congestion'] - 0.01
low = d.choose_state(None, est, dict(Opc=64, Top=1), sched(dict(Opc=64, Top=1), frame=256), 256)
print(f"headroom: congestion {out['res']['congestion']} routes below it: {low is not None and low['res'].get('routes')}")
s.ROUTE_LIMIT = limit
# 200,000 register bits an instance (102 Mbit), a store of 2,778 72-bit columns
# read whole each host cycle, take more block RAM than the region has: Opc
# pages (as deep as half the frame lets its state passes), and
# a memory read combinationally (a 24 x 512 table) pages as rows where its
# distributed RAM costs more: the choice names it (two register-file banks,
# so that ultra RAM is not the pool the choice balances against).
est["Opc"] = dict(lut=20000.0, ff=200000,
                  mems=gprs[:2] + [("ptw", 24, 512, 1, 1, 0, None, 1, True)], regw={200000: 1})
out = d.choose_state(None, est, classes, sched(classes), 4096, vmem=('auto', 120))
print(f"paged: page {out['page']!r} rows {out['rows']} fits {out['res']['fits']}")
# Reads younger than the HBM round trip wait: a state gap of 40 for every
# instance of a class with virtual memories.
h = d.project_h(4096, classes, sched(classes, gap=40), est, '^Opc:g0/ram$')
print(f"bubbles: H {h['h']} bubbles {h['bubbles']} per {h['per']}")
