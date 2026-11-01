"""foldsim_degree.py state --exact: the kinds as a 0-1 program (foldsim_storage
assign_exact), on made-up stores: argv[1] is the foldsim driver, beside the
modules."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_degree as d  # noqa: E402
import foldsim_storage as s  # noqa: E402


def store(lutram, bram):
    zero = dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0.0, ii_floor=0)
    return dict(costs=dict(distributed=dict(zero, lutram=lutram), block=dict(zero, bram=bram)),
                count=1, fixed=None, store='m', **{'class': 'C'})


# Three stores, 10 RAMB36 for them: block RAM is worth most where it frees
# the most distributed RAM per RAMB36 it takes, but the two that fit
# together free more than the largest alone: B and C in block RAM, A in
# distributed (1,000 LUTs plus their RAMs' routing).
units = [store(1000.0, 6), store(900.0, 5), store(800.0, 5)]
caps = dict(sites=1e9, bram=10, uram=0)
choice, used = s.assign_exact(units, {}, caps, dict(luts=1e9))
print('exact', [list(c)[0] for c in choice], 'bram', used['bram'], 'load', round(used['luts']))
# A load cap no choice meets: the least load (the caller sees it over).
choice, used = s.assign_exact(units, {}, caps, dict(luts=500.0))
print('over', [list(c)[0] for c in choice], 'load', round(used['luts']))
# Caps no choice meets: none.
print('none', s.assign_exact([store(1000.0, 6)], {'sites': 2e9}, caps, None))
# assign takes it with ASSIGN_EXACT (and --state auto's priced pools).
s.ASSIGN_EXACT = True
choice, used = s.assign(units, {}, caps, dict(luts=1e9), price_extra=True)
print('assign', [list(c)[0] for c in choice])


def sched(classes, frame=4096):
    return {c: dict(n=n, ready=[], state=[i * (frame // n) for i in range(n)], gap=1000, page=0)
            for c, n in classes.items()}


# The SM-like fold of state-model.test, chosen exactly: it fits and routes,
# H the frame.
classes = dict(Opc=512, Top=1)
gprs = [(f'g{k}/ram', 64, 2048, 1, 1, 1, 'block', 256, True) for k in range(6)]
est = dict(Opc=dict(lut=20000.0, ff=20000, mems=gprs, regw={20000: 1}),
           Top=dict(lut=50000.0, ff=1000, mems=[], regw={}))
out = d.choose_state(None, est, classes, sched(classes), 4096, vmem=('auto', 120))
print(f"opc: frame {out['frame']} H {out['h']['h']} fits {out['res']['fits']} "
      f"routes {out['res']['routes']} moved {len(out['moved'])}")
