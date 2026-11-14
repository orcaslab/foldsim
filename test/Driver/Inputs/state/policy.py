"""The state-home policy of --state auto (foldsim_degree.state_policy): a
per-copy budget in synthesized LUTs within the smallest SLR, stores moved on
chip (block or ultra RAM, a page RAM) before HBM, the --storage rules given
kept, the calibration's factors and its fit (state_fit). Made-up classes;
argv[1] is the foldsim driver, beside the modules."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_degree as d  # noqa: E402
import foldsim_storage as s  # noqa: E402


def sched(classes, gap=None, frame=256):
    return {c: dict(n=n, ready=[], state=[i * (frame // n) for i in range(n)],
                    gap=gap or 100, page=0)
            for c, n in classes.items()}


# A region of one small SLR (120k LUTs, 20k of them LUTRAM-capable) and a
# large one: the budget is the small one's (75 % over placed / synthesized
# 1.0: 90,000 LUTs, 15,000 LUTRAM), its RAM within the region's margin.
REGION = dict(margin=0.85, placed_per_synthesized=dict(lut=1.0),
              slrs=[dict(name='big', luts=600000, lutram_luts=300000, ramb36=1000, uram=500),
                    dict(name='small', luts=120000, lutram_luts=20000, ramb36=300, uram=100)])
cap = d.copy_budget(REGION, 0.75)
print(f"budget {cap['name']} lut {cap['lut']:.0f} lutram {cap['lutram']:.0f} bram {cap['bram']:.0f} "
      f"uram {cap['uram']:.0f}; named {d.copy_budget(REGION, 0.75, 'big')['lut']:.0f}")
classes = dict(Slice=32, Top=1)
TOP = dict(lut=5000.0, ff=100, mems=[], regw={})


def run(tag, est):
    out = d.choose_state(None, est, classes, sched(classes), 256, route=False)
    if out is None:
        print(f'{tag}: none')
        return None
    c = out['res']['copies']['Slice#0']
    rules = sorted(r for r in out['storage'].split(',') if r.startswith('Slice:'))
    print(f"{tag}: copy {c['total']} lutram {c['store_lutram']} H {out['h']['h']} page {out['page'] or '-'} "
          f"off {out['pages_offchip']} moved {sorted(m for _, m in out['moved'])} rules {rules}")
    if d.COPY_CAP:
        assert c['total'] <= cap['lut'] + 1e-6 and c['store_lutram'] <= cap['lutram'] + 1e-6, tag
    return out


# The policy as the driver sets it (state_policy: no calibration file here,
# the region from a file in argv[2]).
import json  # noqa: E402
os.makedirs(sys.argv[2], exist_ok=True)
json.dump(REGION, open(os.path.join(sys.argv[2], 'region.json'), 'w'))
d.state_policy('', '0.75', os.path.join(sys.argv[2], 'region.json'))
policy = d.COPY_CAP
d.COPY_CAP = None
d.STATE_CAL = dict(copy_logic=1.0, copy_lutram=1.0, level=1.0)
# A slice-like class: 32 instances of 20k logic LUTs, 40,000 register bits
# (twenty stores 2,000 wide) and four 2 x 256 queues read through a register.
# Its register stores in distributed RAM (the cheapest kind) pass the
# budget's LUTRAM; in block RAM at a depth of 32 they would take 560 RAMB36,
# past the SLR's: the class pages on chip (a page RAM in block RAM), H the
# frame.
regs = dict(Slice=dict(lut=20000.0, ff=40000, regw={2000: 20},
                       mems=[(f'q{k}/ram', 2, 256, 1, 1, 1, None, 1, True) for k in range(4)]), Top=TOP)
run('regs free', regs)
d.COPY_CAP = policy
run('regs policy', regs)
# A class of queues: eight 2 x 2048 queues an instance (in distributed RAM,
# the cheapest kind at a depth of 64, past the budget's LUTRAM): to block or ultra
# RAM within the budget, H the frame; in an SLR of little block and ultra RAM
# (60 RAMB36, 10 URAM288: the copy's own spent), as many as the budget needs
# leave the chip (virtual memories in HBM: their controllers are the
# kernel's, outside the copy).
queues = dict(Slice=dict(lut=20000.0, ff=2000, regw={2000: 1},
                         mems=[(f'q{k}/ram', 2, 2048, 1, 1, 1, None, 1, True) for k in range(8)]), Top=TOP)
d.COPY_CAP = None
run('queues free', queues)
d.COPY_CAP = cap
qp = run('queues policy', queues)
print(f"queues policy RAM {qp['res']['bram']:g} {qp['res']['uram']:g}")
d.COPY_CAP = d.copy_budget(dict(REGION, slrs=[dict(REGION['slrs'][1], ramb36=60, uram=10)]), 0.75)
run('queues spent', queues)
# Sixteen such queues: more HBM bytes than the schedule's frame carries; the
# frame grows (unpaged) until they fit, H with it.
more = dict(queues, Slice=dict(queues['Slice'], mems=[(f'q{k}/ram', 2, 2048, 1, 1, 1, None, 1, True)
                                                      for k in range(16)]))
run('queues longer', more)
d.COPY_CAP = cap
# One large SLR (a copy budget of 450k that does not bind) and the region's
# LUTs at 6 % of its 600k (36k, below the queues' free 54k with the rest):
# the stores leave distributed RAM until the whole design is within it.
d.COPY_CAP = d.copy_budget(dict(REGION, slrs=REGION['slrs'][:1]), 0.75, region_frac=0.06)
out = run('queues region', queues)
print(f"region cap {d.COPY_CAP['region_lut']:.0f} used {out['res']['luts']} within {out['res']['luts'] <= d.COPY_CAP['region_lut']}")
d.COPY_CAP = cap
# The region's RAM (--state-ram-budget) at 6 % of the V80 region's (144.9
# RAMB36, 72.7 URAM288; the copy budget alone took 199.5 RAMB36): stores
# leave block RAM until the design is within.
d.COPY_CAP = dict(cap, region_ram=0.06)
out = run('queues ram', queues)
print(f"ram cap {0.06 * s.REGION['bram']:.1f} {0.06 * s.REGION['uram']:.1f} used "
      f"{out['res']['bram']:g} {out['res']['uram']:g} within "
      f"{out['res']['bram'] <= 0.06 * s.REGION['bram'] and out['res']['uram'] <= 0.06 * s.REGION['uram']}")
d.COPY_CAP = cap
# A given rule (--storage) fixes its store: priced at its kind, no rule
# emitted for it, the others still chosen within the budget.
d.GIVEN = d.given_rules('Slice:q0/ram=distributed,Top:regs>=1=registers')
run('queues given', queues)
d.GIVEN = None
# The calibration's factors: a class's own (measured) before the defaults.
d.STATE_CAL = dict(copy_logic=1.5, copy_lutram=1.25, level=1.0, classes=dict(Slice=dict(logic=2.0, lutram=1.0)))
print('factors', d.state_factors('Slice_copy1'), d.state_factors('Top'))
run('queues measured', queues)
d.STATE_CAL, d.COPY_CAP = None, None

# The repair's order: a pool over its cap, a store that block RAM or HBM
# relieves, HBM the cheaper by price; on chip first, off chip by price.
def cost(**kw):
    return dict(dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0.0, ii_floor=0), **kw)


units = [dict(costs=dict(distributed=cost(lutram=100.0), block=cost(bram=15.0), hbm=cost(hbm=1.0)),
              count=1, store='q', **{'class': 'A'})]
for first in (True, False):
    s.ONCHIP_FIRST = first
    ch, _ = s.assign([dict(u) for u in units], dict(sites=800.0), dict(sites=1000.0, bram=20.0, hbm=1000.0))
    print(f'onchip-first {first}: {list(ch[0])}')
s.ONCHIP_FIRST = False
# --state-home onchip: block RAM spent (cap 10), HBM never admissible: the
# store stays in distributed RAM (the pool stays over), never off chip.
s.ONCHIP_ONLY = True
for first in (True, False):
    s.ONCHIP_FIRST = first
    ch, _ = s.assign([dict(u) for u in units], dict(sites=800.0), dict(sites=1000.0, bram=10.0, hbm=1000.0))
    print(f'onchip-only first {first}: {list(ch[0])}')
s.ONCHIP_ONLY = s.ONCHIP_FIRST = False

# state_fit: two copies against a report; the level and the kernel by
# difference.
state = dict(res=dict(luts=1000, copies={'A#0': dict(logic=300, store_lut=0, store_lutram=100, total=400),
                                          'B#0': dict(logic=200, store_lut=0, store_lutram=0, total=200)}))
hier = dict(A_copy=dict(lut=550, lutram=150, ff=0), B_copy=dict(lut=260, lutram=0, ff=0),
            sim=dict(lut=1300, lutram=300, ff=0), top_wrapper=dict(lut=1800, lutram=350, ff=0))
cal, rows = d.state_fit(state, hier)
print('fit', {k: cal[k] for k in ('copy_logic', 'copy_lutram', 'level', 'kernel_logic', 'kernel_lutram')})
