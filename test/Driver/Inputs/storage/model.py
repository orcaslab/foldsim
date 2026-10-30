"""Checks of the storage choice and the routability term (foldsim_storage.py,
foldsim_degree.area): argv[1] is the foldsim driver, beside the modules."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_degree as d  # noqa: E402
import foldsim_storage as s  # noqa: E402


def reg(gap, depth=256, width=72):
    return dict(depth=depth, width=width, rports=1, wports=1, rlat=0, mask=1, reg=True,
                gap=gap, threads=depth)


def mem(rlat, wports=1, depth=4096, width=72, mask=1):
    return dict(depth=depth, width=width, rports=1, wports=wports, rlat=rlat, mask=mask,
                reg=False, gap=None, threads=16)


# A register store reads combinationally: block or ultra RAM only at a
# state gap of 4 (read a host cycle early).
print('reg gap 3: block', s.kind_cost(reg(3), 'block'), 'ultra', s.kind_cost(reg(3), 'ultra'))
c = s.kind_cost(reg(4), 'block')
print(f"reg gap 4: block {c['bram']} ultra {s.kind_cost(reg(4), 'ultra')['uram']} "
      f"distributed {s.kind_cost(reg(4), 'distributed')['lutram']:.0f}")
# A memory read combinationally takes neither; a registered read takes both.
print('mem rlat 0: block', s.kind_cost(mem(0), 'block'), 'ultra', s.kind_cost(mem(0), 'ultra'))
print(f"mem rlat 1 4096x72: block {s.kind_cost(mem(1), 'block')['bram']} "
      f"ultra {s.kind_cost(mem(1), 'ultra')['uram']}")
# Masked by 15-bit lanes: a RAM column a lane; by bytes: the byte enables.
print(f"mem 256x120 mask 8: block {s.kind_cost(mem(1, depth=256, width=120, mask=8), 'block')['bram']}"
      f", 512x64 mask 8: block {s.kind_cost(mem(1, depth=512, width=64, mask=8), 'block')['bram']}")
# Two write ports: no RAM primitive serves them in one host cycle.
print('mem two writes: block floor', s.kind_cost(mem(1, wports=2), 'block')['ii_floor'],
      'distributed floor', s.kind_cost(mem(1, wports=2), 'distributed')['ii_floor'])

# Routability: the folds near 0.89M routed, those near 1.45M did not
# (synthesized LUTs, RAMB36, URAM288 of f_hs, img_int, f_k2, o8_t8sf, img_ii8).
for luts, bram, uram in ((905171, 514, 64), (908019, 514, 64), (1179619, 649, 64),
                         (1432069, 592, 64), (1461670, 849, 64)):
    c = s.congestion(s.route_load(luts, bram, uram))
    print(f'luts {luts}: congestion {c:.1f} routes {c <= s.ROUTE_LIMIT}')
print(f'route limit {s.route_luts():.0f} LUTs')

# A class of 256 lanes on one copy, its registers 64 to 80 bits wide, 3,000
# of them: in distributed RAM it does not place; with a state gap of 4 the
# choice moves registers into block and ultra RAM until it does.
est = dict(Lane=dict(lut=20000.0, ff=3000 * 72, mems=[], regw={64: 1000, 72: 1000, 80: 1000}),
           Top=dict(lut=100000.0, ff=0, mems=[], regw={}))
classes = dict(Lane=256, Top=1)


def sched(gap):
    return dict(Lane=dict(n=256, ready=[], state=list(range(256)), gap=gap),
                Top=dict(n=1, ready=[], state=[0], gap=256))


for store, gap, route in (('fixed', 300, False), ('auto', 300, False), ('auto', 2, False),
                          ('auto', 300, True)):
    a = d.area(None, est, classes, {}, 0, sched(gap), 256, 2, store, route)
    print(f"{store} gap {gap}: luts {a['luts']} sites {a['sites']} fits {a['fits']} "
          f"bram {a['bram']} uram {a.get('uram', '-')} routes {a.get('routes', '-')} "
          f"storage {a.get('storage', '-')}")
# Routability: 500 more registers, 80 bits wide, priced into distributed
# RAM at placement; at 1.15M LUTs of logic the fold then takes 1.39M LUTs,
# past the routable 1.33M, and the choice moves them into ultra RAM. At 1.35M
# of logic no storage makes it route.
est['Lane']['regw'] = {64: 1000, 72: 1000, 80: 500}
for logic, route in ((1.15e6, False), (1.15e6, True), (1.35e6, True)):
    est['Top']['lut'] = logic
    a = d.area(None, est, classes, {}, 0, sched(300), 256, 2, 'auto', route)
    print(f"logic {logic:.0f} route {route}: luts {a['luts']} fits {a['fits']} bram {a['bram']} "
          f"uram {a['uram']} congestion {a.get('congestion', '-')} routes {a.get('routes', '-')} "
          f"storage {a['storage']}")
