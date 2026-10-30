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

# Routability: the load and the SLRs it may spread over (synthesized LUTs,
# RAMB36, URAM288). Out of context (the device's 3 SLRs) f_hs, f_k2 and
# o8_t8n route; o8_t8sf, at o8_t8n's load but held in 2 SLRs by its 2 SLR
# groups, does not. In the image's region (1.8 SLRs) img_int routes and
# img_ii8 (7 SLR groups: no fewer SLRs) does not.
print('slrs: region', s.route_slrs(), 'groups 1', s.route_slrs(1), 'groups 2', s.route_slrs(2),
      'ooc groups 2', s.route_slrs(2, s.DEVICE_SLRS),
      'ooc groups 7', s.route_slrs(7, s.DEVICE_SLRS))
for name, luts, bram, uram, slrs in (
        ('f_hs', 905171, 514, 64, s.DEVICE_SLRS), ('img_int', 908019, 514, 64, s.route_slrs()),
        ('f_k2', 1179619, 649, 64, s.DEVICE_SLRS), ('o8_t8n', 1460935, 849, 64, s.DEVICE_SLRS),
        ('o8_t8sf', 1432069, 592, 64, s.route_slrs(2, s.DEVICE_SLRS)),
        ('img_ii8', 1461670, 849, 64, s.route_slrs(7))):
    c = s.congestion(s.route_load(luts, bram, uram), slrs)
    print(f'{name} luts {luts} slrs {slrs:g}: congestion {c:.1f} routes {c <= s.ROUTE_LIMIT}')
print(f'route limit {s.route_luts():.0f} LUTs in the region, '
      f'{s.route_luts(s.DEVICE_SLRS):.0f} over 3 SLRs, {s.route_luts(1):.0f} in one')

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
# SLR groups: one copy of a single class kept whole in one SLR. At 1.1M of
# logic it routes over the region's 1.8 SLRs, not in one.
one = dict(Top=dict(lut=1.1e6, ff=0, mems=[], regw={}))
for groups in (False, True):
    a = d.area(None, one, dict(Top=1), {}, 0, dict(Top=dict(n=1, ready=[], state=[0], gap=1)),
               1, 2, 'auto', True, slr_groups=groups)
    print(f"slr groups {groups}: luts {a['luts']} congestion {a['congestion']} "
          f"routes {a['routes']}")
# Virtual memories (foldsim --virtual-memories): a 64 MiB L2's 64 data
# stores (16384 x 512, a registered read) do not place on chip; taken off
# chip by the fold options' regex they cost only their host caches: with
# --vmem-cache-rows auto, 1024 lines each would take 64 x 14.5 RAMB36, so
# the costliest halve until the caches fit the 120-RAMB36 budget.
data = [[f'l2/bank{b}/data{k}', 16384, 512, 1, 1, 1, 'block', 64, True]
        for b in range(32) for k in range(2)]
tags = [['l2/tags', 1024, 416, 1, 1, 1, 'block', 16, True]]
top = dict(Top=dict(lut=500000.0, ff=0, mems=data + tags, regw={}))
on = d.split_virtual(top, '')
off = d.split_virtual(top, 'l2/.*/data[0-9]+')
print(f"virtual: mems {len(off['Top']['mems'])} vmems {len(off['Top']['vmems'])}")
one_sched = dict(Top=dict(n=1, ready=[], state=[0], gap=8))
for name, e, vmem in (('on chip', on, None), ('in hbm, no cache', off, None),
                      ('in hbm, auto cache', off, ('auto', s.VMEM_CACHE_BUDGET))):
    a = d.area(None, e, dict(Top=1), {}, 0, one_sched, 8, 2, 'auto', False, vmem=vmem)
    print(f"{name}: fits {a['fits']} bram {a['bram']} uram {a['uram']} "
          f"cache {a.get('vmem_cache_bram', 0)}")
rows = s.vmem_cache_rows([dict(instances=1, row_width=14, data_width=512)] * 64, 'auto',
                         s.VMEM_CACHE_BUDGET)
print(f"cache rows {min(rows)}..{max(rows)}, {d.vmem_bram(off, dict(Top=1), {}, ('auto', 120))} RAMB36")
