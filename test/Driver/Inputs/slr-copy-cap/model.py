"""Checks of the per-copy SLR caps (foldsim_degree COPY_CAP, copy_cap_pools;
copy_sites with a schedule): argv[1] is the foldsim driver, beside the modules."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_degree as d  # noqa: E402
import foldsim_storage as s  # noqa: E402


def wide(threads=64):
    # A copy's wide, shallow memory (4 entries an instance, 6,298 bits): a
    # whole URAM288 column per 72 bits however few rows it fills.
    return dict(depth=4 * threads, width=6298, rports=1, wports=1, rlat=1, mask=1, reg=False,
                gap=None, threads=threads)


def units():
    return [d.unit('Big', f'buf{k}', [(0, wide(), 1)], 1.0, None) for k in range(2)]


# Block RAM dear, URAM cheap, distributed RAM dearer (28.7k LUTs each): both
# memories in URAM (176) without a cap.
caps = dict(sites=150000.0, bram=200.0, uram=1000.0)
base = dict(sites=0.0, bram=0.0, uram=0.0)
u = units()
choice, used = s.assign(u, dict(base), dict(caps))
print('no cap:', [list(ch) for ch in choice], 'uram', used['uram'], 'bram', used['bram'])
# A copy's RAM within one SLR's (100 URAM): one memory moves to block RAM.
d.COPY_CAP = dict(name='SLRx', uram=100.0, bram=150.0)
u, c2, b2 = units(), dict(caps), dict(base)
d.copy_cap_pools(u, c2, b2)
choice, used = s.assign(u, b2, c2)
print('cap:', sorted(k for ch in choice for k in ch), 'copy uram', used['uram@Big#0'],
      'copy bram', used['bram@Big#0'], 'cap', c2['uram@Big#0'])
d.COPY_CAP = None
# The region's SLR caps: the smallest by LUTs, by name, with the margin.
region = dict(margin=0.5, slrs=[dict(name='A', luts=100, ramb36=40, uram=20),
                                dict(name='B', luts=10, ramb36=8, uram=2)])
print('smallest', d.slr_copy_cap(region), 'A', d.slr_copy_cap(region, 'A'))
# copy_sites: a copy of 64 instances keeps its 40,000 register bits as a
# distributed RAM of an entry per instance without a schedule; paged (the
# schedule's `page 8`), in page RAMs: far fewer sites.
est = dict(lut=10000.0, ff=40000, mems=[], regw={40000: 1})
plain = d.copy_sites(None, est, 'Big', 64)
paged = d.copy_sites(None, est, 'Big', 64, info=dict(n=64, gap=20, page=8, offchip=False,
                                                      ready=[], state=[]), ii=2000)
print(f'copy sites: unscheduled {plain:.0f}, paged {paged:.0f}, paged smaller {paged < plain / 2}')
