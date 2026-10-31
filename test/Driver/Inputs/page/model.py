"""Paged register state in the cost model (foldsim_storage.page_cost,
foldsim_degree.choose_pages), on made-up classes: argv[1] is the foldsim
driver, beside the modules."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_degree as d  # noqa: E402
import foldsim_storage as s  # noqa: E402

# The page RAM of a copy of 4 instances, 100 bits, at p = 2 and 3.
for p in (1, 2, 3):
    rows, pw, ports = s.page_geometry(100, 4, p)
    c = s.page_cost(100, 4, p, 'block')
    print(f'p {p}: rows {rows} pw {pw} ports {ports} bram {c["bram"]} ff {c["ff"]} lut {c["lut"]} '
          f'frame {c["ii_floor"]} bound {s.page_bound(100, 4, ports)}')
print('distributed', s.page_cost(100, 4, 2, 'distributed'))
# A deep copy: 512 instances of 4,000 bits at p = 8 in ultra RAM (4K rows).
c = s.page_cost(4000, 512, 8, 'ultra')
print(f'ultra 512 x 8: uram {c["uram"]} ports {c["ports"]} frame {c["ii_floor"]}')


def lane(ff):
    return dict(lut=20000.0, ff=ff, mems=[], regw={ff: 1})


def sched(n, page=0):
    return dict(Lane=dict(n=n, ready=[], state=list(range(n)), gap=n, page=page),
                Top=dict(n=1, ready=[], state=[0], gap=1, page=0))


classes = dict(Lane=256, Top=1)
est = dict(Lane=lane(2000), Top=dict(lut=10000.0, ff=1000, mems=[], regw={}))
spec, frame, res = d.choose_pages(None, est, classes, sched(256), 256)
print(f'small state: page {spec!r} frame {frame} fits {res["fits"]}')
# 256 instances of 300,000 bits: unpaged, the registers' distributed RAM
# does not place; paged at the frame of 256 (p = 1), the page word takes
# more block RAM than the device has; at 512 (p = 2) it fits.
est['Lane'] = lane(300000)
un = d.area(None, est, classes, {}, 0, sched(256), 256, 2, 'fixed')
print(f'unpaged: lutram {un["lutram"]} fits {un["fits"]}')
one = d.area(None, est, classes, {}, 0, sched(256, 1), 256, 2, 'fixed')
print(f'p 1: bram {one["bram"]} ports {one["page_ports"]} lutram {one["lutram"]} fits {one["fits"]}')
spec, frame, res = d.choose_pages(None, est, classes, sched(256), 256)
print(f'big state: page {spec!r} frame {frame} fits {res["fits"]} bram {res["bram"]} '
      f'ports {res["page_ports"]} paged bits {res["paged_bits"]}')
spec, frame, res = d.choose_pages(None, est, classes, sched(256), 256, store='auto')
print(f'big state, storage auto: page {spec!r} frame {frame} bram {res["bram"]} uram {res["uram"]} '
      f'storage {res["storage"]!r}')
# A report with links (parse_classes keeps them under '__links__', not a
# class): the same choice, the links' LUTRAM counted.
links = d.parse_links('  link kinds: distributed 2 64 4; block 0 0 0; ultra 0 0 0; registers 1 8\n')
spec, frame, res = d.choose_pages(None, est, classes, dict(sched(256), __links__=links), 256)
print(f'big state with links: page {spec!r} frame {frame} fits {res["fits"]} link lutram {res["link_lutram"]}')
# Schedule reports name the page per class.
info = d.parse_classes('foldsim-fold-static: II = 8\n  Leaf x4 state gap 8 page 2\n    state: 0 2 4 6\n')
print('parsed', info['Leaf']['page'], d.state_form(4, info['Leaf'], 8, 2))
# State beyond the chip: 256 instances of 2,000,000 bits (512 Mbit) fit no
# frame on chip; --page auto moves the model's pages off chip, and the frame
# is its HBM traffic's (every instance read and written once a frame).
est['Lane'] = lane(2000000)
c = s.page_cost(2000000, 256, 2, 'hbm')
print(f'hbm: bram {c["bram"]} uram {c["uram"]} ports {c["ports"]} bytes {c["hbm_bytes"]} '
      f'frame {c["hbm_ii"]} controllers {c["controllers"]}')
spec, frame, res = d.choose_pages(None, est, classes, sched(256), 256, max_ii=8192)
print(f'beyond the chip: page {spec!r} off chip {res["offchip"]} frame {frame} '
      f'bram {res["bram"]} fits {res["fits"]}')
info = d.parse_classes('foldsim-fold-static: II = 8\n  Leaf x4 state gap 8 page 2 off chip\n    state: 0 2 4 6\n')
print('parsed off chip', info['Leaf']['offchip'])
# Small memories page as rows: a 31 x 64 register file read combinationally
# (two ports) and a 4 x 8 queue read through a register join the paged bits
# and leave distributed RAM; a 64 x 128 memory (8,192 bits) keeps its store.
mems = [('rf', 31, 64, 2, 1, 0, None, 1, True), ('q', 4, 8, 1, 1, 1, None, 1, True),
        ('big', 64, 128, 1, 1, 0, None, 1, True)]
e = dict(lut=1000.0, ff=500, mems=mems, regw={500: 1})
bits, mux, rest = d.paged_state(e)
print(f'rows: bits {bits} mux {mux:.0f} kept {[m[0] for m in rest["mems"]]}')
classes = dict(Lane=256, Top=1)
est = dict(Lane=e, Top=dict(lut=10000.0, ff=1000, mems=[], regw={}))
un = d.area(None, est, classes, {}, 0, sched(256), 256, 2, 'fixed')
pg = d.area(None, est, classes, {}, 0, sched(256, 2), 512, 2, 'fixed')
print(f'rows unpaged lutram {un["lutram"]} paged lutram {pg["lutram"]} paged bits {pg["paged_bits"]}')
