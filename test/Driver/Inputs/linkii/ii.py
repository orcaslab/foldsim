# ii.py <the foldsim driver, beside the modules> <planner's schedule report> <link kinds> [built report]: the fold-degree
# model's link_ii (link_spacing x model_passes) for the link kinds on the report the planner priced, and the II the
# compiler built with those kinds.
import re, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.realpath(sys.argv[1])))
import foldsim_degree as fd
links = fd.parse_links(open(sys.argv[2]).read())
sp = fd.link_spacing(links, sys.argv[3])
print('spacing', ' '.join(f'{m}:{s}' for m, s in sorted(sp.items())) or 'none',
      'passes', ' '.join(f'{m}:{n}' for m, n in sorted(fd.model_passes(links).items())))
print('link_ii', fd.link_ii(links, sys.argv[3]))
if len(sys.argv) > 4:
    print('built II', re.search(r'II = (\d+)', open(sys.argv[4]).read()).group(1))
# The paged kinds the planner offers a pair at a frame (link_units): a page spaces every class of both models, so the
# models' most passes (A's other class: 16) x the page must fit the link port share, not only the pair's uses' (4).
use = lambda p, r, pp, rp: dict(p=p, r=r, width=32, depth=8, entries=4, kind='block', sync=True, write_gap=1, read_gap=1,
                                lead=40, reads=4, banked=True, sources=1, producer_passes=pp, reader_passes=rp, page=0)
syn = dict(uses=[use('A', 'B', 4, 4), use('A', 'C', 16, 4)], pairs={}, ff=0)
for f in (32, 64, 128):
    units, _ = fd.link_units(None, syn, f)
    print(f'offer {f}:', '; '.join(f"{u['store']} " + (' '.join(sorted(k for k in u['costs'] if ':' in k)) or '-')
                                  for u in units))
