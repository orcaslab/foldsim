# engines_ff.py <the foldsim driver, beside the modules> <schedule report> [shift] | --regimes: the fold-degree model's page
# engines for a report's fold (foldsim_storage.engines_first_fit, foldsim_degree --engine-rule first-fit) against the
# engines the compiler assigned (the report's `page engines` line) and the upper bound of before ('max': every
# off-chip copy's columns and every stream on engines of their own). Streams: the report's built `link stream`
# pairs, each its share of the frame (link_stream_cost's ports with engines); copies: each class with pages off chip, its
# columns (the engine lines' <class>:region<k>_pages) at its share (max(p, 2) x instances of the frame).
import re
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.realpath(sys.argv[1])))
import foldsim_degree as fd
import foldsim_storage as st
if sys.argv[2] == '--regimes':
    # A large fold's (the 128-SM point: 18 copies of 1 to 4 columns at shares of 1e-3, 26 streams of 1e-2): the
    # streams' engines without the shift, a few with it; a dense one (16-SM g16c: a 9-column
    # copy at 0.5 beside a stream): a run of its own.
    c = [(1 + k % 4, 0.001) for k in range(18)]
    print('large', st.engines_first_fit(c, [0.01] * 26), st.engines_first_fit(c, [0.01] * 26, 256),
          'dense', st.engines_first_fit([(9, 0.5)], [0.01]))
    sys.exit(0)
text = open(sys.argv[2]).read()
shift = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ii = fd.parse_schedule(text)[0]
links = fd.parse_links(text)
streams = []
for s in links['streams']:
    pair = links['pairs'][(s['p'], s['r'])]
    uses = [u for u in links['uses'] if u['p'] == s['p'] and u['r'] == s['r']]
    ports = max(pair['producer_passes'] * pair['pages'], pair['reader_passes'] * (pair['max_need'] + 2))
    streams.append(min(1.0, ports / ii))
cols = {}
for line in re.findall(r'^    page engine \d+ .*$', text, re.M):
    for cls in re.findall(r' (\w+):region\d+_pages', line):
        cols[cls] = cols.get(cls, 0) + 1
copies = []
for m in re.finditer(r'^  (\w+) x(\d+) .*?page (\d+) off chip', text, re.M):
    if m.group(1) in cols:
        t, p = int(m.group(2)), int(m.group(3))
        copies.append((cols[m.group(1)], min(1.0, max(p, 2) * t / ii)))
built = re.search(r'^  page engines: (\d+) engines', text, re.M)
ff = st.engines_first_fit(copies, streams, shift)
mx = sum(c for c, _ in copies) + len(streams)
print(f"frame {ii}: streams {' '.join(f'{u:.3f}' for u in streams)}; copies "
      f"{' '.join(f'{c}x{u:.3f}' for c, u in copies)}")
print(f"engines: first-fit {ff}, max {mx}, built {built.group(1) if built else 0}")
