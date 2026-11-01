# price.py <the foldsim driver, beside the modules> <schedule report> [frame]: the fold-degree model's view of a report's HBM link streams (parse_links'
# `link pair` and `link stream` lines; link_stream_cost per pair at the report's frame; link_streams_built).
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.realpath(sys.argv[1])))
import foldsim_degree as fd
text = open(sys.argv[2]).read()
ii = int(sys.argv[3]) if len(sys.argv) > 3 else fd.parse_schedule(text)[0]
links = fd.parse_links(text)
for (p, r), pair in sorted(links['pairs'].items()):
    uses = [u for u in links['uses'] if u['p'] == p and u['r'] == r]
    c = fd.link_stream_cost(pair, uses, ii)
    print(f"pair {p} {r}: word {pair['word']} rows {pair['read_rows']}+{pair['write_rows']} "
          + (f"stream lut {c['lut']:.0f} lutram {c['lutram']:.0f} hbm {c['hbm']:.0f}" if c else 'no stream'))
b = fd.link_streams_built(links)
print(f"built: {b['ctl']} streams, lut {b['lut']:.0f}, hbm {b['hbm']:.0f}")
