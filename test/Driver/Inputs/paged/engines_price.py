# engines_price.py <the foldsim driver, beside the modules>: the fold-degree model's HBM page streams
# (foldsim_storage page_hbm_cost with an engines frame, page_engine_share, page_engine_ctl) against a
# controller per column, for the vt target's node copy (3,976 paged bits, 16 instances).
import math
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.realpath(sys.argv[1])))
import foldsim_storage as st
for p, frame in ((4, 64), (4, 256), (8, 128)):
    col = st.page_hbm_cost(3976, 16, p)
    eng = st.page_hbm_cost(3976, 16, p, frame)
    print(f"p {p} frame {frame}: columns {eng['engine_columns']} share {eng['ctl']:.3f} "
          f"engines lut {eng['lut']:.0f} lutram {eng['lutram']:.0f}; "
          f"per column {col['ctl']} lut {col['lut']:.0f} lutram {col['lutram']:.0f}")
el, er = st.page_engine_ctl()
shares = [st.page_engine_share(3976, 16, 4, 256), st.page_engine_share(4752, 4, 2, 256)]
n = max(max(c for c, _ in shares), math.ceil(sum(s for _, s in shares) - 1e-9))
print(f"fold at 256: {n} engines, engine lut {el:.0f} lutram {er:.0f}")
# A link stream on the engines: its controller a share of one (the positions its requests take), against its own.
import foldsim_degree as fd
pair = dict(producer_passes=16, pages=1, reader_passes=16, max_need=1, dw=64, word=64, read_rows=16, write_rows=16)
uses = [dict(banked=True, entries=16, lead=200, width=64, depth=32)]
own, eng = fd.link_stream_cost(pair, uses, 128), fd.link_stream_cost(pair, uses, 128, engines=True)
print(f"stream at 128: own ctl {own['ctl']} lut {own['lut']:.0f}; engines ctl {eng['ctl']:.3f} lut {eng['lut']:.0f}")
# Built (a schedule report with page engines): the engines counted, a stream an engine carries no controller of
# its own, one that is not its own.
rep = """  link kinds: distributed 1 64 36; block 0 0 0; ultra 0 0 0; registers 0 0
    link stream A B 4 0 64 64 1 16 16 1 48
    link stream C D 4 0 64 64 1 16 16 1 48
  page engines: 3 engines, 2 page columns
    page engine 0 512 bits, tag 5, row 9: X:region0_pages link:A>B
"""
lk = fd.parse_links(rep)
b = fd.link_streams_built(lk)
print(f"built: engines {lk['engines']} carried {sorted(lk['engine_streams'])} controllers {b['ctl']} "
      f"lut {b['lut']:.0f}")
