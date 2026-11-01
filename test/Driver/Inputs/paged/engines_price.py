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
