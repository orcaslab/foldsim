# engines_depth.py <the foldsim driver, beside the modules>: off-chip pages on engines at their own depth
# (foldsim_storage page_hbm_depth): a copy as wide as the A100-class SM's compute slice (259,000 paged bits,
# 256 instances at 64 SMs) at its on-chip page 15 takes 34 engine columns, so the fold 34 engines; deepened to take
# PAGE_ENGINE_COLUMNS (4) it needs 4, within the page port share of the frame (p x instances <= frame / 2).
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.realpath(sys.argv[1])))
import foldsim_storage as st
bits, n = 259000, 256
for p, frame in ((15, 147456), (15, 16384), (15, 4096), (200, 147456)):
    ph = st.page_hbm_depth(bits, n, p, frame)
    cols, share = st.page_engine_share(bits, n, ph, frame)
    c0, _ = st.page_engine_share(bits, n, p, frame)
    e, e0 = st.page_hbm_cost(bits, n, ph, frame), st.page_hbm_cost(bits, n, p, frame)
    print(f"p {p} frame {frame}: page {ph} columns {cols} (at p {p}: {c0}) share {share:.3f} "
          f"select {e['lut'] - share * st.page_engine_ctl()[0]:.0f} (at p {p}: {e0['lut'] - e0['ctl'] * st.page_engine_ctl()[0]:.0f})")
