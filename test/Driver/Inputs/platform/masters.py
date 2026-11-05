# masters.py <the foldsim driver, beside the modules>: the platform's masters (foldsim_storage.platform_cost 'best')
# by the whole HBM share and by a point's own traffic (bytes_per_cycle, foldsim_degree state --masters-by-traffic):
# one memory-port master (port fold) and 13 or 84 controllers at 35, 72 and 480 HBM bytes a host cycle.
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.realpath(sys.argv[1])))
import foldsim_storage as st
for n, bpc in ((13, None), (13, 35.3), (13, 480.0), (84, None), (84, 72.2)):
    c = st.platform_cost(1, n, 'best', bpc)
    print(f"ctl {n} traffic {bpc}: vmem masters {c['vmem_masters']} masters {c['masters']} shares {c['shares']} "
          f"sc_nodes {c['sc_nodes']} luts {c['logic'] + c['lutram']:.0f}")
