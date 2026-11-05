import os, sys
# top_terms.py <the foldsim driver, beside the modules>
sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_degree as fd
REP = '''  link storage: 3 memories, 100 bits, 10 LUTRAM LUTs; 200 register bits; shared outputs 0; staged: 2 links, 1000 register bits; compact: 0 links, 0 bits saved
  link kinds: distributed 3 100 10; block 0 0 0; ultra 0 0 0; registers 4 1200
    link use a b 32 2 2 distributed 0 1 1 0 1 0 3 1 1 0
    link use a c 8 2 2 distributed 0 1 1 0 1 0 5 1 1 0
    link use b c 16 2 2 distributed 0 1 1 0 1 0 1 1 1 0
'''
lk = fd.parse_links(REP)
print(f"parsed: ff {lk['ff']} staged {lk['staged_ff']} select {lk['select_bits']}")
fd.KERNEL['calibration'] = dict(top_logic_per_link_bit=0.315)
print(f"ratio: {fd.top_link_logic(lk):.1f}")
fd.KERNEL['calibration'] = dict(top_logic_per_link_bit=0.315, top_logic_terms=dict(staged=0.01, reg=0.2, select=0.0833334))
print(f"terms: {fd.top_link_logic(lk):.1f}")
lk = fd.parse_links(REP.replace('; staged: 2 links, 1000 register bits', '').replace('distributed 0 1 1 0 1 0 3', 'distributed 0 1 1 0 1 0 1').replace('distributed 0 1 1 0 1 0 5', 'distributed 0 1 1 0 1 0 1'))
print(f"no staged line: ff {lk['ff']} staged {lk['staged_ff']} select {lk['select_bits']}")
