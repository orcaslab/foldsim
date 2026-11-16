#!/usr/bin/env python3
"""tb_s28_beats.py <foldsim_sim.json> <beats.hex> [<lanes.hex>]: tb_s28.sv's +beats table, entry {g 4, cls 6, mem 4} =
the beats of region g's class cls's memory mem (ceil(row bits / 512); 1 elsewhere), and its +lanes table (log2 of a
page column's lanes; 0 elsewhere), for $readmemh."""
import json
import sys
j = json.load(open(sys.argv[1]))
t, ln = [1] * (1 << 14), [0] * (1 << 14)
for r in j['s28']['regions']:
    for e in r['entries']:
        i = (r['group'] << 10) | (e['cls'] << 4) | e['mem']
        t[i] = -(-e['dw'] // 512)
        ln[i] = e.get('lanes', 1).bit_length() - 1
open(sys.argv[2], 'w').write('\n'.join(f'{x:x}' for x in t) + '\n')
if len(sys.argv) > 3:
    open(sys.argv[3], 'w').write('\n'.join(f'{x:x}' for x in ln) + '\n')
