#!/usr/bin/env python3
"""tb_s28_beats.py <foldsim_sim.json> <out.hex>: tb_s28.sv's +beats table, entry {g 4, cls 6, mem 4} = the beats of
region g's class cls's memory mem (ceil(DW / 512); 1 elsewhere), for $readmemh."""
import json
import sys
j = json.load(open(sys.argv[1]))
t = [1] * (1 << 14)
for r in j['s28']['regions']:
    for e in r['entries']:
        t[(r['group'] << 10) | (e['cls'] << 4) | e['mem']] = -(-e['dw'] // 512)
open(sys.argv[2], 'w').write('\n'.join(f'{x:x}' for x in t) + '\n')
