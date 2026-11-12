#!/usr/bin/env python3
"""port_plan_folds.py <port-plan json>: each folded store placed on a page engine (vmem-fold-on-engines) writes at
its class's issue positions (fold_issue) moved by one stagger d (its beats after): prints 'fold at issues: <member> d <d>' or
'fold NOT at issues: <member>'."""
import json, sys
J = json.load(open(sys.argv[1])); ii = J['ii']
issue = {c['name']: c['fold_issue'] or c['state'] for c in J['classes']}
for e in J['engines']:
    for m in e['members']:
        cls, _, vm = m['name'].partition(':')
        if not vm.startswith('fold') or cls not in issue:
            continue
        w = sorted(x % ii for x in m['wr'])
        nb = max(1, len(w) // max(1, len(issue[cls])))   # (vport-write-beats: a write's beats a position apart)
        ds = [d for d in range(0, 65) if sorted((f + d + j) % ii for f in issue[cls] for j in range(nb)) == w]
        print(f'fold at issues: {m["name"]} d {ds[0]}' if ds else f'fold NOT at issues: {m["name"]}')
