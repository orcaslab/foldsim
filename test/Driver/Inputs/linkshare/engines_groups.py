#!/usr/bin/env python3
"""engines_groups.py <schedule.txt> <class>:<group>,...: the page engines of a
fold and how many carry members of more than one group (a link stream
counts as its reader's): `engines N, mixed M`, then each engine's groups."""
import re
import sys

group = {c: int(g) for c, g in (x.rsplit(':', 1) for x in sys.argv[2].split(','))}
rows = re.findall(r'^\s+page engine (\d+) \d+ bits, tag \d+, row \d+: (.*)$', open(sys.argv[1]).read(), re.M)
mixed, lines = 0, []
for e, members in rows:
    gs = set()
    for x in members.split():
        if x.startswith('link:') and '>' in x:
            gs.add(group.get(x[5:].split('>', 1)[1].split('(')[0], -1))
        elif ':' in x:
            gs.add(group.get(x.split(':', 1)[0], -1))
    mixed += len(gs) > 1
    lines.append(f'engine {e}: groups {",".join(str(g) for g in sorted(gs))}')
print(f'engines {len(rows)}, mixed {mixed}')
print('\n'.join(lines))
