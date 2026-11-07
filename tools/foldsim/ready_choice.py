#!/usr/bin/env python3
"""ready_choice.py <schedule.txt>: the --ready-choice of a fold-static report: the classes with a `(ready pass)` or
`(N ready passes)` line, comma separated, after `split:` when any class has more than one (ready outputs split by level)."""
import re, sys
names, split = [], False
for line in open(sys.argv[1]):
    m = re.match(r'^  (\S+) x\d+ \((?:(\d+) ready passes|ready pass)\)', line)
    if m:
        names.append(m.group(1)); split |= bool(m.group(2)) and int(m.group(2)) > 1
print(('split:' if split else '') + ','.join(names))
