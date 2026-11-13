#!/usr/bin/env python3
"""foldsim_pin_copies.py <gen dir> --assignment <sll.json | CLASS:SLR,...> [-o <floorplan.tcl>] [--append]

The floorplan that pins each class copy of a generated simulator to an SLR, from a given assignment: the
`assignment` of a foldsim_sll_check.py --json (one list of classes a SLR, the SLRs named by its `caps`), or
CLASS:SLR pairs. Each copy is the simulator top's instance `<class>_copy` (of `<class>_static`); a class of the
assignment with no copy in <gen dir>/foldsim_sim.sv is refused (the assignment is another netlist's), and so is a
copy of the netlist the assignment leaves out, unless --partial (those then stay with the placer). Writes, for the
implementation flow's floorplan.tcl (applied before opt_design), one USER_SLR_ASSIGNMENT a copy under the marker line
`# foldsim pinned copies`, which foldsim_sll_check.py --floorplan reads back as the assignment to judge."""
import argparse, json, mmap, os, re, sys

MARK = '# foldsim pinned copies'


def copies(sim_sv):
    """{class: instance} of the simulator top's class copies (`<class>_static <class>_copy (`)."""
    with open(sim_sv, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        top = re.search(rb'^module foldsim_sim\(', mm, re.M)
        if not top:
            sys.exit(f'foldsim_pin_copies.py: {sim_sv} has no module foldsim_sim')
        end = mm.find(b'\nendmodule', top.start())
        body = mm[top.start():end].decode(errors='replace')
    return {m.group(1): m.group(1) + '_copy' for m in re.finditer(r'^\s+(\w+)_static (\w+)_copy \(', body, re.M)
            if m.group(1) == m.group(2)}


def assignment(spec):
    """{class: SLR name} from an sll.json (`assignment` + `caps`) or CLASS:SLR,... pairs."""
    if os.path.exists(spec):
        j = json.load(open(spec))
        names = [c['name'] for c in j['caps']]
        return {c: names[i] for i, cs in enumerate(j['assignment']) for c in cs}
    out = {}
    for x in spec.split(','):
        if ':' not in x:
            sys.exit(f'foldsim_pin_copies.py: {x!r} is not CLASS:SLR')
        c, s = x.split(':', 1)
        out[c] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('gen')
    ap.add_argument('--assignment', required=True)
    ap.add_argument('--partial', action='store_true', help='copies the assignment leaves out stay with the placer')
    ap.add_argument('-o', '--output', default='-')
    ap.add_argument('--append', action='store_true', help='append to the output (a generation\'s own floorplan)')
    a = ap.parse_args()
    have = copies(os.path.join(a.gen, 'foldsim_sim.sv'))
    want = assignment(a.assignment)
    bad = [s for s in set(want.values()) if not re.fullmatch(r'SLR\d+', s)]
    if bad:
        sys.exit(f'foldsim_pin_copies.py: not an SLR name: {", ".join(sorted(bad))}')
    extra = sorted(c for c in want if c not in have)
    if extra:
        sys.exit(f'foldsim_pin_copies.py: the assignment names classes with no copy in this netlist: {" ".join(extra)}')
    left = sorted(c for c in have if c not in want)
    if left and not a.partial:
        sys.exit(f'foldsim_pin_copies.py: the assignment leaves out copies of this netlist (--partial to leave them to '
                 f'the placer): {" ".join(left)}')
    lines = [f'{MARK}: {len(want)} of {len(have)} copies (foldsim_pin_copies.py, {a.assignment})']
    for c in sorted(want, key=lambda c: (want[c], c)):
        lines += [f'set c [get_cells -quiet -hierarchical {have[c]}]',
                  f'if {{[llength $c] == 1}} {{ set_property USER_SLR_ASSIGNMENT {want[c]} $c }}']
    text = '\n'.join(lines) + '\n'
    if a.output == '-':
        sys.stdout.write(text)
    else:
        with open(a.output, 'a' if a.append else 'w') as f:
            f.write(text)


if __name__ == '__main__':
    main()
