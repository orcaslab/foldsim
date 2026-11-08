#!/usr/bin/env python3
"""foldsim_sv_limits.py <file.sv>... [--max-bits N] [--json out]: hard synthesis-tool limits checked on generated
SystemVerilog before any tool runs (generic: any netlist foldsim or firtool emitted).

Checked, an ERROR each (exit 1):
  Synth 8-4556  a declared variable or constant larger than --max-bits bits (Vivado: 1,000,000): wire / logic / reg /
                bit / localparam / parameter, its packed dimensions times its unpacked ones when it is a value
                (an initializer, or no unpacked dimension); a memory (unpacked array without an initializer: a RAM
                the tool infers) is exempt. The 11-08 R128w4 image failed synthesis on `wire [131071:0][10:0]`.
  Synth 8-4556  the same for an `automatic` variable in a procedure.
Reported, not refused: the largest declaration, and the count of declarations past half the limit.
One summary line per file on stdout; exit 0 when no ERROR. Streaming (a 2.6 GB file reads in about a minute)."""
import argparse
import json
import re
import sys

DECL = re.compile(r'^\s*(?:automatic\s+)?(?:wire|logic|reg|bit|localparam|parameter)\b(?:\s+(?:signed|unsigned|logic|bit))*'
                  r'\s*((?:\[[^\]]*\]\s*)*)\s*([A-Za-z_$][\w$]*)\s*((?:\[[^\]]*\]\s*)*)(=?)')
DIM = re.compile(r'\[\s*([0-9]+)\s*:\s*([0-9]+)\s*\]|\[\s*([0-9]+)\s*\]')


def dims(text):
    n = 1
    for a, b, c in DIM.findall(text):
        n *= (abs(int(a) - int(b)) + 1) if a else int(c)
    return n


def scan(path, max_bits):
    errors, biggest, near = [], (0, '', 0), 0
    with open(path, errors='replace') as f:
        for lineno, line in enumerate(f, 1):
            s = line.lstrip()
            if not s or s[0] not in 'wlrbpa':
                continue
            m = DECL.match(line)
            if not m:
                continue
            packed, name, unpacked, init = m.groups()
            if ('[' in packed and not DIM.search(packed)) or ('[' in unpacked and not DIM.search(unpacked)):
                continue  # parameterized dimensions: not a fixed size
            pb = dims(packed)
            value = bool(init) or not unpacked.strip()
            bits = pb * (dims(unpacked) if value else 1)
            if bits > biggest[0]:
                biggest = (bits, name, lineno)
            if bits > max_bits // 2:
                near += 1
            if bits > max_bits:
                errors.append(dict(file=path, line=lineno, name=name, bits=bits,
                                   rule='Synth 8-4556', text=line.strip()[:120]))
    return errors, biggest, near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sv', nargs='+')
    ap.add_argument('--max-bits', type=int, default=1000000)
    ap.add_argument('--json')
    a = ap.parse_args()
    allerr = []
    for p in a.sv:
        errs, big, near = scan(p, a.max_bits)
        allerr += errs
        verdict = 'ERROR' if errs else 'OK'
        print(f'sv_limits: {p}: {verdict}: {len(errs)} declaration(s) over {a.max_bits:,} bits (Synth 8-4556); '
              f'largest {big[1]} {big[0]:,} bits (line {big[2]}); {near} over half the limit')
        for e in errs[:10]:
            print(f'  ERROR {e["rule"]}: {e["name"]} {e["bits"]:,} bits at line {e["line"]}: {e["text"]}')
    if a.json:
        json.dump(allerr, open(a.json, 'w'), indent=1)
    return 1 if allerr else 0


if __name__ == '__main__':
    sys.exit(main())
