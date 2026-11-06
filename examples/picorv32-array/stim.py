#!/usr/bin/env python3
"""stim.py <work.bin> <ports> <T> <out.hex> [tiles]: the gate's stimulus for pico_array: each tile's RAM loaded
(program at 0, 64 words of tile-specific data at 0x800) with the cores in reset, then the cores released for
the rest of T cycles. <ports>: `foldsim_gate.py ports` (name msb lsb per input)."""
import struct, sys
binf, portsf, T, outf = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tiles = int(sys.argv[5]) if len(sys.argv) > 5 else 4
code = open(binf, 'rb').read()
code += b'\0' * (-len(code) % 4)
prog = list(struct.unpack('<%dI' % (len(code) // 4), code))
lay = {l.split()[0]: (int(l.split()[1]), int(l.split()[2])) for l in open(portsf) if l.strip()}
def pack(**v):
    w = 0
    for n, (hi, lo) in lay.items():
        x = v.get(n, 0)
        assert x < (1 << (hi - lo + 1)), (n, x)
        w |= x << lo
    return w
rows = []
for t in range(tiles):
    x = 0x12345 + 7919 * t
    data = []
    for i in range(64):
        x = (x * 1103515245 + 12345) & 0xffffffff
        data.append(x)
    for a, w in list(enumerate(prog)) + [(0x200 + i, d) for i, d in enumerate(data)]:
        rows.append(pack(resetn=0, load_en=1, load_tile=t, load_addr=a, load_data=w))
rows += [pack(resetn=0)] * 4
assert len(rows) < T
rows += [pack(resetn=1)] * (T - len(rows))
width = max(hi for hi, _ in lay.values()) + 1
open(outf, 'w').write(''.join('%0*x\n' % ((width + 3) // 4, r) for r in rows))
print(f'stim: {len(prog)} program words, {tiles} tiles, load {tiles * (len(prog) + 64)} cycles, T {T}')
