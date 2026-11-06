#!/usr/bin/env python3
"""mkload.py <work.bin> <load.hex> [tiles]: what the host writes to pico_v80 before go, one CSR write a line,
hex '<addr:3><data:8>' (addr = tile << 10 | word): each tile's program at word 0 and 64 words of tile-specific
data at word 0x200 (byte 0x800), as stim.py loads pico_array for the Verilator gate."""
import struct, sys
binf, out = sys.argv[1], sys.argv[2]
tiles = int(sys.argv[3]) if len(sys.argv) > 3 else 4
code = open(binf, 'rb').read()
code += b'\0' * (-len(code) % 4)
prog = list(struct.unpack('<%dI' % (len(code) // 4), code))
lines = []
for t in range(tiles):
    x = 0x12345 + 7919 * t
    data = []
    for i in range(64):
        x = (x * 1103515245 + 12345) & 0xffffffff
        data.append(x)
    for a, w in list(enumerate(prog)) + [(0x200 + i, d) for i, d in enumerate(data)]:
        lines.append('%03x%08x' % ((t << 10) | a, w))
open(out, 'w').write('\n'.join(lines) + '\n')
print(f'mkload: {len(prog)} program words + 64 data words x {tiles} tiles = {len(lines)} writes')
