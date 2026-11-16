#!/usr/bin/env python3
"""foldsim_s28_axi_check.py <emulator AXI trace> <foldsim_kernel.vh> (a32-S28C; check_axi.py of B10 for S28 regions): every AR / AW
request of the S28 region controllers' masters checked against the port it is bound to. The trace
(FOLDSIM_EMUL_AXI_TRACE): '<cycle> [<k>:]AR|AW <addr hex>', k the kernel's device-memory master. The S28 masters:
FOLDSIM_S28_MASTER_REPEAT `f(port, r, m)`. A region with PC_LOG > 0 (FOLDSIM_S28_R<r>_PC_LOG / _PC_FIRST, 4 masters on
MEM_SLR<s>_0..3; an SLR2 region's N masters on MEM): each of master m's addresses must lie in its PCs,
PC_FIRST + m + PC_STRIDE i (i < 2^PC_LOG), in the upper half (512 MB in) of that PC's 1 GB window at 0x40_0000_0000 + p GB,
below 512 MB + FOLDSIM_VMEM_PCBYTES. A linear region (PC_LOG 0, on MEM): its addresses are the runtime's virtual-memory
base + lin, so each must lie in [VBASE + LIN_BASE, VBASE + LIN_BASE + BYTES) (FOLDSIM_S28_R<r>_LIN_BASE / _BYTES of the
gen's foldsim_sim.vh beside the kernel header) and outside every spread PC half. VBASE (--vmem-base, default
0x41_0000_0000) is where the runtime places the space on the V80 and in avedsim alike: kMemBase + the target's 4 GiB, its
high word written after the move (VMEM_BASE_LO, then VMEM_BASE_HI): a controller that takes the base before the high word
lands its requests below the HBM window, and fails here.
Prints per master the requests and the PCs seen; exit 1 on any violation, 2 when an S28 master issued nothing."""
import collections
import re
import sys

import os
args = [a for a in sys.argv[1:] if not a.startswith('--vmem-base')]
trace, kvh = args[:2]
VBASE = next((int(a.split('=', 1)[1], 0) for a in sys.argv[1:] if a.startswith('--vmem-base=')), 0x41_0000_0000)
vh = open(kvh).read()
svh = os.path.join(os.path.dirname(kvh), 'foldsim_sim.vh')
if os.path.exists(svh):
    vh += open(svh).read()
def get64(r, k):
    m = re.search(rf"^`define FOLDSIM_S28_R{r}_{k} 64'h([0-9a-fA-F]+)", vh, re.M)
    return int(m.group(1), 16) if m else None
get = lambda r, k: int(re.search(rf'^`define FOLDSIM_S28_R{r}_{k} (\d+)', vh, re.M).group(1))
rep = re.search(r'^`define FOLDSIM_S28_MASTER_REPEAT\(f,s\) (.*)$', vh, re.M).group(1)
masters = {int(p): (int(r), int(m)) for p, r, m in re.findall(r'`f\((\d+), (\d+), (\d+)\)', rep)}
pcb = re.search(r'^`define FOLDSIM_VMEM_PCBYTES (\d+)', vh, re.M)
pcb = int(pcb.group(1)) if pcb else 0
mask = re.search(r"^`define FOLDSIM_VMEM_PCMASK 32'h([0-9a-fA-F]+)", vh, re.M)
mask = int(mask.group(1), 16) if mask else 0
BASE = 0x40_0000_0000
reqs, pcs, bad = collections.Counter(), collections.defaultdict(collections.Counter), []
for ln in open(trace):
    mm = re.match(r'(\d+) (?:(\d+):)?(AR|AW) ([0-9a-f]+)$', ln.strip())
    if not mm:
        continue
    k = int(mm.group(2) or 0)
    a = int(mm.group(4), 16)
    reqs[(k, mm.group(3))] += 1
    pc, off = (a - BASE) >> 30, (a - BASE) & ((1 << 30) - 1)
    pcs[k][pc] += 1
    if k not in masters:
        continue
    r, m = masters[k]
    if get(r, 'PC_LOG'):
        st = re.search(rf'^`define FOLDSIM_S28_R{r}_PC_STRIDE (\d+)', vh, re.M)
        ok = {get(r, 'PC_FIRST') + m + (int(st.group(1)) if st else 4) * i for i in range(1 << get(r, 'PC_LOG'))}
        if a < BASE or pc not in ok or not (512 << 20) <= off < (512 << 20) + pcb:
            bad.append(f'master {k} (region {r} master {m}, PCs {sorted(ok)}) {mm.group(3)} 0x{a:x}: PC {pc} offset 0x{off:x}')
    else:
        lo, n = get64(r, 'LIN_BASE'), get64(r, 'BYTES')
        if lo is not None and n is not None and not VBASE + lo <= a < VBASE + lo + n:
            bad.append(f'master {k} (region {r}, linear) {mm.group(3)} 0x{a:x}: outside the region\'s space '
                       f'0x{VBASE + lo:x}..0x{VBASE + lo + n:x} (runtime base 0x{VBASE:x} + lin)')
        elif a >= BASE and mask >> pc & 1 and off >= (512 << 20):
            bad.append(f'master {k} (region {r}, linear) {mm.group(3)} 0x{a:x}: in spread PC {pc}\'s reserved half')
for k in sorted({k for k, _ in reqs} | set(masters)):
    who = f'region {masters[k][0]} master {masters[k][1]}' if k in masters else 'not S28'
    print(f'master {k} ({who}): AR {reqs[(k, "AR")]} AW {reqs[(k, "AW")]}; PCs ' +
          ' '.join(f'{p}:{n}' for p, n in sorted(pcs[k].items())))
idle = [k for k in masters if not reqs[(k, 'AR')] + reqs[(k, 'AW')]]
for b in bad[:20]:
    print('VIOLATION', b)
print(f'{sum(reqs.values())} requests, {len(bad)} violations, {len(idle)} S28 masters without traffic'
      + (f' ({idle})' if idle else ''))
sys.exit(1 if bad else 2 if idle else 0)
