#!/usr/bin/env python3
"""Storage of folded state on the V80, and the routability of a fold.

Storage. A fold makes, per copy of a class, a store of each of its
registers (one entry per instance, read combinationally into a read
register) and of each of its memories (the target's depth times the
instances). Each store can live in the kinds of the V80:

  kind (ram_style)  capacity per primitive   ports              read
  registers         one flop a bit           any                any
  distributed       a LUT: 64 x 1 (SLICEM)   1 write, a LUT per  any
                                             read port
  block             RAMB36: 512 x 72 ...     2 (a write and a    registered
                    32K x 1, or half (18K)   read, SDP)
  ultra             URAM288: 4K x 72         2 (as block)       registered

A store takes a kind only if the kind meets its ports and read latency: a
register store reads combinationally, so it takes block or ultra RAM only
read a host cycle early (foldsim-fold-static `storage`), which needs a state
gap of 4; a memory read combinationally takes neither. Read ports beyond
the kind's are copies of the RAM (each a write and a read), as Vivado builds
them; write ports beyond the kind's would need slots of several host cycles,
whose II floor is reported (`ii_floor`).

Paged register state (foldsim-fold-static `page`, `page_cost`). A copy of
`threads` instances whose registers page keeps them in one page RAM of
pow2(threads) x pow2(p) rows of ceil(bits / p) bits (block or ultra RAM), a
fetch buffer and a write-back buffer of `bits` flops each and the RAM's
output register; the write-back's page select is ceil((p - 1) / 3) LUTs a
page bit (page_select_luts, calibrated). Its state passes are p host cycles apart, so the frame is at least
p x threads (and p + 4). A RAM of P 72-bit ports moves 72 P bits a host
cycle, so a copy's II is at least threads x ceil(bits / (72 P)) (`page_bound`):
the bits/ports bound, which p x threads meets up to rounding.

`assign` chooses a kind per store against the V80 dynamic region: first the
cheapest kind of each store, pricing every resource by its share of the
region (a URAM is 1/1211 of the URAMs, a LUT site 1/3,269,043 of the
sites), then, while a resource is over its capacity, the moves that relieve
it at the least price, never pushing another over its own. A point that
still exceeds a capacity does not place. Once placed, the same repair
brings the routing load (below) under its limit if it can, never at
placement's cost; a point it cannot bring under does not route.

Routability. The router's global routing utilization (Vivado's initial
estimate, `Global Vertical/Horizontal Routing Utilization`) against the
design's routing load and how densely the layout packs it, fitted over the
V80 OOC builds and images (`fit-route`, foldsim_v80_route.csv: the OOC
builds of fame5_work/static/ooc2x2 by `route-points`, and the images
img_int, img_fo and img_ii8 from their impl_1 runme.log and
place_report_utilization_0.rpt):

  congestion = A * L + B * L * (L / S)   (percent, L in millions)
  L = LUTs + 24 RAMB36 + 26 URAM288
  S = the SLRs the fold may spread over: the region's (3 out of context,
      1.8 for the image's dynamic region), or its SLR groups if fewer

A block RAM's routing in LUTs is its signals over a LUT6's 7 pins: a RAMB36
at 72 bits has about 165 (72 in, 72 out, two addresses, enables), a URAM288
about 180. The builds cannot fit these weights (their block RAMs hardly
vary), so they are set, not fitted; without them, state moved out of LUTs
into block RAM would look free of routing.

The layout term. At 1.42-1.48M of load the full GPU's II 8 folds split by
layout: out of context, o8_t8n and o8_t8a spread over all three SLRs and
route (30.5-30.7 %); o8_t8sc and o8_t8sf, whose `--slr-groups` kept each
socket's subtree whole in one SLR (2 groups), packed into two SLRs and
failed (35.8-37.6 %), as did img_ii8 in the image, whose dynamic region
spans 1.8 SLRs (36.3 %). Their links do not tell them apart: t8n, t8sc and
t8sf have the same 16 copy pairs and 75.2k link bits, t8a and img_ii8 the
same 28 pairs (interleaved copies) and 75.3k bits. What differs is the
SLRs the placement may use, so the second term is the load times the load
per SLR: wires lengthen as a region fills. The region's SLRs are the slices
of its pblock (the rectangles of the compute shell's pblock_slash: about
195k of the device's 321,776 slices; any value from 1.5 to 2.2 classifies
every build alike); a fold with SLR groups (foldsim --slr-groups)
keeps each whole in one SLR, so G groups occupy at most G SLRs. Fewer
groups than SLRs confine; one per copy, as many as the SLRs or more, do not
(t8a's 7 spread over three).

The fit of the builds that routed reaches 28.8 % (o8_t8n); that of those
that failed starts at 34.5 % (o8_t8sf); a fold routes when its estimate is
below the middle, 31.65 %: a load of 1.32M in the image's region, 1.58M
over three SLRs. Leaving any build out, the fit of the rest still classifies
it. The builds count synthesized LUTs (placed for the images); the
fold-degree model estimates routed ones (its calibration's), which are
3.8 % fewer (the median over the 24 OOC builds that routed, 0.7-9.0 %), so a
fold's LUTs are scaled by 1.038 first: in the region it routes up to a load
of 1.27M in the model's LUTs.

usage:
  foldsim_storage.py fit-route [--points foldsim_v80_route.csv]
  foldsim_storage.py route-points <build dir>...   (rows for the points file)
"""
import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import foldsim_cost as cost  # noqa: E402

# The SLASH dynamic region of the V80 (pblock_slash): LUT sites as the placer
# counts them (foldsim_degree.PLACE_SITES), RAMB36, URAM288, and the SLRs its
# slices amount to (about 195k of the device's 321,776 slices over 3 SLRs).
REGION = dict(sites=3269043, bram=2415, uram=1211, slrs=1.8)
# The V80's SLRs: what an out-of-context build may spread over.
DEVICE_SLRS = 3
# What a LUT of distributed RAM costs in sites (foldsim_degree.LUTRAM_SITES).
LUTRAM_SITES = 4.08
# Placement is not routing: the margin kept on every capacity.
MARGIN = 0.9

KINDS = ('registers', 'distributed', 'block', 'ultra')
# RAMB36E5 in simple dual port (a write and a read port): depth x width, and
# its half, RAMB18E5 (0.5 of a RAMB36).
BRAM36 = ((512, 72), (1024, 36), (2048, 18), (4096, 9), (8192, 4), (16384, 2), (32768, 1))
BRAM18 = ((512, 36), (1024, 18), (2048, 9), (4096, 4), (8192, 2), (16384, 1))
URAM = (4096, 72)
# Write ports a primitive has (a store with more needs paging).
WRITE_PORTS = dict(registers=None, distributed=1, block=2, ultra=2)
# Host cycles a register store's block read needs between a state pass's
# write and the next read (foldsim-fold-static `storage`).
BLOCK_STATE_GAP = 4

# The routability fit (fit-route over foldsim_v80_route.csv: 29 OOC builds
# and 3 images; RMS 2.27 points, leave-one-out 2.45) and its limit: midway
# between the fit of the most congested build that routed (28.80 %, o8_t8n)
# and of the least congested that failed (34.51 %, o8_t8sf); 1.2
# leave-one-out errors either side.
ROUTE_A, ROUTE_B = 9.924, 19.211
ROUTE_LIMIT = 31.65
# LUT6s of routing a block RAM brings: its signals over a LUT6's 7 pins.
ROUTE_PER = dict(bram=24, uram=26)
# Synthesized LUTs per routed LUT: the median over the OOC builds that routed
# (synth_util.rpt against routed_util.rpt; 1.007-1.090).
SYNTH_PER_ROUTED = 1.038


def route_load(luts, bram=0.0, uram=0.0):
    """A design's routing load L: its synthesized LUTs and its block RAMs'
    LUT6 equivalents."""
    return luts + ROUTE_PER['bram'] * bram + ROUTE_PER['uram'] * uram


def route_slrs(groups=0, region=None):
    """The SLRs a fold may spread over: the region's (REGION's by default),
    or fewer when it has fewer SLR groups, each kept whole in one SLR
    (`groups`: their number, 0 for none)."""
    s = REGION['slrs'] if region is None else region
    return min(s, groups) if groups else s


def congestion(load, slrs=None):
    """The router's global routing utilization (percent) at a routing load
    spread over `slrs` SLRs (route_slrs; the region's by default)."""
    m = load / 1e6
    s = route_slrs() if slrs is None else slrs
    return ROUTE_A * m + ROUTE_B * m * m / s


def fold_load(luts, bram=0.0, uram=0.0):
    """The routing load of a fold priced by the fold-degree model, whose
    LUTs are routed ones; in the model's LUTs."""
    return luts + route_load(0, bram, uram) / SYNTH_PER_ROUTED


def fold_congestion(luts, bram=0.0, uram=0.0, slrs=None):
    """The congestion of a fold priced by the fold-degree model."""
    return congestion(SYNTH_PER_ROUTED * fold_load(luts, bram, uram), slrs)


def route_luts(slrs=None):
    """The most routing load, in the model's LUTs, that routes over `slrs`
    SLRs (the region's by default): fold_congestion(route_luts(s), slrs=s)
    == ROUTE_LIMIT."""
    s = route_slrs() if slrs is None else slrs
    a, b, c = ROUTE_B / s, ROUTE_A, -ROUTE_LIMIT
    return 1e6 * (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a) / SYNTH_PER_ROUTED


def bram36(depth, width):
    """RAMB36 (18K halves at 0.5) holding depth x width in one port pair."""
    best = min(math.ceil(width / w) * math.ceil(depth / d) for d, w in BRAM36)
    half = min(math.ceil(width / w) * math.ceil(depth / d) for d, w in BRAM18)
    return min(best, 0.5 * half)


# The host caches of the virtual memories (foldsim --vmem-cache-rows auto): a
# sixteenth of a memory's rows, at least 256 lines, and all of them within
# this many RAMB36 (5 % of the V80 dynamic region's 2415).
VMEM_CACHE_FRACTION = 16
VMEM_CACHE_FLOOR = 256
VMEM_CACHE_BUDGET = 120


def pow2_floor(n):
    return 1 << (n.bit_length() - 1) if n > 0 else 0


def vmem_cache_rows(vmems, choice, budget):
    """Each virtual memory's host-cache lines (VMEM_INTERFACE.md, v4): a power
    of two (or 0) up to its rows, the largest power of two at most
    rows x instances. `auto`: rows / 16, at least 256; while the caches' data
    take more than `budget` RAMB36, the costliest halves (0 below 2 lines).
    `vmems`: dicts with instances, row_width (address bits) and data_width.
    A cache's lines live in block RAM in one port pair (bram36). An HBM link
    stream's (class `__link<k>`: every row written once and read once a
    frame, in slot order) has none, nor an HBM page stream's engine
    (`__pages<k>`: each page row written and read once a frame)."""
    caps = [0 if str(v.get('class', '')).startswith(('__link', '__pages')) else
            pow2_floor(v['instances'] << v['row_width']) for v in vmems]
    if choice != 'auto':
        n = int(choice)
        if n < 0 or n == 1 or n & (n - 1):
            sys.exit('--vmem-cache-rows: auto, 0 or a power of two of at least 2')
        return [min(n, c) for c in caps]
    rows = [min(c, max(VMEM_CACHE_FLOOR, pow2_floor(c // VMEM_CACHE_FRACTION))) if c else 0
            for c in caps]
    cost = lambda k: bram36(rows[k], vmems[k]['data_width']) if rows[k] else 0
    while sum(cost(k) for k in range(len(rows))) > budget:
        k = max(range(len(rows)), key=cost)
        rows[k] = rows[k] // 2 if rows[k] > 2 else 0
    return rows


def vmem_cache_bram(vmems, choice, budget):
    """RAMB36 of the virtual memories' host caches (0 without a choice: the
    kernel's default, no cache)."""
    if choice is None or not vmems:
        return 0.0
    return sum(bram36(n, v['data_width']) if n else 0.0
               for v, n in zip(vmems, vmem_cache_rows(vmems, choice, budget)))


def uram(depth, width):
    return math.ceil(width / URAM[1]) * math.ceil(depth / URAM[0])


def lanes(width, mask):
    """A masked store's write lanes: one RAM column per lane unless the lanes
    are bytes (the primitives' byte write enables)."""
    if mask <= 1:
        return 1, width
    grain = width // mask
    return (1, width) if grain in (8, 9) else (mask, grain)


def kind_cost(store, kind):
    """What a store costs in a kind: logic LUTs, distributed-RAM LUTs, flops,
    RAMB36, URAM; and the II floor its write ports imply (0: none). None if
    the kind cannot hold it. `store`: depth, width, rports, wports, rlat (0 or
    1), mask, reg (a register store), gap (the copy's state gap)."""
    d, w = store['depth'], store['width']
    r, wr = max(store['rports'], 1), store['wports']
    c = dict(lut=0.0, lutram=0.0, ff=0, bram=0.0, uram=0, ii_floor=0)
    ports = WRITE_PORTS[kind]
    if ports is not None and wr > ports:
        # Several host cycles a slot (paging): not built.
        c['ii_floor'] = store.get('threads', 1) * math.ceil(wr / ports)
    if kind == 'registers':
        c['ff'] = d * w
        c['lut'] = r * w * math.ceil((d - 1) / 3) if d > 1 else 0.0
        return c
    if kind == 'distributed':
        c['lutram'] = cost.mem_cost(d, w, r, store['rlat'], 'distributed')[0]
        return c
    if store['reg']:
        if store['gap'] is None or store['gap'] < BLOCK_STATE_GAP:
            return None
    elif store['rlat'] < 1:
        return None
    n, grain = lanes(w, store['mask'])
    # A read port beyond the first is a copy of the RAM; a ROM's two ports read.
    copies = r if wr else math.ceil(r / 2)
    if kind == 'block':
        c['bram'] = copies * n * bram36(d, grain)
    else:
        c['uram'] = copies * n * uram(d, grain)
    return c


def page_geometry(bits, threads, p):
    """A paged copy's page RAM: rows, page width and 72-bit read ports."""
    rows = (1 << (max(threads, 1) - 1).bit_length()) * (1 << (p - 1).bit_length())
    pw = max(1, math.ceil(bits / p))
    return rows, pw, math.ceil(pw / 72)


# Off-chip pages (foldsim-fold-static storage <model>:pages=hbm): the HBM
# bandwidth a frame may use, in bytes a host cycle (SLASH's 8 HBM vNoC ports,
# about 128 GB/s a direction, at 200 MHz), and a controller's row.
HBM_BYTES_PER_CYCLE = 640
HBM_COLUMN_BITS = 4096


def page_select_luts(p):
    """LUTs a page bit of the write-back's p:1 select (a paged copy's, a
    paged link's, a memory's rows)."""
    return PAGE_SELECT_LUTS(p) if p > 1 else 0


def _select_mux(p):
    # The page indexed by its binary counter (foldsim-fold-static since
    # 518ccdb; paged links alike): a tree of LUT6 4:1 multiplexers, each
    # taking 3 of the p inputs off, ceil((p - 1) / 3) LUTs a page bit (the
    # Versal CLB has no MUXF7/F8). Calibrated (st32_1031 §10.2, Vivado
    # 2025.2 synthesis + opt_design of a paged copy, the opc unit, out of
    # context on the V80): 1.00 LUT a page bit at p 2, 1.00 at p 4, 2.97 at
    # p 8 (cones of the page RAMs' write-data pins).
    return math.ceil((p - 1) / 3)


PAGE_SELECT_LUTS = _select_mux


def page_cost(bits, threads, p, kind):
    """What a copy's paged register state costs in a kind (block or ultra,
    or hbm: off chip): as kind_cost, with `ii_floor` the frame its state
    passes need, `ports` the page RAM's 72-bit read ports, and off chip
    `hbm_bytes` its traffic a frame (every instance's state read and written
    once), `hbm_ii` the frame that traffic needs at HBM_BYTES_PER_CYCLE and
    `controllers` its virtual memories. None for another kind."""
    if kind not in ('block', 'ultra', 'hbm'):
        return None
    rows, pw, ports = page_geometry(bits, threads, p)
    c = dict(lut=float(pw * page_select_luts(p)), lutram=0.0,
             ff=2 * bits + pw, bram=0.0, uram=0, ii_floor=max(p * threads, p + 4),
             ports=ports)
    if kind == 'block':
        c['bram'] = bram36(rows, pw)
    elif kind == 'ultra':
        c['uram'] = uram(rows, pw)
    else:
        c['ports'] = 0
        c['ii_floor'] = max(p, 2) * threads
        c['hbm_bytes'] = 2 * bits * threads // 8
        c['hbm_ii'] = math.ceil(c['hbm_bytes'] / HBM_BYTES_PER_CYCLE)
        c['controllers'] = math.ceil(pw / HBM_COLUMN_BITS)
    return c


# The off-chip tier (foldsim --state auto). A store kept in HBM costs no
# on-chip RAM but HBM traffic each frame, which the frame must carry: a share
# of HBM_BYTES_PER_CYCLE (the rest is the target's own memory traffic and
# headroom), so `assign` prices traffic in an 'hbm' pool of
# HBM_STATE_SHARE x frame x HBM_BYTES_PER_CYCLE bytes. Off-chip kinds are
# relief kinds: a store takes one only to relieve an on-chip pool that is
# over, the move freeing it at the least HBM traffic per unit freed, so the
# least state that lets the rest fit leaves the chip, deep memories (a row
# of many a frame) before register pages (all of an instance's state, read
# and written each frame).
HBM_STATE_SHARE = 0.75
# The bubbles a frame may take waiting on HBM (the `bubbles` pool): a share of
# the frame, so H stays within (1 + BUBBLE_SHARE) of it.
BUBBLE_SHARE = 0.05
# A virtual memory's read round trip through HBM in host cycles (controller,
# vNoC, HBM and back; a host-cache hit answers in 5): a state pass whose
# instance's read is younger than this waits in bubbles (VMEM_INTERFACE.md,
# controller v1: latency).
HBM_ROUND_TRIP = 64
# The controller's row limit and the AXI beat (512 bits).
VMEM_ROW_BITS = 4096
HBM_BEAT_BYTES = 64
RELIEF_KINDS = ('hbm',)
# The kernel's controllers (foldsim_vmem: request queue, tags, write buffer,
# AXI master; a host cache and the known-zero bits in block RAM where it has
# them), as foldsim_afu instantiates them for a virtual memory or a group of
# them (foldsim --vmem-group), an HBM link stream's own controller and a page
# engine: logic and distributed-RAM LUTs over the features [1, min(dw, 512),
# max(dw - 512, 0), a host cache, a group of more than one member] (`dw` the
# controller's row bits), fitted on every routed V80 image whose kernel had
# them (virtA_1105/cal ctl_fit.py: vtA-vtG, fsd64h, B4; 81 controllers,
# Vivado 2025.2, routed): per controller mean |error| 8.5 % logic, 26 %
# LUTRAM; per image (its controllers summed) within -2.5..+11 %. Before (the
# OOC lines below, cache-less) every image's controllers priced 3-70 % low:
# a host cache adds ~1,560 logic and 400 LUTRAM LUTs, a row past one AXI beat
# about a LUT and two LUTRAM LUTs a bit (the 4096-bit point is fsd64h's alone;
# a group's term is small and negative: its wrapper is ~115 LUTs, the rest is
# the fit).
CTL_LOGIC = (1067.56, 0.4803, 0.9958, 1556.37, -95.68)
CTL_LUTRAM = (299.06, 1.6951, 2.1253, 399.66, -130.94)
# The OOC lines of before (st32_1031 §10.2: the v7 controller IW 9, AW 1,
# known-zero rows, synthesis + opt_design; DW 512: 1,391 + 821; 4096:
# 5,333 + 8,707), kept for comparison (ctl_fit.py 'old').
VMEM_CTL_LOGIC = (828.0, 1.10)
VMEM_CTL_LUTRAM = (-305.0, 2.2)
# The controllers' block RAM (routed): the host cache's lines in true dual
# port with byte lanes (a RAMB36 column of 36 bits, 1,024 rows; port A reads,
# port B writes, reading first), one more for its two tag copies; and the
# known-zero bits (KZ_BITS at most, foldsim --vmem-known-zero-bits) in 16
# one-bit banks, block RAM where a bank is deeper than KZ_BRAM_DEPTH rows
# (Vivado kept 2,368-deep banks in distributed RAM, put 4,096-deep ones in
# block RAM). A page engine or a stream (no cache, no known-zero bits): none
# (the OOC engine's one RAMB36 became distributed RAM routed). Over the 81
# controllers: measured 754.0 RAMB36, priced 736.0 (-2.4 %; before: 166, -78 %).
KZ_BITS = 1 << 16
KZ_BANKS = 16
KZ_BRAM_DEPTH = 3072
# FORWARD 0 (foldsim --vmem-forward 0): every controller priced without
# forwarding from its write buffer (no_forward()).
CTL_FORWARD = 1


def kz_bits(image_rows, budget=KZ_BITS):
    """A controller's known-zero bits (foldsim.py vmem_known_zero): a bit per
    block of 2^shift image rows, the fewest blocks within `budget`."""
    if budget <= 0 or image_rows <= 0:
        return 0
    shift = 0
    while -(-image_rows // (1 << shift)) > budget:
        shift += 1
    return -(-image_rows // (1 << shift))


def ctl_cost(kind, dw, cache_rows=0, members=1, image_rows=0, kz=None):
    """A controller of the kernel (`kind` 'vmem': a virtual memory or a group
    of `members`; 'link': an HBM link stream's own; 'engine': a page engine)
    with rows of `dw` bits: dict(logic, lutram, bram). A virtual memory's
    `cache_rows` host-cache lines and its known-zero bits (`kz`, else those of
    its `image_rows` rows: 2^AW x instances) add block RAM; a stream or an
    engine has neither."""
    dw = max(int(dw), 1)
    cache = 1.0 if kind == 'vmem' and cache_rows else 0.0
    f = (1.0, float(min(dw, 512)), float(max(dw - 512, 0)), cache,
         1.0 if kind == 'vmem' and members > 1 else 0.0)
    logic = sum(a * x for a, x in zip(CTL_LOGIC, f))
    lutram = max(0.0, sum(a * x for a, x in zip(CTL_LUTRAM, f)))
    if CTL_FORWARD == 0:
        fl, fr = NO_FORWARD_ENGINE if kind == 'engine' else NO_FORWARD_VMEM
        logic, lutram = logic * fl, lutram * fr
    bram = 0.0
    if cache:
        bram += math.ceil(((dw + 7) // 8 * 8) / 36) * math.ceil(cache_rows / 1024) + 1
    if kind == 'vmem':
        bits = kz_bits(image_rows) if kz is None else kz
        depth = -(-bits // KZ_BANKS)
        if depth > KZ_BRAM_DEPTH:
            bram += KZ_BANKS * math.ceil(depth / 32768)
    return dict(logic=logic, lutram=lutram, bram=bram)


# The kernel's device-memory masters and the platform's way into device
# memory, which every controller adds to (hw/syn/foldsim/vmem_ports.py,
# SLASH linker/slashkit mem_ctx.py). The kernel has at most MAX_MASTERS
# masters: one per memory port of the target, or one in all where the ports
# are folded (foldsim --fold-ports: the folded endpoint has one master; the
# kernel flow binds only that one, FOLDSIM_MEM_MASTERS), then one per
# controller as far as they go; beyond, controller k shares master k mod M
# (foldsim_vmem_share), or with no master left (or FOLDSIM_VMEM_MASTERS 0)
# the memory ports' master k mod P with its endpoint. `mem_ports` below is
# the memory ports' masters. SLASH binds the masters round robin to SC_MEM_PORTS HBM vNoC
# ports and puts a SmartConnect (sc_mem, at most SC_MAX_SI inputs) on every
# port with more than one. Routed (virtA_1105/cal ctl_fit.py): a share
# 70 logic + 284 LUTRAM LUTs with one input, 986 + 15 more an input
# (fsd64h, 31 shares); an sc_mem node 808 + 2,108 logic and 781 + 801
# LUTRAM a slave input (2 inputs: vtC/vtE 5,024 + 2,383; 4: fsd64h 9,240 +
# 3,985). 32 masters (16 memory ports and 16 controllers, or 32 ports) take
# 8 nodes of 4: 106k LUTs.
MAX_MASTERS = 32
SC_MEM_PORTS = 8
SC_MAX_SI = 16
SHARE_ONE = (70.0, 284.0)
SHARE_EXTRA = (986.0, 15.0)
SC_NODE_LOGIC = (808.0, 2108.0)
SC_NODE_LUTRAM = (781.0, 801.0)


# The masters the controllers' traffic needs where it is known (platform_cost
# `bytes_per_cycle`: a fold's HBM bytes a frame over the frame): its AXI beats
# a host cycle with MASTER_HEADROOM for the static order's bursts (the
# positions where several engines and controllers request at once), at least
# one, at most what the whole HBM share needs (HBM_BYTES_PER_CYCLE, the need
# without it).
MASTER_HEADROOM = 4.0


def platform_cost(mem_ports, controllers, vmem_masters=None, bytes_per_cycle=None):
    """The masters' shares and SLASH's sc_mem tree for a kernel of
    `mem_ports` memory-port masters and `controllers` foldsim_vmem controllers
    (`vmem_masters`: the kernel's FOLDSIM_VMEM_MASTERS, default as many as
    fit; 'best': the count of least LUTs, which the flow's
    FOLDSIM_VMEM_MASTERS sets; `bytes_per_cycle`: the traffic the 'best'
    count must carry, else the whole HBM share): dict(logic, lutram, masters,
    shares, sc_nodes, vmem_masters)."""
    if vmem_masters == 'best':
        # The controllers' traffic needs masters: at least HBM_BYTES_PER_CYCLE
        # of AXI beats a host cycle (or one each), on their own masters or on
        # the memory ports' (FOLDSIM_VMEM_MASTERS 0).
        p, n = max(int(mem_ports), 0), max(int(controllers), 0)
        top = min(n, max(0, MAX_MASTERS - p))
        need = min(n, math.ceil(HBM_BYTES_PER_CYCLE / HBM_BEAT_BYTES))
        if bytes_per_cycle is not None:
            need = min(need, max(1, math.ceil(MASTER_HEADROOM * bytes_per_cycle / HBM_BEAT_BYTES - 1e-9)))
        ms = [m for m in range(top + 1) if (m if m else min(n, p)) >= need] or [top]
        return min((platform_cost(p, n, m) for m in ms),
                   key=lambda c: (c['logic'] + c['lutram'], -c['vmem_masters']))
    p, n = max(int(mem_ports), 0), max(int(controllers), 0)
    room = max(0, MAX_MASTERS - p)
    m = min(n, room if vmem_masters is None else vmem_masters)
    logic = lutram = 0.0
    shares = 0
    if n and m == 0 and p:
        # On the memory ports' masters, each with its endpoint.
        for k in range(min(n, p)):
            on = len(range(k, n, p))
            logic += SHARE_ONE[0] + on * SHARE_EXTRA[0]
            lutram += SHARE_ONE[1] + on * SHARE_EXTRA[1]
            shares += 1
    elif 0 < m < n:
        for k in range(m):
            on = len(range(k, n, m))
            logic += SHARE_ONE[0] + (on - 1) * SHARE_EXTRA[0]
            lutram += SHARE_ONE[1] + (on - 1) * SHARE_EXTRA[1]
            shares += 1
    masters = p + m
    nodes = 0
    for b in range(SC_MEM_PORTS):
        q = len(range(b, masters, SC_MEM_PORTS))
        while q > 1:
            for g in range(0, q, SC_MAX_SI):
                si = min(SC_MAX_SI, q - g)
                logic += SC_NODE_LOGIC[0] + SC_NODE_LOGIC[1] * si
                lutram += SC_NODE_LUTRAM[0] + SC_NODE_LUTRAM[1] * si
                nodes += 1
            q = -(-q // SC_MAX_SI)
    return dict(logic=logic, lutram=lutram, masters=masters, shares=shares, sc_nodes=nodes, vmem_masters=m)


def vmem_ctl(row_bits):
    """A virtual memory controller's (logic LUTs, LUTRAM LUTs) for rows of
    `row_bits`, without a host cache (ctl_cost)."""
    c = ctl_cost('vmem', max(int(row_bits), 64))
    return c['logic'], c['lutram']


def vmem_row_bytes(width, row_bytes_min=1):
    """A virtual memory's row in device memory (layout v7): its bytes rounded
    up to a power of two, at least `row_bytes_min`."""
    return max(row_bytes_min, 1 << ((width + 7) // 8 - 1).bit_length())


def vmem_eligible(m, gap):
    """Why a memory cannot be a virtual memory (foldsim-fold-static
    virtual-memories), or '': a registered read, one read port and at most
    one write port, byte enables or none, a row the controller takes, and a
    state gap of 5 (the pending read set at a state pass seen by the next
    decision). `m`: class_estimates' memory (name, depth, width, reads,
    writes, read latency, style, mask, named); `gap`: the copy's state gap."""
    name, d, w, r, wr, rl, _, mask, named = m[:9]
    if rl != 1:
        return f'read latency {rl}'
    if r != 1 or wr > 1:
        return f'{r} read and {wr} write ports'
    if mask > 1 and (w % mask or (w // mask) % 8):
        return 'mask not whole bytes'
    if w > VMEM_ROW_BITS:
        return f'row of {w} bits'
    if gap is not None and gap < 5:
        return f'state gap {gap}'
    if not named or re.search(r'[\s{},=]', name):
        return 'a name the option cannot carry'
    return ''


def vmem_cost(depth, width, threads, weight=1.0, row_bytes_min=1):
    """What a memory costs kept in HBM (a virtual memory): no RAM of its own
    on chip; its read buffer (a row per instance, read registered: in flops
    for one instance, else distributed or block RAM as Vivado infers it),
    the controller (vmem_ctl: logic in `lut`, its write buffer in `lutram`;
    the logic shared when grouped), and
    `hbm` bytes a frame: each instance's state pass reads a row and writes
    one, `weight` times (accesses per instance and target cycle; 1 at most),
    each at least an AXI beat."""
    row = max(vmem_row_bytes(width, row_bytes_min), HBM_BEAT_BYTES)
    th = 1 << (threads - 1).bit_length() if threads > 1 else 1
    lr, br = cost.mem_cost(th, width, 1, 1) if threads > 1 else (0.0, 0.0)
    c = ctl_cost('vmem', max(width, 64), image_rows=depth * th)
    return dict(lut=c['logic'], lutram=lr + c['lutram'], ff=width, bram=br + c['bram'], uram=0,
                ii_floor=0, hbm=2 * weight * threads * row, ctl=1, ctl_bram=c['bram'],
                ctl_lutram=c['lutram'])


# HBM page streams (foldsim --page-engines): an off-chip page word in columns
# of an AXI beat, carried by engines shared by copies (a page engine is one
# foldsim_vmem of one image, INSTANCES 1, requests tagged by column and
# instance, TAGW): each engine takes a read and a write a host cycle, so a
# copy's page traffic, p x threads positions a frame on each of its columns'
# engines, takes that share of `columns` engines. The engines a fold needs are
# at least the most columns of a copy and the shares' sum (rounded up), as
# many as the schedule's positions let the assignment share (foldsim-fold-
# static assignPageEngines; more where copies' traffic collides).
PAGE_ENGINE_BITS = 512
# An engine's (logic, LUTRAM) LUTs and block RAM: ctl_cost('engine', 512)
# (routed: 1,313 logic + 1,167 LUTRAM, no block RAM; measured on vtC/vtE/vtF
# 1,259-1,391 + 1,356-1,381). The OOC calibration of before (virt_1101 cal/,
# synthesis + opt_design: DW 512 1,657 + 866 + 1 RAMB36; DW 256 1,509 + 550
# + 1) is kept for comparison only.
PAGE_ENGINE_LOGIC = (1361.0, 0.578)
PAGE_ENGINE_LUTRAM = (234.0, 1.234)
PAGE_ENGINE_BRAM = 0.0


# Controllers that answer no read from their write buffer (the kernel's
# foldsim_vmem FORWARD 0, foldsim --vmem-forward 0): the buffer keeps one read
# port and the answer's mux loses its input. Calibrated (a64_1101 cal/, Vivado
# 2025.2 synthesis + opt_design out of context on the V80, V8B + FORWARD, DW
# 512): a page engine 1,657 -> 1,220 logic, 866 -> 570 LUTRAM (1 RAMB36
# both); a virtual memory's controller (IW 9, AW 1, known-zero rows) 1,646 ->
# 1,324 logic, 833 -> 537 LUTRAM. Taken as factors of the calibrated lines.
NO_FORWARD_ENGINE = (1220.0 / 1657.0, 570.0 / 866.0)
NO_FORWARD_VMEM = (1324.0 / 1646.0, 537.0 / 833.0)


def no_forward():
    """Price every controller (virtual memories', page engines', link
    streams' own) as FORWARD 0: the factors above on ctl_cost."""
    global CTL_FORWARD
    CTL_FORWARD = 0


# The engines a fold takes, as foldsim-fold-static assigns them
# (assignPageEngines), from occupancies (`engines_first_fit`): the link
# streams first, then each off-chip copy's columns on the first run of
# `columns` consecutive engines (new ones past the last) whose positions
# leave its page traffic room. Positions are not known before a schedule,
# so a position's use is an occupancy: an engine's share of the frame taken
# (reads; writes alike), in the model's shares (page_engine_share, a copy's
# max(p, 2) x instances; link_stream_cost's ports, a stream's). A copy's page
# reads go in the window between its state passes, in page order after the
# frame before's writes, page 0 two positions before the pass, and a copy's
# state passes may sit close together in the frame, so it shares a run only
# where the run's occupancy plus its own stays within ENGINE_PAGE_OCC. 0.1 is
# the largest that bounds every as-built fold from above (pg_1105 occfit.py,
# built / first-fit at 0.1 / 0.25 / 'max'): a64 m_p64 (II 1,016) 20 / 24 / 16
# / 57, s_p64 14 / 16 / 11 / 26, m_a64 42 / 78 / 70 / 88, s_a64 35 / 59 / 54 /
# 59, g16b and g16c 10 / 10 / 10 / 10, a128 fake_S1_s4 (shift 256) 4 / 5 / 5 /
# 32. A stream's requests sit a lead before its landings: at its own
# positions an engine each (streams landing at the same passes collide: the
# 32-SM schedule g32cs put 17 on 15); with link-hbm-shift (requests up to N
# positions earlier, writes later) on the first engine within
# ENGINE_STREAM_OCC (Driver/engines-first-fit.test: three streams of share
# 0.3 on one; a64 S2_m64, shift 256 at II 81,922: 20 streams 16 -> 7
# engines; a128 fake_S1_s4: 32 streams and a page column on 4).
ENGINE_PAGE_OCC = 0.1
ENGINE_STREAM_OCC = 1.0


def engines_first_fit(copies, streams, shift=0):
    """The page engines of a fold: `copies` [(columns, share of the frame a
    column's page traffic takes)] in the compiler's class order, `streams`
    [share of the frame its requests take]; `shift` the compiler's
    link-hbm-shift. Returns the engine count."""
    occ = []
    for u in streams:
        for e in range(len(occ) if shift > 0 else 0):
            if occ[e] + u <= ENGINE_STREAM_OCC:
                occ[e] += u
                break
        else:
            occ.append(u)
    for cols, u in copies:
        cols = max(int(cols), 1)
        first = 0
        while any(occ[e] + u > ENGINE_PAGE_OCC for e in range(first, min(first + cols, len(occ)))):
            first += 1
        for e in range(first, first + cols):
            if e < len(occ):
                occ[e] += u
            else:
                occ.append(u)
    return len(occ)


def page_engine_ctl(row_bits=PAGE_ENGINE_BITS):
    """A page engine's (logic LUTs, LUTRAM LUTs)."""
    c = ctl_cost('engine', row_bits)
    return c['logic'], c['lutram']


def page_engine_share(bits, threads, p, frame):
    """A copy's off-chip page columns (of PAGE_ENGINE_BITS) and its share of
    engines: its columns times the frame's positions its page reads take."""
    pw = math.ceil(bits / max(p, 1))
    columns = math.ceil(pw / PAGE_ENGINE_BITS)
    return columns, columns * min(1.0, max(p, 2) * threads / max(frame, 1))


# Off-chip pages on engines take their copy's page word in columns of
# PAGE_ENGINE_BITS, and the columns of a copy move together (an engine each),
# so the fold needs at least the widest copy's columns of engines: a copy of
# 259k state bits at p 15 takes 34. Deeper pages narrow the word (the write-
# back select stays about a third of a LUT a state bit at any large p), so off
# chip a copy pages at least deep enough to take PAGE_ENGINE_COLUMNS columns,
# where its state passes still fit the page port share of the frame.
PAGE_ENGINE_COLUMNS = 4


def page_hbm_depth(bits, threads, p, frame, port_share=0.5):
    """The page of a copy's state off chip on engines (page_hbm_cost at a
    `frame`): `p` (its on-chip page), or deeper, up to its word taking
    PAGE_ENGINE_COLUMNS columns, within p x threads <= port_share x frame."""
    top = max(1, int(port_share * frame) // max(threads, 1))
    want = math.ceil(bits / (PAGE_ENGINE_BITS * max(PAGE_ENGINE_COLUMNS, 1)))
    return max(p, min(top, want))


def page_hbm_cost(bits, threads, p, engines_frame=None):
    """Paged register state off chip (pages=hbm) as a relief kind: page_cost's
    'hbm' with its traffic in the `hbm` pool. With `engines_frame` (HBM page
    streams at that frame): its share of page engines (`ctl`, fractional;
    `engine_columns`) instead of a controller per column."""
    c = page_cost(bits, threads, p, 'hbm')
    if engines_frame:
        columns, share = page_engine_share(bits, threads, p, engines_frame)
        cl, cr = page_engine_ctl()
        return dict(c, lut=c['lut'] + share * cl, lutram=c['lutram'] + share * cr,
                    bram=c['bram'] + share * PAGE_ENGINE_BRAM,
                    hbm=c['hbm_bytes'], ctl=share, engine_columns=columns)
    cl, cr = page_ctl(bits, p, c['controllers'])
    return dict(c, lut=c['lut'] + cl, lutram=c['lutram'] + cr, hbm=c['hbm_bytes'],
                ctl=c['controllers'])


def page_ctl(bits, p, controllers):
    """The (logic, LUTRAM) LUTs of an off-chip page word's controllers: a
    column of the page word (ceil(bits / p) bits) each."""
    pw = math.ceil(bits / max(p, 1))
    cl, cr = vmem_ctl(math.ceil(pw / max(controllers, 1)))
    return cl * controllers, cr * controllers


def page_bound(bits, threads, ports):
    """The frame a copy's register state needs through `ports` 72-bit ports,
    read and written once an instance a frame."""
    return threads * math.ceil(bits / (72 * max(ports, 1)))


def sites(c):
    return c['lut'] + LUTRAM_SITES * c['lutram']


# Memories kept as registers (foldsim --memory-registers, the pass
# foldsim-memories-to-registers). A memory read combinationally folds into a
# store of its depth times the instances, copied once per read port (a
# distributed RAM reads one address a copy); kept as registers it is one row
# of its entries per instance in the copy's state, read whole, so that a read
# at a constant address is a wire and a read at another a select of the
# copy's logic. An associative table matched entry by entry (a cache MSHR's
# address table: 192 entries, 196 read ports) is the case: the A100-class
# SM's L2 banks took 688k distributed-RAM LUTs as memories.
#
# Both forms are priced in LUT sites, a flip-flop at FF_SITES (a slice: 8
# LUT6 and 16 flip-flops; the flops the rest of the model leaves unpriced
# count here, where a form can be mostly flops): the memory at its cheapest
# kind (kind_cost; one combinationally read takes no block RAM), the
# registers as a store of one entry per instance (flops for one instance,
# else the cheaper of flops and distributed RAM; block RAM needs a state gap
# the schedule sets later, so it is not counted) plus the logic the pass
# makes as foldsim_cost.py prices it: a select per read at a non-constant
# address over the entries (padded to a power of two), an address compare
# and enable per entry and write port and a lane enable per mask lane, and
# the merge of a later write port's data (the first is the register's
# enable). A memory becomes registers only when they are cheaper for every
# number of instances a copy may hold (1 and all the module's instances):
# both forms grow about linearly in instances beyond, so the two ends bound
# the choice whatever the fold degree.
FF_SITES = 0.5


def register_ineligible(m):
    """Why a memory (memory_ports) cannot be registers, or ''. As the pass's
    check: read combinationally, written a cycle later, no read-write ports,
    no initial content, written at all; and named in a way the pass's
    option can carry."""
    if m['rlat'] != 0:
        return f"reads registered (latency {m['rlat']})"
    if m['wlat'] != 1:
        return f"writes {m['wlat']} cycles late"
    if m['rwports']:
        return 'read-write ports'
    if m['init']:
        return 'initial content'
    if not m['wports']:
        return 'never written'
    if re.search(r'[,{}"\s]', m['name']):
        return 'a name the pass option cannot carry'
    return ''


def register_form_cost(m, threads):
    """kind_cost-style cost of memory `m` kept as registers for a copy of
    `threads` instances: its state store (cheapest kind) and the pass's
    logic."""
    d, w = m['depth'], m['width']
    th = 1 << (threads - 1).bit_length() if threads > 1 else 1
    state = dict(depth=th, width=d * w, rports=1, wports=1, rlat=0, mask=1, reg=True,
                 gap=None, threads=threads)
    kinds = ('registers',) if th == 1 else ('registers', 'distributed')
    best = min((kind_cost(state, k) for k in kinds), key=form_sites)
    a = max(1, m['abits'])
    lanes = m['mask'] if m['mask'] > 1 else 0
    select = m['reads'] - m['const_reads']
    logic = select * w * ((1 << a) - 1) / 3
    compare = math.ceil(a / 6) + math.ceil(a / 36) + 1 + lanes
    logic += m['wports'] * d * compare + max(0, m['wports'] - 1) * d * w * 0.5
    return dict(best, lut=best['lut'] + logic)


def memory_form_cost(m, threads):
    """The cheapest kind_cost of memory `m` folded into a copy of `threads`
    instances."""
    th = 1 << (threads - 1).bit_length() if threads > 1 else 1
    store = dict(depth=m['depth'] * th, width=m['width'], rports=m['reads'],
                 wports=m['wports'], rlat=m['rlat'], mask=m['mask'], reg=False, gap=None,
                 threads=threads)
    costs = [c for c in (kind_cost(store, k) for k in KINDS) if c is not None]
    return min(costs, key=form_sites)


def form_sites(c):
    return sites(c) + FF_SITES * c['ff']


def memory_as_registers(m, instances):
    """Whether memory `m` (memory_ports) is cheaper as registers for every
    copy of 1 to `instances` instances (the two ends), and the prices
    [(instances, as a memory, as registers)] in LUT sites."""
    rows = [(t, form_sites(memory_form_cost(m, t)), form_sites(register_form_cost(m, t)))
            for t in sorted({1, max(1, instances)})]
    return all(r < mm for _, mm, r in rows), rows


def memory_ports(path, top):
    """The memories of an IR (the flat design): per memory its module, name,
    depth, width, mask lanes, address bits, read and write latency, read
    ports (and how many at a constant address), write and read-write ports,
    initial content; and each module's instances in the design under `top`
    (1 for the top or an uninstantiated module)."""
    head = re.compile(r'^\s*hw\.module(?:\.extern)?(?:\s+private)?\s+@([\w$.]+)')
    inst = re.compile(r'hw\.instance\s+"[^"]*"\s+(?:sym\s+@\S+\s+)?@([\w$.]+)')
    decl = re.compile(r'^(%[\w.$-]+) = seq\.firmem\s+(?:sym\s+@\S+\s+)?(?:name\s+"([^"]*)"\s+)?'
                      r'(\d+),\s*(\d+),\s*\w+,\s*\w+(.*):\s*<\s*(\d+)\s*x\s*(\d+)(?:,\s*mask\s+(\d+))?\s*>')
    port = re.compile(r'(%[\w.$-]+)\[(%[\w.$-]+)\]')
    mems, children, module, consts, by_ssa = [], {}, None, set(), {}
    for line in open(path):
        s = line.strip()
        if s.startswith('hw.module'):
            mm = head.match(s)
            module = mm.group(1) if mm else None
            consts, by_ssa = set(), {}
            children.setdefault(module, {})
            continue
        if module is None:
            continue
        if 'hw.instance' in s:
            mm = inst.search(s)
            if mm:
                children[module][mm.group(1)] = children[module].get(mm.group(1), 0) + 1
            continue
        if ' = hw.constant ' in s:
            consts.add(s.split(' = ', 1)[0])
            continue
        if 'seq.firmem' not in s:
            continue
        mm = decl.match(s)
        if mm:
            depth = int(mm.group(6))
            rec = dict(module=module, name=mm.group(2) if mm.group(2) is not None else mm.group(1)[1:],
                       depth=depth, width=int(mm.group(7)), mask=int(mm.group(8) or 1),
                       abits=(depth - 1).bit_length(), rlat=int(mm.group(3)), wlat=int(mm.group(4)),
                       init=' init ' in f' {mm.group(5)} ', reads=0, const_reads=0, wports=0, rwports=0,
                       _addrs=[], _consts=consts)
            by_ssa[mm.group(1)] = rec
            mems.append(rec)
            continue
        mm = port.search(s)
        rec = mm and by_ssa.get(mm.group(1))
        if not rec:
            continue
        if 'seq.firmem.read_write_port' in s:
            rec['rwports'] += 1
        elif 'seq.firmem.write_port' in s:
            rec['wports'] += 1
        elif 'seq.firmem.read_port' in s:
            rec['reads'] += 1
            rec['_addrs'].append(mm.group(2))
    for rec in mems:    # a constant may follow its use in a graph region
        rec['const_reads'] = sum(a in rec['_consts'] for a in rec.pop('_addrs'))
        del rec['_consts']
    count = {}

    def instances(m, seen=()):
        if m == top:
            return 1
        if m not in count:
            n = sum(k * instances(p, seen + (m,)) for p, ch in children.items()
                    for c, k in ch.items() if c == m and p not in seen)
            count[m] = n
        return count[m]
    return mems, {m: max(1, instances(m)) for m in children}


def usage(c, pool):
    if pool == 'sites':
        return sites(c)
    if pool == 'luts':
        return fold_load(c['lut'] + c['lutram'], c['bram'], c['uram'])
    return c.get(pool, 0.0)


def assign(units, base, caps, extra=None, paging=False, price_extra=False):
    """A kind per store. `units`: dicts with 'costs' {kind: cost (kind_cost,
    summed over the unit's stores)}, 'count' stores of it, and optionally
    'fixed' (a kind it keeps); `base`: the usage of everything else per pool;
    `caps`: per pool, the most it may use (placement); `extra`: more pools
    to meet once placed (the LUTs that route), never at placement's cost.
    Returns (per unit, its stores per kind; the usage per pool).

    Each store first takes its cheapest kind, every pool priced by its share
    of the capacity; then, while a pool is over, the move that frees it at
    the least price elsewhere, never onto a pool without room, so the
    overflow only falls and the repair ends. A relief kind (RELIEF_KINDS:
    off chip) is never a first choice, only a move of the repair.

    With `price_extra` (--state auto) the extra pools are priced too, and the
    prices then follow the pools' use: the choice is made again with each
    pool's price weighted by the square of its utilization in the last
    (POLISH_ROUNDS times), so that a pool with room (block RAM nobody needs)
    stops looking dear and the binding one (the routing load) gets the slack;
    of the choices that place (and route), the one whose most used pool is
    least used is kept."""
    pools = list(caps) + list(extra or {})
    limit = dict(caps, **(extra or {}))
    # Kinds of their own (paged links, `block:4`) after the stores' kinds.
    own = sorted({k for u in units for k in u['costs']} - set(KINDS) - {'rows'} -
                 set(RELIEF_KINDS))
    order = KINDS + ('rows',) + tuple(own) + RELIEF_KINDS

    def usage_of(i, k, p):
        return usage(units[i]['costs'][k], p)

    def admissible(u, k):
        c = u['costs'].get(k)
        return c is not None and (paging or not c['ii_floor'])

    def solve(weight):
        def price(c):
            return sum(usage(c, p) * w for p, w in weight.items())

        choice = []
        for u in units:
            kinds = [u['fixed']] if u.get('fixed') in u['costs'] else \
                [k for k in order if k not in RELIEF_KINDS and admissible(u, k)]
            if not kinds:
                kinds = [k for k in RELIEF_KINDS if admissible(u, k)]
            k = min(kinds, key=lambda k: (price(u['costs'][k]), order.index(k)))
            choice.append({k: u['count']})
        used = {p: base.get(p, 0.0) + sum(n * usage_of(i, k, p)
                                          for i, ch in enumerate(choice) for k, n in ch.items())
                for p in pools}

        def relieve(p, active):
            """The move that frees pool p at the least price elsewhere per unit
            of p freed: (key, unit, from, to, stores), or None."""
            best = None
            for i, u in enumerate(units):
                if u.get('fixed') in u['costs']:
                    continue
                for a, n in choice[i].items():
                    for b in order:
                        if b == a or not admissible(u, b):
                            continue
                        relief = usage_of(i, a, p) - usage_of(i, b, p)
                        if relief <= 0:
                            continue
                        m = min(n, math.ceil((used[p] - limit[p]) / relief))
                        for q in active:
                            dq = usage_of(i, b, q) - usage_of(i, a, q)
                            if q != p and dq > 0:
                                room = limit[q] - used[q]
                                m = min(m, int(room // dq) if room > 0 else 0)
                        if m <= 0:
                            continue
                        more = sum((usage_of(i, b, q) - usage_of(i, a, q)) / limit[q]
                                   for q in active if q != p and limit[q])
                        key = (more / (relief / limit[p]), i, order.index(b))
                        if best is None or key < best[0]:
                            best = (key, i, a, b, m)
            return best

        def repair(active):
            while True:
                best = None
                for p in active:
                    if used[p] > limit[p] + 1e-9:
                        best = relieve(p, active)
                        if best:
                            break
                if best is None:
                    return all(used[p] <= limit[p] + 1e-9 for p in active)
                _, i, a, b, m = best
                choice[i][a] -= m
                if not choice[i][a]:
                    del choice[i][a]
                choice[i][b] = choice[i].get(b, 0) + m
                for q in pools:
                    used[q] += m * (usage_of(i, b, q) - usage_of(i, a, q))

        if repair(list(caps)) and extra:
            repair(pools)
        return choice, used

    if ASSIGN_EXACT and price_extra:
        found = assign_exact(units, base, caps, extra, paging)
        if found is not None:
            return found
    priced = dict(caps, **(extra or {})) if price_extra else caps
    weight = {p: 1 / v for p, v in priced.items() if v}
    choice, used = solve(weight)
    if not price_extra:
        return choice, used

    # The off-chip pools are not balanced: the least state leaves the chip
    # (the first choice's), and a polished choice that moves more is not kept.
    offchip = [p for p in OFFCHIP_POOLS if p in limit]
    first = {p: used[p] for p in offchip}

    def peak(used):
        ok = all(used[p] <= limit[p] + 1e-9 for p in pools)
        more = any(used[p] > first[p] + 1e-9 for p in offchip)
        return (not ok, more, max(used[p] / limit[p] for p in pools
                                  if limit[p] and p not in offchip))
    best = (peak(used), choice, used)
    for _ in range(POLISH_ROUNDS):
        weight = {p: (max(used[p], 0.0) / limit[p]) ** 2 / limit[p] for p in pools
                  if limit[p] and p not in offchip}
        weight.update({p: 1 / limit[p] for p in offchip if limit[p]})
        choice, used = solve(weight)
        if peak(used) < best[0]:
            best = (peak(used), choice, used)
    if best[0][0]:
        # Nothing placed: several pools over pull the repair's single moves
        # apart. Search the pools' prices instead (multiplicative weights:
        # each store its cheapest kind at the prices, a pool's price raised
        # by its overflow and lowered by its slack, MW_ROUNDS times), off
        # chip only where a pool on chip is over; repair the least
        # overflowing choice, then bring stores back on chip while they fit.
        found = mw_search(units, base, limit, pools, order, admissible, offchip)
        if found is not None:
            choice, used = found
            if peak(used) < best[0]:
                best = (peak(used), choice, used)
    return best[1], best[2]


def relief_pair(units, kinds, use, ch, used, lim, score, now):
    """mw_search's pair of moves: a store into a relief kind (off chip),
    then the single move that lowers the score most with the room it made:
    (score, [(unit, kind), (unit, kind)], usage), or None."""
    best = None
    for r, u in enumerate(units):
        a = ch[r]
        for b in kinds[r]:
            if b not in RELIEF_KINDS or b == a:
                continue
            mid = {p: used[p] + u['count'] * (use[r][b][p] - use[r][a][p]) for p in lim}
            for i, v in enumerate(units):
                if i == r or len(kinds[i]) < 2:
                    continue
                x = ch[i]
                for y in kinds[i]:
                    if y == x:
                        continue
                    new = {p: mid[p] + v['count'] * (use[i][y][p] - use[i][x][p]) for p in lim}
                    sc = score(new)
                    if sc < now and (best is None or sc < best[0]):
                        best = (sc, [(r, b), (i, y)], new)
    return best


# Rounds and step of assign's price search (mw_search).
MW_ROUNDS = 400
MW_STEP = 0.3
MW_DESCENT = 2000


def mw_search(units, base, limit, pools, order, admissible, offchip):
    """assign's search of the pools' prices when its repair leaves pools
    over: (choice, used) of the least overflow found, its stores brought
    back on chip (off-chip kinds left) while every pool stays within its
    limit; None if no unit can move."""
    kinds = []
    for u in units:
        if u.get('fixed') in u['costs']:
            kinds.append([u['fixed']])
        else:
            kinds.append([k for k in order if admissible(u, k)] or list(u['costs']))
    use = [{k: {p: usage(u['costs'][k], p) for p in pools} for k in ks}
           for u, ks in zip(units, kinds)]
    price = {p: 1.0 / limit[p] for p in pools if limit[p]}

    def pick(price):
        out = []
        for ks, us in zip(kinds, use):
            out.append(min(ks, key=lambda k: (sum(us[k][p] * w for p, w in price.items()),
                                               order.index(k) if k in order else 0)))
        return out

    def totals(ch):
        used = {p: base.get(p, 0.0) for p in pools}
        for i, (k, u) in enumerate(zip(ch, units)):
            for p in pools:
                used[p] += u['count'] * use[i][k][p]
        return used

    def overflow(used):
        return sum(max(0.0, used[p] / limit[p] - 1) for p in pools if limit[p])

    best = None
    for r in range(MW_ROUNDS):
        ch = pick(price)
        used = totals(ch)
        key = (overflow(used), max(used[p] / limit[p] for p in pools
                                   if limit[p] and p not in offchip))
        if best is None or key < best[0]:
            best = (key, ch, used)
        for p in price:
            ratio = used[p] / limit[p]
            price[p] *= math.exp(MW_STEP * max(-1.0, min(1.0, ratio - 1)))
            # Off chip costs nothing until an on-chip pool is over; on chip
            # never drops below a share of its first price.
            price[p] = max(price[p], (1e-3 if p in offchip else 0.05) / limit[p])
    if best is None:
        return None
    _, ch, used = best
    ch = list(ch)
    # Then single moves while one lowers the overflow (steepest first; on a
    # tie in overflow, the one that frees the most of the pools on chip).
    lim = {p: limit[p] for p in pools if limit[p]}

    def score(u):
        # The overflow, then the use of the pools on chip (a move that
        # frees one into another with room, as state into HBM making room
        # in a RAM for links, lowers it).
        return (round(sum(max(0.0, u[p] / lim[p] - 1) for p in lim), 9),
                sum(u[p] / lim[p] for p in lim if p not in offchip))

    for _ in range(MW_DESCENT):
        now = score(used)
        if now[0] <= 0:
            break
        move = None
        for i, u in enumerate(units):
            if len(kinds[i]) < 2:
                continue
            a = ch[i]
            for b in kinds[i]:
                if b == a:
                    continue
                new = {p: used[p] + u['count'] * (use[i][b][p] - use[i][a][p]) for p in lim}
                sc = score(new)
                if sc < now and (move is None or sc < move[0]):
                    move = (sc, i, b, new)
        if move is None:
            # No single move helps: a store off chip (its controller's LUTs
            # on a pool over) that makes room in a RAM, and the best move
            # into that room after it.
            move = relief_pair(units, kinds, use, ch, used, lim, score, now)
            if move is None:
                break
            for i, b in move[1]:
                ch[i] = b
            used = dict(used, **move[2])
            continue
        _, i, b, new = move
        ch[i] = b
        used = dict(used, **new)
    choice = [{k: u['count']} for k, u in zip(ch, units)]
    # Back on chip, where every pool stays within its limit.
    for i, u in enumerate(units):
        k = ch[i]
        if k not in RELIEF_KINDS or u.get('fixed') in u['costs']:
            continue
        for b in sorted((x for x in kinds[i] if x not in RELIEF_KINDS),
                        key=lambda x: sum(use[i][x][p] / limit[p] for p in pools
                                          if limit[p] and p not in offchip)):
            new = {p: used[p] + u['count'] * (use[i][b][p] - use[i][k][p]) for p in pools}
            if all(new[p] <= limit[p] + 1e-9 or new[p] <= used[p] + 1e-9 for p in pools):
                choice[i] = {b: u['count']}
                used = new
                break
    return choice, used


# Rounds of assign's utilization-weighted prices (price_extra), and the pools
# they leave alone (off chip: HBM traffic and the bubbles it costs).
POLISH_ROUNDS = 3
OFFCHIP_POOLS = ('hbm', 'bubbles')

# assign's exact mode (foldsim_degree.py state --exact): the kinds as a
# 0-1 program solved by HiGHS (scipy.optimize.milp), not the repair and the
# polish. Its time limit in seconds (the best choice found by then is kept).
ASSIGN_EXACT = False
EXACT_SECONDS = 120
# What the exact choice weighs beside the routing load: the off-chip pools,
# so that of two choices of one load the one with less state off chip wins
# (the least state leaves the chip).
EXACT_OFFCHIP_WEIGHT = 1e-4


def assign_exact(units, base, caps, extra=None, paging=False):
    """assign's choice as a 0-1 program: a kind per unit (all its stores),
    every pool of `caps` within its cap, the routing load (the 'luts' pool;
    usage) as low as it goes, and within `extra` where some choice is (else
    the least load: the caller sees it over). Off-chip kinds are kinds like
    any other here; the off-chip pools weigh EXACT_OFFCHIP_WEIGHT of their
    share. Returns assign's (choice, used), or None where no choice meets
    `caps`. Needs scipy (HiGHS)."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    limit = dict(caps, **(extra or {}))
    pools = list(limit)
    var = []
    for i, u in enumerate(units):
        if u.get('fixed') in u['costs']:
            kinds = [u['fixed']]
        else:
            kinds = [k for k, c in u['costs'].items()
                     if c is not None and (paging or not c['ii_floor'])]
        var.extend((i, k) for k in kinds)
    if not var:
        return None
    use = {p: np.array([units[i]['count'] * usage(units[i]['costs'][k], p) for i, k in var])
           for p in set(pools) | {'luts'}}
    load = use['luts'] / (limit.get('luts') or route_luts())
    cost = load + sum(EXACT_OFFCHIP_WEIGHT * use[p] / limit[p] for p in OFFCHIP_POOLS
                      if p in limit and limit[p])
    one = np.zeros((len(units), len(var)))
    for j, (i, _) in enumerate(var):
        one[i, j] = 1.0

    def solve(hard):
        rows, hi = [one], [np.ones(len(units))]
        for p in hard:
            s = max(limit[p], 1.0)
            rows.append((use[p] / s)[None, :])
            hi.append(np.array([(limit[p] - base.get(p, 0.0)) / s]))
        lo = np.concatenate([np.ones(len(units))] + [np.full(1, -np.inf)] * len(hard))
        r = milp(cost, constraints=LinearConstraint(np.vstack(rows), lo, np.concatenate(hi)),
                 integrality=np.ones(len(var)), bounds=Bounds(0, 1),
                 options=dict(time_limit=EXACT_SECONDS))
        return r.x if r.x is not None and r.status in (0, 1) else None

    x = solve(pools)
    if x is None and extra:
        x = solve(list(caps))
    if x is None:
        return None
    choice = [dict() for _ in units]
    for j, (i, k) in enumerate(var):
        if x[j] > 0.5:
            choice[i] = {k: units[i]['count']}
    used = {p: base.get(p, 0.0) + sum(units[i]['count'] * usage(units[i]['costs'][k], p)
                                      for i, ch in enumerate(choice) for k in ch)
            for p in pools}
    return choice, used


def read_points(path):
    import csv
    return [r for r in csv.DictReader(open(path))]


def point_slrs(p):
    """The SLRs a build of the points file could spread over: the device's
    out of context, the region's in an image, fewer with fewer SLR groups."""
    region = DEVICE_SLRS if p['source'] == 'ooc' else REGION['slrs']
    return route_slrs(int(p.get('groups') or 0), region)


def fit_route(points, layout=True):
    """Least squares of the router's estimate (mean of vertical and
    horizontal) on L and L * L / S (`layout`; else L^2, the fit without the
    layout term), through the origin; each build's residual and leave-one-out
    error; the routed and failed builds' ranges, the limit between them, the
    builds it misclassifies, and those the fit and limit of the others
    misclassify (leave-one-out)."""
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from foldsim_degree import lstsq
    load = [route_load(float(p['luts']), float(p['bram']), float(p['uram'])) / 1e6 for p in points]
    slrs = [point_slrs(p) if layout else 1.0 for p in points]
    rows = [[x, x * x / s] for x, s in zip(load, slrs)]
    ys = [(float(p['gvu']) + float(p['ghu'])) / 2 for p in points]
    routes = [p['routed'] == 'yes' for p in points]

    def limit(est, keep):
        # The limit separates the builds by the fit itself (what a choice is
        # priced with): midway between the most congested routed build's fit
        # and the least congested failed one's.
        routed = max(est[i] for i in keep if routes[i])
        failed = min(est[i] for i in keep if not routes[i])
        return routed, failed, (routed + failed) / 2

    def fit(keep):
        c = lstsq([rows[i] for i in keep], [ys[i] for i in keep])
        return c and [sum(a * b for a, b in zip(c, r)) for r in rows]

    every = range(len(points))
    est = fit(every)
    routed, failed, lim = limit(est, every)
    out = []
    for i, p in enumerate(points):
        keep = [j for j in every if j != i]
        others = fit(keep)
        loo = others and others[i] - ys[i]
        out.append(dict(name=p['name'], luts=int(float(p['luts'])), load=int(load[i] * 1e6),
                        slrs=slrs[i], routed=p['routed'], y=ys[i], est=est[i],
                        res=est[i] - ys[i], loo=loo,
                        wrong=(est[i] <= lim) != routes[i],
                        loo_wrong=others is None or
                        (others[i] <= limit(others, keep)[2]) != routes[i]))
    rms = math.sqrt(sum(o['res'] ** 2 for o in out) / len(out))
    loo = math.sqrt(sum(o['loo'] ** 2 for o in out if o['loo'] is not None) / len(out))
    return dict(coef=lstsq(rows, ys), rms=rms, loo=loo, routed=routed, failed=failed,
                limit=lim, points=out, layout=layout)


def route_point(d):
    """A points-file row of an out-of-context build directory (vivado.log,
    synth_util.rpt): synthesized LUTs, RAMB36 and URAM, whether routing
    finished legally, the router's global routing utilization, and the SLR
    groups the build applied (floorplan.txt: foldsim_floorplan.tcl's
    assignments); None if the build has not reached the router."""
    log = open(os.path.join(d, 'vivado.log'), errors='replace').read()
    util = open(os.path.join(d, 'synth_util.rpt'), errors='replace').read()
    plan = os.path.join(d, 'floorplan.txt')
    groups = len({line.split()[1] for line in open(plan) if len(line.split()) > 1}) \
        if os.path.exists(plan) else 0

    def row(label):
        m = re.search(r'\|\s*' + re.escape(label) + r'\s*\|\s*([\d.]+)', util)
        return m and float(m.group(1))

    def last(p):
        m = re.findall(p, log)
        return m[-1] if m else None
    gv = last(r'Global Vertical Routing Utilization\s*=\s*([\d.]+)')
    gh = last(r'Global Horizontal Routing Utilization\s*=\s*([\d.]+)')
    if 'not legally routed' in log:
        routed = 'FAIL'
    elif 'Number of Unrouted Nets             = 0' in log and \
            os.path.exists(os.path.join(d, 'routed_timing_summary.rpt')):
        routed = 'yes'
    else:
        return None
    luts = row('CLB LUTs*') or row('CLB LUTs')
    overlaps = last(r'There are (\d+) node overlaps') or ''
    return (f"{os.path.basename(d.rstrip('/'))},ooc,{luts:.0f},{row('LUT as Logic'):.0f},"
            f"{row('LUT as Memory'):.0f},{row('Block RAM Tile'):g},{row('URAM'):.0f},{routed},"
            f"{gv},{gh},{overlaps},{groups}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit-route', help='the routability fit and its residuals')
    f.add_argument('--points', default=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                    'foldsim_v80_route.csv'))
    g = sub.add_parser('route-points', help='points-file rows of OOC build directories')
    g.add_argument('dirs', nargs='+')
    a = ap.parse_args()
    if a.cmd == 'route-points':
        for d in a.dirs:
            r = route_point(d) if all(os.path.exists(os.path.join(d, f)) for f in
                                      ('vivado.log', 'synth_util.rpt')) else None
            if r:
                print(r)
        return
    points = read_points(a.points)
    r = fit_route(points)
    for o in sorted(r['points'], key=lambda o: o['load']):
        loo = f"{o['loo']:+5.1f}" if o['loo'] is not None else '    -'
        mark = (' MISCLASSIFIED' if o['wrong'] else '') + \
            (' (leave-one-out: misclassified)' if o['loo_wrong'] else '')
        print(f"{o['name']:<12} {o['luts']:>9} LUTs, load {o['load']:>9}, S {o['slrs']:<3g}"
              f"{o['routed']:>5} measured {o['y']:5.1f} fit {o['est']:5.1f} ({o['res']:+5.1f}) "
              f"leave-one-out {loo}{mark}")
    a_, b_ = r['coef']

    def load_at(slrs):
        return 1e6 * (-a_ + math.sqrt(a_ * a_ + 4 * b_ / slrs * r['limit'])) / (2 * b_ / slrs)

    def wrong(r, key):
        return ', '.join(o['name'] for o in r['points'] if o[key]) or 'none'
    print(f"congestion = {a_:.3f} L + {b_:.3f} L (L / S) (L in millions, S in SLRs); "
          f"RMS {r['rms']:.2f}, leave-one-out RMS {r['loo']:.2f} over {len(r['points'])} builds")
    print(f"fit of the routed builds up to {r['routed']:.2f} %, of the failed from "
          f"{r['failed']:.2f} %: limit {r['limit']:.2f} % = a load of "
          f"{load_at(REGION['slrs']):.0f} over the region's {REGION['slrs']:g} SLRs, "
          f"{load_at(DEVICE_SLRS):.0f} over {DEVICE_SLRS}")
    print(f"misclassified: {wrong(r, 'wrong')}; leave-one-out: {wrong(r, 'loo_wrong')}")
    base = fit_route(points, layout=False)
    print(f"without the layout term (L and L^2): RMS {base['rms']:.2f}, leave-one-out "
          f"{base['loo']:.2f}; routed up to {base['routed']:.2f} %, failed from "
          f"{base['failed']:.2f} %; misclassified: {wrong(base, 'wrong')}")


if __name__ == '__main__':
    main()
