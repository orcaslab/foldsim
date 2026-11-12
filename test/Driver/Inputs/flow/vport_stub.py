#!/usr/bin/env python3
"""vport_stub.py <N> <out dir> [<M>]: a stand-in for a generated simulator with N static ports (foldsim_sim.sv/.vh/.json), for
the static ports' kernel flow tests (direct HBM, vmem masters): the target ports the kernel top connects (one AXI4 memory port, the control
endpoint's channels by its own names, profile_finish) and, per static port p, the simulator's side of foldsim_vport
(vp<p>_*, the 5.1 contract's signals) with FOLDSIM_VPORT<p>_* in foldsim_sim.vh. Its logic only keeps every input
observable (no behaviour). Ports differ: port 0 copies (COPY), port 1 has 1024-bit rows of two beats, port 2 takes its
write numbers from its sender (NEED_IN), port 3 keeps a write-number table (HZ 4); the rest 512-bit rows. With M, also
M virtual memories (vm<k>_*, FOLDSIM_VMEM_*: 512-bit rows, 16 of them, 2 instances; memory 1 read-first), whose
regions come first in the kernel's vmem space."""
import json, os, sys

n, out = int(sys.argv[1]), sys.argv[2]
nvm = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(out, exist_ok=True)


def params(p):
    q = dict(DW=512, MW=64, TW=8, OW=22, LMAX=1, COPY=0, NEED_IN=0, HZ=0, BYTES=1 << 28, ZERO=1, FS=8, KS=0)
    if p == 0:
        q['COPY'] = 1
    if p == 1:
        q.update(DW=1024, MW=128, LMAX=2)
    if p == 2:
        q['NEED_IN'] = 1
    if p == 3:
        q['HZ'] = 4
    return q


def clog2(x):
    return max(0, (x - 1).bit_length())


vh = ['`define FOLDSIM_NUM_CLASSES 1'] + ([f'`define FOLDSIM_VPORTS {n}'] if n else [])
off = 0
ports = [('in', 'hostClock', 0), ('in', 'hostReset', 1), ('out', 'profile_finish', 1)]
# The control endpoint's channels, by its own names (the kernel's control binding).
for name, w, drives in [('reset', 1, True), ('dcr_req_valid', 1, True), ('dcr_req_rw', 1, True), ('dcr_req_addr', 12, True),
                        ('dcr_req_data', 32, True), ('start', 1, True), ('dcr_rsp_valid', 1, False),
                        ('dcr_rsp_data', 32, False), ('busy', 1, False)]:
    d_in, d_out = ('in', 'out') if drives else ('out', 'in')
    ports += [(d_in, f'{name}_valid', 1), (d_out, f'{name}_ready', 1), (d_in, f'{name}_bits', w)]
# One AXI4 memory port (512 bits, id 4, the kernel's channel names; SPEC 0).
for name, w, d in [('awready', 1, 'in'), ('wready', 1, 'in'), ('arready', 1, 'in'), ('rvalid', 1, 'in'), ('rdata', 512, 'in'),
                   ('rlast', 1, 'in'), ('rid', 4, 'in'), ('rresp', 2, 'in'), ('bvalid', 1, 'in'), ('bresp', 2, 'in'),
                   ('bid', 4, 'in'), ('awvalid', 1, 'out'), ('awaddr', 64, 'out'), ('awid', 4, 'out'), ('awlen', 8, 'out'),
                   ('wvalid', 1, 'out'), ('wdata', 512, 'out'), ('wstrb', 64, 'out'), ('wlast', 1, 'out'),
                   ('arvalid', 1, 'out'), ('araddr', 64, 'out'), ('arid', 4, 'out'), ('arlen', 8, 'out'),
                   ('rready', 1, 'out'), ('bready', 1, 'out')]:
    other = 'out' if d == 'in' else 'in'
    ports += [(d, f'm_axi_mem_0_{name}_valid', 1), (other, f'm_axi_mem_0_{name}_ready', 1), (d, f'm_axi_mem_0_{name}_bits', w)]
for p in range(n):
    q = params(p)
    vh += [f'`define FOLDSIM_VPORT{p}_{k} {v}' for k, v in q.items()]
    # Its region in the kernel's vmem space where the ports are on the vmem masters (foldsim.py's layout).
    vh.append(f"`define FOLDSIM_VPORT{p}_BASE 64'h{off:x}")
    off += -(-q['BYTES'] // 4096) * 4096
    lrw, lw = max(1, clog2(q['LMAX'])), max(1, clog2((q['DW'] + 511) // 512))
    ports += [('out', f'vp{p}_rd_valid', 1), ('out', f'vp{p}_rd_addr', q['OW']), ('out', f'vp{p}_rd_len', lrw),
              ('out', f'vp{p}_rd_tag', q['TW']), ('out', f'vp{p}_rd_fence', 4), ('out', f'vp{p}_rd_new', 1),
              ('out', f'vp{p}_rd_copy', 1), ('out', f'vp{p}_rd_copy_addr', q['OW']), ('out', f'vp{p}_rd_after', 1),
              ('out', f'vp{p}_rd_need', 16), ('out', f'vp{p}_wr_valid', 1), ('out', f'vp{p}_wr_addr', q['OW']),
              ('out', f'vp{p}_wr_len', lw), ('out', f'vp{p}_wr_pair', 1), ('out', f'vp{p}_wr_data', q['DW']),
              ('out', f'vp{p}_wr_mask', q['MW']), ('in', f'vp{p}_rsp_valid', 1), ('in', f'vp{p}_rsp_tag', q['TW']),
              ('in', f'vp{p}_rsp_beat', lrw), ('in', f'vp{p}_rsp_last', 1), ('in', f'vp{p}_rsp_copied', 1),
              ('in', f'vp{p}_rsp_data', 512), ('in', f'vp{p}_rd_free', 16), ('in', f'vp{p}_wr_free', 16),
              ('in', f'vp{p}_w_count', 16)]

if n:
    vh.append(f"`define FOLDSIM_VPORT_REGION_BYTES 64'h{off:x}")
# Virtual memories: foldsim.py's header lines and the simulator's side of their controllers.
vbase = 0
for k in range(nvm):
    size = 16 * 2 * 64
    vh += [f'`define FOLDSIM_VMEM_{k}_{key} {v}' for key, v in (('IW', 1), ('AW', 4), ('DW', 512), ('MW', 64), ('INSTANCES', 2))]
    vh += [f'`define FOLDSIM_VMEM_{k}_RDWRF {int(k == 1)}', f"`define FOLDSIM_VMEM_{k}_BASE 64'h{vbase:x}",
           f'`define FOLDSIM_VMEM_{k}_ROW_BYTES 64', f'`define FOLDSIM_VMEM_{k}_KNOWN_ZERO 1', f'`define FOLDSIM_VMEM_{k}_KZ_SHIFT 0']
    vbase += -(-size // 4096) * 4096
    ports += [('out', f'vm{k}_rd_valid', 1), ('out', f'vm{k}_rd_inst', 1), ('out', f'vm{k}_rd_row', 4),
              ('out', f'vm{k}_wr_valid', 1), ('out', f'vm{k}_wr_inst', 1), ('out', f'vm{k}_wr_row', 4),
              ('out', f'vm{k}_wr_data', 512), ('out', f'vm{k}_wr_mask', 64), ('in', f'vm{k}_rsp_valid', 1),
              ('in', f'vm{k}_rsp_inst', 1), ('in', f'vm{k}_rsp_data', 512), ('in', f'vm{k}_free', 16)]
if nvm:
    vh += [f'`define FOLDSIM_VMEM_COUNT {nvm}', '`define FOLDSIM_VMEM_FORWARD 0', f"`define FOLDSIM_VMEM_BYTES 64'h{vbase:x}"]

# The module: every output a register of the XOR of every input (all observable), reset to 0.
ins = [nm for d, nm, w in ports if d == 'in' and nm not in ('hostClock', 'hostReset')]
outs = [(nm, w) for d, nm, w in ports if d == 'out']
decl = []
for d, nm, w in ports:
    rng = f'[{w - 1}:0] ' if w > 1 else ''
    decl.append(f'    {"input " if d == "in" else "output"} wire {rng}{nm}')
sv = ['// Generated by vport_stub.py: a stand-in simulator with static ports (no behaviour).',
      'module foldsim_sim (', ',\n'.join(decl), ');', '    reg mix;',
      '    always @(posedge hostClock) mix <= hostReset ? 1\'b0 : ^{' + ', '.join(ins) + '};']
sv += [f'    assign {nm} = {{{w}{{mix}}}};' for nm, w in outs]
sv.append('endmodule')
open(os.path.join(out, 'foldsim_sim.sv'), 'w').write('\n'.join(sv) + '\n')
open(os.path.join(out, 'foldsim_sim.vh'), 'w').write('\n'.join(vh) + '\n')
json.dump({'top': 'Stub', 'ports': [dict(dir=d, name=nm, width=w) for d, nm, w in ports]},
          open(os.path.join(out, 'foldsim_sim.json'), 'w'), indent=1)
