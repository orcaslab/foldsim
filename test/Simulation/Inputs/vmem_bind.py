#!/usr/bin/env python3
"""vmem_bind.py <foldsim_sim.json> <out dir>: tb_vmem_multi.sv's binding of the
simulator's virtual memories to vmem_group_model: each controller (each
memory alone, or each of the json's vmem_groups) a model, its members the
memories in member order. Writes vm_decl.svh (wires, models) and vm_ports.svh
(the simulator's port connections) to <out dir>."""
import json
import os
import sys

info = json.load(open(sys.argv[1]))
out = sys.argv[2]
vms = info['vmem']
groups = [g['members'] for g in info['vmem_groups']] if 'vmem_groups' in info else [[k] for k in range(len(vms))]
vms = vms or []
decl, ports = [], []
for k, v in enumerate(vms):
    iw, aw, dw, mw = v['inst_width'], v['row_width'], v['data_width'], v['mask_width']
    decl.append(f'logic vm{k}_rsp_valid; logic [{iw-1}:0] vm{k}_rsp_inst; logic [{dw-1}:0] vm{k}_rsp_data; '
                f'logic [15:0] vm{k}_free;')
    decl.append(f'wire vm{k}_rd_valid, vm{k}_wr_valid; wire [{iw-1}:0] vm{k}_rd_inst, vm{k}_wr_inst; '
                f'wire [{aw-1}:0] vm{k}_rd_row, vm{k}_wr_row; wire [{dw-1}:0] vm{k}_wr_data; wire [{mw-1}:0] vm{k}_wr_mask;')
    for p in ('rsp_valid', 'rsp_inst', 'rsp_data', 'free', 'rd_valid', 'rd_inst', 'rd_row', 'wr_valid', 'wr_inst',
              'wr_row', 'wr_data', 'wr_mask'):
        ports.append(f'.vm{k}_{p}(vm{k}_{p}),')
decl.append(f'localparam G = {max(1, len(groups))};')
decl.append('bit model_ok [G];')
if not groups:
    decl.append('initial model_ok[0] = 1;')
for g, ks in enumerate(groups):
    n = len(ks)
    lst = lambda f: ', '.join(str(f(vms[k])) for k in ks)
    decl.append(f'localparam int G{g}_MMW [{n}] = \'{{{lst(lambda v: v["mask_width"])}}};')
    decl.append(f'localparam int G{g}_MDW [{n}] = \'{{{lst(lambda v: v["data_width"])}}};')
    decl.append(f'localparam int G{g}_RF [{n}] = \'{{{lst(lambda v: int(v["read_first"]))}}};')
    decl.append(f'localparam int G{g}_MIW [{n}] = \'{{{lst(lambda v: v["inst_width"])}}};')
    decl.append(f'localparam int G{g}_TW [{n}] = \'{{{lst(lambda v: v.get("tag_width", 0))}}};')
    decl.append(f'logic g{g}_rd_valid [{n}], g{g}_wr_valid [{n}], g{g}_rsp_valid [{n}];')
    decl.append(f'logic [7:0] g{g}_rd_inst [{n}], g{g}_wr_inst [{n}]; logic [7:0] g{g}_rsp_inst;')
    decl.append(f'logic [15:0] g{g}_rd_row [{n}], g{g}_wr_row [{n}];')
    decl.append(f'logic [1023:0] g{g}_wr_data [{n}]; logic [1023:0] g{g}_rsp_data; logic [127:0] g{g}_wr_mask [{n}];')
    decl.append(f'logic [15:0] g{g}_free [{n}];')
    for i, k in enumerate(ks):
        decl.append(f'assign g{g}_rd_valid[{i}] = vm{k}_rd_valid; assign g{g}_rd_inst[{i}] = 8\'(vm{k}_rd_inst); '
                    f'assign g{g}_rd_row[{i}] = 16\'(vm{k}_rd_row);')
        decl.append(f'assign g{g}_wr_valid[{i}] = vm{k}_wr_valid; assign g{g}_wr_inst[{i}] = 8\'(vm{k}_wr_inst); '
                    f'assign g{g}_wr_row[{i}] = 16\'(vm{k}_wr_row);')
        decl.append(f'assign g{g}_wr_data[{i}] = 1024\'(vm{k}_wr_data); assign g{g}_wr_mask[{i}] = 128\'(vm{k}_wr_mask);')
        decl.append(f'assign vm{k}_rsp_valid = g{g}_rsp_valid[{i}]; assign vm{k}_rsp_inst = g{g}_rsp_inst[{vms[k]["inst_width"]-1}:0]; '
                    f'assign vm{k}_rsp_data = g{g}_rsp_data[{vms[k]["data_width"]-1}:0]; assign vm{k}_free = g{g}_free[{i}];')
    decl.append(f'vmem_group_model #(.N({n}), .MMW(G{g}_MMW), .MDW(G{g}_MDW), .RF(G{g}_RF), .MIW(G{g}_MIW), .TW(G{g}_TW)) model{g} (.clk, .rst, .cyc, '
                f'.lo, .hi, .full, .rd_valid(g{g}_rd_valid), .rd_inst(g{g}_rd_inst), .rd_row(g{g}_rd_row), '
                f'.wr_valid(g{g}_wr_valid), .wr_inst(g{g}_wr_inst), .wr_row(g{g}_wr_row), .wr_data(g{g}_wr_data), '
                f'.wr_mask(g{g}_wr_mask), .rsp_valid(g{g}_rsp_valid), .rsp_inst(g{g}_rsp_inst), .rsp_data(g{g}_rsp_data), '
                f'.free(g{g}_free), .ok(model_ok[{g}]));')
# Static ports (fold-static vport-engines): a foldsim_vport each (SYNC, the
# kernel's parameters) on its own vport_axi_model; `zero` rises after the
# reset (the region clean: no fill).
for p, q in enumerate(info.get('vports', [])):
    lrw = max(1, (q['LMAX'] - 1).bit_length())
    lw = max(1, ((q['DW'] + 511) // 512 - 1).bit_length())
    ow, tw, dw, mw = q['OW'], q['TW'], q['DW'], q['MW']
    decl.append(f'wire vp{p}_rd_valid, vp{p}_rd_new, vp{p}_rd_copy, vp{p}_rd_after, vp{p}_wr_valid, vp{p}_wr_pair; '
                f'wire [{ow-1}:0] vp{p}_rd_addr, vp{p}_rd_copy_addr, vp{p}_wr_addr; wire [{lrw-1}:0] vp{p}_rd_len, '
                f'vp{p}_rsp_beat; wire [{lw-1}:0] vp{p}_wr_len; wire [{tw-1}:0] vp{p}_rd_tag, vp{p}_rsp_tag; wire [3:0] vp{p}_rd_fence; '
                f'wire [15:0] vp{p}_rd_need, vp{p}_w_count; wire [{dw-1}:0] vp{p}_wr_data; wire [{mw-1}:0] vp{p}_wr_mask; '
                f'wire vp{p}_rsp_valid, vp{p}_rsp_last, vp{p}_rsp_copied; wire [511:0] vp{p}_rsp_data; '
                f'wire [5:0] vp{p}_rd_room, vp{p}_wr_room;')
    decl.append(f'wire vp{p}_awvalid, vp{p}_awready, vp{p}_wvalid, vp{p}_wready, vp{p}_wlast, vp{p}_bvalid, vp{p}_bready, '
                f'vp{p}_arvalid, vp{p}_arready, vp{p}_rvalid, vp{p}_rready, vp{p}_rlast; wire [63:0] vp{p}_awaddr, '
                f'vp{p}_araddr, vp{p}_wstrb; wire [7:0] vp{p}_awlen, vp{p}_arlen; wire [511:0] vp{p}_wdata, vp{p}_rdata;')
    decl.append(f"foldsim_vport #(.DW({dw}), .MW({mw}), .TW({tw}), .OW({ow}), .LMAX({q['LMAX']}), .QUEUE(32), "
                f".OUTSTANDING(128), .AXI_RD_OUT(16), .AXI_WR_OUT(16), .MAX_BURST(8), .FS({q.get('FS', 8)}), .KS({q.get('KS', 0)}), "
                f".HZ({q['HZ']}), .NEED_IN({q['NEED_IN']}), .COPY({q['COPY']}), .ZERO_FILL({q['ZERO']}), .ASYNC(0), "
                f".BASE_ALIGNED(1)) vport{p} (.clk, .reset(rst), "
                + ', '.join(f'.{x}(vp{p}_{x})' for x in ('rd_valid', 'rd_addr', 'rd_len', 'rd_tag', 'rd_fence', 'rd_new',
                                                         'rd_copy', 'rd_copy_addr', 'wr_valid', 'wr_addr', 'wr_len',
                                                         'wr_pair', 'wr_data', 'wr_mask', 'rsp_valid', 'rsp_tag',
                                                         'rsp_beat', 'rsp_last', 'rsp_copied', 'rsp_data', 'rd_need',
                                                         'w_count'))
                + f", .rd_after_copies(vp{p}_rd_after), .rd_free(vp{p}_rd_room), .wr_free(vp{p}_wr_room), "
                f".base(64'h{(p + 1) << 32:x}), .bytes(64'd{q['BYTES']}), .zero(!rst), .move(1'b0), .busy(), .stat(), "
                f".m_clk(clk), "
                + ', '.join(f'.m_{x}(vp{p}_{x})' for x in ('awvalid', 'awready', 'awaddr', 'awlen', 'wvalid', 'wready',
                                                           'wdata', 'wstrb', 'wlast', 'bvalid', 'bready', 'arvalid',
                                                           'arready', 'araddr', 'arlen', 'rvalid', 'rready', 'rdata',
                                                           'rlast')) + ');')
    decl.append(f'vport_axi_model axi{p} (.clk, .rst, .cyc, .lo, .hi, .stall(full), '
                + ', '.join(f'.{x}(vp{p}_{x})' for x in ('awvalid', 'awready', 'awaddr', 'awlen', 'wvalid', 'wready',
                                                         'wdata', 'wstrb', 'wlast', 'bvalid', 'bready', 'arvalid',
                                                         'arready', 'araddr', 'arlen', 'rvalid', 'rready', 'rdata',
                                                         'rlast')) + ');')
    for x in ('rd_valid', 'rd_addr', 'rd_len', 'rd_tag', 'rd_fence', 'rd_new', 'rd_copy', 'rd_copy_addr', 'rd_after',
              'rd_need', 'wr_valid', 'wr_addr', 'wr_len', 'wr_pair', 'wr_data', 'wr_mask', 'rsp_valid', 'rsp_tag',
              'rsp_beat', 'rsp_last', 'rsp_copied', 'rsp_data', 'w_count'):
        ports.append(f'.vp{p}_{x}(vp{p}_{x}),')
    ports.append(f'.vp{p}_rd_free(16\'(vp{p}_rd_room)), .vp{p}_wr_free(16\'(vp{p}_wr_room)),')
open(os.path.join(out, 'vm_decl.svh'), 'w').write('\n'.join(decl) + '\n')
open(os.path.join(out, 'vm_ports.svh'), 'w').write('\n'.join(ports) + '\n')
