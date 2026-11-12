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
decl.append(f'localparam G = {len(groups)};')
decl.append('bit model_ok [G];')
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
open(os.path.join(out, 'vm_decl.svh'), 'w').write('\n'.join(decl) + '\n')
open(os.path.join(out, 'vm_ports.svh'), 'w').write('\n'.join(ports) + '\n')
