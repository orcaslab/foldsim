// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The words of the hierarchical, region-local virtual-memory path (S28):
// type adapter (foldsim_vmem_adapter) -> local switch (foldsim_vmem_lsw, one
// per class) -> central switch (foldsim_vmem_csw, one per domain root) ->
// region controller (foldsim_vmem_region, one per SLR).
//
// Request word, one a cycle on every hop (591 + INST_W + ROW_W bits: 619 at
// the defaults; MSB first):
//   cls 6 | mem 4 | inst INST_W | row ROW_W | beat 3 | last 1 | we 1 | wmask 64 | wdata 512
// `cls` is the class's number in its region, `mem` the memory's in its class.
// `inst` is the simulator's instance index of the request (a page column's
// {instance, page}); with LANES > 1 it also names the lane (inst mod LANES)
// of the row the region addresses by inst / LANES.
// A read is one word (beat 0, no data). A write is one word a 512-bit beat,
// `beat` its number, `wmask` its byte enables. `last` closes a local switch
// entry (a host cycle's read and write of one memory): a switch that grants
// an input keeps it until `last`, so an entry's words reach the controller
// together and in order (RDW_READ_FIRST's order for a read-write pair).
//
// Response word, one a cycle (526 + INST_W bits: 542 at the defaults; MSB first):
//   cls 6 | mem 4 | inst INST_W | beat 3 | last 1 | data 512
// A read's beats arrive in order and back to back (`last` on its final one).

`ifndef FOLDSIM_VMEM_S28_VH
`define FOLDSIM_VMEM_S28_VH

`define S28_CLS_W   6
`define S28_MEM_W   4
`define S28_BEAT_W  3

// The instance and row fields are sized per design (the compiler's widest
// memory: inst = {instance, page} bits, row = row address bits): every unit
// that carries a word has `parameter INST_W = 16, ROW_W = 12` (the defaults:
// a 619-bit request, a 542-bit response) and these macros name them, so they
// are used only inside such a module (or one with localparams of those names).
`define S28_INST_W  INST_W
`define S28_ROW_W   ROW_W

`define S28_REQ_W   (591 + INST_W + ROW_W)
`define S28_RSP_W   (526 + INST_W)

`define S28_Q_DATA  511:0
`define S28_Q_MASK  575:512
`define S28_Q_WE    576
`define S28_Q_LAST  577
`define S28_Q_BEAT  580:578
`define S28_Q_ROW   (580 + ROW_W):581
`define S28_Q_INST  (580 + ROW_W + INST_W):(581 + ROW_W)
`define S28_Q_MEM   (584 + ROW_W + INST_W):(581 + ROW_W + INST_W)
`define S28_Q_CLS   (590 + ROW_W + INST_W):(585 + ROW_W + INST_W)

`define S28_R_DATA  511:0
`define S28_R_LAST  512
`define S28_R_BEAT  515:513
`define S28_R_INST  (515 + INST_W):516
`define S28_R_MEM   (519 + INST_W):(516 + INST_W)
`define S28_R_CLS   (525 + INST_W):(520 + INST_W)

`endif
