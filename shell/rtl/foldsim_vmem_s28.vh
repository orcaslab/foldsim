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
// Request word, one a cycle on every hop (619 bits, MSB first):
//   cls 6 | mem 4 | inst 12 | row 16 | beat 3 | last 1 | we 1 | wmask 64 | wdata 512
// `cls` is the class's number in its region, `mem` the memory's in its class.
// A read is one word (beat 0, no data). A write is one word a 512-bit beat,
// `beat` its number, `wmask` its byte enables. `last` closes a local switch
// entry (a host cycle's read and write of one memory): a switch that grants
// an input keeps it until `last`, so an entry's words reach the controller
// together and in order (RDW_READ_FIRST's order for a read-write pair).
//
// Response word, one a cycle (538 bits, MSB first):
//   cls 6 | mem 4 | inst 12 | beat 3 | last 1 | data 512
// A read's beats arrive in order and back to back (`last` on its final one).

`ifndef FOLDSIM_VMEM_S28_VH
`define FOLDSIM_VMEM_S28_VH

`define S28_CLS_W   6
`define S28_MEM_W   4
`define S28_INST_W  12
`define S28_ROW_W   16
`define S28_BEAT_W  3

`define S28_REQ_W   619
`define S28_RSP_W   538

// Request fields.
`define S28_Q_DATA  511:0
`define S28_Q_MASK  575:512
`define S28_Q_WE    576
`define S28_Q_LAST  577
`define S28_Q_BEAT  580:578
`define S28_Q_ROW   596:581
`define S28_Q_INST  608:597
`define S28_Q_MEM   612:609
`define S28_Q_CLS   618:613

// Response fields.
`define S28_R_DATA  511:0
`define S28_R_LAST  512
`define S28_R_BEAT  515:513
`define S28_R_INST  527:516
`define S28_R_MEM   531:528
`define S28_R_CLS   537:532

`endif
