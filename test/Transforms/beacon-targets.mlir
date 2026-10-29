// RUN: not foldsim-opt %s --pass-pipeline='builtin.module(foldsim-beacon-map{target=agilex})' -o /dev/null 2>&1 | FileCheck %s

// A target the table does not have is an error that lists the targets.
// CHECK: no Beacon target 'agilex'; one of versal virtexuplus virtexuplusHBM zynquplus asap7
hw.module @Top() {
  hw.output
}
