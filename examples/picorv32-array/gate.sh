#!/bin/bash
# gate.sh <foldsim bin dir> <out dir> [T] [stall:bubble:seed ...]: pico_array (four PicoRV32 tiles in a mailbox ring) folded
# from pico_array.toml alone, gated cycle-exact against the unfolded RTL (upstream/picorv32.v unmodified) in Verilator on
# the workload sw/work.c (sort, checksum, sieve, ring exchange). Needs: ./fetch.sh, riscv32-unknown-elf-gcc and
# verilator on PATH, CIRCT (circt-verilog, circt-opt, firtool) on PATH or CIRCT_BIN.
# 2026-11-05: class pico_tile x4 (and picorv32 x4), II 6, H 6.00 / 6.18 / 8.57 / 8.61, 400,000 cycles EXACT.
set -eu
E=$(cd "$(dirname "$0")" && pwd); B=$1; O=$(mkdir -p "$2" && cd "$2" && pwd); T=${3:-400000}; shift $(( $# < 3 ? $# : 3 ))
RUNS=${*:-0:0:1 30:0:2 0:30:3 30:30:4}
riscv32-unknown-elf-gcc -march=rv32im -mabi=ilp32 -O2 -nostdlib -ffreestanding -fno-builtin -Wno-stringop-overflow \
  -T $E/sw/link.ld $E/sw/crt0.S $E/sw/work.c -o $O/work.elf
riscv32-unknown-elf-objcopy -O binary $O/work.elf $O/work.bin
cd $E && python3 $B/foldsim --manifest pico_array.toml --out $O/gen ${CIRCT_BIN:+--circt-bin $CIRCT_BIN} > $O/gen.log 2>&1
grep -E '^foldsim:|II =' $O/gen.log | head -2
cd $O && python3 $B/foldsim_gate.py tb --sim-json gen/foldsim_sim.json --clock clk --out .
python3 $B/foldsim_gate.py ports --sim-json gen/foldsim_sim.json > ports.txt
python3 $E/stim.py work.bin ports.txt $T stim.hex
VF="--binary --timing -Wno-fatal -Wno-lint -Wno-style --x-initial unique -O2"
verilator $VF --top-module gate_ref -Mdir ref gate_ref.sv $E/upstream/picorv32.v $E/pico_array.sv > ref.build.log 2>&1
verilator $VF --top-module gate_sim -Mdir sim gate_sim.sv gen/foldsim_sim.sv > sim.build.log 2>&1
ref/Vgate_ref +verilator+rand+reset+0 +cycles=$T +stim=stim.hex +trace=ref.trace | grep -E '^REF'
for r in $RUNS; do IFS=: read st bu sd <<< "$r"
  sim/Vgate_sim +verilator+rand+reset+0 +verilator+seed+$sd +cycles=$T +stim=stim.hex +trace=sim.trace +stall=$st +bubble=$bu | grep -E '^(SIM|TIMEOUT)'
  echo "stall $st bubble $bu: $(python3 $B/foldsim_gate.py compare ref.trace sim.trace --sim-json gen/foldsim_sim.json || true)"
done
