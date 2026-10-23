#!/bin/bash
# Timing screen: map an elaborated design to LUT6 (adders, order comparators
# and multipliers kept word-level), then rank every end point by logic levels.
# usage: screen.sh <design.mlir|mapped.mlirbc> <outdir> [foldsim-estimate-timing options...]
#   A .mlirbc input is taken as already mapped. FOLDSIM_OPT names the
#   foldsim-opt to run (default: the one on the PATH).
# Writes <outdir>/{mapped.mlirbc,endpoints.csv,instances.csv,report.txt,time_*.txt}
set -eu
in=$1; out=$2; shift 2
FOLDSIM_OPT=${FOLDSIM_OPT:-foldsim-opt}
mkdir -p "$out"
cage="systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0"

# foldsim-timing-map is the mapping (see foldsim-opt).
map=${MAP:-'builtin.module(foldsim-timing-map)'}

mapped=$in
if [[ $in != *.mlirbc ]]; then
  mapped=$out/mapped.mlirbc
  $cage /usr/bin/time -v -o "$out/time_map.txt" \
    "$FOLDSIM_OPT" "$in" --pass-pipeline="$map" --emit-bytecode -o "$mapped"
fi

opts="output=$out/endpoints.csv instances=$out/instances.csv report=$out/report.txt"
for o in "$@"; do opts="$opts $o"; done
$cage /usr/bin/time -v -o "$out/time_screen.txt" \
  "$FOLDSIM_OPT" "$mapped" \
  --pass-pipeline="builtin.module(foldsim-estimate-timing{$opts})" -o /dev/null

for t in map screen; do
  [[ -f $out/time_$t.txt ]] || continue
  printf '%-7s' "$t"
  grep -E 'Elapsed|Maximum resident' "$out/time_$t.txt" | sed -E 's/.*: //' | tr '\n' ' '
  echo
done
