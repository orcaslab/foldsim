#!/bin/bash
# host/build.sh <foldsim bin> <work dir> [vrt|emul|test|golden|all]: the PicoRV32 image's host programs and golden.
#   golden  sw/work.c -> work.bin -> load.hex (mkload.py); golden_tb.sv on the unfolded pico_v80 in Verilator ->
#           golden.txt (what the host checks)
#   test    test_pico_host (pico_host against a mock kernel: no Verilator, no xsim) and its run
#   emul    pico_host_emul: pico_host on the kernel (foldsim_flow.py's top + rtl/kernel + <work>/gen) in Verilator
#           in-process (flow/emul/foldsim_emul.cpp), run against golden.txt
#   vrt     pico_host_vrt: pico_host on the V80 through VRT. Built with the system's VRT (/usr/include/vrt,
#           /lib/x86_64-linux-gnu), never /opt/xilinx/slash or ~/dev/.slash-local (CPATH/LIBRARY_PATH/PKG_CONFIG_PATH
#           unset); `ldd -r` must show only /lib/x86_64-linux-gnu VRT libraries and no undefined symbol. Not run.
# Needs riscv32-unknown-elf-gcc and verilator on PATH; <work>/gen (foldsim --manifest pico_v80.toml) for emul.
set -eo pipefail
B=$(cd "$1" && pwd); W=$(mkdir -p "$2" && cd "$2" && pwd); WHAT=${3:-all}
H=$(cd "$(dirname "$0")" && pwd); E=$(dirname $H); K=$B/rtl/kernel; [ -d $K ] || K=$(cd $B/../../rtl/kernel && pwd)
VL=${VERILATOR:-verilator}
golden() {
  riscv32-unknown-elf-gcc -march=rv32im -mabi=ilp32 -O2 -nostdlib -ffreestanding -fno-builtin -Wno-stringop-overflow \
    -T $E/sw/link.ld $E/sw/crt0.S $E/sw/work.c -o $W/work.elf
  riscv32-unknown-elf-objcopy -O binary $W/work.elf $W/work.bin
  python3 $H/mkload.py $W/work.bin $W/load.hex
  $VL --binary --timing -Wno-fatal -Wno-lint -Wno-style -O2 --x-initial unique --top-module golden_tb -Mdir $W/golden_obj \
    $H/golden_tb.sv $E/upstream/picorv32.v $E/pico_array.sv $E/pico_v80.sv > $W/golden.build.log 2>&1
  $W/golden_obj/Vgolden_tb +verilator+rand+reset+0 +load=$W/load.hex +writes=$(wc -l < $W/load.hex) | grep = > $W/golden.txt
  cat $W/golden.txt
}
test_() {
  g++ -std=c++17 -O1 -Wall -Wextra -Werror -I$K -o $W/test_pico_host $H/test_pico_host.cpp $H/pico_host.cpp
  (cd $W && ./test_pico_host)
}
emul() {
  local G=$W/gen EM=$B/flow/emul; [ -d $EM ] || EM=$(cd $B/flow/emul && pwd)
  local M=$(sed -n 's/^`define FOLDSIM_MEM_MASTERS //p' $G/foldsim_kernel.vh)
  $VL --build --exe --trace -O3 -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN --x-initial 0 --x-assign 0 -j 4 \
    -I$K -I$G --cc $(grep -v '^+incdir' $W/kernel_srcs.f) --top-module foldsim_afu \
    $EM/foldsim_emul.cpp $H/pico_host.cpp -CFLAGS "-std=c++17 -O2 -DPICO_HOST_EMUL -I$EM -I$H -I$K -DFOLDSIM_MEM_MASTERS=$M" \
    --Mdir $W/emul_obj -o $W/pico_host_emul > $W/emul.build.log 2>&1
  (cd $W && ./pico_host_emul --load load.hex --golden golden.txt)
}
vrt() {
  env -u CPATH -u LIBRARY_PATH -u PKG_CONFIG_PATH g++ -std=c++17 -O2 -Wall -Wextra -DPICO_HOST_VRT -I$K -I$H \
    -I$E/host/v80 $(env -u PKG_CONFIG_PATH pkg-config --cflags jsoncpp libxml-2.0 2>/dev/null) \
    -o $W/pico_host_vrt $H/pico_host.cpp -L/lib/x86_64-linux-gnu -lvrt -lvrtdpp
  ldd -r $W/pico_host_vrt > $W/vrt.ldd 2>&1 || true
  grep -E "vrt|slash" $W/vrt.ldd
  ! grep -qE "undefined symbol|/opt/xilinx/slash|\.slash-local" $W/vrt.ldd && echo "vrt: ldd -r clean (system VRT only)"
}
case $WHAT in golden) golden;; test) test_;; emul) emul;; vrt) vrt;; all) golden; test_; emul; vrt;; *) echo "?"; exit 2;; esac
