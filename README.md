# foldsim

foldsim (FAME Optimizing Library for Device SIMulation) compiles a
SystemVerilog design into a FAME-5 simulator for an FPGA. The units a design
replicates (cores, issue slices, execution units) are *folded*: each class of
identical instances becomes one physical copy that simulates its instances in
turn, one thread per host cycle, and the copies talk over a token fabric. A
GPU that does not fit on an FPGA as RTL runs on one as a simulator,
cycle-exact to the RTL.

foldsim builds on [CIRCT](https://circt.llvm.org), the way POCL builds on
LLVM: CIRCT imports and lowers the design (`circt-verilog`, `circt-opt`,
`firtool`), and foldsim adds the passes that build the simulator. It uses
[Beacon](https://github.com/orcaslab/beacon) (a submodule, `third_party/beacon`)
for logic levels and area before synthesis: Beacon's mapping sizes a fold's
pipeline stages (`stage-levels-mapped`) and prices a class's area
(`--degree-estimator beacon-area`).

**Versions.** CIRCT: [orcaslab/circt](https://github.com/orcaslab/circt)
`main` (`ab733d13d`), which carries the CIRCT changes foldsim needs on top of
llvm/circt. Beacon: the commit of the `third_party/beacon` submodule.

**Related repositories:** [orcaslab/circt](https://github.com/orcaslab/circt),
[orcaslab/beacon](https://github.com/orcaslab/beacon),
[orcaslab/firesim](https://github.com/orcaslab/firesim) (the FireSim
baseline foldsim is compared with, on the V80),
[orcaslab/slash](https://github.com/orcaslab/slash) (the V80 shell and
runtime images run on), and [Vortex](https://github.com/vortexgpgpu/vortex),
the GPU foldsim folds.

## Contents

| Path | What |
|---|---|
| `lib/Transforms` | the passes: `foldsim-extract-models`, `foldsim-fuse-models`, `foldsim-fame5`, `foldsim-compose`, `foldsim-fold-static`, ... |
| `third_party/beacon` | Beacon: `beacon-map`, `beacon-estimate-timing` and `beacon-estimate-area` (logic levels and area before synthesis), its `beacon` driver and `beacon-opt` |
| `tools/foldsim-opt` | runs them, with Beacon's passes and CIRCT's HW, Comb, Seq, SV and Synth passes |
| `tools/foldsim` | the driver: design in, simulator out; `foldsim-lift`, plain-port boundaries for interface-port units; the fold-degree search, priced by `beacon-estimate-area` |
| `shell/rtl` | the FPGA shell around a simulator: the control endpoint (target reset, start, register writes, busy), a memory endpoint per AXI4 port with its DRAM bank timing model, and the token plumbing |
| `shell/include`, `shell/emul` | the shell's register map for a host runtime, and the shell simulated in-process with Verilator |
| `shell/model` | the normative DRAM bank model the memory endpoint implements, in C++ |
| `shell/test` | the memory endpoint against that model, per endpoint configuration (`check-foldsim-shell`) |
| `examples/picorv32-array` | the quick-start demo: four PicoRV32 tiles folded into one copy, gated cycle-exact against the unfolded RTL |
| `test` | lit tests (`check-foldsim`): the passes' output, and simulators run with Verilator |

## Quick start: build and run a demo

FoldSim depends on two other projects:
- **CIRCT:** [orcaslab/circt](https://github.com/orcaslab/circt) `main`. FoldSim
  links its libraries and runs its tools: `circt-verilog`, `circt-opt` and
  `firtool`.
- **Beacon:** a git submodule, built together with FoldSim. Beacon also builds
  on that CIRCT.

Steps 1 to 4 build CIRCT; skip them if you already have a build of it (step 4
of its README). Tested on Ubuntu 24.04 (x86-64) with clang 18, CMake 3.31,
Ninja 1.11, Python 3.12, Verilator 5.050 and RISC-V GCC 12.2.

**1. Prerequisites.**

```sh
sudo apt-get install -y git cmake ninja-build clang lld python3 ccache curl \
  verilator gcc-riscv64-unknown-elf
```

- **Build:** CMake 3.20 or newer, and clang 16+ or GCC 11+. Without `lld`, set
  `LINKER=gold` in step 3.
- **Python:** 3.11 or newer for the `foldsim` driver; it reads TOML manifests
  with `tomllib`. Some options need more packages: `foldsim-lift` needs
  `pyslang`, and the state and SLR planners need `numpy` and `scipy`, plus
  `mtkahypar` for the SLR split. The demo needs none of them.
- **Verilator 5:** used by the `check-foldsim` simulation tests and the demo.
  Ubuntu 24.04 packages 5.020; the tests were run with 5.050. If the packaged
  one fails, build Verilator from source.
- **RISC-V GCC:** the demo's workload is freestanding rv32im code. Ubuntu's
  `riscv64-unknown-elf-gcc` builds it (with `RISCV_PREFIX=riscv64-unknown-elf-`),
  as does a `riscv32-unknown-elf-gcc` toolchain (the default prefix).

**2. Clone CIRCT and its LLVM.**

```sh
mkdir -p ~/orcaslab && cd ~/orcaslab
git clone https://github.com/orcaslab/circt.git
cd circt
git submodule update --init --depth 1 llvm
```

**3. Build LLVM and MLIR.**

```sh
J=$(nproc); LINKER=lld
cmake -G Ninja -S llvm/llvm -B llvm/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=$LINKER \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON -DLLVM_CCACHE_BUILD=ON
ninja -C llvm/build -j$J
```

**4. Build CIRCT** with its SystemVerilog front end (`circt-verilog`).

```sh
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=$LINKER \
  -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$PWD/llvm/build/bin/llvm-lit \
  -DCIRCT_SLANG_FRONTEND_ENABLED=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_CCACHE_BUILD=ON \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF   # slang's C++20 module scan needs clang-scan-deps
ninja -C build -j$J
export CIRCT=$PWD LLVM_BUILD=$PWD/llvm/build
export PATH=$CIRCT/build/bin:$PATH
```

**5. Build FoldSim and Beacon**, and run their tests. `--recursive` fetches
the Beacon submodule; in an existing clone, run `git submodule update --init`.

```sh
cd ~/orcaslab
git clone --recursive https://github.com/orcaslab/foldsim.git
cd foldsim
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=$LINKER \
  -DCIRCT_DIR=$CIRCT/build/lib/cmake/circt \
  -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$LLVM_BUILD/bin/llvm-lit
ninja -C build                     # build/bin: foldsim, foldsim-opt, beacon, beacon-opt, ...
ninja -C build check-foldsim check-foldsim-shell check-beacon
export PATH=$PWD/build/bin:$PATH
```

The simulation tests under `test/Simulation` need `verilator` on the `PATH`,
or `-DFOLDSIM_VERILATOR=<path>` at configure time; without it they are
reported unsupported. To install instead of running from the build tree,
use `cmake --install build --prefix <prefix>`.

**6. Run the demo:**
[`examples/picorv32-array`](examples/picorv32-array). The design is four
[PicoRV32](https://github.com/YosysHQ/picorv32) cores, each with a private
4 KiB RAM, passing words around a mailbox ring. FoldSim folds the repeated
tile, `pico_tile`, into one physical copy that simulates the four tiles in
turn. Everything the compiler is told about the design is in
`pico_array.toml`.

`gate.sh` does the following:
1. Builds the workload (`sw/work.c`: sort, checksum, sieve and ring exchange)
   with RISC-V GCC.
2. Generates the simulator with `foldsim --manifest pico_array.toml`.
3. Builds both the unfolded RTL and the folded simulator with Verilator.
4. Runs both for the same number of target cycles. The folded simulator runs
   once per stall:bubble:seed setting, with random stalls and bubbles on its
   host-side token channels.
5. Compares every target cycle's outputs.

```sh
cd examples/picorv32-array
./fetch.sh                         # PicoRV32 at a pinned commit, into upstream/ (unmodified)
RISCV_PREFIX=riscv64-unknown-elf- ./gate.sh ../../build/bin out 100000
```

Expected output:

```
foldsim-fold-static: II = 6
foldsim: pico_array_model x1, pico_tile x4
stim: 103 program words, 4 tiles, load 668 cycles, T 100000
REF 100000 cycles
SIM 100000 cycles in 600001 host cycles, H 6.00
stall 0 bubble 0: MATCH 100000 cycles
SIM 100000 cycles in 618372 host cycles, H 6.18
stall 30 bubble 0: MATCH 100000 cycles
SIM 100000 cycles in 857027 host cycles, H 8.57
stall 0 bubble 30: MATCH 100000 cycles
SIM 100000 cycles in 861339 host cycles, H 8.61
stall 30 bubble 30: MATCH 100000 cycles
```

How to read it:
- **`II = 6`:** the one `pico_tile` copy takes 6 host cycles per target cycle,
  so the four tiles are simulated in turn.
- **`MATCH 100000 cycles`:** every output of the folded simulator matched the
  unfolded RTL on every target cycle, with and without stalls and bubbles.
- **H:** host cycles per target cycle. It is 6.00 without stalls or bubbles,
  and grows as they are added.

The run takes about 15 seconds. `out/gen/foldsim_sim.sv` is the generated
simulator, and `out/gen.log` is the compiler's log.

## Use

```bash
foldsim --srcs design.f --top Top --out gen --circt-bin $CIRCT/bin \
  Unit A+B Socket
```

`design.f` lists `+define+`, `+incdir+` and source files, and other lists
with `-f` (paths as written) or `-F` (paths relative to that list); `//`
starts a comment. Each class names a module whose instances are folded; `A+B` fuses two modules of the same parent
into one model. The driver writes `gen/foldsim_sim.sv` (the simulator: a
channel per target port and a `profile_finish` bit per class),
`gen/foldsim_sim.json` (its ports and each class's thread count) and
`gen/foldsim_sim.vh`.

### The shell

The shell (`shell/`, installed as `share/foldsim/shell`) is target-agnostic.
A target joins it through a wrapper module with plain ports: a control
interface (register writes, start, busy) and up to 8 AXI4 memory ports, of
single-beat transactions. `rtl/foldsim_afu.sv` puts the generated simulator
inside the shell. It takes the generated `foldsim_sim.vh` and a
`foldsim_kernel.vh` the target's flow writes: `FOLDSIM_MEM_PORTS`,
`FOLDSIM_MEM_CH_SHIFT`, `FOLDSIM_MEM_ID_W`, `FOLDSIM_MEM_REPEAT` and, for
speculative reads, `FOLDSIM_MEM_SPEC`.

A target project keeps only its glue: the wrapper, the classes it folds, its
image flow and its host runtime. Vortex's glue is in
`hw/rtl/afu/foldsim/target`, `hw/syn/foldsim`, `sim/foldsim` and
`sw/runtime/foldsim`.

## License

Apache License v2.0 with LLVM Exceptions (see `LICENSE`).
