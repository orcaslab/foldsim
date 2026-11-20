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
`main` (`313868c23`), which carries the CIRCT changes foldsim needs on top of
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
| `test` | lit tests (`check-foldsim`): the passes' output, and simulators run with Verilator |

## Build

foldsim needs a CIRCT build or installation (its LLVM, MLIR and CIRCT CMake
packages) of orcaslab/circt, and the Beacon submodule; it builds Beacon with
it (`beacon-opt`, `beacon` and `check-beacon` land in the same `bin`):

```bash
git clone --recursive https://github.com/orcaslab/foldsim.git   # or: git submodule update --init
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
  -DCIRCT_DIR=$CIRCT/lib/cmake/circt \
  -DMLIR_DIR=$CIRCT/lib/cmake/mlir \
  -DLLVM_DIR=$CIRCT/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$CIRCT/bin/llvm-lit
ninja -C build
ninja -C build check-foldsim
cmake --install build --prefix $PREFIX
```

The tests under `test/Simulation` build small simulators and run them with
Verilator; CMake looks for `verilator` on the `PATH`, or takes
`-DFOLDSIM_VERILATOR=<path>`. Without it they are reported unsupported.

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
