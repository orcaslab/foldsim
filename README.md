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
`firtool`), and foldsim adds the passes that build the simulator.

## Contents

| Path | What |
|---|---|
| `lib/Transforms` | the passes: `foldsim-extract-models`, `foldsim-fuse-models`, `foldsim-fame5`, `foldsim-compose` |
| `tools/foldsim-opt` | runs them, with CIRCT's HW, Comb, Seq and SV passes |
| `tools/foldsim` | the driver: design in, simulator out |
| `test` | lit tests (`check-foldsim`): the passes' output, and simulators run with Verilator |

## Build

foldsim needs a CIRCT build or installation (its LLVM, MLIR and CIRCT CMake
packages):

```bash
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

`design.f` lists `+define+`, `+incdir+` and source files. Each class names a
module whose instances are folded; `A+B` fuses two modules of the same parent
into one model. The driver writes `gen/foldsim_sim.sv` (the simulator: a
channel per target port and a `profile_finish` bit per class),
`gen/foldsim_sim.json` (its ports and each class's thread count) and
`gen/foldsim_sim.vh`.

The host side (token endpoints, memory model, runtime) belongs to the
platform that runs the simulator; Vortex's is in `hw/rtl/afu/foldsim`,
`hw/syn/foldsim` and `sw/runtime/foldsim`.

## License

Apache License v2.0 with LLVM Exceptions (see `LICENSE`).
