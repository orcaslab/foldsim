//===- Passes.td - foldsim passes --------------------------*- tablegen -*-===//
//
// Part of the foldsim project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The passes that build a FAME-5 simulator from a flat design: cut the
// separately simulated models out of their parents, fuse models, thread each
// model's instances into one copy, and compose the copies over a token fabric.
//
//===----------------------------------------------------------------------===//

#ifndef FOLDSIM_PASSES
#define FOLDSIM_PASSES

include "mlir/Pass/PassBase.td"

def ExtractModels : Pass<"foldsim-extract-models", "::mlir::ModuleOp"> {
  let summary = "Cut the instances of separately simulated models out of a "
                "parent";
  let description = [{
    FireSim's Golden Gate simulates each partitioned class as its own model,
    connected to the others by channels. This pass removes the instances of
    the `models` modules from the flat `module`, leaving it with ports for
    what it exchanges with them, named `<instance>_<port>`:

    - An input of an instance becomes an output of the parent, unless inputs
      of the parent drive it, with constants: the instance is then connected
      to those directly. One that only constants drive becomes an output of
      no bits, a channel of tokens only.
    - An output of an instance becomes an input of the parent, but for the
      bits that are always the same constant in the model, which the parent
      uses instead. Those are found through wiring and through register bits
      whose next and reset values are that constant. The inputs for the other
      bits are named `<instance>_<port>_<hi>_<lo>_`, or
      `<instance>_<port>_<bit>_` for one, when they are not all of them.

    Clock ports are left out, as models take their clock from the simulator.

    With `netlist`, the parent's own logic becomes `<module>_model`, and
    `<module>` becomes its netlist: a module of the parent's ports that only
    instantiates `<module>_model` and the models, wired as the parent wired
    them, with the `foldsim.netlist` attribute. Instances of the parent elsewhere
    stay valid, so a netlist can be cut out of its own parent in turn. A model
    output the parent's logic does not read then has no input on it, and a
    model input only constants drive is connected to them.

    With `absorb-logic=N`, an input of a model that the parent drives with
    logic of at most N ops over models' outputs, inputs of the parent and the
    parent's state, the same logic for every instance of the model, is
    computed in the model instead: the model takes the logic's operands as
    inputs of its own. A model's input then comes from the models that
    produce its operands rather than from a hop through the parent's model.
  }];
  let options = [
    Option<"moduleName", "module", "std::string", "",
           "The parent to cut the models out of">,
    Option<"absorbLogic", "absorb-logic", "unsigned", "0",
           "Move logic of up to this many ops driving a model's input into "
           "the model">,
    ListOption<"models", "models", "std::string",
               "The modules simulated as separate models">,
    Option<"netlist", "netlist", "bool", "false",
           "Keep the parent as a netlist of its logic and the models">,
    ListOption<"inputBits", "input-bits", "std::string",
               "The inputs to take for the outputs of the models, as "
               "<instance>_<port>:<hi>:<lo>, to match an existing interface; "
               "the bits of such a port they leave out must be constants">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect"
  ];
}

def FuseModels : Pass<"foldsim-fuse-models", "::mlir::ModuleOp"> {
  let summary = "Simulate pairs of connected models of a netlist as one model";
  let description = [{
    For each group `A+B` of `groups`, the instances of `A` and `B` in the
    netlist `module` (as `foldsim-extract-models` with `netlist` leaves it) are
    paired, each `A` with the one `B` it exchanges bits with, and each pair is
    replaced by an instance of one flat model, `A_B`, of both models' logic.
    An input of one wired only from the other's outputs and constants, alike
    in every pair, is connected inside it; the other inputs and every output
    are its ports, named `<model>_<port>`. Both models' clocks must be the
    same clock.

    A combinational path from one model through the other and back is then a
    path inside one model, which a thread follows in one host cycle, rather
    than a round trip through the channels between two models. The pair is
    one copy of each model's logic, threaded as before.
  }];
  let options = [
    Option<"moduleName", "module", "std::string", "",
           "The netlist the models are instantiated in">,
    ListOption<"groups", "groups", "std::string",
               "The pairs of models to fuse, as <A>+<B>">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect"
  ];
}

def Compose : Pass<"foldsim-compose", "::mlir::ModuleOp"> {
  let summary = "Compose a simulator of threaded models from a netlist";
  let description = [{
    Builds the simulator of the netlist `module` (as `foldsim-extract-models`
    with `netlist` leaves it): the netlists it instantiates are inlined, the
    instances of each model are threaded into one module (`foldsim-fame5` with
    `clock-enable` and `thread-index`), and every channel between them is
    built in one fabric, in the module `sim`:

    - `hostClock` and `hostReset`; for each input `x` of the netlist but its
      clock, the channel `x_valid`, `x_ready`, `x_bits`, and for each output
      `y`, `y_valid`, `y_ready`, `y_bits` (no bits port for no bits).
    - Each output of a model, or input of the netlist, is a producer of
      tokens: every thread of it has two slots, written in turn, and a count
      of tokens written. A token is taken by each input it feeds, which counts
      the tokens it has read from it; the producer is ready while every reader
      has read all but at most one of them. This is the two-entry channel of
      FireSim's LI-BDN simulators, with fan-out, so a model's outputs reach
      their inputs on the next host cycle.
    - An input reads its bits from the producers its netlist wiring gives,
      for each thread; it is valid when all of them have a token for it. A
      producer whose readers are all threaded inputs that read it for every
      thread the same way keeps its slots in a memory, read at the reader's
      thread; any other, in registers.
    - An input of the netlist that is exactly one input of a model of one
      instance, and an output of one that is exactly one of the netlist's
      outputs, are connected directly: the shell around the simulator drives
      and takes them, with valid and ready that depend on its state alone.
  }];
  let options = [
    Option<"moduleName", "module", "std::string", "",
           "The netlist to simulate">,
    Option<"simName", "sim", "std::string", "\"foldsim_sim\"",
           "The name of the simulator module">,
    Option<"readySchedule", "ready-schedule", "bool", "true",
           "Advance each model to a thread whose inputs are all there, "
           "rather than to the next in turn">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
}

def FAME5 : Pass<"foldsim-fame5", "::mlir::ModuleOp"> {
  let summary = "Thread a flat model's state for FireSim's FAME-5 simulation";
  let description = [{
    Turns a flat `hw.module`, the model of one class of identical instances,
    into the FAME-5 multithreaded model FireSim's Golden Gate builds for it:
    one module simulating all `threads` instances, one host cycle per thread
    in turn, with each target port a latency-insensitive channel.

    - Every register becomes a memory of one entry per thread, read
      combinationally at the current thread and written on the target clock.
    - Each target port `P` becomes a decoupled channel: an input becomes
      `P_sink_{ready,valid,bits}`, an output `P_source_{ready,valid,bits}`. A
      thread finishes its target cycle once every input is valid and every
      output has fired; outputs that do not depend combinationally on an input
      fire eagerly, with the next cycle's value once the thread has finished.
      An input the model does not read has no channel.
    - The target clock is `hostClock` through a `FireSimFabricClockGate`,
      enabled when the current thread finishes; or, with `clock-enable`, the
      host clock itself, every target register and memory port enabled then.

    The module must be flat, with one clock input, and registers without
    asynchronous reset. A model of one instance keeps its state in registers;
    its target clock may come in on a channel (`clock-channel`), as from
    FireSim's clock bridge: then a cycle whose edge is not enabled has no
    tokens, and the first token only says which inputs the first cycle takes.
  }];
  let options = [
    Option<"moduleName", "module", "std::string", "",
           "The model to thread">,
    Option<"threads", "threads", "unsigned", "0",
           "The number of instances the model simulates">,
    Option<"threadedName", "threaded-name", "std::string", "",
           "The name of the threaded module (default: the model's name with "
           "`_threaded_<name>`)">,
    ListOption<"portSlices", "port-slices", "std::string",
               "Ports split into channels of some of their bits, as "
               "<port>:<hi>:<lo>; a channel is named <port>_<hi>_<lo>_, or "
               "<port>_<hi>_ for one bit">,
    ListOption<"channelFields", "channel-fields", "std::string",
               "Ports carried together as the fields of one channel, as "
               "<channel>.<field>=<port>, first field first; each field has "
               "its bits port <channel>_{sink,source}_<field>">,
    Option<"clockChannel", "clock-channel", "std::string", "",
           "For a model of one instance: the channel its target clock comes "
           "in on, whose tokens say whether each cycle's edge is enabled">,
    Option<"clockEnable", "clock-enable", "bool", "false",
           "Write the target state on the host clock when the thread "
           "finishes, rather than on a gated target clock">,
    Option<"threadIndex", "thread-index", "bool", "false",
           "Output the current thread on a `threadIdx` port">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
}

#endif // FOLDSIM_PASSES
