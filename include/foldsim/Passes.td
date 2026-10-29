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
    A simulator simulates each class of instances as a model of its own,
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
      whose next, reset and preset values, if any, are that constant. The
      inputs for the other bits are named `<instance>_<port>_<hi>_<lo>_`, or
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
      has read all but at most one of them: a two-entry latency-insensitive
      channel, with fan-out, so a model's outputs reach their inputs on the
      next host cycle.
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
    ListOption<"relayModels", "relay", "std::string",
               "Models whose channels to and from other models get a relay "
               "stage, or pairs <model>:<model> whose channels between them "
               "do: four slots, the token written a host cycle after it is "
               "taken, and each side seeing the other's count a host cycle "
               "late, so every signal between the two sides is registered">,
    Option<"regionLoads", "region-loads", "unsigned", "0",
           "Thread each model with regions of at most this many loads, as "
           "`foldsim-fame5`'s `region-loads`">,
    Option<"registeredState", "registered-state", "bool", "false",
           "Thread each model with its state read a host cycle ahead, as "
           "`foldsim-fame5`'s `registered-state`">,
    Option<"commitSelect", "commit-select", "bool", "false",
           "Commit a single thread's state by select, as `foldsim-fame5`'s "
           "`commit-select`">,
    Option<"domains", "domains", "bool", "false",
           "Emit the simulator as a tree of domains, one module per domain "
           "nested as the class tree: a netlist's domain holds its model, "
           "the channels between it and its children and the domains of "
           "what it instantiates; a class without a netlist is a domain of "
           "its own. A domain is `<sim>_<name>`, its instance `<name>_dom`; "
           "domain boundaries are module ports, kept in synthesis where a "
           "relayed channel crosses them">,
    Option<"stateStages", "state-stages", "unsigned", "1",
           "Thread each model with this many host cycles to its state's "
           "write, as `foldsim-fame5`'s `state-stages`">,
    Option<"stagedRamWrites", "staged-ram-writes", "bool", "false",
           "Thread each model with its registered-read memories written a "
           "host cycle late too, as `foldsim-fame5`'s `staged-ram-writes`">,
    Option<"pipelineMultipliers", "pipeline-multipliers", "bool", "false",
           "Thread each model with its multipliers' state pipelined at their "
           "inputs, as `foldsim-fame5`'s `pipeline-multipliers`">,
    Option<"compactStaging", "compact-staging", "bool", "false",
           "Thread each model with compact staging, as `foldsim-fame5`'s "
           "`compact-staging`">,
    Option<"foldPipelines", "fold-pipelines", "bool", "false",
           "Thread each model with its fixed-latency pipelines folded, as "
           "`foldsim-fame5`'s `fold-pipelines`">,
    Option<"idleSlots", "idle-slots", "bool", "false",
           "Thread each model picked in turn with idle host cycles in its "
           "turns where it has too few threads, as `foldsim-fame5`'s "
           "`idle-slots`">,
    Option<"pipelineDepth", "pipeline-depth", "unsigned", "0",
           "Thread each model with its deep state cones split over two host "
           "cycles, as `foldsim-fame5`'s `pipeline-depth`">,
    Option<"memoryLoads", "memory-loads", "unsigned", "0",
           "Split a token memory of more than this many loads (bits read and "
           "written, times its depth in 32-entry LUTs) by bits into memories "
           "of at most that many, each with write registers of its own: one "
           "register no longer drives the address and enable pins of "
           "thousands of LUTs. Each model's memories and thread state are "
           "split the same way, as `foldsim-fame5`'s `memory-loads`. 0: no "
           "split">,
    Option<"channelDepth", "channel-depth", "unsigned", "2",
           "Slots per producer thread of a channel without a relay, a power of "
           "two: deeper channels give producers slack ahead of their readers "
           "at the cost of slot memory">,
    Option<"nest", "nest", "bool", "false",
           "Fold nested classes recursively, with every model picked in "
           "turn: number each class's threads so that the thread current on "
           "a host cycle is a child of its parent's current thread, a fixed "
           "offset of host cycles earlier (the parent's thread index the low "
           "digits of its children's); the order is recorded as "
           "`foldsim.thread_order`">,
    Option<"nestSkew", "nest-skew", "int", "-1",
           "With `nest`: host cycles from a parent's thread to its children's "
           "(negative: a hop of the channels between them, one host cycle, "
           "two behind a relay), rounded up to a multiple of the children's "
           "idle phases">,
    Option<"nestControl", "nest-control", "std::string", "\"local\"",
           "With `nest`: where a class's turns come from. The control tree "
           "gives each class its parent's position with a digit of its own "
           "above it that advances when the parent's wraps (the root's "
           "counts): `local`, each class counts those positions itself from "
           "the one the tree plans for the host cycle after reset (the same "
           "positions on nested folding's fixed schedule, nothing far feeding "
           "them); `wired`, its parent's controller brings them, registered "
           "once per level; `counters`, a counter of each class's own on a "
           "numbering rotated to keep the order">,
    ListOption<"nestOffsets", "nest-offsets", "std::string",
               "With `nest`: models current this many host cycles after "
               "the nested order, as <model>:<cycles> (a multiple of the "
               "model's idle phases), rather than their parent's offset and "
               "the skew">,
    ListOption<"copies", "copies", "std::string",
               "Models simulated by several threaded copies, as "
               "<model>:<count>: each copy simulates a contiguous range of "
               "the instances (with `nest`, the high digits of the thread "
               "index, so each copy's turn must be a multiple of the "
               "parent's), so a turn takes fewer host cycles">,
    Option<"commitEnable", "commit-enable", "bool", "false",
           "With `registered-state` and `commit-select`, and a class of "
           "several threads: a model of one thread takes every other host "
           "cycle and commits its state with the registered edge as clock "
           "enable, rather than by a select per bit">,
    Option<"blockStateDepth", "block-state-depth", "unsigned", "0",
           "With `registered-state`: the thread state of models of at least "
           "this many threads asks for block RAM (0: never)">,
    Option<"bankedTokens", "banked-tokens", "bool", "false",
           "Keep a producer's slots in a memory also when a reader takes, by "
           "thread, the same bits from different producers (a parent's port "
           "per child instance): the reader reads each producer's memory and "
           "selects the one of its thread, rather than every producer "
           "thread's slots being registers">,
    Option<"stagedTokens", "staged-tokens", "bool", "false",
           "Write each token into its slot a host cycle after it is taken, "
           "from registers of the token, its thread and slot, and forward it "
           "from them to a reader of that slot until then: the slots' write "
           "enables are registers, and a token is still readable on the next "
           "host cycle">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect",
    "circt::sv::SVDialect"
  ];
}

def FoldStatic : Pass<"foldsim-fold-static", "::mlir::ModuleOp"> {
  let summary = "Fold a netlist of models into a statically scheduled simulator";
  let description = [{
    Builds the simulator `sim` of the netlist `module` (as
    `foldsim-extract-models` with `netlist` leaves it) on a schedule fixed at
    compile time. A target cycle is a frame of II host cycles; in each slot of
    the frame each class's copy evaluates one instance in one pass, in an
    order the compiler fixes. Each model must register its outputs: an
    output that depends on the model's inputs in the same cycle is a ready
    (or the root's own path to the environment), and orders evaluations
    within the frame.

    - A copy is the class's logic once, its registers memories of one entry
      per instance read into registers (read registers) at the slot's
      instance, and written straight from the logic at their write port. A
      class of one instance keeps its registers, enabled at its slot.
    - An output that depends only on the model's state (early) is written
      into its link at the producer's state pass with its next value, into
      the bank of the next frame. One that depends on an input, or on a
      memory's read data (late), is written in the frame it belongs to, at
      a pass of the producer placed before its readers: a ready pass that
      writes no state, or the state pass.
    - Every input of a copy is a read register: a link's read at the
      consumer's slot, a constant of the instance, or the netlist input of
      the frame.
    - The netlist's inputs and outputs are token channels
      (`<port>_valid`, `_ready`, `_bits`): a frame starts when every input
      has its token and every output has room.

    Writes the schedule to `report` if given.
  }];
  let options = [
    Option<"moduleName", "module", "std::string", "",
           "The netlist to simulate">,
    Option<"simName", "sim", "std::string", "\"foldsim_sim\"",
           "The name of the simulator module">,
    Option<"regionBits", "region-bits", "unsigned", "512",
           "The most state or link bits a copy of the slot controls drives">,
    Option<"report", "report", "std::string", "",
           "A file to write the schedule to">,
    ListOption<"copies", "copies", "std::string",
               "Models simulated by several copies, as <model>:<count>: each "
               "copy simulates a contiguous range of the instances">,
    Option<"nest", "nest", "bool", "false",
           "Nested folding: each class's instances numbered by their "
           "parents', and the slot tables driven by a sequencer tree along "
           "the class tree, each class's sequencer a register from its "
           "parent's">,
    Option<"scheduleOnly", "schedule-only", "bool", "false",
           "Stop once the schedule is chosen and reported, building no "
           "simulator: a search over the copies compares their frames">,
    Option<"oneCycleLinks", "one-cycle-links", "bool", "true",
           "Forward a link's write to a read of the same entry in the same "
           "host cycle, so a reader's slot may follow its producer's directly">,
    Option<"overlapFrames", "overlap-frames", "bool", "true",
           "Decide the next frame before this one's outputs are pushed, "
           "counting the pushes still due, instead of after them">,
    Option<"interleaveCopies", "interleave-copies", "bool", "false",
           "Copies of a model take every k-th instance instead of a "
           "contiguous range, so a parent's children run on all copies at "
           "once rather than one after another on one">,
    ListOption<"specGroups", "spec-groups", "std::string",
               "Inputs the environment offers for either value of a select, "
               "as <output>=<valid>:<input>:...: each such input's token holds "
               "two values, the high one for <valid> and <output> both set in "
               "the frame before, and the environment's next tokens do not "
               "wait for that frame's <output>">,
    Option<"specDepth", "spec-depth", "unsigned", "1",
           "Frames of a select the environment's offers span: at 2, each "
           "spec-groups input's token holds four values, by the select of "
           "the frame before (low bit) and of the one before that, and the "
           "environment's next tokens wait for neither">,
    ListOption<"envSinks", "env-sinks", "std::string",
               "Outputs of the netlist that none of the environment's tokens "
               "depend on: they close no round trip through it">,
    Option<"shortLead", "short-lead", "bool", "false",
           "Decide a frame three host cycles before its first slot instead "
           "of four: the slot tables read the frame's position as the "
           "decision makes it, not a register of it">,
    Option<"stagedEnv", "staged-env", "bool", "true",
           "The environment's endpoints take a token a host cycle after it "
           "is offered (registered token ends), not in that host cycle">,
    Option<"zeroLatencyChains", "zero-latency-chains", "bool", "false",
           "With zero-latency-links, every late link too, also where its "
           "reader's input reaches an output combinationally: links then "
           "chain across classes within a host cycle, as the native design "
           "evaluates them within a cycle (the logic screen checks the depth "
           "against the native netlist's)">,
    Option<"zeroLatencyLinks", "zero-latency-links", "bool", "false",
           "A late link a reader takes in its producer's own host cycle, "
           "around its read register, where the reader's input reaches no "
           "output combinationally and the logic estimated on both sides "
           "stays shallow">,
    Option<"modulo", "modulo", "bool", "false",
           "Overlap consecutive frames' evaluations: a slot past the frame "
           "runs in the next frame's, so the frame is bounded by each copy's "
           "evaluations and the dependences across frames, not by the "
           "longest chain within one">,
    Option<"slrGroups", "slr-groups", "unsigned", "0",
           "Place the classes (each copy its own) in at most this many SLR "
           "groups, each kept whole in one SLR: the assignment within "
           "slr-capacity with the shortest frame, then the fewest link bits "
           "between groups. A link between groups is taken a host cycle "
           "later, registered at its producer's group and read into its "
           "consumer's, never in its producer's host cycle; each group's "
           "logic, copies and links read into it is a module instance "
           "`foldsim_slr<g>` of the simulator. 0: no groups">,
    ListOption<"slrSites", "slr-sites", "std::string",
               "With slr-groups, each class's LUT sites, as <class>:<sites> "
               "(a class not listed takes none)">,
    Option<"slrCapacity", "slr-capacity", "uint64_t", "0",
           "With slr-groups, the most LUT sites of a group (0: any)">,
    ListOption<"slrAssign", "slr-assign", "std::string",
               "With slr-groups, classes pinned to a group, as "
               "<class>:<group>">,
    Option<"registerStateEntries", "register-state-entries", "unsigned", "2",
           "Per-instance state (registers and memory read buffers) of at most "
           "this many instances is kept in flops, not distributed RAM: a "
           "shallow distributed RAM spends a LUT's whole depth on a few words, "
           "and a two-word read folds into the bypass LUT; 0 keeps every "
           "store in distributed RAM">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect",
    "circt::sv::SVDialect"
  ];
}

def FAME5 : Pass<"foldsim-fame5", "::mlir::ModuleOp"> {
  let summary = "Thread a flat model's state for FAME-5 simulation";
  let description = [{
    Turns a flat `hw.module`, the model of one class of identical instances,
    into its FAME-5 multithreaded model: one module simulating all `threads`
    instances, one host cycle per thread
    in turn, with each target port a latency-insensitive channel.

    - Every register becomes a memory of one entry per thread, read
      combinationally at the current thread and written on the target clock.
    - Each target port `P` becomes a decoupled channel: an input becomes
      `P_sink_{ready,valid,bits}`, an output `P_source_{ready,valid,bits}`. A
      thread finishes its target cycle once every input is valid and every
      output has fired; outputs that do not depend combinationally on an input
      fire eagerly, with the next cycle's value once the thread has finished.
      An input the model does not read has no channel.
    - On `hostReset`, every thread's channel flags are cleared and its
      registers with a preset value take it, the target's state before its
      first edge: with several threads, on the host cycles after it, one
      thread each, from thread 0; no thread takes or offers a token until
      then, so a reset of one host cycle is enough. A register with a preset
      value is written on the host clock, on the target clock's edges, and,
      for one thread, keeps its preset value from the FPGA's configuration.
    - The target clock is `hostClock` through a `ClockGate` the platform
      provides (`hw.module.extern @ClockGate(in %I, in %CE, out O)`),
      enabled when the current thread finishes; or, with `clock-enable`, the
      host clock itself, every target register and memory port enabled then.

    The module must be flat, with one clock input, and registers without
    asynchronous reset. A model of one instance keeps its state in registers;
    its target clock may come in on a channel (`clock-channel`), from a
    platform that gates target clocks itself: then a cycle whose edge is not
    enabled has no tokens, and the first token only says which inputs the
    first cycle takes.
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
    Option<"regionLoads", "region-loads", "unsigned", "0",
           "With `clock-enable` and several threads: split the thread state "
           "and memories into regions of at most this many loads (bits of "
           "state read and written per thread), each with its own registered "
           "copies of the thread index and of the target edge; writes land a "
           "host cycle after the edge. 0: one control for the whole model">,
    Option<"registeredState", "registered-state", "bool", "false",
           "With `clock-enable`: read the thread state a host cycle ahead into "
           "registers, the next thread chosen a host cycle earlier and never "
           "the one after the current, writes landing a cycle late and "
           "forwarded">,
    Option<"commitSelect", "commit-select", "bool", "false",
           "With `clock-enable` and one thread: commit the state by select, "
           "its candidate and enable registered every host cycle, rather than "
           "with the target edge as each register's enable">,
    Option<"stateStages", "state-stages", "unsigned", "1",
           "With `registered-state` and at least three threads: host cycles "
           "from a thread's edge to its state's write (1 or 2); with 2 the "
           "extra stage is for synthesis to retime into the logic that "
           "computes the next state, and the next thread but one is never "
           "one that finishes now">,
    Option<"stagedRamWrites", "staged-ram-writes", "bool", "false",
           "With regions: write a memory with a registered read a host cycle "
           "after the edge too, address, data, mask and enable registered; "
           "its read reads on every host cycle with the target's enable, and "
           "where the memory reads new data under a write it takes the staged "
           "write's data when the edge wrote its address">,
    Option<"pipelineMultipliers", "pipeline-multipliers", "bool", "false",
           "With `registered-state` and `state-stages=2`: a thread state whose "
           "next value is computed by a multiplier takes its first stage at "
           "that value's inputs (ports, registers, memory reads, each "
           "registered once) and computes it on the next host cycle, rather "
           "than registering the product: the stage synthesis cannot retime "
           "into a DSP">,
    Option<"compactStaging", "compact-staging", "bool", "false",
           "With regions: locate each staging register at the state it "
           "stages rather than at the whole model, and give staged write "
           "enables no preset value (the edge's copy is zero until its first "
           "edge); on large models the merged locations of the presets grow "
           "the Verilog tenfold">,
    Option<"foldPipelines", "fold-pipelines", "bool", "false",
           "With `clock-enable` and two threads or more: simulate each "
           "fixed-latency pipeline (see `foldsim-find-pipelines`) by one "
           "unthreaded copy on the host clock, fed by the current thread, "
           "and a per-thread store of its results read at the thread's "
           "count of enabled edges, rather than by threading its registers. "
           "A black box with a pipeline contract is simulated this way, by "
           "an instance of itself">,
    Option<"idleSlots", "idle-slots", "bool", "false",
           "With `registered-state`, picked in turn, and too few threads for "
           "a thread's write to land before its state is read ahead again "
           "(fewer than three, four with two state stages): follow each "
           "thread's host cycle with idle ones, which finish nothing and "
           "take or offer no token, rather than forwarding the write to the "
           "read ahead -- a select per state bit">,
    Option<"pipelineDepth", "pipeline-depth", "unsigned", "0",
           "With `registered-state` and `state-stages=2`: a thread state whose "
           "next value's cone is deeper than this many logic levels takes its "
           "first stage inside the cone, where neither host cycle computes "
           "more than this many levels if it can and the fewest bits are "
           "registered, and computes the rest on the next host cycle rather "
           "than registering the next value; 0 does not">,
    Option<"memoryLoads", "memory-loads", "unsigned", "0",
           "With `region-loads`: split a memory or thread state of more than "
           "this many loads (bits, times its depth in 32-entry LUTs) by bits "
           "into memories of at most that many, in whole mask bits, each in "
           "a region of its own with its own staging and thread index, the "
           "ports no region controls with copies of what drives their "
           "address and enable last: no one register drives the address and "
           "enable pins of thousands of LUTs. 0: no split">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect",
    "circt::sv::SVDialect"
  ];
}

def BlackBoxes : Pass<"foldsim-black-boxes", "::mlir::ModuleOp"> {
  let summary = "Make the modules declared black boxes external";
  let description = [{
    Replaces each module with the `black_box` Verilog attribute, a
    declaration of its ports for a core the synthesis flow provides (its
    contents empty or a stand-in), by an `hw.module.extern` of the same ports
    and attributes: flattening keeps its instances, and `foldsim-fame5` folds
    them by their pipeline contract.
  }];
  let dependentDialects = ["circt::hw::HWDialect"];
}

def Sort : Pass<"foldsim-sort"> {
  let summary = "Sort graph regions topologically, in linear time";
  let description = [{
    Orders the ops of each block of a region without SSA dominance as
    `topological-sort` does -- the same order -- in O((n + e) log n) rather
    than a scan of the block per register loop.
  }];
}

def FindPipelines : Pass<"foldsim-find-pipelines", "::mlir::ModuleOp"> {
  let summary = "Report the fixed-latency pipelines of each model";
  let description = [{
    Finds, in each model of `models` (all modules with registers if none),
    the pipelines `foldsim-fame5` with `fold-pipelines` simulates without
    threading their state: registers, and black boxes with a pipeline
    contract, whose outputs depend only on their inputs of a fixed number of
    enabled cycles before (see `Pipelines.h`). For each, prints its layers,
    the bits of each, of its inputs and of a stored result, the threaded
    state it replaces and the estimated cost of both ways for the model's
    thread count. Leaves the design as it is.
  }];
  let options = [
    ListOption<"models", "models", "std::string",
               "The models, as <module>=<threads>">,
    Option<"stagingRegisters", "staging-registers", "unsigned", "3",
           "Registers per bit of threaded state besides its memory (3 with "
           "registered state and two state stages)">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
}

def LowerEquality : Pass<"foldsim-lower-equality", "::circt::hw::HWModuleOp"> {
  let summary = "Lower equality comparisons to logic";
  let description = [{
    Rewrites `comb.icmp` of equality and inequality as the OR of the bits
    of the operands' XOR, so that LUT mapping packs a comparison with the
    logic around it, as Vivado does. Order comparisons stay: they are carry
    chains, not LUT logic.
  }];
  let dependentDialects = ["circt::comb::CombDialect"];
}

def KeepModules : Pass<"foldsim-keep-modules", "::mlir::ModuleOp"> {
  let summary = "Keep the instances of named modules through inlining";
  let description = [{
    Makes the modules named in `modules` public, so that
    `hw-flatten-modules`, which inlines only private modules, keeps their
    instances. A name also matches its specializations (`VX_elastic_buffer`
    matches `VX_elastic_buffer_12`). Run before `foldsim-screen-map` to read
    the depth at those instances' ports (`foldsim-estimate-timing`'s
    `ports`); LUTs then no longer pack across their boundaries.
  }];
  let options = [
    ListOption<"modules", "modules", "std::string",
               "Modules whose instances to keep">,
  ];
}

def SyncAsyncResets : Pass<"foldsim-sync-async-resets", "::circt::hw::HWModuleOp"> {
  let summary = "Simulate registers with an asynchronous reset cycle by cycle";
  let description = [{
    A simulator steps the target a cycle at a time, with the reset one of the
    cycle's inputs. A register reset asynchronously then holds its reset value
    in every cycle the reset is asserted, from that cycle's start, and its
    value otherwise. It becomes a register reset synchronously, which holds
    the reset value from the cycle after, read through a select on the reset:
    the same value in every cycle. Its next value, which a reset overrides,
    is unchanged.
  }];
  let dependentDialects = ["circt::comb::CombDialect", "circt::seq::SeqDialect"];
}

def SplitMaskedMemories
    : Pass<"foldsim-split-masked-memories", "::circt::hw::HWModuleOp"> {
  let summary = "Split byte-masked memories into 64-bit columns";
  let description = [{
    A block RAM's byte lanes are 9 bits wide, so a memory masked in 8-bit
    bytes and wider than 64 bits packs 9 of its bytes in each 72-bit RAM
    column, the ninth in the parity bits under a write enable of its own,
    and merges the other bytes back on a write through logic around the RAM.
    On the V80, such columns lost the bits of one RAM half on byte-masked
    writes, which no simulation of the netlist shows. Split into memories of
    64 bits and 8 mask bits, each byte is one RAM byte lane and the parity
    bits stay unused: a few more RAMs and no merge logic.
  }];
  let dependentDialects = ["circt::comb::CombDialect", "circt::seq::SeqDialect"];
}

def ReplicateControls
    : Pass<"foldsim-replicate-controls", "::circt::hw::HWModuleOp"> {
  let summary = "Bound the loads of the static fold's slot controls by replicas";
  let description = [{
    `foldsim-fold-static` marks each register it makes as a copy of a slot's
    controls (`foldsim.control`): a region's instance index, staged index and
    state pass, a link group's state pass, instance and bank. Such a register
    drives the address or write enable of every store its region or group
    holds, a few hundred to thousands of LUT pins on one net with no logic
    between: on the V80 the route alone takes most of a 4 ns period.

    Each marked register whose loads exceed `loads` becomes several: replicas
    with the same next value, clock, reset and preset, so they all hold the
    register's value on every host cycle, each driving a group of at most
    `loads` loads, the loads in module order so that a group is a submodule's
    stores. A load is a pin per bit of the register: a memory port's address,
    enable or mask is its width times its depth in 32-entry LUTs (a block
    RAM, deeper than 64 entries: a pin per 36 bits and 1024 entries), a
    select's condition is its width, anything else one. Concatenations,
    extractions, bit casts, replications and inversions of the register are
    wiring, copied with it. A memory of at most 64 entries or a select whose
    one port is over `loads` is first split by bits into parts of at most
    `loads` (a memory in whole mask bits). The replicas and the register get
    `DONT_TOUCH`. The simulator's behavior is the same on every host cycle.

    The marks are removed in any case; `loads` 0 replicates nothing.
  }];
  let options = [
    Option<"loads", "loads", "unsigned", "96",
           "The most loads a marked register or a replica drives; 0: none "
           "replicated">,
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::seq::SeqDialect",
                           "circt::sv::SVDialect"];
}

def FoldZeroWidth : Pass<"foldsim-fold-zero-width", "::circt::hw::HWModuleOp"> {
  let summary = "Replace zero-width combinational values with constants";
  let description = [{
    A zero-width value carries no information, so every combinational op
    whose results are all zero-width integers is replaced by the constant 0
    of that width. Elaborated designs keep such ops (a `comb.mux` of `i0`
    from a parameter that sizes a field to nothing), and lowerings that
    expect a bit to exist do not handle them.
  }];
  let dependentDialects = ["circt::hw::HWDialect"];
}

def LowerConstantDivision
    : Pass<"foldsim-lower-constant-division", "::circt::hw::HWModuleOp"> {
  let summary = "Lower unsigned division by a constant as synthesis does";
  let description = [{
    Rewrites `comb.divu` and `comb.modu` by a constant the way synthesis
    propagates it:

    - by a power of two 2^k: the bits above k and the low k bits,
      zero-extended;
    - of a dividend with at most 10 significant bits (the rest zero, as a
      zero-extended index): a table of the results, selected by the dividend
      with a tree of `comb.mux`, which LUT mapping folds into logic.

    Any other division stays.
  }];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

def EstimateTiming : Pass<"foldsim-estimate-timing", "::mlir::ModuleOp"> {
  let summary = "Rank every register-to-register path by its logic levels";
  let description = [{
    Estimates, for every end point of the design elaborated from `top`, the
    logic levels of its longest path the way Vivado counts them on the
    device `family`: LUTs, CARRY8s, DSP cells and distributed-RAM reads. The input is a LUT-mapped design in which
    `comb.truth_table` ops are LUTs and adders, subtracters, comparators,
    multipliers and dividers are still word-level ops:

    - `comb.truth_table`: one level.
    - `comb.add`/`comb.sub`: a LUT and the CARRY8s from an input bit's
      8-bit block to the output bit's.
    - `comb.icmp` of order: a LUT and the CARRY8s from the input bit to the
      most significant one. Of equality: a LUT6 tree over the bits.
    - `comb.mul`: a table in LUTs if its operands have at most ten variable
      bits; otherwise a cascade of DSPs over their significant bits (27 x 24
      each on the V80, 27 x 18 on the U55C): the cells up to the first
      product, then each further DSP. A register the product feeds, with
      its enable logic, is the DSP's own; any other load adds the DSP's
      output stage.
    - `comb.divu`/`divs`/`modu`/`mods`: a table over the bits the operands
      depend on if there are at most ten of them; otherwise a ripple of one
      subtracter per quotient bit, from every input bit to every output bit.
    - `seq.firmem` read of latency 0 (distributed RAM): one level, plus one
      per doubling of the depth past 64 words, from the address.

    `family` is `versal` (the V80) or `virtexuplusHBM` (the U55C), as
    hw/syn/xilinx/platforms names them; the two differ only in their DSPs.

    `family=asap7` counts cell stages of an ASAP7 netlist mapped by
    `foldsim-screen-map-asic` (3-input cuts) instead: a cell
    (`comb.truth_table`, or an and-inverter gate) is a stage; an adder bit from bit j to
    bit i is a prefix adder, 2 + ceil(log2(i - j + 1)); a compare,
    1 + ceil(log2(span)); a product, its partial products, a Dadda tree of
    full adders and a prefix adder; a memory of flops, a decoder and a
    multiplexer tree. The report converts the deepest path to a period with
    the library's cell, clock-to-output and setup delays and a per-path
    overhead fitted on OpenSTA.

    End points are register inputs (data, reset, clock enable), the address,
    data, enable and mask of memory writes, the address and enable of
    registered memory reads, the inputs of external modules (vendor IPs) and
    the outputs of `top`. Start points are registers, registered memory
    reads, outputs of external modules and inputs of `top`.

    Paths are computed per bit over the elaborated instances, keeping only
    the worst input of each bit, so time and memory grow linearly with the
    design. Writes one row per end point bit of at least `min-levels` levels
    to `output` (levels, retimed levels, end kind, maximum fanout and module
    crossings on the path, then start and end as an instance id and a name),
    the instances to `instances` (id, parent, instance name, module), and
    the `top-paths` deepest paths, op by op, to `report`. Leaves the design
    as it is.

    `paths` lists every deep path once per pair of start and end registers,
    as JSON lines, deepest first: the pair's deepest bit, if it has at least
    `path-levels` levels, with its levels, retimed levels, end kind, start
    and end (instance id and name), the number of end point bits the pair
    has, and `ops`, the path from the end point back to its start as
    [levels, kind, fanout, instance id, name, source]: the source is the
    first file:line of the op's location, which names mapped logic.

    With `retime-levels`, the `retimed` column estimates the depth after
    Vivado's global retiming to a period of that many levels: an end point
    deeper than the period borrows the room of its neighbour stages, its
    register moving backward and its start register forward, neither
    through a distributed RAM or a DSP. Top ports and the pins of
    `black-boxes` count as registers with empty neighbour stages, as a gate
    harness registers them. Without it, `retimed` is the depth.
  }];
  let options = [
    Option<"topName", "top", "std::string", "",
           "The top module (the one no other instantiates if empty)">,
    Option<"output", "output", "std::string", "",
           "End point table (CSV)">,
    Option<"instances", "instances", "std::string", "",
           "Instance table (CSV)">,
    Option<"report", "report", "std::string", "\"-\"",
           "Summary and deepest paths (text)">,
    Option<"minLevels", "min-levels", "unsigned", "0",
           "Only write end points of at least this many levels">,
    Option<"topPaths", "top-paths", "unsigned", "50",
           "Deepest paths to print op by op">,
    ListOption<"trace", "trace", "std::string",
               "Also print the deepest path of each end point whose "
               "instance path and name contain one of these">,
    ListOption<"blackBoxes", "black-boxes", "std::string",
               "Modules to treat as external, as a harness stubs them">,
    Option<"family", "family", "std::string", "\"versal\"",
           "The FPGA family whose logic levels to count">,
    Option<"ports", "ports", "std::string", "",
           "Port depth table (CSV) of the instances of `port-modules`">,
    ListOption<"portModules", "port-modules", "std::string",
               "Modules (and their specializations) whose instances' ports "
               "to report: the levels at each input and behind each output">,
    Option<"paths", "paths", "std::string", "",
           "Deep paths, op by op, one per pair of start and end registers "
           "(JSON lines)">,
    Option<"pathLevels", "path-levels", "unsigned", "12",
           "Only write the paths of pairs whose deepest bit has at least "
           "this many levels">,
    Option<"retimeLevels", "retime-levels", "unsigned", "0",
           "Also estimate each end point's depth after Vivado's retiming "
           "to a period of this many levels (0: off)">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
}

#endif // FOLDSIM_PASSES
