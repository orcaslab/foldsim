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
    model input only constants drive is connected to them. With
    `direct-outputs` too, an output of the parent that is only wiring of
    models' outputs, inputs of the parent and constants is connected to them
    in the netlist, as a model input is: the parent's model does not pass it
    on (a port bundle of a child wired straight out of the top stays the
    child's).

    With `absorb-logic=N`, an input of a model that the parent drives with
    logic of at most N ops over models' outputs, inputs of the parent and the
    parent's state, the same logic for every instance of the model, is
    computed in the model instead: the model takes the logic's operands as
    inputs of its own. A model's input then comes from the models that
    produce its operands rather than from a hop through the parent's model.

    For the models `absorb-exclusive` names (globs), an input whose logic is
    over N ops takes the
    part of it that reaches that input alone: a value the parent's other
    logic, state or another input also reads stays in the parent, and the
    model takes it as an input. Logic copied per instance in the parent (each
    port's select in front of a per-port buffer that is a model) is then one
    copy in the model, however wide the shared logic behind it. A value that
    differs among the instances' logic (a constant index into a bus, an input
    one instance has padded with zeros) is then an input too, each instance's
    own.

    With `absorb-registers` too, that logic may pass through registers: a
    register whose value reaches only that input (its own next value aside) is
    moved into the model with the part of its next-value, reset and reset-value
    logic that reaches it alone, at most N ops in all, its clock the model's.
    A per-port pipeline register and the select in front of it are then one
    copy in the model, the register a state of it: the same register, updated
    from the same values every cycle.
  }];
  let options = [
    Option<"moduleName", "module", "std::string", "",
           "The parent to cut the models out of">,
    Option<"absorbLogic", "absorb-logic", "unsigned", "0",
           "Move logic of up to this many ops driving a model's input into "
           "the model">,
    ListOption<"absorbExclusive", "absorb-exclusive", "std::string",
               "The models (globs) whose inputs, over the limit, take the "
               "part of the logic that reaches the input alone">,
    Option<"absorbRegisters", "absorb-registers", "bool", "false",
           "With absorb-exclusive: the logic a model input alone reads may "
           "pass through registers, which move into the model">,
    Option<"absorbRegisterLogic", "absorb-register-logic", "unsigned", "0",
           "With absorb-registers: the most ops a model input's logic through "
           "registers may take (0: absorb-logic's)">,
    ListOption<"models", "models", "std::string",
               "The modules simulated as separate models">,
    Option<"netlist", "netlist", "bool", "false",
           "Keep the parent as a netlist of its logic and the models">,
    Option<"directOutputs", "direct-outputs", "bool", "false",
           "With netlist: an output of the parent wired from models' outputs, "
           "inputs of the parent and constants only (no logic) is wired so in "
           "the netlist, not passed through the parent's model (whose output "
           "is then a constant no one reads)">,
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
      writes no state, or the state pass. With `ready-levels`, a class may
      have several ready passes, each writing the late outputs of one level
      once the inputs they depend on are written.
    - Every input of a copy is a read register: a link's read at the
      consumer's slot, a constant of the instance, or the netlist input of
      the frame.
    - A write reaches a read register loaded in its own host cycle
      (forwarded) or a read in it (taken around the read register) only
      where the schedule puts such a read of its entry there: each control
      is a fixed delay of the frame's position, so the compiler knows which
      reads meet which writes. A forward goes past the read register where
      the reader's logic behind it stays shallow, and bits several classes
      read are written to their links from one register of them, a host
      cycle after the pass: the producer's logic then ends at one register,
      not at every link and read register through the choices.
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
    Option<"bubbles", "bubbles", "bool", "false",
           "An input `advance`: a host cycle it is low puts an idle slot (a "
           "bubble) into the frame where its position is decided; the frame "
           "goes on from where it stopped. No state, link or token moves in "
           "a bubble and the schedule is the same: its distances only grow. "
           "For a slot that waits on data, such as a target memory kept "
           "off-chip">,
    Option<"virtualMemories", "virtual-memories", "std::string", "\"\"",
           "A regex of target memory names (searched in the name and in "
           "<model>:<name>, the class's model, as the storage rules name "
           "it): each such registered-read memory "
           "of a folded class is kept outside the simulator (in HBM, behind "
           "the kernel's foldsim_vmem controller) instead of in a threaded "
           "RAM. A copy issues the memory's reads and writes as requests at "
           "its state pass and keeps each instance's read data from the "
           "responses; a slot whose instance's read has not been answered "
           "waits in a bubble, so the target's cycles never change">,
    Option<"vmemStagger", "vmem-stagger", "unsigned", "0",
           "With vmem-on-engines: the most positions a class virtual memory's "
           "requests may leave after its pass (held in registers, the valids "
           "stepped by the positions run), so that a copy's memories at one "
           "pass share engines at successive positions; bounded by the "
           "class's least gap between passes, the room (a controller's 32 "
           "entries) and the frame's end. Classes whose folded stores stream "
           "take part too (their folded stores keep their controllers). 0: "
           "every request at its pass">,
    Option<"vportEngines", "vport-engines", "bool", "false",
           "With page-engines: each page engine is a static port (the kernel's "
           "foldsim_vport, contract VB/STATUS 5.1): the simulator's vp<p>_* "
           "ports instead of an engine's vm<k>_* row interface and its "
           "controller. A request's address is its row's beat offset in the "
           "port's region ({member, instance, row} x the row's beats), its "
           "tag the engine's; a row of several 512-bit beats is answered beat "
           "by beat into a register of the row. Read after write by a fence "
           "class a member (vport-fence-shift), from the schedule's least "
           "distance from any write of the member to a later read; read-first "
           "members' writes wait for the reads before them (wr_pair)">,
    Option<"vportHz", "vport-hz", "unsigned", "0",
           "With vport-engines: read after write by each port's table of "
           "write numbers by a hash of the row (2^N entries; the fences "
           "unused) instead of fence classes; 0: fence classes">,
    Option<"vportFenceShift", "vport-fence-shift", "unsigned", "4",
           "With vport-engines: the ports' fence sample interval's log2 (FS): "
           "a member whose writes precede its reads by at least (c + 1) x 2^FS "
           "positions reads in fence class c (at most 15); 0 waits for every "
           "earlier write's B">,
    Option<"portPlanOut", "port-plan-out", "std::string", "\"\"",
           "Report only: write every HBM member's request positions (each page "
           "engine member's reads and writes as placed; each class virtual "
           "memory's instances' state passes and folded-store issues, widths "
           "and kinds) to this JSON file, the input of a static-port placement "
           "model; empty: none">,
    Option<"vmemEngines", "vmem-on-engines", "bool", "false",
           "With page-engines: the class virtual memories (the target's RAMs; "
           "not page columns, rings or a class with folded-store streams) ride "
           "the page engines as members at their requests' own positions (each "
           "instance's state pass, a host cycle later for a held read), first "
           "fit on an engine of the same read-during-write order and row bytes "
           "whose positions are free, else a new one: a static port per engine, "
           "no controller and no member queue per memory. A copy's memories "
           "requesting at one position take distinct engines">,
    Option<"vmemCost", "vmem-cost", "bool", "false",
           "A memory the virtual-memory rules pick (pattern or size) stays a "
           "virtual memory only where that costs fewer LUTs than keeping its "
           "threaded image on chip in the copy: a member's select leaves (a "
           "third of a LUT a bit of its row, at the fit's level scale), its "
           "read ring (a row of LUTRAM per 32 entries) and its control, "
           "against the image in LUTRAM (instances x rows <= 64 entries a "
           "LUT6 column, 8 LUTs per 32 x 14 bits) or block RAM; the choices "
           "in the report">,
    Option<"vmemMergeLanes", "vmem-merge-lanes", "bool", "false",
           "With virtual-memories, the arrays a front end split from one "
           "target RAM (byte lanes, ways: the same depth, read-during-write "
           "order, read address and enable, and write address, registers "
           "that always agree counting as the same) are one virtual memory: "
           "a row of whole-byte lanes, each written under its own enable">,
    Option<"vmemAutoBits", "vmem-auto-bits", "uint64_t", "0",
           "Virtual memories by size (the constant shell): every registered-"
           "read target memory of a folded class with one read port and at "
           "most one write port whose threaded image (instances x rows x "
           "width) is at least N bits is a virtual memory, beside those "
           "virtual-memories names, whatever storage rule names it; a memory "
           "that cannot be one (read combinationally, more ports, too short a "
           "state gap) stays on chip and is counted in the report. 0: none by "
           "size">,
    Option<"vmemAutoInstanceBits", "vmem-auto-instance-bits", "uint64_t", "0",
           "As vmem-auto-bits, by the image of one instance (rows x width) "
           "at least N bits, not the threaded image: the same memories are "
           "virtual at every instance count, so the virtual memories (and "
           "their controllers) do not change with N (0: none)">,
    Option<"vmemRings", "vmem-rings", "unsigned", "0",
           "With virtual memories, each one's per-instance read buffer as a "
           "fixed ring (VmemRings.h): a read's answer is written to the "
           "instance's row of an image (a second virtual memory, a row per "
           "instance), read back N positions before the instance's next first "
           "use into entry instance mod R of a ring of R entries, R the "
           "smallest power of two the schedule's window allows. A class whose "
           "schedule leaves no room keeps its buffers (reported). 0: none">,
    Option<"foldStoreRings", "fold-store-rings", "unsigned", "0",
           "With fold-store-stream and without vmem-rings: the streamed folded "
           "stores' read buffers alone as vmem rings of this lead (VmemRings.h): "
           "a ring of R entries and a table of the reads in flight, the same at "
           "every instance count, in place of a read buffer row and a pending "
           "bit per instance; the class's other virtual memories keep theirs. "
           "With vmem-rings the stores take rings already. 0: none">,
    Option<"vmemRingImageEach", "vmem-ring-image-each", "bool", "false",
           "With vmem-rings: each ring its own image (a controller each); by "
           "default a class's rings share one image (a row per ring and "
           "instance, the ring in the requests' tag): one controller a class, "
           "as many at every instance count">,
    Option<"vmemRingEntries", "vmem-ring-entries", "unsigned", "1024",
           "With vmem-rings: the most entries a ring may take; a class that "
           "needs more keeps its buffers">,
    Option<"vmemRingClassControl", "vmem-ring-class-control", "bool", "false",
           "With vmem-rings: the rings of a class whose reads issue together "
           "(the copy's same instance output) share one table of reads in "
           "flight (an entry an issue, a bit per ring), one prefetch queue "
           "and one decision term each, in place of a table, a queue and "
           "terms per ring: a queued prefetch leaves once every one of its "
           "rings has read its instance's image row, each ring's read once "
           "no ring of the class has a read of the instance in flight. "
           "Exact as the rings are (a prefetch may only wait longer)">,
    Option<"vmemRingFlight", "vmem-ring-flight", "unsigned", "32",
           "With vmem-rings: the entries of each ring's table of reads in "
           "flight and of its answer-write queue (a power of two, at least "
           "8; default 32). 0: sized per ring by its read rate, the power of "
           "two covering the decision lead + 2 and the reads of an assumed "
           "256-position answer latency (prefetches a frame x 256 / II), "
           "between 8 and 32: a ring of a sparse class needs few. Never "
           "inexact: a ring whose table is full holds its decisions "
           "(bubbles)">,
    Option<"vmemSkipHeldReads", "vmem-skip-held-reads", "bool", "false",
           "With virtual-memories, a read whose address is provably the "
           "previous target cycle's (the address a function of mux(c, A, q) "
           "alone, q a register without reset whose next value is that mux: "
           "a registered read address with an enable, c low) is not issued "
           "when the row was not written since the instance's last read; the "
           "instance keeps its read data, which is that row's. Exact by "
           "construction; a bit per instance records whether its read data "
           "is current. Such a memory's requests leave a host cycle after the "
           "state pass, from registers (the skip decided there): its room is "
           "12 (10 with short-lead) and it needs a state gap of the lead + 2">,
    Option<"stagedEnv", "staged-env", "bool", "true",
           "The environment's endpoints take a token a host cycle after it "
           "is offered (registered token ends), not in that host cycle">,
    Option<"zeroLatencyChains", "zero-latency-chains", "bool", "false",
           "With zero-latency-links, every late link too, also where its "
           "reader's input reaches an output combinationally: links then "
           "chain across classes within a host cycle, as the native design "
           "evaluates them within a cycle (Beacon checks the depth "
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
               "(a class not listed takes none); with sll-estimate also "
               "<class>:<sites>/<r1>/...: further resources (flops, URAM, "
               "block RAM...) each kept within its SLR's (slr-capacities)">,
    Option<"slrCapacity", "slr-capacity", "uint64_t", "0",
           "With slr-groups, the most LUT sites of a group (0: any)">,
    ListOption<"slrCapacities", "slr-capacities", "std::string",
               "With slr-groups, each group's most LUT sites, one per group: "
               "group g is the region's g-th SLR (the floorplan names it), so "
               "unequal SLRs each take what fits them; replaces slr-capacity; "
               "<sites>/<r1>/... gives sll-estimate the further resources' "
               "capacities (the same number for every group)">,
    Option<"sllEstimate", "sll-estimate", "bool", "false",
           "Report the SLLs the design needs at the least: of the assignments "
           "of the classes to the region's SLRs (slr-capacities, in order, "
           "and slr-sites), the smallest worst boundary of link bits; a class "
           "larger than every SLR is named (its own cut is not known); and "
           "the wires between each pair of classes">,
    Option<"slrSllBudget", "slr-sll-budget", "uint64_t", "0",
           "With slr-groups and slr-capacities, the most link bits an "
           "assignment may put over any one SLR boundary (its SLLs; 0: any)">,
    ListOption<"slrSerialize", "slr-serialize", "std::string",
               "With slr-groups, the links between groups of the given classes "
               "(globs; a copy goes by its class's name) carried over k host "
               "cycles on a k-th of the wires (super long lines), as "
               "<glob>:<k>, or <k> for every class (a later entry wins): the "
               "producer's group holds the bits its pass writes and sends a "
               "k-th of them a host cycle from the held bits (the crossing "
               "register fed by registers alone), the reader's group shifts "
               "them back together and writes the link k + 1 host cycles after "
               "the pass (1 unserialized), so its reader takes it k + 1 host "
               "cycles later. The "
               "class's passes of each kind (its state pass, its ready passes "
               "of a level) are kept k apart; SLL budgets and cuts count a "
               "k-th of its bits. auto or auto:<kmax> (default 8): each class "
               "the largest k <= kmax whose passes, k apart, fit the ungrouped "
               "schedule's modulo lower bound (k x instances <= it)">,
    ListOption<"slrAssign", "slr-assign", "std::string",
               "With slr-groups, classes pinned to a group, as "
               "<class>:<group>">,
    Option<"readyLevels", "ready-levels", "std::string", "\"auto\"",
           "Ready passes split by level: a model's ready outputs written in "
           "as many ready passes as keep the frame acyclic, each pass once "
           "the inputs its outputs depend on are written. `auto` splits them "
           "only where one ready pass per model leaves every choice of ready "
           "passes with a cycle; `off` never; `always` for every choice; "
           "`rank` for every choice, a pass per rank of a model's outputs, "
           "never merged (a test of the split passes)">,
    Option<"registerStateEntries", "register-state-entries", "unsigned", "2",
           "Per-instance state (registers and memory read buffers) of at most "
           "this many instances is kept in flops, not distributed RAM: a "
           "shallow distributed RAM spends a LUT's whole depth on a few words, "
           "and a two-word read folds into the bypass LUT; 0 keeps every "
           "store in distributed RAM">,
    ListOption<"storage", "storage", "std::string",
               "The storage of folded state stores, as <model>:<store>=<kind>, "
               "<kind> a Vivado ram_style (registers, distributed, block, "
               "ultra). <store> is a memory of the model, by name, or `regs` "
               "or `regs>=<width>`: the per-instance stores of its registers, "
               "those at least <width> bits wide (the largest such bound "
               "applies). A copy of <model> takes its rules. A register store "
               "in block or ultra RAM is read a host cycle early, registered, "
               "which needs a state gap of 4. Without rules the stores keep "
               "the default rules">,
    Option<"forwardAfterLevels", "forward-after-levels", "unsigned", "8",
           "A link read the schedule forwards a write to in its host cycle "
           "is forwarded past the reader's read register, from a register of "
           "the written bits, where the reader's input reaches no output and "
           "its logic behind the register stays below this many levels with "
           "the choice: the producer's path then ends at that register, not "
           "through the choice into the read register; 0 forwards every one "
           "into the read register">,
    ListOption<"page", "page", "std::string",
               "Paged register state, as <model>:<p> (or *:<p> for every "
               "model): a copy of the model of several instances keeps its "
               "registers in one page RAM per region (block RAM, or the "
               "<model>:pages=<kind> storage rule) and moves an instance's "
               "state in and out p host cycles at a time: its pages are read "
               "into a fetch buffer over the p slots before its state pass, "
               "and its next state written back from a write-back buffer over "
               "the p host cycles after. The evaluation stays one host cycle; "
               "the copy's state passes are p slots apart and the frame at "
               "least p + 4. Registers a ready pass reads keep their stores">,
    Option<"pageHbmColumnBits", "page-hbm-column-bits", "unsigned", "4096",
           "With a model's pages off chip (storage <model>:pages=hbm): the "
           "most bits of a page word in one virtual memory's row (a multiple "
           "of 8); each such column is a virtual memory of its own, served by "
           "a controller of its own">,
    Option<"pageEngines", "page-engines", "unsigned", "0",
           "With a model's pages off chip: its page columns carried by "
           "shared engines (HBM page streams), at most this many, instead of "
           "a controller per column. An engine is one virtual memory of one "
           "image (INSTANCES 1) whose rows are {member, instance, page}; its "
           "requests are tagged by member and instance (foldsim_vmem TAGW), "
           "and the columns it carries never request at the same position "
           "(the frame's static order, checked): no member FIFO, no "
           "arbitration. Columns of one copy move together, so a copy's "
           "columns need engines of their own; copies whose page traffic "
           "falls in other positions share them. 0: a controller per column">,
    Option<"pageHbmWindow", "page-hbm-window", "unsigned", "0",
           "With a model's pages off chip: an on-chip window of at most this "
           "many instances' pages per copy, filled from the virtual memories "
           "that many instances ahead of their state passes (each instance's "
           "pages prefetched at the copy's state pass that many before), so "
           "the round trip hides behind them; the fetch buffer loads from it "
           "as from an on-chip page RAM. The compiler takes the most slots "
           "the schedule allows (a divisor of the copy's instances). 0: no "
           "window, each fetch from the copy's state pass before">,
    Option<"pageMemoryBits", "page-memory-bits", "unsigned", "4096",
           "With page: a memory read combinationally (a queue, a register "
           "file, a predictor table) of at most this many bits, which no "
           "ready pass reads, pages with its instance's registers: each row "
           "a register, its reads a choice of the rows from the fetch buffer, "
           "its writes the rows' next values (0: registers only)">,
    Option<"pageMemories", "page-memories", "std::string", "\"\"",
           "With page-memory-bits: a regex of the memories that page as rows "
           "(empty: every one that qualifies)">,
    Option<"pageRows", "page-rows", "std::string", "\"\"",
           "With page: a regex of <model>:<memory> names that page as rows "
           "whatever their size (a cost model's choice, foldsim --state "
           "auto), besides those page-memory-bits and page-memories take; "
           "the other conditions of a row memory hold">,
    Option<"pageMasks", "page-masks", "bool", "true",
           "With page: a paged register written under an enable keeps it, "
           "registered at the state pass, as its bytes' write mask in the "
           "page RAM (block RAM's 9-bit bytes, ultra RAM's 8-bit), its next "
           "value written straight into its write-back register; registers "
           "of one enable share bytes, each enable's padded to whole bytes. "
           "Off, the page is dense and such a register writes its old value "
           "where its enable is low, through a mux after the target logic">,
    Option<"shareLinkWrites", "share-link-writes", "bool", "true",
           "A producer's bits read by several classes' inputs are registered "
           "once and written to each reader's link a host cycle after the "
           "pass, forwarded from the register to a read in between: the "
           "producer's path ends at that one register, not at every reader's "
           "link">,
    Option<"zeroChainHops", "zero-chain-hops", "unsigned", "0",
           "With zero-latency-chains: the most links taken in their "
           "producers' host cycle a combinational path may cross, instance "
           "by instance (each adds its reader's choice of source to the "
           "path); a link that would pass them is taken a host cycle later "
           "from its register, where the chain starts anew. 0: no bound">,
    Option<"compactLinks", "compact-links", "bool", "false",
           "A link whose reader takes fewer of its producer's instances than "
           "the producer has (a per-lane port read by its own lane's "
           "instances, say) keeps only those: an entry per instance read, "
           "written at the producer's instance's entry by a table of the "
           "slot's instance, the other instances' writes masked. The schedule "
           "report gives every link's storage, compact or not">,
    Option<"linkBlockDepth", "link-block-depth", "unsigned", "0",
           "A link memory at least this deep (entries x banks) that is the "
           "only source of its reader's bits and never forwarded or taken in "
           "its producer's host cycle is a block RAM, read at the same "
           "registered address a host cycle on into its output register, "
           "which is the read register (0: none)">,
    Option<"linkUltraDepth", "link-ultra-depth", "unsigned", "0",
           "As link-block-depth, in ultra RAM (0: none)">,
    Option<"linkKindCost", "link-kind-cost", "bool", "false",
           "Each link memory that may be block or ultra RAM (read through its "
           "output register) takes the kind (block RAM, ultra RAM, distributed "
           "RAM) by primitive cost against the remaining device budget per "
           "kind, instead of the link-block-depth / link-ultra-depth "
           "thresholds: largest first, the kind that leaves the highest "
           "utilization of the three (URAM288, RAMB36, LUTs) lowest. link-kinds "
           "rules still win. Reported as `link-kind-cost:`">,
    Option<"linkKindDevice", "link-kind-device", "std::string",
           "\"1029,2052,1307904\"",
           "With link-kind-cost: the device's URAM288, RAMB36 and LUTs at the "
           "fill target, \"<uram>,<ramb36>,<luts>\" (default: the V80 "
           "region's at 0.85)">,
    Option<"linkKindGroups", "link-kind-groups", "std::string", "\"\"",
           "With link-kind-cost and SLR groups: each group's budget, "
           "\"<uram>,<ramb36>,<luts>;...\" (group g the region's g-th SLR); a "
           "link's memory is in its reader's group and priced against it. "
           "Empty: the device's (link-kind-device)">,
    Option<"linkKindGroupOthers", "link-kind-group-others", "std::string", "\"\"",
           "With link-kind-groups: what the rest takes of each group's, "
           "\"<uram>,<ramb36>,<luts>;...\" (missing groups: none)">,
    Option<"slrKept", "slr-kept", "unsigned", "8",
           "With slr-groups: the assignments with the fewest bits between "
           "groups kept and each scheduled (a line each), the shortest frame "
           "chosen; the first whose frame is the modulo search's lower bound "
           "is chosen at once (none can beat it). Each is a whole schedule: at "
           "32 SMs tens of seconds">,
    Option<"slrLinkRams", "slr-link-rams", "bool", "false",
           "With SLR groups: links may be block or ultra RAM too (by "
           "link-block-depth / link-ultra-depth or link-kind-cost), the memory "
           "in its reader's group, as without groups; a link read a host cycle "
           "ahead between groups, whose write lands a host cycle later, only "
           "where no write lands in the three host cycles before its read "
           "(false: every link with groups distributed RAM or registers)">,
    Option<"linkKindOthers", "link-kind-others", "std::string", "\"0,0,0\"",
           "With link-kind-cost: what the rest of the simulator takes of each "
           "(\"<uram>,<ramb36>,<luts>\": URAM288, RAMB36, logic and other "
           "LUTRAM LUTs), as a first fold built it "
           "(foldsim_link_kinds.py); 0,0,0: links alone">,
    Option<"linkHbmLead", "link-hbm-lead", "unsigned", "48",
           "HBM link streams (link-kinds kind hbm): the most positions a "
           "row's request precedes its landing (the memory's round trip in "
           "host cycles and a margin; less where a stream's writes come "
           "later)">,
    Option<"pageEnginesPP", "page-engines-pp", "bool", "false",
           "With page-engines and state-in-ram-pp's streamed rows: a PP class's "
           "row chunks ride the page engines (members at their own read and "
           "write positions) instead of a controller each">,
    Option<"ppBeatBits", "pp-beat-bits", "unsigned", "0",
           "With page-engines-pp: a streamed PP class's rows move as beats of "
           "at most this many bits (whole 72-bit columns, a power of two "
           "beats a row): the prefetch reads beat by beat, each pass's row "
           "into a buffer of the class's rows, drained a beat a position "
           "after its last row's pass; the class's reads and drains move by "
           "a constant each to share an engine. One member a class (no "
           "chunks of 56 columns). 0: whole rows">,
    Option<"ppChunkColumns", "pp-chunk-columns", "unsigned", "56",
           "A streamed PP class's row chunks (a virtual memory each): at most "
           "this many 72-bit columns a chunk (56: 4,032 bits, a row of 8 "
           "beats; 7: 504 bits, a beat, so that every chunk is one page-engine "
           "lane and a row's chunks stripe over engines at the same positions)">,
    Option<"linkHbmBeatBits", "link-hbm-beat-bits", "unsigned", "0",
           "HBM link streams on their own controllers: a row wider than this "
           "many bits moves as beats (at most 8) of a controller this wide, "
           "requested at the positions before its landing's request and "
           "written at those after its page's write, the landing slot written "
           "beat by beat (no pass spaced: only where the landings and the "
           "producer's passes leave the positions, else whole rows). A "
           "controller costs its width (0: whole rows)">,
    Option<"linkLandingBlock", "link-landing-block", "unsigned", "0",
           "An HBM link stream's (or mailbox's) landing memory in block RAM, "
           "its read registered and addressed a host cycle ahead (the pop "
           "pointer's next value), where its distributed RAM would cost at "
           "least this many LUTs a RAMB36 it takes (the count's prices: a "
           "masked one 2 LUTs a bit a 64 entries); 0: distributed RAM">,
    Option<"linkHbmBeatRows", "link-hbm-beat-rows", "bool", "false",
           "With link-hbm-beat-bits: a stream of beats lands in rows of a "
           "beat ({slot, beat}: its landing memory as deep as slots x beats "
           "and a beat wide, not a row wide), read beat by beat into the "
           "members' registers from beats - 1 positions before the landing's "
           "own position">,
    Option<"linkHbmRowBits", "link-hbm-row-bits", "unsigned", "512",
           "HBM link streams: the most bits of a stream's row (its "
           "controller's data width, 8 to 4096): a wider word is cut into "
           "that many rows, written one a host cycle after the producer's "
           "pass and landing one a host cycle before the reader's, which "
           "spaces their passes">,
    Option<"linkHbmMinLead", "link-hbm-min-lead", "unsigned", "0",
           "HBM link streams: the least positions from a row's write to its "
           "landing for a link to be streamed; one with less stays an on-chip "
           "link (0: the decision lead + 6). Set to the memory's round trip "
           "in host cycles, no landing waits for it">,
    Option<"linkHbmShift", "link-hbm-shift", "unsigned", "0",
           "HBM link streams on page engines (page-engines): the most "
           "positions a stream's request may move earlier than `lead` before "
           "its landing, and a page write later than its own after the "
           "producer's pass, so that streams whose requests or writes fall "
           "at the same positions share an engine. Within the room from the "
           "row's write to its request (both moves together), requests in "
           "their landings' order, writes in order and before the producer's "
           "next pass; a landing slot for each position more (0: none)">,
    Option<"foldPorts", "fold-ports", "std::string", "\"\"",
           "Netlist ports as bundles folded into two record streams: a "
           "regex whose first capture is the bundle (an index 0..N-1) and "
           "whose second the field (POSIX extended, e.g. m_axi_mem_([0-9]+)_(.*)), "
           "or, without a group, a prefix: <prefix><bundle>_<field>. Each "
           "bundle's outputs of a target cycle leave as one request record "
           "(req_valid / req_ready / req_chan / req_treset / req_bits, the "
           "fields low first in netlist order), its inputs come as one "
           "response record (rsp_valid / rsp_ready / rsp_chan / rsp_bits); "
           "the outputs' token queues are memories indexed by bundle, "
           "written one bundle a host cycle (the pushes spread over the "
           "frame). Empty: none">,
    Option<"foldPortsReset", "fold-ports-reset", "std::string", "\"reset\"",
           "With fold-ports: the netlist input whose frame token rides on "
           "each request record (req_treset)">,
    Option<"foldPortsRows", "fold-ports-rows", "bool", "false",
           "With fold-ports: the bundles' response tokens are rows of one "
           "memory indexed by bundle and frame (four frames), written as the "
           "records come and read where the frame reads them: a class input "
           "whose instances read one field of different bundles is one read "
           "at the instance's bundle. A bundle's token held is one bit (no "
           "token, frame and previous-frame registers per field), the frame's "
           "inputs all in is one counter of bundles. Needs II >= 4 and no "
           "spec group over a bundle's inputs">,
    Option<"readBufferBits", "read-buffer-bits", "unsigned", "0",
           "A threaded memory read through a register keeps each instance's "
           "read register in a buffer of an entry per instance, read at the "
           "next slot's instance and registered. With N > 0, such a buffer of "
           "at least N bits is a RAM read synchronously instead (read first, "
           "address the next slot's instance, its output the registered "
           "value: block RAM rather than LUTs), the read register's bypass of "
           "data just out of the memory registered beside it. The same value "
           "every host cycle. 0: every buffer read combinationally">,
    Option<"foldStoreRam", "fold-store-ram", "uint64_t", "32768",
           "Folded pipelines' result stores of at least N bits (an entry per "
           "slot and instance) read through a register, in block or ultra "
           "RAM: each read a host cycle or two ahead (fold-store-lead), at the "
           "instance then due and its slot (the slot counter physical, an "
           "entry per instance, beside the store), into the RAM's output "
           "register, as far ahead as the schedule shows every write to the "
           "entry landed. Exact. 0: every store read combinationally (LUTs)">,
    Option<"foldStageLevels", "fold-stage-levels", "unsigned", "0",
           "Folded pipelines' logic in stages: each layer of a folded "
           "pipeline's copy (the issue's logic to the first layer, a layer's "
           "to the next, the last's to the store's write data) cut by "
           "registers on the host clock so that no stage spans more than N "
           "estimated levels (as stage-levels counts them); every layer then "
           "takes the deepest layer's stages and the results land that much "
           "later in the store. Checked once the schedule is known: each "
           "result still lands before its first read (the store's depth of "
           "edges II apart and the state gap), else an error. Not for a "
           "pipeline with black boxes. 0: none">,
    ListOption<"foldStageLevelsClass", "fold-stage-levels-class", "std::string",
               "Per class over fold-stage-levels: <glob>:<N> stages that "
               "class's folded pipelines at N levels, <glob>:off not at all "
               "(globs on the model's name; a copy goes by its class's; the "
               "last matching spec wins; a spec of no class is an error)">,
    Option<"foldStoreLead", "fold-store-lead", "unsigned", "2",
           "With fold-store-ram: host cycles a store is read ahead, 2 (the "
           "RAM's register and an output register) or 1 (the RAM's register "
           "alone); less where the state gap or a write landing in between "
           "leaves less room, down to a combinational read">,
    Option<"foldStoreHbm", "fold-store-hbm", "uint64_t", "0",
           "Constant-shell fold: folded pipelines' result stores of at least "
           "N bits (an entry per slot and instance; a depth of 2 or more) "
           "kept in HBM as virtual memories (virtual-memories' controllers): "
           "at an instance's state pass the copy requests the row its next "
           "pass reads (the slot after this pass's issue: a frame ahead) and "
           "writes the row its last issue's results landed for (from a "
           "landing buffer of an entry per instance, written as the results "
           "come back); its next pass waits in a bubble only if the row is "
           "late. A storage rule <model>:fold<k>_store=hbm puts one store in "
           "HBM, another kind keeps it on chip. A store by size goes to HBM "
           "only past the state scratch (state-scratch-bits), which the "
           "stores take first, in class order. 0: none by size">,
    Option<"foldStoreStream", "fold-store-stream", "bool", "false",
           "Constant-shell fold: every folded pipeline's result store of a "
           "depth of 2 or more streams from HBM, whatever its size and the "
           "state scratch (a store row saves 2/depth bits of traffic a frame "
           "per resident bit, a state row 2: the scratch is the rows'). Its "
           "results go out from a landing ring of a fixed number of entries "
           "(L, by the schedule, not the instance count): each issue's "
           "{instance, row} and results wait there L state passes, then that "
           "pass writes the row and reads the row the instance's next pass "
           "reads, so that an instance's read and write still leave together; "
           "the read's answers go to the class's vmem ring (vmem-rings) where "
           "it has one. A class whose schedule leaves no such L keeps a "
           "landing entry per instance (reported). Storage rules naming a "
           "store (<model>:fold<k>_store=<kind>) still apply">,
    Option<"foldStoreStyle", "fold-store-style", "std::string", "\"auto\"",
           "With fold-store-ram: the stores' RAM style, ultra, block, or auto "
           "(ultra where it takes fewer than half as many primitives as "
           "block RAM at its best aspect, else block); a storage rule for "
           "the store by name (<model>:fold<k>_store=<kind>) takes "
           "precedence">,
    Option<"pageWriteGate", "page-write-gate", "bool", "false",
           "Off-chip pages on page engines (page-engines): each page's "
           "write-back bits through a register of its own, loaded the host "
           "cycle before the page's write and held at zero (its synchronous "
           "reset) otherwise, so that a column's write data is the OR of its "
           "pages (a six-input OR a LUT: about (p - 1) / 5 LUTs a page bit) "
           "instead of a p:1 select by the page index (about (p - 1) / 3), "
           "at a register a paged bit; page 0's register loads what the "
           "write-back register loads at the state pass (a 2:1 select a bit "
           "of one page). A class with a page window keeps the select (the "
           "report's `page write gate` line names the classes gated)">,
    Option<"pageWriteGatePages", "page-write-gate-pages", "std::string", "\"\"",
           "With page-write-gate: per class, how many pages are gated "
           "(<model>:<n>, comma separated; the first n pages of each column, "
           "the others chosen by a select of n fewer inputs, ORed in): a "
           "balance between the gated registers (a flop a paged bit of a "
           "gated page) and the select's LUTs. Unlisted: every page">,
    Option<"linkHbmFitSlots", "link-hbm-fit-slots", "bool", "false",
           "HBM link streams on page engines: each takes the fewest landing "
           "slots (a power of two, at least 8) its requests need, each request "
           "at least 8 positions after the landing of its slot's last use (as "
           "lead + 8 slots give at the lead); a stream that shares an engine "
           "only with more takes lead + 8. The report lists them (link stream "
           "slots)">,
    Option<"minII", "min-ii", "unsigned", "0",
           "With modulo: the shortest frame (host cycles) the search starts "
           "from; positions past the schedule's own stay idle (a longer "
           "prefetch distance for every request a pass issues)">,
    Option<"readyChoice", "ready-choice", "std::string", "\"\"",
           "The choice of ready passes, given instead of searched: the "
           "classes (as the report names them) that get a ready pass, comma "
           "separated, after `split:` when the classes' ready outputs are "
           "split by level (the report's `(N ready passes)`, N > 1). The "
           "choice is placed once (with modulo, the period search from the "
           "lower bound as for any choice); it must place. For a schedule a "
           "search found before (the report's `(ready pass)` / `(N ready "
           "passes)` lines): the same frame without the search. Empty: search">,
    Option<"moduloLinear", "modulo-linear", "unsigned", "16",
           "With modulo: the periods tried one at a time from the frame's "
           "lower bound before the search skips ahead (a doubling step until "
           "one schedules, then halving the interval to the last that did "
           "not); the frame found may then be a few periods above the least">,
    Option<"linkKinds", "link-kinds", "std::string", "\"\"",
           "Per pair of models, the RAM of the links between them that "
           "link-block-depth's rules let block or ultra RAM hold, whatever "
           "their depth: <producer model>:<reader model>:distributed|block|"
           "ultra, comma separated (the fold-degree model's link tier, "
           "foldsim --state auto)">,
    Option<"stagedLinks", "staged-links", "bool", "false",
           "An early link of one producer instance (its reader takes one) "
           "kept in two registers where the schedule allows: written at the "
           "producer's pass, copied at a fixed position of the frame to the "
           "one the reader reads, between the last read of the frame before "
           "and the first of this one; no memory and no bank select">,
    Option<"stagedLinkMemories", "staged-link-memories", "bool", "false",
           "With staged-links: the staged links of one producer port (its "
           "bits) to one reader class, two or more, keep their first "
           "registers as one memory of an entry per producer instance they "
           "take (distributed RAM), written at the producer's passes; each "
           "link's second register is loaded from it at its own copy "
           "position (distinct positions in the frame: one read port)">,
    Option<"stagedLinkDirect", "staged-link-direct", "bool", "true",
           "With staged-links: a reader of one instance reads a staged link's "
           "second register in place, without a read register of its own "
           "(the copy then a host cycle later in the frame: after the last "
           "read of the frame before, by the first of this one), so the "
           "link's bits are held in two flops (or a memory row and a flop) "
           "instead of three">,
    Option<"maxTableBits", "max-table-bits", "uint64_t", "1000000",
           "The largest slot / frame table constant in bits: a larger table "
           "is cut into column slices, each its own constant read at the same "
           "index (synthesis bounds a variable's size: Vivado's Synth 8-4556 "
           "at 1,000,000 bits; a 128-SM frame of 2^17 positions met it)">,
    Option<"affineSlots", "affine-slots", "bool", "false",
           "With modulo: each copy's (or ready lane's) passes of a kind are "
           "placed as one affine run over its instances, the k-th at o + k*P "
           "(P a power of two per copy, at least its kinds of passes and its "
           "page, link page or serialized spacing), only the offsets chosen: "
           "a class's slot tables are then a segment a pass (slot-segments), "
           "whatever its instances, and the frame grows instead">,
    Option<"affineAuto", "affine-auto", "bool", "false",
           "With modulo and slot-segments: schedule, then place affine "
           "(affine-slots) only the classes whose own slot rows (each pass "
           "kind's positions, the instance at each position, the PP passes') "
           "need more than slot-segments segments, the others as single "
           "evaluations, and schedule again: their slot tables then decode, "
           "at a smaller cost in the frame than affine-slots">,
    Option<"affineAutoBits", "affine-auto-bits", "uint64_t", "262144",
           "With affine-auto: only a class whose undecoded slot rows hold at "
           "least this many table bits is made affine (the frame's cost paid "
           "where the tables are large)">,
    Option<"affineAutoSegments", "affine-auto-segments", "unsigned", "0",
           "With affine-auto: a class is made affine where its own slot rows "
           "need more than this many segments (0: slot-segments), so a "
           "decoder's segments stay as few at any count of instances">,
    Option<"affineStride", "affine-stride", "std::string", "\"pow2\"",
           "With affine-slots / affine-auto: a group's stride, the most of its "
           "resource's group count and its own least spacing, rounded up to a "
           "power of two (pow2) or kept as it is (exact: a resource's kinds of "
           "passes interleave without idle positions; the slot tables' runs "
           "of a stride not a power of two decode with the position's "
           "quotient and remainder counters, slot-segments; affine-auto then "
           "counts a class's rows decoded by runs of any stride)">,
    Option<"slotColumnAffine", "slot-column-affine", "bool", "false",
           "With slot-segments: a table indexed by a value the position "
           "counters do not know (an instance) that the segments do not "
           "decode is cut into column-affine pieces: on a range, with q and "
           "r the offset's quotient and remainder by a period (a constant "
           "division), a[r] + s[r] * q for a few slopes s (an instance table "
           "of a class nested a fixed count of children a parent: a switch "
           "network's ports), as large at any count of parents">,
    Option<"slotTableCompose", "slot-table-compose", "bool", "false",
           "With slot-segments: a table read at the value of a position table "
           "(an instance decoded from the position) that does not decode is "
           "read through the inner table's position instead (its rows "
           "composed), where the position's decoders and counters take it">,
    Option<"slotSegments", "slot-segments", "unsigned", "0",
           "Slot tables (constants read by the frame's position or an "
           "instance) of at most this many affine segments of nonzero rows "
           "(index s + j*ds holding v + j*dv, ds and dv powers of two) are "
           "decoders of the index, a range and residue compare and a "
           "shift-add a segment, instead of constant arrays of 2^(index "
           "bits) rows; a table of more segments stays a constant and is "
           "counted. 0: every table a constant">,
    Option<"lookaheadTables", "lookahead-tables", "bool", "true",
           "With virtual memories: the readiness a frame's position waits on "
           "reads its slot tables (the next position's use and instance) from "
           "registers loaded a host cycle before, chosen by the decision, not "
           "through the position's increment and the table's decoder in the "
           "decision's host cycle (the same values: no schedule or host cycle "
           "moves; Beacon, 16 SMs: 13 levels on the position, sequencers, "
           "bubbles and output dues)">,
    Option<"dropDeadInputs", "drop-dead-inputs", "bool", "true",
           "A model input its body never reads (left over from a class cut: "
           "its bits go to a sibling class directly) takes no link, no "
           "storage and no read register: a constant to the copy">,
    Option<"shareLinks", "share-links", "bool", "false",
           "Outputs of a model driven by the same value (a bus broadcast to "
           "several children) are one producer: their readers share its "
           "links instead of one link per output">,
    Option<"stageLevels", "stage-levels", "unsigned", "0",
           "Staged evaluation: a class whose state cone (from its state and "
           "inputs to its next state, memory ports and early outputs) is "
           "deeper than this many estimated logic levels evaluates over "
           "several host cycles: its logic cut into segments of at most this "
           "many levels by registers that step with the frame's positions "
           "(none moves in a bubble or between frames), its state and inputs "
           "read the stages' positions before its state pass, which writes "
           "as before. Not for a class with black boxes or folded pipelines. "
           "Every pass of such a class (its ready passes too) reads the "
           "stages before its slot and writes its outputs at its slot from "
           "the last stage: its late outputs take the stages as latency. A "
           "held virtual-memory read's request leaves a host cycle after the "
           "state pass from the last stage's registered row and held row. "
           "The schedule places such a pass at least the stages into the "
           "frame, its inputs' links the stages earlier, and its state gap "
           "from the read. 0: none">,
    Option<"stageLevelsMapped", "stage-levels-mapped", "std::string", "\"\"",
           "With stage-levels: size each class's stages from the depth of its "
           "logic mapped to this Beacon target's cells (foldsim-beacon-map + "
           "foldsim-estimate-timing on one instance of the class's model), "
           "not from the estimate alone: a class whose mapped depth fits "
           "stage-levels is not staged; a deeper one takes ceil(depth / "
           "stage-levels) - 1 stages at most, its cuts placed by the estimate "
           "balanced over them. Empty: the estimate alone">,
    ListOption<"stageLevelsClass", "stage-levels-class", "std::string",
               "Per class over stage-levels: <glob>:<N> stages a class (globs "
               "on the model's name; a copy goes by its class's) at N levels, "
               "<glob>:off not at all; the last matching spec wins. A spec "
               "that matches no class is an error">,
    ListOption<"multicycle", "multicycle", "std::string",
               "With stage-levels: classes (globs; a copy goes by its class's "
               "name) staged without stage registers. A copy evaluates one "
               "instance at a time over M host cycles, M the cone's estimated "
               "levels over stage-levels, or M given as <glob>:<M> (the logic "
               "screen's count of the cone over the levels a host cycle may "
               "take): the state and inputs it reads are "
               "taken into capture registers at the read position (the state "
               "pass's M + 1 before), the logic computes from them over M "
               "host cycles into end registers taken a position before the "
               "state pass, which writes from them; nothing else reads a "
               "capture register or writes an end register. The schedule "
               "spaces the copy's state passes at least M positions apart, so "
               "a capture holds until its end registers are taken. The "
               "multicycle paths (capture to end, M host cycles) are listed "
               "for the flow (`foldsim.multicycle`). Only for a class that "
               "could be staged and reads no memory combinationally">,
    ListOption<"passSplit", "pass-split", "std::string",
               "Pass splitting: classes (globs; a copy goes by its class's "
               "name), as <glob>:<P>[:<S>]. The logic only registers' next "
               "values use (no output, memory port or other register reads "
               "it; an early output's next value counts as a reader) has its "
               "isomorphic fanout-free cones (lanes: the same operations, "
               "attributes and types, leaves by position) run on ceil(R/P) "
               "shared lanes, one replica a pass of S host cycles after the "
               "state pass, from that logic's leaves captured at the state "
               "pass; a leaf position a lane's replicas do not share is chosen "
               "by the pass, each replica's result held in a register. Those "
               "registers are written (P + 1) x S host cycles after the state "
               "pass, whole, at the captured instance (kept in stores, never "
               "paged); the schedule keeps the copy's state passes that far "
               "apart and the state gap that and 4 more. Planned on the model "
               "before scheduling: a class without lanes is left as it is. A "
               "static FPGA's LUTs keep every stage's logic, so only the logic "
               "the passes share is saved. Not for a staged class">,
    ListOption<"stateInRam", "state-in-ram", "std::string",
               "State in RAM (SiR): classes (globs on the model's name; a "
               "copy goes by its class's) as <glob>[:<min-bits>[:<P>]]. A "
               "matching class of more than one instance, whose written "
               "register bits an instance (not the ready passes' registers) "
               "are at least min-bits (default 1), keeps them in a RAM of a "
               "row an instance times P plus a pass (P auto: the fewest that "
               "fill a column's depth, 512 or 4096 rows, within "
               "state-in-ram-h), in columns of whole "
               "72-bit ports, a mask lane a write enable. P = 1 (default): the "
               "row read in place two host cycles ahead, through the RAM's "
               "register and an output register, written whole a host cycle "
               "after the state pass, at a state gap of at least 4. P > 1: "
               "the rows read one a host cycle into capture registers and "
               "written back one a host cycle through a P:1 row select (the "
               "paged form of `page`, with masks). Style by a storage rule "
               "<model>:sir=<block|ultra>; else block where the rows fit "
               "RAMB36 512 x 72, ultra otherwise. It replaces a class's pages "
               "(page, on or off chip)">,
    Option<"stateInRamReadBits", "state-in-ram-read-bits", "unsigned", "0",
           "With state-in-ram: the RAM bits an evaluation reads a host cycle "
           "(N > 0): P = ceil(bits / N) for a class whose spec gives no P">,
    Option<"stateInRamPP", "state-in-ram-pp", "bool", "false",
           "With state-in-ram: a class of P > 1 in the constant-shell fold's "
           "pass-partitioned form (CSF PP): no pages, no capture or write-back "
           "buffer. Its registers are packed into P rows in dependency order; "
           "the class is staged with P stages, and pass j reads row j of every "
           "72-bit column in place (the RAM's output register feeds the logic) "
           "and writes it from the logic in the same host cycle through a P:1 "
           "row select. Values that cross passes are captured once, at their "
           "pass; a register whose next value needs a later pass than its row "
           "stays in a store (a hazard register). A combinationally read memory "
           "of the class is read a stage after its address, from replicated "
           "address registers (state-in-ram-pp-fanout)">,
    Option<"stateInRamPPReplicas", "state-in-ram-pp-replicas", "unsigned", "0",
           "With state-in-ram-pp: each column's output register replicated "
           "per subtree (the first N components of its registers' names) of "
           "the registers it holds, a replica over the bits they take in the "
           "column, kept apart, each register read from its subtree's "
           "replica: a column's fan-out split by the logic it feeds. 0: off">,
    Option<"stateInRamPPPack", "state-in-ram-pp-pack", "bool", "true",
           "With state-in-ram-pp: the registers a class's rows leave out "
           "(hazards, deep ones, the others kept) packed into 72-bit columns "
           "of an instance's row each (block RAM up to 1,024 instances, else "
           "ultra), read at the instance two host cycles ahead and written a "
           "host cycle after the state pass, as state-in-ram P = 1's rows; "
           "given a state gap of 4, else (or false) a store each">,
    Option<"stateInRamPPFanout", "state-in-ram-pp-fanout", "unsigned", "96",
           "With state-in-ram-pp: the most RAM primitives a combinationally "
           "read memory's address register drives (the memory is banked by "
           "width, a registered address copy per bank)">,
    Option<"stateScratchBits", "state-scratch-bits", "uint64_t", "250000000",
           "Constant shell: the on-chip state scratch, in bits, for every "
           "class's state rows (state-in-ram-pp columns and on-chip pages, "
           "all copies). Whole classes stay resident, largest first while "
           "they fit; the others' rows stream from HBM (a PP class through a "
           "fixed landing window, state-stream-window; a paged class as "
           "pages=hbm). A class a storage rule keeps on chip is resident "
           "whatever the budget">,
    Option<"stateScratchDeviceBits", "state-scratch-device-bits", "uint64_t", "0",
           "Constant shell, state-scratch-bits auto: the device's ultra RAM "
           "bits at the placer's fill (its block RAM's: "
           "state-scratch-device-block-bits). The scratch is then this, "
           "less the memories the fold places on chip outside it (the "
           "models' memories past 18 kbit a copy, as instances x rows x "
           "width, unless virtual) and state-scratch-reserve; it overrides "
           "state-scratch-bits. 0: state-scratch-bits as given">,
    Option<"stateScratchDeviceBlockBits", "state-scratch-device-block-bits",
           "uint64_t", "0",
           "With state-scratch-device-bits (the ultra RAM's): the device's "
           "block RAM bits at the placer's fill. The scratch is budgeted per "
           "kind: a class's rows go in its style's RAM, else the other's">,
    Option<"stateScratchReserve", "state-scratch-reserve", "uint64_t", "45000000",
           "With state-scratch-device-bits: block RAM bits kept from the "
           "scratch for what the fold places after it (link memories, read "
           "buffers and rings, the landing windows, the folded pipelines' "
           "smaller stores; 44 Mbit in e1's A100-sized 128-SM fold)">,
    Option<"stateScratchBuiltOthers", "state-scratch-built-others", "std::string",
           "\"\"",
           "With state-scratch-device-bits: the other memories (not the "
           "scratch's) a first fold built, \"<ultra bits>,<block bits>\" as "
           "the fit check prices them; the scratch is the device less them, "
           "with no estimate and no reserve (the driver's as-built rerun)">,
    Option<"stateScratchReport", "state-scratch-report", "std::string", "\"\"",
           "With state-scratch-device-bits: a JSON file of the bits the "
           "planner placed and counted, per kind, and the device's">,
    Option<"stateScratchReserveUltra", "state-scratch-reserve-ultra", "uint64_t",
           "10000000",
           "With state-scratch-device-bits: ultra RAM bits kept from the "
           "scratch for what the fold places after it (deep link memories)">,
    Option<"stateScratchPartial", "state-scratch-partial", "bool", "true",
           "With state-scratch-bits: a state-in-ram-pp class past the scratch "
           "keeps the instances the rest of the scratch holds resident (in "
           "RAMs of their own) and streams only the "
           "others, spread over the frame (false: the whole class streams)">,
    Option<"stateScratchFixed", "state-scratch-fixed", "bool", "false",
           "With state-scratch-bits: the scratch as a fixed allocation, the "
           "same RAM at every instance count. Each scratch class (state-in-"
           "ram-pp, and paged classes, a storage rule's on-chip page kind "
           "included) takes the share of the scratch its rows' share of all "
           "the classes' rows gives, in instances (the same at every count "
           "when the classes scale together); its resident RAMs are that "
           "deep, under-filled at small counts. A PP class streams the rest "
           "(at least one instance in 16, so that its landing windows and "
           "HBM path are there at every count; resident: the first r of each "
           "B instances, B a power of two); a paged class is resident while its share holds "
           "it, else its pages stream (pages=hbm). Folded pipelines' stores "
           "take no scratch">,
    Option<"stateScratchFixedPP", "state-scratch-fixed-pp", "bool", "true",
           "With state-scratch-fixed and state-in-ram-pp (default on): every "
           "paged class of several instances is state in RAM (PP, P its pages), "
           "so that the fixed scratch keeps a share of its instances and streams "
           "the rest, as it does a PP class's (pages are kept or streamed "
           "whole: a RAM step at a count, a share unused)">,
    Option<"stateStreamWindow", "state-stream-window", "unsigned", "0",
           "With state-in-ram-pp, a class streamed from HBM: the most "
           "instance slots of its landing window (each instance's P rows "
           "prefetched into a slot, read in place at its passes); 0: a "
           "RAMB36's depth (512 / P rows)">,
    Option<"stateInRamZeroLinks", "state-in-ram-zero-links", "bool", "false",
           "With state-in-ram and zero-latency-links: links into a class of "
           "P > 1 may be taken in their producer's host cycle (default: from "
           "their read registers, as a staged reader's, so that no "
           "producer's logic chains into the evaluation on the last pass)">,
    Option<"stateInRamH", "state-in-ram-h", "unsigned", "0",
           "With state-in-ram and P auto: the most host cycles a copy's "
           "passes may take, instances x P (0: no bound)">,
    ListOption<"csfReadyCopies", "csf-ready-copies", "std::string",
               "Constant-shell fold (schedule only): models whose ready passes "
               "run on k ready-cone copies of their own, as <model>:<k>. "
               "Instance i's ready passes take lane i mod k; they no longer "
               "take the slots of the model's copies, whose state passes "
               "keep them">,
    Option<"csfReadyCones", "csf-ready-cones", "bool", "false",
           "csf-ready-copies built (schedule and simulator alike): each named "
           "model's late outputs whose cones are logic alone move to a "
           "stateless model <model>_rc of the inputs they read and of the "
           "registers they read (the model's early output csf_ready_regs), "
           "an instance of it beside each of the model's; its k copies "
           "(copies=<model>_rc:k) are the lanes. A ready-choice naming the "
           "model names its cone's classes too. Without it csf-ready-copies "
           "is for schedule-only runs">,
    Option<"csfMailboxLatency", "csf-mailbox-latency", "unsigned", "1",
           "With csf-mailboxes: the host cycles a mailbox read takes past a "
           "register link's (1: the RAM's output register; 2 with a cascade "
           "or a fabric register after it)">,
    Option<"csfSingleBank", "csf-single-bank", "bool", "false",
           "Constant-shell fold: an early link (a memory of several entries) "
           "all of whose reads of a frame come before its producer writes "
           "the entry in that frame keeps one bank, not two; the others "
           "keep two (the schedule decides, link by link)">,
    Option<"csfMailboxHbmChains", "csf-mailbox-hbm-chains", "bool", "false",
           "With csf-mailbox-hbm: early chain links (a ring of siblings' "
           "registered outputs, each entry read by the next instance a frame "
           "after its write) stream through HBM like other early links; "
           "zero-latency chains (late links) stay on chip either way">,
    Option<"csfMailboxHbmSplit", "csf-mailbox-hbm-split", "bool", "false",
           "With csf-mailbox-hbm: a producer-indexed stream sliced by reader "
           "whose reader page is wider than a row (csf-mailbox-hbm-row-bits; "
           "a parent's wide outputs to one child) is split into streams whose "
           "pages each fit a row, a reader pass landing a row of each, rather "
           "than left to on-chip memories">,
    Option<"csfMailboxHbmShare", "csf-mailbox-hbm-share", "bool", "false",
           "With csf-mailbox-hbm: where every source of a reader input's bits "
           "is a member of one stream at one offset (a parent's ports, one a "
           "child, to one input of its children), the sources share one set "
           "of read registers, loaded at each reader pass's landing, and the "
           "reader chooses no source (constant registers and selects)">,
    Option<"linkRings", "link-rings", "unsigned", "0",
           "Constant-shell fold: a link memory whose entries are each live "
           "for a short window of the frame (from the write a read takes to "
           "that read) keeps K entries, K a power of two up to this many, "
           "addressed by the entry's index mod K, where the schedule shows "
           "no other entry of its address written in any read's window "
           "(early links drop their banks); its storage is set by the "
           "schedule's spacing, not by the instances (0: off)">,
    Option<"csfMailboxHbmRowBits", "csf-mailbox-hbm-row-bits", "unsigned",
           "4096",
           "With csf-mailbox-hbm: the most bits of a mailbox stream's row (8 "
           "to 4096): a reader's lanes, or a reader group's slice of a "
           "producer's word (the bits several groups read at the same offset "
           "in every slice), so that a reader pass lands one row">,
    Option<"csfMailboxHbmLead", "csf-mailbox-hbm-lead", "unsigned", "112",
           "With csf-mailbox-hbm: the most positions a mailbox stream's row "
           "request precedes its landing (the HBM round trip at its worst, "
           "80 host cycles, the decision lead and a margin); each stream "
           "takes less where the schedule leaves its reader instances less "
           "room (all but the tenth with the least), the others on chip. "
           "Its landing slots are pow2(lead + 8), whatever the instances">,
    Option<"csfMailboxHbm", "csf-mailbox-hbm", "bool", "false",
           "Constant-shell fold: early links whose producer instances are "
           "each read by one reader instance (chain self-links aside) kept "
           "in HBM as reader-indexed mailboxes: HBM link streams (link-kinds "
           "hbm's machinery) whose rows are reader rows, written byte-masked "
           "at the producer's pass at the row of the reader its instance "
           "feeds, a row of at most link-hbm-row-bits a stream; each reader "
           "pass's row lands in the stream's fixed landing slots. A link "
           "whose room the schedule does not leave stays on chip">,
    Option<"csfMailboxBuilder", "csf-mailbox-builder", "bool", "false",
           "Constant-shell fold: the links from a producer class to a reader "
           "class whose producer instances are each read by one reader "
           "instance become lanes of one reader-indexed mailbox (a row per "
           "reader instance, a lane per link, written masked at the "
           "producer's pass, read whole at the reader's); links that fan "
           "out, are taken now, forwarded, cross SLR groups, paged, staged "
           "or streamed stay as they are">,
    Option<"csfMailboxes", "csf-mailboxes", "bool", "false",
           "Constant-shell fold (schedule only): links are rows of RAM "
           "mailboxes owned by their reader, one read port each: a link not "
           "taken in its producer's host cycle takes a host cycle more (the "
           "RAM's output register), and a reader instance that reads several "
           "producer instances of one class from rows no lane banking keeps "
           "apart holds its mailbox port that many slots (its passes spaced)">,
    ListOption<"splitLatency", "split-latency", "std::string",
               "Classes (globs; a copy goes by its class's name) whose results "
               "take n host cycles more, as <glob>:<n>: a copy split over SLRs "
               "with its crossings registered (slr-split). Its links are never "
               "taken in its own host cycle and every reader takes them n host "
               "cycles later">,
    Option<"splitLatencyFile", "split-latency-file", "std::string", "\"\"",
           "Per class output, its own split-latency: lines `<class> <output "
           "index> <host cycles>` (foldsim-slr-split crossing=txrx lags=: "
           "each output as late as its own paths through the crossings), "
           "over split-latency">,
    ListOption<"registeredLinks", "registered-links", "std::string",
               "Classes (globs; a copy goes by its class's name) whose links "
               "are never taken in their producer's host cycle: with "
               "zero-latency-links, a reader takes them from its read register. "
               "For a class whose late outputs end a deep path (a stateless "
               "address unit, say) that a zero-latency link would chain into "
               "its reader's logic">,
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

def LaneScan : Pass<"foldsim-lane-scan", "::mlir::ModuleOp"> {
  let summary = "Fold an unfolded parent's per-child glue into a lane cell "
                "class and its reductions into scans";
  let description = [{
    In the flat `module` (the unfolded parent of the classes), the instances
    of one module, at least `min-lanes` of them (the modules of `lanes`, all
    if empty), are lanes. Values computed per lane, isomorphic across the
    lanes (packs, SLP-style: per-lane ops, packed lane vectors whose bits j
    are lane j's, registers over them), move into a generated module
    `<module>_lane_<lane module>`, one instance per lane: a class to fold.
    Reductions across the lanes become zero-latency chains through those
    instances, the parent reading the chain's end: XOR / OR / AND / ADD over
    a pack (ac), AND / OR / XOR of a packed lane vector (icmp against all
    ones or zero, parity: vec), priority mux chains in lane order (prio),
    hw.array_get of a pack at a parent index (select), a bit of a packed lane
    vector at a parent index (bitsel), and comb.add of a packed lane vector
    and a constant or a vector of the same layout (carry). Lane-varying
    constants and slices of parent values are per-lane inputs of the cell;
    moved values the parent still reads are cell outputs. Prints a line per
    lane group: `lane-scan: cell <name>`, and the packs, scans and per-lane
    ports. Exact: the same operators on the same operands in the same target
    cycle.
  }];
  let options = [
    Option<"module", "module", "std::string", "\"\"",
           "The unfolded parent (flat but for its class instances)">,
    ListOption<"lanes", "lanes", "std::string",
               "The lane modules (all modules with min-lanes instances if "
               "empty)">,
    Option<"minLanes", "min-lanes", "unsigned", "4",
           "The fewest instances of a module that make lanes">,
    Option<"dryRun", "dry-run", "bool", "false",
           "Report what would move; change nothing">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
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

def LowerWideNegation : Pass<"foldsim-lower-wide-negation",
                             "::circt::hw::HWModuleOp"> {
  let summary = "Wide negations as logarithmic prefix logic";
  let description = [{
    Rewrites a negation `comb.sub 0, x` of at least `min-width` bits as
    `x ^ (P << 1)`, P the inclusive prefix OR of x's bits in log2(width)
    stages of `p | (p << d)` (bit i of -x is x[i] XOR the OR of the bits
    below it). A synthesis tool lays the subtraction on a carry chain, one
    carry block per 8 bits on an FPGA, so the depth of `x & -x`, the lowest
    set bit of a priority select, grows with the number of inputs; the
    prefix grows with their log. Exact: the value is the same for every x.
    For the few wide one-hot selects of a design: narrower negations (the
    datapath's) stay on the carry chain, which is as fast up to a few blocks.
  }];
  let options = [
    Option<"minWidth", "min-width", "unsigned", "65",
           "Negations of at least this many bits are rewritten">,
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

def NarrowShiftAmounts : Pass<"foldsim-narrow-shift-amounts",
                              "::circt::hw::HWModuleOp"> {
  let summary = "Shift amounts narrowed to log2 of the width";
  let description = [{
    Rewrites a shift (`comb.shl`, `comb.shru`, `comb.shrs`) of at least
    `min-width` bits whose amount is not a constant (nor already a zero
    extension of its low ceil(log2 width) bits) as `hi != 0 ? fill : x op
    zext(lo)`, lo the amount's low ceil(log2 width) bits, hi the rest, fill
    zeros (the sign for `shrs`). Exact. The out-of-range test is an OR of
    the high bits (a w-bit `a >= w` compare lowers as a chain, ~w/8 levels),
    and a split evaluation that carries the amount across passes (state in
    RAM, PP captures) carries ceil(log2 width) + 1 bits, not w.
  }];
  let options = [
    Option<"minWidth", "min-width", "unsigned", "17",
           "Shifts of at least this many bits are rewritten">,
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

def CompressAddTrees : Pass<"foldsim-compress-add-trees",
                            "::circt::hw::HWModuleOp"> {
  let summary = "Wide sums of many addends as compressor trees";
  let description = [{
    Gathers a tree of additions into the addends it sums: an addition's
    operand that is another addition of the same width, used there alone, or
    the zero extension of a narrower addition that cannot wrap (the largest
    values its operands' known bits allow fit in its width), used there
    alone. A sum of at least `min-operands` addends is rebuilt as a
    compressor tree of full and half adders (circt::datapath::CompressorTree,
    Dadda's reduction, as `convert-datapath-to-comb` lowers
    `datapath.compress`) reducing the addends' bits to two rows, then one
    addition of the two. A population count over many inputs written as a
    tree of two-input additions (a pairwise adder tree) is the case that
    matters: a synthesis tool lays each addition on a carry chain, so its
    depth grows with a carry chain per level of the tree; the compressor's
    depth grows with log1.5 of the addends, in logic that packs into LUTs.
    Exact: two's-complement sums modulo the width, every addend kept. The
    other additions of the design (fewer addends) are left as they are.
  }];
  let options = [
    Option<"minOperands", "min-operands", "unsigned", "64",
           "Sums of at least this many addends are rewritten">,
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

def KeepModules : Pass<"foldsim-keep-modules", "::mlir::ModuleOp"> {
  let summary = "Keep the instances of named modules through inlining";
  let description = [{
    Makes the modules named in `modules` public, so that
    `hw-flatten-modules`, which inlines only private modules, keeps their
    instances. A name also matches its specializations (`VX_elastic_buffer`
    matches `VX_elastic_buffer_12`). Run before `foldsim-beacon-map` to read
    the depth at those instances' ports (`foldsim-estimate-timing`'s
    `ports`); LUTs then no longer pack across their boundaries.
  }];
  let options = [
    ListOption<"modules", "modules", "std::string",
               "Modules whose instances to keep">,
  ];
}

def BoundaryArrivals : Pass<"foldsim-boundary-arrivals", "::mlir::ModuleOp"> {
  let summary = "Give each module its hierarchy's arrival times at its ports";
  let description = [{
    Reads `file`, the `arrivals` of `foldsim-estimate-timing` over a mapping
    of the same design, and sets `synth.input_arrival_times` on each module
    it lists (its input ports) and `synth.output_arrival_times` on each of
    their instances (their outputs), which `synth-generic-lut-mapper` with a
    `carry-block-size` takes as the arrival times of those values. A module
    mapped apart from the rest of its hierarchy then times a late input, or
    an instance's late output, as a mapping of the flattened design would,
    and keeps it out of deep cones. Ports missing from the file, or of
    another width, arrive at 0.
  }];
  let options = [
    Option<"file", "file", "std::string", "",
           "The arrival times (JSON) of foldsim-estimate-timing's `arrivals`">,
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

def MergeMemories
    : Pass<"foldsim-merge-memories", "::circt::hw::HWModuleOp"> {
  let summary = "Merge memories of equal ports side by side";
  let description = [{
    Memories with one unmasked write port and one read port whose addresses,
    enables and clocks are the same values up to copies (registers and logic
    of one kind and attributes, names aside, over the same operands) take the
    same entries at the same host cycles: laid side by side in one memory of
    at most `max-bits` bits, they read and write as before, in fewer and
    fuller RAM columns. The constant-shell fold's link mailboxes.
  }];
  let options = [
    Option<"maxBits", "max-bits", "unsigned", "512",
           "The widest merged memory, in bits">,
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::seq::SeqDialect"];
}

def MemoriesToRegisters
    : Pass<"foldsim-memories-to-registers", "::mlir::ModuleOp"> {
  let summary = "Keep the named memories as registers, a register an entry";
  let description = [{
    `circt-verilog --detect-memories` makes a memory of every register array
    it can, also of an associative table such as a cache's MSHR address
    table, whose every entry is read every cycle for a match: a memory with
    as many read ports as entries, each at a constant address. Folded, such
    a memory is a store of its depth times the instances, copied once per
    read port (distributed RAM has one read port a copy), where its entries
    kept as registers are a row of the copy's state per instance, read
    whole, and a read at a constant address is a wire.

    Each memory of `memories` (`<module>:<name>`, the memory's `name`)
    becomes a register per entry, `<name>_<entry>`, of the memory's width:
    a read at a constant address is that register (zero past the depth), a
    read at another address a select among them (zero past the depth); each
    write port, in the order of the module, loads the entry it addresses
    when enabled, a mask lane at a time, a later port's write over an
    earlier one's. Only memories read combinationally and written a cycle
    later (read latency 0, write latency 1), without read-write ports or an
    initial content, whose write ports share one clock, can be registers;
    another named is left a memory, with a warning. `foldsim --memory-
    registers` chooses the memories (foldsim_storage.py prices both forms).
  }];
  let options = [
    ListOption<"memories", "memories", "std::string",
               "The memories to keep as registers, as <module>:<name>">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
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
    stores. A load is a pin per bit of the register: a LUT memory's address,
    enable or mask is its width times its depth in 32-entry LUTs, a block
    RAM's a pin per 36 bits and 1024 entries; a select's condition is its
    width, anything else one. A LUT memory is one of at most 64 entries, one
    read asynchronously (read latency 0, which no block RAM has: synthesis
    builds it of LUTs however deep), or one whose `ram_style` says
    distributed or registers; `ram_style` block or ultra makes any memory a
    block RAM. Concatenations, extractions, bit casts, replications and
    inversions of the register are wiring, copied with it. A LUT memory or a
    select whose one port is over `loads` is first split by bits into parts of
    at most `loads` (a memory in whole mask bits). The replicas and the register get
    `DONT_TOUCH`. The simulator's behavior is the same on every host cycle.

    The marks are removed in any case; `loads` 0 replicates nothing.

    With `fanout` N, any other register of more than N loads is replicated
    the same way (groups of at most N), muxes and LUT memories it drives over
    N split first; with `fanout-logic` the loads are counted, and the logic
    copied, through one-LUT logic (`isLutLogic`) as well. Replicas hold the
    register's value on every host cycle and copied logic computes the same
    function of them: cycle for cycle the same simulator.
  }];
  let options = [
    Option<"loads", "loads", "unsigned", "96",
           "The most loads a marked register or a replica drives; 0: none "
           "replicated">,
    Option<"fanout", "fanout", "unsigned", "0",
           "Also every other register (the copies' own state and the "
           "simulator's read selects) of more loads than this: replicated "
           "the same way, each replica at most this many; 0: none">,
    Option<"fanoutLogic", "fanout-logic", "bool", "false",
           "With fanout: the one-LUT logic between such a register and its "
           "loads (a one-bit result of at most six input bits: a reset or "
           "enable decoded from a few flops) is copied with each replica, so "
           "the net past it is bounded too">,
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

def SlrSplit : Pass<"foldsim-slr-split", "::mlir::ModuleOp"> {
  let summary = "Split a module into parts, one per SLR, cut at registers";
  let description = [{
    The module `module` (a class copy of a folded simulator, too large for
    one SLR) becomes `parts` modules `<module>_p<k>`, each instantiated in
    its place as `<prefix><module>_p<k>`, so that a floorplan can hold each
    part in an SLR of its own. The map (`map`, a file: one integer per op of
    the module's body in order, its part, or -1 for an op copied into every
    part that reads it; foldsim_slr_split.py --apply-map writes it from a
    min-cut of the netlist) gives the parts. Constants are copied into every
    part. A memory's ports must be in its part.

    Every value that crosses from one part to another leaves its part from
    a register: a flop, a memory's registered read, or an instance's output.
    The split changes no cycle and no schedule: no register is added, and
    the crossing is a flop's output driving a super long line (the folded
    pipeline's registers, its read registers and stage registers, are the
    cut). Logic of another part that a part reads (`copy-logic`, default) is
    copied into it, up to registers; a combinational memory read across
    parts is an error (its memory would have to move), as is any
    combinational crossing without `copy-logic`. A register or memory the
    map replicates (-1) is copied whole, with its logic, into every part
    that reads it: the fold's counters and their decodes, which each part
    then keeps in lockstep from the same inputs.
  }];
  let options = [
    Option<"moduleName", "module", "std::string", "\"\"",
           "The module to split">,
    Option<"mapPath", "map", "std::string", "\"\"",
           "The map: one part (or -1) per op of the module's body, in order">,
    Option<"parts", "parts", "unsigned", "2", "The parts">,
    Option<"prefix", "prefix", "std::string", "\"fs_split_\"",
           "The parts' instance names' prefix (the floorplan finds them)">,
    Option<"copyLogic", "copy-logic", "bool", "true",
           "Copy another part's logic a part reads into it (else an error)">,
    Option<"crossing", "crossing", "std::string", "\"source\"",
           "source: a crossing leaves its part from a register, no register "
           "added; txrx: every crossing a send and a receive register "
           "(USER_SLL_REG, DONT_TOUCH), the module retimed around them (each "
           "op a lag, a memory's writes up to `slack` later than its reads, "
           "the outputs `latency` late: the class's split-latency)">,
    Option<"latency", "latency", "unsigned", "2",
           "crossing=txrx: the host cycles the outputs come late (the fold's "
           "split-latency for the class)">,
    Option<"lagsPath", "lags", "std::string", "\"\"",
           "crossing=txrx: each output as late as its own value (no padding to "
           "`latency`), its lag written here as `<class> <output> <n>` lines: "
           "the fold's split-latency-file for the class">,
    Option<"alignStyle", "align-style", "std::string", "\"register\"",
           "crossing=txrx: the alignment chains (two stages or more) as "
           "flops (register: srl_style=register, the default; flops are the "
           "plentiful resource where LUTs bind), shift-register LUTs (srl: "
           "srl_style=srl), or Vivado's choice (auto: no attribute)">,
    Option<"slack", "slack", "unsigned", "8",
           "crossing=txrx: host cycles a memory's writes may land later than "
           "its reads are taken (under the class's state gap)">,
  ];
  let statistics = [
    Statistic<"numCutValues", "cut-values", "Values crossing between parts">,
    Statistic<"numCutBits", "cut-bits", "Bits crossing between parts">,
    Statistic<"numCopiedLogic", "copied-logic", "Logic ops copied into a reading part">,
  ];
  let dependentDialects = ["circt::hw::HWDialect", "circt::seq::SeqDialect",
                           "circt::sv::SVDialect"];
}

def RemoveDeadState : Pass<"foldsim-remove-dead-state", "::circt::hw::HWModuleOp"> {
  let summary = "Remove registers and memories that no output observes";
  let description = [{
    A register, memory or combinational value is live when it reaches an
    output of the module, an instance's input or an op with side effects,
    through any number of registers and memories (a memory is live when one
    of its read ports is; its writes then are). Everything else is removed:
    it can change no output in any cycle. Canonicalization already removes
    an unused value, but not state in a cycle of its own, such as a counter
    whose only reader is its own increment: an unconnected performance
    counter (`VX_stream_xbar`'s collision count, quadratic in its inputs)
    keeps its whole cone. Registers and memories with a symbol stay.
  }];
  let statistics = [
    Statistic<"numRemoved", "removed", "Ops removed">,
  ];
  let dependentDialects = ["circt::hw::HWDialect", "circt::seq::SeqDialect"];
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

def BeaconMap : Pass<"foldsim-beacon-map", "::mlir::ModuleOp"> {
  let summary = "Map a design to a Beacon target's cells";
  let description = [{
    Maps a design to the cells of `target` (foldsim/Beacon.h) for
    `foldsim-estimate-timing` with the same `target`: 6-input LUTs on an FPGA
    family, 3-input cuts, each one library cell, on an ASIC PDK. Modules of
    fewer than 300 ops are inlined so cells pack across their boundaries, as
    Vivado maps a flattened design; the mapping is depth-oriented, as Vivado
    restructures chains of associative logic and priority muxes into trees;
    equality comparisons become logic so they pack with their neighbours;
    adders, order comparisons, multipliers and dividers stay word-level, as
    carry chains and DSPs.

    With `area`, the netlist is mapped for `foldsim-estimate-area` instead:
    every module kept, so each instance's resources stay its own (inlining
    changes the count by a few percent); no balancing, common subexpressions
    shared, each cone covered by the fewest LUTs, as Vivado maps the logic
    off its critical paths. A depth-oriented mapping duplicates logic to save
    levels, so its cells are no measure of area.

    For depth, `carry-block-size` makes the mapper time the bits of adders,
    comparators and distributed-RAM reads as the carry blocks of that many
    bits deliver them, so a late carry is not buried under a deep cone as
    Vivado would not bury it (0: every input bit arrives at 0). Each module
    left after inlining (a fold's models) is mapped apart, its input ports and
    its instances' outputs arriving at 0 unless `arrivals` gives them:
    `foldsim-estimate-timing`'s `arrivals` over a first mapping, so a second
    one times them as the whole hierarchy delivers them (as Vivado maps the
    flattened design; `foldsim-boundary-arrivals`).
  }];
  let options = [
    Option<"target", "target", "std::string", "\"versal\"",
           "The Beacon target to map to">,
    Option<"area", "area", "bool", "false",
           "Map for area (foldsim-estimate-area) instead of depth">,
    Option<"carryBlockSize", "carry-block-size", "unsigned", "0",
           "Bits per carry block the depth mapping times adders and "
           "comparators with (0: their input bits arrive at 0)">,
    Option<"arrivals", "arrivals", "std::string", "\"\"",
           "Arrival times at the modules' ports, from foldsim-estimate-timing's "
           "`arrivals` over a first mapping (JSON); none: they arrive at 0">,
  ];
}

def EstimateTiming : Pass<"foldsim-estimate-timing", "::mlir::ModuleOp"> {
  let summary = "Rank every register-to-register path by its logic levels";
  let description = [{
    Estimates, for every end point of the design elaborated from `top`, the
    logic levels of its longest path the way Vivado counts them on the
    `target` device: LUTs, CARRY8s, DSP cells and distributed-RAM reads. The input is a design mapped by
    `foldsim-beacon-map` for the same `target`, in which
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

    `target` is a Beacon target (foldsim/Beacon.h): an FPGA family as
    hw/syn/xilinx/platforms names it (`versal`, `virtexuplus`,
    `virtexuplusHBM`, `zynquplus`, which differ in their DSPs), or the PDK
    of hw/syn/yosys.

    `target=asap7` counts cell stages of an ASAP7 netlist (3-input cuts)
    instead: a cell
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
    crossings on the path, then start and end as an instance id and a name,
    and the largest fanout of a bit into the end point's last cell),
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

    `arrivals` writes, for every module instantiated below `top`, the levels
    at which each bit of its ports arrives, the worst over its instances: an
    input's in the instantiating module, an output's behind it. It is JSON,
    `{"<module>": {"in": {"<port>": [levels per bit]}, "out": {...}}}`, a
    single element when every bit has it, and no entry for a port whose bits
    all arrive at 0. It is what
    `foldsim-boundary-arrivals` reads to map each module again with its
    hierarchy's arrival times at its boundary.
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
    Option<"target", "target", "std::string", "\"versal\"",
           "The Beacon target whose logic levels to count">,
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
    Option<"arrivals", "arrivals", "std::string", "",
           "Per-bit arrival times at every instantiated module's ports, "
           "the worst over its instances (JSON)">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
}

def EstimateArea : Pass<"foldsim-estimate-area", "::mlir::ModuleOp"> {
  let summary = "Count the cells, memories and DSPs a design maps to";
  let description = [{
    Estimates the resources of the design elaborated from `top` on the
    `target` device, per instance: the input is a design mapped by
    `foldsim-beacon-map` for the same `target`, as for
    `foldsim-estimate-timing`, so both read one netlist.

    On an FPGA family:

    - `comb.truth_table`: a LUT. An and-inverter gate the mapper left: a
      LUT per five more inputs.
    - `comb.add`/`comb.sub`: a LUT per bit into the carry chain; an order
      `comb.icmp`, a LUT per two bits.
    - `comb.mul`: times a constant, shifts and adders, a LUT per bit; a
      table in LUTs, one per output bit up to the inputs' count, if its
      operands have at most ten variable bits; otherwise DSPs, the fewer of
      the two orientations over the significant bits of the operands
      (27 x 24 on Versal, 27 x 18 on UltraScale+).
    - dividers: a table over at most ten variable bits, else a subtracter
      per quotient bit; as many output bits as the dividend has.
    - registers: a flip-flop per bit.
    - `seq.firmem`: placed as Vivado infers it. A `ram_style` the RTL asks
      for (`block`, `distributed`, `ultra`, `registers`) wins; otherwise a
      read of latency 0 is distributed RAM, and a registered read of at
      least 32 words and 2 Kb is block RAM. Distributed RAM takes 0.57 LUTs
      a bit up to 32 words (RAM32M16) and 1.14 per 64 words beyond (RAM64M8)
      per read port, with a multiplexer tree over the 64-word LUTs. Block
      RAM takes the fewest 36 Kb blocks (or 18 Kb halves) of any shape, a
      copy per two ports (a read-write port is one); UltraRAM, 4096 x 72.
    - Vivado floating-point cores (`xil_*`, external modules): their
      measured LUTs, flip-flops and DSPs.

    On an ASIC library (`asap7`), `lut` counts cells: a 3-input cut is a
    cell, an adder a prefix adder, a product partial products and a Dadda
    tree, a memory without a macro flip-flops and multiplexers; an external
    module (an SRAM macro) counts as a macro.

    The mapping's LUTs are a cut mapper's cover: every output scales them by
    the target's `lutScale` (`lut-scale` overrides it), fitted on routed
    Vivado builds, and adds the memories' read multiplexers and vendor IP's
    measured LUTs, which no mapper covers.

    Writes the instances depth first to `output` (CSV: the instance's own
    resources and its subtree's), the memories and their placement to
    `memories`, each module's totals to `modules` (JSON), and a summary to
    `report`. With `state-threads`, `modules` also gives, per module and
    thread count, the distributed and block RAM of its registers (one memory
    of a word per thread) and of its memories that many times deeper, as a
    folded copy stores its instances' state. Leaves the design as it is.
  }];
  let options = [
    Option<"topName", "top", "std::string", "",
           "The top module (the one no other instantiates if empty)">,
    Option<"target", "target", "std::string", "\"versal\"",
           "The Beacon target whose cells to count">,
    Option<"output", "output", "std::string", "",
           "Instance table (CSV)">,
    Option<"memoryTable", "memories", "std::string", "",
           "Memory table (CSV)">,
    Option<"moduleTable", "modules", "std::string", "",
           "Per-module totals (JSON)">,
    Option<"report", "report", "std::string", "\"-\"",
           "Summary (text)">,
    ListOption<"blackBoxes", "black-boxes", "std::string",
               "Modules to leave out, as a harness stubs them">,
    ListOption<"stateThreads", "state-threads", "unsigned",
               "Thread counts at which `modules` prices folded state">,
    Option<"lutScale", "lut-scale", "double", "-1",
           "Device LUTs per mapped LUT in `output` and `report` (default: "
           "the target's lutScale)">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
}

#endif // FOLDSIM_PASSES
