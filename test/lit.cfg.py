# -*- Python -*-

import os
import shutil

import lit.formats
from lit.llvm import llvm_config

config.name = "foldsim"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir", ".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.foldsim_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py", "Inputs"]

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
# FOLDSIM_SLOT_SEGMENTS: run the suite with slot-table decoders (foldsim-fold-static).
llvm_config.with_system_environment(["FOLDSIM_SLOT_SEGMENTS", "FOLDSIM_SLOT_MIN_INDEX_BITS"])
# The CIRCT tools (circt-verilog, circt-opt, firtool) of the CIRCT foldsim is
# built against, ahead of the LLVM tools: a CIRCT built on its own has them
# in a directory of its own.
if config.circt_tools_dir and config.circt_tools_dir != config.llvm_tools_dir:
    llvm_config.with_environment("PATH", config.circt_tools_dir, append_path=True)
# The driver's Python tools beside it (foldsim_manifest.py, foldsim_gate.py, ...).
config.substitutions.append(("%foldsim_tools", config.foldsim_tools_dir))
llvm_config.add_tool_substitutions(["foldsim-opt", "foldsim"],
                                  [config.foldsim_tools_dir])

# The simulation tests build a simulator with firtool and run it with
# Verilator (FOLDSIM_VERILATOR at configuration).
if (config.verilator and os.path.isfile(config.verilator) and
        shutil.which("firtool", path=os.pathsep.join([config.circt_tools_dir, config.llvm_tools_dir]))):
    config.available_features.add("verilator")
    config.substitutions.append(("%verilator", config.verilator))

# foldsim_degree.py state --exact solves a 0-1 program with scipy (HiGHS):
# FOLDSIM_PYTHON, a python that has it, else python3 if it has it.
import subprocess
_py = os.environ.get("FOLDSIM_PYTHON") or shutil.which("python3")
if _py and subprocess.run([_py, "-c", "import scipy.optimize"],
                          capture_output=True).returncode == 0:
    config.available_features.add("scipy")
    config.substitutions.append(("%scipy_python", _py))
