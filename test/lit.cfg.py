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
llvm_config.add_tool_substitutions(["foldsim-opt", "foldsim"],
                                  [config.foldsim_tools_dir])

# The simulation tests build a simulator with firtool and run it with
# Verilator (FOLDSIM_VERILATOR at configuration).
if (config.verilator and os.path.isfile(config.verilator) and
        shutil.which("firtool", path=config.llvm_tools_dir)):
    config.available_features.add("verilator")
    config.substitutions.append(("%verilator", config.verilator))
