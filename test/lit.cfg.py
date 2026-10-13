# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "foldsim"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.foldsim_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.add_tool_substitutions(["foldsim-opt"], [config.foldsim_tools_dir])
