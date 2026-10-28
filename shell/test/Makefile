# The shell's unit tests in Verilator: the memory endpoint (rtl/foldsim_mem.sv)
# against the normative bank model (model/dram_bank_model.h), once per endpoint
# configuration.
#
#   make [VERILATOR=<verilator>] [BUILD_DIR=<dir>] [-j]   build and run them all
#   make run-<variant>                                     one of them
#
# A variant is the endpoint's parameters and the test's matching defines.

SHELL_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
RTL_DIR := $(SHELL_DIR)/rtl
TEST_DIR := $(SHELL_DIR)/test/foldsim_mem
VERILATOR ?= verilator
BUILD_DIR ?= $(CURDIR)/build
THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(min(16, mp.cpu_count()))')

VARIANTS := base staged spec spec2 s8 spec_s8 spec2_s8 ch2

PARAMS_base :=
PARAMS_staged := -GSTAGED=1
PARAMS_spec := -GSTAGED=1 -GSPEC=1
PARAMS_spec2 := -GSTAGED=1 -GSPEC=2
PARAMS_s8 := -GSTAGED=1 -GSLOTS=8
PARAMS_spec_s8 := -GSTAGED=1 -GSLOTS=8 -GSPEC=1
PARAMS_spec2_s8 := -GSTAGED=1 -GSLOTS=8 -GSPEC=2
PARAMS_ch2 := -GSTAGED=1 -GCH_SHIFT=1

DEFS_base :=
DEFS_staged :=
DEFS_spec := -DFOLDSIM_MEM_SPEC=1
DEFS_spec2 := -DFOLDSIM_MEM_SPEC=2
DEFS_s8 := -DFOLDSIM_MEM_READ_SLOTS=8
DEFS_spec_s8 := -DFOLDSIM_MEM_READ_SLOTS=8 -DFOLDSIM_MEM_SPEC=1
DEFS_spec2_s8 := -DFOLDSIM_MEM_READ_SLOTS=8 -DFOLDSIM_MEM_SPEC=2
DEFS_ch2 := -DFOLDSIM_MEM_CH_SHIFT=1

CXXFLAGS := -std=c++17 -Wall -Wextra -Wfatal-errors -Werror -fPIC -O2 -DNDEBUG -I$(TEST_DIR) -I$(SHELL_DIR)/model
VL_FLAGS := --exe --language 1800-2012 --assert -Wall -Wpedantic \
            -Wno-DECLFILENAME -Wno-REDEFMACRO -Wno-GENUNNAMED \
            --x-initial unique --x-assign unique -DSIMULATION -DNDEBUG -j $(THREADS) \
            -I$(RTL_DIR) $(RTL_DIR)/foldsim_token.sv --cc foldsim_mem --top-module foldsim_mem
RTL_SRCS := $(wildcard $(RTL_DIR)/*.sv $(RTL_DIR)/*.vh)

# run-<variant> is never a file, so it always runs (make applies no pattern
# rule to a phony target).
.PHONY: all build clean

all: $(addprefix run-,$(VARIANTS))
build: $(foreach v,$(VARIANTS),$(BUILD_DIR)/$(v)/foldsim_mem)

# verilator --build runs its own make, which cannot reach this make's
# jobserver, so it gets a clean MAKEFLAGS.
$(BUILD_DIR)/%/foldsim_mem: $(TEST_DIR)/main.cpp $(SHELL_DIR)/model/dram_bank_model.h $(SHELL_DIR)/model/dram_timing.h $(RTL_SRCS)
	@mkdir -p $(@D)
	MAKEFLAGS= $(VERILATOR) --build $(VL_FLAGS) $(PARAMS_$*) $(TEST_DIR)/main.cpp \
		-CFLAGS '$(CXXFLAGS) $(DEFS_$*)' --Mdir $(@D)/obj_dir -o ../foldsim_mem

run-%: $(BUILD_DIR)/%/foldsim_mem
	$<

clean:
	rm -rf $(BUILD_DIR)
