# The FPGA shell a foldsim simulator runs in: its control and memory endpoints
# (rtl), the host's register map and in-process emulator (include, emul), the
# normative DRAM bank model the memory endpoint implements (model), and the
# endpoint's unit tests (test). Staged in the build tree and installed as
# share/foldsim/shell, beside the driver that builds the simulator.
set(FOLDSIM_SHELL_DIRS rtl include emul model test)
set(FOLDSIM_SHELL_STAGE ${CMAKE_BINARY_DIR}/share/foldsim/shell)
file(GLOB_RECURSE FOLDSIM_SHELL_FILES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/rtl/* ${CMAKE_CURRENT_SOURCE_DIR}/include/*
  ${CMAKE_CURRENT_SOURCE_DIR}/emul/* ${CMAKE_CURRENT_SOURCE_DIR}/model/*
  ${CMAKE_CURRENT_SOURCE_DIR}/test/*)
set(FOLDSIM_SHELL_STAMP ${CMAKE_CURRENT_BINARY_DIR}/shell.stamp)
set(FOLDSIM_SHELL_COPY)
foreach(dir ${FOLDSIM_SHELL_DIRS})
  list(APPEND FOLDSIM_SHELL_COPY COMMAND ${CMAKE_COMMAND} -E copy_directory
       ${CMAKE_CURRENT_SOURCE_DIR}/${dir} ${FOLDSIM_SHELL_STAGE}/${dir})
endforeach()
add_custom_command(OUTPUT ${FOLDSIM_SHELL_STAMP}
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${FOLDSIM_SHELL_STAGE}
  ${FOLDSIM_SHELL_COPY}
  COMMAND ${CMAKE_COMMAND} -E touch ${FOLDSIM_SHELL_STAMP}
  DEPENDS ${FOLDSIM_SHELL_FILES}
  COMMENT "Staging the foldsim shell")
add_custom_target(foldsim-shell ALL DEPENDS ${FOLDSIM_SHELL_STAMP})
install(DIRECTORY ${FOLDSIM_SHELL_DIRS} DESTINATION share/foldsim/shell)

# The memory endpoint against the bank model, in Verilator (test/Makefile).
if(FOLDSIM_VERILATOR)
  add_custom_target(check-foldsim-shell
    COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/test VERILATOR=${FOLDSIM_VERILATOR}
            BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/test
    COMMENT "Running the foldsim shell's unit tests"
    USES_TERMINAL)
endif()
