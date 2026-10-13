# The simulation tests run the simulators they build with Verilator, if found.
find_program(FOLDSIM_VERILATOR verilator
  DOC "Verilator, for the simulation tests (test/Simulation)")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
add_lit_testsuite(check-foldsim "Running the foldsim regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS foldsim-opt
)
