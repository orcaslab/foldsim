# The driver: elaborates a design with CIRCT and builds its simulator.
configure_file(foldsim.py ${CMAKE_BINARY_DIR}/bin/foldsim COPYONLY)
install(PROGRAMS foldsim.py DESTINATION bin RENAME foldsim)
# The fold-degree search (--copies auto): prices each class with foldsim-opt's
# foldsim-estimate-area, the one area model.
configure_file(foldsim_degree.py ${CMAKE_BINARY_DIR}/bin/foldsim_degree.py COPYONLY)
configure_file(foldsim_cost.py ${CMAKE_BINARY_DIR}/bin/foldsim_cost.py COPYONLY)
configure_file(foldsim_storage.py ${CMAKE_BINARY_DIR}/bin/foldsim_storage.py COPYONLY)
# Target manifests: a design's whole recipe in one file (--manifest).
configure_file(foldsim_manifest.py ${CMAKE_BINARY_DIR}/bin/foldsim_manifest.py COPYONLY)
install(PROGRAMS foldsim_manifest.py DESTINATION bin)
# The design-independent steps of an image flow from a manifest (boundary
# lifting, the kernel shell's header).
configure_file(foldsim_flow.py ${CMAKE_BINARY_DIR}/bin/foldsim_flow.py COPYONLY)
install(PROGRAMS foldsim_flow.py DESTINATION bin)
# The flow's scripts (flow/: lift_boundaries.py; v80/: vmem_ports.py, package_ip.tcl, the implementation
# hooks, portfold_gen.sh; platform/: gen_platform.py and the card descriptors) and the kernel shell
# (shell/rtl, also reached as rtl/kernel: the memory and control endpoints, the virtual-memory controllers,
# foldsim_afu.sv.in),
# beside the tools in the build and the install.
file(GLOB_RECURSE FOLDSIM_FLOW_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/flow/*)
foreach(f ${FOLDSIM_FLOW_FILES})
  configure_file(${f} ${CMAKE_BINARY_DIR}/bin/${f} COPYONLY)
endforeach()
install(DIRECTORY flow DESTINATION bin USE_SOURCE_PERMISSIONS)
# The area models' calibrations (calibrations/*.json: --state auto's default, state-v80-*.json, among them).
file(GLOB FOLDSIM_CALIBRATIONS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/calibrations/*.json)
foreach(f ${FOLDSIM_CALIBRATIONS})
  configure_file(${f} ${CMAKE_BINARY_DIR}/bin/${f} COPYONLY)
endforeach()
install(DIRECTORY calibrations DESTINATION bin FILES_MATCHING PATTERN "*.json")
file(GLOB FOLDSIM_KERNEL_RTL RELATIVE ${PROJECT_SOURCE_DIR} CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/rtl/kernel/*)
foreach(f ${FOLDSIM_KERNEL_RTL})
  configure_file(${PROJECT_SOURCE_DIR}/${f} ${CMAKE_BINARY_DIR}/bin/${f} COPYONLY)
endforeach()
install(DIRECTORY ${PROJECT_SOURCE_DIR}/shell/rtl/ DESTINATION bin/rtl/kernel)
# A design's simulator gated against its own RTL in Verilator, by its ports.
configure_file(foldsim_gate.py ${CMAKE_BINARY_DIR}/bin/foldsim_gate.py COPYONLY)
install(PROGRAMS foldsim_gate.py DESTINATION bin)
install(PROGRAMS foldsim_degree.py DESTINATION bin)
install(FILES foldsim_cost.py DESTINATION bin)
install(PROGRAMS foldsim_storage.py DESTINATION bin)
# Judges a logic screen's end points with the multicycle paths per host cycle.
configure_file(foldsim_mc_judge.py ${CMAKE_BINARY_DIR}/bin/foldsim_mc_judge.py COPYONLY)
install(PROGRAMS foldsim_mc_judge.py DESTINATION bin)
configure_file(ready_choice.py ${CMAKE_BINARY_DIR}/bin/ready_choice.py COPYONLY)
install(PROGRAMS ready_choice.py DESTINATION bin)
# The SLLs an existing netlist needs against the region's budget (from its logged fold pipeline).
configure_file(foldsim_sll_check.py ${CMAKE_BINARY_DIR}/bin/foldsim_sll_check.py COPYONLY)
install(PROGRAMS foldsim_sll_check.py DESTINATION bin)
# A floorplan that pins each class copy to its SLR from a given assignment (judged by foldsim_sll_check --floorplan).
configure_file(foldsim_pin_copies.py ${CMAKE_BINARY_DIR}/bin/foldsim_pin_copies.py COPYONLY)
install(PROGRAMS foldsim_pin_copies.py DESTINATION bin)
configure_file(foldsim_locality.py ${CMAKE_BINARY_DIR}/bin/foldsim_locality.py COPYONLY)
install(PROGRAMS foldsim_locality.py DESTINATION bin)
configure_file(foldsim_xcheck.py ${CMAKE_BINARY_DIR}/bin/foldsim_xcheck.py COPYONLY)
install(PROGRAMS foldsim_xcheck.py DESTINATION bin)
# The fit calibration database (every synthesis of a netlist against its count) and the bound check of a fold
# against its region (floor, median, upper bound; per copy against the SLRs; the image's post-synthesis gate).
configure_file(foldsim_fitdb.py ${CMAKE_BINARY_DIR}/bin/foldsim_fitdb.py COPYONLY)
install(PROGRAMS foldsim_fitdb.py DESTINATION bin)
configure_file(foldsim_scratch_built.py ${CMAKE_BINARY_DIR}/bin/foldsim_scratch_built.py COPYONLY)
install(PROGRAMS foldsim_scratch_built.py DESTINATION bin)
configure_file(foldsim_link_kinds.py ${CMAKE_BINARY_DIR}/bin/foldsim_link_kinds.py COPYONLY)
install(PROGRAMS foldsim_link_kinds.py DESTINATION bin)
configure_file(foldsim_pp_columns.py ${CMAKE_BINARY_DIR}/bin/foldsim_pp_columns.py COPYONLY)
install(PROGRAMS foldsim_pp_columns.py DESTINATION bin)
configure_file(foldsim_fit_bound.py ${CMAKE_BINARY_DIR}/bin/foldsim_fit_bound.py COPYONLY)
install(PROGRAMS foldsim_fit_bound.py DESTINATION bin)
configure_file(foldsim_sv_limits.py ${CMAKE_BINARY_DIR}/bin/foldsim_sv_limits.py COPYONLY)
install(PROGRAMS foldsim_sv_limits.py DESTINATION bin)
# Beacon's routing terms (fanout, SLR crossings): fit on a placed build, score a screen.
configure_file(foldsim_beacon_terms.py ${CMAKE_BINARY_DIR}/bin/foldsim_beacon_terms.py COPYONLY)
install(PROGRAMS foldsim_beacon_terms.py DESTINATION bin)
configure_file(foldsim_v80_timing.json ${CMAKE_BINARY_DIR}/bin/foldsim_v80_timing.json COPYONLY)
install(FILES foldsim_v80_timing.json DESTINATION bin)
# The V80 builds that calibrate the fold degree's area and period
# (foldsim_degree.py calibrate), the default of --copies auto.
configure_file(foldsim_v80.json ${CMAKE_BINARY_DIR}/bin/foldsim_v80.json COPYONLY)
install(FILES foldsim_v80.json DESTINATION bin)
# The V80 builds that calibrate the routability term (foldsim_storage.py fit-route).
configure_file(foldsim_v80_route.csv ${CMAKE_BINARY_DIR}/bin/foldsim_v80_route.csv COPYONLY)
install(FILES foldsim_v80_route.csv DESTINATION bin)
# The V80 compute shell's dynamic region per SLR (foldsim --slr-region v80).
configure_file(foldsim_v80_slr.json ${CMAKE_BINARY_DIR}/bin/foldsim_v80_slr.json COPYONLY)
install(FILES foldsim_v80_slr.json DESTINATION bin)
configure_file(foldsim_v80_asl9_slr.json ${CMAKE_BINARY_DIR}/bin/foldsim_v80_asl9_slr.json COPYONLY)
install(FILES foldsim_v80_asl9_slr.json DESTINATION bin)
# Interface boundary lifting: plain-port wrappers that let interface-port units
# be classes, on a rewritten copy of the sources (needs pyslang).
configure_file(foldsim_lift.py ${CMAKE_BINARY_DIR}/bin/foldsim-lift COPYONLY)
install(PROGRAMS foldsim_lift.py DESTINATION bin RENAME foldsim-lift)
