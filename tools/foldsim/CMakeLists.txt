# The driver: elaborates a design with CIRCT and builds its simulator.
configure_file(foldsim.py ${CMAKE_BINARY_DIR}/bin/foldsim COPYONLY)
install(PROGRAMS foldsim.py DESTINATION bin RENAME foldsim)
configure_file(foldsim_degree.py ${CMAKE_BINARY_DIR}/bin/foldsim_degree.py COPYONLY)
configure_file(foldsim_cost.py ${CMAKE_BINARY_DIR}/bin/foldsim_cost.py COPYONLY)
configure_file(foldsim_storage.py ${CMAKE_BINARY_DIR}/bin/foldsim_storage.py COPYONLY)
# Target manifests: a design's whole recipe in one file (--manifest).
configure_file(foldsim_manifest.py ${CMAKE_BINARY_DIR}/bin/foldsim_manifest.py COPYONLY)
install(PROGRAMS foldsim_manifest.py DESTINATION bin)
# The design-independent steps of an image flow from a manifest (boundary
# lifting, the kernel shell's header).
configure_file(foldsim_flow.py ${CMAKE_BINARY_DIR}/bin/foldsim_flow.py COPYONLY)
configure_file(flow/lift_boundaries.py ${CMAKE_BINARY_DIR}/bin/flow/lift_boundaries.py COPYONLY)
install(PROGRAMS foldsim_flow.py DESTINATION bin)
install(PROGRAMS flow/lift_boundaries.py DESTINATION bin/flow)
# A design's simulator gated against its own RTL in Verilator, by its ports.
configure_file(foldsim_gate.py ${CMAKE_BINARY_DIR}/bin/foldsim_gate.py COPYONLY)
install(PROGRAMS foldsim_gate.py DESTINATION bin)
install(PROGRAMS foldsim_degree.py DESTINATION bin)
install(FILES foldsim_cost.py DESTINATION bin)
install(PROGRAMS foldsim_storage.py DESTINATION bin)
# Judges a logic screen's end points with the multicycle paths per host cycle.
configure_file(foldsim_mc_judge.py ${CMAKE_BINARY_DIR}/bin/foldsim_mc_judge.py COPYONLY)
install(PROGRAMS foldsim_mc_judge.py DESTINATION bin)
# The V80 builds that calibrate the fold degree's area and period
# (foldsim_degree.py calibrate), the default of --copies auto.
configure_file(foldsim_v80.json ${CMAKE_BINARY_DIR}/bin/foldsim_v80.json COPYONLY)
install(FILES foldsim_v80.json DESTINATION bin)
# The V80 builds that calibrate the routability term (foldsim_storage.py fit-route).
configure_file(foldsim_v80_route.csv ${CMAKE_BINARY_DIR}/bin/foldsim_v80_route.csv COPYONLY)
install(FILES foldsim_v80_route.csv DESTINATION bin)
