# The driver: elaborates a design with CIRCT and builds its simulator.
configure_file(foldsim.py ${CMAKE_BINARY_DIR}/bin/foldsim COPYONLY)
install(PROGRAMS foldsim.py DESTINATION bin RENAME foldsim)
# The fold-degree search (--copies auto): prices each class with foldsim-opt's
# foldsim-estimate-area, the one area model.
configure_file(foldsim_degree.py ${CMAKE_BINARY_DIR}/bin/foldsim_degree.py COPYONLY)
install(PROGRAMS foldsim_degree.py DESTINATION bin)
# The V80 builds that calibrate the fold degree's area and period
# (foldsim_degree.py calibrate), the default of --copies auto.
configure_file(foldsim_v80.json ${CMAKE_BINARY_DIR}/bin/foldsim_v80.json COPYONLY)
install(FILES foldsim_v80.json DESTINATION bin)
# Interface boundary lifting: plain-port wrappers that let interface-port units
# be classes, on a rewritten copy of the sources (needs pyslang).
configure_file(foldsim_lift.py ${CMAKE_BINARY_DIR}/bin/foldsim-lift COPYONLY)
install(PROGRAMS foldsim_lift.py DESTINATION bin RENAME foldsim-lift)
