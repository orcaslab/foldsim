# The driver: elaborates a design with CIRCT and builds its simulator.
configure_file(foldsim.py ${CMAKE_BINARY_DIR}/bin/foldsim COPYONLY)
install(PROGRAMS foldsim.py DESTINATION bin RENAME foldsim)
configure_file(foldsim_degree.py ${CMAKE_BINARY_DIR}/bin/foldsim_degree.py COPYONLY)
configure_file(foldsim_cost.py ${CMAKE_BINARY_DIR}/bin/foldsim_cost.py COPYONLY)
install(PROGRAMS foldsim_degree.py DESTINATION bin)
install(FILES foldsim_cost.py DESTINATION bin)
# The V80 builds that calibrate the fold degree's area and period
# (foldsim_degree.py calibrate), the default of --copies auto.
configure_file(foldsim_v80.json ${CMAKE_BINARY_DIR}/bin/foldsim_v80.json COPYONLY)
install(FILES foldsim_v80.json DESTINATION bin)
