# The driver: elaborates a design with CIRCT and builds its simulator.
configure_file(foldsim.py ${CMAKE_BINARY_DIR}/bin/foldsim COPYONLY)
install(PROGRAMS foldsim.py DESTINATION bin RENAME foldsim)
