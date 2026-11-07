add_llvm_executable(foldsim-opt foldsim-opt.cpp)
llvm_update_compile_flags(foldsim-opt)
add_dependencies(foldsim-opt foldsim-version)
target_link_libraries(foldsim-opt PRIVATE
  FoldsimTransforms
  CIRCTComb
  CIRCTCombToSynth
  CIRCTCombTransforms
  CIRCTDatapath
  CIRCTHW
  CIRCTHWTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSV
  CIRCTSVTransforms
  CIRCTSupport
  CIRCTSynth
  CIRCTSynthTransforms
  MLIROptLib
  MLIRTransforms
)
install(TARGETS foldsim-opt RUNTIME DESTINATION bin)
