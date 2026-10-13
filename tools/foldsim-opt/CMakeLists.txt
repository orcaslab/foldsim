add_llvm_executable(foldsim-opt foldsim-opt.cpp)
llvm_update_compile_flags(foldsim-opt)
target_link_libraries(foldsim-opt PRIVATE
  FoldsimTransforms
  CIRCTComb
  CIRCTCombTransforms
  CIRCTHW
  CIRCTHWTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSV
  CIRCTSVTransforms
  CIRCTSupport
  MLIROptLib
  MLIRTransforms
)
install(TARGETS foldsim-opt RUNTIME DESTINATION bin)
