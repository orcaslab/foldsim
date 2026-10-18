add_mlir_library(FoldsimTransforms
  Compose.cpp
  ExtractModels.cpp
  FAME5.cpp
  FindPipelines.cpp
  FuseModels.cpp
  Pipelines.cpp

  DEPENDS
  FoldsimPassesIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTHWTransforms
  CIRCTSeq
  CIRCTSV
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransforms
)
