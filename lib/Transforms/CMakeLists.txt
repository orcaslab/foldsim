add_mlir_library(FoldsimTransforms
  Beacon.cpp
  BlackBoxes.cpp
  Compose.cpp
  EstimateTiming.cpp
  ExtractModels.cpp
  FAME5.cpp
  FindPipelines.cpp
  FuseModels.cpp
  Pipelines.cpp
  Sort.cpp
  StaticFold.cpp

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
