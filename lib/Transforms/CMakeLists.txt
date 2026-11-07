add_mlir_library(FoldsimTransforms
  Beacon.cpp
  BlackBoxes.cpp
  Compose.cpp
  EstimateArea.cpp
  EstimateTiming.cpp
  ExtractModels.cpp
  FAME5.cpp
  FindPipelines.cpp
  MemoriesToRegisters.cpp
  FuseModels.cpp
  LowerWideNegation.cpp
  Pipelines.cpp
  RemoveDeadState.cpp
  ReplicateControls.cpp
  Sort.cpp
  PassSplit.cpp
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
  CIRCTSynth
  MLIRIR
  MLIRPass
  MLIRTransforms
)
