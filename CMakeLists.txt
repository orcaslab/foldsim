cmake_minimum_required(VERSION 3.20)
project(foldsim LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# foldsim builds on an installed (or built) CIRCT, as POCL builds on LLVM:
#   cmake -G Ninja -B build -DCIRCT_DIR=<circt>/lib/cmake/circt \
#     -DMLIR_DIR=<circt>/lib/cmake/mlir -DLLVM_DIR=<circt>/lib/cmake/llvm
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CIRCT_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# The revision foldsim-opt --version prints, refreshed on every build.
set(FOLDSIM_VERSION_INC ${PROJECT_BINARY_DIR}/include/foldsim/Version.inc)
add_custom_target(foldsim-version
  COMMAND ${CMAKE_COMMAND} -DSRC=${PROJECT_SOURCE_DIR}
          -DOUT=${FOLDSIM_VERSION_INC}
          -P ${PROJECT_SOURCE_DIR}/cmake/GenerateVersion.cmake
  BYPRODUCTS ${FOLDSIM_VERSION_INC}
  COMMENT "Checking the foldsim revision")

# Beacon (orcaslab/beacon, a submodule): its passes map a class's model for
# depth (stage-levels-mapped) and area (--degree-estimator beacon-area).
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/beacon/CMakeLists.txt)
  message(FATAL_ERROR "third_party/beacon is missing: git submodule update --init")
endif()
add_subdirectory(third_party/beacon)
include_directories(${PROJECT_SOURCE_DIR}/third_party/beacon/include
                    ${PROJECT_BINARY_DIR}/third_party/beacon/include)

add_subdirectory(include/foldsim)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
add_subdirectory(shell)
