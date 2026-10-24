# Writes OUT, the header that names the foldsim revision (git describe of SRC),
# and rewrites it only when the revision changes, so that a rebuild with no
# change recompiles nothing. A tree outside git is "unknown".
#   cmake -DSRC=<source dir> -DOUT=<header> -P GenerateVersion.cmake
execute_process(
  COMMAND git -C "${SRC}" describe --always --dirty --abbrev=9
  OUTPUT_VARIABLE revision
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE result)
if(NOT result EQUAL 0 OR revision STREQUAL "")
  set(revision "unknown")
endif()
set(text "#define FOLDSIM_REVISION \"${revision}\"\n")
set(old "")
if(EXISTS "${OUT}")
  file(READ "${OUT}" old)
endif()
if(NOT old STREQUAL text)
  file(WRITE "${OUT}" "${text}")
endif()
