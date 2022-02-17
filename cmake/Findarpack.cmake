include(FindPackageHandleStandardArgs)

message(STATUS "Finding arpack")

# if components not specified default to serial library
set(_supported_components serial parallel)
if(NOT arpack_FIND_COMPONENTS)
  set(arpack_FIND_COMPONENTS serial)
endif()

# check request components
foreach(_component ${arpack_FIND_COMPONENTS})
  if (NOT ${_component} IN_LIST _supported_components)
    message(FATAL_ERROR "${_component} is not a valid component (serial,parallel)")
  endif()
endforeach()

message(STATUS "${arpack_FIND_COMPONENTS}")

if(DEFINED ENV{ARPACK_ROOT})
  set(ARPACK_ROOT "$ENV{ARPACK_ROOT}")
endif()

set(arpack_serial_FOUND FALSE)
set(arpack_parallel_FOUND FALSE)
set(arpack_LIBRARIES "")

if ("serial" IN_LIST arpack_FIND_COMPONENTS)
  find_library(ARPACK_LIBRARY NAMES arpack HINTS ${ARPACK_ROOT})
  if(ARPACK_LIBRARY)
    set(arpack_serial_FOUND TRUE)
    message(STATUS "ARPACK_LIBRARY: ${ARPACK_LIBRARY}")
    add_library(ARPACK::ARPACK INTERFACE IMPORTED)
    set_target_properties(ARPACK::ARPACK PROPERTIES INTERFACE_LINK_LIBRARIES "${ARPACK_LIBRARY}")
    list(APPEND arpack_LIBRARIES ${ARPACK_LIBRARY})
  endif()
endif()

if ("parallel" IN_LIST arpack_FIND_COMPONENTS)
  find_library(PARPACK_LIBRARY NAMES parpack HINTS ${ARPACK_ROOT})
  if(PARPACK_LIBRARY)
    set(arpack_parallel_FOUND TRUE)
    message(STATUS "PARPACK_LIBRARY: ${PARPACK_LIBRARY}")
    add_library(ARPACK::PARPACK INTERFACE IMPORTED)
    set_target_properties(ARPACK::PARPACK PROPERTIES INTERFACE_LINK_LIBRARIES "${PARPACK_LIBRARY}")
    list(APPEND arpack_LIBRARIES ${PARPACK_LIBRARY})
  endif()
endif()

find_package_handle_standard_args(arpack REQUIRED_VARS "arpack_LIBRARIES" HANDLE_COMPONENTS)
