cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project (EDRIXS Fortran)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(MPI REQUIRED)

find_package(LAPACK REQUIRED)

# see https://gitlab.kitware.com/cmake/cmake/-/issues/21779
if (CMAKE_VERSION VERSION_LESS 3.20)
  if(LAPACK_FOUND)
    set(_lapack_libs "${LAPACK_LIBRARIES}")
    if(_lapack_libs AND TARGET BLAS::BLAS)
      # remove the ${BLAS_LIBRARIES} from the interface and replace it
      # with the BLAS::BLAS target
      list(REMOVE_ITEM _lapack_libs "${BLAS_LIBRARIES}")
      list(APPEND _lapack_libs BLAS::BLAS)
    endif()
    if(_lapack_libs)
      set_target_properties(LAPACK::LAPACK PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_lapack_libs}"
      )
    endif()
    unset(_lapack_libs)
  endif()
endif()

# use a custom find module because arpack-ng config module doesn't provide full paths to libs
# see https://github.com/opencollab/arpack-ng/pull/311
find_package(arpack MODULE REQUIRED COMPONENTS serial parallel)

add_subdirectory(src)
