set(src
  m_constants.f90
  m_control.f90
  m_types.f90
  m_global.f90
  utils.f90
  io.f90
  fock.f90
  spmv.f90
  linsys.f90
  ham.f90
  full_diag.f90
  lanczos.f90
  arpack.f90
  ed_driver.f90
  xas_driver.f90
  rixs_driver.f90
  opavg_driver.f90)

add_library(edrixs ${src})
target_link_libraries(edrixs PUBLIC MPI::MPI_Fortran LAPACK::LAPACK ARPACK::PARPACK)

install(TARGETS edrixs DESTINATION lib)

set(progs ed xas rixs opavg)
foreach(prog ${progs})
  add_executable(${prog}.x ${prog}_main.f90)
  target_link_libraries(${prog}.x PUBLIC edrixs)
  install(TARGETS ${prog}.x DESTINATION bin)
endforeach(prog)
