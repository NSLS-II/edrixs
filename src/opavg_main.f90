program opavg_main
    use m_control, only: myid, nprocs
    use mpi 

    implicit none
 
    integer :: ierror

    call MPI_INIT(ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid,   ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierror)
    call MPI_BARRIER(MPI_COMM_WORLD, ierror)

    call config()  
    call opavg_driver()

    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    call MPI_FINALIZE(ierror)

end program opavg_main
