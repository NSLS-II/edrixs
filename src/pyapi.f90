subroutine ed_fsolver(my_id, num_procs)
    use m_control, only: myid, nprocs

    implicit none

    integer, intent(in) :: my_id
    integer, intent(in) :: num_procs

!F2PY intent(in) my_id
!F2PY intent(in) num_procs
    myid = my_id
    nprocs = num_procs

    call config()  
    call ed_driver()

    return
end subroutine ed_fsolver

subroutine xas_fsolver(my_id, num_procs)
    use m_control, only: myid, nprocs

    implicit none

    integer, intent(in) :: my_id
    integer, intent(in) :: num_procs

!F2PY intent(in) my_id
!F2PY intent(in) num_procs
    myid = my_id
    nprocs = num_procs

    call config()  
    call xas_driver()

    return
end subroutine xas_fsolver

subroutine rixs_fsolver(my_id, num_procs)
    use m_control, only: myid, nprocs

    implicit none

    integer, intent(in) :: my_id
    integer, intent(in) :: num_procs

!F2PY intent(in) my_id
!F2PY intent(in) num_procs
    myid = my_id
    nprocs = num_procs

    call config()  
    call rixs_driver()

    return
end subroutine rixs_fsolver

subroutine opavg_fsolver(my_id, num_procs)
    use m_control, only: myid, nprocs

    implicit none

    integer, intent(in) :: my_id
    integer, intent(in) :: num_procs

!F2PY intent(in) my_id
!F2PY intent(in) num_procs
    myid = my_id
    nprocs = num_procs

    call config()  
    call opavg_driver()

    return
end subroutine opavg_fsolver
