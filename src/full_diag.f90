subroutine full_diag_ham(ndim, ham, eval, evec)
    use m_constants, only : dp, zero

    implicit none

    integer,     intent(in)  :: ndim
    complex(dp), intent(in)  :: ham(ndim, ndim)
    real(dp),    intent(out) :: eval(ndim)
    complex(dp), intent(out) :: evec(ndim,ndim)

    integer :: info
    integer :: lwork
    integer :: lrwork
    real(dp), allocatable    :: rwork(:)
    complex(dp), allocatable :: work(:)

    lwork = 2 * ndim - 1
    lrwork = 3 * ndim - 2

    allocate(work(lwork))
    allocate(rwork(lrwork))

    eval = zero
    evec = ham

    call ZHEEV('V', 'U', ndim, evec, ndim, eval, work, lwork, rwork, info)

    if ( allocated(work ) ) deallocate(work )
    if ( allocated(rwork) ) deallocate(rwork)

    return
end subroutine full_diag_ham

