!! Use parallel arpack to find a few lowest eigenstates of a large sparse Hamiltonian
subroutine diag_ham_arpack(nblock, end_indx, needed, nloc, nev, ham, eval, &
                   n_vecs, evec, ncv, maxiter, tol, info, actual_step, nconv)

    use m_constants, only: dp, czero
    use m_control,   only: nprocs, new_comm
    use m_types
    use mpi

    implicit none

    ! external variables
    integer, intent(in)       :: nblock
    integer, intent(in)       :: end_indx(2,2,nprocs)
    integer, intent(in)       :: needed(nprocs, nprocs)
    integer, intent(in)       :: nloc
    integer, intent(in)       :: nev
    type (T_csr)              :: ham(nblock)
    real(dp), intent(out)     :: eval(nev)
    integer, intent(in)       :: n_vecs
    complex(dp), intent(out)  :: evec(nloc,n_vecs)
    integer, intent(in)       :: ncv
    integer, intent(in)       :: maxiter
    real(dp), intent(in)      :: tol
    integer, intent(out)      :: info
    integer, intent(out)      :: actual_step
    integer, intent(out)      :: nconv

    ! local variables
    integer      :: ido
    integer      :: ishfts
    integer      :: mode
    integer      :: lworkl
    integer      :: iparam(11)
    integer      :: ipntr(14)
    integer      :: sorted_indx(nev)
    integer      :: i

    logical      :: rvec
    character(1) :: bmat
    character(2) :: which
    complex(dp)  :: sigma

    real(dp)   , allocatable  :: rwork(:)
    logical    , allocatable  :: selec(:)
    complex(dp), allocatable  :: resid(:)
    complex(dp), allocatable  :: v(:,:)
    complex(dp), allocatable  :: workd(:)
    complex(dp), allocatable  :: workl(:)
    complex(dp), allocatable  :: workev(:)
    complex(dp), allocatable  :: d(:)

    lworkl  = 3*ncv**2 + 5*ncv 

    allocate(rwork(ncv))
    allocate(selec(ncv))
    allocate(resid(nloc))
    allocate(v(nloc,ncv))
    allocate(workd(3*nloc))
    allocate(workl(lworkl))
    allocate(workev(3*ncv))
    allocate(d(ncv))

    ishfts    = 1
    mode      = 1
    iparam(1) = ishfts
    iparam(3) = maxiter 
    iparam(7) = mode 
    bmat      = 'I'
    which     = 'SR'
    ido       = 0
    info      = 0
    selec     = .true.

    do while (.true.)
        call pznaupd(new_comm, ido, bmat, nloc, which, nev, tol, resid, ncv, &
                     v, nloc, iparam, ipntr, workd, workl, lworkl, rwork, info)

        if (ido .eq. -1 .or. ido .eq. 1) then
            workd(ipntr(2):ipntr(2)+nloc-1) = czero
            call pspmv_csr(new_comm, nblock, end_indx, needed, nloc, nloc, ham, &
                                                     workd(ipntr(1)), workd(ipntr(2)))
        else
            EXIT
        endif
    enddo
    actual_step = iparam(3)    
    nconv = iparam(5)    

    if (info .lt. 0) then
        return
    else 
        rvec = .true.
        call pzneupd(new_comm, rvec, 'A', selec, d, v, nloc, sigma, workev, &
                          bmat, nloc, which, nev, tol, resid, ncv, v, nloc, iparam, &
                          ipntr, workd, workl, lworkl, rwork, info)
        if (info .ne. 0) then
            return
        endif
        ! sort eigenvalues to ascending order
        eval = real(d(1:nev))        
        do i=1,nev
            sorted_indx(i) = i
        enddo
        call sort_eigvals(nev, eval, sorted_indx)
        do i=1,n_vecs
            evec(:,i) = v(:,sorted_indx(i))
        enddo
    endif

    return
end subroutine diag_ham_arpack
