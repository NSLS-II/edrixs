module m_lanczos

    contains
   
    !! use parallel Lanczos method to find the ground state
    subroutine diag_ham_lanczos(nblock, end_indx, needed, nloc, nev, max_niter, eval_tol, ham, eval, n_vec, evec)
        use m_constants, only: dp, zero, czero, mystd
        use m_control, only: nprocs , myid, master, new_comm
        use m_types
        use mpi
    
        implicit none
    
        ! external variables
        integer, intent(in)       :: nblock
        integer, intent(in)       :: end_indx(2,2,nprocs)
        integer, intent(in)       :: needed(nprocs, nprocs)
        integer, intent(in)       :: nloc
        integer, intent(in)       :: nev
        integer, intent(in)       :: max_niter
        real(dp), intent(in)      :: eval_tol
        type (T_csr)              :: ham(nblock)
        real(dp), intent(out)     :: eval(nev)
        integer, intent(in)       :: n_vec
        complex(dp), intent(out)  :: evec(nloc,n_vec)
    
        ! local variables
        real(dp),    allocatable :: temp_vec(:)
        complex(dp), allocatable :: work_vec(:,:)
        real(dp),    allocatable :: alpha(:)
        real(dp),    allocatable :: beta(:)
        real(dp),    allocatable :: krylov_eigval(:)
        real(dp),    allocatable :: krylov_eigvec(:,:)
    
        integer :: curr
        integer :: prev
        integer :: temp_indx
        integer :: i,j
        integer :: nkrylov
        integer :: ierror
    
        real(dp) :: res
    
        complex(dp) :: temp1
        complex(dp) :: temp1_mpi
    
        allocate(temp_vec(nloc))
        allocate(work_vec(nloc,2))
        allocate(alpha(max_niter))
        allocate(beta(0:max_niter))
        allocate(krylov_eigval(max_niter))
        allocate(krylov_eigvec(max_niter,max_niter))
    
        temp_vec = zero
        eval     = zero
        work_vec = czero
        alpha    = zero
        beta     = zero
    
        call random_seed()
        call random_number(temp_vec) 
        work_vec(:,1) = temp_vec - 0.5
        if (allocated(temp_vec)) deallocate(temp_vec)
    
        ! normalize v1
        temp1 = czero 
        temp1_mpi = czero 
        temp1 = dot_product(work_vec(:,1), work_vec(:,1))
        call MPI_ALLREDUCE(temp1, temp1_mpi, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, new_comm, ierror)
        temp1 = sqrt(real(temp1_mpi))
        work_vec(:,1) = work_vec(:,1) / temp1
        evec(:,1) = work_vec(:,1)
    
        curr = 1
        prev = 2
        nkrylov = 0
    
        do i=1,max_niter
            ! matrix-vector product
            call MPI_BARRIER(new_comm, ierror)
            work_vec(:,prev) = -beta(i-1) * work_vec(:,prev)
            call pspmv_csr(new_comm, nblock, end_indx, needed, nloc, nloc, ham, work_vec(:,curr), work_vec(:,prev))
            call MPI_BARRIER(new_comm, ierror)
            ! compute alpha_i
            temp1 = czero 
            temp1_mpi = czero 
            temp1 = dot_product(work_vec(:,curr), work_vec(:,prev))
            call MPI_ALLREDUCE(temp1, temp1_mpi, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, new_comm, ierror)
            alpha(i) = real(temp1_mpi)
    
            if (mod(i,10) == 0 .or. i==max_niter) then
                krylov_eigval = zero
                krylov_eigvec = zero
                call krylov_eigs(i, alpha(1:i), beta(1:i-1), krylov_eigval(1:i), krylov_eigvec(1:i,1:i))
                ! check convergence
                if (i>=nev) then
                    res = krylov_eigval(nev) - eval(nev)
                    if (myid==master) then
                        write(mystd,"(a25, i5, E10.2, a5, E10.2)")  "Lanczos iteration:  ", i, abs(res), "-->", eval_tol
                    endif
                    if (abs(res) < eval_tol) then 
                        eval(1:nev) = krylov_eigval(1:nev)
                        nkrylov = i
                        EXIT
                    endif
                endif
                eval(1:nev) = krylov_eigval(1:nev)
            endif
    
            work_vec(:,prev) = work_vec(:,prev) - alpha(i) * work_vec(:,curr)
            ! compute beta_{i+1}
            temp1 = czero 
            temp1_mpi = czero 
            temp1 = dot_product(work_vec(:,prev), work_vec(:,prev))
            call MPI_ALLREDUCE(temp1, temp1_mpi, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, new_comm, ierror)
            beta(i) = sqrt(real(temp1_mpi))
            ! beta is too small, exit
            if (abs(beta(i)) < 1E-10) then
                krylov_eigval = zero
                krylov_eigvec = zero
                if (i>=2) then
                    call krylov_eigs(i, alpha(1:i), beta(1:i-1), krylov_eigval(1:i), krylov_eigvec(1:i,1:i))
                    eval(1:nev) = krylov_eigval(1:nev)
                else
                    eval(1) = alpha(1)
                    krylov_eigvec(1,1) = 1.0_dp
                endif
                nkrylov = i
                EXIT
            endif
            work_vec(:,prev) = work_vec(:,prev) / beta(i)
            temp_indx = curr
            curr = prev
            prev = temp_indx     
        enddo
    
        ! build eigenvectors if wanted
        work_vec = czero
        work_vec(:,1) = evec(:,1)
        evec  = czero
        curr  = 1
        prev  = 2
        do i=1,nkrylov
            do j=1,n_vec
                evec(:,j) = evec(:,j) + krylov_eigvec(i,j) * work_vec(:,curr)
            enddo 
            if (i<nkrylov) then
                call MPI_BARRIER(new_comm, ierror)
                work_vec(:,prev) = -alpha(i) * work_vec(:,curr) - beta(i-1) * work_vec(:,prev)
                call pspmv_csr(new_comm, nblock, end_indx, needed, nloc, nloc, ham, work_vec(:,curr), work_vec(:,prev))
                call MPI_BARRIER(new_comm, ierror)
                work_vec(:,prev) = work_vec(:,prev) / beta(i)
                temp_indx = curr
                curr = prev
                prev = temp_indx     
            endif
        enddo 

    
        if (allocated(work_vec))        deallocate(work_vec)
        if (allocated(alpha))           deallocate(alpha)
        if (allocated(beta))            deallocate(beta)
        if (allocated(krylov_eigval))   deallocate(krylov_eigval)
        if (allocated(krylov_eigvec))   deallocate(krylov_eigvec)
    
        return
    end subroutine diag_ham_lanczos
    
    subroutine krylov_eigs(n, alpha, beta, eval, evec)
        use m_constants, only: dp, zero
        implicit none
    
        ! external variables
        integer, intent(in) :: n
        real(dp), intent(in) :: alpha(n)
        real(dp), intent(in) :: beta(n-1)
        real(dp), intent(out) :: eval(n)
        real(dp), intent(out) :: evec(n,n)
    
        ! local variables
        real(dp), allocatable :: tmp_beta(:)
        real(dp), allocatable :: work(:)
        integer :: i
        integer :: info
        
        allocate(tmp_beta(n-1))
        allocate(work(max(1,2*n-2)))
        eval = alpha
        tmp_beta = beta
        work = zero
        ! init evec to identity matrix
        evec = zero
        do i=1,n
            evec(i,i) = 1.0_dp
        enddo    
        call dsteqr('I', n, eval, tmp_beta, evec, n, work, info) 
        if (info /= 0 ) then
            print *, "error in dster, info: ", info
            STOP
        endif
    
        if (allocated(tmp_beta)) deallocate(tmp_beta)
        if (allocated(work))     deallocate(work)
    
        return
    end subroutine krylov_eigs

    !! given a Hamiltonian H, an initial vector V1, build its Krylov space with desired dimension
    !! return the actual dimension of the Krylov space, the normalization parameter of V0, alpha and beta 
    subroutine build_krylov_mp(nblock, end_indx, needed, nloc, ham, init_vec, maxn, neff, alpha, beta, norm)
        use m_constants, only: dp, zero, czero, mystd
        use m_control, only: nprocs, myid, master, new_comm
        use m_types
        use mpi
    
        implicit none
    
        ! external variables
        integer, intent(in)      :: nblock
        integer, intent(in)      :: end_indx(2,2,nprocs)
        integer, intent(in)      :: needed(nprocs, nprocs)
        integer, intent(in)      :: nloc
        type (T_csr)             :: ham(nblock)
        complex(dp), intent(in)  :: init_vec(nloc) 
        integer, intent(in)      :: maxn
        integer, intent(out)     :: neff
        real(dp), intent(out)    :: alpha(maxn)
        real(dp), intent(out)    :: beta(0:maxn)
        real(dp), intent(out)    :: norm
     
        ! local variables
        complex(dp), allocatable :: work_vec(:,:)
    
        integer :: curr
        integer :: prev
        integer :: temp_indx
        integer :: i
        integer :: ierror

    
        complex(dp) :: temp1
        complex(dp) :: temp1_mpi
    
        allocate(work_vec(nloc,2))
    
        work_vec = czero
        alpha    = zero
        beta     = zero
        neff     = 0
        norm     = zero
    
        work_vec(:,1) = init_vec
        ! normalize V1
        temp1 = czero 
        temp1_mpi = czero 
        temp1 = dot_product(work_vec(:,1), work_vec(:,1))
        call MPI_ALLREDUCE(temp1, temp1_mpi, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, new_comm, ierror)
        norm = real(temp1_mpi)
        temp1 = sqrt(real(temp1_mpi))
        work_vec(:,1) = work_vec(:,1) / temp1
    
        curr = 1
        prev = 2
    
        ! begin iteration
        do i=1,maxn
            if (myid == master .and. mod(i,50)==0) then
                write(mystd,"(a24, i5)")  "Krylov iteration:  ", i
            endif
            neff = i
            ! matrix-vector product
            call MPI_BARRIER(new_comm, ierror)
            work_vec(:,prev) = -beta(i-1) * work_vec(:,prev)
            call pspmv_csr(new_comm, nblock, end_indx, needed, nloc, nloc, ham, work_vec(:,curr), work_vec(:,prev))
            call MPI_BARRIER(new_comm, ierror)
            ! compute alpha_i
            temp1 = czero 
            temp1_mpi = czero 
            temp1 = dot_product(work_vec(:,curr), work_vec(:,prev))
            call MPI_ALLREDUCE(temp1, temp1_mpi, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, new_comm, ierror)
            alpha(i) = real(temp1_mpi)
    
            work_vec(:,prev) = work_vec(:,prev) - alpha(i) * work_vec(:,curr)
            ! compute beta_{i+1}
            temp1 = czero 
            temp1_mpi = czero 
            temp1 = dot_product(work_vec(:,prev), work_vec(:,prev))
            call MPI_ALLREDUCE(temp1, temp1_mpi, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, new_comm, ierror)
            beta(i) = sqrt(real(temp1_mpi))
            ! beta is too small, exit
            if (abs(beta(i)) < 1E-10) then
                EXIT
            endif
            work_vec(:,prev) = work_vec(:,prev) / beta(i)
            temp_indx = curr
            curr = prev
            prev = temp_indx     
        enddo

        if (allocated(work_vec)) deallocate(work_vec)
    
        return
    end subroutine build_krylov_mp

    ! given alpha, beta and normalization parameter to build the spectrum function
    subroutine build_spectrum(nkrylov, alpha, beta, norm, nw, om_mesh, e_gs, gamma_final, spec)
        use m_constants, only: dp, czero, pi
        
        implicit none

        ! external variables
        integer, intent(in) :: nkrylov
        real(dp), intent(in) :: alpha(nkrylov)
        real(dp), intent(in) :: beta(0:nkrylov)
        real(dp), intent(in) :: norm
        integer,  intent(in) :: nw
        real(dp), intent(in) :: om_mesh(nw)
        real(dp), intent(in) :: e_gs
        real(dp), intent(in) :: gamma_final(nw)
        real(dp), intent(out) :: spec(nw)

        ! local variables
        integer :: i
        complex(dp) :: tmp_vec(nw)        
        ! loop over krylov space

        tmp_vec = czero
        do i=nkrylov, 2, -1
            tmp_vec = beta(i-1)**2 / (om_mesh + dcmplx(0.0, 1.0)*gamma_final + e_gs - alpha(i) - tmp_vec)
        enddo
        tmp_vec = 1.0/(om_mesh + dcmplx(0.0,1.0) * gamma_final + e_gs - alpha(1) - tmp_vec)
        
        spec = -1.0/pi * aimag(tmp_vec) * norm

        return
    end subroutine build_spectrum

end module m_lanczos
