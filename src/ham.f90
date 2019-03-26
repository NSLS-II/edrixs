!! Give hopping, Coulomb interaction terms, and the Fock basis, build many particle Hamiltonian
subroutine build_ham_i(ncfgs, fock, nblock, end_indx, nhopp, hopping, &
                       ncoul, coulomb, omega, ham_sign, ham_csr)
    use m_constants, only: dp
    use m_control, only: myid, master, nprocs
    use m_types
    use mpi

    implicit none

    ! external variables
    ! the dimension of this Hilbert space
    integer, intent(in) :: ncfgs
    ! the fock basis of this Hilbert space
    integer(dp), intent(in) :: fock(ncfgs)
    ! number of blocks of this Hamiltonian divided by cpu cores
    integer, intent(in) :: nblock
    ! end index of the Fock basis for each cpu core
    integer, intent(in) :: end_indx(2,2,nprocs)
    ! number of nonzero hopping terms
    integer, intent(in) :: nhopp
    ! the hopping terms
    type (T_tensor2), intent(in) :: hopping(nhopp)
    ! number of nonzero interaction terms
    integer, intent(in) :: ncoul
    ! the interaction terms
    type (T_tensor4), intent(in) :: coulomb(ncoul)
    ! omega for intermediate states
    complex(dp), intent(in) :: omega
    ! the sign of Hamiltonian
    real(dp), intent(in) :: ham_sign
    ! the Hamiltonian needed
    type (T_csr) :: ham_csr(nblock)

    ! local variables
    integer, external :: binary_search
    integer(dp) :: old, new
    integer :: icfg, jcfg
    integer :: alpha,beta,gama,delta
    integer :: tot_sign
    integer :: sgn
    integer :: i,j
    integer :: which_block
    integer :: row_shift
    integer :: begin_col

    integer, allocatable :: num_of_cols(:,:)
    integer :: num_of_tot(nblock)

    type (T_col), pointer :: next_col, curr_col
    type (T_container_col) :: cols(nblock)

    which_block = 0
    row_shift = end_indx(1,1,myid+1) - 1 
    allocate(num_of_cols(end_indx(2,1,myid+1)-end_indx(1,1,myid+1)+1, nblock))
    num_of_cols = 0
    num_of_tot = 0

    do icfg=end_indx(1,1,myid+1), end_indx(2,1,myid+1)
        do i=1,nblock
            cols(i)%col_ptr => null()
        enddo
        do i=1,nhopp
            alpha = hopping(i)%ind1
            beta  = hopping(i)%ind2

            if (.not. btest(fock(icfg), alpha-1)) cycle
            old = fock(icfg)

            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn) 
            tot_sign = tot_sign * sgn
            old = new

            if (btest(old, beta-1)) cycle
            call make_newfock('+', beta,  old, new, sgn) 
            tot_sign = tot_sign * sgn

            jcfg = binary_search(ncfgs, fock, new)
            
            if (jcfg == -1) cycle

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            ! init one row
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, hopping(i)%val * tot_sign * ham_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, hopping(i)%val * tot_sign * ham_sign)
            endif
        enddo

        do i=1,ncoul
            alpha = coulomb(i)%ind1
            beta  = coulomb(i)%ind2
            gama  = coulomb(i)%ind3
            delta = coulomb(i)%ind4

            if (.not. btest(fock(icfg), alpha-1)) cycle
            old = fock(icfg)
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new

            if (.not. btest(old, beta-1)) cycle
            call make_newfock('-', beta, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new

            if (btest(old, gama-1)) cycle
            call make_newfock('+', gama, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new

            if (btest(old, delta-1)) cycle
            call make_newfock('+', delta, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new

            jcfg = binary_search(ncfgs, fock, new)
            if (jcfg == -1) cycle

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, coulomb(i)%val * tot_sign * ham_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, coulomb(i)%val * tot_sign * ham_sign)
            endif
        enddo

        ! diagonal elements: omega
        if (ham_sign < 0) then
            jcfg = icfg
            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif
            
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, omega)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, omega) 
            endif
        endif

        do i=1,nblock
            next_col => cols(i)%col_ptr
            do while (associated(next_col)) 
                num_of_cols(icfg-row_shift, i) = num_of_cols(icfg-row_shift, i) + 1 
                curr_col => next_col
                next_col => next_col%next
                deallocate(curr_col)
            enddo 
            num_of_tot(i) = num_of_tot(i) + num_of_cols(icfg-row_shift, i)
        enddo 
    enddo

    if (myid==master) then
        print *, "    Allocate memory for ham_csr ..."
    endif
    ! change to csr format
    do i=1,nblock
        if (num_of_tot(i) /= 0) then
            ham_csr(i)%m = end_indx(2,1,myid+1) - end_indx(1,1,myid+1) + 1
            ham_csr(i)%nnz = num_of_tot(i)
            ham_csr(i)%row_shift = end_indx(1,1,myid+1) - 1 
            ham_csr(i)%col_shift = end_indx(1,2,i) - 1 
            call alloc_csr(ham_csr(i)%m, ham_csr(i)%nnz, ham_csr(i))
            ham_csr(i)%iaa(1) = 1
            do j=2,ham_csr(i)%m+1
                ham_csr(i)%iaa(j) = ham_csr(i)%iaa(j-1) + num_of_cols(j-1, i)
            enddo
        endif
    enddo
    deallocate(num_of_cols)

    ! really building Hamiltonian here
    if (myid==master) then
        print *, "    Really building Hamiltonian begin here ..."
    endif
    do icfg=end_indx(1,1,myid+1), end_indx(2,1,myid+1)
        do i=1,nblock
            cols(i)%col_ptr => null()
        enddo
        do i=1,nhopp
            alpha = hopping(i)%ind1
            beta  = hopping(i)%ind2

            if (.not. btest(fock(icfg), alpha-1)) cycle
            old = fock(icfg)

            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn) 
            tot_sign = tot_sign * sgn
            old = new

            if (btest(old, beta-1)) cycle
            call make_newfock('+', beta,  old, new, sgn) 
            tot_sign = tot_sign * sgn

            jcfg = binary_search(ncfgs, fock, new)
            
            if (jcfg == -1) cycle

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            ! init one row
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, hopping(i)%val * tot_sign * ham_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, hopping(i)%val * tot_sign * ham_sign)
            endif
        enddo

        do i=1,ncoul
            alpha = coulomb(i)%ind1
            beta  = coulomb(i)%ind2
            gama  = coulomb(i)%ind3
            delta = coulomb(i)%ind4

            if (.not. btest(fock(icfg), alpha-1)) cycle
            old = fock(icfg)
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new

            if (.not. btest(old, beta-1)) cycle
            call make_newfock('-', beta, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new

            if (btest(old, gama-1)) cycle
            call make_newfock('+', gama, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new

            if (btest(old, delta-1)) cycle
            call make_newfock('+', delta, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new

            jcfg = binary_search(ncfgs, fock, new)
            if (jcfg == -1) cycle

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, coulomb(i)%val * tot_sign * ham_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, coulomb(i)%val * tot_sign * ham_sign)
            endif
        enddo

        ! diagonal elements: omega
        if (ham_sign < 0) then
            jcfg = icfg
            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif
            
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, omega)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, omega) 
            endif
        endif

        ! copy to T_csr
        do i=1,nblock
            if (num_of_tot(i) == 0) cycle
            begin_col = ham_csr(i)%iaa(icfg-row_shift) - 1
            next_col => cols(i)%col_ptr
            do while (associated(next_col)) 
                begin_col = begin_col + 1
                ham_csr(i)%jaa(begin_col) = next_col%col
                ham_csr(i)%aa(begin_col)  = next_col%val
                curr_col => next_col
                next_col => next_col%next
                deallocate(curr_col)
            enddo 
        enddo 
    enddo

    return
end subroutine build_ham_i

!! Give hopping, Coulomb interaction terms, and Fock basis with a core-hole, build many particle Hamiltonian
subroutine build_ham_n(ncfgs, fock, num_val_orbs, num_core_orbs, nblock, &
       end_indx, nhopp, hopping, ncoul, coulomb, omega, ham_sign, ham_csr)
    use m_constants, only: dp
    use m_control, only: myid, master, nprocs
    use m_types
    use mpi

    implicit none

    ! external variables
    ! the dimension of this Hilbert space
    integer, intent(in) :: ncfgs
    ! the fock basis of this Hilbert space
    integer(dp), intent(in) :: fock(ncfgs)
    ! number of valence orbitals
    integer, intent(in) :: num_val_orbs 
    ! number of core orbitals
    integer, intent(in) :: num_core_orbs 
    ! number of blocks of this Hamiltonian divided by cpu cores
    integer, intent(in) :: nblock
    ! end index of the Fock basis for each cpu core
    integer, intent(in) :: end_indx(2,2,nprocs)
    ! number of nonzero hopping terms
    integer, intent(in) :: nhopp
    ! the hopping terms
    type (T_tensor2), intent(in) :: hopping(nhopp)
    ! number of nonzero interaction terms
    integer, intent(in) :: ncoul
    ! the interaction terms
    type (T_tensor4), intent(in) :: coulomb(ncoul)
    ! omega for intermediate states
    complex(dp), intent(in) :: omega
    ! the sign of Hamiltonian
    real(dp), intent(in) :: ham_sign
    ! the Hamiltonian needed
    type (T_csr) :: ham_csr(nblock)

    ! local variables
    integer, external :: binary_search
    integer(dp) :: old, new
    integer :: icfg, jcfg, kcfg, lcfg
    integer :: alpha,beta,gama,delta
    integer :: tot_sign
    integer :: sgn
    integer :: i,j
    integer :: which_block
    integer :: row_shift
    integer :: begin_col
    
    integer :: core_indx
    integer :: val_indx
    integer(dp) :: fock_core(num_core_orbs)
    integer(dp) :: new_core
    integer(dp) :: new_val

    integer, allocatable :: num_of_cols(:,:)
    integer :: num_of_tot(nblock)

    type (T_col), pointer :: next_col, curr_col
    type (T_container_col) :: cols(nblock)

    which_block = 0
    row_shift = end_indx(1,1,myid+1) - 1 
    allocate(num_of_cols(end_indx(2,1,myid+1)-end_indx(1,1,myid+1)+1, nblock))
    num_of_cols = 0
    num_of_tot = 0

    do i=1,num_core_orbs
        fock_core(i) = (2_dp)**num_core_orbs - 1_dp - (2_dp)**(num_core_orbs-i)
    enddo

    do icfg=end_indx(1,1,myid+1), end_indx(2,1,myid+1)
        core_indx = (icfg-1)/ncfgs + 1
        val_indx = mod(icfg-1, ncfgs) + 1
        do i=1,nblock
            cols(i)%col_ptr => null()
        enddo
        do i=1,nhopp
            alpha = hopping(i)%ind1
            beta  = hopping(i)%ind2
            old = fock_core(core_indx) * (2_dp)**num_val_orbs + fock(val_indx)
            if (.not. btest(old, alpha-1)) cycle
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn) 
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, beta-1)) cycle
            call make_newfock('+', beta,  old, new, sgn) 
            tot_sign = tot_sign * sgn
            new_core = new/((2_dp)**num_val_orbs)
            new_val  = new - new_core*(2_dp)**num_val_orbs
            kcfg = binary_search(num_core_orbs, fock_core, new_core)
            lcfg = binary_search(ncfgs, fock, new_val)
            if (kcfg == -1 .or. lcfg == -1) cycle
            jcfg = (kcfg-1)*ncfgs + lcfg

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            ! init one row
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, hopping(i)%val * tot_sign * ham_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, hopping(i)%val * tot_sign * ham_sign)
            endif
        enddo

        do i=1,ncoul
            alpha = coulomb(i)%ind1
            beta  = coulomb(i)%ind2
            gama  = coulomb(i)%ind3
            delta = coulomb(i)%ind4
            old = fock_core(core_indx) * (2_dp)**num_val_orbs + fock(val_indx)
            if (.not. btest(old, alpha-1)) cycle
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new
            if (.not. btest(old, beta-1)) cycle
            call make_newfock('-', beta, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, gama-1)) cycle
            call make_newfock('+', gama, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, delta-1)) cycle
            call make_newfock('+', delta, old, new, sgn)
            tot_sign = tot_sign * sgn
            new_core = new/((2_dp)**num_val_orbs)
            new_val  = new - new_core*(2_dp)**num_val_orbs
            kcfg = binary_search(num_core_orbs, fock_core, new_core)
            lcfg = binary_search(ncfgs, fock, new_val)
            if (kcfg == -1 .or. lcfg == -1) cycle
            jcfg = (kcfg-1)*ncfgs + lcfg

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, coulomb(i)%val * tot_sign * ham_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, coulomb(i)%val * tot_sign * ham_sign)
            endif
        enddo

        ! diagonal elements: omega
        if (ham_sign < 0) then
            jcfg = icfg
            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif
            
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, omega)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, omega) 
            endif
        endif

        do i=1,nblock
            next_col => cols(i)%col_ptr
            do while (associated(next_col)) 
                num_of_cols(icfg-row_shift, i) = num_of_cols(icfg-row_shift, i) + 1 
                curr_col => next_col
                next_col => next_col%next
                deallocate(curr_col)
            enddo 
            num_of_tot(i) = num_of_tot(i) + num_of_cols(icfg-row_shift, i)
        enddo 
    enddo

    if (myid==master) then
        print *, "    Allocate memory for ham_csr ..."
    endif
    ! change to csr format
    do i=1,nblock
        if (num_of_tot(i) /= 0) then
            ham_csr(i)%m = end_indx(2,1,myid+1) - end_indx(1,1,myid+1) + 1
            ham_csr(i)%nnz = num_of_tot(i)
            ham_csr(i)%row_shift = end_indx(1,1,myid+1) - 1 
            ham_csr(i)%col_shift = end_indx(1,2,i) - 1 
            call alloc_csr(ham_csr(i)%m, ham_csr(i)%nnz, ham_csr(i))
            ham_csr(i)%iaa(1) = 1
            do j=2,ham_csr(i)%m+1
                ham_csr(i)%iaa(j) = ham_csr(i)%iaa(j-1) + num_of_cols(j-1, i)
            enddo
        endif
    enddo
    deallocate(num_of_cols)

    ! really building Hamiltonian here
    if (myid==master) then
        print *, "    Really building Hamiltonian begin here ..."
    endif
    do icfg=end_indx(1,1,myid+1), end_indx(2,1,myid+1)
        core_indx = (icfg-1)/ncfgs + 1
        val_indx = mod(icfg-1, ncfgs) + 1
        do i=1,nblock
            cols(i)%col_ptr => null()
        enddo
        do i=1,nhopp
            alpha = hopping(i)%ind1
            beta  = hopping(i)%ind2
            old = fock_core(core_indx) * (2_dp)**num_val_orbs + fock(val_indx)
            if (.not. btest(old, alpha-1)) cycle
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn) 
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, beta-1)) cycle
            call make_newfock('+', beta,  old, new, sgn) 
            tot_sign = tot_sign * sgn
            new_core = new/((2_dp)**num_val_orbs)
            new_val  = new - new_core*(2_dp)**num_val_orbs
            kcfg = binary_search(num_core_orbs, fock_core, new_core)
            lcfg = binary_search(ncfgs, fock, new_val)
            if (kcfg == -1 .or. lcfg == -1) cycle
            jcfg = (kcfg-1)*ncfgs + lcfg

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            ! init one row
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, hopping(i)%val * tot_sign * ham_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, hopping(i)%val * tot_sign * ham_sign)
            endif
        enddo

        do i=1,ncoul
            alpha = coulomb(i)%ind1
            beta  = coulomb(i)%ind2
            gama  = coulomb(i)%ind3
            delta = coulomb(i)%ind4
            old = fock_core(core_indx) * (2_dp)**num_val_orbs + fock(val_indx)
            if (.not. btest(old, alpha-1)) cycle
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new
            if (.not. btest(old, beta-1)) cycle
            call make_newfock('-', beta, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, gama-1)) cycle
            call make_newfock('+', gama, old, new, sgn)
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, delta-1)) cycle
            call make_newfock('+', delta, old, new, sgn)
            tot_sign = tot_sign * sgn
            new_core = new/((2_dp)**num_val_orbs)
            new_val  = new - new_core*(2_dp)**num_val_orbs
            kcfg = binary_search(num_core_orbs, fock_core, new_core)
            lcfg = binary_search(ncfgs, fock, new_val)
            if (kcfg == -1 .or. lcfg == -1) cycle
            jcfg = (kcfg-1)*ncfgs + lcfg

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, coulomb(i)%val * tot_sign * ham_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, coulomb(i)%val * tot_sign * ham_sign)
            endif
        enddo

        ! diagonal elements: omega
        if (ham_sign < 0) then
            jcfg = icfg
            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif
            
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, omega)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, omega) 
            endif
        endif

        ! copy to T_csr
        do i=1,nblock
            if (num_of_tot(i) == 0) cycle
            begin_col = ham_csr(i)%iaa(icfg-row_shift) - 1
            next_col => cols(i)%col_ptr
            do while (associated(next_col)) 
                begin_col = begin_col + 1
                ham_csr(i)%jaa(begin_col) = next_col%col
                ham_csr(i)%aa(begin_col)  = next_col%val
                curr_col => next_col
                next_col => next_col%next
                deallocate(curr_col)
            enddo 
        enddo 
    enddo

    return
end subroutine build_ham_n

!! building transition operator from initial configuration to intermediate configuration
subroutine build_transop_i(mcfgs, ncfgs, fock_left, fock_right, num_val_orbs, &
                     num_core_orbs, nblock, end_indx, ntran, transop, tran_csr)
    use m_constants, only: dp
    use mpi
    use m_control, only: myid, nprocs
    use m_types

    implicit none

    ! external variables
    ! the dimension of left Hilbert space
    integer, intent(in) :: mcfgs
    ! the dimension of right Hilbert space
    integer, intent(in) :: ncfgs
    ! the fock basis of left Hilbert space
    integer(dp), intent(in) :: fock_left(mcfgs)
    ! the fock basis of right Hilbert space
    integer(dp), intent(in) :: fock_right(ncfgs)
    ! number of valence orbitals
    integer, intent(in) :: num_val_orbs
    ! number of core orbitals
    integer, intent(in) :: num_core_orbs
    ! number of blocks of right Hilbert space divided by cpu cores
    integer, intent(in) :: nblock
    ! end index of the Fock basis for each cpu core
    integer, intent(in) :: end_indx(2,2,nprocs)
    integer, intent(in) :: ntran
    type(T_tensor2), intent(in) :: transop(ntran)
    ! transition matrix in csr format
    type (T_csr) :: tran_csr(nblock)

    ! local variables
    integer, external :: binary_search
    integer :: icfg, jcfg
    integer :: alpha, beta
    integer(dp) :: old, new
    integer :: tot_sign 
    integer :: sgn
    integer :: i,j
    integer :: which_block
    integer :: row_shift
    integer :: begin_col

    integer :: core_indx
    integer :: val_indx
    integer(dp) :: fock_core(num_core_orbs)
    integer(dp) :: new_core
    integer(dp) :: new_val

    integer, allocatable :: num_of_cols(:,:)
    integer :: num_of_tot(nblock)

    type (T_col), pointer :: next_col, curr_col
    type (T_container_col) :: cols(nblock)

    which_block = 0
    row_shift = end_indx(1,1,myid+1) - 1 
    allocate(num_of_cols(end_indx(2,1,myid+1)-end_indx(1,1,myid+1)+1, nblock))
    num_of_cols = 0
    num_of_tot = 0

    do i=1,num_core_orbs
        fock_core(i) = (2_dp)**num_core_orbs - (1_dp) - (2_dp)**(num_core_orbs-i)
    enddo

    do icfg=end_indx(1,1,myid+1), end_indx(2,1,myid+1)
        core_indx = (icfg-1)/mcfgs + 1
        val_indx = mod(icfg-1, mcfgs) + 1
        do i=1,nblock
            cols(i)%col_ptr => null()
        enddo
        do i=1,ntran
            alpha = transop(i)%ind1
            beta  = transop(i)%ind2
            old = fock_core(core_indx) * (2_dp)**num_val_orbs + fock_left(val_indx)
            if (.not. btest(old, alpha-1)) cycle
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn) 
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, beta-1)) cycle
            call make_newfock('+', beta,  old, new, sgn) 
            tot_sign = tot_sign * sgn
            new_core = new/((2_dp)**num_val_orbs)
            new_val  = new - new_core*(2_dp)**num_val_orbs
            jcfg = binary_search(ncfgs, fock_right, new_val)
            if (jcfg == -1) cycle

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            ! init one row
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, transop(i)%val * tot_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, transop(i)%val * tot_sign)
            endif
        enddo

        do i=1,nblock
            next_col => cols(i)%col_ptr
            do while (associated(next_col)) 
                num_of_cols(icfg-row_shift, i) = num_of_cols(icfg-row_shift, i) + 1 
                curr_col => next_col
                next_col => next_col%next
                deallocate(curr_col)
            enddo 
            num_of_tot(i) = num_of_tot(i) + num_of_cols(icfg-row_shift, i)
        enddo 
    enddo

    do i=1,nblock
        if (num_of_tot(i) /= 0) then
            tran_csr(i)%m = end_indx(2,1,myid+1) - end_indx(1,1,myid+1) + 1
            tran_csr(i)%nnz = num_of_tot(i)
            tran_csr(i)%row_shift = end_indx(1,1,myid+1) - 1 
            tran_csr(i)%col_shift = end_indx(1,2,i) - 1 
            call alloc_csr(tran_csr(i)%m, tran_csr(i)%nnz, tran_csr(i))
            tran_csr(i)%iaa(1) = 1
            do j=2,tran_csr(i)%m+1
                tran_csr(i)%iaa(j) = tran_csr(i)%iaa(j-1) + num_of_cols(j-1, i)
            enddo
        endif
    enddo
    deallocate(num_of_cols)

    ! really building begins
    do icfg=end_indx(1,1,myid+1), end_indx(2,1,myid+1)
        core_indx = (icfg-1)/mcfgs + 1
        val_indx = mod(icfg-1, mcfgs) + 1
        do i=1,nblock
            cols(i)%col_ptr => null()
        enddo
        do i=1,ntran
            alpha = transop(i)%ind1
            beta  = transop(i)%ind2
            old = fock_core(core_indx) * (2_dp)**num_val_orbs + fock_left(val_indx)
            if (.not. btest(old, alpha-1)) cycle
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn) 
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, beta-1)) cycle
            call make_newfock('+', beta,  old, new, sgn) 
            tot_sign = tot_sign * sgn
            new_core = new/((2_dp)**num_val_orbs)
            new_val  = new - new_core*(2_dp)**num_val_orbs
            jcfg = binary_search(ncfgs, fock_right, new_val)
            if (jcfg == -1) cycle

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            ! init one row
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, transop(i)%val * tot_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, transop(i)%val * tot_sign)
            endif
        enddo

        ! copy to T_csr
        do i=1,nblock
            if (num_of_tot(i) == 0) cycle
            begin_col = tran_csr(i)%iaa(icfg-row_shift) - 1
            next_col => cols(i)%col_ptr
            do while (associated(next_col)) 
                begin_col = begin_col + 1
                tran_csr(i)%jaa(begin_col) = next_col%col
                tran_csr(i)%aa(begin_col)  = next_col%val
                curr_col => next_col
                next_col => next_col%next
                deallocate(curr_col)
            enddo 
        enddo 
    enddo

    return
end subroutine build_transop_i

!! building transition operator from intermediate configuration to final configuration
subroutine build_transop_f(mcfgs, ncfgs, fock_left, fock_right, num_val_orbs, &
                     num_core_orbs, nblock, end_indx, ntran, transop, tran_csr)
    use m_constants, only: dp
    use mpi
    use m_control, only: myid, nprocs
    use m_types

    implicit none

    ! external variables
    ! the dimension of left Hilbert space
    integer, intent(in) :: mcfgs
    ! the dimension of right Hilbert space
    integer, intent(in) :: ncfgs
    ! the fock basis of left Hilbert space
    integer(dp), intent(in) :: fock_left(mcfgs)
    ! the fock basis of right Hilbert space
    integer(dp), intent(in) :: fock_right(ncfgs)
    ! number of valence orbitals
    integer, intent(in) :: num_val_orbs
    ! number of core orbitals
    integer, intent(in) :: num_core_orbs
    ! number of blocks of right Hilbert space divided by cpu cores
    integer, intent(in) :: nblock
    ! end index of the Fock basis for each cpu core
    integer, intent(in) :: end_indx(2,2,nprocs)
    integer, intent(in) :: ntran
    type(T_tensor2), intent(in) :: transop(ntran)
    ! transition matrix in csr format
    type (T_csr) :: tran_csr(nblock)

    ! local variables
    integer, external :: binary_search
    integer :: icfg, jcfg, kcfg, lcfg
    integer :: alpha, beta
    integer(dp) :: old, new
    integer :: tot_sign 
    integer :: sgn
    integer :: i,j
    integer :: which_block
    integer :: row_shift
    integer :: begin_col

    integer :: core_indx
    integer :: val_indx
    integer(dp) :: fock_core(num_core_orbs)
    integer(dp) :: new_core
    integer(dp) :: new_val

    integer, allocatable :: num_of_cols(:,:)
    integer :: num_of_tot(nblock)

    type (T_col), pointer :: next_col, curr_col
    type (T_container_col) :: cols(nblock)

    which_block = 0
    row_shift = end_indx(1,1,myid+1) - 1 
    allocate(num_of_cols(end_indx(2,1,myid+1)-end_indx(1,1,myid+1)+1, nblock))
    num_of_cols = 0
    num_of_tot = 0

    do i=1,num_core_orbs
        fock_core(i) = (2_dp)**num_core_orbs - (1_dp) - (2_dp)**(num_core_orbs-i)
    enddo

    do icfg=end_indx(1,1,myid+1), end_indx(2,1,myid+1)
        do i=1,nblock
            cols(i)%col_ptr => null()
        enddo
        do i=1,ntran
            alpha = transop(i)%ind1
            beta  = transop(i)%ind2
            old = ((2_dp)**num_core_orbs - (1_dp)) * (2_dp)**num_val_orbs + fock_left(icfg)
            if (.not. btest(old, alpha-1)) cycle
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn) 
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, beta-1)) cycle
            call make_newfock('+', beta,  old, new, sgn) 
            tot_sign = tot_sign * sgn
            new_core = new/((2_dp)**num_val_orbs)
            new_val  = new - new_core*(2_dp)**num_val_orbs
            kcfg = binary_search(num_core_orbs, fock_core, new_core)
            lcfg = binary_search(ncfgs, fock_right, new_val)
            if (kcfg == -1 .or. lcfg == -1) cycle
            jcfg = (kcfg-1)*ncfgs + lcfg

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            ! init one row
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, transop(i)%val * tot_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, transop(i)%val * tot_sign)
            endif
        enddo

        do i=1,nblock
            next_col => cols(i)%col_ptr
            do while (associated(next_col)) 
                num_of_cols(icfg-row_shift, i) = num_of_cols(icfg-row_shift, i) + 1 
                curr_col => next_col
                next_col => next_col%next
                deallocate(curr_col)
            enddo 
            num_of_tot(i) = num_of_tot(i) + num_of_cols(icfg-row_shift, i)
        enddo 
    enddo

    do i=1,nblock
        if (num_of_tot(i) /= 0) then
            tran_csr(i)%m = end_indx(2,1,myid+1) - end_indx(1,1,myid+1) + 1
            tran_csr(i)%nnz = num_of_tot(i)
            tran_csr(i)%row_shift = end_indx(1,1,myid+1) - 1 
            tran_csr(i)%col_shift = end_indx(1,2,i) - 1 
            call alloc_csr(tran_csr(i)%m, tran_csr(i)%nnz, tran_csr(i))
            tran_csr(i)%iaa(1) = 1
            do j=2,tran_csr(i)%m+1
                tran_csr(i)%iaa(j) = tran_csr(i)%iaa(j-1) + num_of_cols(j-1, i)
            enddo
        endif
    enddo
    deallocate(num_of_cols)

    ! really building begins
    do icfg=end_indx(1,1,myid+1), end_indx(2,1,myid+1)
        core_indx = (icfg-1)/mcfgs + 1
        val_indx = mod(icfg-1, mcfgs) + 1
        do i=1,nblock
            cols(i)%col_ptr => null()
        enddo
        do i=1,ntran
            alpha = transop(i)%ind1
            beta  = transop(i)%ind2
            old = ((2_dp)**num_core_orbs - (1_dp)) * (2_dp)**num_val_orbs + fock_left(icfg)
            if (.not. btest(old, alpha-1)) cycle
            tot_sign = 1
            call make_newfock('-', alpha, old, new, sgn) 
            tot_sign = tot_sign * sgn
            old = new
            if (btest(old, beta-1)) cycle
            call make_newfock('+', beta,  old, new, sgn) 
            tot_sign = tot_sign * sgn
            new_core = new/((2_dp)**num_val_orbs)
            new_val  = new - new_core*(2_dp)**num_val_orbs
            kcfg = binary_search(num_core_orbs, fock_core, new_core)
            lcfg = binary_search(ncfgs, fock_right, new_val)
            if (kcfg == -1 .or. lcfg == -1) cycle
            jcfg = (kcfg-1)*ncfgs + lcfg

            if (nblock==nprocs) then
                do j=1,nprocs
                    if (jcfg >= end_indx(1,2,j) .and. jcfg <= end_indx(2,2,j)) then
                        which_block = j 
                    endif
                enddo
            elseif (nblock==1) then
                which_block = 1 
            else
                print *, "nblock /= 1 and nblock /= nprocs"
                STOP
            endif

            ! init one row
            if (.not. associated(cols(which_block)%col_ptr)) then
                call init_col(cols(which_block)%col_ptr, jcfg, transop(i)%val * tot_sign)
            else
                call insert_into_row(cols(which_block)%col_ptr, jcfg, transop(i)%val * tot_sign)
            endif
        enddo

        ! copy to T_csr
        do i=1,nblock
            if (num_of_tot(i) == 0) cycle
            begin_col = tran_csr(i)%iaa(icfg-row_shift) - 1
            next_col => cols(i)%col_ptr
            do while (associated(next_col)) 
                begin_col = begin_col + 1
                tran_csr(i)%jaa(begin_col) = next_col%col
                tran_csr(i)%aa(begin_col)  = next_col%val
                curr_col => next_col
                next_col => next_col%next
                deallocate(curr_col)
            enddo 
        enddo 
    enddo

    return
end subroutine build_transop_f
