!! define some global variables
module m_global
    use m_constants, only: dp
    use m_types

    implicit none

    ! nonzero elements of hopping and Coulomb interaction terms
    type(T_tensor2), public, allocatable, save :: hopping_i(:)
    type(T_tensor4), public, allocatable, save :: coulomb_i(:)
    type(T_tensor2), public, allocatable, save :: hopping_n(:)
    type(T_tensor4), public, allocatable, save :: coulomb_n(:)

    ! nonzeros of transition operators
    type(T_tensor2), public, allocatable, save :: transop_xas(:)
    type(T_tensor2), public, allocatable, save :: transop_rixs_i(:)
    type(T_tensor2), public, allocatable, save :: transop_rixs_f(:)

    ! fock basis
    integer(dp), public, allocatable, save :: fock_i(:)
    integer(dp), public, allocatable, save :: fock_n(:)
    integer(dp), public, allocatable, save :: fock_f(:)

    ! Hamiltonian 
    type (T_csr), public, allocatable, save :: ham_csr(:)
   
    ! transition operator
    type (T_csr), public, allocatable, save :: tran_csr(:)

    ! alpha and beta when building Krylov space
    real(dp), public, allocatable, save :: krylov_alpha(:)
    real(dp), public, allocatable, save :: krylov_beta(:)

    contains

    ! subroutines used to allocate and deallocate memory
    subroutine alloc_hopping_i()
        use m_constants, only: czero
        use m_control,   only: nhopp_i

        implicit none
         
        integer :: i

        allocate(hopping_i(nhopp_i))
        do i=1,nhopp_i
            hopping_i(i)%ind1 = -1
            hopping_i(i)%ind2 = -1
            hopping_i(i)%val  = czero
        enddo

        return
    end subroutine alloc_hopping_i

    subroutine dealloc_hopping_i()
        implicit none
         
        if (allocated(hopping_i)) deallocate(hopping_i)

        return
    end subroutine dealloc_hopping_i

    subroutine alloc_coulomb_i()
        use m_constants, only: czero
        use m_control,   only: ncoul_i

        implicit none
         
        integer :: i

        allocate(coulomb_i(ncoul_i))
        do i=1,ncoul_i
            coulomb_i(i)%ind1 = -1
            coulomb_i(i)%ind2 = -1
            coulomb_i(i)%ind3 = -1
            coulomb_i(i)%ind4 = -1
            coulomb_i(i)%val  = czero
        enddo

        return
    end subroutine alloc_coulomb_i

    subroutine dealloc_coulomb_i()
        implicit none
         
        if (allocated(coulomb_i)) deallocate(coulomb_i)

        return
    end subroutine dealloc_coulomb_i

    subroutine alloc_fock_i()
        use m_control, only: ndim_i

        implicit none
         
        allocate(fock_i(ndim_i))
        fock_i = -1

        return
    end subroutine alloc_fock_i

    subroutine dealloc_fock_i()
        implicit none
         
        if (allocated(fock_i)) deallocate(fock_i)

        return
    end subroutine dealloc_fock_i

    subroutine alloc_hopping_n()
        use m_constants, only: czero
        use m_control,   only: nhopp_n

        implicit none
         
        integer :: i

        allocate(hopping_n(nhopp_n))
        do i=1,nhopp_n
            hopping_n(i)%ind1 = -1
            hopping_n(i)%ind2 = -1
            hopping_n(i)%val  = czero
        enddo

        return
    end subroutine alloc_hopping_n

    subroutine dealloc_hopping_n()
        implicit none
         
        if (allocated(hopping_n)) deallocate(hopping_n)

        return
    end subroutine dealloc_hopping_n

    subroutine alloc_coulomb_n()
        use m_constants, only: czero
        use m_control,   only: ncoul_n

        implicit none
         
        integer :: i

        allocate(coulomb_n(ncoul_n))
        do i=1,ncoul_n
            coulomb_n(i)%ind1 = -1
            coulomb_n(i)%ind2 = -1
            coulomb_n(i)%ind3 = -1
            coulomb_n(i)%ind4 = -1
            coulomb_n(i)%val  = czero
        enddo

        return
    end subroutine alloc_coulomb_n

    subroutine dealloc_coulomb_n()
        implicit none
         
        if (allocated(coulomb_n)) deallocate(coulomb_n)

        return
    end subroutine dealloc_coulomb_n

    subroutine alloc_fock_n()
        use m_control, only: ndim_n_nocore

        implicit none
         
        allocate(fock_n(ndim_n_nocore))
        fock_n = -1

        return
    end subroutine alloc_fock_n

    subroutine dealloc_fock_n()
        implicit none
         
        if (allocated(fock_n)) deallocate(fock_n)

        return
    end subroutine dealloc_fock_n

    subroutine alloc_fock_f()
        use m_control, only: ndim_f

        implicit none
         
        allocate(fock_f(ndim_f))
        fock_f = -1

        return
    end subroutine alloc_fock_f

    subroutine dealloc_fock_f()
        implicit none
         
        if (allocated(fock_f)) deallocate(fock_f)

        return
    end subroutine dealloc_fock_f

    subroutine alloc_transop_xas()
        use m_constants, only: czero
        use m_control,   only: ntran_xas

        implicit none
         
        integer :: i

        allocate(transop_xas(ntran_xas))
        do i=1,ntran_xas
            transop_xas(i)%ind1 = -1
            transop_xas(i)%ind2 = -1
            transop_xas(i)%val  = czero
        enddo

        return
    end subroutine alloc_transop_xas

    subroutine dealloc_transop_xas()
        implicit none
         
        if (allocated(transop_xas)) deallocate(transop_xas)

        return
    end subroutine dealloc_transop_xas


    subroutine alloc_transop_rixs_i()
        use m_constants, only: czero
        use m_control,   only: ntran_rixs_i

        implicit none
         
        integer :: i

        allocate(transop_rixs_i(ntran_rixs_i))
        do i=1,ntran_rixs_i
            transop_rixs_i(i)%ind1 = -1
            transop_rixs_i(i)%ind2 = -1
            transop_rixs_i(i)%val  = czero
        enddo

        return
    end subroutine alloc_transop_rixs_i

    subroutine dealloc_transop_rixs_i()
        implicit none
         
        if (allocated(transop_rixs_i)) deallocate(transop_rixs_i)

        return
    end subroutine dealloc_transop_rixs_i

    subroutine alloc_transop_rixs_f()
        use m_constants, only: czero
        use m_control,   only: ntran_rixs_f

        implicit none
         
        integer :: i

        allocate(transop_rixs_f(ntran_rixs_f))
        do i=1,ntran_rixs_f
            transop_rixs_f(i)%ind1 = -1
            transop_rixs_f(i)%ind2 = -1
            transop_rixs_f(i)%val  = czero
        enddo

        return
    end subroutine alloc_transop_rixs_f

    subroutine dealloc_transop_rixs_f()
        implicit none
         
        if (allocated(transop_rixs_f)) deallocate(transop_rixs_f)

        return
    end subroutine dealloc_transop_rixs_f

    subroutine alloc_ham_csr(nblock)
        implicit none

        integer, intent(in) :: nblock

        integer :: i

        allocate(ham_csr(nblock))
        do i=1,nblock
            call init_csr(ham_csr(i))
        enddo

        return
    end subroutine alloc_ham_csr

    subroutine dealloc_ham_csr(nblock)
        implicit none

        integer, intent(in) :: nblock

        integer :: i

        do i=1,nblock
            call dealloc_csr(ham_csr(i))
        enddo
        if (allocated(ham_csr))    deallocate(ham_csr)

        return
    end subroutine dealloc_ham_csr

    subroutine alloc_tran_csr(nblock)
        implicit none

        integer, intent(in) :: nblock

        integer :: i

        allocate(tran_csr(nblock))
        do i=1,nblock
            call init_csr(tran_csr(i))
        enddo

        return
    end subroutine alloc_tran_csr

    subroutine dealloc_tran_csr(nblock)
        implicit none

        integer, intent(in) :: nblock

        integer :: i

        do i=1,nblock
            call dealloc_csr(tran_csr(i))
        enddo
        if (allocated(tran_csr))    deallocate(tran_csr)

        return
    end subroutine dealloc_tran_csr

end module m_global
