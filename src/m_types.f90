!!>>> module used to define some types
module m_types
    use m_constants, only: dp, czero
    implicit none
 
    ! 2-index tensor
    type T_tensor2
        integer :: ind1
        integer :: ind2
        complex(dp) :: val
    end type T_tensor2

    ! 4-index tensor
    type T_tensor4
        integer :: ind1
        integer :: ind2
        integer :: ind3
        integer :: ind4
        complex(dp) :: val
    end type T_tensor4

    ! one column in one row of a sparse matrix
    type T_col
        ! current column index
        integer :: col
        ! value in this position
        complex(dp) :: val
        ! next column position
        type (T_col), pointer :: next
    end type T_col

    ! simulate array of pointers of T_col
    type T_container_col
        ! just a pointer to T_col
        type (T_col), pointer ::  col_ptr
    end type T_container_col

    ! use CSR to represent a sparse matrix 
    type T_csr
        ! number of nonzero matrix elements
        integer :: nnz
        ! number of rows of the matrix
        integer :: m
        ! the shift of rows
        integer :: row_shift
        ! the shift of columns
        integer :: col_shift
        ! the row index
        integer, allocatable :: iaa(:)
        ! the column index
        integer, allocatable :: jaa(:)
        ! the matrix elements
        complex(dp), allocatable :: aa(:)
    end type T_csr

    contains

    subroutine init_col(self, col, val)
        implicit none

        ! external variables
        type (T_col), pointer :: self
        integer, intent(in) :: col
        complex(dp), intent(in) :: val

        allocate(self)
        self%col = col
        self%val = val
        self%next => null()

        return
    end subroutine init_col

    subroutine insert_into_row(self, col, val)
        implicit none

        ! external variables 
        type (T_col), pointer :: self
        integer, intent(in) :: col
        complex(dp), intent(in) :: val

        ! local variables
        type (T_col), pointer :: curr
        type (T_col), pointer :: prev
        type (T_col), pointer :: tmp
  
        curr => self
        prev => null()
        do while (associated(curr))
            ! compare col with current node
            if (col == curr%col) then
                curr%val = curr%val + val
                return
            endif

            ! insert before current node
            if (col < curr%col) then
                allocate(tmp)
                tmp%col = col
                tmp%val = val 
                tmp%next => curr
                if (.not. associated(prev)) then
                    self => tmp
                    return
                endif 
                prev%next => tmp
                return 
            endif
            prev => curr
            curr => curr%next 
        enddo

        ! arrive at the tail of the list
        allocate(tmp)
        tmp%col = col
        tmp%val = val 
        tmp%next => null()
        prev%next => tmp

        return
    end subroutine insert_into_row

    subroutine init_csr(self)
        implicit none

        ! external variables
        type (T_csr) :: self

        self%m = 0
        self%nnz = 0
        self%row_shift = 0
        self%col_shift = 0

        return
    end subroutine init_csr


    subroutine alloc_csr(m, nnz, self)
        implicit none

        ! external variables
        integer, intent(in) :: m
        integer, intent(in) :: nnz
        type (T_csr) :: self

        integer :: istat

        allocate(self%iaa(m+1), stat=istat)
        allocate(self%jaa(nnz), stat=istat)
        allocate(self%aa(nnz),  stat=istat)

        self%iaa = 0
        self%jaa = 0
        self%aa  = czero

        return
    end subroutine alloc_csr

    subroutine dealloc_csr(self)
        implicit none

        ! external variables
        type (T_csr) :: self

        if (allocated(self%iaa)) deallocate(self%iaa)
        if (allocated(self%jaa)) deallocate(self%jaa)
        if (allocated(self%aa))  deallocate(self%aa)

        return
    end subroutine dealloc_csr

    ! copy elements from csr to full
    subroutine csr_to_full(m, n, m_csr, m_full)
        implicit none

        ! external variables
        integer, intent(in)      :: m
        integer, intent(in)      :: n
        type (T_csr)             :: m_csr
        complex(dp), intent(out) :: m_full(m,n)
        
        ! local variables
        integer :: i
        integer :: j
        integer :: begin_col
        integer :: end_col

        ! init m_full
        do i=1,m_csr%m
            begin_col = m_csr%iaa(i) 
            end_col = m_csr%iaa(i+1)-1 
            if (begin_col > end_col) cycle
            do j=begin_col, end_col
                m_full(i+m_csr%row_shift, m_csr%jaa(j)) = m_csr%aa(j)
            enddo
        enddo

        return
    end subroutine csr_to_full

end module m_types 
