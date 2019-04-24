subroutine pspmv_csr(comm, nblock, end_indx, needed, mloc, nloc, ham, vec_in, vec_out)
    use m_constants, only: dp, czero
    use m_control, only: myid, nprocs
    use m_types, only: T_csr
    use mpi

    implicit none

    ! externval variables
    integer, intent(in)         :: comm
    integer, intent(in)         :: nblock
    integer, intent(in)         :: end_indx(2, 2, nprocs)
    integer, intent(in)         :: needed(nprocs, nprocs)
    integer, intent(in)         :: mloc
    integer, intent(in)         :: nloc
    type (T_csr)                :: ham(nblock)
    complex(dp), intent(in)     :: vec_in(nloc)
    complex(dp), intent(inout)  :: vec_out(mloc)

    ! local variables
    integer :: i
    integer :: ireq
    integer :: ierror
    integer :: request(nprocs)
    integer :: send_stat(MPI_STATUS_SIZE, nprocs)
    integer :: stat(MPI_STATUS_SIZE)
    integer :: other_size
    integer :: max_size
    complex(dp), allocatable :: tmp_vec(:)

    ! send the data
    ireq = 0
    do i=1,nprocs
       if (needed(i,myid+1) == 1) then
           ireq = ireq + 1
           call MPI_ISEND(vec_in, nloc, MPI_DOUBLE_COMPLEX, i-1, i*(10*nprocs)+myid+1, &
           comm, request(ireq), ierror)
       endif
    enddo

    ! do matrix vector multiplication
    ! diagonal part
    call matvec_csr(mloc, nloc, ham(myid+1), vec_in, vec_out)
 
    ! off-diagonal parts
    if (nblock == nprocs .and. nprocs > 1) then
        max_size = end_indx(2,2,1)-end_indx(1,2,1) + 1
        do i=2,nprocs
            if ( (end_indx(2,2,i) - end_indx(1,2,i) + 1) > max_size ) then
                max_size = end_indx(2,2,i) - end_indx(1,2,i) + 1
            endif
        enddo
        allocate(tmp_vec(max_size))
        do i=1,nprocs
           if (needed(myid+1,i) == 1) then
               ! receive data from other processor
               other_size = end_indx(2,2,i) - end_indx(1,2,i) + 1
               tmp_vec = czero
               call MPI_RECV(tmp_vec(1:other_size), other_size, MPI_DOUBLE_COMPLEX, i-1, (myid+1)*(10*nprocs)+i,&
                comm, stat, ierror)
               ! do matrix vector multiplication
               call matvec_csr(mloc, other_size, ham(i), tmp_vec(1:other_size), vec_out)
           endif
        enddo
        if (allocated(tmp_vec)) deallocate(tmp_vec)
    endif

    call MPI_WAITALL(ireq, request, send_stat, ierror)
     
    return
end subroutine pspmv_csr

subroutine matvec_csr(m, n, mat, vec_in, vec_out)
    use m_constants, only: dp
    use m_types

    implicit none

    ! externval variables
    integer, intent(in) :: m
    integer, intent(in) :: n
    type (T_csr), intent(in) :: mat
    complex(dp), intent(in) :: vec_in(n)
    complex(dp), intent(inout) :: vec_out(m)

    ! local variables
    complex(dp) :: tmp 
    integer :: i,j
    integer :: begin_col, end_col

    do i=1,mat%m
       begin_col = mat%iaa(i) 
       end_col = mat%iaa(i+1)-1 
       if (begin_col > end_col) cycle
       tmp = 0
       do j=begin_col, end_col
           tmp = tmp + mat%aa(j) * vec_in(mat%jaa(j) - mat%col_shift)      
       enddo
       vec_out(i) = vec_out(i) + tmp
    enddo

    return
end subroutine matvec_csr
