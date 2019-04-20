!!>>> binary search
function binary_search(n, vec, m) result(val)
    use m_constants, only: dp

    implicit none

! external variables
    integer, intent(in) :: n
    integer(dp), intent(in) :: vec(n)
    integer(dp), intent(in) :: m

! return value
    integer :: val

! local variables
    integer :: left
    integer :: right
    integer :: middle

    val = -1

    left=1
    right=n
    do while (left <= right)
        middle = left + (right-left)/2 

        ! find the index
        if (vec(middle) < m) then
            left = middle + 1
        elseif (vec(middle) > m) then
            right = middle -1
        else
            val = middle
            EXIT
        endif
    enddo

    return 

end function binary_search

subroutine sort_eigvals(n, eigvals, sorted_indx)
    use m_constants, only: dp, zero
    implicit none

    ! external variables
    integer,  intent(in)    :: n
    real(dp), intent(inout) :: eigvals(n)
    integer,  intent(inout) :: sorted_indx(n)

    ! local variables
    real(dp) :: tmp_eigval(n)
    integer  :: tmp_indx(n)
    integer  :: i
    integer  :: j

    tmp_eigval = eigvals
    tmp_indx   = sorted_indx

    ! we use simple insert sort algorithm 
    eigvals = zero
    sorted_indx = 0
    eigvals(1) = tmp_eigval(1)
    sorted_indx(1) = tmp_indx(1) 
    do i=2, n
        j=i-1
        do while (tmp_eigval(i) < eigvals(j))
            eigvals(j+1)     = eigvals(j) 
            sorted_indx(j+1) = sorted_indx(j)
            j = j - 1
            if (j==0) EXIT
        enddo 
        eigvals(j+1)     = tmp_eigval(i)
        sorted_indx(j+1) = tmp_indx(i)
    enddo 

    return
end subroutine sort_eigvals

subroutine partition_task(nprocs, m, n, end_indx)
    implicit none

    ! external variables
    integer, intent(in)  :: nprocs
    integer, intent(in)  :: m
    integer, intent(in)  :: n
    integer, intent(out) :: end_indx(2, 2, nprocs)

    ! local variables
    integer :: step
    integer :: i

    end_indx = 0
    ! nprocs <= m, n
    step = m / nprocs
    do i=1,nprocs-1
        end_indx(1,1,i) = (i-1)*step+1
        end_indx(2,1,i) =     i*step
    enddo 
    end_indx(1,1,nprocs) = (nprocs-1)*step+1
    end_indx(2,1,nprocs) = m

    step = n / nprocs
    do i=1,nprocs-1
        end_indx(1,2,i) = (i-1)*step+1
        end_indx(2,2,i) =     i*step
    enddo 
    end_indx(1,2,nprocs) = (nprocs-1)*step+1
    end_indx(2,2,nprocs) = n

    return
end subroutine partition_task 

subroutine get_needed_indx(nblock, ham_csr, needed)
    use m_control, only: myid, nprocs, new_comm
    use m_types
    use mpi
    implicit none

    integer, intent(in)  :: nblock
    type (T_csr)         :: ham_csr(nblock)
    integer, intent(out) :: needed(nprocs, nprocs)

    integer :: i
    integer :: needed_mpi(nprocs,nprocs)
    integer :: ierror

    needed = 0
    needed_mpi = 0
    if (nblock==nprocs) then
        do i=1,nprocs
            if (myid+1 /= i .and. ham_csr(i)%nnz > 0) then
                needed(myid+1, i) =  1 
            endif
        enddo
        call MPI_BARRIER(new_comm, ierror)
        call MPI_ALLREDUCE(needed, needed_mpi, size(needed), MPI_INTEGER, MPI_SUM, new_comm, ierror)
        call MPI_BARRIER(new_comm, ierror)
        needed = needed_mpi
    endif

    return
end subroutine get_needed_indx

subroutine get_number_nonzeros(nblock, ham_csr, num_nonzeros)
    use m_constants, only: dp
    use m_control, only: new_comm
    use m_types
    use mpi
    implicit none

    integer, intent(in)  :: nblock
    type (T_csr)         :: ham_csr(nblock)
    integer(dp), intent(out) :: num_nonzeros

    integer :: i
    integer(dp) :: num_nonzeros_mpi
    integer :: ierror

    num_nonzeros = 0_dp
    num_nonzeros_mpi = 0_dp
    do i=1,nblock
        num_nonzeros = num_nonzeros + ham_csr(i)%nnz 
    enddo
    call MPI_BARRIER(new_comm, ierror)
    call MPI_ALLREDUCE(num_nonzeros, num_nonzeros_mpi, 1, MPI_INTEGER8, MPI_SUM, new_comm, ierror)
    call MPI_BARRIER(new_comm, ierror)
    num_nonzeros = num_nonzeros_mpi

    return
end subroutine get_number_nonzeros

subroutine get_prob(n, eigval, beta, prob)
    use m_constants
    implicit none

    integer, intent(in) :: n
    real(dp), intent(in) :: eigval(n)
    real(dp), intent(in) :: beta
    real(dp), intent(out) :: prob(n)

    real(dp) :: rtemp(n)
    real(dp) :: norm
    integer :: i

    rtemp = eigval - minval(eigval)
    norm = 0.0
    do i=1,n
        norm = norm + exp(-beta * rtemp(i))
    enddo 
     
    prob = exp(-beta * rtemp) / norm

    return
end subroutine get_prob
