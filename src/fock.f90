subroutine cal_combination(n, m, cnm)
    use m_constants, only: dp

    implicit none

    ! external variables
    integer, intent(in) :: n
    integer, intent(in) :: m
    integer, intent(out) :: cnm

    ! local variables
    integer :: small
    integer :: large
    integer :: i

    real(dp) :: x
    real(dp) :: y

    small = min(m,n-m)
    large = max(m,n-m)
    
    x = 1.0_dp
    y = 1.0_dp
    do i=large+1, n
        x = x * dble(i)
    enddo 

    do i=1, small
        y = y * dble(i)
    enddo

    cnm = nint(x / y)

    return
end subroutine cal_combination

!!>>> get new fock state and the sign
subroutine make_newfock(c_type, pos, old, new, sgn)
    use m_constants, only: mystd, dp

    implicit none

    character(1), intent(in)  :: c_type
    integer,      intent(in)  :: pos
    integer(dp),  intent(in)  :: old
    integer(dp),  intent(out) :: new
    integer,      intent(out) :: sgn

    integer :: i

    sgn = 0
    do i=1,pos-1
       if (btest(old, i-1) .eqv. .true.) sgn = sgn + 1
    enddo 
    sgn = mod(sgn,2)
    sgn = (-1)**sgn

    if (c_type == '+') then
        new = old + (2_dp)**(pos-1)
    elseif (c_type == '-') then
        new = old - (2_dp)**(pos-1)
    else
        write(mystd, "(58a,a)")  " fedrixs >>> error of create and destroy operator type: ", c_type
        STOP
    endif

    return
end subroutine make_newfock
