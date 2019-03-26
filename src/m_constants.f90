module m_constants
    implicit none

    integer, public, parameter :: dp = kind(1.0d0)
    integer, public, parameter :: mystd = 6
    integer, public, parameter :: mytmp = 100

    real(dp), public, parameter :: zero = 0.0_dp
    real(dp), public, parameter :: one  = 1.0_dp
    real(dp), public, parameter :: pi   = 3.1415926535897932_dp

    complex(dp), public, parameter :: czi   = dcmplx(0.0_dp, 1.0_dp)
    complex(dp), public, parameter :: cone  = dcmplx(1.0_dp, 0.0_dp)
    complex(dp), public, parameter :: czero = dcmplx(0.0_dp, 0.0_dp)

    real(dp), public, parameter :: ev2k = 11604.505008098_dp

end module m_constants
