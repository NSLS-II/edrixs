!!>>> set control parameters
module m_control
    use m_constants, only: dp

    implicit none

    ! which ed solver is used
    integer, public, save :: ed_solver = 1

    ! number of valence orbitals
    integer, public, save :: num_val_orbs = 1

    ! number of core orbitals
    integer, public, save :: num_core_orbs = 1

    ! the dimension of initial Hilbert space 
    integer, public, save :: ndim_i = 1

    ! the dimension of intermediate Hilbert space without core orbitals 
    integer, public, save :: ndim_n_nocore = 1

    ! the dimension of intermediate Hilbert space with core orbitals 
    ! ndim_n = ndim_n_nocore * num_core_orbs
    integer, public, save :: ndim_n = 1

    ! the dimension of final Hilbert space
    integer, public, save :: ndim_f = 1

    ! number of nonzero elements of hopping and Coulomb interaction terms
    ! for initial Hamiltonian
    integer, public, save :: nhopp_i =1 
    integer, public, save :: ncoul_i = 1

    ! number of nonzero elements of hopping and Coulomb interaction terms
    ! for intermediate Hamiltonian
    integer, public, save :: nhopp_n =1 
    integer, public, save :: ncoul_n = 1

    ! number of nonzeros of transition operators
    integer, public, save :: ntran_xas = 1
    integer, public, save :: ntran_rixs_i = 1
    integer, public, save :: ntran_rixs_f = 1
   
    ! how many smallest eigenvalues to be found
    integer, public, save :: neval = 1

    ! indicate how to dump the eigenvectors
    logical, public, save :: idump = .false.

    ! how many eigenvectors are dumped out
    integer, public, save :: nvector = 1

    ! how many ground state are used to calculate XAS or RIXS
    integer, public, save :: num_gs = 1

    ! maximum of the dimension of Krylov space
    ! for diagonalizing initial Hamiltonian
    integer, public, save :: maxiter = 1

    ! maximum step for linear system solver, minres
    integer, public, save :: linsys_max = 1

    ! minimum dimension for Lanczos or Arpack
    integer, public, save :: min_ndim = 100

    ! ncv, for arpack
    integer, public, save :: ncv = 1

    ! dimension of Krylov space, for building XAS and RIXS
    integer, public, save :: nkryl = 500

    ! the tolerance of eigen values
    real(dp), public, save :: eigval_tol = 1E-10 

    ! the tolerance for linear system
    real(dp), public, save :: linsys_tol = 1E-12 

    ! the frequency of incident photon, for RIXS
    real(dp), public, save :: omega_in 

    ! life-time of core-hole (in eV)
    real(dp), public, save :: gamma_in 

    ! number of process
    integer, public, save :: origin_nprocs = 1
    integer, public, save :: nprocs = 1

    ! the index of process
    integer, public, save :: origin_myid = 0
    integer, public, save :: myid = 0

    ! the index of master process
    integer, public, save :: master = 0 

    ! MPI communicator
    integer, public, save :: new_comm = 0
    integer, public, save :: origin_comm = 0

end module m_control
