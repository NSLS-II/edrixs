import numpy as np
import scipy
import edrixs
import os
from time import sleep


def test_edx(norb=10, noccu=2,
             F0=4.0, F2=2.0, F4=0.5, ten_dq=1.5):
    emat = edrixs.cf_cubic_d(ten_dq)
    umat = edrixs.get_umat_slater('d', F0, F2, F4)
    basis = edrixs.get_fock_bin_by_N(norb, noccu)

    H = (edrixs.build_opers(2, emat, basis) +
         edrixs.build_opers(4, umat, basis))

    e, v = scipy.linalg.eigh(H)

    with open("config.in", 'w') as f:
        f.write("&control\n"
                "ed_solver     =   0\n"
                "num_val_orbs  =   20\n"
                "ncv           =   150\n"
                f"neval         =   {len(v)}\n"
                f"nvector       =   {len(v)}\n"
                f"num_gs        =   {len(v)}\n"
                "maxiter       =   1000\n"
                "eigval_tol    =   1E-10\n"
                "idump         =   .true.\n"
                "&end\n")

    edrixs.write_fock_dec_by_N(norb, noccu)
    edrixs.write_emat(emat, 'hopping_i.in')
    edrixs.write_umat(umat, 'coulomb_i.in')
    os.system("ed.x")

    while not os.path.exists('eigvals.dat'):
        sleep(1)

    _, e_fortran = np.loadtxt('eigvals.dat', unpack=True)

    assert np.all(np.isclose(e, e_fortran))
