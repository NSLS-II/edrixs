#!/usr/bin/env python


import numpy as np
import scipy
import edrixs

if __name__ == "__main__":
    norbs = 14

    F2_f, F4_f, F6_f = 9.711 * 0.77, 6.364 * 0.77, 4.677 * 0.77
    Uf_av = 0.0
    F0_f = Uf_av + edrixs.get_F0('f', F2_f, F4_f, F6_f)

    zeta_f_i = 0.261 * 0.9
    params = [F0_f, F2_f, F4_f, F6_f]
    umat_i = edrixs.get_umat_slater('f', *params)
    edrixs.write_umat(umat_i, 'coulomb_i.in', 1E-10)

    hsoc_i = edrixs.atom_hsoc('f', zeta_f_i)
    edrixs.write_emat(hsoc_i, 'hopping_i.in', 1E-10)

    edrixs.write_fock_dec_by_N(14, 7, "fock_i.in")

    config_in = [
        "&control",
        "ed_solver    =   2",
        "num_val_orbs =   14",
        "neval        =   100",
        "ncv          =   200",
        "nvector      =   2",
        "maxiter      =   1000",
        "eigval_tol   =   1E-10",
        "idump        =   .false.",
        "&end"
    ]
    f = open("config.in", 'w')
    for line in config_in:
        f.write(line + "\n")
    f.close()

    # diagonalize it by using python code
    basis_i = edrixs.get_fock_bin_by_N(norbs, 7)
    ncfgs_i = len(basis_i)

    print("edrixs >>> building Hamiltonian ...")
    hmat_i = np.zeros((ncfgs_i, ncfgs_i), dtype=np.complex128)
    hmat_i[:, :] += edrixs.two_fermion(hsoc_i, basis_i, basis_i)
    hmat_i[:, :] += edrixs.four_fermion(umat_i, basis_i)
    print("edrixs >>> Done!")

    print("edrixs >>> diagonalize Hamiltonian ...")
    eval_i, evec_i = scipy.linalg.eigh(hmat_i)
    print("edrixs >>> Done!")
    edrixs.write_tensor(eval_i, "eval_i.dat", fmt_float='{:20.10f}')
