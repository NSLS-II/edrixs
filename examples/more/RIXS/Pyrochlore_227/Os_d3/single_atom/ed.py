#!/usr/bin/env python

import numpy as np
import edrixs


if __name__ == "__main__":
    # local axis
    loc2g = np.zeros((4, 3, 3), dtype=np.float64)
    loc2g[0] = [[-2.0 / 3.0, +1.0 / 3.0, -2.0 / 3.0],
                [+1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
                [-2.0 / 3.0, -2.0 / 3.0, +1.0 / 3.0]]

    loc2g[1] = [[+2.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0],
                [-1.0 / 3.0, +2.0 / 3.0, -2.0 / 3.0],
                [+2.0 / 3.0, +2.0 / 3.0, +1.0 / 3.0]]

    loc2g[2] = [[+2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
                [+2.0 / 3.0, +1.0 / 3.0, +2.0 / 3.0],
                [-1.0 / 3.0, -2.0 / 3.0, +2.0 / 3.0]]

    loc2g[3] = [[-2.0 / 3.0, +2.0 / 3.0, -1.0 / 3.0],
                [-2.0 / 3.0, -1.0 / 3.0, +2.0 / 3.0],
                [+1.0 / 3.0, +2.0 / 3.0, +2.0 / 3.0]]

    U, J = 1.5, 0.25
    Ud, JH = edrixs.UJ_to_UdJH(U, J)
    F0_d, F2_d, F4_d = edrixs.UdJH_to_F0F2F4(Ud, JH)

    G1_dp, G3_dp = 0.957, 0.569
    F0_dp, F2_dp = edrixs.get_F0('dp', G1_dp, G3_dp), 1.107

    zeta_d, zeta_p = 0.45, 1009.3333333333334

    emat_i = np.zeros((12, 12), dtype=np.complex128)
    emat_n = np.zeros((12, 12), dtype=np.complex128)

    umat_i = edrixs.get_umat_slater('t2g', F0_d, F2_d, F4_d)
    umat_i = edrixs.transform_utensor(umat_i, edrixs.tmat_c2r('t2g', True))

    params = [
        F0_d, F2_d, F4_d,    # Fk for d
        F0_dp, F2_dp,        # Fk for dp
        G1_dp, G3_dp,        # Gk for dp
        0.0, 0.0             # Fk for p
    ]
    umat_n = edrixs.get_umat_slater('t2gp', *params)
    tmat = np.zeros((12, 12), dtype=np.complex128)
    tmat[0:6, 0:6] = edrixs.tmat_c2r('t2g', True)
    tmat[6:12, 6:12] = edrixs.tmat_c2r('p', True)
    umat_n = edrixs.transform_utensor(umat_n, tmat)

    # SOC
    hsoc_d = edrixs.cb_op(edrixs.atom_hsoc('t2g', zeta_d), edrixs.tmat_c2r('t2g', True))
    hsoc_p = edrixs.cb_op(edrixs.atom_hsoc('p', zeta_p), edrixs.tmat_c2r('p', True))
    emat_i[0:6, 0:6] += hsoc_d
    emat_n[0:6, 0:6] += hsoc_d
    emat_n[6:12, 6:12] += hsoc_p

    # Crystal field
    c = -0.10
    crys = np.array([[0, c, c],
                     [c, 0, c],
                     [c, c, 0]], dtype=np.complex128)
    emat_i[0:6:2, 0:6:2] += crys
    emat_i[1:6:2, 1:6:2] += crys
    emat_n[0:6:2, 0:6:2] += crys
    emat_n[1:6:2, 1:6:2] += crys

    basis_i = edrixs.get_fock_bin_by_N(6, 3, 6, 6)
    basis_n = edrixs.get_fock_bin_by_N(6, 4, 6, 5)
    ncfgs_i, ncfgs_n = len(basis_i), len(basis_n)

    print("edrixs >>> diagonalize Hamitlonian ...")
    hmat_i = np.zeros((ncfgs_i, ncfgs_i), dtype=np.complex128)
    hmat_n = np.zeros((ncfgs_n, ncfgs_n), dtype=np.complex128)
    hmat_i[:, :] += edrixs.two_fermion(emat_i, basis_i, basis_i)
    hmat_i[:, :] += edrixs.four_fermion(umat_i, basis_i)
    hmat_n[:, :] += edrixs.two_fermion(emat_n, basis_n, basis_n)
    hmat_n[:, :] += edrixs.four_fermion(umat_n, basis_n)

    eval_i, evec_i = np.linalg.eigh(hmat_i)
    eval_n, evec_n = np.linalg.eigh(hmat_n)

    edrixs.write_tensor(eval_i, "eval_i.dat")
    edrixs.write_tensor(eval_n, "eval_n.dat")
    print("edrixs >>> done !")

    # Transition operators
    print("edrixs >>> building dipolar transition operators...")
    dipole_op = np.zeros((3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    dipole_op_all = np.zeros((4, 3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    dipole = np.zeros((3, 12, 12), dtype=np.complex128)
    tmp = edrixs.get_trans_oper('t2gp')
    for i in range(3):
        tmp[i] = edrixs.cb_op2(tmp[i], edrixs.tmat_c2r('t2g', True), edrixs.tmat_c2r('p', True))
        dipole[i, 0:6, 6:12] = tmp[i]
        dipole_op[i] = edrixs.two_fermion(dipole[i], basis_n, basis_i)
        dipole_op[i] = edrixs.cb_op2(dipole_op[i], evec_n, evec_i)

    for i in range(4):
        for j in range(3):
            dipole_op_all[i, j] = loc2g[i, 0, j] * dipole_op[0] + \
                loc2g[i, 1, j] * dipole_op[1] + loc2g[i, 2, j] * dipole_op[2]

    edrixs.write_tensor(dipole_op_all, "trans_mat.dat")
    print("edrixs >>> Done !")
