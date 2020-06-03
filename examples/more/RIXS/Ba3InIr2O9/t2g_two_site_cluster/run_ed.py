#!/usr/bin/env python


import scipy
import numpy as np
import edrixs

if __name__ == "__main__":
    norbs = 24

    Ud, JH = edrixs.UJ_to_UdJH(2, 0.3)
    F0_d, F2_d, F4_d = edrixs.UdJH_to_F0F2F4(Ud, JH)

    G1_dp, G3_dp = 0.957, 0.569
    F0_dp, F2_dp = edrixs.get_F0('dp', G1_dp, G3_dp), 1.107

    zeta_d_i, zeta_p_n = 0.35, 1072.6666666666667

    Ir_loc2g = np.zeros((2, 3, 3), dtype=np.float64)
    Ir_loc2g[0] = [
        [+0.70710678, +0.40824829, +0.57735027],
        [-0.70710678, +0.40824829, +0.57735027],
        [+0.00000000, -0.81649658, +0.57735027]
    ]
    Ir_loc2g[1] = [
        [-0.70710678, +0.40824829, -0.57735027],
        [+0.70710678, +0.40824829, -0.57735027],
        [+0.00000000, -0.81649658, -0.57735027],
    ]

    umat_t2g_i = edrixs.get_umat_slater('t2g', F0_d, F2_d, F4_d)
    umat_t2g_i = edrixs.transform_utensor(umat_t2g_i, edrixs.tmat_c2r('t2g', True))
    params = [
        0.0, 0.0, 0.0,       # Fk for d
        F0_dp, F2_dp,        # Fk for dp
        G1_dp, G3_dp,        # Gk for dp
        0.0, 0.0           # Fk for p
    ]
    umat_t2gp_n = edrixs.get_umat_slater('t2gp', *params)
    tmat = np.zeros((12, 12), dtype=np.complex128)
    tmat[0:6, 0:6] = edrixs.tmat_c2r('t2g', True)
    tmat[6:12, 6:12] = edrixs.tmat_c2r('p', True)
    umat_t2gp_n = edrixs.transform_utensor(umat_t2gp_n, tmat)

    umat_i = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    umat_n = np.zeros((2, norbs, norbs, norbs, norbs), dtype=np.complex128)
    umat_i[0:6, 0:6, 0:6, 0:6] = umat_t2g_i
    umat_i[6:12, 6:12, 6:12, 6:12] = umat_t2g_i
    for i in range(2):
        umat_n[i, 0:12, 0:12, 0:12, 0:12] += umat_i[0:12, 0:12, 0:12, 0:12]

    indx = np.array([[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
                     [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]])

    for m in range(2):
        for i in range(12):
            for j in range(12):
                for k in range(12):
                    for ll in range(12):
                        umat_n[m, indx[m, i], indx[m, j], indx[m, k],
                               indx[m, ll]] += umat_t2gp_n[i, j, k, ll]

    emat_i = np.zeros((norbs, norbs), dtype=np.complex128)
    emat_n = np.zeros((2, norbs, norbs), dtype=np.complex128)

    soc_d = edrixs.cb_op(edrixs.atom_hsoc('t2g', zeta_d_i), edrixs.tmat_c2r('t2g', True))
    soc_p = edrixs.cb_op(edrixs.atom_hsoc('p', zeta_p_n), edrixs.tmat_c2r('p', True))

    emat_i[0:6, 0:6] += soc_d
    emat_i[6:12, 6:12] += soc_d
    emat_n[0, 0:6, 0:6] += soc_d
    emat_n[0, 6:12, 6:12] += soc_d
    emat_n[0, 12:18, 12:18] += soc_p
    emat_n[1, 0:6, 0:6] += soc_d
    emat_n[1, 6:12, 6:12] += soc_d
    emat_n[1, 18:24, 18:24] += soc_p

    # crystal field and hopping between the two Ir atom
    a, b, c = -0.18, 0.036, -0.03
    # a, b, c = -0.0, 0.00, -0.00
    dmu = 0.0
    m1, m2 = dmu / 2.0, -dmu / 2.0
    crys_tmp = np.array(
        [[m1, c, c, a, b, a],
         [c, m1, c, b, a, a],
         [c, c, m1, a, a, b],
         [a, b, a, m2, c, c],
         [b, a, a, c, m2, c],
         [a, a, b, c, c, m2]
         ], dtype=np.complex128)

    # transform spin to local axis
    dmat = np.zeros((2, 2, 2), dtype=np.complex128)
    ang1, ang2, ang3 = edrixs.rmat_to_euler(np.transpose(Ir_loc2g[0]))
    dmat[0] = edrixs.dmat_spinor(ang1, ang2, ang3)
    ang1, ang2, ang3 = edrixs.rmat_to_euler(np.transpose(Ir_loc2g[1]))
    dmat[1] = edrixs.dmat_spinor(ang1, ang2, ang3)

    t_spinor = np.zeros((12, 12), dtype=np.complex128)
    for i in range(2):
        off = i * 6
        t_spinor[off + 0:off + 2, off + 0:off + 2] = dmat[i]
        t_spinor[off + 2:off + 4, off + 2:off + 4] = dmat[i]
        t_spinor[off + 4:off + 6, off + 4:off + 6] = dmat[i]

    crys_spin = np.zeros((12, 12), dtype=np.complex128)
    crys_spin[0:12:2, 0:12:2] = crys_tmp
    crys_spin[1:12:2, 1:12:2] = crys_tmp
    crys_spin[:, :] = edrixs.cb_op(crys_spin, t_spinor)

    emat_i[0:12, 0:12] += crys_spin
    emat_n[0, 0:12, 0:12] += crys_spin
    emat_n[1, 0:12, 0:12] += crys_spin

    # static core-hole potential
    core_v = 2.0
    for i in range(0, 6):
        emat_n[0, i, i] -= core_v
    for i in range(6, 12):
        emat_n[1, i, i] -= core_v

    basis_i = edrixs.get_fock_bin_by_N(12, 9, 12, 12)
    basis_n = []
    tmp = edrixs.get_fock_bin_by_N(12, 10, 6, 5, 6, 6)
    basis_n.append(tmp)
    tmp = edrixs.get_fock_bin_by_N(12, 10, 6, 6, 6, 5)
    basis_n.append(tmp)
    ncfgs_i, ncfgs_n = len(basis_i), len(basis_n[0])

    hmat_i = np.zeros((ncfgs_i, ncfgs_i), dtype=np.complex128)
    hmat_n = np.zeros((2, ncfgs_n, ncfgs_n), dtype=np.complex128)

    print("edrixs >>> building Hamiltonian without core-hole ...")
    hmat_i[:, :] += edrixs.two_fermion(emat_i, basis_i, basis_i)
    hmat_i[:, :] += edrixs.four_fermion(umat_i, basis_i)
    eval_i, evec_i = scipy.linalg.eigh(hmat_i)
    print("edrixs >>> Done!")

    print("edrixs >>> Building Hamiltonian with a core-hole ...")
    eval_n = np.zeros((2, ncfgs_n), dtype=np.float64)
    evec_n = np.zeros((2, ncfgs_n, ncfgs_n), dtype=np.complex128)
    for i in range(2):
        hmat_n[i] += edrixs.two_fermion(emat_n[i], basis_n[i], basis_n[i])
        hmat_n[i] += edrixs.four_fermion(umat_n[i], basis_n[i])
        eval_n[i], evec_n[i] = scipy.linalg.eigh(hmat_n[i])
    print("edrixs >>> Done!")

    edrixs.write_tensor(eval_i, 'eval_i.dat')
    edrixs.write_tensor(eval_n, 'eval_n.dat')

    # Building transition operators
    print("edrixs >>> building transition operators ...")
    tran_ptod = edrixs.get_trans_oper('t2gp')
    for i in range(3):
        tran_ptod[i, :, :] = edrixs.cb_op2(
            tran_ptod[i], edrixs.tmat_c2r('t2g', True), edrixs.tmat_c2r('p', True))
    dipole_n = np.zeros((2, 3, norbs, norbs), dtype=np.complex128)
    dipole_n[0, :, 0:6, 12:18] = tran_ptod
    dipole_n[1, :, 6:12, 18:24] = tran_ptod
    trop_n = np.zeros((2, 3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    for i in range(2):
        for j in range(3):
            trop_n[i, j] = edrixs.two_fermion(dipole_n[i, j], basis_n[i], basis_i)
            trop_n[i, j] = edrixs.cb_op2(trop_n[i, j], evec_n[i], evec_i)

    Ir_trop = np.zeros((2, 3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    for i in range(2):
        for j in range(3):
            Ir_trop[i, j] = Ir_loc2g[i, 0, j] * trop_n[i, 0] + \
                Ir_loc2g[i, 1, j] * trop_n[i, 1] + Ir_loc2g[i, 2, j] * trop_n[i, 2]
    edrixs.write_tensor(Ir_trop, 'trans_mat.dat')
    print("edrixs >>> Done !")
