#!/usr/bin/env python


import numpy as np
import edrixs


if __name__ == "__main__":
    """
    Purpose:   Os: L2/L3 edge  2p --> 5d
    """

    # PARAMETERS
    # ----------

    # 10 5d-orbitals, 6 2p-orbitals (including spin)
    ndorb, nporb = 10, 6
    # total orbitals
    norbs = ndorb + nporb

    # Slater integrals
    scale_d = 0.335
    F2_d, F4_d = 8.885 * scale_d, 5.971 * scale_d
    # averaged Coulomb interaction for 5d
    Ud = 0.0
    # F0_d is split into two parts: (1) Ud, (2) related to F2_d, F4_d
    F0_d = Ud + edrixs.get_F0('d', F2_d, F4_d)

    scale_dp = 0.9
    F2_dp = 1.036 * scale_dp
    G1_dp, G3_dp = 0.894 * scale_dp, 0.531 * scale_dp

    # averaged Coulomb interaction between 3d and 2p shells
    Udp = 0.0
    # F0_dp is split into two parts: (1) Udp, (2) related to G1_dp, G3_dp
    F0_dp = Udp + edrixs.get_F0('dp', G1_dp, G3_dp)

    # Spin-Orbit Coupling (SOC) zeta
    # 5d, without core-hole
    zeta_d_i = 0.33
    # 5d, with core-hole
    zeta_d_n = 0.33
    # 2p core electron, and usually we will adjust it to match the energy
    # difference of the two edges,
    # for example, roughly, E(L2-edge) - E(L3-edge) = 1.5 * zeta_p_n
    zeta_p_n = 1009.333333

    # Cubic crystal field
    dq10 = 4.3
    dq = dq10 / 10.0

    # END of PARAMETERS
    # -----------------

    params = [
        F0_d, F2_d, F4_d,    # Fk for d
        0.0, 0.0,            # Fk for dp
        0.0, 0.0,            # Gk for dp
        0.0, 0.0             # Fk for p
    ]

    # Coulomb interaction tensors without a core-hole
    umat_i = edrixs.get_umat_slater('dp', *params)

    params = [
        F0_d, F2_d, F4_d,    # Fk for d
        F0_dp, F2_dp,        # Fk for dp
        G1_dp, G3_dp,        # Gk for dp
        0.0, 0.0             # Fk for p
    ]

    # Coulomb interaction tensors with a core-hole
    umat_n = edrixs.get_umat_slater('dp', *params)

    # SOC matrix
    hsoc_i = np.zeros((norbs, norbs), dtype=np.complex128)
    hsoc_n = np.zeros((norbs, norbs), dtype=np.complex128)
    hsoc_i[0:ndorb, 0:ndorb] += edrixs.atom_hsoc('d', zeta_d_i)
    hsoc_n[0:ndorb, 0:ndorb] += edrixs.atom_hsoc('d', zeta_d_n)
    hsoc_n[ndorb:norbs, ndorb:norbs] += edrixs.atom_hsoc('p', zeta_p_n)

    # cubic crystal field
    crys = np.zeros((norbs, norbs), dtype=np.complex128)
    tmp = np.zeros((5, 5), dtype=np.complex128)
    tmp[0, 0] = 6 * dq    # d3z2-r2
    tmp[1, 1] = -4 * dq    # dzx
    tmp[2, 2] = -4 * dq    # dzy
    tmp[3, 3] = 6 * dq    # dx2-y2
    tmp[4, 4] = -4 * dq    # dxy

    # change basis to complex spherical harmonics, and then add spin degree of freedom
    tmp[:, :] = edrixs.cb_op(tmp, edrixs.tmat_r2c('d'))
    crys[0:ndorb:2, 0:ndorb:2] = tmp
    crys[1:ndorb:2, 1:ndorb:2] = tmp

    # build Fock basis, without core-hole
    # 3 5d-electrons and 6-2p electrons
    basis_i = edrixs.get_fock_bin_by_N(ndorb, 3, nporb, nporb)
    ncfgs_i = len(basis_i)

    # build Fock basis, with a core-hole
    # 9 5d-electrons and 5 2p-electrons
    basis_n = edrixs.get_fock_bin_by_N(ndorb, 4, nporb, nporb - 1)
    ncfgs_n = len(basis_n)

    # build many body Hamiltonian, without core-hole
    print("edrixs >>> building Hamiltonian without core-hole ...")
    hmat_i = np.zeros((ncfgs_i, ncfgs_i), dtype=np.complex128)
    hmat_n = np.zeros((ncfgs_n, ncfgs_n), dtype=np.complex128)

    emat = hsoc_i + crys
    hmat_i[:, :] += edrixs.two_fermion(emat, basis_i, basis_i)
    hmat_i[:, :] += edrixs.four_fermion(umat_i, basis_i)
    print("edrixs >>> Done!")

    # build many body Hamiltonian, with a core-hole
    print("edrixs >>> building Hamiltonian with a core-hole ...")
    emat = hsoc_n + crys
    hmat_n[:, :] += edrixs.two_fermion(emat, basis_n, basis_n)
    hmat_n[:, :] += edrixs.four_fermion(umat_n, basis_n)
    print("edrixs >>> Done!")

    # diagonalize many body Hamiltonian
    eval_i, evec_i = np.linalg.eigh(hmat_i)
    eval_n, evec_n = np.linalg.eigh(hmat_n)

    # write eigenvalues to files, the eigenvalues are needed for XAS and RIXS calculations
    edrixs.write_tensor(eval_i, "eval_i.dat")
    edrixs.write_tensor(eval_n, "eval_n.dat")

    # build dipolar transition operator from 2p to 5d
    print("edrixs >>> building dipolar transition operators...")
    dipole_op = np.zeros((3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    dipole = np.zeros((3, norbs, norbs), dtype=np.complex128)
    # Please NOTE that, here the function returns operators x, y, z
    tmp_op = edrixs.get_trans_oper('dp')

    # build the transition operator in the Fock basis
    dipole_op = np.zeros((3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    for i in range(3):
        dipole[i, 0:ndorb, ndorb:norbs] = tmp_op[i]
        dipole_op[i] = edrixs.two_fermion(dipole[i], basis_n, basis_i)
    # and then transform it to the eigenvalue basis
    for i in range(3):
        dipole_op[i] = edrixs.cb_op2(dipole_op[i], evec_n, evec_i)
    # write it to file, will be used by XAS and RIXS calculations
    edrixs.write_tensor(dipole_op, "trans_mat.dat")

    print("edrixs >>> Done !")
