#!/usr/bin/env python


import os
import shutil
import numpy as np
import edrixs


def build_dirs():
    for path in ["search_gs", "ed", "xas", "rixs_pp", "rixs_ps"]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)


def set_config():
    config = [
        "&control",
        "num_val_orbs=24    ! Number of total valence orbitals: 4x6=24",
        "num_core_orbs=4    ! Number of total core orbitals (2p-3/2)",
        "ed_solver=2        ! Type of ED solver, full diagonalization of H_{i}",
        "neval=10           ! Number of eigenvalues obtained",
        "ncv=50             ! For Arpack library",
        "maxiter=1000       ! Maximum iteration for Lanczos or Arpack",
        "eigval_tol=1E-10   ! Tolerance for eigenvalues",
        "nvector=4          ! Number of eigenvectors obtained",
        "idump=.true.       ! Dump eigenvectors to file eigvec.xxx",
        "num_gs=4           ! Numer of gound states are used for XAS and RIXS calculations",
        "linsys_tol=1E-10   ! Tolerance for the termination of solving linear equations",
        "nkryl=500          ! Maximum number of Krylov vectors",
        "gamma_in=2.5       ! Core-hole life-time in eV",
        "omega_in=-6.2      ! Relative incident x-ray energy at which RIXS calculations \
are performed",
        "&end"
    ]

    f = open('ed/config.in', 'w')
    for item in config:
        f.write(item + "\n")
    f.close()
    shutil.copy('ed/config.in', 'xas/config.in')
    shutil.copy('ed/config.in', 'rixs_pp/config.in')
    shutil.copy('ed/config.in', 'rixs_ps/config.in')


def get_hopping_coulomb():
    norbs = 28

    Ud, JH = edrixs.UJ_to_UdJH(6, 0.45)
    F0_d, F2_d, F4_d = edrixs.UdJH_to_F0F2F4(Ud, JH)

    scale_dp = 0.9
    G1_dp, G3_dp = 0.894 * scale_dp, 0.531 * scale_dp
    F2_dp = 1.036 * scale_dp
    Udp_av = 0.0
    F0_dp = Udp_av + edrixs.get_F0('dp', G1_dp, G3_dp)

    params = [
        F0_d, F2_d, F4_d
    ]
    umat_i = edrixs.get_umat_slater('t2g', *params)

    params = [
        F0_d, F2_d, F4_d,
        F0_dp, F2_dp,
        G1_dp, G3_dp,
        0.0, 0.0
    ]

    umat_tmp = edrixs.get_umat_slater('t2gp', *params)
    tmat = np.zeros((12, 12), dtype=np.complex128)
    tmat[0:6, 0:6] = edrixs.tmat_c2j(1)
    tmat[6:12, 6:12] = edrixs.tmat_c2j(1)
    umat_i[:, :, :, :] = edrixs.transform_utensor(umat_i, edrixs.tmat_c2j(1))

    umat_tmp[:, :, :, :] = edrixs.transform_utensor(umat_tmp, tmat)
    id1 = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
    id2 = [0, 1, 2, 3, 4, 5, 24, 25, 26, 27]
    umat_n = np.zeros((norbs, norbs, norbs, norbs), dtype=complex)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                for ll in range(10):
                    umat_n[id2[i], id2[j], id2[k], id2[ll]
                           ] = umat_tmp[id1[i], id1[j], id1[k], id1[ll]]

    emat_i = np.zeros((norbs, norbs), dtype=np.complex128)
    emat_n = np.zeros((norbs, norbs), dtype=np.complex128)
    # bath energy level & hybridization strength
    N_site = 3
    data = np.loadtxt('bath_sites.in')
    e1, v1 = data[:, 0], data[:, 1]
    e2, v2 = data[:, 2], data[:, 3]

    h = np.zeros((24, 24), dtype=complex)
    h[0, 0] = h[1, 1] = -14.7627972353
    h[2, 2] = h[3, 3] = h[4, 4] = h[5, 5] = -15.4689430453
    for i in range(N_site):
        o = 6 + 6 * i
        # orbital j=1/2
        h[o + 0, o + 0] = h[o + 1, o + 1] = e1[i]
        h[0, o + 0] = h[1, o + 1] = h[o + 0, 0] = h[o + 1, 1] = v1[i]
        # orbital j=3/2
        h[o + 2, o + 2] = h[o + 3, o + 3] = h[o + 4, o + 4] = h[o + 5, o + 5] = e2[i]
        h[2, o + 2] = h[3, o + 3] = h[4, o + 4] = h[5, o + 5] = v2[i]
        h[o + 2, 2] = h[o + 3, 3] = h[o + 4, 4] = h[o + 5, 5] = v2[i]

    emat_i[0:24, 0:24] += h
    emat_n[0:24, 0:24] += h

    # static core-hole potential
    static_V = 2.0
    for i in range(6):
        emat_n[i, i] -= static_V

    # Write to files
    # Search GS inputs
    edrixs.write_emat(emat_i, "search_gs/hopping_i.in")
    edrixs.write_umat(umat_i, "search_gs/coulomb_i.in")

    # ED inputs
    edrixs.write_emat(emat_i, "ed/hopping_i.in")
    edrixs.write_umat(umat_i, "ed/coulomb_i.in")

    # XAS inputs
    edrixs.write_emat(emat_n, "xas/hopping_n.in")
    edrixs.write_umat(umat_n, "xas/coulomb_n.in")

    # RIXS inputs
    edrixs.write_emat(emat_i, "rixs_pp/hopping_i.in")
    edrixs.write_umat(umat_i, "rixs_pp/coulomb_i.in")
    edrixs.write_emat(emat_n, "rixs_pp/hopping_n.in")
    edrixs.write_umat(umat_n, "rixs_pp/coulomb_n.in")

    edrixs.write_emat(emat_i, "rixs_ps/hopping_i.in")
    edrixs.write_umat(umat_i, "rixs_ps/coulomb_i.in")
    edrixs.write_emat(emat_n, "rixs_ps/hopping_n.in")
    edrixs.write_umat(umat_n, "rixs_ps/coulomb_n.in")


def get_transop():
    tmp = edrixs.get_trans_oper('t2gp')
    dipole = np.zeros((3, 28, 28), dtype=complex)
    for i in range(3):
        tmp[i] = edrixs.cb_op2(tmp[i], edrixs.tmat_c2j(1), edrixs.tmat_c2j(1))
        dipole[i, 0:6, 24:28] = tmp[i, 0:6, 2:6]

    # for RIXS
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0.0
    # polarization pi-pi and pi-sigma
    for key, alpha, beta in [('pp', 0, 0), ('ps', 0, np.pi)]:
        ei, ef = edrixs.dipole_polvec_rixs(thin, thout, phi, alpha, beta)
        transop_rixs_i = dipole[0] * ei[0] + dipole[1] * ei[1] + dipole[2] * ei[2]
        transop_rixs_f = np.conj(np.transpose(dipole[0] * ef[0] + dipole[1] * ef[1] +
                                 dipole[2] * ef[2]))
        edrixs.write_emat(transop_rixs_i, "rixs_" + key + "/transop_rixs_i.in")
        edrixs.write_emat(transop_rixs_f, "rixs_" + key + "/transop_rixs_f.in")

    # For XAS
    ei = np.array([1, 1, 1]) / np.sqrt(3.0)
    transop_xas = dipole[0] * ei[0] + dipole[1] * ei[1] + dipole[2] * ei[2]
    edrixs.write_emat(transop_xas, "xas/transop_xas.in")


if __name__ == '__main__':
    print("edrixs >>> building directories ...")
    build_dirs()
    print("edrixs >>> Done !")

    print("edrixs >>> set control file config.in ...")
    set_config()
    print("edrixs >>> Done !")

    print("edrixs >>> building hopping and coulomb parameters ...")
    get_hopping_coulomb()
    print("edrixs >>> Done !")

    print("edrixs >>> building transition operators from 2p to 5d(t2g) ...")
    get_transop()
    print("edrixs >>> Done !")
