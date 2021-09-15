#!/usr/bin/env python


import os
import shutil
import numpy as np
import edrixs


def build_dirs():
    for path in ["ed", "xas", "rixs_pp", "rixs_ps"]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)


def set_config():
    config = [
        "&control",
        "num_val_orbs=12    ! Number of total valence orbitals (t2g): 2x6=12",
        "num_core_orbs=12   ! Number of total core orbitals (2p): 2x6=12",
        "ed_solver=0        ! Type of ED solver, full diagonalization of H_{i}",
        "neval=220          ! Number of eigenvalues obtained",
        "nvector=4          ! Number of eigenvectors obtained",
        "idump=.true.       ! Dump eigenvectors to file eigvec.xxx",
        "num_gs=4           ! Numer of gound states are used for XAS and RIXS calculations",
        "linsys_tol=1E-10   ! Tolerance for the termination of solving linear equations",
        "nkryl=500          ! Maximum number of Krylov vectors",
        "gamma_in=2.5       ! Core-hole life-time in eV",
        "omega_in=-540.4    ! Relative incident x-ray energy at which RIXS calculations \
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


def get_hopping_coulomb(locaxis):
    # Number of orbitals
    nt2g, nporb, norbs = 6, 6, 24

    # On-site Coulomb interaction tensor
    Ud, JH = edrixs.UJ_to_UdJH(2, 0.3)
    F0_d, F2_d, F4_d = edrixs.UdJH_to_F0F2F4(Ud, JH)

    G1_dp, G3_dp = 0.957, 0.569
    F0_dp, F2_dp = edrixs.get_F0('dp', G1_dp, G3_dp), 1.107

    umat_t2g_i = edrixs.get_umat_slater('t2g', F0_d, F2_d, F4_d)

    params = [
        F0_d, F2_d, F4_d,   # Fk for d
        F0_dp, F2_dp,        # Fk for dp
        G1_dp, G3_dp,        # Gk for dp
        0.0, 0.0           # Fk for p
    ]
    umat_t2gp_n = edrixs.get_umat_slater('t2gp', *params)

    # static core-hole potential
    static_v = 2.0
    for i in range(0, nt2g):
        for j in range(nt2g, nt2g + nporb):
            umat_t2gp_n[i, j, j, i] += static_v

    umat_i = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    umat_n = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)

    umat_i[0:6, 0:6, 0:6, 0:6] = umat_t2g_i
    umat_i[6:12, 6:12, 6:12, 6:12] = umat_t2g_i

    indx = np.array([[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
                     [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]])
    for m in range(2):
        for i in range(12):
            for j in range(12):
                for k in range(12):
                    for ll in range(12):
                        umat_n[indx[m, i], indx[m, j], indx[m, k],
                               indx[m, ll]] += umat_t2gp_n[i, j, k, ll]

    emat_i = np.zeros((norbs, norbs), dtype=np.complex128)
    emat_n = np.zeros((norbs, norbs), dtype=np.complex128)

    # SOC
    zeta_d_i, zeta_p_n = 0.35, 1072.6666666666667
    soc_d = edrixs.atom_hsoc('t2g', zeta_d_i)
    soc_p = edrixs.atom_hsoc('p', zeta_p_n)

    emat_i[0:6, 0:6] += soc_d
    emat_i[6:12, 6:12] += soc_d

    emat_n[0:6, 0:6] += soc_d
    emat_n[6:12, 6:12] += soc_d

    emat_n[12:18, 12:18] += soc_p
    emat_n[18:24, 18:24] += soc_p
    for i in range(2 * nt2g):
        emat_n[i, i] -= 6 * static_v

    # Crystal field and hoppings between the two Ir-sites
    t1, t2, delta = -0.18, 0.036, -0.03
    # Uncomment the following line to do calculation without hopping and crystal filed splitting.
    # t1, t2, delta = 0, 0, -0.03
    crys_tmp = np.array(
        [[0, delta, delta, t1, t2, t1],
         [delta, 0, delta, t2, t1, t1],
         [delta, delta, 0, t1, t1, t2],
         [t1, t2, t1, 0, delta, delta],
         [t2, t1, t1, delta, 0, delta],
         [t1, t1, t2, delta, delta, 0]
         ], dtype=complex)

    # transform spin to local axis
    dmat = np.zeros((2, 2, 2), dtype=np.complex128)
    ang1, ang2, ang3 = edrixs.rmat_to_euler(locaxis[0])
    dmat[0] = edrixs.dmat_spinor(ang1, ang2, ang3)
    ang1, ang2, ang3 = edrixs.rmat_to_euler(locaxis[1])
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
    t_orb = np.zeros((12, 12), dtype=np.complex128)
    t_orb[0:6, 0:6] = edrixs.tmat_r2c('t2g', True)
    t_orb[6:12, 6:12] = edrixs.tmat_r2c('t2g', True)
    crys_spin[:, :] = edrixs.cb_op(crys_spin, np.dot(t_spinor, t_orb))

    emat_i[0:12, 0:12] += crys_spin
    emat_n[0:12, 0:12] += crys_spin

    # Write to files
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


def get_fock_basis():
    edrixs.write_fock_dec_by_N(12, 9, 'ed/fock_i.in')
    shutil.copy('ed/fock_i.in', 'xas/fock_i.in')
    shutil.copy('ed/fock_i.in', 'rixs_pp/fock_i.in')
    shutil.copy('ed/fock_i.in', 'rixs_ps/fock_i.in')
    shutil.copy('ed/fock_i.in', 'rixs_pp/fock_f.in')
    shutil.copy('ed/fock_i.in', 'rixs_ps/fock_f.in')

    edrixs.write_fock_dec_by_N(12, 10, 'xas/fock_n.in')
    shutil.copy('xas/fock_n.in', 'rixs_pp/fock_n.in')
    shutil.copy('xas/fock_n.in', 'rixs_ps/fock_n.in')


def get_transop(loc, pos):
    dop = edrixs.get_trans_oper('t2gp')
    dop_g = np.zeros((2, 3, 6, 6), dtype=complex)
    for i in range(2):
        dop_g[i, 0] = loc[i, 0, 0] * dop[0] + loc[i, 0, 1] * dop[1] + loc[i, 0, 2] * dop[2]
        dop_g[i, 1] = loc[i, 1, 0] * dop[0] + loc[i, 1, 1] * dop[1] + loc[i, 1, 2] * dop[2]
        dop_g[i, 2] = loc[i, 2, 0] * dop[0] + loc[i, 2, 1] * dop[1] + loc[i, 2, 2] * dop[2]

    # for RIXS
    thin, thout, phi = 30 / 180.0 * np.pi, 60 / 180.0 * np.pi, 0.0
    ein, eout = 11215.0, 11215.0   # eV
    kin, kout = edrixs.get_wavevector_rixs(thin, thout, phi, ein, eout)
    # polarization pi-pi and pi-sigma
    for key, alpha, beta in [('pp', 0, 0), ('ps', 0, np.pi)]:
        ei, ef = edrixs.dipole_polvec_rixs(thin, thout, phi, alpha, beta)
        T_i = np.zeros((2, 6, 6), dtype=complex)
        T_o = np.zeros((2, 6, 6), dtype=complex)
        for i in range(2):
            T_i[i] = (dop_g[i, 0] * ei[0] + dop_g[i, 1] * ei[1] +
                      dop_g[i, 2] * ei[2]) * np.exp(1j * np.dot(pos[i], kin))
            T_o[i] = np.exp(-1j * np.dot(pos[i], kout)) * np.conj(
                np.transpose(dop_g[i, 0] * ef[0] + dop_g[i, 1] * ef[1] + dop_g[i, 2] * ef[2]))
        transop_rixs_i = np.zeros((24, 24), dtype=complex)
        transop_rixs_f = np.zeros((24, 24), dtype=complex)
        for i in range(2):
            off1, off2 = i * 6, 12 + i * 6
            transop_rixs_i[off1:off1 + 6, off2:off2 + 6] = T_i[i]
            transop_rixs_f[off2:off2 + 6, off1:off1 + 6] = T_o[i]
        edrixs.write_emat(transop_rixs_i, "rixs_" + key + "/transop_rixs_i.in")
        edrixs.write_emat(transop_rixs_f, "rixs_" + key + "/transop_rixs_f.in")

    # For XAS
    ei = np.array([1, 1, 1]) / np.sqrt(3.0)
    T_i = np.zeros((2, 6, 6), dtype=complex)
    for i in range(2):
        T_i[i] = dop_g[i, 0] * ei[0] + dop_g[i, 1] * ei[1] + dop_g[i, 2] * ei[2]
    transop_xas = np.zeros((24, 24), dtype=np.complex128)
    for i in range(2):
        off1, off2 = i * 6, 12 + i * 6
        transop_xas[off1:off1 + 6, off2:off2 + 6] = T_i[i]
    edrixs.write_emat(transop_xas, "xas/transop_xas.in")


if __name__ == "__main__":
    # site positions
    site_pos = np.array([[2.9164499, 1.6838131, 2.3064262],
                         [2.9164499, 1.6838131, 4.9373740]])
    # local axis for the two sites, the Wannier orbitals are defined with respect to them
    loc = np.zeros((2, 3, 3), dtype=np.float64)
    loc[0] = np.transpose(np.array([[+0.70710678, +0.40824829, +0.57735027],
                                    [-0.70710678, +0.40824829, +0.57735027],
                                    [+0.00000000, -0.81649658, +0.57735027]]))
    loc[1] = np.transpose(np.array([[-0.70710678, +0.40824829, -0.57735027],
                                    [+0.70710678, +0.40824829, -0.57735027],
                                    [+0.00000000, -0.81649658, -0.57735027]]))

    print("edrixs >>> building directories ...")
    build_dirs()
    print("edrixs >>> Done !")

    print("edrixs >>> set control file config.in ...")
    set_config()
    print("edrixs >>> Done !")

    print("edrixs >>> building hopping and coulomb parameters ...")
    get_hopping_coulomb(loc)
    print("edrixs >>> Done !")

    print("edrixs >>> building fock_basis ...")
    get_fock_basis()
    print("edrixs >>> Done !")

    print("edrixs >>> building transition operators from 2p to 5d(t2g) ...")
    get_transop(loc, site_pos)
    print("edrixs >>> Done !")
