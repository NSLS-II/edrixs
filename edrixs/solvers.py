#!/usr/bin/env python

import numpy as np
from .utils import info_atomic_shell, boltz_dist
from .coulomb_utensor import get_umat_slater
from .iostream import write_tensor
from .soc import atom_hsoc
from .angular_momentum import get_sx, get_sy, get_sz, get_lx, get_ly, get_lz
from .angular_momentum import rmat_to_euler, get_wigner_dmat
from .manybody_operator import two_fermion, four_fermion
from .fock_basis import get_fock_bin_by_N
from .basis_transform import cb_op2, tmat_r2c
from .photon_transition import get_trans_oper, quadrupole_polvec
from .photon_transition import dipole_polvec_xas, dipole_polvec_rixs, unit_wavevector
from .rixs_utils import scattering_mat


def ed_1v1c(v_name='d', c_name='p', v_level=0.0, c_level=0.0,
            v_soc=(0.0, 0.0), c_soc=0.0,
            v_noccu=1, slater=([0], [0]), ext_B=np.zeros(3),
            cf_mat=None, other_mat=None, local_axis=np.eye(3)):
    """
    Perform ED for two atomic shell case, one Valence-shell plus one Core-shell.
    For example, for Ni-L3 edge RIXS, they are 3d valence shell and 2p core shell.

    Parameters
    ----------

    Returns
    -------
    """
    v_name = v_name.strip()
    c_name = c_name.strip()
    info_shell = info_atomic_shell()
    # Quantum numbers of angular momentum
    v_orbl = info_shell[v_name][0]
    c_orbl = info_shell[c_name][0]
    # Effective quantum numbers of angular momentum
    v_orbl_eff = info_shell[v_name][1]

    # number of orbitals without spin
    v_norb1 = info_shell[v_name][2]
    c_norb1 = info_shell[c_name][2]
    # number of orbitals with spin
    v_norb = 2 * v_norb1
    c_norb = 2 * c_norb1
    # total number of orbitals
    ntot = v_norb + c_norb

    emat_i = np.zeros((ntot, ntot), dtype=np.complex)
    emat_n = np.zeros((ntot, ntot), dtype=np.complex)

    # Coulomb interaction
    slater_name = []
    slater_name.extend(['F' + str(i) + '_vv' for i in range(0, 2 * v_orbl + 1, 2)])
    slater_name.extend(['F' + str(i) + '_vc' for i in range(0, min(2*v_orbl, 2*c_orbl) + 1, 2)])
    slater_name.extend(['G' + str(i) + '_vc' for i in
                        range(abs(v_orbl - c_orbl), v_orbl + c_orbl + 1, 2)])
    slater_name.extend(['F' + str(i) + '_cc' for i in range(0, 2 * c_orbl + 1, 2)])
    slater_i = np.zeros(len(slater_name), dtype=np.float)
    slater_n = np.zeros(len(slater_name), dtype=np.float)
    if len(slater_name) > len(slater[0]):
        slater_i[0:len(slater[0])] = slater[0]
    else:
        slater_i[:] = slater[0][0:len(slater_name)]
    if len(slater_name) > len(slater[1]):
        slater_n[0:len(slater[1])] = slater[1]
    else:
        slater_n[:] = slater[1][0:len(slater_name)]

    # print summary of slater integrals
    print("Summary of Slater integrals:")
    print("---------------------------")
    print("Terms,  Initial,  Intermediate")
    for i in range(len(slater_name)):
        print(slater_name[i], ": ", slater_i[i], "    ", slater_n[i])

    case = v_name + c_name
    umat_i = get_umat_slater(case, *slater_i)
    umat_n = get_umat_slater(case, *slater_n)

    write_tensor(umat_i, 'umat_i.dat', only_nonzeros=True)
    write_tensor(umat_n, 'umat_n.dat', only_nonzeros=True)

    # SOC
    emat_i[0:v_norb, 0:v_norb] += atom_hsoc(v_name, v_soc[0])
    emat_n[0:v_norb, 0:v_norb] += atom_hsoc(v_name, v_soc[1])
    emat_n[v_norb:ntot, v_norb:ntot] += atom_hsoc(c_name, c_soc)

    # crystal field
    if cf_mat is not None:
        emat_i[0:v_norb, 0:v_norb] += cf_mat
        emat_n[0:v_norb, 0:v_norb] += cf_mat

    # other hopping matrix
    if other_mat is not None:
        emat_i[0:v_norb, 0:v_norb] += other_mat
        emat_n[0:v_norb, 0:v_norb] += other_mat

    # external magnetic field
    lx, ly, lz = get_lx(v_orbl_eff, True), get_ly(v_orbl_eff, True), get_lz(v_orbl_eff, True)
    sx, sy, sz = get_sx(v_orbl_eff), get_sy(v_orbl_eff), get_sz(v_orbl_eff)
    if v_name == 't2g':
        lx, ly, lz = -lx, -ly, -lz
    zeeman = ext_B[0] * (lx + 2 * sx) + ext_B[1] * (ly + 2 * sy) + ext_B[2] * (lz + 2 * sz)

    emat_i[0:v_norb, 0:v_norb] += zeeman
    emat_n[0:v_norb, 0:v_norb] += zeeman

    basis_i = get_fock_bin_by_N(v_norb, v_noccu, c_norb, c_norb)
    basis_n = get_fock_bin_by_N(v_norb, v_noccu+1, c_norb, c_norb - 1)
    ncfg_i, ncfg_n = len(basis_i), len(basis_n)
    print("Dimension of the initial Hamiltonian: ", ncfg_i)
    print("Dimension of the intermediate Hamiltonian: ", ncfg_n)

    # Build many-body Hamiltonian in Fock basis
    hmat_i = np.zeros((ncfg_i, ncfg_i), dtype=np.complex)
    hmat_n = np.zeros((ncfg_n, ncfg_n), dtype=np.complex)
    hmat_i[:, :] += two_fermion(emat_i, basis_i, basis_i)
    hmat_i[:, :] += four_fermion(umat_i, basis_i)
    hmat_n[:, :] += two_fermion(emat_n, basis_n, basis_n)
    hmat_n[:, :] += four_fermion(umat_n, basis_n)

    # Do exact-diagonalization to get eigenvalues and eigenvectors
    eval_i, evec_i = np.linalg.eigh(hmat_i)
    eval_n, evec_n = np.linalg.eigh(hmat_n)

    # Build dipolar transition operators in local-xyz axis
    tmp = get_trans_oper(case)
    npol, n, m = tmp.shape
    tmp_g = np.zeros((npol, n, m), dtype=np.complex)
    # Transform the transition operators to global-xyz axis
    # dipolar transition
    if npol == 3:
        for i in range(3):
            for j in range(3):
                tmp_g[i] += local_axis[i, j] * tmp[j]

    # quadrupolar transition
    elif npol == 5:
        alpha, beta, gamma = rmat_to_euler(local_axis)
        wignerD = get_wigner_dmat(4, alpha, beta, gamma)
        rotmat = np.dot(np.dot(tmat_r2c('d'), wignerD), np.conj(np.transpose(tmat_r2c('d'))))
        for i in range(5):
            for j in range(5):
                tmp_g[i] += rotmat[i, j] * tmp[j]
    else:
        raise Exception("Have NOT implemented this case: ", npol)

    tmp2 = np.zeros((npol, ntot, ntot), dtype=np.complex)
    T_abs = np.zeros((npol, ncfg_n, ncfg_i), dtype=np.complex)
    for i in range(npol):
        tmp2[i, 0:v_norb, v_norb:ntot] = tmp_g[i]
        T_abs[i] = two_fermion(tmp2[i], basis_n, basis_i)
        T_abs[i] = cb_op2(T_abs[i], evec_n, evec_i)

    return eval_i, eval_n, T_abs


def xas_1v1c(eval_i, eval_n, T_abs,
             thin=np.pi/2.0,
             phi=0,
             alpha=0,
             case='linear',
             scattering_plane_axis=np.eye(3),
             ominc_mesh=np.linspace(1e-6, 10, 100),
             gamma_c=0.1,
             gs=[0],
             temperature=1.0):

    n_om = len(ominc_mesh)
    npol, ncfg_n = T_abs.shape[0], T_abs.shape[1]
    xas = np.zeros(n_om, dtype=np.float)
    gamma_core = np.zeros(n_om, dtype=np.float)
    prob = boltz_dist([eval_i[i] for i in gs], temperature)
    if np.isscalar(gamma_c):
        gamma_core[:] = np.ones(n_om) * gamma_c
    else:
        gamma_core[:] = gamma_c

    polvec = np.zeros(npol, dtype=np.complex)
    if npol == 3:  # Dipolar transition
        polvec[:] = dipole_polvec_xas(thin, phi, alpha, scattering_plane_axis, case)
    if npol == 5:  # Quadrupolar transition
        pol = dipole_polvec_xas(thin, phi, alpha, scattering_plane_axis, case)
        kvec = unit_wavevector(thin, phi, scattering_plane_axis, 'in')
        polvec[:] = quadrupole_polvec(pol, kvec)

    for i, om in enumerate(ominc_mesh):
        for j in gs:
            F_mag = np.zeros(ncfg_n, dtype=np.complex)
            for k in range(npol):
                F_mag += T_abs[k, :, j] * polvec[k]
            xas[i] += (prob[j] * np.sum(np.abs(F_mag)**2 * gamma_core[i] / np.pi /
                       ((om - (eval_n[:] - eval_i[j]))**2 + gamma_core[i]**2)))
    return xas


def rixs_1v1c(eval_i, eval_n, T_abs,
              thin=np.pi/4.0,
              thout=np.pi/4.0,
              phi=0,
              alpha=[(0, 0)],
              pol=[('linear', 'linear')],
              scattering_plane_axis=np.eye(3),
              ominc_mesh=np.linspace(1E-6, 10, 100),
              eloss_mesh=np.linspace(-0.1, 1, 100),
              gamma_c=0.1,
              gamma_f=0.01,
              gs=[0],
              temperature=1.0):

    n_ominc = len(ominc_mesh)
    n_eloss = len(eloss_mesh)
    gamma_core = np.zeros(n_ominc, dtype=np.float)
    gamma_final = np.zeros(n_eloss, dtype=np.float)
    if np.isscalar(gamma_c):
        gamma_core[:] = np.ones(n_ominc) * gamma_c
    else:
        gamma_core[:] = gamma_c

    if np.isscalar(gamma_f):
        gamma_final[:] = np.ones(n_eloss) * gamma_f
    else:
        gamma_final[:] = gamma_f

    prob = boltz_dist([eval_i[i] for i in gs], temperature)
    rixs = np.zeros((len(pol), len(eloss_mesh), len(ominc_mesh)), dtype=np.float)
    ntrans, n, m = T_abs.shape
    T_emi = np.zeros((ntrans, m, n), dtype=np.complex128)
    for i in range(ntrans):
        T_emi[i] = np.conj(np.transpose(T_abs[i]))
    polvec_i = np.zeros(ntrans, dtype=np.complex)
    polvec_f = np.zeros(ntrans, dtype=np.complex)

    # Calculate RIXS
    for i, om in enumerate(ominc_mesh):
        F_fi = scattering_mat(eval_i, eval_n, T_abs[:, :, gs], T_emi, om, gamma_core[i])
        for j, pol_type in enumerate(pol):
            ei, ef = dipole_polvec_rixs(thin, thout, phi, alpha[j][0], alpha[j][1],
                                        scattering_plane_axis, pol_type)
            # dipolar transition
            if ntrans == 3:
                polvec_i[:] = ei
                polvec_f[:] = ef
            # quadrupolar transition
            elif ntrans == 5:
                ki = unit_wavevector(thin, phi, scattering_plane_axis, direction='in')
                kf = unit_wavevector(thout, phi, scattering_plane_axis, direction='out')
                polvec_i[:] = quadrupole_polvec(ei, ki)
                polvec_f[:] = quadrupole_polvec(ef, kf)
            else:
                raise Exception("Have NOT implemented this type of transition operators")
            # scattering magnitude with polarization vectors
            F_mag = np.zeros((len(eval_i), len(gs)), dtype=np.complex)
            for m in range(ntrans):
                for n in range(ntrans):
                    F_mag[:, :] += np.conj(polvec_f[m]) * F_fi[m, n] * polvec_i[n]
            for m in gs:
                for n in range(len(eval_i)):
                    rixs[j, :, i] += (prob[m] * np.abs(F_mag[n, m])**2 * gamma_final / np.pi /
                                      ((eloss_mesh - (eval_i[n] - eval_i[m]))**2 + gamma_final**2))
    return rixs
