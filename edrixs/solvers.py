#!/usr/bin/env python

import numpy as np
from .utils import info_atomic_shell, boltz_dist
from .coulomb_utensor import get_umat_slater
from .iostream import write_tensor
from .soc import atom_hsoc
from .angular_momentum import get_sx, get_sy, get_sz, get_lx, get_ly, get_lz
from .manybody_operator import two_fermion, four_fermion
from .fock_basis import get_fock_bin_by_N
from .basis_transform import cb_op2
from .photon_transition import get_trans_oper, get_quadrupolar_polvec


def ed_1v1c(v_name='d', c_name='p', v_level=0.0, c_level=0.0, v_soc=(0.0, 0.0), c_soc=0.0,
            v_noccu=1, slater=([0], [0]), ext_B=np.zeros(3, dtype=np.float),
            cf_mat=None, other_mat=None, loc_axis=np.eye(3, dtype=np.float)):
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
    npol = tmp.shape[0]
    tmp2 = np.zeros((npol, ntot, ntot), dtype=np.complex)
    T_abs = np.zeros((npol, ncfg_n, ncfg_i), dtype=np.complex)
    T_abs_g = np.zeros((npol, ncfg_n, ncfg_i), dtype=np.complex)

    for i in range(npol):
        tmp2[i, 0:v_norb, v_norb:ntot] = tmp[i]
        T_abs[i] = two_fermion(tmp2[i], basis_n, basis_i)
        T_abs[i] = cb_op2(T_abs[i], evec_n, evec_i)

    # Transform the transition operators to global-xyz axis
    # dipolar transition
    if npol == 3:
        for i in range(3):
            T_abs_g[i] = (T_abs[0] * loc_axis[i, 0] +
                          T_abs[1] * loc_axis[i, 1] +
                          T_abs[2] * loc_axis[i, 2])
    # TODO
    # quadrupolar transition
    # else if npol == 5:

    return eval_i, eval_n, T_abs_g


def xas_1v1c(eval_i, eval_n, T_abs,
             ominc_mesh=np.linspace(-10, 10, 100),
             gamma=np.ones(100)*0.1,
             pol=np.ones(3)/np.sqrt(3.0),
             kvec=np.array([0, 0, 1]),
             gs=[0],
             temperature=1.0):

    n_om = len(ominc_mesh)
    npol, ncfg_n = T_abs.shape[0], T_abs.shape[1]
    xas = np.zeros(n_om, dtype=np.float)
    gamma_core = np.zeros(n_om, dtype=np.float)
    prob = boltz_dist([eval_i[i] for i in gs], temperature)
    if np.isscalar(gamma):
        gamma_core[:] = np.ones(n_om) * gamma
    else:
        gamma_core[:] = gamma

    polvec = np.zeros(npol, dtype=np.float)
    if npol == 3:  # Dipolar transition
        polvec[:] = pol
    if npol == 5:  # Quadrupolar transition
        polvec[:] = get_quadrupolar_polvec(pol, kvec)

    for i, om in enumerate(ominc_mesh):
        for j in gs:
            F_mag = np.zeros(ncfg_n, dtype=np.complex)
            for k in range(npol):
                F_mag += T_abs[k, :, j] * polvec[k]
            xas[i] += (prob[j] * np.sum(np.abs(F_mag)**2 * gamma_core[i] / np.pi /
                       ((om - (eval_n[:] - eval_i[j]))**2 + gamma_core[i]**2)))
    return xas
