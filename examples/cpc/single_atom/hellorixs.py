#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import edrixs

font = {'weight': 'medium',
        'size': '18'}
plt.rc('font', **font)


def ed():
    # 1-10: Ni-3d valence orbitals, 11-16: Ni-2p core orbitals
    # Single particle basis: complex shperical Harmonics
    ndorb, nporb, ntot = 10, 6, 16
    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)

    # 4-index Coulomb interaction tensor, parameterized by
    # Slater integrals, which are obtained from Cowan's code
    F2_d, F4_d = 7.9521, 4.9387
    # Averaged dd Coulomb interaction is set to be zero
    F0_d = edrixs.get_F0('d', F2_d, F4_d)
    G1_dp, G3_dp = 4.0509, 2.3037
    # Averaged dp Coulomb interaction is set to be zero
    F0_dp, F2_dp = edrixs.get_F0('dp', G1_dp, G3_dp), 7.33495
    umat_i = edrixs.get_umat_slater('dp', F0_d, F2_d, F4_d,  # dd
                                    0, 0, 0, 0,        # dp
                                    0, 0)              # pp
    umat_n = edrixs.get_umat_slater('dp', F0_d, F2_d, F4_d,           # dd
                                    F0_dp, F2_dp, G1_dp, G3_dp,  # dp
                                    0, 0)                       # pp

    # Atomic spin-orbit coupling
    zeta_d, zeta_p = 0.083, 11.24
    emat_i[0:ndorb, 0:ndorb] += edrixs.atom_hsoc('d', zeta_d)
    emat_n[0:ndorb, 0:ndorb] += edrixs.atom_hsoc('d', zeta_d)
    emat_n[ndorb:ntot, ndorb:ntot] += edrixs.atom_hsoc('p', zeta_p)

    # Tetragonal crystal field splitting terms,
    # which are first defined in the real cubic Harmonics basis,
    # and then transformed to complex shperical Harmonics basis.
    dt, ds, dq = 0.011428, 0.035714, 0.13
    tmp = np.zeros((5, 5), dtype=complex)
    tmp[0, 0] = 6 * dq - 2 * ds - 6 * dt   # d3z2-r2
    tmp[1, 1] = -4 * dq - 1 * ds + 4 * dt   # dzx
    tmp[2, 2] = -4 * dq - 1 * ds + 4 * dt   # dzy
    tmp[3, 3] = 6 * dq + 2 * ds - 1 * dt   # dx2-y2
    tmp[4, 4] = -4 * dq + 2 * ds - 1 * dt   # dxy
    tmp[:, :] = edrixs.cb_op(tmp, edrixs.tmat_r2c('d'))
    emat_i[0:ndorb:2, 0:ndorb:2] += tmp
    emat_i[1:ndorb:2, 1:ndorb:2] += tmp
    emat_n[0:ndorb:2, 0:ndorb:2] += tmp
    emat_n[1:ndorb:2, 1:ndorb:2] += tmp

    # Build Fock basis in its binary form
    basis_i = edrixs.get_fock_bin_by_N(ndorb, 8, nporb, nporb)
    basis_n = edrixs.get_fock_bin_by_N(ndorb, 9, nporb, nporb - 1)
    ncfg_i, ncfg_n = len(basis_i), len(basis_n)

    # Build many-body Hamiltonian in Fock basis
    hmat_i = np.zeros((ncfg_i, ncfg_i), dtype=complex)
    hmat_n = np.zeros((ncfg_n, ncfg_n), dtype=complex)
    hmat_i[:, :] += edrixs.two_fermion(emat_i, basis_i, basis_i)
    hmat_i[:, :] += edrixs.four_fermion(umat_i, basis_i)
    hmat_n[:, :] += edrixs.two_fermion(emat_n, basis_n, basis_n)
    hmat_n[:, :] += edrixs.four_fermion(umat_n, basis_n)

    # Do exact-diagonalization to get eigenvalues and eigenvectors
    eval_i, evec_i = np.linalg.eigh(hmat_i)
    eval_n, evec_n = np.linalg.eigh(hmat_n)

    # Build dipolar transition operators
    dipole = np.zeros((3, ntot, ntot), dtype=complex)
    T_abs = np.zeros((3, ncfg_n, ncfg_i), dtype=complex)
    T_emi = np.zeros((3, ncfg_i, ncfg_n), dtype=complex)
    tmp = edrixs.get_trans_oper('dp')
    for i in range(3):
        dipole[i, 0:ndorb, ndorb:ntot] = tmp[i]
        # First, in the Fock basis
        T_abs[i] = edrixs.two_fermion(dipole[i], basis_n, basis_i)
        # Then, transfrom to the eigenvector basis
        T_abs[i] = edrixs.cb_op2(T_abs[i], evec_n, evec_i)
        T_emi[i] = np.conj(np.transpose(T_abs[i]))

    return eval_i, eval_n, T_abs, T_emi


def xas(eval_i, eval_n, T_abs):
    gs = list(range(0, 3))  # three lowest eigenstates are used
    prob = edrixs.boltz_dist([eval_i[i] for i in gs], 300)
    off = 857.4  # offset of the energy of the incident x-ray
    Gam_c = 0.2  # core-hole life-time broadening
    pol = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)  # isotropic
    omega = np.linspace(-10, 20, 1000)
    xas = np.zeros(len(omega), dtype=float)
    # Calculate XAS spectrum
    for i, om in enumerate(omega):
        for j in gs:
            F_mag = (T_abs[0, :, j] * pol[0] +
                     T_abs[1, :, j] * pol[1] +
                     T_abs[2, :, j] * pol[2])

            xas[i] += prob[j] * np.sum(np.abs(F_mag)**2 * Gam_c / np.pi /
                                       ((om - (eval_n[:] - eval_i[j]))**2 + Gam_c**2))

    # plot XAS
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.ylim([0, 0.6])
    plt.plot(omega + off, xas, '-', color="blue")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.xlabel(r'Energy of incident photon (eV)')
    plt.ylabel(r'XAS Intensity (a.u.)')
    plt.text(852, 0.52, r'$L_{3}$', fontsize=20)
    plt.text(869.5, 0.15, r'$L_{2}$', fontsize=20)
    plt.text(846.5, 0.55, r'(a)', fontsize=25)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.13,
                        top=0.95, wspace=0.05, hspace=0.00)
    plt.savefig('xas_ni.pdf')


def rixs(eval_i, eval_n, T_abs, T_emi):
    gs = list(range(0, 3))
    prob = edrixs.boltz_dist([eval_i[i] for i in gs], 300)
    off = 857.4
    phi, thin, thout = 0, 15 / 180.0 * np.pi, 75 / 180.0 * np.pi
    Gam_c, Gam = 0.2, 0.1  # core-hole life-time and RIXS resolution
    pol = [(0, 0), (0, np.pi / 2.0)]  # pi-pi and pi-sigma polarization
    omega = np.linspace(-5.9, -0.9, 100)
    eloss = np.linspace(-0.5, 5.0, 1000)
    rixs = np.zeros((len(pol), len(eloss), len(omega)), dtype=float)

    # Calculate RIXS
    for i, om in enumerate(omega):
        F_fi = edrixs.scattering_mat(eval_i, eval_n, T_abs[:, :, gs], T_emi, om, Gam_c)
        for j, (alpha, beta) in enumerate(pol):
            ei, ef = edrixs.dipole_polvec_rixs(thin, thout, phi, alpha, beta)
            F_mag = np.zeros((len(eval_i), len(gs)), dtype=complex)
            for m in range(3):
                for n in range(3):
                    F_mag[:, :] += ef[m] * F_fi[m, n] * ei[n]
            for m in gs:
                for n in range(len(eval_i)):
                    rixs[j, :, i] += (prob[m] * np.abs(F_mag[n, m])**2 * Gam / np.pi /
                                      ((eloss - (eval_i[n] - eval_i[m]))**2 + Gam**2))

    # plot RIXS map
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    a, b, c, d = min(omega) + off, max(omega) + off, min(eloss), max(eloss)
    plt.imshow(rixs[0] + rixs[1], extent=[a, b, c, d],
               origin='lower', aspect='auto', cmap='rainbow',
               interpolation='bicubic')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.xlabel(r'Energy of incident photon (eV)')
    plt.ylabel(r'Energy loss (eV)')
    plt.text(851.6, 4.5, r'(b)', fontsize=25)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.13,
                        top=0.95, wspace=0.05, hspace=0.00)
    plt.savefig("rixs_map_ni.pdf")


if __name__ == "__main__":
    print("edrixs >>> ED ...")
    eval_i, eval_n, T_abs, T_emi = ed()
    print("edrixs >>> Done ! ")

    print("edrixs >>> Calculate XAS ...")
    xas(eval_i, eval_n, T_abs)
    print("edrixs >>> Done ! ")

    print("edrixs >>> Calculate RIXS ...")
    rixs(eval_i, eval_n, T_abs, T_emi)
    print("edrixs >>> Done ! ")
