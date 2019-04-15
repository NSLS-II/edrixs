#!/usr/bin/env python

import numpy as np
import edrixs


if __name__ == "__main__":
    '''
    Purpose: This example shows how to calculate RIXS spectrum.
             RIXS(eloss, om_inc) = \\sum_{fi} ( |\\sum_{mn} e_{f}^{m} *F_{fi}^{mn} * e_{i}^{n}|^2
             * Gamma/pi / [ (eloss - (E_f-E_i))^2 + Gamma^2 ] * Prob[i] ),
             where, F_{fi}^{mn} = <f|r^m|n><n|r^n|i> / (om_inc - (E_n-E_i) + i*Gamma_c).

             This example use purely python code.
    '''

    # Parameters
    # ----------
    # the parameters for the experimental RIXS geometry are taken from [PRL 117, 147401 (2016)]

    # which initial states are used
    gs_list = list(range(0, 9))
    # temperature
    T = 300

    # the incident angle of X-ray
    thin, thout = 30 / 180.0 * np.pi, 60 / 180.0 * np.pi
    # azimuthal angle
    phi = 0.0

    # resolution of RIXS excitations
    gamma_f = 0.10

    # energy shift of the incident X-ray
    om_shift = 99.7

    # energy mesh for incident X-ray (eV)
    om1, om2 = -5, 20
    nom = 100
    om_mesh = np.linspace(om1, om2, nom)

    # energy loss mesh
    eloss_1, eloss_2 = -0.2, 2.5
    neloss = 1000
    eloss_mesh = np.linspace(eloss_1, eloss_2, neloss)

    # core-hole life-time broadening
    # here, we use an energy dependent one to reproduce the experimental XAS spectrum
    gamma = np.zeros(nom, dtype=np.float64)
    begin, end = 101.5, 104
    npt = 0
    for i in om_mesh:
        if begin < i + om_shift < end:
            npt += 1
    step = (6.5 - 1.05) / 2.0 / npt

    last_one = 0
    for i in range(nom):
        if om_mesh[i] + om_shift <= begin:
            gamma[i] = 1.05 / 2.0
            last_one += 1
        elif begin < om_mesh[i] + om_shift < end:
            gamma[i] = 1.05 / 2.0 + step * (i - last_one)
        else:
            gamma[i] = 6.5 / 2.0

    # END of PARAMETERS
    # -----------------

    # load data, the eigenvalues of the initial and the intermediate
    # Hamiltonian, and the transition matrix
    data = np.loadtxt('eval_i.dat')
    eval_i = data[:, 1]
    data = np.loadtxt('eval_n.dat')
    eval_n = data[:, 1]
    ncfgs_n, ncfgs_i = len(eval_n), len(eval_i)
    # the transition operator for the absorption process
    data = np.loadtxt('trans_mat.dat')
    trans_mat_abs = data[:, 3].reshape((3, ncfgs_n, ncfgs_i)) + \
        1j * data[:, 4].reshape((3, ncfgs_n, ncfgs_i))
    # the transition operator for the emission process
    trans_mat_emi = np.zeros((3, ncfgs_i, ncfgs_n), dtype=np.complex128)
    for i in range(3):
        trans_mat_emi[i] = np.conj(np.transpose(trans_mat_abs[i]))

    print("edrixs >>> calculating RIXS ...")
    rixs = np.zeros((4, neloss, nom), dtype=np.float64)
    gs_prob = edrixs.boltz_dist([eval_i[i] for i in gs_list], T)

    # We calculate RIXS for \pi-\pi, \pi-\sigma, \sigma-\pi, \sigma-\sigma polarizations
    polvec_list = [(0, 0), (0, np.pi / 2.0), (np.pi / 2.0, 0), (np.pi / 2.0, np.pi / 2.0)]

    for i, om_inc in enumerate(om_mesh):
        print(("  incident X-ray energy:   ", i, "    ", om_inc + om_shift))
        F_fi = edrixs.scattering_mat(
            eval_i, eval_n, trans_mat_abs[:, :, gs_list], trans_mat_emi, om_inc, gamma[i])
        for j, (alpha, beta) in enumerate(polvec_list):
            ei, ef = edrixs.dipole_polvec_rixs(thin, thout, phi, alpha, beta)
            F_magnitude = np.zeros((ncfgs_i, len(gs_list)), dtype=np.complex128)
            for m in range(3):
                for n in range(3):
                    F_magnitude[:, :] += ef[m] * F_fi[m, n] * ei[n]
            for m in gs_list:
                for n in range(ncfgs_i):
                    rixs[j, :, i] += np.abs(F_magnitude[n, m])**2 * gamma_f / np.pi / \
                        ((eloss_mesh - (eval_i[n] - eval_i[m]))**2 + gamma_f**2) * gs_prob[m]
    f = open('rixs.dat', 'w')
    for i in range(neloss):
        for j in range(nom):
            str_form = "{:20.10f}" * 6 + "\n"
            line = str_form.format(eloss_mesh[i], om_mesh[j] + om_shift,
                                   rixs[0, i, j], rixs[1, i, j], rixs[2, i, j], rixs[3, i, j])
            f.write(line)
    f.close()
    print("edrixs >>> done !")
