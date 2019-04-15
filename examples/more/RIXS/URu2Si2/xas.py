#!/usr/bin/env python

import numpy as np
import edrixs

if __name__ == "__main__":
    '''
    Purpose: This example shows how to calculate XAS spectrum:
        XAS(om) = \\sum_{i,n} |<n|D|i>|^2 * Gamma/Pi / [ (om-(E_n-E_i))^2 + Gamma^2 ] * Prob(i),
        where D=ei_{x} * x + ei_{y} * y + ei_{z} *z.

        This example use purely python code.
    '''

    # PARAMETERS
    # ----------

    # which initial states are used
    gs_list = list(range(0, 9))
    # temperature
    T = 300

    # energy mesh of the incident X-ray (eV)
    om1, om2 = -5, 20
    nom = 1000
    om_mesh = np.linspace(om1, om2, nom)

    # energy shift to match the experimental XAS spectrum
    om_shift = 99.7

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

    # isotropic polarization
    ei = np.array([1, 1, 1]) / np.sqrt(3.0)

    # END of PARAMETERS
    # -----------------

    print("edrixs >>> calculating XAS ...")
    # load data, the eigenvalues of the initial and the intermediate Hamiltonian,
    # and the transition matrix
    data = np.loadtxt('eval_i.dat')
    eval_i = data[:, 1]
    data = np.loadtxt('eval_n.dat')
    eval_n = data[:, 1]
    ncfgs_n, ncfgs_i = len(eval_n), len(eval_i)
    data = np.loadtxt('trans_mat.dat')
    trans_mat = (data[:, 3].reshape((3, ncfgs_n, ncfgs_i)) +
                 1j * data[:, 4].reshape((3, ncfgs_n, ncfgs_i)))

    xas = np.zeros(nom, dtype=np.float64)

    # get boltzmann distribution at T for these ground states
    gs_prob = edrixs.boltz_dist([eval_i[i] for i in gs_list], T)

    for i, om_inc in enumerate(om_mesh):
        for j in gs_list:
            scatt_mat = (trans_mat[0, :, j] * ei[0] + trans_mat[1, :, j] * ei[1]
                         + trans_mat[2, :, j] * ei[2])
            xas[i] += np.sum(np.abs(scatt_mat)**2 * gamma[i] / np.pi /
                             ((om_inc - (eval_n[:] - eval_i[j]))**2 + gamma[i]**2)) * gs_prob[j]

    f = open('xas.dat', 'w')
    f.write("#       omega_inc              isotropic \n")
    for i in range(nom):
        f.write("{:20.10f}{:20.10f}\n".format(om_shift + om_mesh[i], xas[i]))
    f.close()
    print("edrixs >>> Done !")
