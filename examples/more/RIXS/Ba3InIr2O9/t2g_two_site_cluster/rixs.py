#!/usr/bin/env python


import numpy as np
import copy
import edrixs

np.set_printoptions(precision=10, linewidth=200, suppress=True)

if __name__ == '__main__':
    Ir_pos = np.array([[2.9164499, 1.6838131, 2.3064262],
                       [2.9164499, 1.6838131, 4.9373740]], dtype=np.float64)

    ncfgs_gs, ncfgs_ex = 220, 396
    data = np.loadtxt('eval_i.dat')
    eval_gs = data[:, 1]
    data = np.loadtxt('eval_n.dat')
    eval_ex = data[:, 2].reshape((2, ncfgs_ex))

    data = np.loadtxt('trans_mat.dat')
    trans_mat_abs = (data[:, 4].reshape((2, 3, ncfgs_ex, ncfgs_gs)) +
                     1j * data[:, 5].reshape((2, 3, ncfgs_ex, ncfgs_gs)))
    trans_mat_emi = np.zeros((2, 3, ncfgs_gs, ncfgs_ex), dtype=np.complex128)
    for i in range(2):
        for j in range(3):
            trans_mat_emi[i, j] = np.conj(np.transpose(trans_mat_abs[i, j]))

    gamma_c, gamma = 5.0 / 2.0, 0.08
    phi, thin, thout = 0.0, 30 / 180.0 * np.pi, 60 / 180.0 * np.pi
    pol_vec = [(0, 0), (0, np.pi / 2.0)]

    omi_shift = 11755
    omi_mesh = np.linspace(-546, -536, 21)
    eloss_mesh = np.linspace(-0.2, 1.5, 1000)
    gs_list = range(0, 2)
    gs_dist = edrixs.boltz_dist([eval_gs[i] for i in gs_list], 30)

    rixs = np.zeros((len(pol_vec), len(eloss_mesh), len(omi_mesh)), dtype=np.float64)
    for omi_ind, omi in enumerate(omi_mesh):
        print(omi_ind, omi, omi + omi_shift)
        Ffg_Ir = np.zeros((2, 3, 3, ncfgs_gs, len(gs_list)), dtype=np.complex128)
        for i in range(0, 2):
            abs_trans = copy.deepcopy(trans_mat_abs[i])
            emi_trans = copy.deepcopy(trans_mat_emi[i])
            Ffg_Ir[i] = edrixs.scattering_mat(
                eval_gs, eval_ex[i], abs_trans[:, :, gs_list], emi_trans, omi, gamma_c)
        print('------------------------')
        for ipol, (alpha, beta) in enumerate(pol_vec):
            ein, eout = omi + omi_shift, omi + omi_shift
            ei, ef = edrixs.dipole_polvec_rixs(thin, thout, phi, alpha, beta, local_axis=np.eye(3))
            kin, kout = edrixs.get_wavevector_rixs(
                thin, thout, phi, ein, eout, local_axis=np.eye(3))
            Q = kout - kin
            phase_Ir = np.zeros(2, dtype=np.complex128)
            for i in range(0, 2):
                phase_Ir[i] = np.exp(-1j * np.dot(Q, Ir_pos[i]))
            Ffg = np.zeros((ncfgs_gs, len(gs_list)), dtype=np.complex128)
            for i in range(0, 2):
                for j in range(3):
                    for k in range(3):
                        Ffg[:, :] += ef[j] * Ffg_Ir[i, j, k] * phase_Ir[i] * ei[k]
            for i in gs_list:
                for j in range(ncfgs_gs):
                    rixs[ipol, :, omi_ind] += np.abs(Ffg[j, i])**2 * gamma / (2 * np.pi) / (
                        (eloss_mesh - (eval_gs[j] - eval_gs[i]))**2 + (gamma / 2.0)**2) * gs_dist[i]

    f = open('rixs_tot.dat', 'w')
    rixs_tot = np.zeros(len(eloss_mesh), dtype=np.float64)
    for i in range(len(eloss_mesh)):
        rixs_tot[i] = np.sum(rixs[0, i, :] + rixs[1, i, :]) / len(omi_mesh)
        line = "{:20.10f}{:20.10f}\n".format(eloss_mesh[i], rixs_tot[i])
        f.write(line)
    f.close()

    f = open('rixs.dat', 'w')
    for i in range(len(eloss_mesh)):
        for j in range(len(omi_mesh)):
            line = "{:20.10f}{:20.10f}{:20.10f}{:20.10f}\n".format(
                eloss_mesh[i], omi_mesh[j] + omi_shift, rixs[0, i, j], rixs[1, i, j])
            f.write(line)
    f.close()
