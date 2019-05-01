#!/usr/bin/env python


import numpy as np
import edrixs

if __name__ == "__main__":
    '''
    U-5f2 :math:`O_{4,5}`-edge, :math:`5d_{3/2,5/2}\\rightarrow 5f` transition.

    How to run
    ----------
    python run_rixs_pysolver.py
    '''

    # PARAMETERS
    # -----------
    # The parameters used in this calculation are taken from [PRL 114, 236401 (2015)]
    # 14 5f-orbitals, 10 5d-orbitals (including spin)

    # Slater integrals
    # The Slater integrals are calculated by the Cowan's code:
    # [https://www.tcd.ie/Physics/people/Cormac.McGuinness/Cowan/]
    # by using Hartree-Fock mean-field method, and then re-scaled,
    # for example, Fff: 77%, Ffd: 60%, Gfd: 60%
    F2_ff, F4_ff, F6_ff = 9.711 * 0.77, 6.364 * 0.77, 4.677 * 0.77
    # Here, Uf_av is the averaged Coulomb interaction in 5f shell, we set it to be zero
    # It doesn't matter here because we are doing single atomic calculation with fixed 5f occupancy.
    Uf_av = 0.0
    # F0_ff is split into two parts: (1) Uf_av, (2) related to F2_f, F4_f, F6_f
    F0_ff = Uf_av + edrixs.get_F0('f', F2_ff, F4_ff, F6_ff)

    F2_fd, F4_fd = 10.652 * 0.6, 6.850 * 0.6
    G1_fd, G3_fd, G5_fd = 12.555 * 0.6, 7.768 * 0.6, 5.544 * 0.6
    Ufd_av = 0.0
    F0_fd = Ufd_av + edrixs.get_F0('fd', G1_fd, G3_fd, G5_fd)
    slater = (
        [F0_ff, F2_ff, F4_ff, F6_ff],  # Initial
        [F0_ff, F2_ff, F4_ff, F6_ff, F0_fd, F2_fd, F4_fd, G1_fd, G3_fd, G5_fd]   # Intermediate
    )

    # Spin-Orbit Coupling (SOC) zeta
    # 5f, without core-hole, from Cowan's code
    zeta_f_i = 0.261 * 0.9
    # 5f, with core-hole, from Cowan's code
    zeta_f_n = 0.274 * 0.9
    # 5d core electron, from Cowan's code, and usually we will adjust it to match the
    # energy difference of the two edges,
    # for example, roughly, E(O4-edge) - E(O5-edge) = 2.5 * zeta_d_n
    zeta_d_n = 3.2

    # Occupancy of U 5f orbitals
    noccu = 2

    # Energy shift of the core level
    om_shift = 99.7

    # XAS and RIXS settings
    # ---------------------
    # The incident, scattered and azimuthal angles
    thin, thout, phi = 30 / 180.0 * np.pi, 60 / 180.0 * np.pi, 0.0

    # Which initial states are used
    gs_list = list(range(0, 9))

    # energy mesh of the incident X-ray (eV)
    nom = 100
    ominc = np.linspace(-5 + om_shift, 20 + om_shift, nom)
    neloss = 1000
    eloss = np.linspace(-0.2, 2.5, neloss)

    # core-hole and final states life-time broadening
    # here, we use an energy dependent one to reproduce the experimental XAS spectrum
    gamma_c = np.zeros(nom, dtype=np.float64)
    gamma_f = 0.1
    begin, end = 101.5, 104
    npt = 0
    for i in ominc:
        if begin < i < end:
            npt += 1
    step = (6.5 - 1.05) / 2.0 / npt
    last_one = 0
    for i in range(nom):
        if ominc[i] <= begin:
            gamma_c[i] = 1.05 / 2.0
            last_one += 1
        elif begin < ominc[i] < end:
            gamma_c[i] = 1.05 / 2.0 + step * (i - last_one)
        else:
            gamma_c[i] = 6.5 / 2.0

    poltype_xas = [('isotropic', 0.0)]
    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0),
                    ('linear', np.pi / 2.0, 'linear', 0.0),
                    ('linear', np.pi / 2.0, 'linear', np.pi / 2.0)]
    # END of Parameters
    # ------------------

    # Run ED
    eval_i, eval_n, trans_op = edrixs.ed_1v1c_py(
        ('f', 'd'), shell_level=(0, -om_shift), v_soc=(zeta_f_i, zeta_f_n),
        c_soc=zeta_d_n, v_noccu=noccu, slater=slater
    )

    # Run XAS
    xas = edrixs.xas_1v1c_py(
        eval_i, eval_n, trans_op, ominc, gamma_c=gamma_c, thin=thin, phi=phi,
        pol_type=poltype_xas, gs_list=gs_list, temperature=300
    )

    np.savetxt('xas.dat', np.concatenate((np.array([ominc]).T, xas), axis=1))

    # Run RIXS
    rixs = edrixs.rixs_1v1c_py(
        eval_i, eval_n, trans_op, ominc, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
        thin=thin, thout=thout, phi=phi, pol_type=poltype_rixs, gs_list=gs_list,
        temperature=300
    )

    rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
    rixs_sigma = np.sum(rixs[:, :, 2:4], axis=2)
    np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
    np.savetxt('rixs_sigma.dat', np.concatenate((np.array([eloss]).T, rixs_sigma.T), axis=1))
    # Plot RIXS map
    edrixs.plot_rixs_map(rixs_pi, ominc, eloss, "rixsmap_pi.pdf")
    edrixs.plot_rixs_map(rixs_sigma, ominc, eloss, "rixsmap_sigma.pdf")
