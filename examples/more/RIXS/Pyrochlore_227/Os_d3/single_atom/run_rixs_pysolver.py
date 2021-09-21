#!/usr/bin/env python

import numpy as np
import edrixs


if __name__ == "__main__":
    """
    Os-:math:`d^3`, :math:`L_{3}`-edge, :math:`2p_{3/2}\\rightarrow 5d (t_{2g})` transition.

    How to run
    ----------
    python run_rixs_pysolver.py
    """
    natom = 4
    # local axis
    loc_axis = np.zeros((natom, 3, 3), dtype=np.float64)
    loc_axis[0] = np.array([[-2.0 / 3.0, +1.0 / 3.0, -2.0 / 3.0],
                            [+1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
                            [-2.0 / 3.0, -2.0 / 3.0, +1.0 / 3.0]]).T
    loc_axis[1] = np.array([[+2.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0],
                            [-1.0 / 3.0, +2.0 / 3.0, -2.0 / 3.0],
                            [+2.0 / 3.0, +2.0 / 3.0, +1.0 / 3.0]]).T
    loc_axis[2] = np.array([[+2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
                            [+2.0 / 3.0, +1.0 / 3.0, +2.0 / 3.0],
                            [-1.0 / 3.0, -2.0 / 3.0, +2.0 / 3.0]]).T
    loc_axis[3] = np.array([[-2.0 / 3.0, +2.0 / 3.0, -1.0 / 3.0],
                            [-2.0 / 3.0, -1.0 / 3.0, +2.0 / 3.0],
                            [+1.0 / 3.0, +2.0 / 3.0, +2.0 / 3.0]]).T

    # Kanamori U, J
    U, J = 1.5, 0.25
    Ud, JH = edrixs.UJ_to_UdJH(U, J)
    F0_dd, F2_dd, F4_dd = edrixs.UdJH_to_F0F2F4(Ud, JH)
    F0_dd = edrixs.get_F0('d', F2_dd, F4_dd)

    G1_dp, G3_dp = 0.957, 0.569
    F0_dp, F2_dp = edrixs.get_F0('dp', G1_dp, G3_dp), 1.107

    slater = (
        [F0_dd, F2_dd, F4_dd, 0.000, 0.000, 0.000, 0.000],  # Initial
        [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp]   # Intermediate
    )

    # SOC of 5d
    zeta_d = 0.45

    # Trigonal crystal field
    cf = edrixs.cf_trigonal_t2g(-0.1)

    # Occupancy of 5d
    noccu = 3

    # Core-hole and final states life-time broadening
    gamma_c, gamma_f = 2.5, 0.075

    # RIXS settings
    thin, thout, phi = 56 / 180.0 * np.pi, 34 / 180.0 * np.pi, 0
    scattering_plane = edrixs.zx_to_rmat([0, 1 / np.sqrt(2.0), 1 / np.sqrt(2.0)], [-1, 0, 0])

    om_shift = 10871
    ominc = np.linspace(-5, 5, 100)
    eloss = np.linspace(-0.1, 2, 1000)
    gs_list = range(0, 2)
    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0),
                    ('linear', np.pi / 2.0, 'linear', 0.0),
                    ('linear', np.pi / 2.0, 'linear', np.pi / 2.0)]

    rixs = np.zeros((len(ominc), len(eloss), len(poltype_rixs), natom), dtype=float)
    # Loop over all the atoms
    for iatom in range(natom):
        print()
        print("edrixs >>> atom: ", iatom)
        c_name = edrixs.edge_to_shell_name('L3')
        eval_i, eval_n, trans_op = edrixs.ed_1v1c_py(
            ('t2g', c_name), v_soc=(zeta_d, zeta_d), v_noccu=noccu, v_cfmat=cf,
            slater=slater, loc_axis=loc_axis[iatom]
        )

        rixs[:, :, :, iatom] = edrixs.rixs_1v1c_py(
            eval_i, eval_n, trans_op, ominc, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
            thin=thin, thout=thout, phi=phi, pol_type=poltype_rixs, gs_list=gs_list,
            scatter_axis=scattering_plane, temperature=300
        )

    # Summation of all the atoms
    rixs_tot = np.sum(rixs, axis=3)
    # Summation of polarization
    rixs_pi = np.sum(rixs_tot[:, :, 0:2], axis=2)
    rixs_sigma = np.sum(rixs_tot[:, :, 2:4], axis=2)
    np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
    np.savetxt('rixs_sigma.dat', np.concatenate((np.array([eloss]).T, rixs_sigma.T), axis=1))
    # Plot RIXS map
    edrixs.plot_rixs_map(rixs_pi, ominc + om_shift, eloss, "rixsmap_pi.pdf")
    edrixs.plot_rixs_map(rixs_sigma, ominc + om_shift, eloss, "rixsmap_sigma.pdf")
