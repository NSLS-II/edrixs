#!/usr/bin/env python

import numpy as np
import edrixs


if __name__ == "__main__":
    # local axis
    natom = 4
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

    U, J = 2.0, 0.3
    Ud, JH = edrixs.UJ_to_UdJH(U, J)
    F0_dd, F2_dd, F4_dd = edrixs.UdJH_to_F0F2F4(Ud, JH)
    F0_dd = edrixs.get_F0('d', F2_dd, F4_dd)

    G1_dp, G3_dp = 0.957, 0.569
    F0_dp, F2_dp = edrixs.get_F0('dp', G1_dp, G3_dp), 1.107

    slater = ([F0_dd, F2_dd, F4_dd], [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp])

    zeta_d, zeta_p = 0.45, 1072.6666666666667

    # Trigonal crystal field
    c = -0.15
    tmp = np.array([[0, c, c],
                    [c, 0, c],
                    [c, c, 0]])
    cf = np.zeros((6, 6), dtype=np.complex)
    cf[0:6:2, 0:6:2] += tmp
    cf[1:6:2, 1:6:2] += tmp
    cf[:,:] = edrixs.cb_op(cf, edrixs.tmat_r2c('t2g', True))

    gamma_c, gamma_f = 2.5, 0.075
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0
    scattering_plane = np.array([[1 / np.sqrt(6.0), 1 / np.sqrt(6.0), -2 / np.sqrt(6.0)],
                                 [-1 / np.sqrt(2.0), 1 / np.sqrt(2.0), 0],
                                 [1 / np.sqrt(3.0), 1 / np.sqrt(3.0), 1 / np.sqrt(3.0)]]).T
    om_shift = 11751.333333
    ominc = np.linspace(-540 + om_shift, -530 + om_shift, 100)
    eloss = np.linspace(-0.1, 1.2, 1000)
    gs_list = range(0, 2)
    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0), 
                    ('linear', np.pi / 2.0, 'linear', 0.0), 
                    ('linear', np.pi / 2.0, 'linear', np.pi / 2.0)] 

    rixs = np.zeros((len(ominc), len(eloss), len(poltype_rixs), natom), dtype=np.float)
    for iatom in range(natom):
        result = edrixs.ed_1v1c(v_name='t2g', c_name='p', v_soc=(zeta_d, zeta_d),
                                c_soc=zeta_p, c_level=-om_shift, v_noccu=5, cf_mat=cf,
                                slater=slater, local_axis=loc_axis[iatom])
        eval_i, eval_n, trans_op = result
        rixs[:, :, :, iatom] = edrixs.rixs_1v1c(eval_i, eval_n, trans_op, ominc, eloss,
                                                gamma_c, gamma_f, thin, thout, phi,
                                                poltype=poltype_rixs, gs_list=gs_list,
                                                scattering_plane_axis=scattering_plane,
                                                temperature=300)
    rixs_tot = np.sum(rixs, axis=3)
    rixs_pi = np.sum(rixs_tot[:, :, 0:2], axis=2)
    rixs_sigma = np.sum(rixs_tot[:, :, 2:4], axis=2)
    np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
    np.savetxt('rixs_sigma.dat', np.concatenate((np.array([eloss]).T, rixs_sigma.T), axis=1))
    # Plot RIXS map
    edrixs.plot_rixs_map(rixs_pi, ominc, eloss, "rixsmap_pi.pdf")
    edrixs.plot_rixs_map(rixs_sigma, ominc, eloss, "rixsmap_sigma.pdf")
