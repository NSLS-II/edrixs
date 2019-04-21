#!/usr/bin/env python


import numpy as np
import edrixs

if __name__ == "__main__":

    # PARAMETERS
    # -----------
    nt2g, nporb = 6, 6
    norbs = nt2g + nporb

    Ud, JH = 0, 0.355
    tmp, F2_dd, F4_dd = edrixs.UdJH_to_F0F2F4(Ud, JH)
    Ud_av = 0.0
    F0_dd = Ud_av + edrixs.get_F0('d', F2_dd, F4_dd)

    scale_dp = 0.9
    G1_dp, G3_dp = 0.894 * scale_dp, 0.531 * scale_dp
    F2_dp = 1.036 * scale_dp
    Udp_av = 0.0
    F0_dp = Udp_av + edrixs.get_F0('dp', G1_dp, G3_dp)

    slater = ([F0_dd, F2_dd, F4_dd], [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp])

    zeta_d_i = 0.33
    zeta_d_n = 0.33
    zeta_p_n = 1009.333333

    om_shift = 11380
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0.0
    gamma_c = 2.5
    gamma_f = 0.075
    ominc = np.linspace(-508 + om_shift, -500 + om_shift, 100)
    eloss = np.linspace(-0.5, 2.0, 1000)
    gs_list = list(range(0, 4))
    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0), 
                    ('linear', np.pi / 2.0, 'linear', 0.0), 
                    ('linear', np.pi / 2.0, 'linear', np.pi / 2.0)] 
    # END of PARAMETERS
    # -----------------
    # Run ED
    result = edrixs.ed_1v1c(v_name='t2g', c_name='p', v_soc=(zeta_d_i, zeta_d_n), c_soc=zeta_p_n,
                            c_level=-om_shift, v_noccu=3, slater=slater)
    eval_i, eval_n, trans_op = result

    # Run RIXS
    rixs = edrixs.rixs_1v1c(eval_i, eval_n, trans_op, ominc, eloss, gamma_c, gamma_f,
                            thin, thout, phi, poltype=poltype_rixs,
                            gs_list=gs_list, temperature=300)
    rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
    rixs_sigma = np.sum(rixs[:, :, 2:4], axis=2)
    np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
    np.savetxt('rixs_sigma.dat', np.concatenate((np.array([eloss]).T, rixs_sigma.T), axis=1))
    # Plot RIXS map
    edrixs.plot_rixs_map(rixs_pi, ominc, eloss, "rixsmap_pi.pdf")
    edrixs.plot_rixs_map(rixs_sigma, ominc, eloss, "rixsmap_sigma.pdf")
