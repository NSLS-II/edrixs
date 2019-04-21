#!/usr/bin/env python


import numpy as np
import edrixs


if __name__ == "__main__":
    """
    Purpose:   Os: L2/L3 edge  2p --> 5d
    """

    # PARAMETERS
    # ----------

    # 10 5d-orbitals, 6 2p-orbitals (including spin)
    ndorb, nporb = 10, 6
    # total orbitals
    norbs = ndorb + nporb

    # Slater integrals
    scale_d = 0.335
    F2_dd, F4_dd = 8.885 * scale_d, 5.971 * scale_d
    # averaged Coulomb interaction for 5d
    Ud = 0.0
    # F0_d is split into two parts: (1) Ud, (2) related to F2_d, F4_d
    F0_dd = Ud + edrixs.get_F0('d', F2_dd, F4_dd)

    scale_dp = 0.9
    F2_dp = 1.036 * scale_dp
    G1_dp, G3_dp = 0.894 * scale_dp, 0.531 * scale_dp

    # averaged Coulomb interaction between 3d and 2p shells
    Udp = 0.0
    # F0_dp is split into two parts: (1) Udp, (2) related to G1_dp, G3_dp
    F0_dp = Udp + edrixs.get_F0('dp', G1_dp, G3_dp)

    slater = ([F0_dd, F2_dd, F4_dd], [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp])

    # Spin-Orbit Coupling (SOC) zeta
    # 5d, without core-hole
    zeta_d_i = 0.33
    # 5d, with core-hole
    zeta_d_n = 0.33
    # 2p core electron, and usually we will adjust it to match the energy
    # difference of the two edges,
    # for example, roughly, E(L2-edge) - E(L3-edge) = 1.5 * zeta_p_n
    zeta_p_n = 1009.333333

    # Cubic crystal field
    dq10 = 4.3

    cf = edrixs.cf_cubic_d(dq10)

    om_shift = 11380
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0.0
    gamma_c = 2.5
    gamma_f = 0.075
    ominc = np.linspace(-508 + om_shift, -500 + om_shift, 100)
    eloss = np.linspace(-0.5, 5.5, 1000)
    gs_list = list(range(0, 4))
    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0), 
                    ('linear', np.pi / 2.0, 'linear', 0.0), 
                    ('linear', np.pi / 2.0, 'linear', np.pi / 2.0)] 

    # END of PARAMETERS
    # -----------------

    # Run ED
    result = edrixs.ed_1v1c(v_name='d', c_name='p', v_soc=(zeta_d_i, zeta_d_n), c_soc=zeta_p_n,
                            c_level=-om_shift, v_noccu=3, slater=slater, cf_mat=cf)
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
