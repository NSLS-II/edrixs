#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import edrixs

if __name__ == "__main__":
    F2_dd, F4_dd = 12.234 * 0.65, 7.598 * 0.65
    F0_dd = edrixs.get_F0('d', F2_dd, F4_dd)

    G1_dp, G3_dp = 5.787 * 0.7, 3.291 * 0.7
    F0_dp = edrixs.get_F0('dp', G1_dp, G3_dp)
    F2_dp = 7.721 * 0.95

    slater = ([F0_dd, F2_dd, F4_dd], [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp])

    zeta_d_i = 0.083
    zeta_d_n = 0.102
    zeta_p_n = 11.24

    dq10, d1, d3 = 1.3, 0.05, 0.2
    cf = edrixs.cf_tetragonal_d(dq10, d1, d3)

    off = 857.4
    eval_i, eval_n, trans_op = edrixs.ed_1v1c(v_name='d', c_name='p',
                                           v_soc=(zeta_d_i, zeta_d_n),
                                           c_soc=zeta_p_n,
                                           c_level=-off, v_noccu=8,
                                           slater=slater, cf_mat=cf)

    omega=np.linspace(off - 10, off + 20, 1000)
    thin = 15 / 180.0 * np.pi
    poltype=[('linear', 0.0), ('linear', np.pi / 2.0),
             ('left', 0.0), ('right', 0.0), ('isotropic', 0.0)]

    xas = edrixs.xas_1v1c(eval_i, eval_n, trans_op, thin,
                          poltype=poltype,
                          ominc_mesh=omega, gamma_c=0.2,
                          gs_list=[0, 1, 2], temperature=300)

    np.savetxt('xas_2.dat', np.concatenate((np.array([omega]).T, xas), axis=1))

    thin, thout = 15 / 180.0 * np.pi, 75 / 180.0 * np.pi
    phi=0
    om1, om2 = -5.9+off, -0.9+off
    ominc_mesh = np.linspace(-5.9 + off, -0.9 + off, 100)
    #ominc_mesh = np.linspace(10.9 + off, 14.9 + off, 100)
    eloss_mesh = np.linspace(-0.5, 5.0, 1000)
    poltype=[('linear', 0, 'linear', 0),
             ('linear', 0, 'linear', np.pi/2.0),
             ('linear', np.pi/2.0, 'linear', 0.0),
             ('linear', np.pi/2.0, 'linear', np.pi/2.0)]
    rixs = edrixs.rixs_1v1c(eval_i, eval_n, trans_op, 
                            thin, thout, phi,
                            poltype=poltype,
                            ominc_mesh=ominc_mesh,
                            eloss_mesh=eloss_mesh,
                            gamma_c=0.2,
                            gamma_f=0.1,
                            gs_list=[0, 1, 2],
                            temperature=300)

    plt.figure(figsize=(16, 8))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(np.sum(rixs[:,:,0:2], axis=2), extent=[min(ominc_mesh), max(ominc_mesh), min(eloss_mesh), max(eloss_mesh)],
               origin='lower', aspect='auto', cmap='rainbow', interpolation='gaussian')
    plt.ylabel(r'Energy loss (eV)')
    plt.xlabel(r'Energy of incident photon (eV)')
    
    ax = plt.subplot(1, 2, 2)
    plt.imshow(np.sum(rixs[:,:,2:4], axis=2), extent=[min(ominc_mesh), max(ominc_mesh), min(eloss_mesh), max(eloss_mesh)],
               origin='lower', aspect='auto', cmap='rainbow', interpolation='gaussian')
    plt.ylabel(r'Energy loss (eV)')
    plt.xlabel(r'Energy of incident photon (eV)')

    plt.savefig('rixs_map_2.pdf')
