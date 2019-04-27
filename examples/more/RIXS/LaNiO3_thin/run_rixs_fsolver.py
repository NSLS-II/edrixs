#!/usr/bin/env python

import numpy as np
import edrixs
from mpi4py import MPI

if __name__ == "__main__":
    # Atomic shell settings
    # ---------------------
    # Slater integrals
    F2_dd, F4_dd = 12.234 * 0.65, 7.598 * 0.65
    F0_dd = edrixs.get_F0('d', F2_dd, F4_dd)
    G1_dp, G3_dp = 5.787 * 0.7, 3.291 * 0.7
    F0_dp = edrixs.get_F0('dp', G1_dp, G3_dp)
    F2_dp = 7.721 * 0.95
    slater = (
        [F0_dd, F2_dd, F4_dd],  # Initial
        [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp]   # Intermediate
    )

    # Spin-orbit coupling strength
    zeta_d_i, zeta_d_n, zeta_p_n = 0.083, 0.102, 11.24

    # Tetragonal crystal field splitting
    dq10, d1, d3 = 1.3, 0.05, 0.2
    cf = edrixs.cf_tetragonal_d(dq10, d1, d3)

    # Energy shift of the core level
    off = 857.4

    # XAS and RIXS settings
    # ---------------------
    ominc_xas = np.linspace(off - 10, off + 20, 1000)
    thin, thout, phi = 15 / 180.0 * np.pi, 75 / 180.0 * np.pi, 0.0
    poltype_xas = [('linear', 0.0), ('linear', np.pi / 2.0),
                   ('left', 0.0), ('right', 0.0), ('isotropic', 0.0)]
    gamma_c, gamma_f = 0.2, 0.1
    # L3-edge
    ominc_rixs = np.linspace(-5.9 + off, -0.9 + off, 100)
    # L2-edge
    # ominc_rixs = np.linspace(10.9 + off, 14.9 + off, 100)
    eloss = np.linspace(-0.5, 5.0, 1000)
    poltype_rixs = [('linear', 0, 'linear', 0),
                    ('linear', 0, 'linear', np.pi/2.0),
                    ('linear', np.pi/2.0, 'linear', 0.0),
                    ('linear', np.pi/2.0, 'linear', np.pi/2.0)]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    shell_name = ('d', 'p')
    # Run ED
    v_norb, c_norb, eval_i, denmat = edrixs.ed_1v1c_fort(
        comm, shell_name, shell_level=(0, -off), v_soc=(zeta_d_i, zeta_d_n), c_soc=zeta_p_n,
        v_noccu=8, slater=slater, v_cfmat=cf, ed_solver=0, neval=10, nvector=3, idump=True
    )

    # Run XAS
    xas, xas_poles = edrixs.xas_1v1c_fort(
        comm, shell_name, ominc_xas, gamma_c=gamma_c, v_noccu=8, thin=thin, phi=phi,
        num_gs=3, nkryl=400, pol_type=poltype_xas, temperature=300
    )
    np.savetxt('xas.dat', np.concatenate((np.array([ominc_xas]).T, xas), axis=1))

    # Run RIXS
    rixs, rixs_poles = edrixs.rixs_1v1c_fort(
        comm, shell_name, ominc_rixs, eloss, gamma_c=gamma_c, gamma_f=gamma_f, thin=thin, thout=thout,
        phi=phi, v_noccu=8, pol_type=poltype_rixs, num_gs=3, nkryl=200, temperature=300
    )

    rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
    rixs_sigma = np.sum(rixs[:, :, 2:4], axis=2)
    np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
    np.savetxt('rixs_sigma.dat', np.concatenate((np.array([eloss]).T, rixs_sigma.T), axis=1))
    # Plot RIXS map
    edrixs.plot_rixs_map(rixs_pi, ominc_rixs, eloss, "rixsmap_pi.pdf")
    edrixs.plot_rixs_map(rixs_sigma, ominc_rixs, eloss, "rixsmap_sigma.pdf")
