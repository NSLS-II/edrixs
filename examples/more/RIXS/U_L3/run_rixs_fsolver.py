#!/usr/bin/env python

import json
import numpy as np
import edrixs
from mpi4py import MPI

if __name__ == "__main__":
    '''
    U-5f2 :math:`L_{3}`-edge, :math:`2p_{3/2}\\rightarrow 6d` transition.

    How to run
    ----------
    mpiexec -n 4 python run_rixs_fsolver.py
    '''

    F2_ff, F4_ff, F6_ff = 8.278, 5.447, 4.011
    F0_ff = edrixs.get_F0('f', F2_ff, F4_ff, F6_ff)

    F2_dd, F4_dd = 0.0, 0.0
    F0_dd = edrixs.get_F0('d', F2_dd, F4_dd)

    F2_fd, F4_fd = 3.750, 2.050
    G1_fd, G3_fd, G5_fd = 1.938, 1.562, 1.213
    F0_fd = edrixs.get_F0('fd', G1_fd, G3_fd, G5_fd)

    F2_fp = 0.528
    G2_fp, G4_fp = 0.087, 0.056
    F0_fp = edrixs.get_F0('fp', G2_fp, G4_fp)

    F2_dp = 0.272
    G1_dp, G3_dp = 0.238, 0.142
    F0_dp = edrixs.get_F0('dp', G1_dp, G3_dp)

    slater = (
        [F0_ff, F2_ff, F4_ff, F6_ff,  # FX_11
         F0_fd, F2_fd, F4_fd, G1_fd, G3_fd, G5_fd,  # FX_12, GX_12
         F0_dd, F2_dd, F4_dd  # FX_22
         ],  # Initial
        [F0_ff, F2_ff, F4_ff, F6_ff,  # FX_11
         F0_fd, F2_fd, F4_fd, G1_fd, G3_fd, G5_fd,  # FX_12, GX_12
         F0_dd, F2_dd, F4_dd,  # FX_22
         F0_fp, F2_fp, G2_fp, G4_fp,  # FX_13, GX_13
         F0_dp, F2_dp, G1_dp, G3_dp   # FX_23, GX_23
         ]   # Intermediate
    )

    # Spin-Orbit Coupling (SOC) zeta
    zeta_f_i = 0.261
    zeta_f_n = 0.321

    zeta_d_i = 0.435
    zeta_d_n = 0.435

    # Occupancy of U 5f orbitals
    noccu = 2

    # RIXS settings
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0.0
    gamma_c = 1.0
    gamma_f = 0.1
    ominc_xas = np.linspace(-10, 20, 1000)
    ominc_rixs = np.linspace(0, 10, 10)
    eloss = np.linspace(-0.2, 5, 1000)

    poltype_xas = [('isotropic', 0.0)]

    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0)]

    shell_name = ('f', 'd', 'p32')

    # mpi4py env
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Run ED
    eval_i, denmat = edrixs.ed_2v1c_fort(
        comm, shell_name, shell_level=(0, 5.0, 0),
        v1_soc=(zeta_f_i, zeta_f_n), v2_soc=(zeta_d_i, zeta_d_n),
        v_tot_noccu=noccu, slater=slater, ed_solver=2, neval=20,
        nvector=2, ncv=50, idump=True
    )
    if rank == 0:
        print('eigvals:', eval_i)
        print('occupancy numbers:', denmat[0].diagonal())

    # Run XAS
    xas, xas_poles = edrixs.xas_2v1c_fort(
        comm, shell_name, ominc_xas, gamma_c=gamma_c,
        v_tot_noccu=noccu, trans_to_which=2, thin=thin, phi=phi,
        pol_type=poltype_xas, num_gs=1, nkryl=200, temperature=300
    )
    if rank == 0:
        np.savetxt('xas.dat', np.concatenate((np.array([ominc_xas]).T, xas), axis=1))
        with open('xas_poles.json', 'w') as f:
            json.dump(xas_poles, f, indent=4)

    # Run RIXS
    rixs, rixs_poles = edrixs.rixs_2v1c_fort(
        comm, shell_name, ominc_rixs, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
        v_tot_noccu=noccu, trans_to_which=2, thin=thin, thout=thout, phi=phi,
        pol_type=poltype_rixs, num_gs=1, nkryl=200, linsys_max=1000, temperature=300
    )

    if rank == 0:
        with open('rixs_poles.json', 'w') as f:
            json.dump(rixs_poles, f, indent=4)
        rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
        np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
        edrixs.plot_rixs_map(rixs_pi, ominc_rixs, eloss, "rixsmap_pi.pdf")
