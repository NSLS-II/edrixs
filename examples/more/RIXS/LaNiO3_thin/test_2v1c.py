#!/usr/bin/env python

import numpy as np
import edrixs
from mpi4py import MPI

if __name__ == "__main__":
    '''
    Valence shells: 3d4p, Core shell: 2p
    Ni :math:`L_{2,3}`-edge, :math:`2p_{1/2,3/2}\\rightarrow 3d` transition.

    Used to test the ed_2v1c_fort, xas_2v1c_fort, rixs_2v1c_fort solvers.

    How to run
    ----------
    mpiexec -n 2 python test_2v1c.py
    '''

    F2_dd, F4_dd = 12.234 * 0.65, 7.598 * 0.65
    F0_dd = edrixs.get_F0('d', F2_dd, F4_dd)

    G1_d4p, G3_d4p = 5.787 * 0.7, 3.291 * 0.7
    F0_d4p = edrixs.get_F0('dp', G1_d4p, G3_d4p)
    F2_d4p = 7.721 * 0.95

    F0_4p4p, F2_4p4p = 0.0, 0.0

    F2_d2p = 7.721 * 0.95
    G1_d2p, G3_d2p = 5.787 * 0.7, 3.291 * 0.7
    F0_d2p = edrixs.get_F0('dp', G1_d2p, G3_d2p)

    F0_4p2p = 2.0
    F2_4p2p = 2.0
    G0_4p2p = 2.0
    G2_4p2p = 2.0

    slater = (
        [F0_dd, F2_dd, F4_dd,  # FX_11
         F0_d4p, F2_d4p, G1_d4p, G3_d4p,  # FX_12, GX_12
         F0_4p4p, F2_4p4p
         ],  # Initial
        [F0_dd, F2_dd, F4_dd,  # FX_11
         F0_d4p, F2_d4p, G1_d4p, G3_d4p,  # FX_12, GX_12
         F0_4p4p, F2_4p4p,  # FX_22
         F0_d2p, F2_d2p, G1_d2p, G3_d2p,  # FX_13, GX_13
         F0_4p2p, F2_4p2p, G0_4p2p, G2_4p2p  # FX_23, GX_23
         ]   # Intermediate
    )

    # Spin-Orbit Coupling (SOC) zeta
    zeta_d_i = 0.083
    zeta_d_n = 0.102
    zeta_p_n = 11.24

    # Occupancy of U 5f orbitals
    noccu = 8

    # Tetragonal crystal field splitting
    dq10, d1, d3 = 1.3, 0.05, 0.2
    cf = edrixs.cf_tetragonal_d(dq10, d1, d3)

    # Energy shift of the core level
    off = 857.4

    # XAS and RIXS settings
    # ---------------------
    ominc_xas = np.linspace(off - 10, off + 20, 1000)
    thin, thout, phi = 15 / 180.0 * np.pi, 75 / 180.0 * np.pi, 0.0
    poltype_xas = [('isotropic', 0.0)]
    gamma_c, gamma_f = 0.2, 0.1
    # L3-edge
    ominc_rixs = np.linspace(-5.9 + off, -0.9 + off, 10)
    # L2-edge
    # ominc_rixs = np.linspace(10.9 + off, 14.9 + off, 100)
    eloss = np.linspace(-0.5, 5.0, 1000)
    poltype_rixs = [('linear', 0, 'linear', 0),
                    ('linear', 0, 'linear', np.pi/2.0)]

    shell_name = ('d', 'p', 'p')

    # mpi4py env
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Run ED
    eval_i, denmat = edrixs.ed_2v1c_fort(
        comm, shell_name, shell_level=(0, 3, -off),
        v1_soc=(zeta_d_i, zeta_d_n), c_soc=zeta_p_n, v1_cfmat=cf,
        v_tot_noccu=noccu, slater=slater,
        ed_solver=2, neval=20, nvector=3, ncv=50, idump=True
    )

    if rank == 0:
        print("Eigenvalues: ", eval_i)
        print("Occupancy of orbitals: ", denmat[0].diagonal())

    # Run XAS
    xas, xas_poles = edrixs.xas_2v1c_fort(
        comm, shell_name, ominc_xas, gamma_c=gamma_c, v_tot_noccu=noccu, trans_to_which=1,
        thin=thin, phi=phi, pol_type=poltype_xas, num_gs=3, nkryl=100, temperature=300
    )
    if rank == 0:
        np.savetxt('xas.dat', np.concatenate((np.array([ominc_xas]).T, xas), axis=1))
        edrixs.dump_poles(xas_poles, 'xas_poles')
    # Run RIXS
    rixs, rixs_poles = edrixs.rixs_2v1c_fort(
        comm, shell_name, ominc_rixs, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
        v_tot_noccu=noccu, trans_to_which=1, thin=thin, thout=thout, phi=phi,
        pol_type=poltype_rixs, num_gs=3, nkryl=100, temperature=300
    )
    if rank == 0:
        edrixs.dump_poles(rixs_poles, 'rixs_poles')
        rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
        np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
        edrixs.plot_rixs_map(rixs_pi, ominc_rixs, eloss, "rixsmap_pi.pdf")
