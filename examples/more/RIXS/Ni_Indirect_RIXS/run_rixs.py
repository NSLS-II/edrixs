#!/usr/bin/env python


import numpy as np
import edrixs
from mpi4py import MPI

if __name__ == "__main__":
    '''
    Ni :math:`K`-edge, :math:`1s\\rightarrow 3p` transition.
    '''

    F2_dd, F4_dd = 7.0, 5.0
    F0_dd = edrixs.get_F0('d', F2_dd, F4_dd)

    F2_pp = 7.0
    F0_pp = edrixs.get_F0('p', F2_pp)

    F2_dp = 6.0
    G1_dp, G3_dp = 7.0, 5.0
    F0_dp = edrixs.get_F0('dp', G1_dp, G3_dp)

    G2_ds = 0.5
    F0_ds = edrixs.get_F0('ds', G2_ds)

    G1_ps = 0.5
    F0_ps = edrixs.get_F0('ps', G1_ps)

    slater = (
        [F0_dd, F2_dd, F4_dd,  # FX_11
         F0_dp, F2_dp, G1_dp, G3_dp,  # FX_12, GX_12
         F0_pp, F2_pp  # FX_22
         ],  # Initial
        [F0_dd, F2_dd, F4_dd,  # FX_11
         F0_dp, F2_dp, G1_dp, G3_dp,  # FX_12, GX_12
         F0_pp, F2_pp,  # FX_22
         F0_ds, G2_ds,  # FX_13, GX_13
         F0_ps, G1_ps   # FX_23, GX_23
         ]   # Intermediate
    )

    # Spin-Orbit Coupling (SOC) zeta
    zeta_d_i = 0.05
    zeta_d_n = 0.05
    zeta_p_i = 0.0
    zeta_p_n = 0.0

    # Occupancy of U 5f orbitals
    noccu = 12

    # RIXS settings
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0.0
    gamma_c = 0.2
    gamma_f = 0.1
    ominc_xas = np.linspace(3, 10, 1000)
    ominc_rixs = np.linspace(3, 10, 10)
    eloss = np.linspace(-0.2, 5, 1000)
    poltype_xas = [('isotropic', 0.0)]

    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0)]

    shell_name = ('d', 'p', 's')

    # mpi4py env
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Run ED
    v_norb, c_norb, eval_i, denmat = edrixs.ed_2v1c(
        comm, shell_name, shell_level=(0, 3.0, 0),
        v1_soc=(zeta_d_i, zeta_d_n), v2_soc=(zeta_p_i, zeta_p_n),
        v_tot_noccu=noccu, slater=slater,
        ed_solver=2, neval=30, nvector=10, ncv=50, idump=True
    )

    if rank == 0:
        print("Eigenvalues: ", eval_i)
        print("Occupancy of orbitals: ", denmat[0].diagonal())

    # Run XAS
    xas, poles_dict = edrixs.xas_2v1c(
        comm, shell_name, ominc_xas, gamma_c=gamma_c, v_tot_noccu=noccu, trans_to_which=2,
        thin=thin, phi=phi, pol_type=poltype_xas, num_gs=1, nkryl=300, temperature=300
    )

    if rank == 0:
        np.savetxt('xas.dat', np.concatenate((np.array([ominc_xas]).T, xas), axis=1))

    # Run RIXS
    rixs, poles_dict = edrixs.rixs_2v1c(
        comm, shell_name, ominc_rixs, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
        v_tot_noccu=noccu, trans_to_which=2, thin=thin, thout=thout, phi=phi,
        pol_type=poltype_rixs, num_gs=1, nkryl=300, temperature=300
    )

    if rank == 0:
        rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
        np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
        edrixs.plot_rixs_map(rixs_pi, ominc_rixs, eloss, "rixsmap_pi.pdf")
