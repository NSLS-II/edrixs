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
    zeta_d_i = 0.3
    zeta_d_n = 0.3
    zeta_p_i = 0.3
    zeta_p_n = 0.3

    # Occupancy of U 5f orbitals
    noccu = 13

    # RIXS settings
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0.0
    gamma_c = 0.2
    gamma_f = 0.1
    ominc = np.linspace(0, 20, 100)
    #poltype_xas = [('isotropic', 0.0), ('left', 0), ('right', 0)]
    poltype_xas = [('isotropic', 0)]

    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0),
                    ('linear', np.pi / 2.0, 'linear', 0.0),
                    ('linear', np.pi / 2.0, 'linear', np.pi / 2.0)]

    # mpi4py env
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Run ED
    result = edrixs.ed_2v1c(
        comm, v1_name='d', v2_name='p', c_name='s', v1_soc=(zeta_d_i, zeta_d_n), v2_soc=(zeta_p_i, zeta_p_n),
        v2_level=1.0, v_tot_noccu=noccu, slater=slater,
        ed_solver=0, neval=20, nvector=2, ncv=40, idump=True
    )
    v_norb, c_norb, emat_i, emat_n, umat_i, umat_n, eval_i, denmat = result

    # Run XAS
    xas, poles_dict = edrixs.xas_2v1c(
        comm, ominc, gamma_c, v1_name='d', v2_name='p', c_name='s',
        v_tot_noccu=noccu, trans_to_which=2, thin=thin, phi=phi, poltype=poltype_xas,
        num_gs=1, nkryl=300, temperature=300
    )
    if rank == 0:
        np.savetxt('xas.dat', np.concatenate((np.array([ominc]).T, xas), axis=1))
