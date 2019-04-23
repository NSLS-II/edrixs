#!/usr/bin/env python


import numpy as np
import edrixs
from mpi4py import MPI

if __name__ == "__main__":
    '''
    Sm :math:`L_{3}`-edge, :math:`2p_{3/2}\\rightarrow 5d` transition.
    '''

    F2_ff, F4_ff, F6_ff = 7.0, 5.0, 4.0
    F0_ff = edrixs.get_F0('f', F2_ff, F4_ff, F6_ff)

    F2_dd, F4_dd = 7.0, 5.0
    F0_dd = edrixs.get_F0('d', F2_dd, F4_dd)

    F2_fd, F4_fd = 6.0, 4.0
    G1_fd, G3_fd, G5_fd = 7.0, 5.0, 3.0
    F0_fd = edrixs.get_F0('fd', G1_fd, G3_fd, G5_fd)

    F2_fp = 0.5
    G2_fp, G4_fp = 0.5, 0.5
    F0_fp = edrixs.get_F0('fp', G2_fp, G4_fp)

    F2_dp = 0.5
    G1_dp, G3_dp = 0.5, 0.5
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
    zeta_f_i = 0.3
    zeta_f_n = 0.3

    # Occupancy of U 5f orbitals
    noccu = 6

    # mpi4py env
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Run ED
    result = edrixs.ed_2v1c(comm, v1_name='f', v2_name='d', c_name='p32',
                            v1_soc=(zeta_f_i, zeta_f_n),
                            v2_level=1, v_tot_noccu=noccu, slater=slater,
                            ed_solver=2, neval=10, nvector=1, ncv=30, idump=True)
    eval_i, denmat, num_val_orbs, num_core_orbs = result

    if rank == 0:
        print(num_val_orbs, num_core_orbs)
        print('eigvals:')
        print(eval_i)
        print('occupancy numbers:')
        print(denmat[0].diagonal())
        print('total occupancy:')
        print(np.sum(denmat[0].diagonal()).real)
