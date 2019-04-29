#!/usr/bin/env python


import numpy as np
from mpi4py import MPI
import edrixs


if __name__ == "__main__":
    """
    Os :math:`L_{3}`-edge, :math:`2p_{3/2}\\rightarrow t_{2g}` transition.
    Single Impurity Anderson Model, only :math:`t_{2g}` orbitals are considered.
    1 impuirty site plus 3 bath sites
    """

    U, J = 6.0, 0.45  # Kanamori U, J
    Ud, JH = edrixs.UJ_to_UdJH(U, J)
    F0_dd, F2_dd, F4_dd = edrixs.UdJH_to_F0F2F4(Ud, JH)

    scale_dp = 0.9
    G1_dp, G3_dp = 0.894 * scale_dp, 0.531 * scale_dp
    F2_dp = 1.036 * scale_dp
    Udp_av = 0.0
    F0_dp = Udp_av + edrixs.get_F0('dp', G1_dp, G3_dp)

    slater = ([F0_dd, F2_dd, F4_dd],  # initial
              [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp])  # intermediate

    nbath = 3  # number of bath sites
    norb = 6   # number of orbitals for each site

    # bath energy level & hybridization strength
    bath_level = np.zeros((nbath, norb), dtype=np.complex)
    hyb = np.zeros((nbath, norb), dtype=np.complex)
    e1 = [-3.528372861516216, -1.207615555008166, 6.183392524170465]
    e2 = [-3.538890277572901, -1.308137924011529, 5.578400781316763]
    v1 = [2.131943931119928, 0.481470627582356, 1.618130554107574]
    v2 = [2.108388983547592, 0.517200653226272, 1.674013405873617]
    bath_level[:, 0] = bath_level[:, 1] = e1  # j=1/2
    bath_level[:, 2] = bath_level[:, 3] = bath_level[:, 4] = bath_level[:, 5] = e2  # j=3/2
    hyb[:, 0] = hyb[:, 1] = v1  # j=1/2
    hyb[:, 2] = hyb[:, 3] = hyb[:, 4] = hyb[:, 5] = v2  # j=3/2

    # impurity matrix
    imp_mat = np.zeros((6, 6), dtype=np.complex)
    imp_mat[0, 0] = imp_mat[1, 1] = -14.7627972353
    imp_mat[2, 2] = imp_mat[3, 3] = imp_mat[4, 4] = imp_mat[5, 5] = -15.4689430453

    trans_c2n = edrixs.tmat_c2j(1)
  
    shell_name = ('t2g', 'p32')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    do_ed = 1
    eval_i, denmat, noccu_gs = edrixs.ed_siam_fort(
        comm, shell_name, nbath,
        siam_type=0, imp_mat=imp_mat, bath_level=bath_level, hyb=hyb,
        static_core_pot=2.0, slater=slater, trans_c2n=trans_c2n, v_noccu=5,
        do_ed=do_ed, ed_solver=2, neval=20, nvector=5, ncv=50, idump=True
    )

    if rank == 0:
        if do_ed != 2:
            print(noccu_gs)
            print(eval_i)
            print(np.sum(denmat[0].diagonal()[0:norb]).real) 
