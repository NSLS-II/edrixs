#!/usr/bin/env python


import numpy as np
from mpi4py import MPI
import edrixs


if __name__ == "__main__":
    """
    Os :math:`L_{3}`-edge, :math:`2p_{3/2}\\rightarrow t_{2g}` transition.
    Single Impurity Anderson Model, only :math:`t_{2g}` orbitals are considered.
    1 impuirty site plus 3 bath sites

    How to run
    ----------
    mpiexec -n 4 python run_rixs_fsolver.py
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
    bath_level = np.zeros((nbath, norb), dtype=complex)
    hyb = np.zeros((nbath, norb), dtype=complex)
    e1 = [-3.528372861516216, -1.207615555008166, 6.183392524170465]
    e2 = [-3.538890277572901, -1.308137924011529, 5.578400781316763]
    v1 = [2.131943931119928, 0.481470627582356, 1.618130554107574]
    v2 = [2.108388983547592, 0.517200653226272, 1.674013405873617]
    bath_level[:, 0] = bath_level[:, 1] = e1  # j=1/2
    bath_level[:, 2] = bath_level[:, 3] = bath_level[:, 4] = bath_level[:, 5] = e2  # j=3/2
    hyb[:, 0] = hyb[:, 1] = v1  # j=1/2
    hyb[:, 2] = hyb[:, 3] = hyb[:, 4] = hyb[:, 5] = v2  # j=3/2

    # impurity matrix
    imp_mat = np.zeros((6, 6), dtype=complex)
    imp_mat[0, 0] = imp_mat[1, 1] = -14.7627972353
    imp_mat[2, 2] = imp_mat[3, 3] = imp_mat[4, 4] = imp_mat[5, 5] = -15.4689430453

    trans_c2n = edrixs.tmat_c2j(1)
    shell_name = ('t2g', 'p32')

    # RIXS settings
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0.0
    gamma_c = 2.5
    gamma_f = 0.075
    om_shift = 10877.2
    ominc_xas = np.linspace(-20+om_shift, 10+om_shift, 1000)
    ominc_rixs = np.linspace(-6.2+om_shift, -6.2+om_shift, 1)
    eloss = np.linspace(-0.2, 7, 1000)
    poltype_xas = [('isotropic', 0)]
    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0)]

    # mpi4py
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    do_ed = 0
    eval_i, denmat, noccu_gs = edrixs.ed_siam_fort(
        comm, shell_name, nbath, siam_type=0, imp_mat=imp_mat,
        bath_level=bath_level, hyb=hyb, c_level=-om_shift,
        static_core_pot=2.0, slater=slater, trans_c2n=trans_c2n, v_noccu=15,
        do_ed=do_ed, ed_solver=2, neval=10, nvector=4, ncv=50, idump=True
    )
    if rank == 0 and do_ed != 2:
        print(noccu_gs)
        print(eval_i)
        print(np.sum(denmat[0].diagonal()[0:norb]).real)

    v_noccu = noccu_gs
    # Run XAS
    xas, xas_poles = edrixs.xas_siam_fort(
        comm, shell_name, nbath, ominc_xas, gamma_c=gamma_c, v_noccu=v_noccu, thin=thin,
        phi=phi, num_gs=4, nkryl=200, pol_type=poltype_xas, temperature=300
    )
    np.savetxt('xas.dat', np.concatenate((np.array([ominc_xas]).T, xas), axis=1))
    edrixs.dump_poles(xas_poles, 'xas_poles')

    # Run RIXS
    rixs, rixs_poles = edrixs.rixs_siam_fort(
        comm, shell_name, nbath, ominc_rixs, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
        thin=thin, thout=thout, phi=phi, v_noccu=v_noccu, pol_type=poltype_rixs,
        num_gs=4, nkryl=500, temperature=300, linsys_tol=1e-10
    )
    rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
    np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
    edrixs.dump_poles(rixs_poles, 'rixs_poles')
