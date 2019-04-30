#!/usr/bin/env python

import pickle
import numpy as np
from mpi4py import MPI
import edrixs

if __name__ == "__main__":
    '''
    Pu-5f6 :math:`O_{4,5}`-edge, :math:`5d_{3/2,5/2}\\rightarrow 5f` transition.
    '''

    # PARAMETERS
    # -----------
    F2_ff, F4_ff, F6_ff = 9.711 * 0.77, 6.364 * 0.77, 4.677 * 0.77
    Uf_av = 0.0
    F0_ff = Uf_av + edrixs.get_F0('f', F2_ff, F4_ff, F6_ff)

    F2_fd, F4_fd = 10.652 * 0.6, 6.850 * 0.6
    G1_fd, G3_fd, G5_fd = 12.555 * 0.6, 7.768 * 0.6, 5.544 * 0.6
    Ufd_av = 0.0
    F0_fd = Ufd_av + edrixs.get_F0('fd', G1_fd, G3_fd, G5_fd)
    slater = (
        [F0_ff, F2_ff, F4_ff, F6_ff],  # Initial
        [F0_ff, F2_ff, F4_ff, F6_ff, F0_fd, F2_fd, F4_fd, G1_fd, G3_fd, G5_fd]   # Intermediate
    )

    # Spin-Orbit Coupling (SOC) zeta
    # 5f, without core-hole, from Cowan's code
    zeta_f_i = 0.261 * 0.9
    # 5f, with core-hole, from Cowan's code
    zeta_f_n = 0.274 * 0.9
    zeta_d_n = 4.4

    # Occupancy of U 5f orbitals
    noccu = 6

    # Energy shift of the core level
    om_shift = 106.4

    # XAS and RIXS settings
    # ---------------------
    # The incident, scattered and azimuthal angles
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0.0

    # Number of initial states are used
    num_gs = 1

    # energy mesh of the incident X-ray (eV)
    ominc_xas = np.linspace(-5 + om_shift, 20 + om_shift, 1000)
    ominc_rixs = np.linspace(-5 + om_shift, 20 + om_shift, 10)
    neloss = 1000
    eloss = np.linspace(-0.2, 5, neloss)

    # core-hole and final states life-time broadening
    # here, we use an energy dependent one to reproduce the experimental XAS spectrum
    gamma_c = 0.2
    gamma_f = 0.1
    poltype_xas = [('isotropic', 0.0)]
    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0)]
    shell_name = ('f', 'd')
    T = 300

    # END of Parameters
    # ------------------

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Run ED
    eval_i, denmat = edrixs.ed_1v1c_fort(
        comm, shell_name, shell_level=(0, -om_shift), v_soc=(zeta_f_i, zeta_f_n),
        c_soc=zeta_d_n, v_noccu=noccu, slater=slater, ed_solver=2,
        neval=30, ncv=60, nvector=9, idump=True

    )

    # Run XAS
    xas, xas_poles = edrixs.xas_1v1c_fort(
        comm, shell_name, ominc_xas, gamma_c=gamma_c, v_noccu=noccu, thin=thin, phi=phi,
        num_gs=num_gs, nkryl=200, pol_type=poltype_xas, temperature=T
    )
    if rank == 0:
        np.savetxt('xas.dat', np.concatenate((np.array([ominc_xas]).T, xas), axis=1))
        f=open('xas_poles.pkl', 'wb')
        pickle.dump(xas_poles, f)
        f.close() 

    # Run RIXS
    rixs, rixs_poles = edrixs.rixs_1v1c_fort(
        comm, shell_name, ominc_rixs, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
        thin=thin, thout=thout, phi=phi, v_noccu=noccu, pol_type=poltype_rixs,
        num_gs=num_gs, nkryl=200, temperature=T
    )
    if rank == 0:
        f=open('rixs_poles.pkl', 'wb')
        pickle.dump(rixs_poles, f)
        f.close() 

        rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
        rixs_sigma = np.sum(rixs[:, :, 2:4], axis=2)
        np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
        np.savetxt('rixs_sigma.dat', np.concatenate((np.array([eloss]).T, rixs_sigma.T), axis=1))
        # Plot RIXS map
        edrixs.plot_rixs_map(rixs_pi, ominc_rixs, eloss, "rixsmap_pi.pdf")
        edrixs.plot_rixs_map(rixs_sigma, ominc_rixs, eloss, "rixsmap_sigma.pdf")
