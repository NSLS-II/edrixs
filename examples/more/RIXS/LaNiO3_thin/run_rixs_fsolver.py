#!/usr/bin/env python

import numpy as np
import edrixs
from mpi4py import MPI

if __name__ == "__main__":
    """
    Ni-3d8: :math:`L_{2,3}`-edge, transition from :math:`2p_{1/2,3/2}\\rightarrow 3d`.

    Use Fortran ED, XAS, RIXS solvers.

    In this example, we call function edrixs.get_atom_data to return the default atomic
    parameters of Ni, inlucding slater integrals, SOC, resonant energy of :math:`L_{2,3}`-edge
    and core-hole life-time broadening.

    How to run
    ----------
    mpiexec -n 2 python run_rixs_fsolver.py
    """
    # Atomic shell settings
    # ---------------------
    res = edrixs.get_atom_data('Ni', v_name='3d', v_noccu=8, edge='L23')
    name_i, slater_i = [list(i) for i in zip(*res['slater_i'])]  # Initial
    name_n, slater_n = [list(i) for i in zip(*res['slater_n'])]  # Intermediate

    # Slater integrals
    si = edrixs.rescale(slater_i, ([1, 2], [0.65, 0.65]))
    si[0] = edrixs.get_F0('d', si[1], si[2])  # F0_dd

    sn = edrixs.rescale(slater_n, ([1, 2, 4, 5, 6], [0.65, 0.65, 0.95, 0.7, 0.7]))
    sn[0] = edrixs.get_F0('d', sn[1], sn[2])  # F0_dd
    sn[3] = edrixs.get_F0('dp', sn[5], sn[6])  # F0_dp

    slater = (si, sn)

    # Spin-orbit coupling strength
    zeta_d_i, zeta_d_n = res['v_soc_i'][0], res['v_soc_n'][0]
    zeta_p_n = (res['edge_ene'][0] - res['edge_ene'][1]) / 1.5

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

    # L2-edge
    # gamma_c, gamma_f = res['gamma_c'][0], 0.1
    # ominc_rixs = np.linspace(10.9 + off, 14.9 + off, 100)

    # L3-edge
    gamma_c, gamma_f = res['gamma_c'][1], 0.1
    ominc_rixs = np.linspace(-5.9 + off, -0.9 + off, 10)

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
    eval_i, denmat = edrixs.ed_1v1c_fort(
        comm, shell_name, shell_level=(0, -off), v_soc=(zeta_d_i, zeta_d_n), c_soc=zeta_p_n,
        v_noccu=8, slater=slater, v_cfmat=cf, ed_solver=0, neval=10, nvector=3, idump=True
    )

    # Run XAS
    xas, xas_poles = edrixs.xas_1v1c_fort(
        comm, shell_name, ominc_xas, gamma_c=gamma_c, v_noccu=8, thin=thin, phi=phi,
        num_gs=3, nkryl=100, pol_type=poltype_xas, temperature=300
    )
    if rank == 0:
        np.savetxt('xas.dat', np.concatenate((np.array([ominc_xas]).T, xas), axis=1))
        # Save XAS pole files for later plots
        edrixs.dump_poles(xas_poles, 'xas_poles')

    # Run RIXS
    rixs, rixs_poles = edrixs.rixs_1v1c_fort(
        comm, shell_name, ominc_rixs, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
        thin=thin, thout=thout, phi=phi, v_noccu=8, pol_type=poltype_rixs,
        num_gs=3, nkryl=100, temperature=300
    )
    if rank == 0:
        # Save RIXS pole files for later plots
        edrixs.dump_poles(rixs_poles, 'rixs_poles')
        # Save RIXS spectra
        rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
        rixs_sigma = np.sum(rixs[:, :, 2:4], axis=2)
        np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
        np.savetxt('rixs_sigma.dat', np.concatenate((np.array([eloss]).T, rixs_sigma.T), axis=1))
        # Plot RIXS map
        edrixs.plot_rixs_map(rixs_pi, ominc_rixs, eloss, "rixsmap_pi.pdf")
        edrixs.plot_rixs_map(rixs_sigma, ominc_rixs, eloss, "rixsmap_sigma.pdf")
