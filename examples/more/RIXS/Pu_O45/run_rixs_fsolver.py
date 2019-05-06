#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import edrixs

if __name__ == "__main__":
    '''
    Pu-5f6 :math:`O_{4,5}`-edge, :math:`5d_{3/2,5/2}\\rightarrow 5f` transition.

    How to run
    ----------
    mpiexec -n 4 python run_rixs_fsolver.py
    '''

    # PARAMETERS
    # -----------
    # Occupancy of U 5f orbitals
    noccu = 6
    res = edrixs.get_atom_data('Pu', v_name='5f', v_noccu=noccu, edge='O45')
    name_i, slat_i = [list(i) for i in zip(*res['slater_i'])]
    name_n, slat_n = [list(i) for i in zip(*res['slater_n'])]

    # Initial Hamiltonian
    si = edrixs.rescale(slat_i, ([1, 2, 3], [0.77]*3))
    si[0] = edrixs.get_F0('f', si[1], si[2], si[3])

    # Intermediate Hamiltonian
    sn = edrixs.rescale(slat_n, ([1, 2, 3, 5, 6, 7, 8, 9], [0.77]*3+[0.6]*4))
    sn[0] = edrixs.get_F0('f', sn[1], sn[2], sn[3])
    sn[4] = edrixs.get_F0('fd', sn[7], sn[8], sn[9])

    slater = (si, sn)

    # Spin-Orbit Coupling (SOC) zeta
    # 5f, without core-hole, from Cowan's code
    zeta_f_i = res['v_soc_i'][0] * 0.9
    # 5f, with core-hole, from Cowan's code
    zeta_f_n = res['v_soc_n'][0] * 0.9
    zeta_d_n = (res['edge_ene'][0] - res['edge_ene'][1]) / 2.5

    # Energy shift of the core level
    om_shift = (2 * res['edge_ene'][0] + 3 * res['edge_ene'][1]) / 5.0

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
        edrixs.dump_poles(xas_poles, 'xas_poles')

    # Run RIXS
    rixs, rixs_poles = edrixs.rixs_1v1c_fort(
        comm, shell_name, ominc_rixs, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
        thin=thin, thout=thout, phi=phi, v_noccu=noccu, pol_type=poltype_rixs,
        num_gs=num_gs, nkryl=200, temperature=T
    )
    if rank == 0:
        edrixs.dump_poles(rixs_poles, 'rixs_poles')
        rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
        np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
        edrixs.plot_rixs_map(rixs_pi, ominc_rixs, eloss, "rixsmap_pi.pdf")
