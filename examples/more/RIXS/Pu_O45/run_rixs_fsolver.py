#!/usr/bin/env python

import json
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
    slat_i[1], slat_i[2], slat_i[3] = slat_i[1] * 0.77, slat_i[2] * 0.77, slat_i[3] * 0.77
    Uf_ave = 0
    slat_i[0] = Uf_ave + edrixs.get_F0('f', slat_i[1], slat_i[2], slat_i[3])

    # Intermediate Hamiltonian
    slat_n[1], slat_n[2], slat_n[3] = slat_n[1] * 0.77, slat_n[2] * 0.77, slat_n[3] * 0.77
    slat_n[0] = Uf_ave + edrixs.get_F0('f', slat_n[1], slat_n[2], slat_n[3])

    slat_n[5], slat_n[6] = slat_n[5] * 0.6, slat_n[6] * 0.6
    slat_n[7], slat_n[8], slat_n[9] = slat_n[7] * 0.6, slat_n[8] * 0.6, slat_n[9] * 0.6

    Ufd_ave = 0.0
    slat_n[4] = Ufd_ave + edrixs.get_F0('fd', slat_n[7], slat_n[8], slat_n[9])

    slater = (slat_i, slat_n)

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
        with open('xas_poles.json', 'w') as f:
            json.dump(xas_poles, f, indent=4)

    # Run RIXS
    rixs, rixs_poles = edrixs.rixs_1v1c_fort(
        comm, shell_name, ominc_rixs, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
        thin=thin, thout=thout, phi=phi, v_noccu=noccu, pol_type=poltype_rixs,
        num_gs=num_gs, nkryl=200, temperature=T
    )
    if rank == 0:
        with open('rixs_poles.json', 'w') as f:
            json.dump(rixs_poles, f, indent=4)
        rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)
        np.savetxt('rixs_pi.dat', np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
        edrixs.plot_rixs_map(rixs_pi, ominc_rixs, eloss, "rixsmap_pi.pdf")
