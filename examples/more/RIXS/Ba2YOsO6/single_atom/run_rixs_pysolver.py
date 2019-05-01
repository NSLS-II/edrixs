#!/usr/bin/env python


import numpy as np
import edrixs


if __name__ == "__main__":
    """
    Os :math:`L_{2,3}`-edge, :math:`2p_{1/2,3/2}\\rightarrow 5d` transition, cubic crystal field.

    How to run
    ----------
    python run_rixs_pysolver.py
    """

    # PARAMETERS
    # ----------
    # Slater integrals
    scale_d = 0.335
    F2_dd, F4_dd = 8.885 * scale_d, 5.971 * scale_d
    # Set averaged Coulomb interaction for 5d to zero
    Ud = 0.0
    # F0_d contatins two parts: (1) related to Ud (2) related to F2_dd, F4_dd
    F0_dd = Ud + edrixs.get_F0('d', F2_dd, F4_dd)

    scale_dp = 0.9
    F2_dp = 1.036 * scale_dp
    G1_dp, G3_dp = 0.894 * scale_dp, 0.531 * scale_dp
    Udp = 0.0
    F0_dp = Udp + edrixs.get_F0('dp', G1_dp, G3_dp)

    # The tuple containing Slater integrals for (1) initial Hamiltonian (2) intermediate
    # Hamiltonian, the order of these parameters is important.
    # First FX for valence, then FX for valence-core, then GX for valence-core, and
    # then FX for core, where X=0, 2, 4, ... or X=1, 3, 5, ...
    slater = (
        [F0_dd, F2_dd, F4_dd],  # Initial
        [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp]   # Intermediate
    )

    # Spin-Orbit Coupling (SOC) zeta
    # 5d, without core-hole
    zeta_d_i = 0.33
    # 5d, with core-hole
    zeta_d_n = 0.33

    # Cubic crystal field
    dq10 = 4.3
    cf = edrixs.cf_cubic_d(dq10)

    # Occupancy of 5d shell
    noccu = 3

    # RIXS settings
    thin, thout, phi = 45 / 180.0 * np.pi, 45 / 180.0 * np.pi, 0.0
    gamma_c = 2.5
    gamma_f = 0.075
    ominc = np.linspace(-4, 4, 100)
    gs_list = list(range(0, 4))
    poltype_rixs = [('linear', 0.0, 'linear', 0.0),
                    ('linear', 0.0, 'linear', np.pi / 2.0),
                    ('linear', np.pi / 2.0, 'linear', 0.0),
                    ('linear', np.pi / 2.0, 'linear', np.pi / 2.0)]

    transition = [('L2', 'd', 12385 + 5), ('L3', 'd', 10871 + 5),  # Keep all the 5d orbitals
                  ('L2', 't2g', 12385 + 5), ('L3', 't2g', 10871 + 5)  # Keep only t2g orbitals
                  ]

    for case, v_name, om_shift in transition:
        # Run ED
        c_name = edrixs.edge_to_shell_name(case)
        print("edrixs >>> ", c_name, " to ", v_name)
        if v_name == 'd':
            eloss = np.linspace(-0.5, 5.5, 1000)
            result = edrixs.ed_1v1c_py(
                shell_name=(v_name, c_name), v_soc=(zeta_d_i, zeta_d_n),
                v_noccu=noccu, slater=slater, v_cfmat=cf  # add CF
            )
        else:
            eloss = np.linspace(-0.5, 2, 1000)
            result = edrixs.ed_1v1c_py(
                shell_name=(v_name, c_name), v_soc=(zeta_d_i, zeta_d_n),
                v_noccu=noccu, slater=slater  # NO CF
            )
        eval_i, eval_n, trans_op = result

        # Run RIXS
        rixs = edrixs.rixs_1v1c_py(
            eval_i, eval_n, trans_op, ominc, eloss, gamma_c=gamma_c, gamma_f=gamma_f, thin=thin,
            thout=thout, phi=phi, pol_type=poltype_rixs, gs_list=gs_list, temperature=300
        )
        # In rixs: axis-0 is for ominc, axis-1 is for eloss, axis-2 is for polarization
        rixs_pi = np.sum(rixs[:, :, 0:2], axis=2)   # pi-pi + pi-sigma
        rixs_sigma = np.sum(rixs[:, :, 2:4], axis=2)   # sigma-pi + sigma-sigma
        # The first column is eloss, other columns are RIXS cut, useful when plotting with gnuplot
        fname = 'rixs_pi_' + case + v_name + '.dat'
        np.savetxt(fname, np.concatenate((np.array([eloss]).T, rixs_pi.T), axis=1))
        fname = 'rixs_sigma_' + case + v_name + '.dat'
        np.savetxt(fname, np.concatenate((np.array([eloss]).T, rixs_sigma.T), axis=1))
        # Plot RIXS map
        fname = 'rixsmap_pi_' + case + v_name + '.pdf'
        edrixs.plot_rixs_map(rixs_pi, ominc + om_shift, eloss, fname)
        fname = 'rixsmap_sigma_' + case + v_name + '.pdf'
        edrixs.plot_rixs_map(rixs_sigma, ominc + om_shift, eloss, fname)
