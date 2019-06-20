#!/usr/bin/env python

# This example does a simlar calculation to that in
# Phys. Rev. Lett. 118, 156402 (2017), but with an added
# spin degree of freedom.

import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import edrixs


def do_ed(dq10=1.6, d1=0.1, d3=0.75,
          ex_x=0., ex_y=0., ex_z=0.):
    """
    Diagonalize Hamiltonian for Ni 3d8 system in tetragonal crytals field and
    applied exchange field.

    Parameters
    ----------
    dq10: float
        Cubic crystal field parameter (eV)
        Cubic splitting
    dq1: float
        Cubic crystal field parameter (eV)
        t2g splitting
    dq3: float
        Cubic crystal field parameter (eV)
        eg splitting
    ex_x: float
        Magnitude of the magnetic exchange field in the x direction (eV)
    ex_y: float
        Magnitude of the magnetic exchange field in the y direction (eV)
    ex_z: float
        Magnitude of the magnetic exchange field in the z direction (eV)

    Returns
    --------
    eval_i: 1d array
        Eigenvalues of initial state Hamiltonian
    eval_n: 1d array
        Eigenvalues of intermediate state Hamiltonian
    dipole_op: 3d array
        Dipole operator from 2p to 3d state
    """
    # PARAMETERS:
    # ------------
    F2_dd, F4_dd = 12.234 * 0.65, 7.598 * 0.65
    Ud_av = 0.0
    F0_dd = Ud_av + edrixs.get_F0('d', F2_dd, F4_dd)

    G1_dp, G3_dp = 5.787 * 0.7, 3.291 * 0.7
    F2_dp = 7.721 * 0.95
    Udp_av = 0.0
    F0_dp = Udp_av + edrixs.get_F0('dp', G1_dp, G3_dp)

    slater = (
        [F0_dd, F2_dd, F4_dd],  # Initial
        [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp]   # Intermediate
    )

    zeta_d_i = 0.083
    zeta_d_n = 0.102
    zeta_p_n = 11.24

    cf = edrixs.cf_tetragonal_d(dq10, d1, d3)

    ext_B = np.array([ex_x, ex_y, ex_z])
    result = edrixs.ed_1v1c_py(
        shell_name=('d', 'p'), v_soc=(zeta_d_i, zeta_d_n), c_soc=zeta_p_n,
        v_noccu=8, slater=slater, ext_B=ext_B, on_which='spin', v_cfmat=cf
    )
    return result


def get_xas(eval_i, eval_n, dipole_ops,
            om_mesh=np.linspace(-10, 20, 1000), om_offset=857.40,
            thin=115 / 180.0 * np.pi, phi=0.0,
            gamma_c=0.2, T=30.):
    '''
    Calculate XAS spectrum.

    Parameters
    ----------
    eval_i: array
        Eigenvalues of initial state Hamiltonian
    eval_n: array
        Eigenvalues of intermediate state Hamiltonian
    dipole_op: array
        Dipole operator from 2p to 3d state
    om_mesh: array
        Incident energies to calculate the spectrum.
        In eV without the offset.
    om_offset: float
        Offset between theoretical and experimental energy axes
    thin: float
        Incident x-ray angle in radians w.r.t. plane perpendicular to the
        c-axis
    phi: float
        Azimutal x-ray angle in radians
    gamma_c: float
        core-hole life-time broadening
    T: float
        Temperature in Kelvin

    Returns
    -------
    om_mesh: array
        Incident energies to calculate the spectrum.
        In eV without the offset.

    om_offset: float
        Offset between theoretical and experimental energy axes

    xas: array
        X-ray absorption intensity. Shape (len(om_mesh), 5)
        Where the last axis denotes the polarization
        Linear pi
        Linear sigma
        Circular left
        Circular right
        Isotropic
    '''
    poltype = [('linear', 0.0), ('linear', np.pi / 2.0),
               ('left', 0.0), ('right', 0.0), ('isotropic', 0.0)]
    xas = edrixs.xas_1v1c_py(
        eval_i, eval_n, dipole_ops, om_mesh, gamma_c=gamma_c, thin=thin,
        phi=phi, pol_type=poltype, gs_list=[0, 1, 2], temperature=T
    )

    return om_mesh, om_offset, xas


def get_rixs(eval_i, eval_n, dipole_ops,
             om_mesh=np.linspace(-8, 0, 100), om_offset=857.40,
             eloss_mesh=np.linspace(-1, 5.0, 1000),
             thin=115 / 180.0 * np.pi, thout=15 / 180.0 * np.pi, phi=0.0,
             gamma_c=0.5, gamma_f=0.10, emi_res=0, T=30.):
    """
    Calculate RIXS map.

    Parameters
    ----------
    eval_i: array
        Eigenvalues of initial state Hamiltonian
    eval_n: array
        Eigenvalues of intermediate state Hamiltonian
    dipole_op: array
        Dipole operator from 2p to 3d state
    om_mesh: array
        Incident energies to calculate the spectrum.
        In eV without the offset.
    om_offset: float
        Offset between theoretical and experimental energy axes
    eloss_mesh: array
        axis to evalulate RIXS on.
    thin: float
        Incident x-ray angle in radians w.r.t. plane perpendicular to the
        c-axis
    th_out: float
        Emitted x-ray angle in radians w.r.t. plane perpendicular to the
        c-axis
    phi: float
        Azimutal x-ray angle in radians
    gamma_c: float
        core-hole life-time Lorentzian broadening
    gamma_f: float
        final state life-time Lorentzian broadening
    emi_res: float
        resolution for the emitted x-rays
        Gaussian convolution defined as sigma (not FWHM)
    T: float
        Temperature in Kelvin

    Returns
    -------
    om_mesh: array
        Incident energies to calculate the spectrum.
        In eV without the offset.
    om_offset: float
        Offset between theoretical and experimental energy axes
    eloss_mesh: array
        axis to evalulate RIXS on.
    rixs: array
        X-ray absorption intensity. Shape (len(om_mesh), len(eloss_mesh), 4)
        Where the last axis denotes the polarization
        pi-pi
        pi-sigma
        sigma-pi
        sigma-sigma
    """
    poltype_rixs = [('linear', 0, 'linear', 0),
                    ('linear', 0, 'linear', np.pi/2.0),
                    ('linear', np.pi/2.0, 'linear', 0.0),
                    ('linear', np.pi/2.0, 'linear', np.pi/2.0)]
    rixs = edrixs.rixs_1v1c_py(
        eval_i, eval_n, dipole_ops, om_mesh, eloss_mesh, gamma_c=gamma_c, gamma_f=gamma_f,
        thin=thin, thout=thout, phi=phi, pol_type=poltype_rixs, gs_list=[0, 1, 2], temperature=T
    )
    # Apply gaussian broadening
    if emi_res > 0:
        dx = np.mean(np.abs(eloss_mesh[1:] - eloss_mesh[:-1]))
        rixs = np.apply_along_axis(gaussian_filter1d, 2, rixs, emi_res / dx)

    return om_mesh, om_offset, eloss_mesh, rixs


fig, (ax_XAS, ax_RIXS, ax_spin) = plt.subplots(1, 3, figsize=(10, 3))
# Compute XAS
eval_i, eval_n, dipole_op = do_ed(dq10=1.6, d1=0.1, d3=0.75)
om_mesh, om_offset, xas = get_xas(eval_i, eval_n, dipole_op, gamma_c=.5)
ax_XAS.plot(om_mesh + om_offset, xas[:, -1])
ax_XAS.set_xlabel('Incident energy (eV)')
ax_XAS.set_ylabel('XAS')

# Resonant energy at the peak of the XAS
res_e = om_mesh[np.argmax(xas[:, -1])]

# Compute a RIXS MAP
om_mesh, om_offset, eloss_mesh, rixs = get_rixs(eval_i, eval_n, dipole_op, gamma_c=0.5, emi_res=.1)
art = ax_RIXS.pcolorfast(eloss_mesh, om_mesh + om_offset, rixs[:, :, 0:2].sum(2))
ax_RIXS.set_ylabel('Incident energy (eV)')
ax_RIXS.set_xlabel('Energy loss')
plt.colorbar(art, ax=ax_RIXS)
ax_RIXS.xaxis.set_major_locator(MultipleLocator(2))

# Compute RIXS as a function of the spin direction
phis = np.linspace(0, np.pi / 2, 5)
colors = plt.cm.viridis(np.linspace(0, 1, len(phis)))
for phi, color in zip(phis, colors):
    ex_x = 0.1 * np.sin(phi)
    ex_y = 0
    ex_z = 0.1 * np.cos(phi)

    eval_i, eval_n, dipole_op = do_ed(dq10=1.6, d1=0.1, d3=0.75,
                                      ex_x=ex_x, ex_y=ex_y, ex_z=ex_z)

    om_mesh, om_offset, eloss_mesh, rixs = get_rixs(eval_i, eval_n, dipole_op,
                                                    om_mesh=np.array([res_e]),
                                                    gamma_c=0.5, emi_res=.07, T=20)
    intensity = rixs[:, :, 0:2].sum((0, 2))
    ax_spin.plot(eloss_mesh, intensity, color=color,
                 label=r'$\phi = {:.3f}$'.format(phi))

ax_spin.set_xlabel('Energy loss (eV)')
ax_spin.set_ylabel('RIXS intensity')

ax_spin.legend(fontsize=8)

for ax in [ax_XAS, ax_RIXS, ax_spin]:
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

fig.subplots_adjust(bottom=0.2, wspace=0.4)
plt.savefig('figures.pdf')

# Use interativate mode in ipython
# otherwise just display plot
try:
    __IPYTHON__
except NameError:
    plt.show(block=True)
else:
    plt.ion()
    plt.show()
