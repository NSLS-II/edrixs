#!/usr/bin/env python

# This example does a simlar calculation to that in
# Phys. Rev. Lett. 118, 156402 (2017), but with an added
# spin degree of freedom.

import sys
import numpy as np
from scipy.ndimage      import gaussian_filter1d
from matplotlib         import pyplot as plt
from matplotlib.ticker  import AutoMinorLocator, MultipleLocator 
from soc                import  atom_hsoc
from iostream           import  write_tensor, write_tensor_nonzeros
from fock_basis         import  get_fock_bin_by_N
from coulomb_utensor    import  get_umat_slater, get_F0
from basis_transform    import  cb_op, cb_op2, tmat_r2c
from photon_transition  import  get_trans_oper, dipole_polvec_xas, dipole_polvec_rixs
from angular_momentum   import  get_sx, get_sy, get_sz

from manybody_operator  import  two_fermion, four_fermion
from utils              import boltz_dist
from rixs_utils         import   scattering_mat

def do_ed(dq10=1.6, d1=0.1, d3=0.75,
          ex_x=0., ex_y=0., ex_z=0.):
    """Diagonalize Hamiltonian for Ni 3d8 system in tetragonal crytals field and
    applied exchange field.

    Parameters
    ----------
    dq10 : float
        Cubic crystal field parameter (eV)
        Cubic splitting
    dq1 : float
        Cubic crystal field parameter (eV)
        t2g splitting
    dq3 : float
        Cubic crystal field parameter (eV)
        eg splitting
    ex_x : float
        Magnitude of the magnetic exchange field in the x direction (eV)
    ex_y : float
        Magnitude of the magnetic exchange field in the y direction (eV)
    ex_z : float
        Magnitude of the magnetic exchange field in the z direction (eV)

    Returns
    --------
    eval_i : array
        Eigenvalues of initial state Hamiltonian
    eval_n : array
        Eigenvalues of intermediate state Hamiltonian
    dipole_op
        Dipole operator from 2p to 3d state
    """
    # PARAMETERS:
    #------------

    # Number of orbitals
    # 10 3d-orbitals, 6 2p-orbitals (including spin)
    ndorb, nporb = 10, 6
    # total orbitals for this "d + p" model
    norbs = ndorb + nporb

    # Slater Integrals for Coulomb interaction terms
    # They can be calculated by, for example, the Cowan's code [https://www.tcd.ie/Physics/people/Cormac.McGuinness/Cowan/]
    # by using Hartree-Fock mean-field method, and then are re-scaled, for example, Fdd: 65%, Fdp: 95%, Gdp: 70%
    # F^{k}_ij, k=0,2,4, ..., min(2*l_{i}, 2*l_{j})
    # G^{k}_ij, k=|l_{i}-l_{j}|, |l_{i}-l_{j}|+2, ..., l_{i}+l_{j}, where i,j are shell indices
    F2_d, F4_d = 12.234*0.65,  7.598*0.65
    # Here, Ud_av is the averaged Coulomb interaction in 3d shell, we set it to be zero for simplicity.
    # It doesn't matter here because we are doing single atomic calculation with fixed 3d occupancy.
    Ud_av = 0.0
    # F0_dd is split into two parts: (1) Ud_av, (2) another part is related to F2_d and F4_d
    F0_d = Ud_av + get_F0('d', F2_d, F4_d)

    G1_dp, G3_dp = 5.787*0.7, 3.291*0.7
    F2_dp = 7.721*0.95
    # Udp is the averaged Coulomb interaction between 3d and 2p shells, we set it to be zero here.
    Udp_av = 0.0
    # F0_dp is split intwo two parts: (1) Udp_av, (2) another part is related to G1_dp and G3_dp
    F0_dp = Udp_av + get_F0('dp', G1_dp, G3_dp)

    # Spin-Orbit Coupling (SOC) zeta
    # for 3d, without core-hole, from Cowan's code
    zeta_d_i = 0.083
    # for 3d, with core-hole, from Cowan's code
    zeta_d_n = 0.102
    # for 2p core electron, from Cowan's code, and usually we will adjust it to match the energy difference of the two edges,
    # for example, roughly, E(L2-edge) - E(L3-edge) = 1.5 * zeta_p_n
    zeta_p_n = 11.24

    # Crystal Field notation is converted to dt, ds, dq notation
    dt = (3.0*d3 - 4.0*d1)/35
    ds = (d3 + d1)/7.0
    dq = dq10/10.0

    # NOTE: the RIXS scattering involves 2 process, the absorption and emission, so we have 3 states:
    # (1) before the absorption, only valence shells are involved, we use the letter (i) to label it
    # (2) after the absorption, a core hole is created in the core shell, both the valence and core shells are involved, we use
    #     the letter (n) to label it
    # (3) after the emission, the core hole is re-filled by an electron, we use the letter (f) to label it.

    # Coulomb interaction tensors without core-hole
    params=[
            F0_d, F2_d, F4_d,     # Fk for d
            0.0, 0.0,             # Fk for dp
            0.0, 0.0,             # Gk for dp
            0.0, 0.0              # Fk for p
           ]
    umat_i = get_umat_slater('dp', *params)

    # Coulomb interaction tensors with a core-hole
    params=[
            F0_d, F2_d, F4_d,     # Fk for d
            F0_dp, F2_dp,         # Fk for dp
            G1_dp, G3_dp,         # Gk for dp
            0.0,   0.0            # Fk for p
           ]
    umat_n = get_umat_slater('dp', *params)

    # SOC matrix
    hsoc_i = np.zeros((norbs,norbs), dtype=np.complex128)
    hsoc_n = np.zeros((norbs,norbs), dtype=np.complex128)
    hsoc_i[0:ndorb, 0:ndorb] += atom_hsoc('d', zeta_d_i)
    hsoc_n[0:ndorb, 0:ndorb] += atom_hsoc('d', zeta_d_n)
    hsoc_n[ndorb:norbs, ndorb:norbs] += atom_hsoc('p', zeta_p_n)

    # Crystal Field matrix
    crys = np.zeros((norbs,norbs),dtype=np.complex128)
    # For 3d real harmonics, the default orbital ordering in our code is:
    # d3z2-r2, dzx, dzy, dx2-y2, dxy
    tmp = np.zeros((5,5),dtype=np.complex128)
    tmp[0,0] =  6*dq - 2*ds - 6*dt   # d3z2-r2
    tmp[1,1] = -4*dq - 1*ds + 4*dt   # dzx
    tmp[2,2] = -4*dq - 1*ds + 4*dt   # dzy
    tmp[3,3] =  6*dq + 2*ds - 1*dt   # dx2-y2
    tmp[4,4] = -4*dq + 2*ds - 1*dt   # dxy
    # change basis to complex spherical harmonics, and then add spin degree of freedom
    tmp[:,:] = cb_op(tmp, tmat_r2c('d'))
    crys[0:ndorb:2, 0:ndorb:2] = tmp
    crys[1:ndorb:2, 1:ndorb:2] = tmp

    # Exchange field in complex harmonic basis. Orbital angular momentum l=2 in
    # the d-orbitals
    exchange = ex_x * get_sx(2) + ex_y * get_sy(2) + ex_z * get_sz(2)

    # build Fock basis (binary representation), without core-hole
    # 8 3d-electrons and 6 2p-electrons
    basis_i = get_fock_bin_by_N(ndorb, 8, nporb, nporb)
    ncfgs_i = len(basis_i)

    # build Fock basis (binary representation), with a core-hole
    # 9 3d-electrons and 5 2p-electrons
    basis_n = get_fock_bin_by_N(ndorb, 9, nporb, nporb-1)
    ncfgs_n = len(basis_n)

    # build many-body Hamiltonian, without core-hole
    hmat_i = np.zeros((ncfgs_i, ncfgs_i),dtype=np.complex128)
    hmat_n = np.zeros((ncfgs_n, ncfgs_n),dtype=np.complex128)
    emat = hsoc_i + crys
    # add the exchange field
    emat[0:ndorb,0:ndorb] += exchange

    hmat_i[:,:] += two_fermion(emat,  basis_i, basis_i)
    hmat_i[:,:] += four_fermion(umat_i, basis_i)

    # build many-body Hamiltonian, with a core-hole
    emat = hsoc_n + crys
    hmat_n[:,:] += two_fermion(emat,   basis_n, basis_n)
    hmat_n[:,:] += four_fermion(umat_n, basis_n)

    # diagonalize the many-body Hamiltonian to get eigenvalues and eigenvectors
    eval_i, evec_i = np.linalg.eigh(hmat_i)
    eval_n, evec_n = np.linalg.eigh(hmat_n)

    # build dipolar transition operator from 2p to 3d
    # build the transition operator in the Fock basis
    dipole = np.zeros((3, norbs, norbs), dtype=np.complex128)
    dipole_op = np.zeros((3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    # Please note that, here the function returns operators x, y, z
    tmp = get_trans_oper('dp')
    for i in range(3):
        dipole[i, 0:ndorb, ndorb:norbs] = tmp[i]
        dipole_op[i] = two_fermion(dipole[i], basis_n, basis_i)
    # and then transform it to the eigenstate basis
    for i in range(3):
        dipole_op[i] = cb_op2(dipole_op[i], evec_n, evec_i)

    return eval_i, eval_n, dipole_op


def get_xas(eval_i, eval_n, dipole_ops,
        om_mesh=np.linspace(-10, 20, 1000), om_offset=857.40,
        thin=115/180.0*np.pi, phi = 0.0,
        gamma=0.2, T=30.):
    '''Calculate XAS spectrum.

    Parameters
    ----------
    eval_i : array
        Eigenvalues of initial state Hamiltonian
    eval_n : array
        Eigenvalues of intermediate state Hamiltonian
    dipole_op : array
        Dipole operator from 2p to 3d state
    om_mesh : array
        Incident energies to calculate the spectrum.
        In eV without the offset.
    om_offset : float
        Offset between theoretical and experimental energy axes
    thin : float
        Incident x-ray angle in radians w.r.t. plane perpendicular to the
        c-axis
    phi : float
        Azimutal x-ray angle in radians
    gamma : float
        core-hole life-time broadening
    T : float
        Temperature in Kelvin

    Returns
    -------
    om_mesh : array
        Incident energies to calculate the spectrum.
        In eV without the offset.
    om_offset : float
        Offset between theoretical and experimental energy axes
    xas : array
        X-ray absorption intensity. Shape (5, len(om_mesh))
        Where the first axis denotes the polarization
        Linear pi
        Linear sigma
        Circular left
        Circular right
        Isotropic
    '''
    # PARAMETERS
    #-----------
    # In our calculations, we set the energy level of 2p core to zero, so we need to
    # adjust om_offset to match the experimental X-ray energy, it can be roughly estimated by, for example, Ni-L2/L3 edge,
    # om_offset = (2*E_L3 + E_L2)/3, E_L3 and E_L2 can be found in software, such as, Hephaestus [https://bruceravel.github.io/demeter/aug/hephaestus.html].
    # for Ni, E_L3=852.7 eV and E_L2=870 eV, so the calculated om_offset is 858.47 eV, however, due to the different
    # crystal environment, we fine tune it to be 857.40 according to the experimental XAS and RIXS data.

    # which initial states are used to calculate the spectra
    gs_list=list(range(0, 3))

    ncfgs_n, ncfgs_i = len(eval_n), len(eval_i)

    # get boltzmann distribution at T for these ground states
    gs_prob = boltz_dist([eval_i[i] for i in gs_list], T)

    # 5 different polarizations are considered
    # (0) linear \pi-pol, the polarization vector is in the scattering plane
    pol_pi    = dipole_polvec_xas(thin, phi, 0)
    # (1) linear \sigma-pol, the polarization vector is normal to the scattering plane
    pol_sigma = dipole_polvec_xas(thin, phi, np.pi/2.0)
    polvec = [pol_pi,                                   # Linear pi
              pol_sigma,                                # Linear sigma
              (pol_pi + 1j * pol_sigma) / np.sqrt(2.0), # circular left
              (pol_pi - 1j * pol_sigma) / np.sqrt(2.0),  # Circular right
              np.array([1,1,1])/np.sqrt(3.0)            # Isotropic
    ]

    # Compute XAS
    xas = np.zeros((5, len(om_mesh)), dtype=np.float64)
    for m, ei in enumerate(polvec):
        for i, om_inc in enumerate(om_mesh):
            for j in gs_list:
                scatt_mat = dipole_op[0,:,j] * ei[0] + dipole_op[1,:,j] * ei[1] + dipole_op[2,:,j] * ei[2]
                xas[m, i] += np.sum( np.abs( scatt_mat )**2  * gamma/np.pi / ((om_inc - (eval_n[:] - eval_i[j]))**2 + gamma**2) ) * gs_prob[j]

    return om_mesh, om_offset, xas

def get_rixs(eval_i, eval_n, dipole_ops,
            om_mesh=np.linspace(-8, 0, 100), om_offset=857.40,
            eloss_mesh = np.linspace(-1, 5.0, 1000),
            thin=115/180.0*np.pi,  thout=15/180.0*np.pi, phi = 0.0,
            gamma_n=0.5, gamma_f = 0.10, emi_res=0.1, T=30.):
    """Calculate RIXS map.

    Parameters
    ----------
    eval_i : array
        Eigenvalues of initial state Hamiltonian
    eval_n : array
        Eigenvalues of intermediate state Hamiltonian
    dipole_op : array
        Dipole operator from 2p to 3d state
    om_mesh : array
        Incident energies to calculate the spectrum.
        In eV without the offset.
    om_offset : float
        Offset between theoretical and experimental energy axes
    eloss_mesh : array
        axis to evalulate RIXS on.
    thin : float
        Incident x-ray angle in radians w.r.t. plane perpendicular to the
        c-axis
    th_out : float
        Emitted x-ray angle in radians w.r.t. plane perpendicular to the
        c-axis
    phi : float
        Azimutal x-ray angle in radians
    gamma : float
        core-hole life-time broadening
    gamma_f : float
        final state life-time broadening
    emi_res : float
        resolution for the emitted x-rays
        Gaussian convolution defined as sigma (not FWHM)
    T : float
        Temperature in Kelvin
    """
    # Allow these states to contribute to the ground state based on a boltzmann
    # distribution
    gs_list=list(range(0, 3))

    ncfgs_i, ncfgs_n = len(eval_i), len(eval_n)
    # the transition operator for the absorption process
    trans_mat_abs = dipole_op
    # the transition operator for the emission process
    trans_mat_emi = np.zeros((3, ncfgs_i, ncfgs_n), dtype=np.complex128)
    for i in range(3):
        trans_mat_emi[i] = np.conj(np.transpose(trans_mat_abs[i]))

    # We calculate RIXS for \pi-\pi, \pi-\sigma, \sigma-\pi, \sigma-\sigma polarizations
    rixs = np.zeros((4, len(eloss_mesh), len(om_mesh)), dtype=np.float64)
    gs_prob = boltz_dist([eval_i[i] for i in gs_list], T)
    polvec_list = [(0,0), (0,np.pi/2.0), (np.pi/2.0, 0), (np.pi/2.0, np.pi/2.0)]

    for i, om_inc in enumerate(om_mesh):
        F_fi = scattering_mat(eval_i, eval_n, trans_mat_abs[:,:,gs_list], trans_mat_emi, om_inc, gamma_n)
        for j, (alpha, beta) in enumerate(polvec_list):
            ei, ef = dipole_polvec_rixs(thin, thout, phi, alpha, beta)
            F_magnitude = np.zeros((ncfgs_i, len(gs_list)), dtype=np.complex128)
            for m in range(3):
                for n in range(3):
                    F_magnitude[:,:] += ef[m] * F_fi[m,n] * ei[n]
            for m in gs_list:
                for n in range(ncfgs_i):
                    rixs[j, :, i] += np.abs(F_magnitude[n,m])**2 * gamma_f/np.pi / ( (eloss_mesh-(eval_i[n]-eval_i[m]))**2 + gamma_f**2 ) * gs_prob[m]

    # Apply gaussian broadening
    dx = np.mean(np.abs(eloss_mesh[1:] - eloss_mesh[:-1]))
    rixs = np.apply_along_axis(gaussian_filter1d, 1, rixs, emi_res/dx)

    return om_mesh, om_offset, eloss_mesh, rixs


fig, (ax_XAS, ax_RIXS, ax_spin) = plt.subplots(1, 3, figsize=(10, 3))

# Compute XAS
eval_i, eval_n, dipole_op = do_ed(dq10=1.6, d1=0.1, d3=0.75)
om_mesh, om_offset, xas = get_xas(eval_i, eval_n, dipole_op, gamma=.5)
ax_XAS.plot(om_mesh + om_offset, xas[-1])
ax_XAS.set_xlabel('Incident energy (eV)')
ax_XAS.set_ylabel('XAS')

# Resonant energy at the peak of the XAS
res_e = om_mesh[np.argmax(xas[-1])]

# Compute a RIXS MAP
om_mesh, om_offset, eloss_mesh, rixs = get_rixs(eval_i, eval_n, dipole_op, gamma_n=0.5, emi_res=.1)
art = ax_RIXS.pcolorfast(om_mesh + om_offset, eloss_mesh, rixs[:2].sum(0))
ax_RIXS.set_xlabel('Incident energy (eV)')
ax_RIXS.set_ylabel('Energy loss')
plt.colorbar(art, ax=ax_RIXS)
ax_RIXS.xaxis.set_major_locator(MultipleLocator(2))

# Compute RIXS as a function of the spin direction
phis = np.linspace(0, np.pi/2, 5)
colors = plt.cm.viridis(np.linspace(0, 1, len(phis)))
for phi, color in zip(phis, colors):
    ex_x = 0.1 * np.sin(phi)
    ex_y = 0
    ex_z = 0.1 * np.cos(phi)

    eval_i, eval_n, dipole_op = do_ed(dq10=1.6, d1=0.1, d3=0.75,
                                      ex_x=ex_x, ex_y=ex_y, ex_z=ex_z)

    om_mesh, om_offset, eloss_mesh, rixs = get_rixs(eval_i, eval_n, dipole_op,
                                                    om_mesh=np.array([res_e]),
                                                    gamma_n=1, emi_res=.07, T=20)
    I = rixs[:2].sum((0, 2))
    ax_spin.plot(eloss_mesh, I, color=color,
                 label='$\phi = {:.3f}$'.format(phi))

ax_spin.set_xlabel('Energy loss (eV)')
ax_spin.set_ylabel('RIXS intensity')

ax_spin.legend(fontsize=8)

for ax in [ax_XAS, ax_RIXS, ax_spin]:
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

fig.subplots_adjust(bottom=0.2, wspace=0.4)

# Use interativate mode in ipython
# otherwise just display plot
try:
    __IPYTHON__
except NameError:
    plt.show(block=True)
else:
    plt.ion()
    plt.show()
