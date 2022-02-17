#!/usr/bin/env python
"""
X-ray transitions
================================================================================
This example explains how to calculate x-ray transition amplitudes between
specific orbital and spin states. We take the case of a cuprate with a
:math:`d_{x^2-y^2}` ground state and compute the angular dependence of spin-flip
and non-spin-flip processes.

This case was chosen because the eigenvectors in question are simple enough
for us to write them out more-or-less by hand, so this example helps the reader
to understand what happens under the hood in more complex cases.

Some of the code here is credited to Yao Shen who used this approach for the
analysis of a low valence nickelate material [1]_. The task performed repeats
analysis done by many researchers e.g. Luuk Ament et al [2]_ as well as
several other groups.
"""
import edrixs
import numpy as np
import matplotlib.pyplot as plt
import scipy

################################################################################
# Eigenvectors
# ------------------------------------------------------------------------------
# Let us start by determining the eigenvectors involved in the transitions.
# If we wanted to, we could build and diagonalize a suitable Hamiltonian, but
# it will be more convenient and pedagogical to write a function that generates
# them directly. Using hole language and the real harmonic basis we can generate
# the eignvector by making an array of zeroes and setting the relevant
# :code:`orbital_index` to one. The array can then be transformed into the
# complex harmonic basis. The spin direction can be set using a vector
# :math:`\vec{B}` to represent a magnetic field in terms of generalized spin
# operator :math:`\tilde{\sigma}=\vec{B}\cdot\sigma` based on the Pauli matrices
# :math:`\sigma`. A full spin-orbital
# eignvector with an arbitrary spin direction :math:`\Psi` can be
# generated in terms of orbital eigenvector :math:`\psi` as
#
#     .. math::
#        \Psi^{\uparrow} = \frac{1}{\sqrt{2}}
#        \left[U_- \psi^{\uparrow} + D_- \psi^{\uparrow}\right]
#
# where :math:`U_-` and :math:`D_-` are the first components of the two
# eigenvectors of :math:`\tilde{\sigma}`.


def make_eigenvector(orbital_index, B, case='d'):
    tmat_r2c = edrixs.tmat_r2c(case)
    norb = tmat_r2c.shape[0]
    eigenvector_rhb = np.zeros(norb, dtype=complex)
    eigenvector_rhb[orbital_index] = 1
    psi = np.dot(eigenvector_rhb, tmat_r2c)

    Spauli = edrixs.get_pauli()
    Emat_spin = sum(B[i]*Spauli[i] for i in range(3))
    _, v = scipy.linalg.eigh(Emat_spin)
    U, D = v[:, 0], v[:, 1]

    Psi = np.zeros(2*norb, dtype=complex)
    Psi[::2] = psi*U[0]
    Psi[1::2] = psi*D[0]
    return Psi


################################################################################
# Let's put the spin along the :math:`[1 1 0]` direction and
# recall from the :ref:`sphx_glr_auto_examples_example_1_crystal_field.py`
# example that edrixs uses the standard orbital order of
# :math:`d_{3z^2-r^2}, d_{xz}, d_{yz}, d_{x^2-y^2}, d_{xy}`, so we want the
# :code:`orbital_index = 3` element. Using this, we can build spin-up and -down
# eigenvectors.
orbital_index = 3
B = np.array([1, 1, 0])

groundstate_vector = make_eigenvector(orbital_index, B)
excitedstate_vector = make_eigenvector(orbital_index, -B)

################################################################################
# Transition operators
# ------------------------------------------------------------------------------
# Here we are considering a :math:`L_3`-edge :math:`2p_{3/2} \rightarrow 3d`
# transition. We can read this matrix from the edrixs database, keeping in
# mind that there are in fact three operations for :math:`x, y,` & :math:`z` directions.
# We will also need the :math:`3d \rightarrow 2p_{3/2}` operation, which we can
# generate as the conjugate transpose.

Tmat = edrixs.get_trans_oper('dp32')
Tmat_dag = np.array([np.conj(Tpol).T for Tpol in Tmat])

################################################################################
# Scattering matrix
# ------------------------------------------------------------------------------
# The angular dependence of a RIXS transition can be conveniently described
# using the scattering matrix, which is a :math:`3\times3` element object that
# specifies the transition amplitude for each incoming and outgoing x-ray
# polarization.


def get_F(vector_i, vector_f):
    F = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            F[i, j] = np.dot(np.dot(Tmat_dag[i], vector_f),
                             np.dot(vector_i, Tmat[j]))
    return F


################################################################################
# Using this function, we can obtain non-spin-flip (NSF) and spin-flip (SF)
# scattering matrices by choosing whether we return to the ground state or
# whether we access the excited state with the spin flipped.
F_NSF = get_F(groundstate_vector, groundstate_vector)
F_SF = get_F(groundstate_vector, excitedstate_vector)

################################################################################
# Angular dependence
# ------------------------------------------------------------------------------
# Let's consider the common case of fixing the total scattering angle at
# :code:`two_theta = 90` and choosing a series of incident angles :code:`thins`.
# Since the detector does not resolve polarization, we need to add both outgoing
# polarizations. It is then convenient to use function :func:`.dipole_polvec_rixs`
# to obtain the incoming and outgoing polarization vectors.
thins = np.linspace(0, 90)
two_theta = 90
phi = 0


def get_I(thin, alpha, F):
    intensity = 0
    for beta in [0, np.pi/2]:
        thout = two_theta - thin
        ei, ef = edrixs.dipole_polvec_rixs(thin*np.pi/180, thout*np.pi/180,
                                           phi*np.pi/180, alpha, beta)
        intensity += np.abs(np.dot(ef, np.dot(F, ei)))**2
    return intensity


################################################################################
# Plot
# ------------------------------------------------------------------------------
# We now run through a few configurations specified in terms of incoming
# polarization angle :math:`\alpha` (defined in radians w.r.t. the scattering
# plane), :math:`F`, plotting label, and plotting color.
fig, ax = plt.subplots()

config = [[0, F_NSF, r'$\pi$ NSF', 'C0'],
          [np.pi/2, F_NSF, r'$\sigma$ NSF', 'C1'],
          [0, F_SF, r'$\pi$ SF', 'C2'],
          [np.pi/2, F_SF, r'$\sigma$ SF', 'C3']]

for alpha, F, label, color in config:
    Is = np.array([get_I(thin, alpha, F) for thin in thins])
    ax.plot(thins, Is, label=label, color=color)

ax.legend()
ax.set_xlabel(r'Theta ($^\circ$)')
ax.set_ylabel('Relative intensity')
plt.show()
##############################################################################
#
# .. rubric:: Footnotes
#
# .. [1] Yao Shen et al.,
#        `arXiv:2110.08937 (2022) <https://arxiv.org/abs/2110.08937>`_.
# .. [2] Luuk J. P. Ament et al.,
#        `Phys. Rev. Lett. 103, 117003 (2009) <https://doi.org/10.1103/PhysRevLett.103.117003>`_
