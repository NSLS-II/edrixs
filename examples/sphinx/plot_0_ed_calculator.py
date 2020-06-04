#!/usr/bin/env python
"""
Using edrixs as a ED calculator
===============================
Here we show how to find the eigenvalues and eigenvectors of a many-body
Hamiltonian of fermions with Coulomb interactions. We then determine their spin
and orbital angular momentum and how this changes when we switch on spin-orbit
coupling.
"""

################################################################################
# Import the necessary modules and set print options for later.
import numpy as np
import scipy
import edrixs

np.set_printoptions(precision=6, linewidth=200, suppress=True)


################################################################################
# Define the number of orbitals, the orbital occupancy and the Slater integrals.
# :math:`F^{k}` with :math:`k=1,2`:
norb = 6
noccu = 2
F0, F2 = 4.0, 1.0
################################################################################
# We start by calculating the Coulomb interaction tensor which is parameterized
# by
#
#    .. math::
#        \begin{gather*}
#        U_{m_{l_i}m_{s_i}, m_{l_j}m_{s_j}, m_{l_t}m_{s_t},
#        m_{l_u}m_{s_u}}^{i,j,t,u}
#        = \\ \frac{1}{2} \delta_{m_{s_i},m_{s_t}}\delta_{m_{s_j},m_{s_u}}
#        \delta_{m_{l_i}+m_{l_j}, m_{l_t}+m_{l_u}}
#        \sum_{k}C_{l_i,l_t}(k,m_{l_i},m_{l_t})C_{l_u,l_j}
#        (k,m_{l_u},m_{l_j})F^{k}_{i,j,t,u}
#        \end{gather*}
#
#    where :math:`m_s` is the magnetic quantum number for spin
#    and :math:`m_l` is the magnetic quantum number for orbital.
#    :math:`F^{k}_{i,j,t,u}` are Slater integrals.
#    :math:`C_{l_i,l_j}(k,m_{l_i},m_{l_j})` are Gaunt coefficients.
#
umat = edrixs.get_umat_slater('p', F0, F2)

################################################################################
# Build the binary form of the Fock basis.
# This is the simplest legitimate form for the basis
basis = edrixs.get_fock_bin_by_N(norb, noccu)
print(np.array(basis))

################################################################################
# Build the Hamiltonian, i.e. the four fermion terms.
#
#.. math::
#
#    <F_l|\sum_{ij}U_{ijkl}\hat{f}_{i}^{\dagger}\hat{f}_{j}^{\dagger}
#    \hat{f}_{k}\hat{f}_{l}|F_r>
#
#

n_fermion = 4
H = edrixs.build_opers(n_fermion, umat, basis)

################################################################################
# For a small problem it is convenient to use the
e1, v1 = scipy.linalg.eigh(H)

orb_mom = edrixs.get_orb_momentum(1, True)
spin_mom = edrixs.get_spin_momentum(1)
tot_mom = orb_mom + spin_mom
opL, opS, opJ = edrixs.build_opers(2, [orb_mom, spin_mom, tot_mom], basis)
# L^2 = Lx^2 + Ly^2 +Lz^2, S^2 = Sx^2 + Sy^2 + Sz^2, J^2 = Jx^2 + Jy^2 + Jz^2
L2 = np.dot(opL[0], opL[0]) + np.dot(opL[1], opL[1]) + np.dot(opL[2], opL[2])
S2 = np.dot(opS[0], opS[0]) + np.dot(opS[1], opS[1]) + np.dot(opS[2], opS[2])
J2 = np.dot(opJ[0], opJ[0]) + np.dot(opJ[1], opJ[1]) + np.dot(opJ[2], opJ[2])

# print out results
print("Without SOC")
L2_val = edrixs.cb_op(L2, v1).diagonal().real
S2_val = edrixs.cb_op(S2, v1).diagonal().real
print("    #           eigenvalue     L(L+1)    S(S+1)")
for i, e in enumerate(e1):
    print("{:5d}{:20.6f}{:10.1f}{:10.1f}".format(i, e, L2_val[i], S2_val[i]))

# add spin-orbit coupling (SOC)
soc = edrixs.atom_hsoc('p', 0.2)
# build Hamiltonian, two_fermion terms
H2 = H + edrixs.build_opers(2, soc, basis)
# do ED
e2, v2 = scipy.linalg.eigh(H2)
print("With SOC")
print("    #           eigenvalue     J(J+1)")
J2_val = edrixs.cb_op(J2, v2).diagonal().real
for i, e in enumerate(e2):
    print("{:5d}{:20.6f}{:10.1f}".format(i, e, J2_val[i]))
