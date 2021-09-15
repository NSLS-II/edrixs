#!/usr/bin/env python
# usage: python run_ed.py

import numpy as np
import scipy
import edrixs


"""
An example showing how to do exact diagonalization of a many-body Hamiltonian

.. math::

    \\hat{H}=\\sum_{\\alpha,\\beta}E_{\\alpha,\\beta}\\hat{f}^{\\dagger}_{\\alpha}
    \\hat{f}_{\\beta} + \\sum_{\\alpha,\\beta,\\gamma,\\delta}
    U_{\\alpha,\\beta,\\gamma,\\delta}\\hat{f}^{\\dagger}_{\\alpha}\\hat{f}^{\\dagger}_{\\beta}
    \\hat{f}_{\\gamma}\\hat{f}_{\\delta}

based on the second quantization language, where :math:`E_{\\alpha,\\beta}` is the matrix
for two-fermion terms and :math:`U_{\\alpha,\\beta,\\gamma,\\delta}` is the Coulomb
U-tensor for the four-fermion terms.

To build the Hamiltonian or other many-body operators, we need a Fock basis that is defined
on top of particular single-particle basis. In this example, we do ED on a :math:`t_{2g}`
system with the default single-particle basis of complex spherical harmonics (same to :math:`p`
orbital system) and the orbital ordering: |-1,up>, |-1,dn>, |0,up>, |0,dn>, |+1,up>, |+1,dn>.

To get the expectation values of some operators in the many-body states, one should first
get the matrix or 4-rank tensor in the single-particle basis, and then call build_opers()
to get the many-body operators.
"""
# t2g system, 6 orbitals and occupied by 2 electrons
norb = 6
noccu = 2
# Hubbard U and Hund's coupling J in the form of Kanamori Coulomb interaction
U, J = 4, 1
# Hubbard Ud and Hund's coupling JH in the form of Slater Coulomb interaction
Ud, JH = edrixs.UJ_to_UdJH(U, J)
# Slater integrals
F0, F2, F4 = edrixs.UdJH_to_F0F2F4(Ud, JH)
# Two fermion terms: spin-orbital coupling (SOC)
# The matrix is in the complex spherical Harmonics basis
soc_zeta = 0.4
emat_soc = edrixs.atom_hsoc('t2g', soc_zeta)
# Four fermion terms: Coulomb interaction
# The 4-rank tensor is in the complex spherical Harmonics basis
umat = edrixs.get_umat_slater('t2g', F0, F2, F4)
# Fock basis in the complex spherical Harmonics basis with the orbital ordering:
# |-1,up>, |-1,dn>, |0,up>, |0,dn>, |+1,up>, |+1,dn>
# basis: 2d list of integers with 1 or 0, the shape is (15, 6) in this case
# where, 15=6*5/2 is the total number of Fock basis and 6 is the total number of
# single-particle orbitals
basis = edrixs.get_fock_bin_by_N(norb, noccu)

# quantum number of orbital angular momentum for t2g: l=1
ll = 1
# Matrices of lx,ly,lz,sx,sy,sz,jx,jy,jz in the single-particle basis
# lx: l_orb[0], ly: l_orb[1], lz: l_orb[2]
l_orb = edrixs.get_orb_momentum(ll, True)
# sx: s_spin[0], sy: s_spin[1], sz: s_spin[2]
s_spin = edrixs.get_spin_momentum(ll)
# jx: j_so[0], jy: j_so[1], jz: j_so[2]
j_so = l_orb + s_spin

# very small Zeeman splitting along z-direction
emat_zeeman = (l_orb[2] + s_spin[2]) * 1e-10

# many-body operators of L^2, Lz
Lxyz = edrixs.build_opers(2, l_orb, basis)
L2 = np.dot(Lxyz[0], Lxyz[0]) + np.dot(Lxyz[1], Lxyz[1]) + np.dot(Lxyz[2], Lxyz[2])
Lz = Lxyz[2]
# many-body operators of S^2, Sz
Sxyz = edrixs.build_opers(2, s_spin, basis)
S2 = np.dot(Sxyz[0], Sxyz[0]) + np.dot(Sxyz[1], Sxyz[1]) + np.dot(Sxyz[2], Sxyz[2])
Sz = Sxyz[2]
# many-body operators of J^2, Jz
Jxyz = edrixs.build_opers(2, j_so, basis)
J2 = np.dot(Jxyz[0], Jxyz[0]) + np.dot(Jxyz[1], Jxyz[1]) + np.dot(Jxyz[2], Jxyz[2])
Jz = Jxyz[2]

# We can also calculate the expectation values of the occupancy number of
# t2g orbitals, dxz, dyz, dxy, :math:`\\hat{d}^{\\dagger}_{\\alpha}\\hat{d}_{\\alpha}`
# First, write their matrice in the real harmonics basis
# |dxz,up>, |dxz,dn>, |dyz,up>, |dyz,dn>, |dxy,up>, |dxy,dn>
# In this basis, they take simple form, only the diagonal terms have element 1
nd_oper = np.zeros((norb, norb, norb), dtype=complex)
nd_oper[0, 0, 0] = 1
nd_oper[1, 1, 1] = 1
nd_oper[2, 2, 2] = 1
nd_oper[3, 3, 3] = 1
nd_oper[4, 4, 4] = 1
nd_oper[5, 5, 5] = 1
# Then transform to the complex harmonics basis
# |-1,up>, |-1,dn>, |0,up>, |0,dn>, |+1,up>, |+1,dn>
# comment the following line to calculate the occupancy number
# of the complex harmonics orbitals
nd_oper = edrixs.cb_op(nd_oper, edrixs.tmat_r2c('t2g', True))
# Build the many-body operators for nd
nd_manybody_oper = edrixs.build_opers(2, nd_oper, basis)

# Build many-body Hamiltonian for four-fermion terms in the Fock basis
# H has the dimension of 15*15
H_U = edrixs.build_opers(4, umat, basis)
# Build many-body Hamiltonian for two-fermion terms in the Fock basis
H_soc = edrixs.build_opers(2, emat_soc, basis)
H_zeeman = edrixs.build_opers(2, emat_zeeman, basis)

# The scipy diagonalization returns eigenvalues in ascending order, each repeated according
# to its multiplicity. eigenvectors are returned as a set of column vectors.
# eigenvalue eigenval[n] is associated with eignevector eigenvec[:, n]
# case 1: without SOC
H = H_U + H_zeeman
eigenval, eigenvec = scipy.linalg.eigh(H)

# print out eigenval
edrixs.write_tensor(eigenval, "eigenval_nosoc.dat")
# print out the non-zeros of coefficients of wave function by one by
# By default, write_tensor print row after row, so we print out the transpose
# of eigenvec, so for example, one line of " 1 3 0.5 0.5" is the 3rd coefficient
# 0.5+0.5i of the 1st eigenvector
edrixs.write_tensor(eigenvec.T, "eigenvec_nosoc.dat", only_nonzeros=True)

# expectation values
# without SOC, good quantum numbers without Zeeman splitting are: [L2, Lz, S2, Sz]
[L2_expt, Lz_expt, S2_expt, Sz_expt] = edrixs.cb_op([L2, Lz, S2, Sz], eigenvec)
edrixs.write_tensor(L2_expt.diagonal().real, "L2_nosoc.dat")
edrixs.write_tensor(Lz_expt.diagonal().real, "Lz_nosoc.dat")
edrixs.write_tensor(S2_expt.diagonal().real, "S2_nosoc.dat")
edrixs.write_tensor(Sz_expt.diagonal().real, "Sz_nosoc.dat")

# get the expectation values for nd
nd_expt = np.array([edrixs.cb_op(i, eigenvec).diagonal().real for i in nd_manybody_oper])
edrixs.write_tensor(nd_expt.T, "nd_nosoc.dat")

# case 2: with SOC
H = H_U + H_soc + H_zeeman
eigenval, eigenvec = scipy.linalg.eigh(H)

edrixs.write_tensor(eigenval, "eigenval_withsoc.dat")
edrixs.write_tensor(eigenvec.T, "eigenvec_withsoc.dat", only_nonzeros=True)

# with SOC, since the SOC Hamiltonian does not commute with L2, S2,
# so good quantum numbers without Zeeman splitting are: [J2, Jz]
[L2_expt, S2_expt, J2_expt, Jz_expt] = edrixs.cb_op([L2, S2, J2, Jz], eigenvec)
edrixs.write_tensor(L2_expt.diagonal().real, "L2_withsoc.dat")
edrixs.write_tensor(S2_expt.diagonal().real, "S2_withsoc.dat")
edrixs.write_tensor(J2_expt.diagonal().real, "J2_withsoc.dat")
edrixs.write_tensor(Jz_expt.diagonal().real, "Jz_withsoc.dat")

# get the expectation values for nd
nd_expt = np.array([edrixs.cb_op(i, eigenvec).diagonal().real for i in nd_manybody_oper])
edrixs.write_tensor(nd_expt.T, "nd_withsoc.dat")
