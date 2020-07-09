#!/usr/bin/env python
"""
Ground state analysis for NiO 
================================================================================
This example follows :ref:`sphx_glr_auto_examples_example_2_AIM_XAS.py` and
considers the same model for NiO. This time we show how to analyze the ground 
wavefunction in terms of $d^8L^{10}$ $d^9L^9$ $d^{10}L^8$ contributions.

As an exercise we will see how build and diagonalize the Hamiltonian
describing this model. We need to consider :code:ntot=20` spin orbitals
for this problem. 
"""
import edrixs
import numpy as np
import scipy

################################################################################
# Hamiltonian
# ------------------------------------------------------------------------------
# edrixs build models based on the 
#    .. math::
#     \begin{equation}
#     \hat{H}_{i} = \sum_{\alpha,\beta} t_{\alpha,\beta}
#     \hat{f}^{\dagger}_{\alpha} \hat{f}_{\beta} 
#     + \sum_{\alpha,\beta,\gamma,\delta} U_{\alpha,\beta,\gamma,\delta} 
#     \hat{f}^{\dagger}_{\alpha}\hat{f}^{\dagger}_{\beta}\hat{f}_{\gamma}\hat{f}_{\delta},
#     \end{equation} 
#



################################################################################
# Import paramaters
# ------------------------------------------------------------------------------
# Let's get the parammeters we need from the previous example.

from example_2_AIM_XAS import (F0_dd, F2_dd, F4_dd,
                               nd, norb_d, norb_bath, v_noccu,
                               imp_mat, CF, bath_level,
                               hyb, ext_B)
ntot = 20
                               
################################################################################
# Four fermion matrix 
# ------------------------------------------------------------------------------
# The Coulomb matrix is a four dimensional matrice
umat_delectrons = edrixs.get_umat_slater('d', F0_dd, F2_dd, F4_dd)
umat = np.zeros((ntot, ntot, ntot, ntot), dtype=np.complex)
umat[:norb_d, :norb_d, :norb_d, :norb_d] += umat_delectrons


################################################################################
# Two fermion matrix 
# ------------------------------------------------------------------------------
emat = np.zeros((ntot, ntot), dtype='complex')
emat[0:norb_d, 0:norb_d] += imp_mat

indx = np.arange(norb_d, norb_d*2)
emat[indx, indx] += bath_level[0]
    
indx1  = np.arange(norb_d)
indx2 = np.arange(norb_d, norb_d*2)
emat[indx1, indx2] += hyb[0]
emat[indx2, indx1] += np.conj(hyb[0])
    
v_orbl = 2
sx = edrixs.get_sx(v_orbl)
sy = edrixs.get_sy(v_orbl)
sz = edrixs.get_sz(v_orbl)
zeeman = ext_B[0] * (2 * sx) + ext_B[1] * (2 * sy) + ext_B[2] * (2 * sz)
emat[0:norb_d, 0:norb_d] += zeeman


################################################################################
# Build Fock basis
# ------------------------------------------------------------------------------
basis = np.array(edrixs.get_fock_bin_by_N(ntot, v_noccu))


################################################################################
# Build and diagonalize Hamtiltonain
# ------------------------------------------------------------------------------
H = (edrixs.build_opers(2, emat, basis)
  + edrixs.build_opers(4, umat, basis))

e, v = scipy.linalg.eigh(H)
gs_v = v[:, 0]

num_d_electrons = basis[:, :norb_d].sum(1)
alpha = np.sum(np.abs(gs_v[num_d_electrons==8])**2)
beta = np.sum(np.abs(gs_v[num_d_electrons==9])**2)
gamma = np.sum(np.abs(gs_v[num_d_electrons==10])**2)
print("alpha={:.3f}\tbeta={:.3f}\tgamma={:.3f}".format(alpha, beta, gamma))

alpha+beta+gamma