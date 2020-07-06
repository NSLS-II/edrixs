#!/usr/bin/env python
"""
Anderson impurity model for NiO XAS (Under contruction)
================================================================================
Here we calculate the :math:`L`-edge XAS spectrum of an Anderson impurity model.
This is sometimes also called a charge-transfer multiplet model.
Everyone's favorite test case for this is NiO and we won't risk being original!

The model considers the Ni :math:`3d` orbitals including crystal field, 
magnetic field, spin- orbit couping and Coulomb interactions and a surrounding 
set of O :math:`2p` orbitals. NiO has a rocksalt structure in which all Ni 
atoms are surrounded by six O atoms in cubic symmetry. Based on this, 
one would assume that the NiO cluster used to simulate the crystal would 
need to contain :math:`6` spin-orbitals per O 
:math:`\\times 6` O atoms :math:`=36` oxygen 
spin-orbitals. As explained by, for example, Maurits Haverkort et al., in
`Phys. Rev. B 85, 165113 (2012) 
<https://doi.org/10.1103/PhysRevB.85.165113>`_ only 10 of these orbitals
interact with the Ni atom and solving the problem with :math:`10+10 = 20`
spin-orbitals is far faster. For more general calculations of this type you 
may see the Ni states being referred to as the impurity or metal and the
O states being called the bath or ligand.

We consider on-site crystal field, spin-orbit coupling, magnetic field and
Coulomb interactions for the impurity, which is hybridized with the bath by
defining hopping parameters and an energy difference between the impurity 
and bath. In this way, our spectrum can include processes where
electrons transition from the bath to the impurity.

The crystal field and hopping parameters for such a calculation can be
calculated by DFT. We will use values for NiO from
`Phys. Rev. B 85, 165113 (2012) 
<https://doi.org/10.1103/PhysRevB.85.165113>`_ [1]_. 
"""
import edrixs
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpi4py import MPI
import sympy

################################################################################
# Number of electrons
# ------------------------------------------------------------------------------
# When formulating problems of this type, one usually thinks of a nominal 
# valence for the impurity atom in this case :code:`nd = 8` and assume that the
# bath of :code:`norb_bath  = 10` O spin-orbitals is full. The solver that we will
# use can simulate multiple bath sites. In our case we specify 
# :code:`nbath  = 1` sites. The total number of valance electrons :code:`v_noccu`
# will be conserved.
nd = 8
norb_d = 10
norb_bath  = 10 # number of spin-orbitals for each bath site
nbath = 1
v_noccu  = nd + nbath*norb_d
shell_name = ('d', 'p') # valence and core shells for XAS calculation

################################################################################
# Coulomb interactions 
# ------------------------------------------------------------------------------
# The atomic Coulomb interactions are usually initialized based on Hartree-Fock
# calculations from, for example,  
# `Cowan's code <https://www.tcd.ie/Physics/people/Cormac.McGuinness/Cowan/>`_.
# edrixs has a database of these. We get our desired values for Ni via
info  = edrixs.utils.get_atom_data('Ni', '3d', nd, edge='L3')

################################################################################
# The atomic values are typically scaled to account for screening in the solid. 
# Here we use 80% scaling. Let's write these out in full, so that nothing is 
# hidden. A note about implementing finite :code:`Udd` and :code:`Udp` will be
# added shortly. 
scale_dd = 0.8
F2_dd = info['slater_i'][1][1] * scale_dd
F4_dd = info['slater_i'][2][1] * scale_dd
Udd = 0
F0_dd = Udd + edrixs.get_F0('d', F2_dd, F4_dd)

scale_dp = 0.8
F2_dp = info['slater_n'][4][1] * scale_dp
G1_dp = info['slater_n'][5][1] * scale_dp
G3_dp = info['slater_n'][6][1] * scale_dp
Udp = 0
F0_dp = Udp + edrixs.get_F0('dp', G1_dp, G3_dp)

slater = ([F0_dd, F2_dd, F4_dd],  # initial
          [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp])  # with core hole

################################################################################
# The spin-orbit coupling for the valence electrons in the ground state, the
# valence electrons with the core hole present, and for the core hole itself
# are initialized using the atomic values.
zeta_d_i = info['v_soc_i'][0]
zeta_d_n = info['v_soc_n'][0]
c_soc = info['c_soc']

################################################################################
# Build matrices describing interactions
# ------------------------------------------------------------------------------
# edrixs uses complex spherical harmonics as its default basis set. If we want to
# use another basis set, we need to pass a matrix to the solver, which transforms
# from complex spherical harmonics into the basis we use. 
# The solver will use this matrix when implementing the Coulomb intereactions 
# using the :code:`slater` list of Coulomb parameters.
# Here it is easiest to 
# use real harmoics. We make the complex harmoics to real harmonics transformation
# matrix via
trans_c2n = edrixs.tmat_c2r('d',True)

################################################################################
# The crystal field and SOC needs to be passed to the solver by constructing
# the impurity matrix in the real harmonic basis. For cubic symmetry, we need
# to set the energies of the orbitals along the
# diagonal of the matrix. These need to be in pairs as there are two 
# spin-orbitals for each orbital energy. Python 
# `list comprehension <https://realpython.com/list-comprehension-python/>`_
# and
# `numpy indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_
# are used here.
ten_dq = 0.56
CF = np.zeros((10, 10), dtype=np.complex)
diagonal_indices = np.arange(norb_d)

orbital_energies = np.array([e for orbital_energy in
                             [+0.6 * ten_dq, # dz2
                              -0.4 * ten_dq, # dzx
                              -0.4 * ten_dq, # dzy
                              +0.6 * ten_dq, # dx2-y2
                              -0.4 * ten_dq] # dxy)
                             for e in [orbital_energy]*2]),


CF[diagonal_indices, diagonal_indices] = orbital_energies                  

################################################################################
# The valence band SOC is constructed in the normal way and transformed into the
# real harmonic basis.
soc = edrixs.cb_op(edrixs.atom_hsoc('d', zeta_d_i), edrixs.tmat_c2r('d', True))

################################################################################
# The total impurity matrix is then
imp = CF + soc

################################################################################
# The charge transfer :code:`Delta` and  monopole Coulomb :code:`Udd` parameters
# set the centers of the different electonic configurations before they are
# split. We choose
Delta = 4.7

################################################################################
# We define the state with a full set of bath orbitals to be zero
# energy and write the energy levels using the same definitions as the famous
# J. Zaanen, G. A. Sawatzky, and J. W. Allen paper
# `Phys. Rev. Lett. 55, 418 (1985) <https://doi.org/10.1103/PhysRevLett.55.418>`_
# . Papers by Maurits Haverkort et al., 
# `Phys. Rev. B 85, 165113 (2012) <https://doi.org/10.1103/PhysRevB.85.165113>`_
# and A. E. Bocquet et al., 
# `Phys. Rev. B 53, 1161 (1996) <https://doi.org/10.1103/PhysRevB.53.1161>`_
# were also used in preparing this section. 
#   
# * :math:`d^{nd}L^{10}` has energy :math:`0`
#
# * :math:`d^{nd+1}L^9` has energy :math:`\Delta`
#
# * :math:`d^{nd+2}L^8` has energy :math:`2\Delta + U`
#
# Using this we can write and solve three linear equations. The exercise of 
# doing this by hand is not so hard, but if you are familiar with the
# python package `sympy <http://www.sympy.org>`_ you can make the computer
# do it for you.
D, U, E_p, E_d, n = sympy.symbols('\Delta U E_p E_d, n')
eq1 = sympy.Eq(10*E_p +n*E_d     +  n*(n-1)*U/2,    rhs=0)
eq2 = sympy.Eq(9*E_p  +(n+1)*E_d + (n+1)*n*U/2,     rhs=D)
eq3 = sympy.Eq(8*E_p  +(n+2)*E_d + (n+1)*(n+2)*U/2, rhs=2*D+U)

################################################################################
#
#    .. math::
#        \begin{aligned}
#        n E_{d}  + 10 E_{p} + \frac{U n \left(n - 1\right)}{2} &= 0\\
#        E_{d} \left(n + 1\right) + 9 E_{p} + \frac{U n \left(n + 1\right)}{2} &= \Delta \\
#        E_{d} \left(n + 2\right) + 8 E_{p} + \frac{U \left(n + 1\right) \left(n + 2\right)}{2}  &= U + 2 \Delta
#        \end{aligned}
#
# and 

answer = sympy.solve([eq1, eq2, eq3], (E_d, E_p))
################################################################################
#
#    .. math::
#        \begin{aligned}
#        E_p &= - \frac{U n^{2} + 19 U n - 20 \Delta}{2 n + 20} \\
#        E_d &= \frac{n \left(U n + U - 2 \Delta\right)}{2 \left(n + 10\right)}
#        \end{aligned}
#

d_energy = answer[E_d].evalf(subs={U:Udd, n:nd, D:Delta})
p_energy = answer[E_p].evalf(subs={U:Udd, n:nd, D:Delta})
print("impurity_energy={:.3f}\nbath_energy={:.3f}".format(d_energy, p_energy))

################################################################################
# another print method
sympy.init_printing(use_latex='mathjax')
sympy.pprint(eq1)
################################################################################
# The energy level of the bath(s) is described by a matrix where the row index 
# denotes which bath and the column index denotes which orbital. Here we have
# only one bath, with 10 spin-orbitals all of same energy set as above.
bath_level = np.full((nbath, norb_d), p_energy, dtype=np.complex)
bath_level

################################################################################
# The hybridization matrix describes the hopping between the bath
# to the impuirty. This is the same shape as the bath matrix. We take our
# values from Maurits Haverkort et al.'s DFT calculations
# `Phys. Rev. B 85, 165113 (2012) <https://doi.org/10.1103/PhysRevB.85.165113>`_ 
Tb1 = 2.06 # x2-y2
Ta1 = 2.06 # 3z2-r2
Tb2 = 1.21 # xy
Te  = 1.21 # zx/yz
    
hyb = np.zeros((nbath, norb_d), dtype=np.complex)
hyb[0, :2] = Ta1  # 3z2-r2
hyb[0, 2:6] = Te  # zx/yz
hyb[0, 6:8] = Tb1  # x2-y2
hyb[0, 8:] = Tb2  # xy


################################################################################
# We now need to define the parameters describing the XAS. X-ray polariztion
# can be linear, circular, isotropic (appropriate for a powder).
poltype_xas = [('isotropic', 0)]
################################################################################
# edrixs uses the tempretur in Kelvin to work out the population of the low-lying
# states via a Boltzmann distribution.
temperature = 300
################################################################################
# The x-ray beam is specified by the incident angle and azimuthal angle in radians
thin =  0 / 180.0 * np.pi
phi = 0.0
################################################################################
# these are with respect to the crystal field :math:`z` and :math:`x` axes 
# written above. (That is, unless you specify the code:`loc_axis` parameter
# described in the :code:`edrixs.xas_siam_fort` function documenation.)

################################################################################
# The calculation does not know the energy cost of a core hole or the inverse 
# core hole lifetime so we need to specify these. Usually this these are though 
# of as adjustablparameters for comparing theory to experiment. We use this to 
# specify the range we want to compute the spectrum over.
om_shift = 852.7
ominc_xas = om_shift + np.linspace(-15, 25, 1000)
gamma_c_stat = 0.48/2

################################################################################
# The final broadening is specified in terms of half width at half maxium
# You can either pass a constant value of an array the same size as
# :code:`om_shift` with varying values to simulate, for example, different state
# lifetimes for higher energy states.
gamma_c = np.full(ominc_xas.shape, 0.48/2)   

################################################################################
# Magnetic field is a three-component vector in eV specified with respect to the
# same local axis as the x-ray beam. Since we are considering a powder here
# we create an isotropic normalized vector. :code:`on_which = 'both'` specifies to
# apply the operator to the total spin plus orbital angular momentum as is
# appropriate for a physical external magnetic field. You can use 
# :code:`on_which = 'spin' to apply the operator to spin in order to simulate
# magnetic order in the sample. The value of the Bohr Magneton can
# be useful for converting here :math:`\mu_B = 5.7883818012\times 10^{−5}` 
# eV/T.
ext_B = np.array([0.01, 0.01, .01])/np.sqrt(3)
on_which = 'both'
    
################################################################################
# The number crunching uses
# `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_. You can safely ignore 
# this for most purposes, but see 
# `Y. L. Wang et al., Computer Physics Communications 243, 151-165 (2019) <https://doi.org/10.1016/j.cpc.2019.04.018>`_ if you would like more details.
# The main thing to remember is that you should call this script via::
#
#        mpirun -n <number of processors> python example_AIM_XAS.py
#
# where :code:`<number of processors>` is the number of processsors
# you'd like to us. Running it as normal will work, it will just be slower.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() 

################################################################################
# Calling the :code:`edrixs.ed_siam_fort` solver will find the ground state and
# write input files, *hopping_i.in*, *hopping_n.in*, *coulomb_i.in*, *coulomb_n.in*
# for following XAS (or RIXS) calculation. We need to specify :code:`siam_type=0`
# which says that we will pass *imp_mat*, *bath_level* and *hyb*.
# We need to specify :code:`do_ed = 1`. More details of why will be included
# later. 

do_ed = 1
eval_i, denmat, noccu_gs = edrixs.ed_siam_fort(
    comm, shell_name, nbath, siam_type=0, imp_mat=imp,
    bath_level=bath_level, hyb=hyb, c_level=-om_shift, c_soc=c_soc,
    static_core_pot=gamma_c_stat, slater=slater, ext_B=ext_B, on_which=on_which,
    trans_c2n=trans_c2n, v_noccu=v_noccu, do_ed=do_ed, ed_solver=2, neval=20,
    nvector=3, ncv=40, idump=True)

################################################################################
# Let's check that we have all the electrons we think we have and print how 
# the electron are distributed between the Ni (impurity) and O (bath).
assert np.abs(noccu_gs - v_noccu) < 1e-6
impurity_occupation = np.sum(denmat[0].diagonal()[0:norb_d]).real
bath_occupation = np.sum(denmat[0].diagonal()[norb_d:]).real
print('Impurity occupation = {:.6f}\n'.format(impurity_occupation))
print('Bath occupation = {:.6f}\n'.format(bath_occupation))

################################################################################
# We see that 0.36 electrons move from the O to the Ni in the ground state. 
# 
# We can now construct the XAS spectrum edrixs is applying a transition
# operator to create the excitated state.  
xas, xas_poles = edrixs.xas_siam_fort(
    comm, shell_name, nbath, ominc_xas, gamma_c=gamma_c, v_noccu=v_noccu, thin=thin,
    phi=phi, num_gs=3, nkryl=200, pol_type=poltype_xas, temperature=temperature
)


################################################################################
# Let's plot the data and save it just in case
fig, ax = plt.subplots()

ax.plot(ominc_xas, xas)
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('XAS intensity')
ax.set_title('Anderson impurity model for NiO')

np.savetxt('xas.dat', np.concatenate((np.array([ominc_xas]).T, xas), axis=1))


################################################################################
#
# .. rubric:: Footnotes
# 
# .. [1] If you use these values in a paper you should, of course, cite it.
#