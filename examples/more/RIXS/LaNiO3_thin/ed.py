#!/usr/bin/env python

import sys
import numpy as np
from soc                import  atom_hsoc

from edrixs.iostream           import  write_tensor, write_tensor_nonzeros
from edrixs.fock_basis         import  get_fock_bin_by_N
from edrixs.coulomb_utensor    import  get_umat_slater, get_F0
from edrixs.basis_transform    import  cb_op, cb_op2, tmat_r2c
from edrixs.photon_transition  import  get_trans_oper
from edrixs.manybody_operator  import  two_fermion, four_fermion

if __name__ == "__main__":
    '''
    Purpose: This is a "Hello World !" case for RIXS simulation.
             This example shows how to exactly diagonalize a Hamiltonian for valence electrons and for valence + core electrons,
             and also shows how to generate dipolar transition operators. 
             This example use purely python code.
    '''
    # The parameters used in this calculation are taken from [PRL 117, 147401 (2016)]
    # First, we need to know which atomic shells are involved in the calculations, here, we are performing Ni-L2, L3 edges X-ray scattering,
    # which means Ni-2p --> Ni-3d transitions, so we have two atomic shells: one is the valence shell 3d(l=2) and another is core-electron 
    # shell 2p(l=1). Here, we use a convention that the orbitals of valence shells are always in front of that of the core-electron shells.
    # For this case, we work on the complex spherical harmonics basis, so the orbital order will be: 
    # |2,-2,up> |2,-2,dn>, |2,-1,up>, |2,-1,dn>, ..., |2,2,up>, |2,2,dn>     |1,-1,up>, |1,-1,dn>, ..., |1,1,up>, |1,1,dn> 
    # where, ``up" means spin up and ``dn" means spin down

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

    # Crystal Field 
    # For this case, it has a tetrgonal symmetry, see [https://doi.org/10.1016/j.ccr.2004.03.018] for details.
    # cubic splitting
    dq10 = 1.3
    # for tetrgonal splitting
    d1, d3 = 0.05, 0.2
    # we use dt, ds, dq finally
    dt = (3.0*d3 - 4.0*d1)/35
    ds = (d3 + d1)/7.0
    dq = dq10/10.0

    # END of PARAMETERS:
    #-------------------



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
    # For 3d real harmonics, the default orbital in our code is: d3z2-r2, dzx, dzy, dx2-y2, dxy
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

    # build Fock basis (binary representation), without core-hole
    # 8 3d-electrons and 6 2p-electrons
    basis_i = get_fock_bin_by_N(ndorb, 8, nporb, nporb)
    ncfgs_i = len(basis_i)
    
    # build Fock basis (binary representation), with a core-hole
    # 9 3d-electrons and 5 2p-electrons
    basis_n = get_fock_bin_by_N(ndorb, 9, nporb, nporb-1)
    ncfgs_n = len(basis_n)

    # build many-body Hamiltonian, without core-hole
    print("edrixs >>> building Hamiltonian without core-hole ...")
    hmat_i = np.zeros((ncfgs_i,ncfgs_i),dtype=np.complex128)
    hmat_n = np.zeros((ncfgs_n,ncfgs_n),dtype=np.complex128)
    emat = hsoc_i + crys
    hmat_i[:,:] += two_fermion(emat,  basis_i, basis_i)
    hmat_i[:,:] += four_fermion(umat_i, basis_i)
    print("edrixs >>> Done!")

    # build many-body Hamiltonian, with a core-hole
    print("edrixs >>> building Hamiltonian with a core-hole ...")
    emat = hsoc_n + crys
    hmat_n[:,:] += two_fermion(emat,   basis_n, basis_n)
    hmat_n[:,:] += four_fermion(umat_n, basis_n)
    print("edrixs >>> Done!")

    # diagonalize the many-body Hamiltonian to get eigenvalues and eigenvectors
    print("edrixs >>> diagonalizing the many-body Hamiltonian ... ")
    eval_i, evec_i = np.linalg.eigh(hmat_i)
    eval_n, evec_n = np.linalg.eigh(hmat_n)
    print("edrixs >>> Done!")

    # write eigenvalues to files, the eigenvalues are needed for XAS and RIXS calculations
    write_tensor(eval_i, "eval_i.dat")
    write_tensor(eval_n, "eval_n.dat")
 
    # build dipolar transition operator from 2p to 3d
    print("edrixs >>> building dipolar transition operators from 2p to 3d ...")
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
 
    # write it to file, will be used by XAS and RIXS calculations
    write_tensor(dipole_op, "trans_mat.dat")
    print("edrixs >>> Done!")
    print("edrixs >>> ED Done !")
