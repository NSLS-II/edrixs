#!/usr/bin/env python

import sys
import numpy as np
from soc                import  atom_hsoc
from iostream           import  write_tensor, write_tensor_nonzeros
from fock_basis         import  get_fock_bin_by_N
from coulomb_utensor    import  get_umat_slater, get_F0
from basis_transform    import  cb_op, cb_op2, tmat_r2c, tmat_c2j, transform_utensor
from photon_transition  import  get_trans_oper
from manybody_operator  import  two_fermion, four_fermion
from angular_momentum   import  get_sx, get_sy, get_sz, get_lx, get_ly, get_lz

if __name__ == "__main__":
    '''
    Purpose: This example shows how to exactly diagonalize a Hamiltonian for valence electrons (5f) and for valence + core electrons (5d),
             and also shows how to generate dipolar transition operators from 5d to 5f. 
             This example use purely python code.
    '''

    # PARAMETERS
    #-----------

    # The parameters used in this calculation are taken from [PRL 114, 236401 (2015)]
    # First, we need to know which atomic shells are involved in the calculations, here, we are performing U-O4/O5 edges X-ray scattering,
    # which means U-5d --> U-5f transitions, so we have two atomic shells: one is the valence shell 5f(l=3) and another is core-electron 
    # shell 5d(l=2). We will use a convention that the orbitals of valence shells are always in front of that of the core-electron shells.
    # For this case, we work on the complex spherical harmonics basis, so the orbital order will be: 
    # |3,-3,up> |3,-3,dn>, |3,-2,up>, |3,-2,dn>, ..., |3,3,up>, |3,3,dn>     |2,-2,up>, |2,-2,dn>, ..., |2,2,up>, |2,2,dn> 
    # where, ``up" means spin up and ``dn" means spin down

    # 14 5f-orbitals, 10 5d-orbitals (including spin)
    nforb, ndorb = 14, 10
    # total orbitals
    norbs = nforb + ndorb
 
    # Slater integrals
    # The Slater integrals are calculated by the Cowan's code [https://www.tcd.ie/Physics/people/Cormac.McGuinness/Cowan/] 
    # by using Hartree-Fock mean-field method, and then re-scaled, for example, Fff: 77%, Ffd: 60%, Gfd: 60%
    F2_f, F4_f, F6_f = 9.711*0.77,  6.364*0.77,  4.677*0.77
    # Here, Uf_av is the averaged Coulomb interaction in 5f shell, we set it to be zero for simplicity.
    # It doesn't matter here because we are doing single atomic calculation with fixed 5f occupancy.
    Uf_av = 0.0 
    # F0_ff is split into two parts: (1) Uf_av, (2) related to F2_f, F4_f, F6_f
    F0_f = Uf_av + get_F0('f', F2_f, F4_f, F6_f)

    F2_fd, F4_fd = 10.652*0.6,  6.850*0.6 
    G1_fd, G3_fd, G5_fd = 12.555*0.6,  7.768*0.6,  5.544*0.6 
    # Ufd_av is the averaged Coulomb interaction between 5f and 5d shells, we set it to be zero here.
    Ufd_av = 0.0
    # F0_fd is split into two parts: (1) Ufd_av, (2) related to G1_fd, G3_fd, G5_fd
    F0_fd = Ufd_av + get_F0('fd', G1_fd, G3_fd, G5_fd)

    # Spin-Orbit Coupling (SOC) zeta
    # 5f, without core-hole, from Cowan's code
    zeta_f_i = 0.261*0.9
    # 5f, with core-hole, from Cowan's code
    zeta_f_n = 0.274*0.9
    # 5d core electron, from Cowan's code, and usually we will adjust it to match the energy difference of the two edges,
    # for example, roughly, E(O4-edge) - E(O5-edge) = 2.5 * zeta_d_n 
    zeta_d_n = 3.2

    # END of Parameters
    #------------------


    # NOTE: the RIXS scattering involves 2 process, the absorption and emission, so we have 3 states:
    # (1) before the absorption, only valence shells are involved, we use the letter (i) to label it
    # (2) after the absorption, a core hole is created in the core shell, both the valence and core shells are involved, we use
    #     the letter (n) to label it
    # (3) after the emission, the core hole is re-filled by an electron, we use the letter (f) to label it.

    tmat = np.zeros((norbs, norbs), dtype=np.complex128)
    tmat[0:nforb, 0:nforb] = tmat_c2j(3)
    tmat[nforb:norbs, nforb:norbs] = tmat_c2j(2)
    # Coulomb interaction tensors without core-hole
    params=[
             F0_f, F2_f, F4_f, F6_f,       # Fk for f
             0.0, 0.0, 0.0,                # Fk for fd
             0.0, 0.0, 0.0,                # Gk for fd
             0.0, 0.0, 0.0                 # Fk for d
           ]
    umat_tmp = get_umat_slater('fd', *params)
    umat_i = transform_utensor(umat_tmp, tmat)

    # Coulomb interaction tensors with a core-hole
    params=[
             F0_f, F2_f, F4_f, F6_f,       # Fk for f
             F0_fd, F2_fd, F4_fd,          # Fk for fd
             G1_fd, G3_fd, G5_fd,          # Gk for fd
             0.0, 0.0, 0.0                 # Fk for d
           ] 
    umat_tmp = get_umat_slater('fd', *params)
    umat_n = transform_utensor(umat_tmp, tmat)
    
    # build SOC matrix
    hsoc_i = np.zeros((norbs,norbs), dtype=np.complex128)
    hsoc_n = np.zeros((norbs,norbs), dtype=np.complex128)
    hsoc_i[0:nforb, 0:nforb] += cb_op(atom_hsoc('f', zeta_f_i), tmat_c2j(3))
    hsoc_n[0:nforb, 0:nforb] += cb_op(atom_hsoc('f', zeta_f_n), tmat_c2j(3))
    hsoc_n[nforb:norbs, nforb:norbs] += cb_op(atom_hsoc('d', zeta_d_n), tmat_c2j(2))
    
    # build Fock basis, without core-hole
    # 2 5f-electrons and 10 5d-electrons
    basis_i = get_fock_bin_by_N(nforb, 2, ndorb, ndorb)
    ncfgs_i = len(basis_i)
    
    # build Fock basis, with a core-hole
    # 3 5f-electrons and 9 5d-electrons
    basis_n = get_fock_bin_by_N(nforb, 3, ndorb, ndorb-1)
    ncfgs_n = len(basis_n)

    # build many body Hamiltonian, without core-hole
    print("edrixs >>> building Hamiltonian without core-hole ...")
    hmat_i = np.zeros((ncfgs_i,ncfgs_i),dtype=np.complex128)
    hmat_n = np.zeros((ncfgs_n,ncfgs_n),dtype=np.complex128)

    hmat_i[:,:] += two_fermion(hsoc_i, basis_i, basis_i)
    hmat_i[:,:] += four_fermion(umat_i, basis_i)
    print("edrixs >>> Done!")

    # build many body Hamiltonian, with a core-hole
    print("edrixs >>> building Hamiltonian with a core-hole ...")
    hmat_n[:,:] += two_fermion(hsoc_n, basis_n, basis_n)
    hmat_n[:,:] += four_fermion(umat_n, basis_n)
    print("edrixs >>> Done!")

    # diagonalize hamt to get eigenvalues and eigenvectors
    print("edrixs >>> diagonalize Hamiltonian ...")
    eval_i, evec_i = np.linalg.eigh(hmat_i)
    eval_n, evec_n = np.linalg.eigh(hmat_n)
    print("edrixs >>> Done!")

    # write eigenvalues to files, the eigenvalues are needed for XAS and RIXS calculations
    write_tensor(eval_i, "eval_i.dat")
    write_tensor(eval_n, "eval_n.dat")

    # build dipolar transition operator from 2p to 3d
    print("edrixs >>> building dipolar transition operators from 5d to 5f ...")
    # build the transition operator in the Fock basis
    dipole = np.zeros((3, norbs, norbs), dtype=np.complex128)
    dipole_op = np.zeros((3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    # Please note that, here the function returns operators x, y, z
    tmp = get_trans_oper('fd')
    for i in range(3):
        dipole[i, 0:nforb, nforb:norbs] = cb_op2(tmp[i], tmat_c2j(3), tmat_c2j(2))
        dipole_op[i] = two_fermion(dipole[i], basis_n, basis_i)
    # and then transform it to the eigenstate basis
    for i in range(3):
        dipole_op[i] = cb_op2(dipole_op[i], evec_n, evec_i)
 
    # write it to file, will be used by XAS and RIXS calculations
    write_tensor(dipole_op, "trans_mat.dat")

    print("edrixs >>> Done!")
    print("edrixs >>> ED Done !")
