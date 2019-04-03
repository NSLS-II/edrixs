#!/usr/bin/env python

import sys
import numpy as np

from edrixs.coulomb_utensor    import get_umat_slater, get_F0
from edrixs.iostream           import write_tensor, write_tensor_nonzeros
from edrixs.fock_basis         import get_fock_bin_by_N
from edrixs.basis_transform    import cb_op, cb_op2, tmat_r2c, tmat_c2r
from edrixs.photon_transition  import get_trans_oper
from edrixs.manybody_operator  import two_fermion, four_fermion
from edrixs.soc                import atom_hsoc
from edrixs.utils              import UdJH_to_F0F2F4

if __name__ == "__main__":

    # PARAMETERS
    # -----------
    nt2g, nporb = 6, 6
    norbs = nt2g + nporb
 
    Ud, JH = 0, 0.355
    tmp, F2_d, F4_d = UdJH_to_F0F2F4(Ud, JH)
    Ud_av = 0.0 
    F0_d = Ud_av + get_F0('d', F2_d, F4_d)

    scale_dp = 0.9
    G1_dp, G3_dp = 0.894 * scale_dp,  0.531 * scale_dp
    F2_dp = 1.036 * scale_dp
    Udp_av = 0.0
    F0_dp = Udp_av + get_F0('dp', G1_dp, G3_dp)

    zeta_d_i = 0.33
    zeta_d_n = 0.33
    zeta_p_n = 1009.333333

    # END of PARAMETERS
    # -----------------

    params=[
            F0_d, F2_d, F4_d,    # Fk for d
            0.0,  0.0,           # Fk for dp
            0.0,  0.0,           # Gk for dp
            0.0,  0.0            # Fk for p
           ]
    # only keep t2g subspace, discard eg subspace
    umat_i = get_umat_slater('t2gp', *params)

    params=[
            F0_d,  F2_d, F4_d,   # Fk for d
            F0_dp, F2_dp,        # Fk for dp
            G1_dp, G3_dp,        # Gk for dp
            0.0,   0.0           # Fk for p
           ]
    # only keep t2g subspace, discard eg subspace
    umat_n = get_umat_slater('t2gp', *params)

    hsoc_i = np.zeros((norbs,norbs), dtype=np.complex128)
    hsoc_n = np.zeros((norbs,norbs), dtype=np.complex128)
    hsoc_i[0:nt2g, 0:nt2g] = atom_hsoc('t2g', zeta_d_i)
    hsoc_n[0:nt2g, 0:nt2g] = atom_hsoc('t2g', zeta_d_n)
    hsoc_n[nt2g:norbs, nt2g:norbs] = atom_hsoc('p', zeta_p_n)
    
    basis_i = get_fock_bin_by_N(nt2g, 3, nporb, nporb)
    ncfgs_i = len(basis_i)
    
    basis_n = get_fock_bin_by_N(nt2g, 4, nporb, nporb-1)
    ncfgs_n = len(basis_n)

    print("edrixs >>> building Hamiltonian without core-hole ...")
    hmat_i = np.zeros((ncfgs_i,ncfgs_i),dtype=np.complex128)
    hmat_n = np.zeros((ncfgs_n,ncfgs_n),dtype=np.complex128)
    hmat_i[:,:] += two_fermion(hsoc_i, basis_i, basis_i)
    hmat_i[:,:] += four_fermion(umat_i, basis_i)
    print("edrixs >>> Done!")

    print("edrixs >>> building Hamiltonian with a core-hole ...")
    hmat_n[:,:] += two_fermion(hsoc_n, basis_n, basis_n)
    hmat_n[:,:] += four_fermion(umat_n, basis_n)
    print("edrixs >>> Done!")

    eval_i, evec_i = np.linalg.eigh(hmat_i)
    eval_n, evec_n = np.linalg.eigh(hmat_n)

    write_tensor(eval_i, "eval_i.dat")
    write_tensor(eval_n, "eval_n.dat")
 
    print("edrixs >>> building dipolar transition operators...")
    dipole_op = np.zeros((3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    dipole = np.zeros((3, norbs, norbs), dtype=np.complex128)
    tmp = get_trans_oper('t2gp')
    for i in range(3):
        dipole[i,0:nt2g, nt2g:norbs] = tmp[i]
        dipole_op[i] = two_fermion(dipole[i], basis_n, basis_i)
    for i in range(3):
        dipole_op[i] = cb_op2(dipole_op[i], evec_n, evec_i)
    write_tensor(dipole_op, "trans_mat.dat")
    print("edrixs >>> Done !")
