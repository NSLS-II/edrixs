#!/usr/bin/env python

import sys
import numpy as np

from edrixs.soc                import   atom_hsoc
from edrixs.utils              import   UJ_to_UdJH, UdJH_to_F0F2F4
from edrixs.iostream           import   write_tensor, write_tensor_nonzeros
from edrixs.fock_basis         import   get_fock_bin_by_N
from edrixs.coulomb_utensor    import   get_umat_slater, get_F0
from edrixs.basis_transform    import   cb_op, cb_op2, tmat_c2r, transform_utensor
from edrixs.manybody_operator  import   two_fermion, four_fermion
from edrixs.photon_transition  import   get_trans_oper

if __name__ == "__main__":
    # local axis 
    loc2g=np.zeros((4,3,3),dtype=np.float64)
    loc2g[0] = [[-2.0/3.0,+1.0/3.0,-2.0/3.0],
                [+1.0/3.0,-2.0/3.0,-2.0/3.0],
                [-2.0/3.0,-2.0/3.0,+1.0/3.0]]

    loc2g[1] = [[+2.0/3.0,-1.0/3.0,-2.0/3.0],
                [-1.0/3.0,+2.0/3.0,-2.0/3.0],
                [+2.0/3.0,+2.0/3.0,+1.0/3.0]]

    loc2g[2] = [[+2.0/3.0,-2.0/3.0,-1.0/3.0],
                [+2.0/3.0,+1.0/3.0,+2.0/3.0],
                [-1.0/3.0,-2.0/3.0,+2.0/3.0]]

    loc2g[3] = [[-2.0/3.0,+2.0/3.0,-1.0/3.0],
                [-2.0/3.0,-1.0/3.0,+2.0/3.0],
                [+1.0/3.0,+2.0/3.0,+2.0/3.0]]

    U, J = 2.0, 0.3
    Ud, JH = UJ_to_UdJH(U,J)
    F0_d, F2_d, F4_d = UdJH_to_F0F2F4(Ud, JH)

    G1_dp, G3_dp = 0.957, 0.569
    F0_dp, F2_dp = get_F0('dp', G1_dp, G3_dp), 1.107

    zeta_d, zeta_p = 0.45, 1072.6666666666667

    emat_i = np.zeros((12,12), dtype=np.complex128)
    emat_n = np.zeros((12,12), dtype=np.complex128)

    umat_i = get_umat_slater('t2g', F0_d, F2_d, F4_d)
    umat_i = transform_utensor(umat_i, tmat_c2r('t2g', True))

    params=[
            F0_d, F2_d, F4_d,    # Fk for d
            F0_dp, F2_dp,        # Fk for dp
            G1_dp, G3_dp,        # Gk for dp
            0.0, 0.0             # Fk for p
           ]
    umat_n = get_umat_slater('t2gp', *params)
    tmat = np.zeros((12,12), dtype=np.complex128)
    tmat[0:6, 0:6] = tmat_c2r('t2g', True)
    tmat[6:12, 6:12] = tmat_c2r('p', True)
    umat_n = transform_utensor(umat_n, tmat)

    # SOC
    hsoc_d = cb_op(atom_hsoc('t2g', zeta_d), tmat_c2r('t2g', True))
    hsoc_p = cb_op(atom_hsoc('p', zeta_p), tmat_c2r('p', True))
    emat_i[0:6,0:6]   += hsoc_d
    emat_n[0:6,0:6]   += hsoc_d
    emat_n[6:12,6:12] += hsoc_p

    # Crystal field
    c=-0.15
    crys = np.array([[0, c, c],
                     [c, 0, c],
                     [c, c, 0]], dtype=np.complex128)
    emat_i[0:6:2,0:6:2] += crys
    emat_i[1:6:2,1:6:2] += crys
    emat_n[0:6:2,0:6:2] += crys
    emat_n[1:6:2,1:6:2] += crys

    basis_i = get_fock_bin_by_N(6, 5, 6, 6)
    basis_n = get_fock_bin_by_N(6, 6, 6, 5)
    ncfgs_i, ncfgs_n = len(basis_i), len(basis_n)

    print("edrixs >>> diagonalize Hamitlonian ...")
    hmat_i = np.zeros((ncfgs_i, ncfgs_i), dtype=np.complex128)
    hmat_n = np.zeros((ncfgs_n, ncfgs_n), dtype=np.complex128)
    hmat_i[:,:] += two_fermion(emat_i, basis_i, basis_i)
    hmat_i[:,:] += four_fermion(umat_i, basis_i)
    hmat_n[:,:] += two_fermion(emat_n, basis_n, basis_n)
    hmat_n[:,:] += four_fermion(umat_n, basis_n)

    eval_i, evec_i = np.linalg.eigh(hmat_i)
    eval_n, evec_n = np.linalg.eigh(hmat_n)

    write_tensor(eval_i, "eval_i.dat")
    write_tensor(eval_n, "eval_n.dat")
    print("edrixs >>> done !")
 
    # Transition operators
    print("edrixs >>> building dipolar transition operators...")
    dipole_op = np.zeros((3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    dipole_op_all = np.zeros((4, 3, ncfgs_n, ncfgs_i), dtype=np.complex128)
    dipole = np.zeros((3, 12, 12), dtype=np.complex128)
    tmp = get_trans_oper('t2gp')
    for i in range(3):
        tmp[i] = cb_op2(tmp[i], tmat_c2r('t2g',True), tmat_c2r('p',True))
        dipole[i,0:6, 6:12] = tmp[i]
        dipole_op[i] = two_fermion(dipole[i], basis_n, basis_i)
        dipole_op[i] = cb_op2(dipole_op[i], evec_n, evec_i)

    for i in range(4):
        for j in range(3):
            dipole_op_all[i,j] = loc2g[i,0,j]*dipole_op[0] + loc2g[i,1,j]*dipole_op[1] + loc2g[i,2,j]*dipole_op[2]

    write_tensor(dipole_op_all, "trans_mat.dat")
    print("edrixs >>> Done !")
     
