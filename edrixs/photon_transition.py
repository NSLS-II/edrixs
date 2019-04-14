#!/usr/bin/env python

import numpy as np
from .basis_transform      import tmat_c2r, tmat_r2c
from sympy.physics.wigner  import clebsch_gordan 

def dipole_trans_oper(l1, l2):
    from sympy import N

    n1, n2 = 2*l1+1, 2*l2+1
    op=np.zeros((3,n1,n2), dtype=np.complex128)
    for i1, m1 in enumerate(range(-l1,l1+1)): 
        for i2, m2 in enumerate(range(-l2,l2+1)):
            tmp1 = clebsch_gordan(l2, 1, l1, m2, -1, m1)
            tmp2 = clebsch_gordan(l2, 1, l1, m2,  1, m1)
            tmp3 = clebsch_gordan(l2, 1, l1, m2,  0, m1)
            tmp1, tmp2, tmp3 = N(tmp1), N(tmp2), N(tmp3)
            op[0, i1, i2] = (tmp1 - tmp2) * np.sqrt(2.0)/2.0
            op[1, i1, i2] = (tmp1 + tmp2) * 1j * np.sqrt(2.0)/2.0
            op[2, i1, i2] = tmp3
    op_spin = np.zeros((3, 2*n1, 2*n2), dtype=np.complex128)
    for i in range(3):
        op_spin[i,0:2*n1:2, 0:2*n2:2] = op[i]
        op_spin[i,1:2*n1:2, 1:2*n2:2] = op[i]

    return op_spin
    
def quadrupole_trans_oper(l1, l2):
    from sympy import N
    n1, n2 = 2*l1+1, 2*l2+1
    op=np.zeros((5,n1,n2), dtype=np.complex128)
    for i1, m1 in enumerate(range(-l1,l1+1)): 
        for i2, m2 in enumerate(range(-l2,l2+1)):
            t1 = clebsch_gordan(l2, 2, l1, m2, -2, m1)
            t2 = clebsch_gordan(l2, 2, l1, m2,  2, m1)
            t3 = clebsch_gordan(l2, 2, l1, m2,  0, m1)
            t4 = clebsch_gordan(l2, 2, l1, m2, -1, m1)
            t5 = clebsch_gordan(l2, 2, l1, m2,  1, m1)
            t1, t2, t3, t4, t5 = N(t1), N(t2), N(t3), N(t4), N(t5)

            op[0, i1, i2] =  (t1+t2)/np.sqrt(2.0)
            op[1, i1, i2] =  t3
            op[2, i1, i2] =  (t4+t5) * 1j/np.sqrt(2.0)
            op[3, i1, i2] =  (t4-t5) / np.sqrt(2.0)
            op[4, i1, i2] =  (t1-t2) * 1j/np.sqrt(2.0)

    op_spin = np.zeros((5, 2*n1, 2*n2), dtype=np.complex128)
    for i in range(5):
        op_spin[i,0:2*n1:2, 0:2*n2:2] = op[i]
        op_spin[i,1:2*n1:2, 1:2*n2:2] = op[i]

    return op_spin

def trans_oper_ps():
    return dipole_trans_oper(1,0)

def trans_oper_pp():
    return quadrupole_trans_oper(1,1)

def trans_oper_pd():
    return dipole_trans_oper(1,2)

def trans_oper_pf():
    return quadrupole_trans_oper(1,3)

def trans_oper_ds():
    return quadrupole_trans_oper(2, 0)

def trans_oper_dp():
    return dipole_trans_oper(2, 1)

def trans_oper_dd():
    return quadrupole_trans_oper(2, 2)

def trans_oper_df():
    return dipole_trans_oper(2, 3)

def trans_oper_t2gs():
    op = quadrupole_trans_oper(2, 0)
    op_t2g = np.zeros((5, 6, 2), dtype=np.complex128)
    indx=[2,3,4,5,8,9]
    for i in range(5):
        op[i] = np.dot(np.conj(np.transpose(tmat_c2r('d', True))), op[i])
        op_t2g[i] = op[i,indx] 
        op_t2g[i] = np.dot(np.conj(np.transpose(tmat_r2c('t2g', True))), op_t2g[i])

    return op_t2g

def trans_oper_t2gp():
    op = dipole_trans_oper(2, 1)
    op_t2g = np.zeros((3, 6, 6), dtype=np.complex128)
    indx=[2,3,4,5,8,9]
    for i in range(3):
        op[i] = np.dot(np.conj(np.transpose(tmat_c2r('d', True))), op[i])
        op_t2g[i] = op[i,indx] 
        op_t2g[i] = np.dot(np.conj(np.transpose(tmat_r2c('t2g', True))), op_t2g[i])

    return op_t2g

def trans_oper_t2gd():
    op = quadrupole_trans_oper(2, 2)
    op_t2g = np.zeros((5, 6, 10), dtype=np.complex128)
    indx=[2,3,4,5,8,9]
    for i in range(5):
        op[i] = np.dot(np.conj(np.transpose(tmat_c2r('d', True))), op[i])
        op_t2g[i] = op[i,indx] 
        op_t2g[i] = np.dot(np.conj(np.transpose(tmat_r2c('t2g', True))), op_t2g[i])

    return op_t2g

def trans_oper_t2gf():
    op = dipole_trans_oper(2, 3)
    op_t2g = np.zeros((3, 6, 14), dtype=np.complex128)
    indx=[2,3,4,5,8,9]
    for i in range(3):
        op[i] = np.dot(np.conj(np.transpose(tmat_c2r('d', True))), op[i])
        op_t2g[i] = op[i,indx] 
        op_t2g[i] = np.dot(np.conj(np.transpose(tmat_r2c('t2g', True))), op_t2g[i])

    return op_t2g

def trans_oper_fs():
    return dipole_trans_oper(3,0)

def trans_oper_fp():
    return quadrupole_trans_oper(3,1)

def trans_oper_fd():
    return dipole_trans_oper(3,2)

def trans_oper_ff():
    return quadrupole_trans_oper(3,3)

transop_func_dict={
                  'ps': trans_oper_ps,  
                  'pp': trans_oper_pp,  
                  'pd': trans_oper_pd,  
                  'pf': trans_oper_pf,  
                't2gs': trans_oper_t2gs,  
                't2gp': trans_oper_t2gp,  
                't2gd': trans_oper_t2gd,  
                't2gf': trans_oper_t2gf,  
                  'ds': trans_oper_ds,  
                  'dp': trans_oper_dp,  
                  'dd': trans_oper_dd,  
                  'df': trans_oper_df,  
                  'fs': trans_oper_fs,  
                  'fp': trans_oper_fp,  
                  'fd': trans_oper_fd,  
                  'ff': trans_oper_ff  
                  }

def get_trans_oper(case):
    """
    Get the matrix of transition operators between two atomic shell in the complex 
    spherical harmonics basis.

    Parameters
    ----------
    case : str
        A string indicating the two atomic shells, possible options are:
 
        -   'ps':   :math:`s \\rightarrow p` transition
        -   'pp':   :math:`p \\rightarrow p` transition
        -   'pd':   :math:`d \\rightarrow p` transition
        -   'pf':   :math:`f \\rightarrow p` transition
        -   't2gs':   :math:`s \\rightarrow t_{2g}` transition
        -   't2gp':   :math:`p \\rightarrow t_{2g}` transition
        -   't2gd':   :math:`d \\rightarrow t_{2g}` transition
        -   't2gf':   :math:`f \\rightarrow t_{2g}` transition
        -   'ds':   :math:`s \\rightarrow d` transition
        -   'dp':   :math:`p \\rightarrow d` transition
        -   'dd':   :math:`d \\rightarrow d` transition
        -   'df':   :math:`f \\rightarrow d` transition
        -   'fs':   :math:`s \\rightarrow f` transition
        -   'fp':   :math:`p \\rightarrow f` transition
        -   'fd':   :math:`d \\rightarrow f` transition
        -   'ff':   :math:`f \\rightarrow f` transition
 
    Returns
    -------
    res : 2d complex array
        The calculated transition matrix.
    """

    res = transop_func_dict[case.strip()]()
    return res

def get_wavevector_rixs(thin, thout, phi, ein, eout, local_axis=np.eye(3)):
    """
    Return the wave vector of incident and scattered photons, for RIXS calculation.

    Parameters
    ----------
    thin : float
        The incident angle in radian.

    thout : float
        The scattered angle in radian.

    phi : float
        The azimuthal angle in radian.

    ein : float
        Energy of the incident photon (eV). 

    eout : float
        Energy of the scattered photon (eV). 

    local_axis : :math:`3 \\times 3` float array
        The local :math:`z` -axis, the angle thin and thout are defined with respect to this axis.

    Returns
    -------
    K_in_global : 3-length float array
        The wave vector of the incident photon, with respect to the global :math:`xyz` -axis.

    K_out_global : 3-length float array
        The wave vector of the scattered photon, with respect to the global :math:`xyz` -axis.
    """

    hbarc=1.973270533*1000 # eV*A
    kin_len =ein/hbarc
    kout_len=eout/hbarc
    K_in  = kin_len  * np.array([-np.cos(thin)*np.cos(phi),  
                                 -np.cos(thin)*np.sin(phi),  
                                 -np.sin(thin)])
    K_out = kout_len * np.array([-np.cos(thout)*np.cos(phi), 
                                 -np.cos(thout)*np.sin(phi), 
                                  np.sin(thout)]) 

    K_in_global = np.dot(local_axis, K_in) 
    K_out_global = np.dot(local_axis, K_out) 

    return K_in_global, K_out_global


def dipole_polvec_rixs(thin, thout, phi, alpha, beta, local_axis=np.eye(3)):
    """
    Return the polarization vector of incident and scattered photons, for RIXS calculation.

    Parameters
    ----------
    thin : float
        The incident angle  (radian).

    thout : float
        The scattered angle (radian).

    phi : float
        The azimuthal angle (radian).

    alpha : float
        The angle between the polarization vector of the incident photon and 
        the scattering plane (radian)

    beta : float
        The angle between the polarization vector of the scattered photon and 
        the scattering plane (radian)

    local_axis : :math:`3 \\times 3` float array
        The local :math:`z` -axis, the angle thin and thout are defined with 
        respect to this axis.

    Returns
    -------
    ei_in_global : 3-length float array
        The polarization vector of the incident photon, 
        with respect to the global :math:`xyz` -axis.

    ef_out_global : 3-length float array
        The polarization vector of the scattered photon 
        with respect to the global :math:`xyz` -axis.
    """

    ei = np.cos(alpha)*np.array([-np.cos(phi)*np.cos(np.pi/2.0-thin), 
                                 -np.sin(phi)*np.cos(np.pi/2.0-thin),  
         np.sin(np.pi/2.0-thin)]) + np.sin(alpha) * np.array([-np.sin(phi), np.cos(phi), 0])

    ef = np.cos(beta) *np.array([ np.cos(phi)*np.cos(np.pi/2.0-thout), 
                                  np.sin(phi)*np.cos(np.pi/2.0-thout), 
         np.sin(np.pi/2.0-thout)]) + np.sin(beta)  * np.array([-np.sin(phi), np.cos(phi), 0])

    ei_global = np.dot(local_axis, ei)
    ef_global = np.dot(local_axis, ef)

    return ei_global, ef_global

def dipole_polvec_xas(thin, phi, alpha, local_axis=np.eye(3)):
    """
    Return the polarization vector of incident photons, for XAS calculation.

    Parameters
    ----------
    thin : float
        The incident angle  (radian).

    phi : float
        The azimuthal angle (radian).

    alpha : float
        The angle between the polarization vector of the incident photon and 
        the scattering plane (radian)

    local_axis : :math:`3 \\times 3` float array
        The local :math:`z` -axis, the angle thin and thout are defined with 
        respect to this axis.

    Returns
    -------
    ei_in_global : 3-length float array
        The polarization vector of the incident photon, with resepct to the 
        global :math:`xyz` -axis.
    """

    ei = np.cos(alpha)*np.array([-np.cos(phi)*np.cos(np.pi/2.0-thin), 
                                 -np.sin(phi)*np.cos(np.pi/2.0-thin),  
         np.sin(np.pi/2.0-thin)]) + np.sin(alpha) * np.array([-np.sin(phi), np.cos(phi), 0])

    ei_global = np.dot(local_axis, ei)

    return ei_global
