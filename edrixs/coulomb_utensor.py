#!/usr/bin/env python
import numpy as np
from edrixs.basis_transform import tmat_c2r, tmat_r2c, transform_utensor
from sympy.physics.wigner import gaunt


def get_gaunt(l1, l2):
    """
    Calculate the Gaunt coefficents :math:`C_{l_1,l_2}(k,m_1,m_2)`

    .. math::

        C_{l_1,l_2}(k,m_1,m_2)=\\sqrt{\\frac{4\pi}{2k+1}} \\int \\mathop{d\\phi} \\mathop{d\\theta} sin(\\theta) Y_{l_1}^{m_1\\star}(\\theta,\\phi) 
                                Y_{k}^{m_1-m_2}(\\theta,\\phi) Y_{l_2}^{m_2}(\\theta,\\phi)

    Parameters
    ----------
    l1 : int
        The first quantum number of angular momentum.
    l2 : int 
        The second quantum number of angular momentum.

    Returns
    -------
    res: 3d float array
        The calculated Gaunt coefficents. 

        The 1st index (:math:`= 0, 1, ..., l_1+l_2+1`) is the order :math:`k`.

        The 2nd index (:math:`= 0, 1, ... ,2l_1`) is the magnetic quantum number :math:`m_1` plus :math:`l_1`

        The 3nd index (:math:`= 0, 1, ... ,2l_2`) is the magnetic quantum number :math:`m_2` plus :math:`l_2`

    Notes
    -----
    It should be noted that :math:`C_{l_1,l_2}(k,m_1,m_2)` is nonvanishing only when


    :math:`k + l_1 + l_2 = \\text{even}`,

    and

    :math:`|l_1 -  l_2| \\leq k \leq l_1 + l_2`. 

    Please see Ref. [1]_ p. 10 for more details.

    References
    ----------
    .. [1] Sugano S, Tanabe Y and Kamimura H. 1970. Multiplets of Transition-Metal Ions in Crystals. Academic Press, New York and London.

    Examples
    --------
    >>> from coulomb_utensor import get_gaunt
    
    Get gaunt coefficients between :math:`p`-shell and :math:`d`-shell

    >>> g = get_gaunt(1, 2)

    """

    from sympy import N
    res = np.zeros((l1+l2+1, 2*l1+1, 2*l2+1), dtype=np.float64) 
    for k in range(l1+l2+1):
        if not (np.mod(l1+l2+k,2) == 0 and np.abs(l1-l2) <= k <= l1+l2):
            continue 
        for i1, m1 in enumerate(range(-l1, l1+1)):
            for i2, m2 in enumerate(range(-l2, l2+1)):
                res[k,i1,i2] = N(gaunt(l1, k, l2, -m1, m1-m2, m2))*(-1.0)**m1*np.sqrt(4*np.pi/(2*k+1))
    return res

def umat_slater(l_list, fk):
    """
    Calculate the Coulomb interaction tensor which is parameterized by Slater integrals :math:`F^{k}`:

    .. math::

        U_{m_{l_i}m_{s_i}, m_{l_j}m_{s_j}, m_{l_t}m_{s_t}, m_{l_u}m_{s_u}}^{i,j,t,u} =\\frac{1}{2} \\delta_{m_{s_i},m_{s_t}}\\delta_{m_{s_j},m_{s_u}}
        \\delta_{m_{l_i}+m_{l_j}, m_{l_t}+m_{l_u}} \\sum_{k}C_{l_i,l_t}(k,m_{l_i},m_{l_t})C_{l_u,l_j}(k,m_{l_u},m_{l_j})F^{k}_{i,j,t,u}

    where :math:`m_s` is the magnetic quantum number for spin and :math:`m_l` is the magnetic quantum number for orbital. :math:`F^{k}_{i,j,t,u}`
    are Slater integrals. :math:`C_{l_i,l_j}(k,m_{l_i},m_{l_j})` are Gaunt coefficients.

    Parameters
    ----------
    l_list: list of int
        contains the quantum number of orbital angular momentum :math:`l` for each shell.
    fk: dict of float
        contains all the possible Slater integrals between the shells in l_list, the key is a tuple of 5 ints (:math:`k,i,j,t,u`), where :math:`k` is the order,
        :math:`i,j,t,u` are the shell indices begin with 1.

    Returns
    -------
    umat: 4d array of complex
        contains the Coulomb interaction tensor.

    See also
    --------
    coulomb_utensor.get_umat_slater
    coulomb_utensor.get_umat_kanamori
    coulomb_utensor.get_umat_kanamori_ge

    Examples
    --------
    >>> from coulomb_utensor import umat_slater

    For only one :math:`d`-shell

    >>> l_list = [2]
    >>> fk={}
    >>> F0, F2, F4 = 5.0, 4.0 2.0
    >>> fk[(0,1,1,1,1)] = F0
    >>> fk[(2,1,1,1,1)] = F2
    >>> fk[(4,1,1,1,1)] = F4
    >>> umat_d = umat_slater(l_list, fk) 

    For one :math:`d`-shell and one :math:`p`-shell

    >>> l_list = [2,1]
    >>> fk={}
    >>> F0_dd, F2_dd, F4_dd = 5.0, 4.0, 2.0
    >>> F0_dp, F2_dp = 4.0, 2.0
    >>> G1_dp, G3_dp = 2.0, 1.0
    >>> F0_pp, F2_pp = 2.0, 1.0

    >>> fk[(0,1,1,1,1)] = F0_dd
    >>> fk[(2,1,1,1,1)] = F2_dd
    >>> fk[(4,1,1,1,1)] = F4_dd

    >>> fk[(0,1,2,1,2)] = F0_dp
    >>> fk[(0,2,1,2,1)] = F0_dp
    >>> fk[(2,1,2,1,2)] = F2_dp
    >>> fk[(2,2,1,2,1)] = F2_dp

    >>> fk[(1,1,2,2,1)] = G1_dp
    >>> fk[(1,2,1,1,2)] = G1_dp
    >>> fk[(3,1,2,2,1)] = G3_dp
    >>> fk[(3,2,1,1,2)] = G3_dp

    >>> fk[(0,2,2,2,2)] = F0_pp
    >>> fk[(2,2,2,2,2)] = F2_pp

    """

    k_list=list(range(0, 2*max(l_list)+1))
    ck={}
    orb_label=[]
    # for each shell
    for i,l in enumerate(l_list):
        # magnetic quantum number
        for m in range(-l,l+1):
            # spin part, up,dn
            for spin in range(2):
                orb_label.append((i+1,l,m,spin))

    # all the possible gaunt coefficient
    for l1 in l_list:
        for l2 in l_list:
            ck_tmp = get_gaunt(l1,l2)
            for m1 in range(-l1,l1+1):
                for m2 in range(-l2,l2+1):
                    for k in k_list:
                        if np.mod(l1+l2+k,2) == 0 and np.abs(l1-l2) <= k <= l1+l2:
                            ck[(k,l1,m1,l2,m2)] = ck_tmp[k, m1+l1, m2+l2]
                        else:
                            ck[(k,l1,m1,l2,m2)] = 0

    # build the coulomb interaction tensor
    norbs = len(orb_label)
    umat = np.zeros((norbs,norbs,norbs,norbs),dtype=np.complex128)
    for orb1 in range(norbs):
        for orb2 in range(norbs):
            for orb3 in range(norbs): 
                for orb4 in range(norbs): 
                    i1,l1,m1,sigma1 = orb_label[orb1]
                    i2,l2,m2,sigma2 = orb_label[orb2]
                    i3,l3,m3,sigma3 = orb_label[orb3]
                    i4,l4,m4,sigma4 = orb_label[orb4]
                    if ( m1 + m2 ) != ( m3 + m4 ):
                        continue
                    if ( sigma1 != sigma3 ) or ( sigma2 != sigma4 ):
                        continue
                    res = 0.0
                    for k in k_list:
                        tmp_key = (k,i1,i2,i3,i4)
                        if tmp_key in list(fk.keys()):
                            res += ck[(k,l1,m1,l3,m3)] * ck[(k,l4,m4,l2,m2)] * fk[tmp_key]
                    umat[orb1,orb2,orb4,orb3] = res

    umat = umat / 2.0
    return umat

def get_umat_slater_s(F0):
    l_list=[0]
    fk={}
    fk[(0, 1, 1, 1, 1)] = F0
    umat = umat_slater(l_list, fk)

    return umat

def get_umat_slater_p(F0,F2):
    l_list=[1]
    fk={}
    fk[(0, 1, 1, 1, 1)] = F0
    fk[(2, 1, 1, 1, 1)] = F2
    umat = umat_slater(l_list, fk)

    return umat

def get_umat_slater_d(F0, F2, F4):
    l_list = [2]
    fk={}
    fk[(0,1,1,1,1)] = F0
    fk[(2,1,1,1,1)] = F2
    fk[(4,1,1,1,1)] = F4
    umat = umat_slater(l_list, fk)
    return umat

def get_umat_slater_t2g(F0, F2, F4):
    umat = get_umat_slater_d(F0, F2, F4)
    # c to r
    umat = transform_utensor(umat, tmat_c2r('d', True))
    t2g_indx = [2, 3, 4, 5, 8, 9]
    umat_t2g = np.zeros((6,6,6,6), dtype=np.complex128)
    umat_t2g[:,:,:,:] = umat[t2g_indx][:,t2g_indx][:,:,t2g_indx][:,:,:,t2g_indx]
    # t2g to c
    umat_t2g[:,:,:,:] = transform_utensor(umat_t2g, tmat_r2c('t2g', True))
    
    return umat_t2g

def get_umat_slater_f(F0, F2, F4, F6):
    l_list = [3]
    fk={}
    fk[(0,1,1,1,1)] = F0
    fk[(2,1,1,1,1)] = F2
    fk[(4,1,1,1,1)] = F4
    fk[(6,1,1,1,1)] = F6
    umat = umat_slater(l_list, fk)

    return umat

def get_umat_slater_ps(F0_pp, F2_pp, 
                       F0_ps, 
                       G1_ps, 
                       F0_ss):
    l_list = [1,0]
    fk={}
    fk[(0,1,1,1,1)] = F0_pp
    fk[(2,1,1,1,1)] = F2_pp

    fk[(0,1,2,1,2)] = F0_ps
    fk[(0,2,1,2,1)] = F0_ps

    fk[(1,1,2,2,1)] = G1_ps
    fk[(1,2,1,1,2)] = G1_ps

    fk[(0,2,2,2,2)] = F0_ss

    return umat_slater(l_list, fk)

def get_umat_slater_pp(F0_p1p1, F2_p1p1, 
                       F0_p1p2, F2_p1p2, 
                       G0_p1p2, G2_p1p2, 
                       F0_p2p2, F2_p2p2):
    l_list = [1,1]
    fk={}
    fk[(0,1,1,1,1)] = F0_p1p1
    fk[(2,1,1,1,1)] = F2_p1p1

    fk[(0,1,2,1,2)] = F0_p1p2
    fk[(0,2,1,2,1)] = F0_p1p2
    fk[(2,1,2,1,2)] = F2_p1p2
    fk[(2,2,1,2,1)] = F2_p1p2

    fk[(0,1,2,2,1)] = G0_p1p2
    fk[(0,2,1,1,2)] = G0_p1p2
    fk[(2,1,2,2,1)] = G2_p1p2
    fk[(2,2,1,1,2)] = G2_p1p2

    fk[(0,2,2,2,2)] = F0_p2p2
    fk[(2,2,2,2,2)] = F2_p2p2

    return umat_slater(l_list, fk)

def get_umat_slater_ds(F0_dd, F2_dd, F4_dd, 
                       F0_ds, 
                       G2_ds, 
                       F0_ss):
    l_list = [2,0]
    fk={}
    fk[(0,1,1,1,1)] = F0_dd
    fk[(2,1,1,1,1)] = F2_dd
    fk[(4,1,1,1,1)] = F4_dd

    fk[(0,1,2,1,2)] = F0_ds
    fk[(0,2,1,2,1)] = F0_ds

    fk[(2,1,2,2,1)] = G2_ds
    fk[(2,2,1,1,2)] = G2_ds

    fk[(0,2,2,2,2)] = F0_ss

    return umat_slater(l_list, fk)

def get_umat_slater_dp(F0_dd, F2_dd, F4_dd, 
                       F0_dp, F2_dp, 
                       G1_dp, G3_dp,
                       F0_pp, F2_pp):
    l_list = [2,1]
    fk={}
    fk[(0,1,1,1,1)] = F0_dd
    fk[(2,1,1,1,1)] = F2_dd
    fk[(4,1,1,1,1)] = F4_dd

    fk[(0,1,2,1,2)] = F0_dp
    fk[(0,2,1,2,1)] = F0_dp
    fk[(2,1,2,1,2)] = F2_dp
    fk[(2,2,1,2,1)] = F2_dp

    fk[(1,1,2,2,1)] = G1_dp
    fk[(1,2,1,1,2)] = G1_dp

    fk[(3,1,2,2,1)] = G3_dp
    fk[(3,2,1,1,2)] = G3_dp

    fk[(0,2,2,2,2)] = F0_pp
    fk[(2,2,2,2,2)] = F2_pp

    return umat_slater(l_list, fk)

def get_umat_slater_t2gp(F0_dd, F2_dd, F4_dd,
                         F0_dp, F2_dp,
                         G1_dp, G3_dp,
                         F0_pp, F2_pp):

    umat = get_umat_slater_dp(F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp, F0_pp, F2_pp)
    tmat = np.eye(16, dtype=np.complex128)
    tmat[0:10,0:10] = tmat_c2r('d', True)
    umat = transform_utensor(umat, tmat)
    umat_t2gp = np.zeros((12,12,12,12), dtype=np.complex128)
    indx = [2,3,4,5,8,9,10,11,12,13,14,15] 
    umat_t2gp[:,:,:,:] = umat[indx][:,indx][:,:,indx][:,:,:,indx]
    tmat = np.eye(12, dtype=np.complex128)
    tmat[0:6,0:6] = tmat_r2c('t2g', True)
    umat_t2gp[:,:,:,:] = transform_utensor(umat_t2gp, tmat)

    return umat_t2gp

def get_umat_slater_dd(F0_d1d1, F2_d1d1, F4_d1d1, 
                       F0_d1d2, F2_d1d2, F4_d1d2, 
                       G0_d1d2, G2_d1d2, G4_d1d2, 
                       F0_d2d2, F2_d2d2, F4_d2d2):
    l_list = [2,2]
    fk={}
    fk[(0,1,1,1,1)] = F0_d1d1
    fk[(2,1,1,1,1)] = F2_d1d1
    fk[(4,1,1,1,1)] = F4_d1d1

    fk[(0,1,2,1,2)] = F0_d1d2
    fk[(0,2,1,2,1)] = F0_d1d2
    fk[(2,1,2,1,2)] = F2_d1d2
    fk[(2,2,1,2,1)] = F2_d1d2
    fk[(4,1,2,1,2)] = F4_d1d2
    fk[(4,2,1,2,1)] = F4_d1d2

    fk[(0,1,2,2,1)] = G0_d1d2
    fk[(0,2,1,1,2)] = G0_d1d2
    fk[(2,1,2,2,1)] = G2_d1d2
    fk[(2,2,1,1,2)] = G2_d1d2
    fk[(4,1,2,2,1)] = G4_d1d2
    fk[(4,2,1,1,2)] = G4_d1d2

    fk[(0,2,2,2,2)] = F0_d2d2
    fk[(2,2,2,2,2)] = F2_d2d2
    fk[(4,2,2,2,2)] = F4_d2d2

    return umat_slater(l_list, fk)

def get_umat_slater_fs(F0_ff, F2_ff, F4_ff, F6_ff, 
                       F0_fs, 
                       G3_fs, 
                       F0_ss):
    l_list = [3,0]
    fk={}
    fk[(0,1,1,1,1)] = F0_ff 
    fk[(2,1,1,1,1)] = F2_ff
    fk[(4,1,1,1,1)] = F4_ff
    fk[(6,1,1,1,1)] = F4_ff

    fk[(0,1,2,1,2)] = F0_fs 
    fk[(0,2,1,2,1)] = F0_fs

    fk[(3,1,2,2,1)] = G3_fs
    fk[(3,2,1,1,2)] = G3_fs

    fk[(0,2,2,2,2)] = F0_ss

    return umat_slater(l_list, fk)

def get_umat_slater_fp(F0_ff, F2_ff, F4_ff, F6_ff, 
                       F0_fp, F2_fp, 
                       G2_fp, G4_fp, 
                       F0_pp, F2_pp):
    l_list = [3,1]
    fk={}
    fk[(0,1,1,1,1)] = F0_ff 
    fk[(2,1,1,1,1)] = F2_ff
    fk[(4,1,1,1,1)] = F4_ff
    fk[(6,1,1,1,1)] = F4_ff

    fk[(0,1,2,1,2)] = F0_fp 
    fk[(0,2,1,2,1)] = F0_fp
    fk[(2,1,2,1,2)] = F2_fp 
    fk[(2,2,1,2,1)] = F2_fp

    fk[(2,1,2,2,1)] = G2_fp
    fk[(2,2,1,1,2)] = G2_fp
    fk[(4,1,2,2,1)] = G4_fp
    fk[(4,2,1,1,2)] = G4_fp

    fk[(0,2,2,2,2)] = F0_pp
    fk[(2,2,2,2,2)] = F2_pp

    return umat_slater(l_list, fk)

def get_umat_slater_fd(F0_ff, F2_ff, F4_ff, F6_ff, 
                       F0_fd, F2_fd, F4_fd, 
                       G1_fd, G3_fd, G5_fd,
                       F0_dd, F2_dd, F4_dd):
    l_list = [3,2]
    fk={}
    fk[(0,1,1,1,1)] = F0_ff 
    fk[(2,1,1,1,1)] = F2_ff
    fk[(4,1,1,1,1)] = F4_ff
    fk[(6,1,1,1,1)] = F6_ff

    fk[(0,1,2,1,2)] = F0_fd 
    fk[(0,2,1,2,1)] = F0_fd

    fk[(2,1,2,1,2)] = F2_fd
    fk[(2,2,1,2,1)] = F2_fd

    fk[(4,1,2,1,2)] = F4_fd
    fk[(4,2,1,2,1)] = F4_fd

    fk[(1,1,2,2,1)] = G1_fd
    fk[(1,2,1,1,2)] = G1_fd

    fk[(3,1,2,2,1)] = G3_fd
    fk[(3,2,1,1,2)] = G3_fd

    fk[(5,1,2,2,1)] = G5_fd
    fk[(5,2,1,1,2)] = G5_fd

    fk[(0,2,2,2,2)] = F0_dd
    fk[(2,2,2,2,2)] = F2_dd
    fk[(4,2,2,2,2)] = F4_dd

    return umat_slater(l_list, fk)

def get_umat_slater_ff(F0_f1f1, F2_f1f1, F4_f1f1, F6_f1f1, 
                       F0_f1f2, F2_f1f2, F4_f1f2, F6_f1f2, 
                       G0_f1f2, G2_f1f2, G4_f1f2, G6_f1f2, 
                       F0_f2f2, F2_f2f2, F4_f2f2, F6_f2f2):
    l_list = [3,3]
    fk={}
    fk[(0,1,1,1,1)] = F0_f1f1 
    fk[(2,1,1,1,1)] = F2_f1f1
    fk[(4,1,1,1,1)] = F4_f1f1
    fk[(6,1,1,1,1)] = F6_f1f1

    fk[(0,1,2,1,2)] = F0_f1f2 
    fk[(0,2,1,2,1)] = F0_f1f2

    fk[(2,1,2,1,2)] = F2_f1f2 
    fk[(2,2,1,2,1)] = F2_f1f2

    fk[(4,1,2,1,2)] = F4_f1f2 
    fk[(4,2,1,2,1)] = F4_f1f2

    fk[(6,1,2,1,2)] = F6_f1f2 
    fk[(6,2,1,2,1)] = F6_f1f2

    fk[(0,1,2,2,1)] = G0_f1f2
    fk[(0,2,1,1,2)] = G0_f1f2

    fk[(2,1,2,2,1)] = G2_f1f2
    fk[(2,2,1,1,2)] = G2_f1f2

    fk[(4,1,2,2,1)] = G4_f1f2
    fk[(4,2,1,1,2)] = G4_f1f2

    fk[(6,1,2,2,1)] = G6_f1f2
    fk[(6,2,1,1,2)] = G6_f1f2

    fk[(0,2,2,2,2)] = F0_f2f2 
    fk[(2,2,2,2,2)] = F2_f2f2
    fk[(4,2,2,2,2)] = F4_f2f2
    fk[(6,2,2,2,2)] = F6_f2f2

    return umat_slater(l_list, fk)

def get_umat_kanamori_ge(norbs, U1, U2, J, Jx, Jp):
    """
    Calculate the Coulomb interaction tensor for a Kanamori-type interaction. For the general case, it is parameterized by :math:`U_1, U_2, J, J_x, J_p`.

    Parameters
    ----------
    norbs: int
        number of orbitals (including spin).
    U1: float
        Hubbard :math:`U` for electrons residing on the same orbital with opposite spin.
    U2: float
        Hubbard :math:`U` for electrons residing on different orbitals.
    J: float
        Hund's coupling for density-density interaction.
    Jx: float
        Hund's coupling for spin flip.
    Jp: float
        Hund's coupling for pair-hopping.

    Returns
    -------
    umat: 4d complex array
        The calculated Coulomb interaction tensor

    Notes
    -----
    The order of spin index is: up, down, up, down, ..., up, down.

    See Also
    --------
    coulomb_utensor.get_umat_kanamori 
    coulomb_utensor.get_umat_slater
    coulomb_utensor.umat_slater

    """

    umat=np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    for alpha in range(0,norbs-1):
        for beta in range(alpha+1, norbs):
            for gamma in range(0, norbs-1):
                for delta in range(gamma+1, norbs):
                    aband =  alpha / 2
                    aspin =  alpha % 2
                    bband =  beta  / 2
                    bspin =  beta  % 2
                    gband =  gamma / 2
                    gspin =  gamma % 2
                    dband =  delta / 2
                    dspin =  delta % 2

                    dtmp=0.0
                    if ( ( alpha == gamma ) and ( beta == delta ) ):
                        if ( ( aband == bband ) and ( aspin != bspin ) ):
                            dtmp = dtmp + U1

                    if ( ( alpha == gamma ) and ( beta == delta ) ):
                        if ( aband != bband ):
                            dtmp = dtmp + U2

                    if ( ( alpha == gamma ) and ( beta == delta ) ):
                        if ( ( aband != bband ) and ( aspin == bspin ) ):
                            dtmp = dtmp - J

                    if ( ( aband == gband ) and ( bband == dband ) ):
                        if ( ( aspin != gspin ) and ( bspin != dspin ) and ( aspin != bspin ) ):
                            dtmp = dtmp - Jx

                    if ( ( aband == bband ) and ( dband == gband ) and ( aband != dband ) ):
                        if ( ( aspin != bspin ) and ( dspin != gspin ) and ( aspin == gspin ) ):
                            dtmp = dtmp + Jp

                    umat[alpha,beta,delta,gamma] = dtmp

    return umat

def get_F0(case, *args):
    case=case.strip()
    if case == 's':
        return 0.0 

    elif case == 'p':
        F2 = args[0]
        return 2.0/25.0*F2

    elif case == 'd':
        F2, F4 = args[0:2]
        return 2.0/63.0*F2 + 2.0/63.0*F4

    elif case == 'f':
        F2, F4, F6 = args[0:3]
        return 4.0/195.0*F2 + 2.0/143.0*F4 + 100.0/5577.0*F6

    elif case == 'ss':
        G0=args[0]
        return 0.5*G0

    elif case == 'sp' or case == 'ps':
        G1 = args[0]
        return 1.0/6.0*G1

    elif case == 'sd' or case == 'ds':
        G2 = args[0]
        return 1.0/10.0*G2

    elif case == 'sf' or case == 'fs':
        G3 = args[0]
        return 1.0/14.0*G3

    elif case == 'pp':
        G0, G2 = args[0:2]
        return 1.0/6.0*G0 + 1.0/15.0*G2

    elif case == 'pd' or case == 'dp':
        G1, G3 = args[0:2]
        return 1.0/15.0*G1 + 3.0/70.0*G3

    elif case == 'pf' or case =='fp':
        G2, G4 = args[0:2]
        return 3.0/70.0*G2 + 2.0/63.0*G4

    elif case == 'dd':
        G0, G2, G4 = args[0:3]
        return 1.0/10.0*G0 + 1.0/35.0*G2 + 1.0/35.0*G4

    elif case == 'df' or case =='fd':
        G1, G3, G5 = args[0:3]
        return 3.0/70.0*G1 + 2.0/105.0*G3 + 5.0/231.0*G5

    elif case == 'ff':
        G0, G2, G4, G6= args[0:4]
        return 1.0/14.0*G0 + 2.0/105.0*G2 + 1.0/77.0*G4 + 50.0/3003.0*G6

    else:
        print("error in get_F0():  Unknown case name")

umat_func_dict={
                's':  get_umat_slater_s,
                'p':  get_umat_slater_p,
                'd':  get_umat_slater_d,
              't2g':  get_umat_slater_t2g,
                'f':  get_umat_slater_f,
               'ps':  get_umat_slater_ps,
               'pp':  get_umat_slater_pp,
               'ds':  get_umat_slater_ds,
               'dp':  get_umat_slater_dp,
             't2gp':  get_umat_slater_t2gp,
               'dd':  get_umat_slater_dd,
               'fs':  get_umat_slater_fs,
               'fp':  get_umat_slater_fp,
               'fd':  get_umat_slater_fd,
               'ff':  get_umat_slater_ff
                }
               
def get_umat_slater(case, *args):
    """
    Convenient adapter function to return the Coulomb interaction tensor for common case.

    Parameters
    ----------
    case: string
        indicates which atomic shells we will include, should be one of
        
        - 's':    a single :math:`s`-shell (:math:`l=0`)
        - 'p':    a single :math:`p`-shell (:math:`l=1`)
        - 'd':    a single :math:`d`-shell (:math:`l=2`)
        - 'f':    a single :math:`f`-shell (:math:`l=3`)
        - 'ps':   a :math:`p`-shell (:math:`l=1`) and another :math:`s`-shell (:math:`l=0`)
        - 'pp':   a :math:`p`-shell (:math:`l=1`) and another :math:`p`-shell (:math:`l=1`)
        - 'ds':   a :math:`d`-shell (:math:`l=2`) and another :math:`s`-shell (:math:`l=0`)
        - 'dp':   a :math:`d`-shell (:math:`l=2`) and another :math:`p`-shell (:math:`l=1`)
        - 't2gp': a :math:`t2g`-shell (:math:`l_{\\text{eff}}=1`) and another :math:`p`-shell (:math:`l=1`)
        - 'dd':   a :math:`d`-shell (:math:`l=2`) and another :math:`d`-shell (:math:`l=2`)
        - 'fs':   a :math:`f`-shell (:math:`l=3`) and another :math:`s`-shell (:math:`l=0`)
        - 'fp':   a :math:`f`-shell (:math:`l=3`) and another :math:`p`-shell (:math:`l=1`)
        - 'fd':   a :math:`f`-shell (:math:`l=3`) and another :math:`d`-shell (:math:`l=2`)
        - 'ff':   a :math:`f`-shell (:math:`l=3`) and another :math:`f`-shell (:math:`l=3`)

    args: floats
        Slater integrals, should be one of
 
        - for case 's':  F0_ss
        - for case 'p':  F0_pp, F2_pp
        - for case 'd':  F0_dd, F2_dd, F4_dd
        - for case 'f':  F0_ff, F2_ff, F4_ff, F6_ff
        - for case 'ps': F0_pp, F2_pp, F0_ps, G1_ps, F0_ss
        - for case 'pp': F0_p1p1, F2_p1p1, F0_p1p2, F2_p1p2, G0_p1p2, G2_p1p2, F0_p2p2, F2_p2p2
        - for case 'ds': F0_dd, F2_dd, F4_dd, F0_ds, G2_ds, F0_ss
        - for case 'dp': F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp, F0_pp, F2_pp
        - for case 't2gp': F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp, F0_pp, F2_pp
        - for case 'dd': F0_d1d1, F2_d1d1, F4_d1d1, F0_d1d2, F2_d1d2, F4_d1d2, G0_d1d2, G2_d1d2, G4_d1d2, F0_d2d2, F2_d2d2, F4_d2d2
        - for case 'fs':  F0_ff, F2_ff, F4_ff, F6_ff, F0_fs, G3_fs, F0_ss
        - for case 'fp':  F0_ff, F2_ff, F4_ff, F6_ff, F0_fp, F2_fp, G2_fp, G4_fp, F0_pp, F2_pp
        - for case 'fd':  F0_ff, F2_ff, F4_ff, F6_ff, F0_fd, F2_fd, F4_fd, G1_fd, G3_fd, G5_fd, F0_dd, F2_dd, F4_dd
        - for case 'ff':  F0_f1f1, F2_f1f1, F4_f1f1, F6_f1f1, F0_f1f2, F2_f1f2, F4_f1f2, F6_f1f2, G0_f1f2, G2_f1f2, G4_f1f2, G6_f1f2, F0_f2f2, F2_f2f2, F4_f2f2, F6_f2f2
                                                             
        The order of these arguments are important.          
                                                             
    Returns
    -------
    umat: 4d array of complex
        the Coulomb interaction tensor 

    See Also
    --------
    coulomb_utensor.umat_slater
    coulomb_utensor.get_umat_kanamori
    coulomb_utensor.get_umat_kanamori_ge

    """
    return umat_func_dict[case.strip()](*args)

def get_umat_kanamori(norbs, U, J):
    """
    Calculate the Coulomb interaction tensor for a Kanamori-type interaction. For the :math:`t2g`-shell case, it is parameterized by :math:`U, J`.

    Parameters
    ----------
    norbs: int
        number of orbitals (including spin).
    U: float
        Hubbard :math:`U` for electrons residing on the same orbital with opposite spin.
    J: float
        Hund's coupling.

    Returns
    -------
    umat: 4d complex array
        The calculated Coulomb interaction tensor

    Notes
    -----
    The order of spin index is: up, down, up, down, ..., up, down.

    See Also
    --------
    coulomb_utensor.get_umat_kanamori_ge
    coulomb_utensor.get_umat_slater
    coulomb_utensor.umat_slater

    """

    return get_umat_kanamori_ge(norbs, U, U-2*J, J, J, J)
