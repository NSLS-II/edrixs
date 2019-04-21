#!/usr/bin/env python

import numpy as np
from sympy.physics.wigner import clebsch_gordan
from .basis_transform import tmat_c2r, tmat_r2c, tmat_c2j, cb_op2
from .utils import case_to_shell_name, info_atomic_shell


def dipole_trans_oper(l1, l2):
    from sympy import N

    n1, n2 = 2 * l1 + 1, 2 * l2 + 1
    op = np.zeros((3, n1, n2), dtype=np.complex128)
    for i1, m1 in enumerate(range(-l1, l1 + 1)):
        for i2, m2 in enumerate(range(-l2, l2 + 1)):
            tmp1 = clebsch_gordan(l2, 1, l1, m2, -1, m1)
            tmp2 = clebsch_gordan(l2, 1, l1, m2, 1, m1)
            tmp3 = clebsch_gordan(l2, 1, l1, m2, 0, m1)
            tmp1, tmp2, tmp3 = N(tmp1), N(tmp2), N(tmp3)
            op[0, i1, i2] = (tmp1 - tmp2) * np.sqrt(2.0) / 2.0
            op[1, i1, i2] = (tmp1 + tmp2) * 1j * np.sqrt(2.0) / 2.0
            op[2, i1, i2] = tmp3
    op_spin = np.zeros((3, 2 * n1, 2 * n2), dtype=np.complex128)
    for i in range(3):
        op_spin[i, 0:2 * n1:2, 0:2 * n2:2] = op[i]
        op_spin[i, 1:2 * n1:2, 1:2 * n2:2] = op[i]

    return op_spin


def quadrupole_trans_oper(l1, l2):
    from sympy import N
    n1, n2 = 2 * l1 + 1, 2 * l2 + 1
    op = np.zeros((5, n1, n2), dtype=np.complex128)
    for i1, m1 in enumerate(range(-l1, l1 + 1)):
        for i2, m2 in enumerate(range(-l2, l2 + 1)):
            t1 = clebsch_gordan(l2, 2, l1, m2, -2, m1)
            t2 = clebsch_gordan(l2, 2, l1, m2, 2, m1)
            t3 = clebsch_gordan(l2, 2, l1, m2, 0, m1)
            t4 = clebsch_gordan(l2, 2, l1, m2, -1, m1)
            t5 = clebsch_gordan(l2, 2, l1, m2, 1, m1)
            t1, t2, t3, t4, t5 = N(t1), N(t2), N(t3), N(t4), N(t5)

            op[0, i1, i2] = t3
            op[1, i1, i2] = (t4 - t5) / np.sqrt(2.0)
            op[2, i1, i2] = (t4 + t5) * 1j / np.sqrt(2.0)
            op[3, i1, i2] = (t1 + t2) / np.sqrt(2.0)
            op[4, i1, i2] = (t1 - t2) * 1j / np.sqrt(2.0)

    op_spin = np.zeros((5, 2 * n1, 2 * n2), dtype=np.complex128)
    for i in range(5):
        op_spin[i, 0:2 * n1:2, 0:2 * n2:2] = op[i]
        op_spin[i, 1:2 * n1:2, 1:2 * n2:2] = op[i]

    return op_spin


def get_trans_oper(case):
    """
    Get the matrix of transition operators between two atomic shell in the complex
    spherical harmonics basis.

    Parameters
    ----------
    case : string
        A string indicating the two atomic shells, possible transitions are:

        -   'ss':     :math:`s \\rightarrow s`
        -   'ps':     :math:`s \\rightarrow p`
        -   't2gs':   :math:`s \\rightarrow t2g`
        -   'ds':     :math:`s \\rightarrow d`
        -   'fs':     :math:`s \\rightarrow f`
        -   'sp':     :math:`p \\rightarrow s`
        -   'sp12':   :math:`p_{1/2} \\rightarrow s`
        -   'sp32':   :math:`p_{3/2} \\rightarrow s`
        -   'pp':     :math:`p \\rightarrow p`
        -   'pp12':   :math:`p_{1/2} \\rightarrow p`
        -   'pp32':   :math:`p_{3/2} \\rightarrow p`
        -   't2gp':   :math:`p \\rightarrow t2g`
        -   't2gp12': :math:`p_{1/2} \\rightarrow t2g`
        -   't2gp32': :math:`p_{3/2} \\rightarrow t2g`
        -   'dp':     :math:`p \\rightarrow d`
        -   'dp12':   :math:`p_{1/2} \\rightarrow d`
        -   'dp32':   :math:`p_{3/2} \\rightarrow d`
        -   'fp':     :math:`p \\rightarrow f`
        -   'fp12':   :math:`p_{1/2} \\rightarrow f`
        -   'fp32':   :math:`p_{3/2} \\rightarrow f`
        -   'sd':     :math:`d \\rightarrow s`
        -   'sd32':   :math:`d_{3/2} \\rightarrow s`
        -   'sd52':   :math:`d_{5/2} \\rightarrow s`
        -   'pd':     :math:`d \\rightarrow p`
        -   'pd32':   :math:`d_{3/2} \\rightarrow p`
        -   'pd52':   :math:`d_{5/2} \\rightarrow p`
        -   't2gd':   :math:`d \\rightarrow t2g`
        -   't2gd32': :math:`d_{3/2} \\rightarrow t2g`
        -   't2gd52': :math:`d_{5/2} \\rightarrow t2g`
        -   'dd':     :math:`d \\rightarrow d`
        -   'dd32':   :math:`d_{3/2} \\rightarrow d`
        -   'dd52':   :math:`d_{5/2} \\rightarrow d`
        -   'fd':     :math:`d \\rightarrow f`
        -   'fd32':   :math:`d_{3/2} \\rightarrow f`
        -   'fd52':   :math:`d_{5/2} \\rightarrow f`
        -   'sf':     :math:`f \\rightarrow s`
        -   'sf52':   :math:`f_{5/2} \\rightarrow s`
        -   'sf72':   :math:`f_{7/2} \\rightarrow s`
        -   'pf':     :math:`f \\rightarrow p`
        -   'pf52':   :math:`f_{5/2} \\rightarrow p`
        -   'pf72':   :math:`f_{7/2} \\rightarrow p`
        -   't2gf':   :math:`f \\rightarrow t2g`
        -   't2gf52': :math:`f_{5/2} \\rightarrow t2g`
        -   't2gf72': :math:`f_{7/2} \\rightarrow t2g`
        -   'df':     :math:`f \\rightarrow d`
        -   'df52':   :math:`f_{5/2} \\rightarrow d`
        -   'df72':   :math:`f_{7/2} \\rightarrow d`
        -   'ff':     :math:`f \\rightarrow f`
        -   'ff52':   :math:`f_{5/2} \\rightarrow f`
        -   'ff72':   :math:`f_{7/2} \\rightarrow f`

    Returns
    -------
    res : 2d complex array
        The calculated transition matrix.
    """
    info = info_atomic_shell()
    v_name, c_name = case_to_shell_name(case.strip())
    v_orbl, c_orbl = info[v_name][0], info[c_name][0]
    v_norb, c_norb = 2 * (2 * v_orbl + 1), 2 * (2 * c_orbl + 1)
    if (v_orbl + c_orbl) % 2 == 0:
        op = quadrupole_trans_oper(v_orbl, c_orbl)
    else:
        op = dipole_trans_oper(v_orbl, c_orbl)

    # truncate to a sub-shell if necessary
    shell_list2 = ['p12', 'p32', 'd32', 'd52', 'f52', 'f72']
    left_tmat = np.eye(v_norb, dtype=np.complex)
    right_tmat = np.eye(c_norb, dtype=np.complex)
    indx1 = list(range(0, v_norb))
    indx2 = list(range(0, c_norb))
    if v_name == 't2g':
        left_tmat[0:v_norb, 0:v_norb] = tmat_c2r('d', True)
        indx1 = [2, 3, 4, 5, 8, 9]
    if c_name in shell_list2:
        right_tmat[0:c_norb, 0:c_norb] = tmat_c2j(c_orbl)
        if c_name == 'p12':
            indx2 = [0, 1]
        if c_name == 'p32':
            indx2 = [2, 3, 4, 5]
        if c_name == 'd32':
            indx2 = [0, 1, 2, 3]
        if c_name == 'd52':
            indx2 = [4, 5, 6, 7, 8, 9]
        if c_name == 'f52':
            indx2 = [0, 1, 2, 3, 4, 5]
        if c_name == 'f72':
            indx2 = [6, 7, 8, 9, 10, 11, 12, 13]

    if (v_orbl + c_orbl) % 2 == 0:
        npol = 5
    else:
        npol = 3

    op_tmp = np.zeros((npol, len(indx1), len(indx2)), dtype=np.complex)
    for i in range(npol):
        op[i] = cb_op2(op[i], left_tmat, right_tmat)
        op_tmp[i] = op[i, indx1][:, indx2]
        if v_name == 't2g':
            op_tmp[i] = np.dot(np.conj(np.transpose(tmat_r2c('t2g', True))), op_tmp[i])
    res = op_tmp
    return res


def unit_wavevector(theta, phi, local_axis=np.eye(3), direction='in'):
    if direction.strip() == 'in':
        unit_k = np.array([-np.cos(theta) * np.cos(phi),
                           -np.cos(theta) * np.sin(phi),
                           -np.sin(theta)])
        unit_k = np.dot(local_axis, unit_k)
    elif direction.strip() == 'out':
        unit_k = np.array([-np.cos(theta) * np.cos(phi),
                           -np.cos(theta) * np.sin(phi),
                           np.sin(theta)])
        unit_k = np.dot(local_axis, unit_k)
    else:
        raise Exception("Unknown direction in unit_wavevector: ", direction)

    return unit_k


def wavevector_with_length(theta, phi, energy, local_axis=np.eye(3), direction='in'):
    hbarc = 1.973270533 * 1000  # eV*A
    k_len = energy / hbarc
    return k_len * unit_wavevector(theta, phi, local_axis, direction)


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

    K_in_global = wavevector_with_length(thin, phi, ein, local_axis, direction='in')
    K_out_global = wavevector_with_length(thout, phi, eout, local_axis, direction='out')

    return K_in_global, K_out_global


def linear_polvec(theta, phi, alpha, local_axis=np.eye(3), case='in'):
    if case.strip() == 'in':
        polvec = (np.array([-np.cos(phi) * np.cos(np.pi / 2.0 - theta),
                            -np.sin(phi) * np.cos(np.pi / 2.0 - theta),
                            np.sin(np.pi / 2.0 - theta)]) * np.cos(alpha)
                  + np.array([-np.sin(phi), np.cos(phi), 0]) * np.sin(alpha))
        polvec = np.dot(local_axis, polvec)
    elif case.strip() == 'out':
        polvec = (np.array([np.cos(phi) * np.cos(np.pi / 2.0 - theta),
                            np.sin(phi) * np.cos(np.pi / 2.0 - theta),
                            np.sin(np.pi / 2.0 - theta)]) * np.cos(alpha)
                  + np.array([-np.sin(phi), np.cos(phi), 0]) * np.sin(alpha))
        polvec = np.dot(local_axis, polvec)
    else:
        raise Exception("Unknown case in linear_polvec: ", case)

    return polvec


def dipole_polvec_rixs(thin, thout, phi=0, alpha=0, beta=0, local_axis=np.eye(3),
                       case=('linear', 'linear')):
    """
    Return polarization vector of incident and scattered photons, for RIXS calculation.

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

    case : tuple of two strings
        Specify types of polarization for incident and scattered photons.
        case[0] for incident photon, case[1] for scattered photon.
        'linear': Linear polarization
        'left'  : Left-circular polarization.
        'right' : Right-circular polarization.

    Returns
    -------
    ei_in_global : 3-length complex array
        The linear polarization vector of the incident photon,
        with respect to the global :math:`xyz` -axis.

    ef_out_global : 3-length complex array
        The linear polarization vector of the scattered photon
        with respect to the global :math:`xyz` -axis.
    """

    ex = linear_polvec(thin, phi, 0, local_axis, case='in')
    ey = linear_polvec(thin, phi, np.pi/2.0, local_axis, case='in')
    if case[0].strip() == 'linear':
        ei_global = linear_polvec(thin, phi, alpha, local_axis, case='in')
    elif case[0].strip() == 'left':
        ei_global = (ex + 1j * ey) / np.sqrt(2.0)
    elif case[0].strip() == 'right':
        ei_global = (ex - 1j * ey) / np.sqrt(2.0)
    else:
        raise Exception("Unknown polarization case for incident photon: ", case[0])

    ex = linear_polvec(thout, phi, 0, local_axis, case='out')
    ey = linear_polvec(thout, phi, np.pi/2.0, local_axis, case='out')
    if case[1].strip() == 'linear':
        ef_global = linear_polvec(thout, phi, beta, local_axis, case='out')
    elif case[1].strip() == 'left':
        ef_global = (ex + 1j * ey) / np.sqrt(2.0)
    elif case[1].strip() == 'right':
        ef_global = (ex - 1j * ey) / np.sqrt(2.0)
    else:
        raise Exception("Unknown polarization case for scattered photon: ", case[1])

    return ei_global, ef_global


def dipole_polvec_xas(thin, phi=0, alpha=0, local_axis=np.eye(3), case='linear'):
    """
    Return the linear polarization vector of incident photons, for XAS calculation.

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

    case : string
        'linear': Linear polarization.
        'left'  : Left-circular polarization.
        'right' : Right-circular polarization.

    Returns
    -------
    ei_global : 3-length float array
        The linear polarization vector of the incident photon, with resepct to the
        global :math:`xyz` -axis.
    """

    ex = linear_polvec(thin, phi, 0, local_axis, case='in')
    ey = linear_polvec(thin, phi, np.pi/2.0, local_axis, case='in')
    if case.strip() == 'linear':
        ei_global = linear_polvec(thin, phi, alpha, local_axis, case='in')
    elif case.strip() == 'left':
        ei_global = (ex + 1j * ey) / np.sqrt(2.0)
    elif case.strip() == 'right':
        ei_global = (ex - 1j * ey) / np.sqrt(2.0)
    else:
        raise Exception("Unknown polarization case for incident photon: ", case)

    return ei_global


def quadrupole_polvec(polvec, wavevec):
    """
    Given dipolar polarization vector and wave-vector, return quadrupolar polarization vector.

    Parameters
    ----------
    polvec : 3 elements of complex array
        Dipolar polarization vector of photon, :math:`\\epsilon_{x}, \\epsilon_{y}, \\epsilon_{z}`,
        NOTE: they can be complex when the polarization is circular.

    wavevec : 3 elements of float array
        Wavevector of photon, :math:`k_{x}, k_{y}, k_{z}`.

    Returns
    -------
    quad_vec : 5 elements of float array
        Quadrupolar polarization vector.
    """

    quad_vec = np.zeros(5, dtype=np.complex)
    kvec = wavevec / np.sqrt(np.dot(wavevec, wavevec))

    quad_vec[0] = 0.5 * (2 * polvec[2] * kvec[2] - polvec[0] * kvec[0] - polvec[1] * kvec[1])
    quad_vec[1] = np.sqrt(3.0)/2.0 * (polvec[2] * kvec[0] + kvec[0] * kvec[2])
    quad_vec[2] = np.sqrt(3.0)/2.0 * (polvec[1] * kvec[2] + kvec[2] * kvec[1])
    quad_vec[3] = np.sqrt(3.0)/2.0 * (polvec[0] * kvec[0] - polvec[1] * kvec[1])
    quad_vec[4] = np.sqrt(3.0)/2.0 * (polvec[0] * kvec[1] + kvec[1] * kvec[0])

    return quad_vec
