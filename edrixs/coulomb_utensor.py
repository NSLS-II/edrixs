#!/usr/bin/env python

import numpy as np
from sympy.physics.wigner import gaunt
from .basis_transform import tmat_c2r, tmat_r2c, tmat_c2j, transform_utensor
from .utils import info_atomic_shell, case_to_shell_name


def get_gaunt(l1, l2):
    """
    Calculate the Gaunt coefficents :math:`C_{l_1,l_2}(k,m_1,m_2)`

    .. math::

        C_{l_1,l_2}(k,m_1,m_2)=\\sqrt{\\frac{4\\pi}{2k+1}} \\int
        \\mathop{d\\phi} \\mathop{d\\theta} sin(\\theta)
        Y_{l_1}^{m_1\\star}(\\theta,\\phi) Y_{k}^{m_1-m_2}(\\theta,\\phi)
        Y_{l_2}^{m_2}(\\theta,\\phi)

    Parameters
    ----------
    l1: int
        The first quantum number of angular momentum.
    l2: int
        The second quantum number of angular momentum.

    Returns
    -------
    res: 3d float array
        The calculated Gaunt coefficents.

        The 1st index (:math:`= 0, 1, ..., l_1+l_2+1`) is the order :math:`k`.

        The 2nd index (:math:`= 0, 1, ... ,2l_1`) is the magnetic quantum
        number :math:`m_1` plus :math:`l_1`

        The 3nd index (:math:`= 0, 1, ... ,2l_2`) is the magnetic quantum
        number :math:`m_2` plus :math:`l_2`

    Notes
    -----
    It should be noted that :math:`C_{l_1,l_2}(k,m_1,m_2)` is
    nonvanishing only when

    :math:`k + l_1 + l_2 = \\text{even}`,

    and

    :math:`|l_1 -  l_2| \\leq k \\leq l_1 + l_2`.

    Please see Ref. [1]_ p. 10 for more details.

    References
    ----------
    .. [1] Sugano S, Tanabe Y and Kamimura H. 1970. Multiplets of
       Transition-Metal Ions in Crystals. Academic Press, New York and London.

    Examples
    --------
    >>> import edrixs

    Get gaunt coefficients between :math:`p`-shell and :math:`d`-shell

    >>> g = edrixs.get_gaunt(1, 2)

    """

    from sympy import N
    res = np.zeros((l1 + l2 + 1, 2 * l1 + 1, 2 * l2 + 1), dtype=np.float64)
    for k in range(l1 + l2 + 1):
        if not (np.mod(l1 + l2 + k, 2) == 0 and np.abs(l1 - l2) <= k <= l1 + l2):
            continue
        for i1, m1 in enumerate(range(-l1, l1 + 1)):
            for i2, m2 in enumerate(range(-l2, l2 + 1)):
                res[k, i1, i2] = (N(gaunt(l1, k, l2, -m1, m1 - m2, m2)) *
                                  (-1.0)**m1 * np.sqrt(4 * np.pi / (2 * k + 1)))
    return res


def umat_slater(l_list, fk):
    """
    Calculate the Coulomb interaction tensor which is parameterized by
    Slater integrals :math:`F^{k}`:

    .. math::

        U_{m_{l_i}m_{s_i}, m_{l_j}m_{s_j}, m_{l_t}m_{s_t},
        m_{l_u}m_{s_u}}^{i,j,t,u}
        =\\frac{1}{2} \\delta_{m_{s_i},m_{s_t}}\\delta_{m_{s_j},m_{s_u}}
        \\delta_{m_{l_i}+m_{l_j}, m_{l_t}+m_{l_u}}
        \\sum_{k}C_{l_i,l_t}(k,m_{l_i},m_{l_t})C_{l_u,l_j}
        (k,m_{l_u},m_{l_j})F^{k}_{i,j,t,u}

    where :math:`m_s` is the magnetic quantum number for spin
    and :math:`m_l` is the magnetic quantum number for orbital.
    :math:`F^{k}_{i,j,t,u}` are Slater integrals.
    :math:`C_{l_i,l_j}(k,m_{l_i},m_{l_j})` are Gaunt coefficients.

    Parameters
    ----------
    l_list: list of int
        contains the quantum number of orbital angular momentum
        :math:`l` for each shell.
    fk: dict of float
        contains all the possible Slater integrals between the shells
        in l_list, the key is a tuple of 5 ints (:math:`k,i,j,t,u`),
        where :math:`k` is the order,
        :math:`i,j,t,u` are the shell indices begin with 1.

    Returns
    -------
    umat: 4d array of complex
        contains the Coulomb interaction tensor.

    Examples
    --------
    >>> import edrixs

    For only one :math:`d`-shell

    >>> l_list = [2]
    >>> fk={}
    >>> F0, F2, F4 = 5.0, 4.0 2.0
    >>> fk[(0,1,1,1,1)] = F0
    >>> fk[(2,1,1,1,1)] = F2
    >>> fk[(4,1,1,1,1)] = F4
    >>> umat_d = edrixs.umat_slater(l_list, fk)

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

    >>> umat_dp = edrixs.umat_slater(l_list, fk)

    See also
    --------
    coulomb_utensor.get_umat_slater
    coulomb_utensor.get_umat_kanamori
    coulomb_utensor.get_umat_kanamori_ge
    """
    k_list = list(range(0, 2 * max(l_list) + 1))
    ck = {}
    orb_label = []
    # for each shell
    for i, l in enumerate(l_list):
        # magnetic quantum number
        for m in range(-l, l + 1):
            # spin part, up,dn
            for spin in range(2):
                orb_label.append((i + 1, l, m, spin))

    # all the possible gaunt coefficient
    for l1 in l_list:
        for l2 in l_list:
            ck_tmp = get_gaunt(l1, l2)
            for m1 in range(-l1, l1 + 1):
                for m2 in range(-l2, l2 + 1):
                    for k in k_list:
                        if (np.mod(l1 + l2 + k, 2) == 0 and np.abs(l1 - l2) <= k <= l1 + l2):
                            ck[(k, l1, m1, l2, m2)] = ck_tmp[k, m1 + l1, m2 + l2]
                        else:
                            ck[(k, l1, m1, l2, m2)] = 0

    # build the coulomb interaction tensor
    norbs = len(orb_label)
    umat = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    for orb1 in range(norbs):
        for orb2 in range(norbs):
            for orb3 in range(norbs):
                for orb4 in range(norbs):
                    i1, l1, m1, sigma1 = orb_label[orb1]
                    i2, l2, m2, sigma2 = orb_label[orb2]
                    i3, l3, m3, sigma3 = orb_label[orb3]
                    i4, l4, m4, sigma4 = orb_label[orb4]
                    if (m1 + m2) != (m3 + m4):
                        continue
                    if (sigma1 != sigma3) or (sigma2 != sigma4):
                        continue
                    res = 0.0
                    for k in k_list:
                        tmp_key = (k, i1, i2, i3, i4)
                        if tmp_key in list(fk.keys()):
                            res += ck[(k, l1, m1, l3, m3)] * ck[(k, l4, m4, l2, m2)] * fk[tmp_key]
                    umat[orb1, orb2, orb4, orb3] = res

    umat = umat / 2.0
    return umat


def get_umat_kanamori_ge(norbs, U1, U2, J, Jx, Jp):
    """
    Calculate the Coulomb interaction tensor for a Kanamori-type interaction.
    For the general case, it is parameterized by :math:`U_1, U_2, J, J_x, J_p`.

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

    umat = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    for alpha in range(0, norbs - 1):
        for beta in range(alpha + 1, norbs):
            for gamma in range(0, norbs - 1):
                for delta in range(gamma + 1, norbs):
                    aband = alpha / 2
                    aspin = alpha % 2
                    bband = beta / 2
                    bspin = beta % 2
                    gband = gamma / 2
                    gspin = gamma % 2
                    dband = delta / 2
                    dspin = delta % 2

                    dtmp = 0.0
                    if ((alpha == gamma) and (beta == delta)):
                        if ((aband == bband) and (aspin != bspin)):
                            dtmp = dtmp + U1

                    if ((alpha == gamma) and (beta == delta)):
                        if (aband != bband):
                            dtmp = dtmp + U2

                    if ((alpha == gamma) and (beta == delta)):
                        if ((aband != bband) and (aspin == bspin)):
                            dtmp = dtmp - J

                    if ((aband == gband) and (bband == dband)):
                        if ((aspin != gspin) and (bspin != dspin)
                                and (aspin != bspin)):
                            dtmp = dtmp - Jx

                    if ((aband == bband) and (dband == gband)
                            and (aband != dband)):
                        if ((aspin != bspin) and (dspin != gspin)
                                and (aspin == gspin)):
                            dtmp = dtmp + Jp

                    umat[alpha, beta, delta, gamma] = dtmp

    return umat


def get_F0(case, *args):
    case = case.strip()
    if case == 's':
        return 0.0

    elif case == 'p':
        F2 = args[0]
        return 2.0 / 25.0 * F2

    elif case == 'd':
        F2, F4 = args[0:2]
        return 2.0 / 63.0 * F2 + 2.0 / 63.0 * F4

    elif case == 'f':
        F2, F4, F6 = args[0:3]
        return 4.0 / 195.0 * F2 + 2.0 / 143.0 * F4 + 100.0 / 5577.0 * F6

    elif case == 'ss':
        G0 = args[0]
        return 0.5 * G0

    elif case == 'sp' or case == 'ps':
        G1 = args[0]
        return 1.0 / 6.0 * G1

    elif case == 'sd' or case == 'ds':
        G2 = args[0]
        return 1.0 / 10.0 * G2

    elif case == 'sf' or case == 'fs':
        G3 = args[0]
        return 1.0 / 14.0 * G3

    elif case == 'pp':
        G0, G2 = args[0:2]
        return 1.0 / 6.0 * G0 + 1.0 / 15.0 * G2

    elif case == 'pd' or case == 'dp':
        G1, G3 = args[0:2]
        return 1.0 / 15.0 * G1 + 3.0 / 70.0 * G3

    elif case == 'pf' or case == 'fp':
        G2, G4 = args[0:2]
        return 3.0 / 70.0 * G2 + 2.0 / 63.0 * G4

    elif case == 'dd':
        G0, G2, G4 = args[0:3]
        return 1.0 / 10.0 * G0 + 1.0 / 35.0 * G2 + 1.0 / 35.0 * G4

    elif case == 'df' or case == 'fd':
        G1, G3, G5 = args[0:3]
        return 3.0 / 70.0 * G1 + 2.0 / 105.0 * G3 + 5.0 / 231.0 * G5

    elif case == 'ff':
        G0, G2, G4, G6 = args[0:4]
        return 1.0 / 14.0 * G0 + 2.0 / 105.0 * G2 + 1.0 / 77.0 * G4 + 50.0 / 3003.0 * G6

    else:
        raise Exception("error in get_F0():  Unknown case name:", case)


def get_umat_slater(case, *args):
    """
    Convenient adapter function to return the Coulomb interaction tensor for common case.

    Parameters
    ----------
    case: string
        indicates which atomic shells we will include, should be one of

        - 's':    single :math:`s`-shell (:math:`l=0`)
        - 'p':    single :math:`p`-shell (:math:`l=1`)
        - 't2g':  single :math:`t2g`-shell (:math:`l_{\\text{eff}}=1`)
        - 'd':    single :math:`d`-shell (:math:`l=2`)
        - 'f':    single :math:`f`-shell (:math:`l=3`)
        - 'ss':   :math:`s`-shell (:math:`l=0`) plus :math:`s`-shell (:math:`l=0`)
        - 'ps':   :math:`p`-shell (:math:`l=1`) plus :math:`s`-shell (:math:`l=0`)
        - 't2gs': :math:`t2g`-shell (:math:`l_{\\text{eff}}=1`) plus :math:`s`-shell (:math:`l=0`)
        - 'ds':   :math:`d`-shell (:math:`l=2`) plus :math:`s`-shell (:math:`l=0`)
        - 'fs':   :math:`f`-shell (:math:`l=3`) plus :math:`s`-shell (:math:`l=0`)
        - 'sp':   :math:`s`-shell (:math:`l=0`) plus :math:`p`-shell (:math:`l=1`)
        - 'sp12': :math:`s`-shell (:math:`l=0`) plus :math:`p_{1/2}`-shell (:math:`j=1/2`)
        - 'sp32': :math:`s`-shell (:math:`l=0`) plus :math:`p_{3/2}`-shell (:math:`j=3/2`)
        - 'pp':   :math:`p`-shell (:math:`l=1`) plus :math:`p`-shell (:math:`l=1`)
        - 'pp12': :math:`p`-shell (:math:`l=1`) plus :math:`p_{1/2}`-shell (:math:`j=1/2`)
        - 'pp32': :math:`p`-shell (:math:`l=1`) plus :math:`p_{3/2}`-shell (:math:`j=3/2`)
        - 't2gp': :math:`t_{2g}`-shell (:math:`l_{\\text{eff}}=1`)
          plus :math:`p`-shell (:math:`l=1`)
        - 't2gp12': :math:`t_{2g}`-shell (:math:`l_{\\text{eff}}=1`)
          plus :math:`p_{1/2}`-shell (:math:`j=1/2`)
        - 't2gp32': :math:`t_{2g}`-shell (:math:`l_{\\text{eff}}=1`)
          plus :math:`p_{3/2}`-shell (:math:`j=3/2`)
        - 'dp':     :math:`d`-shell (:math:`l=2`) plus :math:`p`-shell (:math:`l=1`)
        - 'dp12':   :math:`d`-shell (:math:`l=2`) plus :math:`p_{1/2}`-shell (:math:`j=1/2`)
        - 'dp32':   :math:`d`-shell (:math:`l=2`) plus :math:`p_{3/2}`-shell (:math:`j=3/2`)
        - 'fp':     :math:`f`-shell (:math:`l=3`) plus :math:`p`-shell (:math:`l=1`)
        - 'fp12':   :math:`f`-shell (:math:`l=3`) plus :math:`p_{1/2}`-shell (:math:`j=1/2`)
        - 'fp32':   :math:`f`-shell (:math:`l=3`) plus :math:`p_{3/2}`-shell (:math:`j=3/2`)
        - 'sd':     :math:`s`-shell (:math:`l=0`) plus :math:`d`-shell (:math:`l=2`)
        - 'sd32':   :math:`s`-shell (:math:`l=0`) plus :math:`d_{3/2}`-shell (:math:`j=3/2`)
        - 'sd52':   :math:`s`-shell (:math:`l=0`) plus :math:`d_{5/2}`-shell (:math:`j=5/2`)
        - 'pd':     :math:`p`-shell (:math:`l=1`) plus :math:`d`-shell (:math:`l=2`)
        - 'pd32':   :math:`p`-shell (:math:`l=1`) plus :math:`d_{3/2}`-shell (:math:`j=3/2`)
        - 'pd52':   :math:`p`-shell (:math:`l=1`) plus :math:`d_{5/2}`-shell (:math:`j=5/2`)
        - 't2gd':   :math:`t_{2g}`-shell (:math:`l_{\\text{eff}}=1`)
          plus :math:`d`-shell (:math:`l=2`)
        - 't2gd32': :math:`t_{2g}`-shell (:math:`l_{\\text{eff}}=1`)
          plus :math:`d_{3/2}`-shell (:math:`j=3/2`)
        - 't2gd52': :math:`t_{2g}`-shell (:math:`l_{\\text{eff}}=1`)
          plus :math:`d_{5/2}`-shell (:math:`j=5/2`)
        - 'dd':     :math:`d`-shell (:math:`l=2`) plus :math:`d`-shell (:math:`l=2`)
        - 'dd32':   :math:`d`-shell (:math:`l=2`) plus :math:`d_{3/2}`-shell (:math:`j=3/2`)
        - 'dd52':   :math:`d`-shell (:math:`l=2`) plus :math:`d_{5/2}`-shell (:math:`j=5/2`)
        - 'fd':     :math:`f`-shell (:math:`l=3`) plus :math:`d`-shell (:math:`l=2`)
        - 'fd32':   :math:`f`-shell (:math:`l=3`) plus :math:`d_{3/2}`-shell (:math:`j=3/2`)
        - 'fd52':   :math:`f`-shell (:math:`l=3`) plus :math:`d_{5/2}`-shell (:math:`j=5/2`)
        - 'sf':     :math:`s`-shell (:math:`l=0`) plus :math:`f`-shell (:math:`l=3`)
        - 'sf52':   :math:`s`-shell (:math:`l=0`) plus :math:`f_{5/2}`-shell (:math:`j=5/2`)
        - 'sf72':   :math:`s`-shell (:math:`l=0`) plus :math:`f_{7/2}`-shell (:math:`j=7/2`)
        - 'pf':     :math:`p`-shell (:math:`l=1`) plus :math:`f`-shell (:math:`l=3`)
        - 'pf52':   :math:`p`-shell (:math:`l=1`) plus :math:`f_{5/2}`-shell (:math:`j=5/2`)
        - 'pf72':   :math:`p`-shell (:math:`l=1`) plus :math:`f_{7/2}`-shell (:math:`j=7/2`)
        - 't2gf':   :math:`t_{2g}`-shell (:math:`l_{\\text{eff}}=1`)
          plus :math:`f`-shell (:math:`l=3`)
        - 't2gf52': :math:`t_{2g}`-shell (:math:`l_{\\text{eff}}=1`)
          plus :math:`f_{5/2}`-shell (:math:`j=5/2`)
        - 't2gf72': :math:`t_{2g}`-shell (:math:`l_{\\text{eff}}=1`)
          plus :math:`f_{7/2}`-shell (:math:`j=7/2`)
        - 'df':     :math:`d`-shell (:math:`l=2`) plus :math:`f`-shell (:math:`l=3`)
        - 'df52':   :math:`d`-shell (:math:`l=2`) plus :math:`f_{5/2}`-shell (:math:`j=5/2`)
        - 'df72':   :math:`d`-shell (:math:`l=2`) plus :math:`f_{7/2}`-shell (:math:`j=7/2`)
        - 'ff':     :math:`f`-shell (:math:`l=3`) plus :math:`f`-shell (:math:`l=3`)
        - 'ff52':   :math:`f`-shell (:math:`l=3`) plus :math:`f_{5/2}`-shell (:math:`j=5/2`)
        - 'ff72':   :math:`f`-shell (:math:`l=3`) plus :math:`f_{7/2}`-shell (:math:`j=7/2`)
    *args: floats
        Variable length argument list. Slater integrals, should be one of the following cases:

        - 's':  [F0_ss]
        - 'p':  [F0_pp, F2_pp]
        - 'd', 't2g':  [F0_dd, F2_dd, F4_dd]
        - 'f':  [F0_ff, F2_ff, F4_ff, F6_ff]
        - 'ss': [F0_s1s1, F0_s1s2, G0_s1s2, F0_s2s2]
        - 'ps': [F0_pp, F2_pp, F0_ps, G1_ps, F0_ss]
        - 'ds', 't2gs': [F0_dd, F2_dd, F4_dd, F0_ds, G2_ds, F0_ss]
        - 'fs':  [F0_ff, F2_ff, F4_ff, F6_ff, F0_fs, G3_fs, F0_ss]
        - 'sp', 'sp12', 'sp32': [F0_ss, F0_sp, G1_sp, F0_pp, F2_pp]
        - 'pp', 'pp12', 'pp32':
          [F0_p1p1, F2_p1p1, F0_p1p2, F2_p1p2, G0_p1p2, G2_p1p2, F0_p2p2, F2_p2p2]
        - 'dp', 'dp12', 'dp32', 't2gp', 't2gp12', t2gp32':
          [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp, F0_pp, F2_pp]
        - 'fp', 'fp12', 'fp32':
          [F0_ff, F2_ff, F4_ff, F6_ff, F0_fp, F2_fp, G2_fp, G4_fp, F0_pp, F2_pp]
        - 'sd', 'sd32', 'sd52':
          [F0_ss, F0_sd, G2_sd, F0_ddd, F2_dd, F4_dd]
        - 'pd', 'pd32', 'pd52':
          [F0_pp, F2_pp, F0_pd, F2_pd, G1_pd, G3_pd, F0_dd, F2_dd, F4_dd]
        - 'dd', 'dd32', 'dd52', 't2gd', 't2gd32', 't2gd52':
          [F0_d1d1, F2_d1d1, F4_d1d1, F0_d1d2, F2_d1d2, F4_d1d2, G0_d1d2,
          G2_d1d2, G4_d1d2, F0_d2d2, F2_d2d2, F4_d2d2]
        - 'fd', 'fd32', 'fd52':
          [F0_ff, F2_ff, F4_ff, F6_ff, F0_fd, F2_fd, F4_fd, G1_fd, G3_fd, G5_fd,
          F0_dd, F2_dd, F4_dd]
        - 'sf', 'sf52', 'sf72':
          [F0_ss, F0_sf, G3_sf, F0_ff, F2_ff, F4_ff, F6_ff]
        - 'pf', 'pf52', 'pf72':
          [F0_pp, F2_pp, F0_pf, F2_pf, G2_pf, G4_pf, F0_ff, F2_ff, F4_ff, F6_ff]
        - 'df', 'df52', 'df72', 't2gf', 't2gf52', 't2gf72':
          [F0_pp, F2_pp, F0_pf, F2_pf, G2_pf, G4_pf, F0_ff, F2_ff, F4_ff, F6_ff]
        - 'ff', 'ff52', 'ff72':
          [F0_f1f1, F2_f1f1, F4_f1f1, F6_f1f1, F0_f1f2, F2_f1f2, F4_f1f2, F6_f1f2,
          G0_f1f2, G2_f1f2, G4_f1f2, G6_f1f2, F0_f2f2, F2_f2f2, F4_f2f2, F6_f2f2]

        The order of these arguments are important.

    Returns
    -------
    umat: 4d array of complex
        the Coulomb interaction tensor

    Examples
    --------
    >>> import edrixs
    >>> F0_dd, F2_dd, F4_dd = 3.0, 1.0, 0.5
    >>> F0_dp, F2_dp, G1_dp, G3_dp = 2.0, 1.0, 0.2, 0.1
    >>> F0_pp, F2_pp = 0.0, 0.0
    >>> slater = [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp, F0_pp, F2_pp]
    >>> umat_d = edrixs.get_umat_slater('d', F0_dd, F2_dd, F4_dd)
    >>> umat_dp = edrixs.get_umat_slater('dp', *slater)
    >>> umat_t2gp = edrixs.get_umat_slater('t2gp', *slater)
    >>> umat_dp32 = edrixs.get_umat_slater('dp32', *slater)

    See Also
    --------
    coulomb_utensor.umat_slater
    coulomb_utensor.get_umat_kanamori
    coulomb_utensor.get_umat_kanamori_ge

    """
    info = info_atomic_shell()
    shells = case_to_shell_name(case)
    # only one shell
    if len(shells) == 1:
        orbl = info[shells[0]][0]
        l_list = [orbl]
        fk = {}
        it = 0
        for rank in range(0, 2*orbl+1, 2):
            fk[(rank, 1, 1, 1, 1)] = args[it]
            it += 1
        umat = umat_slater(l_list, fk)
        if shells[0] == 't2g':
            # c to t2g
            umat = transform_utensor(umat, tmat_c2r('d', True))
            indx = [2, 3, 4, 5, 8, 9]
            umat_tmp = np.zeros((6, 6, 6, 6), dtype=np.complex128)
            umat_tmp[:, :, :, :] = umat[indx][:, indx][:, :, indx][:, :, :, indx]
            # t2g to c
            umat_tmp[:, :, :, :] = transform_utensor(umat_tmp, tmat_r2c('t2g', True))
            umat = umat_tmp
    # two shells
    elif len(shells) == 2:
        name1, name2 = shells
        l1, l2 = info[name1][0], info[name2][0]
        n1, n2 = 2*(2*l1+1), 2*(2*l2+1)
        ntot = n1 + n2
        l_list = [l1, l2]
        fk = {}
        it = 0
        for rank in range(0, 2 * l1 + 1, 2):
            fk[(rank, 1, 1, 1, 1)] = args[it]
            it += 1
        for rank in range(0, min(2 * l1, 2 * l2) + 1, 2):
            fk[(rank, 1, 2, 1, 2)] = args[it]
            fk[(rank, 2, 1, 2, 1)] = args[it]
            it += 1
        for rank in range(abs(l1 - l2), l1 + l2 + 1, 2):
            fk[(rank, 1, 2, 2, 1)] = args[it]
            fk[(rank, 2, 1, 1, 2)] = args[it]
            it += 1
        for rank in range(0, 2 * l2 + 1, 2):
            fk[(rank, 2, 2, 2, 2)] = args[it]
            it += 1
        umat = umat_slater(l_list, fk)
        # truncate to a sub-shell if necessary
        shell_list2 = ['p12', 'p32', 'd32', 'd52', 'f52', 'f72']
        if (name1 == 't2g') or (name2 in shell_list2):
            tmat = np.eye(ntot, dtype=np.complex)
            indx1 = list(range(0, n1))
            if name1 == 't2g':
                tmat[0:n1, 0:n1] = tmat_c2r('d', True)
                indx1 = [2, 3, 4, 5, 8, 9]
            indx2 = [n1 + i for i in range(0, n2)]
            if name2 in shell_list2:
                tmat[n1:ntot, n1:ntot] = tmat_c2j(l2)
                if name2 == 'p12':
                    indx2 = [n1 + i for i in [0, 1]]
                if name2 == 'p32':
                    indx2 = [n1 + i for i in [2, 3, 4, 5]]
                if name2 == 'd32':
                    indx2 = [n1 + i for i in [0, 1, 2, 3]]
                if name2 == 'd52':
                    indx2 = [n1 + i for i in [4, 5, 6, 7, 8, 9]]
                if name2 == 'f52':
                    indx2 = [n1 + i for i in [0, 1, 2, 3, 4, 5]]
                if name2 == 'f72':
                    indx2 = [n1 + i for i in [6, 7, 8, 9, 10, 11, 12, 13]]
            indx = indx1 + indx2
            umat = transform_utensor(umat, tmat)
            umat_tmp = np.zeros((len(indx), len(indx), len(indx), len(indx)), dtype=np.complex)
            umat_tmp[:, :, :, :] = umat[indx][:, indx][:, :, indx][:, :, :, indx]
            if name1 == 't2g':
                tmat = np.eye(len(indx), dtype=np.complex128)
                tmat[0:6, 0:6] = tmat_r2c('t2g', True)
                umat_tmp[:, :, :, :] = transform_utensor(umat_tmp, tmat)
            umat = umat_tmp
    else:
        raise Exception("Not implemented!")

    return umat


def get_umat_kanamori(norbs, U, J):
    """
    Calculate the Coulomb interaction tensor for a Kanamori-type interaction.
    For the :math:`t2g`-shell case, it is parameterized by :math:`U, J`.

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

    return get_umat_kanamori_ge(norbs, U, U - 2 * J, J, J, J)
