__all__ = ['beta_to_kelvin', 'kelvin_to_beta', 'boltz_dist', 'UJ_to_UdJH',
           'UdJH_to_UJ', 'UdJH_to_F0F2F4', 'UdJH_to_F0F2F4F6', 'F0F2F4_to_UdJH',
           'F0F2F4_to_UJ', 'F0F2F4F6_to_UdJH', 'info_atomic_shell',
           'case_to_shell_name', 'edge_to_shell_name', 'slater_integrals_name']

import numpy as np


def beta_to_kelvin(beta):
    """
    Convert :math:`\\beta` to Kelvin.

    Parameters
    ----------
    beta: float
        Inversion temperature.

    Returns
    -------
    T: float
        Temperature (K).
    """

    kb = 8.6173303E-5
    ev = 1.0 / float(beta)
    T = ev / kb
    return T


def kelvin_to_beta(k):
    """
    Convert temperature from Kelvin to :math:`\\beta`.

    Parameters
    ----------
    k: float
        Temperature in Kelvin.

    Returns
    -------
    beta: float
        Inversion temperature.
    """

    kb = 8.6173303E-5
    beta = 1.0 / (kb * k)
    return beta


def boltz_dist(gs, T):
    """
    Return Boltzmann distributition.

    Parameters
    ----------
    gs: 1d float array
        Energy levels.
    T: float
        Temperature in Kelvin.

    Returns
    -------
    res: 1d float array
        The Boltzmann distributition.
    """
    tmp_gs = np.array(gs)
    beta = kelvin_to_beta(T)
    res = np.exp(-beta * (tmp_gs - min(tmp_gs))) / np.sum(np.exp(-beta * (tmp_gs - min(tmp_gs))))
    return res


def UJ_to_UdJH(U, J):
    """
    Given Kanamori :math:`U` and :math:`J`, return :math:`U_d` and :math:`J_H`,
    for :math:`t2g`-orbitals.

    Parameters
    ----------
    U: float
        Coulomb interaction :math:`U`.
    J: float
        Hund's coupling :math:`J`.

    Returns
    -------
    Ud: float
        Coulomb interaction :math:`U_d`.
    JH: float
        Hund's coupling :math:`J_{H}`.
    """

    F2 = J / (3.0 / 49.0 + 20 * 0.625 / 441.0)
    F4 = 0.625 * F2
    JH = (F2 + F4) / 14.0
    Ud = U - 4.0 / 49.0 * (F2 + F4)

    return Ud, JH


def UdJH_to_UJ(Ud, JH):
    """
    Given :math:`U_d` and :math:`J_H`, return Kanamori :math:`U` and :math:`J`,
    for :math:`t2g`-orbitals.

    Parameters
    ----------
    Ud: float
        Coulomb interaction :math:`U_d`.
    JH: float
        Hund's coupling :math:`J_H`.

    Returns
    -------
    U: float
        Coulomb interaction :math:`U` in Kanamori form.
    J: float
        Hund's coupling :math:`J` in Kanamori form.
    """

    F2 = 14.0 / 1.625 * JH
    F4 = 0.625 * F2
    J = 3.0 / 49.0 * F2 + 20 / 441.0 * F4
    U = Ud + 4.0 / 49.0 * (F2 + F4)

    return U, J


def UdJH_to_F0F2F4(Ud, JH):
    """
    Given :math:`U_d` and :math:`J_H`, return :math:`F_0`, :math:`F_2` and :math:`F_4`,
    for :math:`d`-orbitals.

    Parameters
    ----------
    Ud: float
        Coulomb interaction :math:`U_d`.
    JH: float
        Hund's coupling :math:`J_H`.

    Returns
    -------
    F0: float
        Slater integral :math:`F_0`.
    F2: float
        Slater integral :math:`F_2`.
    F4: float
        Slater integral :math:`F_4`.
    """

    F0 = Ud
    F2 = 14 / 1.625 * JH
    F4 = 0.625 * F2

    return F0, F2, F4


def UdJH_to_F0F2F4F6(Ud, JH):
    """
    Given :math:`U_d` and :math:`J_H`, return :math:`F_0`, :math:`F_2`, :math:`F_4` and :math:`F_6`,
    for :math:`f`-orbitals.

    Parameters
    ----------
    Ud: float
        Coulomb interaction :math:`U_d`.
    JH: float
        Hund's coupling :math:`J_H`.

    Returns
    -------
    F0: float
        Slater integral :math:`F_0`.
    F2: float
        Slater integral :math:`F_2`.
    F4: float
        Slater integral :math:`F_4`.
    F6: float
        Slater integral :math:`F_6`.

    """

    F0 = Ud
    F2 = 6435 / (286.0 + (195 * 451) / 675.0 + (250 * 1001) / 2025.0) * JH
    F4 = 451 / 675.0 * F2
    F6 = 1001 / 2025.0 * F2

    return F0, F2, F4, F6


def F0F2F4_to_UdJH(F0, F2, F4):
    """
    Given :math:`F_0`, :math:`F_2` and :math:`F_4`, return :math:`U_d` and :math:`J_H`,
    for :math:`d`-orbitals.

    Parameters
    ----------
    F0: float
        Slater integral :math:`F_0`.
    F2: float
        Slater integral :math:`F_2`.
    F4: float
        Slater integral :math:`F_4`.

    Returns
    -------
    Ud: float
        Coulomb interaction :math:`U_d`.
    JH: float
        Hund's coupling :math:`J_H`.
    """

    Ud = F0
    JH = (F2 + F4) / 14.0

    return Ud, JH


def F0F2F4_to_UJ(F0, F2, F4):
    """
    Given :math:`F_0`, :math:`F_2` and :math:`F_4`, return :math:`U` and :math:`J`,
    for :math:`t2g`-orbitals.

    Parameters
    ----------
    F0: float
        Slater integral :math:`F_0`.
    F2: float
        Slater integral :math:`F_2`.
    F4: float
        Slater integral :math:`F_4`.

    Returns
    -------
    U: float
        Coulomb interaction :math:`U`.
    J: float
        Hund's coupling :math:`J`.
    """

    U = F0 + 4.0 / 49.0 * (F2 + F4)
    J = 3.0 / 49.0 * F2 + 20 / 441.0 * F4

    return U, J


def F0F2F4F6_to_UdJH(F0, F2, F4, F6):
    """
    Given :math:`F_0`, :math:`F_2`, :math:`F_4` and :math:`F_6`,
    return :math:`U_d` and :math:`J_H`, for :math:`f`-orbitals.

    Parameters
    ----------
    F0: float
        Slater integral :math:`F_0`.
    F2: float
        Slater integral :math:`F_2`.
    F4: float
        Slater integral :math:`F_4`.
    F6: float
        Slater integral :math:`F_6`.

    Returns
    -------
    Ud: float
        Coulomb interaction :math:`U_d`.
    JH: float
        Hund's coupling :math:`J_H`.
    """

    Ud = F0
    JH = (286 * F2 + 195 * F4 + 250 * F6) / 6435.0
    return Ud, JH


def info_atomic_shell():
    """
    Return a dict to describe the information of atomic shell.
    The key is a string of the shell's name, the value is a 2-elments tuple,
    the first value is the orbital angular moment number and
    the second value is the dimension of the Hilbert space.

    Returns
    -------
    info: dict
        The dict describing info of atomic shell.
    """

    info = {'s':   (0, 2),
            'p':   (1, 6),
            'p12': (1, 2),
            'p32': (1, 4),
            't2g': (2, 6),
            'd':   (2, 10),
            'd32': (2, 4),
            'd52': (2, 6),
            'f':   (3, 14),
            'f52': (3, 6),
            'f72': (3, 8)
            }

    return info


def case_to_shell_name(case):
    """
    Return the shell names for different cases.

    Parameters
    ----------
    case: string
        A string describing the shells included.

    Returns
    -------
    shell_name: tuple of one or two strings
        The name of shells.

    Examples
    --------
    >>> import edrixs
    >>> edrixs.case_to_shell_name('d')
    ('d',)
    >>> edrixs.case_to_shell_name('t2gp32')
    ('t2g', 'p32')
    """

    shell = ['s', 'p', 'p12', 'p32', 't2g', 'd', 'd32', 'd52', 'f', 'f52', 'f72']

    shell_name = {}
    for str1 in shell:
        shell_name[str1] = (str1,)

    for str1 in shell:
        for str2 in shell:
            shell_name[str1+str2] = (str1, str2)

    return shell_name[case.strip()]


def edge_to_shell_name(edge_name, with_main_qn=False):
    """
    Given edge name, return shell name.
    If one needs to include both spin-orbit split edges of one shell,
    one can use string, for example, 'L23' means both L2 and L3 edges
    are considered in the calculations, and the shell name will be 'p'.

    Parameters
    ----------
    edge_name: string
        Standard edge name.
    with_main_qn: logical
        If true, the shell name will include the main quantum number.
    Returns
    -------
    shell_name: string
        Shell name.

        - with_main_qn=True, shell_name will include the main quantum number,
          for example, *2p*

        - with_main_qn=False, shell_name will not include the main quantum number,
          for example, *p*
    """
    shell_name = {
        'K': ('s', '1s'),
        'L1': ('s', '2s'),
        'L2': ('p12', '2p12'),
        'L3': ('p32', '2p32'),
        'L23': ('p', '2p'),
        'M1': ('s', '3s'),
        'M2': ('p12', '3p12'),
        'M3': ('p32', '3p32'),
        'M23': ('p', '3p'),
        'M4': ('d32', '3d32'),
        'M5': ('d52', '3d52'),
        'M45': ('d', '3d'),
        'N1': ('s', '4s'),
        'N2': ('p12', '4p12'),
        'N3': ('p32', '4p32'),
        'N23': ('p', '4p'),
        'N4': ('d32', '4d32'),
        'N5': ('d52', '4d52'),
        'N45': ('d', '4d'),
        'N6': ('f52', '4f52'),
        'N7': ('f72', '4f72'),
        'N67': ('f', '4f'),
        'O1': ('s', '5s'),
        'O2': ('p12', '5p12'),
        'O3': ('p32', '5p32'),
        'O23': ('p', '5p'),
        'O4': ('d32', '5d32'),
        'O5': ('d52', '5d52'),
        'O45': ('d', '5d'),
        'P1': ('s', '6s'),
        'P2': ('p12', '6p12'),
        'P3': ('p32', '6p32'),
        'P23': ('p', '6p')
    }

    if with_main_qn:
        return shell_name[edge_name.strip()][1]
    else:
        return shell_name[edge_name.strip()][0]


def slater_integrals_name(shell_name, label=None):
    """
    Given shell names, return the names of required Slater integrals.
    The order of these names in the return list is according to the convention:

    - For 1 shell: [FX_11]

    - For 2 shells: [FX_11, FX_12, GX_12, FX_22]

    - For 3 shells: [FX_11, FX_12, GX_12, FX_22, FX_13, GX_13, FX_23, GX_23, FX_33]

    where, X=0, 2, 4, 6, or X=1, 3, 5, X shoule be in ascending order.

    Parameters
    ----------
    shell_name: tuple of strings
        Name of shells. Its length should be less and equal than 3.
    label: tuple of strings
        Label of shells, same shape as shell_name.

        If not provided, label will be set to

        - label=(1,) for one shell, or

        - label=(1,2) for two shells, or

        - label=(1,2,3) for three shells

    Returns
    -------
    res: list of strings
        Names of Slater integrals.
    """
    info = info_atomic_shell()
    # one shell
    if len(shell_name) == 1:
        res = []
        l1 = info[shell_name[0]][0]
        if label is not None:
            x = label[0]
        else:
            x = '1'
        res.extend(['F' + str(i) + '_' + x + x for i in range(0, 2*l1+1, 2)])
    elif len(shell_name) == 2:
        res = []
        l1 = info[shell_name[0]][0]
        l2 = info[shell_name[1]][0]
        if label is not None:
            x, y = label[0], label[1]
        else:
            x, y = '1', '2'
        res.extend(['F' + str(i) + '_' + x + x for i in range(0, 2*l1+1, 2)])
        res.extend(['F' + str(i) + '_' + x + y for i in range(0, min(2*l1, 2*l2)+1, 2)])
        res.extend(['G' + str(i) + '_' + x + y for i in range(abs(l1-l2), l1+l2+1, 2)])
        res.extend(['F' + str(i) + '_' + y + y for i in range(0, 2*l2+1, 2)])
    elif len(shell_name) == 3:
        res = []
        l1 = info[shell_name[0]][0]
        l2 = info[shell_name[1]][0]
        l3 = info[shell_name[2]][0]
        if label is not None:
            x, y, z = label[0], label[1], label[2]
        else:
            x, y, z = '1', '2', '3'
        res.extend(['F' + str(i) + '_' + x + x for i in range(0, 2*l1+1, 2)])
        res.extend(['F' + str(i) + '_' + x + y for i in range(0, min(2*l1, 2*l2)+1, 2)])
        res.extend(['G' + str(i) + '_' + x + y for i in range(abs(l1-l2), l1+l2+1, 2)])
        res.extend(['F' + str(i) + '_' + y + y for i in range(0, 2*l2+1, 2)])
        res.extend(['F' + str(i) + '_' + x + z for i in range(0, min(2*l1, 2*l3)+1, 2)])
        res.extend(['G' + str(i) + '_' + x + z for i in range(abs(l1-l3), l1+l3+1, 2)])
        res.extend(['F' + str(i) + '_' + y + z for i in range(0, min(2*l2, 2*l3)+1, 2)])
        res.extend(['G' + str(i) + '_' + y + z for i in range(abs(l2-l3), l2+l3+1, 2)])
        res.extend(['F' + str(i) + '_' + z + z for i in range(0, 2*l3+1, 2)])
    else:
        raise Exception("Not implemented for this case: ", shell_name)

    return res
