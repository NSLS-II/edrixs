import numpy as np


def beta_to_kelvin(beta):
    """
    Convert :math:`\\beta` to Kelvin.

    Parameters
    ----------
    beta : float
        Inversion temperature.

    Returns
    -------
    T : float
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
    k : float
        Temperature in Kelvin.

    Returns
    -------
    beta : float
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
    gs : 1d float array
        Energy levels.

    T : float
        Temperature in Kelvin.

    Returns
    -------
    res : 1d float array
        The Boltzmann distributition.
    """

    beta = kelvin_to_beta(T)
    res = np.exp(-beta * (gs - min(gs))) / \
        np.sum(np.exp(-beta * (gs - min(gs))))
    return res


def UJ_to_UdJH(U, J):
    """
    Given Kanamori :math:`U` and :math:`J`, return :math:`U_d` and :math:`J_H`,
    for :math:`t2g`-orbitals.

    Parameters
    ----------
    U : float
        Coulomb interaction :math:`U`.

    J : float
        Hund's coupling :math:`J`.

    Returns
    -------
    Ud : float
        Coulomb interaction :math:`U_d`.

    JH : float
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
    Ud : float
        Coulomb interaction :math:`U_d`.

    JH : float
        Hund's coupling :math:`J_H`.

    Returns
    -------
    U : float
        Coulomb interaction :math:`U` in Kanamori form.

    J : float
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
    Ud : float
        Coulomb interaction :math:`U_d`.

    JH : float
        Hund's coupling :math:`J_H`.

    Returns
    -------
    F0 : float
        Slater integral :math:`F_0`.

    F2 : float
        Slater integral :math:`F_2`.

    F4 : float
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
    Ud : float
        Coulomb interaction :math:`U_d`.

    JH : float
        Hund's coupling :math:`J_H`.

    Returns
    -------
    F0 : float
        Slater integral :math:`F_0`.

    F2 : float
        Slater integral :math:`F_2`.

    F4 : float
        Slater integral :math:`F_4`.

    F6 : float
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
    F0 : float
        Slater integral :math:`F_0`.

    F2 : float
        Slater integral :math:`F_2`.

    F4 : float
        Slater integral :math:`F_4`.

    Returns
    -------
    Ud : float
        Coulomb interaction :math:`U_d`.

    JH : float
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
    F0 : float
        Slater integral :math:`F_0`.

    F2 : float
        Slater integral :math:`F_2`.

    F4 : float
        Slater integral :math:`F_4`.

    Returns
    -------
    U : float
        Coulomb interaction :math:`U`.

    J : float
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
    F0 : float
        Slater integral :math:`F_0`.

    F2 : float
        Slater integral :math:`F_2`.

    F4 : float
        Slater integral :math:`F_4`.

    F6 : float
        Slater integral :math:`F_6`.

    Returns
    -------
    Ud : float
        Coulomb interaction :math:`U_d`.

    JH : float
        Hund's coupling :math:`J_H`.
    """

    Ud = F0
    JH = (286 * F2 + 195 * F4 + 250 * F6) / 6435.0
    return Ud, JH
